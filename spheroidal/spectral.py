"""Module containing functions for computing spin-weighted spheroidal harmonics using the spherical expansion method."""
from .spherical import *

_NMAX = 500


def eigenvalue_spectral(s, ell, m, g, num_terms=None, n_max=_NMAX):
    """Computes the spin-weighted spheroidal eigenvalue with spin-weight s,
    degree l, order m, and spheroidicity g

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    ell : int or half-integer float
        degree
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    num_terms : int
        number of terms in the spherical expansion, automatic by default
    n_max : int
        maximum number of terms in the spherical expansion, defaults to
        100

    Returns
    -------
    double
        spin-weighted spheroidal eigenvalue :math:`{}_{s}\lambda_{lm}`
    """
    l_min = max(abs(s), abs(m))

    if num_terms is None:
        prev_sep_const = separation_constants(s, m, g, num_terms = ell - l_min + 10)[int(ell - l_min)]

        for i in range(20, n_max, 10):
            sep_const = separation_constants(s, m, g, num_terms= ell - l_min + i)[int(ell - l_min)]
            # return eigenvalue once machine precision is reached
            if sep_const == prev_sep_const:
                return sep_const + g**2 - 2 * m * g
            prev_sep_const = sep_const
        return sep_const + g**2 - 2 * m * g
    else:
        return (
            separation_constants(s, m, g, num_terms)[int(ell - l_min)]
            + g**2
            - 2 * m * g
        )


def harmonic_spectral(s, ell, m, g, num_terms=None, n_max=_NMAX):
    r"""Computes the spin-weighted spheroidal harmonic with spin-weight s,
    degree l, order m, and spheroidicity g using the spherical expansion method.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    ell : int or half-integer float
        degree
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    num_terms : int
        number of terms in the expansion
    n_max : int
        maximum number of terms in the expansion

    Returns
    -------
    function
        spin-weighted spheroidal harmonic
        :math:`{}_{s}S_{lm}(\theta,\phi)`
    """
    l_min = max(abs(s), abs(m))
    if num_terms is None:
        # adaptively increase the number of terms until the final coefficient is zero
        for i in range(20, n_max, 10):
            coefficients = mixing_coefficients(s, ell, m, g, ell - l_min + i)
            if coefficients[-1] == 0:
                break
    else:
        # compute specified number of coefficients
        coefficients = mixing_coefficients(s, ell, m, g, ell - l_min + num_terms)

    def Sslm(theta, phi):
        spherical_harmonics = np.array(
            [
                sphericalY(s, l, m)(theta, phi)
                for l in np.arange(l_min, l_min + len(coefficients), 1)
            ]
        )
        # tensordot allows theta and phi to be multidimensional arrays
        return np.tensordot(coefficients,spherical_harmonics,axes=1)

    return Sslm


def harmonic_spectral_deriv(s, ell, m, g, num_terms=None, n_max=_NMAX):
    r"""Computes the derivative with respect to theta of the spin-weighted
    spheroidal harmonic with spin-weight s, degree l, order m, and spheroidicity g
    using the spherical expansion method.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    ell : int or half-integer float
        degree
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    num_terms : int
        number of terms in the expansion

    Returns
    -------
    function
        derivative of the spin-weighted spheroidal harmonic
        :math:`\frac{d}{d\theta}\left({}_{s}S_{lm}(\theta,\phi)\right)`
    """
    l_min = max(abs(s), abs(m))
    if num_terms is None:
        # adaptively increase the number of terms until the final coefficient is zero
        for i in range(20, n_max, 10):
            coefficients = mixing_coefficients(s, ell, m, g, ell - l_min + i)
            if coefficients[-1] == 0:
                break
    else:
        coefficients = mixing_coefficients(s, ell, m, g, ell - l_min + num_terms)

    def dS(theta, phi):
        spherical_harmonics = np.array(
            [
                sphericalY_deriv(s, l, m)(theta, phi)
                for l in np.arange(l_min, l_min + len(coefficients))
            ]
        )
        return np.tensordot(coefficients, spherical_harmonics, axes=1)

    return dS


def harmonic_spectral_deriv2(s, ell, m, g, num_terms=None, n_max=_NMAX):
    r"""Computes the second derivative with respect to theta of the spin-weighted
    spheroidal harmonic with spin-weight s, degree l, order m, and
    spheroidicity g using the spherical expansion method.

    Parameters
    ----------
    s : int or half-integer float
        spin weight
    ell : int or half-integer float
        degree
    m : int or half-integer float
        order
    g : complex
        spheroidicity
    num_terms : int
        number of terms in the expansion
    n_max : int


    Returns
    -------
    function
        derivative of the spin-weighted spheroidal harmonic
        :math:`\frac{d}{d\theta}\left({}_{s}S_{lm}(\theta,\phi)\right)`
    """
    eigenvalue = eigenvalue_spectral(s, ell, m, g, num_terms, n_max)

    S = harmonic_spectral(s, ell, m, g, num_terms, n_max)
    dS = harmonic_spectral_deriv(s, ell, m, g, num_terms, n_max)

    def dS2(theta, phi):
        theta = np.where(abs(theta) < 1e-6, 1e-6, theta)
        return (
            g**2 * sin(theta) ** 2
            + (m + s * cos(theta)) ** 2 / sin(theta) ** 2
            + 2 * g * s * cos(theta)
            - s
            - 2 * m * g
            - eigenvalue
        ) * S(theta, phi) - cos(theta) / sin(theta) * dS(theta, phi)

    return dS2

class SpinWeightedSpheroidalHarmonicSpectral:
    def __init__(self, s, ell, m, g, num_terms = None, n_max = _NMAX, **kwargs):
        if n_max < 30:
            n_max = 30

        self.s = s
        self.ell = ell
        self.m = m
        self.gamma = g
        self.num_terms = num_terms
        self.n_max = n_max
        self.l_min = max(abs(s), abs(m))

        if "eigenvalue" in kwargs.keys() and "coefficients" in kwargs.keys():
            self.coefficients = kwargs["coefficients"]
            self.eigenvalue = kwargs["eigenvalue"]
            self.num_terms = self.coefficients.shape[0] - (ell - self.l_min)
        else:
            # calculate eigenvalues
            if num_terms is None:
                # adaptively increase the number of terms until the final coefficient is zero
                for i in range(20, n_max, 10):
                    self.coefficients, la = mixing_coefficients_and_separation_constants(s, ell, m, g, ell - self.l_min + i)
                    if self.coefficients[-1] == 0:
                        break
            else:
                # compute specified number of coefficients
                self.coefficients, la = mixing_coefficients_and_separation_constants(s, ell, m, g, ell - self.l_min + num_terms)
            self.num_terms = i
            self.eigenvalue = la[int(ell - self.l_min)] + g**2 - 2 * m * g

        self.shifted_eigenvalue = self.eigenvalue + self.s*(self.s + 1)

    def mixing_coefficient(self, l):
        if l < self.l_min or l - self.l_min > + len(self.coefficients) - 1:
            return 0.
        return self.coefficients[l - self.l_min]

    def __call__(self, theta, phi):
        spherical_harmonics = np.array(
            [
                sphericalY(self.s, l, self.m)(theta, 0.)
                for l in np.arange(self.l_min, self.l_min + len(self.coefficients), 1)
            ]
        )
        # tensordot allows theta and phi to be multidimensional arrays
        return np.tensordot(self.coefficients, spherical_harmonics, axes=1)*np.exp(1j*self.m*phi)
    
    def deriv(self, theta, phi):
        spherical_harmonics = np.array(
            [
                sphericalY_deriv(self.s, l, self.m)(theta, 0.)
                for l in np.arange(self.l_min, self.l_min + len(self.coefficients), 1)
            ]
        )
        # tensordot allows theta and phi to be multidimensional arrays
        return np.tensordot(self.coefficients, spherical_harmonics, axes=1)*np.exp(1j*self.m*phi)
    
    def deriv2(self, theta, phi):
        theta = theta % (2.*np.pi)
        Yslm = np.array([
            sphericalY(self.s, l, self.m)(theta, 0.)
            for l in np.arange(self.l_min, self.l_min + len(self.coefficients), 1)
        ])
        dYslm = np.array([
            sphericalY_deriv(self.s, l, self.m)(theta, 0.)
            for l in np.arange(self.l_min, self.l_min + len(self.coefficients), 1)
        ])
        Yslm_coeff = 0.*self.gamma*theta
        dYslm_coeff = 0.*self.gamma*theta

        ctheta = np.cos(theta)
        stheta = np.sin(theta)

        # select out arguments where sin(th) is non-zero
        nonzero_args = (np.abs(stheta) > 0.)
        if np.any(nonzero_args):
            if np.all(nonzero_args):
                cth = ctheta
                sth = stheta
                Yslm_coeff = (
                                self.gamma**2 * sth ** 2
                                + (self.m + self.s * cth) ** 2 / sth ** 2
                                + 2 * self.gamma * self.s * cth
                                - self.s
                                - 2 * self.m * self.gamma
                                - self.eigenvalue
                            )
                dYslm_coeff = - cth / sth
            else:
                cth = ctheta[nonzero_args]
                sth = stheta[nonzero_args]
                Yslm_coeff[nonzero_args] = (
                                self.gamma**2 * sth ** 2
                                + (self.m + self.s * cth) ** 2 / sth ** 2
                                + 2 * self.gamma * self.s * cth
                                - self.s
                                - 2 * self.m * self.gamma
                                - self.eigenvalue
                            )
                dYslm_coeff[nonzero_args] = - cth / sth

        # deal with case where theta is zero
        zero_args = (theta == 0.)
        if np.any(zero_args):
            if np.all(zero_args):
                cth = ctheta
                sth = stheta
                Yslm_coeff = 2.*(
                                self.s**2 + self.s*(1. + self.m - 2*self.gamma) 
                                + 2.*self.m*self.gamma + self.eigenvalue
                            ) / ((self.m + self.s)**2 - 4)
            else:
                cth = ctheta[zero_args]
                sth = stheta[zero_args]
                Yslm_coeff[zero_args] = 2.*(
                                self.s**2 + self.s*(1. + self.m - 2*self.gamma) 
                                + 2.*self.m*self.gamma + self.eigenvalue
                            ) / ((self.m + self.s)**2 - 4)

        # deal with case where theta is pi
        zero_args = (theta == np.pi)
        if np.any(zero_args):
            if np.all(zero_args):
                cth = ctheta
                sth = stheta
                Yslm_coeff = 2.*(
                                self.s**2 + self.s*(1. - self.m + 2*self.gamma) 
                                + 2.*self.m*self.gamma + self.eigenvalue
                            ) / ((self.m - self.s)**2 - 4)
            else:
                cth = ctheta[zero_args]
                sth = stheta[zero_args]
                Yslm_coeff[zero_args] = 2.*(
                                self.s**2 + self.s*(1. - self.m + 2*self.gamma) 
                                + 2.*self.m*self.gamma + self.eigenvalue
                            ) / ((self.m - self.s)**2 - 4)

        spherical_harmonics = Yslm_coeff*Yslm + dYslm_coeff*dYslm

        # tensordot allows theta and phi to be multidimensional arrays
        return np.tensordot(self.coefficients, spherical_harmonics, axes=1)*np.exp(1j*self.m*phi)

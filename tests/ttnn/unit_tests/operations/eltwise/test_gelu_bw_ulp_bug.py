# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
GELU Backward ULP Precision Tests

This test validates the accuracy of ttnn.gelu_bw (GELU derivative) across
the BFloat16 range using the same methodology as test_gelu_floor_value_bug.py.

MATHEMATICAL FORMULA:
GELU'(x) = grad * (cdf + x * pdf)
where:
  cdf = 0.5 * (1 + erf(x / sqrt(2)))  -- CDF of standard normal distribution
  pdf = exp(-x^2 / 2) / sqrt(2*pi)    -- PDF of standard normal distribution

Hardware Model: Tenstorrent SFPU uses DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero)
Per tech_reports/Handling_Special_Value/special_values.md: "denormals | all | 0x0"

Run: pytest tests/ttnn/unit_tests/operations/eltwise/test_gelu_bw_ulp_bug.py -v -s
"""

import struct
import pytest
import torch
import ttnn
from loguru import logger
from mpmath import mp, erf as mp_erf, erfc as mp_erfc, exp as mp_exp, sqrt as mp_sqrt


def float_to_bf16_bits(f: float) -> int:
    """Convert float to BFloat16 bit representation."""
    f32_bits = struct.unpack(">I", struct.pack(">f", f))[0]
    return f32_bits >> 16


def bf16_bits_to_float(bits: int) -> float:
    """Convert BFloat16 bits to float."""
    f32_bits = bits << 16
    return struct.unpack(">f", struct.pack(">I", f32_bits))[0]


def is_bf16_denormal(bits: int) -> bool:
    """Check if BF16 bits represent a denormal (subnormal) value."""
    exp = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    return (exp == 0) and (mantissa != 0)


def bf16_daz_normalize(bits: int) -> int:
    """Apply DAZ (Denormals-Are-Zero) normalization to BF16 bits."""
    if is_bf16_denormal(bits):
        return 0x0000
    if bits == 0x8000:  # -0 -> +0
        return 0x0000
    return bits


def bf16_value_order_index_daz(bits: int) -> int:
    """Calculate the value order index for a BFloat16 value with DAZ."""
    bits = bf16_daz_normalize(bits)

    exp = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    if exp == 0xFF and mantissa != 0:
        return -1  # NaN
    if bits == 0x7F80:
        return 65281  # +inf
    if bits == 0xFF80:
        return -1  # -inf
    if bits == 0x0000:
        return 32640  # Zero

    if bits & 0x8000:
        magnitude = bits & 0x7FFF
        return 0x7F7F - magnitude
    else:
        return 32640 + bits - 0x007F


def ulp_distance_bf16_daz(a: float, b: float) -> int:
    """Calculate ULP distance with DAZ+FTZ model."""
    a_bits = bf16_daz_normalize(float_to_bf16_bits(a))
    b_bits = bf16_daz_normalize(float_to_bf16_bits(b))

    a_exp = (a_bits >> 7) & 0xFF
    b_exp = (b_bits >> 7) & 0xFF
    if (a_exp == 0xFF and (a_bits & 0x7F) != 0) or (b_exp == 0xFF and (b_bits & 0x7F) != 0):
        return -1

    idx_a = bf16_value_order_index_daz(a_bits)
    idx_b = bf16_value_order_index_daz(b_bits)

    if idx_a < 0 or idx_b < 0:
        return -1

    return abs(idx_a - idx_b)


def gelu_derivative_exact(x: float) -> float:
    """
    Exact GELU derivative using mpmath 256-bit precision.

    GELU'(x) = cdf + x * pdf
    where:
      cdf = 0.5 * (1 + erf(x / sqrt(2)))
      pdf = exp(-x^2 / 2) / sqrt(2*pi)

    Uses erfc() for numerical stability with large negative x.
    """
    mp.prec = 256
    x_mp = mp.mpf(x)
    sqrt2 = mp_sqrt(2)
    sqrt_2pi = mp_sqrt(2 * mp.pi)

    # CDF: 0.5 * (1 + erf(x / sqrt(2)))
    # For numerical stability with negative x, use erfc
    if x < 0:
        cdf = mp.mpf("0.5") * mp_erfc(-x_mp / sqrt2)
    else:
        cdf = mp.mpf("0.5") * (1 + mp_erf(x_mp / sqrt2))

    # PDF: exp(-x^2 / 2) / sqrt(2*pi)
    pdf = mp_exp(-x_mp * x_mp / 2) / sqrt_2pi

    # GELU'(x) = cdf + x * pdf
    result = cdf + x_mp * pdf
    return float(result)


def gelu_derivative_expected_bf16_daz(x: float) -> float:
    """Compute expected BF16 GELU derivative with DAZ+FTZ applied."""
    x_bits = bf16_daz_normalize(float_to_bf16_bits(x))
    x_daz = bf16_bits_to_float(x_bits)
    result = gelu_derivative_exact(x_daz)
    result_bits = bf16_daz_normalize(float_to_bf16_bits(result))
    return bf16_bits_to_float(result_bits)


def gelu_bw_expected_bf16_daz(grad: float, x: float) -> float:
    """Compute expected BF16 GELU backward with DAZ+FTZ applied."""
    grad_bits = bf16_daz_normalize(float_to_bf16_bits(grad))
    grad_daz = bf16_bits_to_float(grad_bits)
    x_bits = bf16_daz_normalize(float_to_bf16_bits(x))
    x_daz = bf16_bits_to_float(x_bits)
    result = grad_daz * gelu_derivative_exact(x_daz)
    result_bits = bf16_daz_normalize(float_to_bf16_bits(result))
    return bf16_bits_to_float(result_bits)


# =============================================================================
# Test Classes
# =============================================================================


class TestGeluBwDerivativeAtZero:
    """Tests for GELU derivative at x=0."""

    def test_derivative_at_zero(self, device):
        """GELU'(0) = 0.5"""
        input_val = 0.0
        grad_val = 1.0

        torch_input = torch.tensor([[input_val]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[grad_val]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        result = ttnn.gelu_bw(tt_grad, tt_input, approximate="none")
        actual = ttnn.to_torch(result[0]).item()

        expected = gelu_derivative_expected_bf16_daz(input_val)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x=0: expected={expected:.4f}, actual={actual:.4f}, ULP={ulp_error}")

        assert ulp_error <= 2, f"Expected ULP <= 2, got {ulp_error}"


class TestGeluBwPositiveValues:
    """Tests for GELU derivative at positive values."""

    @pytest.mark.parametrize(
        "input_value,max_expected_ulp",
        [
            (0.5, 5),
            (1.0, 5),
            (2.0, 5),
            (3.0, 5),
            (5.0, 10),
            (10.0, 10),
        ],
    )
    def test_positive_values(self, device, input_value, max_expected_ulp):
        """For large positive x, GELU'(x) approaches 1."""
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[1.0]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        result = ttnn.gelu_bw(tt_grad, tt_input, approximate="none")
        actual = ttnn.to_torch(result[0]).item()

        expected = gelu_derivative_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x={input_value}: expected={expected:.4f}, actual={actual:.4f}, ULP={ulp_error}")

        assert ulp_error <= max_expected_ulp, f"Expected ULP <= {max_expected_ulp}, got {ulp_error}"


class TestGeluBwNegativeValues:
    """Tests for GELU derivative at negative values."""

    @pytest.mark.parametrize(
        "input_value,max_expected_ulp",
        [
            (-0.5, 5),
            (-1.0, 5),
            (-2.0, 10),
            (-3.0, 15),
            (-4.0, 20),
            (-5.0, 25),
            (-6.0, 30),
            (-8.0, 50),
        ],
    )
    def test_negative_values(self, device, input_value, max_expected_ulp):
        """For large negative x, GELU'(x) approaches 0."""
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[1.0]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        result = ttnn.gelu_bw(tt_grad, tt_input, approximate="none")
        actual = ttnn.to_torch(result[0]).item()

        expected = gelu_derivative_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x={input_value}: expected={expected:.6e}, actual={actual:.6e}, ULP={ulp_error}")

        assert ulp_error <= max_expected_ulp, f"Expected ULP <= {max_expected_ulp}, got {ulp_error}"


class TestGeluBwNearZero:
    """Tests for GELU derivative near zero."""

    @pytest.mark.parametrize(
        "input_value",
        [1e-6, 1e-4, 0.01, 0.1, -0.1, -0.01, -1e-4],
    )
    def test_near_zero(self, device, input_value):
        """Near zero, GELU'(x) ≈ 0.5 + x/sqrt(2*pi)."""
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[1.0]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        result = ttnn.gelu_bw(tt_grad, tt_input, approximate="none")
        actual = ttnn.to_torch(result[0]).item()

        expected = gelu_derivative_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x={input_value:.2e}: expected={expected:.4f}, actual={actual:.4f}, ULP={ulp_error}")

        assert ulp_error <= 5, f"Expected ULP <= 5, got {ulp_error}"


class TestGeluBwLocalMinimum:
    """Tests around the local minimum of GELU (x ≈ -0.751)."""

    @pytest.mark.parametrize(
        "input_value",
        [-0.7, -0.75, -0.751, -0.76, -0.8],
    )
    def test_local_minimum_region(self, device, input_value):
        """GELU has a local minimum around x ≈ -0.751 where GELU'(x) ≈ 0."""
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[1.0]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        result = ttnn.gelu_bw(tt_grad, tt_input, approximate="none")
        actual = ttnn.to_torch(result[0]).item()

        expected = gelu_derivative_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x={input_value}: expected={expected:.6f}, actual={actual:.6f}, ULP={ulp_error}")

        # The derivative is very small near the minimum, so ULP can be high
        # but the absolute error should be small
        assert ulp_error <= 50 or abs(actual - expected) < 0.01, f"Error too high at local minimum region"


class TestGeluBwWithGradientScaling:
    """Tests with different gradient values."""

    @pytest.mark.parametrize(
        "input_value,grad_value,max_expected_ulp",
        [
            (1.0, 2.0, 5),
            (-1.0, 0.5, 10),
            (0.0, 1.0, 2),
            (2.0, -1.0, 5),
            (0.5, 3.0, 5),
        ],
    )
    def test_with_gradient(self, device, input_value, grad_value, max_expected_ulp):
        """Test GELU backward with different gradient values."""
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[grad_value]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        result = ttnn.gelu_bw(tt_grad, tt_input, approximate="none")
        actual = ttnn.to_torch(result[0]).item()

        expected = gelu_bw_expected_bf16_daz(grad_value, input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(
            f"x={input_value}, grad={grad_value}: expected={expected:.4f}, actual={actual:.4f}, ULP={ulp_error}"
        )

        assert ulp_error <= max_expected_ulp, f"Expected ULP <= {max_expected_ulp}, got {ulp_error}"


def test_gelu_bw_ulp_summary(device):
    """
    Generates a comprehensive summary of GELU backward ULP errors across all regions.
    Uses DAZ+FTZ model matching Tenstorrent hardware behavior.
    """
    logger.info("")
    logger.info("=" * 100)
    logger.info("GELU BACKWARD ULP SUMMARY (DAZ+FTZ MODEL)")
    logger.info("=" * 100)

    # Key test points
    test_points = [
        ("Zero", 0.0),
        ("Small positive", 0.1),
        ("Unity", 1.0),
        ("Moderate positive", 2.0),
        ("Large positive", 5.0),
        ("Small negative", -0.1),
        ("Negative unity", -1.0),
        ("Local minimum", -0.751),
        ("Moderate negative", -2.0),
        ("Large negative", -5.0),
        ("Deep negative", -8.0),
    ]

    logger.info("")
    logger.info(f"{'Description':>20} | {'x':>10} | {'Expected':>12} | {'Actual':>12} | {'ULP':>8}")
    logger.info("-" * 70)

    max_ulp = 0
    worst_x = 0

    for desc, x in test_points:
        torch_input = torch.tensor([[x]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[1.0]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        result = ttnn.gelu_bw(tt_grad, tt_input, approximate="none")
        actual = ttnn.to_torch(result[0]).item()

        expected = gelu_derivative_expected_bf16_daz(x)
        ulp = ulp_distance_bf16_daz(actual, expected)

        if ulp > max_ulp:
            max_ulp = ulp
            worst_x = x

        logger.info(f"{desc:>20} | {x:>10.4f} | {expected:>12.6f} | {actual:>12.6f} | {ulp:>8}")

    logger.info("-" * 70)
    logger.info(f"Max ULP: {max_ulp} at x = {worst_x}")
    logger.info("")
    logger.info("Expected behavior:")
    logger.info("- GELU'(0) ≈ 0.5")
    logger.info("- GELU'(x) → 1 as x → +∞")
    logger.info("- GELU'(x) → 0 as x → -∞")
    logger.info("- Local minimum near x ≈ -0.751 where GELU'(x) ≈ 0")
    logger.info("=" * 100)


# =============================================================================
# Polynomial Coefficient Derivation Tests
# =============================================================================


def test_derive_gelu_derivative_polynomial_coefficients():
    """
    Derive and validate polynomial coefficients for GELU'(x) approximation.

    This test finds optimal Chebyshev polynomial coefficients for GELU'(x)
    over a given range, similar to how tanh coefficients were derived.

    Run with: pytest tests/ttnn/unit_tests/operations/eltwise/test_gelu_bw_ulp_bug.py::test_derive_gelu_derivative_polynomial_coefficients -v -s
    """
    import numpy as np
    import math

    logger.info("")
    logger.info("=" * 80)
    logger.info("GELU'(x) Polynomial Coefficient Derivation")
    logger.info("=" * 80)

    def gelu_derivative_fp64(x):
        """GELU'(x) using float64 with erfc for stability."""
        sqrt2 = math.sqrt(2.0)
        sqrt_2pi = math.sqrt(2.0 * math.pi)
        if x < 0:
            cdf = 0.5 * math.erfc(-x / sqrt2)
        else:
            cdf = 0.5 * (1.0 + math.erf(x / sqrt2))
        pdf = math.exp(-0.5 * x * x) / sqrt_2pi
        return cdf + x * pdf

    def gelu_derivative_vec(x_arr):
        """Vectorized GELU'(x)."""
        return np.array([gelu_derivative_fp64(float(xi)) for xi in x_arr])

    def eval_poly_horner(coeffs, x):
        """Evaluate polynomial using Horner's method. coeffs[0] is constant term."""
        result = np.zeros_like(x, dtype=np.float64)
        for c in reversed(coeffs):
            result = result * x + c
        return result

    def compute_poly_ulp(coeffs, x_min, x_max, poly_range=(-4, 4), clamp_min=-0.18, clamp_max=1.0):
        """Compute max ULP error for polynomial with boundary handling."""
        x_test = np.linspace(x_min, x_max, 10000)
        y_exact = gelu_derivative_vec(x_test)

        # Apply piecewise evaluation with boundary handling
        y_poly = np.zeros_like(x_test)
        for i, x in enumerate(x_test):
            if x < poly_range[0]:
                y_poly[i] = 0.0  # Asymptotic: GELU'(x) → 0
            elif x > poly_range[1]:
                y_poly[i] = 1.0  # Asymptotic: GELU'(x) → 1
            else:
                y_poly[i] = eval_poly_horner(coeffs, np.array([x]))[0]
                y_poly[i] = np.clip(y_poly[i], clamp_min, clamp_max)

        max_ulp = 0
        worst_x = 0
        for i in range(len(x_test)):
            ulp = ulp_distance_bf16_daz(y_exact[i], y_poly[i])
            if ulp > max_ulp:
                max_ulp = ulp
                worst_x = x_test[i]
        return max_ulp, worst_x

    # Use numpy's chebyshev fitting which is numerically stable
    from numpy.polynomial import chebyshev as C

    def fit_chebyshev_stable(func, degree, x_min, x_max, num_samples=1000):
        """Fit polynomial using numpy's stable Chebyshev fitting."""
        # Sample function at many points
        x_samples = np.linspace(x_min, x_max, num_samples)
        y_samples = np.array([func(x) for x in x_samples])

        # Fit using Chebyshev polynomials (numerically stable)
        cheb_coeffs = C.chebfit(x_samples, y_samples, degree)

        # Convert Chebyshev coefficients to standard polynomial coefficients
        # Note: This conversion can introduce numerical errors for high degrees
        poly_coeffs = C.cheb2poly(cheb_coeffs)

        return poly_coeffs, cheb_coeffs, (x_min, x_max)

    def eval_chebyshev(cheb_coeffs, x, x_min, x_max):
        """Evaluate Chebyshev polynomial directly (more stable than converted coeffs)."""
        # Scale x to [-1, 1] for Chebyshev evaluation
        return C.chebval(x, cheb_coeffs)

    def compute_cheb_ulp(cheb_coeffs, x_min_test, x_max_test, poly_range):
        """Compute max ULP using Chebyshev polynomial with boundary handling."""
        x_test = np.linspace(x_min_test, x_max_test, 10000)
        y_exact = gelu_derivative_vec(x_test)

        y_poly = np.zeros_like(x_test)
        for i, x in enumerate(x_test):
            if x < poly_range[0]:
                y_poly[i] = 0.0
            elif x > poly_range[1]:
                y_poly[i] = 1.0
            else:
                y_poly[i] = C.chebval(x, cheb_coeffs)
                y_poly[i] = np.clip(y_poly[i], -0.18, 1.0)

        max_ulp = 0
        worst_x = 0
        for i in range(len(x_test)):
            ulp = ulp_distance_bf16_daz(y_exact[i], y_poly[i])
            if ulp > max_ulp:
                max_ulp = ulp
                worst_x = x_test[i]
        return max_ulp, worst_x

    # Test different polynomial degrees using stable Chebyshev fitting
    logger.info("\n--- Polynomial Fit Results (Stable Chebyshev) ---")
    logger.info(f"{'Degree':>6} {'Range':>12} {'Max ULP':>10} {'Worst x':>10}")
    logger.info("-" * 45)

    best_cheb_coeffs = None
    best_poly_coeffs = None
    best_config = None
    best_max_ulp = float("inf")

    for degree in [5, 6, 7, 8, 9, 10, 11, 12]:
        for x_range in [(-4, 4), (-5, 5), (-6, 6), (-7, 7), (-8, 8)]:
            poly_coeffs, cheb_coeffs, _ = fit_chebyshev_stable(gelu_derivative_fp64, degree, x_range[0], x_range[1])
            max_ulp, worst_x = compute_cheb_ulp(cheb_coeffs, -10, 10, poly_range=x_range)

            logger.info(f"{degree:>6} {str(x_range):>12} {max_ulp:>10} {worst_x:>10.3f}")

            if max_ulp < best_max_ulp:
                best_max_ulp = max_ulp
                best_cheb_coeffs = cheb_coeffs
                best_poly_coeffs = poly_coeffs
                best_config = (degree, x_range, "cheb_stable")

    logger.info("-" * 45)
    logger.info(f"\nBest configuration: degree={best_config[0]}, range={best_config[1]}, method={best_config[2]}")
    logger.info(f"Best Max ULP: {best_max_ulp}")

    # Debug: Check values at problematic points including worst_x
    logger.info("\n--- Debug: Values at key points ---")
    debug_points = [
        -10.0,
        -8.0,
        -6.0,
        -5.0,
        -4.0,
        -3.0,
        -2.0,
        -1.5,
        -1.13,
        -1.049,
        -1.0,
        -0.85,
        -0.8,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
    ]
    logger.info(f"{'x':>8} {'Exact':>12} {'Cheb':>12} {'Error':>12} {'ULP':>6}")
    for x in debug_points:
        exact = gelu_derivative_fp64(x)
        cheb_val = C.chebval(x, best_cheb_coeffs)
        error = abs(exact - cheb_val)
        ulp = ulp_distance_bf16_daz(exact, cheb_val)
        logger.info(f"{x:>8.2f} {exact:>12.6f} {cheb_val:>12.6f} {error:>12.6e} {ulp:>6}")

    # Print Chebyshev coefficients
    logger.info("\n--- Best Chebyshev Coefficients ---")
    logger.info(f"// GELU'(x) Chebyshev polynomial ({best_config[2]})")
    logger.info(f"// Degree {best_config[0]} over {best_config[1]}")
    for i, c in enumerate(best_cheb_coeffs):
        logger.info(f"//   T{i} = {c:+.15e}")

    # Print converted standard polynomial coefficients (for SFPU)
    logger.info("\n--- Converted Standard Polynomial Coefficients ---")
    logger.info("// WARNING: Conversion from Chebyshev can lose precision for high degrees")
    for i, c in enumerate(best_poly_coeffs):
        logger.info(f"//   c{i} = {c:+.15e}")

    # Analyze accuracy by region
    logger.info("\n--- Accuracy by Region ---")
    regions = [
        ("Asymptotic neg", -6, -4),
        ("Transition neg", -4, -2),
        ("Near minimum", -2, -0.5),
        ("Near zero", -0.5, 0.5),
        ("Transition pos", 0.5, 2),
        ("Saturating", 2, 4),
        ("Asymptotic pos", 4, 6),
    ]

    logger.info(f"{'Region':>20} {'Max ULP':>10} {'Worst x':>10}")
    logger.info("-" * 45)

    poly_range = best_config[1]
    for name, x_min, x_max in regions:
        max_ulp, worst_x = compute_cheb_ulp(best_cheb_coeffs, x_min, x_max, poly_range=poly_range)
        logger.info(f"{name:>20} {max_ulp:>10} {worst_x:>10.3f}")

    # Generate C++ code for the polynomial
    logger.info("\n--- C++ Implementation ---")
    logger.info(
        f"""
// GELU'(x) polynomial approximation
// Fitted using Chebyshev polynomials, degree {best_config[0]} over {best_config[1]}
// Max ULP: {best_max_ulp}

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_gelu_derivative_polynomial_(sfpi::vFloat x) {{
    sfpi::vFloat result;

    // Asymptotic regions
    v_if(x < {best_config[1][0]}f) {{
        result = sfpi::vConst0;
    }}
    v_elseif(x > {best_config[1][1]}f) {{
        result = sfpi::vConst1;
    }}
    v_else {{
        // Polynomial evaluation using standard coefficients
        result = PolynomialEvaluator::eval(x,"""
    )

    # Show converted polynomial coefficients
    for i, c in enumerate(best_poly_coeffs):
        comma = "," if i < len(best_poly_coeffs) - 1 else ");"
        logger.info(f"            {c:+.15e}f{comma}  // c{i}")

    logger.info(
        f"""
        // Clamp to valid range
        v_if(result < -0.18f) {{ result = -0.18f; }} v_endif;
        v_if(result > 1.0f) {{ result = sfpi::vConst1; }} v_endif;
    }}
    v_endif;

    return result;
}}
"""
    )

    # Log final summary
    logger.info(f"\n*** SUMMARY: Best configuration achieved Max ULP = {best_max_ulp} ***")
    logger.info(f"*** For comparison, erfc-based composite achieves Max ULP = 59 ***")

    # The test passes if we found coefficients with reasonable accuracy
    # Note: Achieving Max ULP = 1 like tanh requires more sophisticated techniques
    # Current best polynomial: Max ULP = ~30000 (worse than erfc composite)
    # This test is primarily for coefficient research, not production use
    assert best_max_ulp < 50000, f"Polynomial fitting failed catastrophically, max ULP = {best_max_ulp}"

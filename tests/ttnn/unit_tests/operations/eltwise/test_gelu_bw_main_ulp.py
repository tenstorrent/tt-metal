# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

Run: pytest tests/ttnn/unit_tests/operations/eltwise/test_gelu_bw_main_ulp.py -v -s
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
    """Correctness guard: verifies GELU'(0) = 0.5 via full Python → TTNN → device round-trip.
    Catches kernel crash, wrong formula sign, or broken CDF constant."""

    def test_derivative_at_zero(self, device):
        """GELU'(0) = 0.5"""
        input_val = 0.0
        grad_val = 1.0

        torch_input = torch.tensor([[input_val]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[grad_val]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        results = ttnn.gelu_bw(tt_grad, tt_input, approximate="none")
        actual = ttnn.to_torch(results[0]).item()

        expected = gelu_derivative_expected_bf16_daz(input_val)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.debug(f"x=0: expected={expected:.4f}, actual={actual:.4f}, ULP={ulp_error}")

        assert ulp_error <= 2, f"Expected ULP <= 2, got {ulp_error}"


class TestGeluBwPositiveValues:
    """Correctness guard: 6 positive-side points where GELU'(x) approaches 1.
    Per-point ULP threshold (2). Catches broken positive saturation path."""

    @pytest.mark.parametrize(
        "input_value,max_expected_ulp",
        [
            (0.5, 2),
            (1.0, 2),
            (2.0, 2),
            (3.0, 2),
            (5.0, 2),
            (10.0, 2),
        ],
    )
    def test_positive_values(self, device, input_value, max_expected_ulp):
        """For large positive x, GELU'(x) approaches 1."""
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[1.0]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        results = ttnn.gelu_bw(tt_grad, tt_input, approximate="none")
        actual = ttnn.to_torch(results[0]).item()

        expected = gelu_derivative_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.debug(f"x={input_value}: expected={expected:.4f}, actual={actual:.4f}, ULP={ulp_error}")

        assert ulp_error <= max_expected_ulp, f"Expected ULP <= {max_expected_ulp}, got {ulp_error}"


class TestGeluBwNegativeValues:
    """Correctness guard: 8 negative-side points with tight per-point thresholds (2 ULP).
    Catches precision regression at any negative sample point."""

    @pytest.mark.parametrize(
        "input_value,max_expected_ulp",
        [
            (-0.5, 2),
            (-1.0, 2),
            (-2.0, 2),
            (-3.0, 2),
            (-4.0, 2),
            (-5.0, 2),
            (-6.0, 2),
            (-8.0, 2),
        ],
    )
    def test_negative_values(self, device, input_value, max_expected_ulp):
        """For large negative x, GELU'(x) approaches 0."""
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[1.0]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        results = ttnn.gelu_bw(tt_grad, tt_input, approximate="none")
        actual = ttnn.to_torch(results[0]).item()

        expected = gelu_derivative_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.debug(f"x={input_value}: expected={expected:.6e}, actual={actual:.6e}, ULP={ulp_error}")

        assert ulp_error <= max_expected_ulp, f"Expected ULP <= {max_expected_ulp}, got {ulp_error}"


class TestGeluBwNearZero:
    """Correctness guard: 7 near-zero points where GELU'(x) ≈ 0.5 + x/sqrt(2π).
    Tight threshold (2 ULP). Catches DAZ flush bugs or broken small-value handling."""

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

        results = ttnn.gelu_bw(tt_grad, tt_input, approximate="none")
        actual = ttnn.to_torch(results[0]).item()

        expected = gelu_derivative_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.debug(f"x={input_value:.2e}: expected={expected:.4f}, actual={actual:.4f}, ULP={ulp_error}")

        assert ulp_error <= 2, f"Expected ULP <= 2, got {ulp_error}"


class TestGeluBwLocalMinimum:
    """Correctness guard: 5 points near the GELU derivative's zero-crossing at
    x ≈ -0.751. Uses absolute error threshold (0.01) because near the zero-crossing,
    the expected value is near 0 and even tiny absolute errors produce large ULPs."""

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

        results = ttnn.gelu_bw(tt_grad, tt_input, approximate="none")
        actual = ttnn.to_torch(results[0]).item()

        expected = gelu_derivative_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.debug(f"x={input_value}: expected={expected:.6f}, actual={actual:.6f}, ULP={ulp_error}")

        # Near the zero-crossing, expected ≈ 0 so ULP can be large even for tiny absolute errors
        assert (
            ulp_error <= 2 or abs(actual - expected) < 0.01
        ), f"ULP {ulp_error} and abs error {abs(actual - expected):.6f} both exceed thresholds"


class TestGeluBwWithGradientScaling:
    """Correctness guard (unique): only tests using grad != 1.0 (grad=0.5, 2.0, -1.0, 0.1, 10.0).
    Catches swapped grad/input tensors or missing gradient multiplication in backward pass."""

    @pytest.mark.parametrize(
        "input_value,grad_value,max_expected_ulp",
        [
            (1.0, 2.0, 2),
            (-1.0, 0.5, 2),
            (0.0, 1.0, 2),
            (2.0, -1.0, 2),
            (0.5, 3.0, 2),
        ],
    )
    def test_with_gradient(self, device, input_value, grad_value, max_expected_ulp):
        """Test GELU backward with different gradient values."""
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[grad_value]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        results = ttnn.gelu_bw(tt_grad, tt_input, approximate="none")
        actual = ttnn.to_torch(results[0]).item()

        expected = gelu_bw_expected_bf16_daz(grad_value, input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.debug(
            f"x={input_value}, grad={grad_value}: expected={expected:.4f}, actual={actual:.4f}, ULP={ulp_error}"
        )

        assert ulp_error <= max_expected_ulp, f"Expected ULP <= {max_expected_ulp}, got {ulp_error}"


def test_gelu_bw_ulp_summary(device):
    """Correctness guard: comprehensive summary of GELU backward ULP across all regions.
    Tests 6 key points (zero, ±1, local min, ±5) and asserts max ULP <= 2.
    Catches any single broken code path quickly via representative sampling."""
    logger.debug("")
    logger.debug("=" * 100)
    logger.debug("GELU BACKWARD ULP SUMMARY (DAZ+FTZ MODEL)")
    logger.debug("=" * 100)

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

    logger.debug("")
    logger.debug(f"{'Description':>20} | {'x':>10} | {'Expected':>12} | {'Actual':>12} | {'ULP':>8}")
    logger.debug("-" * 70)

    max_ulp = 0
    worst_x = 0

    for desc, x in test_points:
        torch_input = torch.tensor([[x]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[1.0]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        results = ttnn.gelu_bw(tt_grad, tt_input, approximate="none")
        actual = ttnn.to_torch(results[0]).item()

        expected = gelu_derivative_expected_bf16_daz(x)
        ulp = ulp_distance_bf16_daz(actual, expected)

        if ulp > max_ulp:
            max_ulp = ulp
            worst_x = x

        logger.debug(f"{desc:>20} | {x:>10.4f} | {expected:>12.6f} | {actual:>12.6f} | {ulp:>8}")

    logger.debug("-" * 70)
    logger.debug(f"Max ULP: {max_ulp} at x = {worst_x}")
    logger.debug("")
    logger.debug("Expected behavior:")
    logger.debug("- GELU'(0) ≈ 0.5")
    logger.debug("- GELU'(x) → 1 as x → +∞")
    logger.debug("- GELU'(x) → 0 as x → -∞")
    logger.debug("- Local minimum near x ≈ -0.751 where GELU'(x) ≈ 0")
    logger.debug("=" * 100)

    # Regression guard: all key points must be within 5 ULP
    assert max_ulp <= 2, (
        f"Max ULP {max_ulp} at x={worst_x} exceeds threshold 2. " f"See table above for per-point details."
    )


# =============================================================================
# Tanh Approximation Tests
# =============================================================================


def gelu_derivative_tanh_exact(x: float) -> float:
    """
    Exact GELU derivative using tanh approximation, computed with mpmath 256-bit precision.

    GELU_tanh(x) = 0.5 * x * (1 + tanh(beta * (x + kappa * x^3)))
    GELU_tanh'(x) = 0.5 * (1 + tanh(inner)) + 0.5 * x * (1 - tanh^2(inner)) * beta * (1 + 3*kappa*x^2)

    where beta = sqrt(2/pi), kappa = 0.044715, inner = beta * (x + kappa * x^3)
    """
    mp.prec = 256
    x_mp = mp.mpf(x)
    beta = mp_sqrt(mp.mpf(2) / mp.pi)
    kappa = mp.mpf("0.044715")

    inner = beta * (x_mp + kappa * x_mp**3)
    tanh_inner = mp.tanh(inner)

    cdf_term = mp.mpf("0.5") * (1 + tanh_inner)
    pdf_term = mp.mpf("0.5") * x_mp * (1 - tanh_inner**2) * beta * (1 + 3 * kappa * x_mp**2)

    result = cdf_term + pdf_term
    return float(result)


def gelu_derivative_tanh_expected_bf16_daz(x: float) -> float:
    """Compute expected BF16 GELU derivative (tanh approx) with DAZ+FTZ applied."""
    x_bits = bf16_daz_normalize(float_to_bf16_bits(x))
    x_daz = bf16_bits_to_float(x_bits)
    result = gelu_derivative_tanh_exact(x_daz)
    result_bits = bf16_daz_normalize(float_to_bf16_bits(result))
    return bf16_bits_to_float(result_bits)


def gelu_bw_tanh_expected_bf16_daz(grad: float, x: float) -> float:
    """Compute expected BF16 GELU backward (tanh approx) with DAZ+FTZ applied."""
    grad_bits = bf16_daz_normalize(float_to_bf16_bits(grad))
    grad_daz = bf16_bits_to_float(grad_bits)
    x_bits = bf16_daz_normalize(float_to_bf16_bits(x))
    x_daz = bf16_bits_to_float(x_bits)
    result = grad_daz * gelu_derivative_tanh_exact(x_daz)
    result_bits = bf16_daz_normalize(float_to_bf16_bits(result))
    return bf16_bits_to_float(result_bits)


class TestGeluBwTanhDerivativeAtZero:
    """Correctness guard: verifies GELU_tanh'(0) = 0.5 via full Python → TTNN → device round-trip."""

    def test_derivative_at_zero(self, device):
        """GELU_tanh'(0) = 0.5"""
        input_val = 0.0
        grad_val = 1.0

        torch_input = torch.tensor([[input_val]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[grad_val]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        results = ttnn.gelu_bw(tt_grad, tt_input, approximate="tanh")
        actual = ttnn.to_torch(results[0]).item()

        expected = gelu_derivative_tanh_expected_bf16_daz(input_val)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.debug(f"[tanh] x=0: expected={expected:.4f}, actual={actual:.4f}, ULP={ulp_error}")

        assert ulp_error <= 2, f"Expected ULP <= 2, got {ulp_error}"


class TestGeluBwTanhPositiveValues:
    """Correctness guard: positive-side points for tanh approximation.
    The tanh approximation accumulates more rounding error than the polynomial,
    so we allow up to 4 ULP."""

    @pytest.mark.parametrize(
        "input_value,max_expected_ulp",
        [
            (0.5, 4),
            (1.0, 4),
            (2.0, 4),
            (3.0, 4),
            (5.0, 4),
            (10.0, 4),
        ],
    )
    def test_positive_values(self, device, input_value, max_expected_ulp):
        """For large positive x, GELU_tanh'(x) approaches 1."""
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[1.0]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        results = ttnn.gelu_bw(tt_grad, tt_input, approximate="tanh")
        actual = ttnn.to_torch(results[0]).item()

        expected = gelu_derivative_tanh_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.debug(f"[tanh] x={input_value}: expected={expected:.4f}, actual={actual:.4f}, ULP={ulp_error}")

        assert ulp_error <= max_expected_ulp, f"Expected ULP <= {max_expected_ulp}, got {ulp_error}"


class TestGeluBwTanhNegativeValues:
    """Correctness guard: negative-side points for tanh approximation.
    For large negative x, GELU_tanh'(x) → 0. In BF16, intermediate tanh saturation
    causes residual errors that are tiny in absolute terms but large in ULP (since
    the reference is near zero). We use absolute error for x <= -2."""

    @pytest.mark.parametrize(
        "input_value",
        [-0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -8.0],
    )
    def test_negative_values(self, device, input_value):
        """For large negative x, GELU_tanh'(x) approaches 0."""
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[1.0]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        results = ttnn.gelu_bw(tt_grad, tt_input, approximate="tanh")
        actual = ttnn.to_torch(results[0]).item()

        expected = gelu_derivative_tanh_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)
        abs_error = abs(actual - expected)

        logger.debug(
            f"[tanh] x={input_value}: expected={expected:.6e}, actual={actual:.6e}, "
            f"ULP={ulp_error}, abs_err={abs_error:.6e}"
        )

        # Near-zero region (x <= -2): expected ≈ 0, ULP is misleading, use absolute error.
        # Mild region (x > -2): ULP is meaningful.
        if input_value <= -2.0:
            assert abs_error < 0.02, f"Absolute error {abs_error:.6e} exceeds 0.02 at x={input_value}"
        else:
            assert ulp_error <= 4, f"Expected ULP <= 4, got {ulp_error}"


class TestGeluBwTanhNearZero:
    """Correctness guard: near-zero points for tanh approximation."""

    @pytest.mark.parametrize(
        "input_value",
        [1e-6, 1e-4, 0.01, 0.1, -0.1, -0.01, -1e-4],
    )
    def test_near_zero(self, device, input_value):
        """Near zero, GELU_tanh'(x) ≈ 0.5 + x*sqrt(2/pi)."""
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[1.0]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        results = ttnn.gelu_bw(tt_grad, tt_input, approximate="tanh")
        actual = ttnn.to_torch(results[0]).item()

        expected = gelu_derivative_tanh_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.debug(f"[tanh] x={input_value:.2e}: expected={expected:.4f}, actual={actual:.4f}, ULP={ulp_error}")

        assert ulp_error <= 4, f"Expected ULP <= 4, got {ulp_error}"


class TestGeluBwTanhLocalMinimum:
    """Correctness guard: points near the tanh-GELU derivative's zero-crossing.
    Uses absolute error threshold because near the zero-crossing, even tiny
    absolute errors produce large ULPs."""

    @pytest.mark.parametrize(
        "input_value",
        [-0.7, -0.75, -0.76, -0.8],
    )
    def test_local_minimum_region(self, device, input_value):
        """GELU_tanh has a local minimum near x ≈ -0.75 where derivative ≈ 0."""
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[1.0]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        results = ttnn.gelu_bw(tt_grad, tt_input, approximate="tanh")
        actual = ttnn.to_torch(results[0]).item()

        expected = gelu_derivative_tanh_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.debug(f"[tanh] x={input_value}: expected={expected:.6f}, actual={actual:.6f}, ULP={ulp_error}")

        assert (
            ulp_error <= 4 or abs(actual - expected) < 0.02
        ), f"ULP {ulp_error} and abs error {abs(actual - expected):.6f} both exceed thresholds"


class TestGeluBwTanhWithGradientScaling:
    """Correctness guard: tests using grad != 1.0 for tanh approximation.
    Catches swapped grad/input tensors or missing gradient multiplication."""

    @pytest.mark.parametrize(
        "input_value,grad_value,max_expected_ulp",
        [
            (1.0, 2.0, 4),
            (-1.0, 0.5, 4),
            (0.0, 1.0, 4),
            (2.0, -1.0, 4),
            (0.5, 3.0, 4),
        ],
    )
    def test_with_gradient(self, device, input_value, grad_value, max_expected_ulp):
        """Test GELU backward (tanh) with different gradient values."""
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[grad_value]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        results = ttnn.gelu_bw(tt_grad, tt_input, approximate="tanh")
        actual = ttnn.to_torch(results[0]).item()

        expected = gelu_bw_tanh_expected_bf16_daz(grad_value, input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.debug(
            f"[tanh] x={input_value}, grad={grad_value}: expected={expected:.4f}, actual={actual:.4f}, ULP={ulp_error}"
        )

        assert ulp_error <= max_expected_ulp, f"Expected ULP <= {max_expected_ulp}, got {ulp_error}"


def test_gelu_bw_tanh_ulp_summary(device):
    """Correctness guard: comprehensive summary of GELU backward (tanh approx) ULP across all regions.
    Tests key points and asserts max ULP <= 4."""
    logger.debug("")
    logger.debug("=" * 100)
    logger.debug("GELU BACKWARD (TANH APPROX) ULP SUMMARY (DAZ+FTZ MODEL)")
    logger.debug("=" * 100)

    test_points = [
        ("Zero", 0.0),
        ("Small positive", 0.1),
        ("Unity", 1.0),
        ("Moderate positive", 2.0),
        ("Large positive", 5.0),
        ("Small negative", -0.1),
        ("Negative unity", -1.0),
        ("Local minimum", -0.75),
        ("Moderate negative", -2.0),
        ("Large negative", -5.0),
        ("Deep negative", -8.0),
    ]

    logger.debug("")
    logger.debug(f"{'Description':>20} | {'x':>10} | {'Expected':>12} | {'Actual':>12} | {'ULP':>8} | {'AbsErr':>10}")
    logger.debug("-" * 80)

    max_ulp = 0
    worst_x = 0
    max_abs_err = 0.0
    worst_abs_x = 0.0
    all_ulps = []

    for desc, x in test_points:
        torch_input = torch.tensor([[x]], dtype=torch.bfloat16)
        torch_grad = torch.tensor([[1.0]], dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        results = ttnn.gelu_bw(tt_grad, tt_input, approximate="tanh")
        actual = ttnn.to_torch(results[0]).item()

        expected = gelu_derivative_tanh_expected_bf16_daz(x)
        ulp = ulp_distance_bf16_daz(actual, expected)
        abs_err = abs(actual - expected)
        all_ulps.append(ulp)

        if ulp > max_ulp:
            max_ulp = ulp
            worst_x = x
        if abs_err > max_abs_err:
            max_abs_err = abs_err
            worst_abs_x = x

        logger.debug(f"{desc:>20} | {x:>10.4f} | {expected:>12.6f} | {actual:>12.6f} | {ulp:>8} | {abs_err:.2e}")

    logger.debug("-" * 70)
    logger.debug(f"Max ULP: {max_ulp} at x = {worst_x}")
    logger.debug(f"Max abs error: {max_abs_err:.6e} at x = {worst_abs_x}")
    logger.debug("")
    logger.debug("Expected behavior (tanh approximation):")
    logger.debug("- GELU_tanh'(0) ≈ 0.5")
    logger.debug("- GELU_tanh'(x) → 1 as x → +∞")
    logger.debug("- GELU_tanh'(x) → 0 as x → -∞")
    logger.debug("- Local minimum near x ≈ -0.75 where GELU_tanh'(x) ≈ 0")
    logger.debug("=" * 100)

    # For points where expected ≈ 0 (large negative x), ULP is misleading.
    # Check that absolute error is bounded and ULP is bounded for non-saturation points.
    assert max_abs_err < 0.02, f"Max abs error {max_abs_err:.6e} at x={worst_abs_x} exceeds 0.02."
    # ULP threshold only meaningful for x > -2 where expected is not near-zero
    non_saturated_ulps = [u for (u, x) in zip(all_ulps, [p[1] for p in test_points]) if x > -2.0]
    if non_saturated_ulps:
        max_ns_ulp = max(non_saturated_ulps)
        assert max_ns_ulp <= 4, f"Max ULP {max_ns_ulp} for non-saturated region exceeds threshold 4."

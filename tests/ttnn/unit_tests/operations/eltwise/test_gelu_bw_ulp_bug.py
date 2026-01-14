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

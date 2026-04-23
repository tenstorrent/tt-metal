# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tanh Backward ULP Precision Tests

This test validates the accuracy of ttnn.tanh_bw (sech²(x) · grad) across the
BFloat16 range using mpmath 256-bit precision as the independent reference.

Validates the operation through the Python binding round-trip
(ttnn.from_torch → ttnn.tanh_bw → ttnn.to_torch). The companion C++ test
(test_tanh_bw_ulp.cpp) covers exhaustive ~65K BF16 sweep + statistical analysis;
this Python test covers sample-point correctness through the public binding.

MATHEMATICAL FORMULA:
tanh'(x) = sech²(x) = 1 / cosh²(x)
tanh_bw(grad, x) = grad * sech²(x)

Hardware Model: Tenstorrent SFPU uses DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero)
Per tech_reports/Handling_Special_Value/special_values.md: "denormals | all | 0x0"

Run: pytest tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_ulp.py -v -s
"""

import struct
import pytest
import torch
import ttnn
from loguru import logger
from mpmath import mp, cosh as mp_cosh


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


def bf16_quantize_rne(x: float) -> float:
    """RNE-quantize a float to BF16 (matches torch's BFloat16 conversion).
    Required because the bit-level helpers above truncate, but torch — and therefore
    the device input — uses round-to-nearest-even. For test points that are not
    exact BF16 values (e.g., 2.9, 3.01), truncation and RNE diverge."""
    return float(torch.tensor([x], dtype=torch.bfloat16).item())


def sech2_exact(x: float) -> float:
    """
    Exact tanh derivative using mpmath 256-bit precision.

    tanh'(x) = sech²(x) = 1 / cosh²(x)

    Uses 1/cosh²(x) form (not 1 - tanh²(x)) to avoid the catastrophic cancellation
    that motivated this PR's existence (the original buggy composite kernel).
    """
    mp.prec = 256
    x_mp = mp.mpf(x)
    cosh_x = mp_cosh(x_mp)
    result = 1 / (cosh_x * cosh_x)
    return float(result)


def tanh_derivative_expected_bf16_daz(x: float) -> float:
    """Compute expected BF16 tanh derivative (sech²(x)) with DAZ+FTZ applied.
    Quantizes the input via torch RNE so the reference matches what the device sees."""
    x_bf16 = bf16_quantize_rne(x)
    x_bits = bf16_daz_normalize(float_to_bf16_bits(x_bf16))
    x_daz = bf16_bits_to_float(x_bits)
    result = sech2_exact(x_daz)
    result_bits = bf16_daz_normalize(float_to_bf16_bits(result))
    return bf16_bits_to_float(result_bits)


def tanh_bw_expected_bf16_daz(grad: float, x: float) -> float:
    """Compute expected BF16 tanh backward with DAZ+FTZ applied.
    Quantizes inputs via torch RNE so the reference matches what the device sees."""
    grad_bf16 = bf16_quantize_rne(grad)
    x_bf16 = bf16_quantize_rne(x)
    grad_bits = bf16_daz_normalize(float_to_bf16_bits(grad_bf16))
    grad_daz = bf16_bits_to_float(grad_bits)
    x_bits = bf16_daz_normalize(float_to_bf16_bits(x_bf16))
    x_daz = bf16_bits_to_float(x_bits)
    result = grad_daz * sech2_exact(x_daz)
    result_bits = bf16_daz_normalize(float_to_bf16_bits(result))
    return bf16_bits_to_float(result_bits)


def _run_tanh_bw(device, x: float, grad: float = 1.0) -> float:
    """Common helper: run ttnn.tanh_bw through the Python binding round-trip."""
    torch_input = torch.tensor([[x]], dtype=torch.bfloat16)
    torch_grad = torch.tensor([[grad]], dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    results = ttnn.tanh_bw(tt_grad, tt_input)
    return ttnn.to_torch(results[0]).item()


# =============================================================================
# Test Classes
# =============================================================================


class TestTanhBwDerivativeAtZero:
    """Correctness guard: verifies sech²(0) = 1.0 via full Python → TTNN → device round-trip.
    Catches kernel crash, wrong formula sign, or broken polynomial constant term."""

    def test_derivative_at_zero(self, device):
        """sech²(0) = 1.0"""
        x = 0.0
        actual = _run_tanh_bw(device, x)
        expected = tanh_derivative_expected_bf16_daz(x)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x=0: expected={expected:.4f}, actual={actual:.4f}, ULP={ulp_error}")

        assert ulp_error <= 2, f"Expected ULP <= 2, got {ulp_error}"


class TestTanhBwPositiveValues:
    """Correctness guard: 6 positive-side points where sech²(x) decays toward 0.
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
        """For increasing positive x, sech²(x) decays toward 0."""
        actual = _run_tanh_bw(device, input_value)
        expected = tanh_derivative_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x={input_value}: expected={expected:.6e}, actual={actual:.6e}, ULP={ulp_error}")

        assert ulp_error <= max_expected_ulp, f"Expected ULP <= {max_expected_ulp}, got {ulp_error}"


class TestTanhBwNegativeValues:
    """Correctness guard: 6 negative-side points (sech² is symmetric: sech²(-x)=sech²(x)).
    Catches kernel mishandling of negative inputs (e.g., missing abs(x))."""

    @pytest.mark.parametrize(
        "input_value,max_expected_ulp",
        [
            (-0.5, 2),
            (-1.0, 2),
            (-2.0, 2),
            (-3.0, 2),
            (-5.0, 2),
            (-10.0, 2),
        ],
    )
    def test_negative_values(self, device, input_value, max_expected_ulp):
        """sech² is symmetric — negative x must give the same result as positive."""
        actual = _run_tanh_bw(device, input_value)
        expected = tanh_derivative_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x={input_value}: expected={expected:.6e}, actual={actual:.6e}, ULP={ulp_error}")

        assert ulp_error <= max_expected_ulp, f"Expected ULP <= {max_expected_ulp}, got {ulp_error}"


class TestTanhBwNearZero:
    """Correctness guard: 7 near-zero points where sech²(x) ≈ 1 - x².
    Tight threshold (2 ULP). Catches DAZ flush bugs or broken small-value handling."""

    @pytest.mark.parametrize(
        "input_value",
        [1e-6, 1e-4, 0.01, 0.1, -0.1, -0.01, -1e-4],
    )
    def test_near_zero(self, device, input_value):
        """Near zero, sech²(x) ≈ 1 - x²."""
        actual = _run_tanh_bw(device, input_value)
        expected = tanh_derivative_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x={input_value:.2e}: expected={expected:.6f}, actual={actual:.6f}, ULP={ulp_error}")

        assert ulp_error <= 2, f"Expected ULP <= 2, got {ulp_error}"


class TestTanhBwBoundaryRegion:
    """Correctness guard (tanh_bw-specific): points near |x| = 3, where the SFPU
    kernel switches from the polynomial core to the inline Cody-Waite exp tail.
    Catches polynomial/exp boundary discontinuities."""

    @pytest.mark.parametrize(
        "input_value",
        [2.9, 2.99, 3.0, 3.01, 3.1, -2.99, -3.0, -3.01],
    )
    def test_boundary_region(self, device, input_value):
        """Region transition: polynomial (|x|<3) → inline exp (3<=|x|<45)."""
        actual = _run_tanh_bw(device, input_value)
        expected = tanh_derivative_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x={input_value}: expected={expected:.6e}, actual={actual:.6e}, ULP={ulp_error}")

        assert ulp_error <= 2, f"Expected ULP <= 2, got {ulp_error}"


class TestTanhBwDeepTail:
    """Correctness guard (tanh_bw-specific): points in the exp tail and near the
    saturation cliff at |x|=45 where exp(-2|x|+ln4) approaches BF16 min normal.
    Catches FP32 FTZ underflow regressions in the inline exp implementation."""

    @pytest.mark.parametrize(
        "input_value",
        [10.0, 20.0, 30.0, 43.75, 44.0, 44.99, 45.0, 50.0, -20.0, -44.0, -45.0],
    )
    def test_deep_tail(self, device, input_value):
        """Exp tail and saturation boundary: kernel must produce 0 for |x|>=45."""
        actual = _run_tanh_bw(device, input_value)
        expected = tanh_derivative_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x={input_value}: expected={expected:.6e}, actual={actual:.6e}, ULP={ulp_error}")

        # Near the BF16 underflow boundary, allow absolute-error tolerance
        # because expected values approach BF16 min normal (1.18e-38)
        assert (
            ulp_error <= 2 or abs(actual - expected) < 1e-37
        ), f"ULP {ulp_error} and abs error {abs(actual - expected):.3e} both exceed thresholds"


class TestTanhBwWithGradientScaling:
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
        """Test tanh backward with different gradient values."""
        actual = _run_tanh_bw(device, input_value, grad_value)
        expected = tanh_bw_expected_bf16_daz(grad_value, input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(
            f"x={input_value}, grad={grad_value}: expected={expected:.6e}, actual={actual:.6e}, ULP={ulp_error}"
        )

        assert ulp_error <= max_expected_ulp, f"Expected ULP <= {max_expected_ulp}, got {ulp_error}"


def test_tanh_bw_ulp_summary(device):
    """Correctness guard: comprehensive summary of tanh backward ULP across all regions.
    Tests representative points (zero, ±1, ±2.99, ±3.01, ±5, ±10, ±44.99) and asserts
    max ULP <= 2. Catches any single broken code path quickly via representative sampling."""
    logger.info("")
    logger.info("=" * 100)
    logger.info("TANH BACKWARD ULP SUMMARY (DAZ+FTZ MODEL)")
    logger.info("=" * 100)

    # Key test points
    test_points = [
        ("Zero", 0.0),
        ("Small positive", 0.1),
        ("Unity", 1.0),
        ("Pre-boundary", 2.99),
        ("Post-boundary", 3.01),
        ("Mid-tail", 5.0),
        ("Deep tail", 10.0),
        ("Near saturation", 44.99),
        ("Small negative", -0.1),
        ("Negative unity", -1.0),
        ("Negative pre-boundary", -2.99),
        ("Negative post-boundary", -3.01),
        ("Mid negative tail", -5.0),
        ("Deep negative tail", -10.0),
    ]

    logger.info("")
    logger.info(f"{'Description':>22} | {'x':>10} | {'Expected':>14} | {'Actual':>14} | {'ULP':>5}")
    logger.info("-" * 78)

    max_ulp = 0
    worst_x = 0

    for desc, x in test_points:
        actual = _run_tanh_bw(device, x)
        expected = tanh_derivative_expected_bf16_daz(x)
        ulp = ulp_distance_bf16_daz(actual, expected)

        if ulp > max_ulp:
            max_ulp = ulp
            worst_x = x

        logger.info(f"{desc:>22} | {x:>10.4f} | {expected:>14.6e} | {actual:>14.6e} | {ulp:>5}")

    logger.info("-" * 78)
    logger.info(f"Max ULP: {max_ulp} at x = {worst_x}")
    logger.info("")
    logger.info("Expected behavior:")
    logger.info("- sech²(0) = 1.0 (peak)")
    logger.info("- sech²(x) → 0 as |x| → ∞")
    logger.info("- sech²(-x) = sech²(x) (even symmetry)")
    logger.info("- Smooth across the kernel's |x|=3 polynomial/exp boundary")
    logger.info("=" * 100)

    assert max_ulp <= 2, (
        f"Max ULP {max_ulp} at x={worst_x} exceeds threshold 2. " f"See table above for per-point details."
    )

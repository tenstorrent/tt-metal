# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tanh Forward ULP Precision Tests

This test validates the accuracy of ttnn.tanh across the BFloat16 range using
mpmath 256-bit precision as the independent reference.

Validates the operation through the Python binding round-trip
(ttnn.from_torch → ttnn.tanh → ttnn.to_torch). The companion C++ test
(test_tanh_fw_ulp.cpp) covers the exhaustive ~65K BF16 sweep + statistical analysis;
this Python test covers sample-point correctness through the public binding.

This is a regression baseline: ttnn.tanh forward already achieves Max ULP = 1
(via polynomial approximation), and this test guards against any regression
that the tanh_bw fix might inadvertently introduce.

MATHEMATICAL FORMULA:
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Hardware Model: Tenstorrent SFPU uses DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero)
Per tech_reports/Handling_Special_Value/special_values.md: "denormals | all | 0x0"

Run: pytest tests/ttnn/unit_tests/operations/eltwise/test_tanh_fw_ulp.py -v -s
"""

import struct
import pytest
import torch
import ttnn
from loguru import logger
from mpmath import mp, tanh as mp_tanh


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


def tanh_exact(x: float) -> float:
    """
    Exact tanh using mpmath 256-bit precision.
    """
    mp.prec = 256
    x_mp = mp.mpf(x)
    result = mp_tanh(x_mp)
    return float(result)


def tanh_expected_bf16_daz(x: float) -> float:
    """Compute expected BF16 tanh with DAZ+FTZ applied.
    Quantizes the input via torch RNE so the reference matches what the device sees."""
    x_bf16 = bf16_quantize_rne(x)
    x_bits = bf16_daz_normalize(float_to_bf16_bits(x_bf16))
    x_daz = bf16_bits_to_float(x_bits)
    result = tanh_exact(x_daz)
    result_bits = bf16_daz_normalize(float_to_bf16_bits(result))
    return bf16_bits_to_float(result_bits)


def _run_tanh(device, x: float) -> float:
    """Common helper: run ttnn.tanh through the Python binding round-trip."""
    torch_input = torch.tensor([[x]], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    result = ttnn.tanh(tt_input)
    return ttnn.to_torch(result).item()


# =============================================================================
# Test Classes
# =============================================================================


class TestTanhFwAtZero:
    """Correctness guard: verifies tanh(0) = 0 via full Python → TTNN → device round-trip.
    Catches kernel crash, wrong sign, or broken zero handling."""

    def test_at_zero(self, device):
        """tanh(0) = 0"""
        x = 0.0
        actual = _run_tanh(device, x)
        expected = tanh_expected_bf16_daz(x)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x=0: expected={expected:.4f}, actual={actual:.4f}, ULP={ulp_error}")

        assert ulp_error <= 2, f"Expected ULP <= 2, got {ulp_error}"


class TestTanhFwPositiveValues:
    """Correctness guard: 6 positive-side points where tanh(x) saturates to +1.
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
        """For increasing positive x, tanh(x) saturates to +1."""
        actual = _run_tanh(device, input_value)
        expected = tanh_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x={input_value}: expected={expected:.6f}, actual={actual:.6f}, ULP={ulp_error}")

        assert ulp_error <= max_expected_ulp, f"Expected ULP <= {max_expected_ulp}, got {ulp_error}"


class TestTanhFwNegativeValues:
    """Correctness guard: 6 negative-side points where tanh(x) saturates to -1.
    tanh is odd: tanh(-x) = -tanh(x). Catches broken sign handling."""

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
        """For increasing negative x, tanh(x) saturates to -1."""
        actual = _run_tanh(device, input_value)
        expected = tanh_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x={input_value}: expected={expected:.6f}, actual={actual:.6f}, ULP={ulp_error}")

        assert ulp_error <= max_expected_ulp, f"Expected ULP <= {max_expected_ulp}, got {ulp_error}"


class TestTanhFwNearZero:
    """Correctness guard: 7 near-zero points where tanh(x) ≈ x.
    Tight threshold (2 ULP). Catches DAZ flush bugs or broken small-value handling."""

    @pytest.mark.parametrize(
        "input_value",
        [1e-6, 1e-4, 0.01, 0.1, -0.1, -0.01, -1e-4],
    )
    def test_near_zero(self, device, input_value):
        """Near zero, tanh(x) ≈ x."""
        actual = _run_tanh(device, input_value)
        expected = tanh_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x={input_value:.2e}: expected={expected:.6e}, actual={actual:.6e}, ULP={ulp_error}")

        assert ulp_error <= 2, f"Expected ULP <= 2, got {ulp_error}"


class TestTanhFwTransitionRegion:
    """Correctness guard: the transition zone (|x| in [2, 5]) where tanh(x) goes from
    'growing fast' to 'approaching ±1'. This is where any forward kernel regression
    is most likely to manifest — the curve is steep and any approximation error
    is visible. Catches polynomial fit regressions and LUT discontinuities."""

    @pytest.mark.parametrize(
        "input_value",
        [2.0, 2.5, 2.9, 3.0, 3.01, 3.1, 3.5, 4.0, 4.5, 5.0, -2.5, -3.0, -3.01, -4.0, -5.0],
    )
    def test_transition_region(self, device, input_value):
        """Steepest part of the curve and onset of saturation."""
        actual = _run_tanh(device, input_value)
        expected = tanh_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x={input_value}: expected={expected:.6f}, actual={actual:.6f}, ULP={ulp_error}")

        assert ulp_error <= 2, f"Expected ULP <= 2, got {ulp_error}"


class TestTanhFwSaturation:
    """Correctness guard: deep saturation region where tanh(x) ≈ ±1 exactly.
    Catches saturation overshoot, sign-flip bugs, or numerical instability at large |x|."""

    @pytest.mark.parametrize(
        "input_value",
        [15.0, 20.0, 50.0, 100.0, -15.0, -20.0, -50.0, -100.0],
    )
    def test_saturation(self, device, input_value):
        """For |x| >> 1, tanh(x) saturates to ±1."""
        actual = _run_tanh(device, input_value)
        expected = tanh_expected_bf16_daz(input_value)
        ulp_error = ulp_distance_bf16_daz(actual, expected)

        logger.info(f"x={input_value}: expected={expected:.6f}, actual={actual:.6f}, ULP={ulp_error}")

        assert ulp_error <= 2, f"Expected ULP <= 2, got {ulp_error}"


def test_tanh_fw_ulp_summary(device):
    """Correctness guard: comprehensive summary of tanh forward ULP across all regions.
    Tests representative points (zero, ±1, ±5, ±20, ±100) and asserts max ULP <= 2.
    Catches any single broken code path quickly via representative sampling."""
    logger.info("")
    logger.info("=" * 100)
    logger.info("TANH FORWARD ULP SUMMARY (DAZ+FTZ MODEL)")
    logger.info("=" * 100)

    # Key test points
    test_points = [
        ("Zero", 0.0),
        ("Small positive", 0.1),
        ("Unity", 1.0),
        ("Mid positive", 2.0),
        ("Mid-tail positive", 5.0),
        ("Saturation positive", 20.0),
        ("Deep saturation positive", 100.0),
        ("Small negative", -0.1),
        ("Negative unity", -1.0),
        ("Mid negative", -2.0),
        ("Mid-tail negative", -5.0),
        ("Saturation negative", -20.0),
        ("Deep saturation negative", -100.0),
    ]

    logger.info("")
    logger.info(f"{'Description':>26} | {'x':>10} | {'Expected':>14} | {'Actual':>14} | {'ULP':>5}")
    logger.info("-" * 82)

    max_ulp = 0
    worst_x = 0

    for desc, x in test_points:
        actual = _run_tanh(device, x)
        expected = tanh_expected_bf16_daz(x)
        ulp = ulp_distance_bf16_daz(actual, expected)

        if ulp > max_ulp:
            max_ulp = ulp
            worst_x = x

        logger.info(f"{desc:>26} | {x:>10.4f} | {expected:>14.6f} | {actual:>14.6f} | {ulp:>5}")

    logger.info("-" * 82)
    logger.info(f"Max ULP: {max_ulp} at x = {worst_x}")
    logger.info("")
    logger.info("Expected behavior:")
    logger.info("- tanh(0) = 0")
    logger.info("- tanh(x) → +1 as x → +∞")
    logger.info("- tanh(x) → -1 as x → -∞")
    logger.info("- tanh(-x) = -tanh(x) (odd symmetry)")
    logger.info("=" * 100)

    assert max_ulp <= 2, (
        f"Max ULP {max_ulp} at x={worst_x} exceeds threshold 2. " f"See table above for per-point details."
    )

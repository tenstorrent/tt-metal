# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_equal,
    assert_with_ulp,
    assert_allclose,
    assert_with_pcc,
    flush_subnormal_values_to_zero,
)
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    generate_bfloat16_bits_in_range,
    flush_to_zero,
    to_tt_tensor,
    MAX_BF16,
    SMALLEST_NORMAL_BF16,
)

pytestmark = pytest.mark.use_module_device

"""
Category 1: basic_unary_math (no extra parameters)
Trigonometric, hyperbolic, comparison, rounding, special math, logical, and utility ops

 1. ttnn.abs              - Absolute value
 2. ttnn.acos             - Arc cosine
 3. ttnn.asin             - Arc sine
 4. ttnn.atan             - Arc tangent
 5. ttnn.atanh            - Inverse hyperbolic tangent
 6. ttnn.cos              - Cosine
 7. ttnn.acosh            - Inverse hyperbolic cosine
 8. ttnn.asinh            - Inverse hyperbolic sine
 9. ttnn.sin              - Sine
10. ttnn.sinh             - Hyperbolic sine
11. ttnn.cosh             - Hyperbolic cosine
12. ttnn.tan              - Tangent
13. ttnn.erfinv           - Inverse error function
14. ttnn.erfc             - Complementary error function
15. ttnn.exp              - Exponential
16. ttnn.exp2             - Base-2 exponential
17. ttnn.expm1            - exp(x) - 1
18. ttnn.floor            - Floor
19. ttnn.ceil             - Ceiling
20. ttnn.trunc            - Truncate
21. ttnn.frac             - Fractional part
22. ttnn.neg              - Negate
23. ttnn.reciprocal       - 1/x
24. ttnn.square           - Square
25. ttnn.cbrt             - Cube root
26. ttnn.sign             - Sign function
27. ttnn.signbit          - Sign bit
28. ttnn.deg2rad          - Degrees to radians
29. ttnn.rad2deg          - Radians to degrees
30. ttnn.i0               - Modified Bessel function (order 0)
31. ttnn.i1               - Modified Bessel function (order 1)
32. ttnn.lgamma           - Log gamma
33. ttnn.digamma          - Digamma function
34. ttnn.multigammaln     - Multivariate log gamma
35. ttnn.eqz              - Equal to zero
36. ttnn.gez              - Greater than or equal to zero
37. ttnn.gtz              - Greater than zero
38. ttnn.lez              - Less than or equal to zero
39. ttnn.ltz              - Less than zero
40. ttnn.nez              - Not equal to zero
41. ttnn.isfinite         - Is finite
42. ttnn.isinf            - Is infinite
43. ttnn.isnan            - Is NaN
44. ttnn.isneginf         - Is negative infinity
45. ttnn.isposinf         - Is positive infinity
46. ttnn.logical_not      - Logical NOT
47. ttnn.identity         - Identity (copy)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Comparison-to-zero ops (output is 0.0 or 1.0) - includes special value checks
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.eqz,
        ttnn.gez,
        ttnn.gtz,
        ttnn.lez,
        ttnn.ltz,
        ttnn.nez,
    ],
)
def test_comparison_to_zero_ops(device, ttnn_op):
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16, include_spl_values=True)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_equal(result, golden)


# ─────────────────────────────────────────────────────────────────────────────
# Finite/Inf/NaN check ops (output is 0.0 or 1.0)
# These ops are tested with include_spl_values=True to exercise inf/nan paths
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.isfinite,
        ttnn.isinf,
        ttnn.isnan,
        ttnn.isneginf,
        ttnn.isposinf,
    ],
)
def test_spl_value_check_ops(device, ttnn_op):
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16, include_spl_values=True)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_equal(result, golden)


# ─────────────────────────────────────────────────────────────────────────────
# Logical NOT
# ─────────────────────────────────────────────────────────────────────────────


def test_logical_not_ops(device):
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16, include_spl_values=True)
    ttnn_op = ttnn.logical_not
    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_equal(result, golden)


# ─────────────────────────────────────────────────────────────────────────────
# Identity, negation, absolute value, sign ops (exact output expected)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, include_spl_values",
    [
        (ttnn.identity, False),
        (ttnn.neg, False),
        (ttnn.abs, False),
        (ttnn.sign, False),
        (ttnn.signbit, True),
    ],
)
def test_identity_neg_abs_sign_ops(device, ttnn_op, include_spl_values):
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16, include_spl_values=include_spl_values)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_equal(result, golden)


# ─────────────────────────────────────────────────────────────────────────────
# Rounding ops (floor, ceil, trunc, frac) - exact output expected
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.floor,
        ttnn.ceil,
        ttnn.trunc,
        ttnn.frac,
    ],
)
def test_rounding_ops(device, ttnn_op):
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_equal(result, golden)


# ─────────────────────────────────────────────────────────────────────────────
# Trigonometric and hyperbolic ops
# Each op is tested with its valid input domain using generate_bfloat16_bits_in_range
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, low, high",
    [
        (ttnn.acos, -1.0, 1.0),
        (ttnn.cos, -10.0, 10.0),
        (ttnn.acosh, 1.0, 100.0),
        (ttnn.asinh, -100.0, 100.0),
        (ttnn.sin, -10.0, 10.0),
        (ttnn.sinh, -9.0, 9.0),
        (ttnn.cosh, -9.0, 9.0),
        (ttnn.tan, -1.45, 1.45),
    ],
)
def test_trig_ops(device, ttnn_op, low, high):
    input_tensor = generate_bfloat16_bits_in_range(low, high)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


# ─────────────────────────────────────────────────────────────────────────────
# Trig ops with output FTZ handling (asin, atan, atanh)
# These ops produce near-zero outputs for small inputs where the device returns
# the smallest normal (2^-126) instead of exact zero. We zero out the device
# result where golden is 0 and device produced the smallest normal.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, low, high",
    [
        (ttnn.asin, -1.0, 1.0),
        (ttnn.atan, -100.0, 100.0),
    ],
)
def test_trig_ops_out_ftz(device, ttnn_op, low, high):
    input_tensor = generate_bfloat16_bits_in_range(low, high)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    # Device SFPU cannot produce exact zero — it returns the smallest normal instead.
    # Flush both outputs: anything at or below smallest normal becomes zero.
    result = flush_to_zero(result)
    golden = flush_to_zero(golden)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


# ─────────────────────────────────────────────────────────────────────────────
# atanh - uses PCC due to inherently higher SFPU error
# ─────────────────────────────────────────────────────────────────────────────


def test_atanh(device):
    input_tensor = generate_bfloat16_bits_in_range(-100, 100)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.atanh)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.atanh(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_pcc(golden, result, pcc=0.999)


# ─────────────────────────────────────────────────────────────────────────────
# deg2rad: multiply by pi/180 (safe for full range but FTZ near zero)
# rad2deg: multiply by 180/pi (large inputs overflow, limit range)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, low, high",
    [
        (ttnn.deg2rad, -1e6, 1e6),
        (ttnn.rad2deg, -5e36, 5e36),
    ],
)
def test_angle_conversion_ops(device, ttnn_op, low, high):
    input_tensor = generate_bfloat16_bits_in_range(low, high)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    result = flush_to_zero(result)
    golden = flush_to_zero(golden)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


# ─────────────────────────────────────────────────────────────────────────────
# erfinv: domain (-1, 1), outputs ±inf at boundaries
# erfc: complementary error function, valid for all finite inputs but clamps
#        to 0 or 2 for large magnitude inputs
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, low, high",
    [
        (ttnn.erfinv, -0.999, 0.999),
        (ttnn.erfc, -10.0, 10.0),
    ],
)
def test_error_functions(device, ttnn_op, low, high):
    input_tensor = generate_bfloat16_bits_in_range(low, high)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_pcc(golden, result, 0.999)


# ─────────────────────────────────────────────────────────────────────────────
# reciprocal: 1/x, swept over both signs
#
# Split at |x| = 2^126, whose reciprocal is the smallest normal BF16.
# Wormhole preserves that boundary result; the other architectures retain
# their existing flush-to-zero behavior there. Larger magnitudes produce
# subnormal reciprocals and are flushed to +0. Test the boundary separately
# so the normal-domain comparison does not mask underflow errors.
# ─────────────────────────────────────────────────────────────────────────────

RECIPROCAL_BOUNDARY_INPUT = 2.0**126
RECIPROCAL_MAX_INPUT = RECIPROCAL_BOUNDARY_INPUT * (1 - 2.0**-8)  # largest bfloat16 below it


@pytest.mark.parametrize(
    "low, high",
    [
        (SMALLEST_NORMAL_BF16, RECIPROCAL_MAX_INPUT),
        (-RECIPROCAL_MAX_INPUT, -SMALLEST_NORMAL_BF16),
    ],
    ids=["positive", "negative"],
)
def test_reciprocal(device, low, high):
    """Every normal bfloat16 of one sign below the underflow boundary.
    This covers |x| < 1, where 1/x amplifies instead of shrinking and the
    output runs all the way up to 2^126, as well as the shrinking half.
    """
    input_tensor = generate_bfloat16_bits_in_range(low, high)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.reciprocal)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.reciprocal(tt_in)
    result = ttnn.to_torch(tt_result)

    ulp_threshold = 0 if device.arch() == ttnn.device.Arch.WORMHOLE_B0 else 1
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp_threshold)


@pytest.mark.parametrize(
    "low, high",
    [
        (RECIPROCAL_BOUNDARY_INPUT, MAX_BF16),
        (-MAX_BF16, -RECIPROCAL_BOUNDARY_INPUT),
    ],
    ids=["positive", "negative"],
)
def test_reciprocal_underflow_boundary(device, low, high):
    """Wormhole returns +/-2^-126 at +/-2^126 and +0 beyond that boundary.
    Other architectures retain their existing +0 result at the boundary.
    Every exact reciprocal in this range is nonzero and BF16-representable
    after rounding, but subnormal outputs are flushed on the device.
    """
    input_tensor = generate_bfloat16_bits_in_range(low, high)

    golden = ttnn.get_golden_function(ttnn.reciprocal)(input_tensor, device=device)
    result = ttnn.to_torch(ttnn.reciprocal(to_tt_tensor(input_tensor, device)))

    assert (golden != 0).all(), "expected every reciprocal in this range to be representable"
    expected = torch.zeros_like(result)
    if device.arch() == ttnn.device.Arch.WORMHOLE_B0:
        boundary = input_tensor.abs() == RECIPROCAL_BOUNDARY_INPUT
        expected[boundary] = golden[boundary]
    assert_equal(expected, result)
    assert not torch.signbit(result[expected == 0]).any(), "flushed results must be +0 for both input signs"


def test_reciprocal_zero_and_nonfinite(device):
    """Signed zero divides the way torch does; two of the non-finite inputs do
    not. The device and torch results are as follows:

        input    | torch (IEEE 754) | device (actual)
        ---------+------------------+----------------
        +0       | +inf             | +inf            ✓
        -0       | -inf             | -inf            ✓
        +inf     | +0               | +0              ✓
        -inf     | -0               | +0              ✗ sign lost
        NaN      | NaN              | +0              ✗ not propagated

    The last two are pinned as the device's current behavior rather than
    endorsed as correct.
    """
    special = [0.0, -0.0, float("inf"), float("-inf"), float("nan")]
    input_tensor = torch.ones(32, 32, dtype=torch.bfloat16)
    input_tensor.view(-1)[: len(special)] = torch.tensor(special, dtype=torch.bfloat16)

    result = ttnn.to_torch(ttnn.reciprocal(to_tt_tensor(input_tensor, device))).view(-1)
    pos_zero, neg_zero, pos_inf, neg_inf, nan = (result[i] for i in range(len(special)))

    assert pos_zero == float("inf") and not torch.signbit(pos_zero), "1/+0 must be +inf"
    assert neg_zero == float("-inf") and torch.signbit(neg_zero), "1/-0 must be -inf"

    assert pos_inf == 0.0 and not torch.signbit(pos_inf), "1/+inf is +0, matching torch"
    assert neg_inf == 0.0 and not torch.signbit(neg_inf), "1/-inf is +0 on device, -0 in torch"
    assert nan == 0.0, "the device returns 0 for NaN instead of propagating it"


# ─────────────────────────────────────────────────────────────────────────────
# square: x^2, overflow when |x| > ~1.84e19; use (-1e19, 1e19)
# ─────────────────────────────────────────────────────────────────────────────


def test_square(device):
    input_tensor = generate_bfloat16_bits_in_range(-1e19, 1e19)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.square)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.square(tt_in)
    result = ttnn.to_torch(tt_result)

    threshold = 2 * SMALLEST_NORMAL_BF16
    result = torch.where(torch.abs(result) <= threshold, torch.zeros_like(result), result)
    golden = torch.where(torch.abs(golden) <= threshold, torch.zeros_like(golden), golden)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


# ─────────────────────────────────────────────────────────────────────────────
# cbrt: cube root, valid for all finite inputs.
# ─────────────────────────────────────────────────────────────────────────────
"""
Golden must be evaluated in float64 because bfloat16's non-representable 1/3
exponent rounds the reference incorrectly; the non-representable 1/3 was rounding
the reference up to 2 ULP short of the true cube root while the kernel was correct.
Subnormal bf16 inputs are flushed to zero on device
"""


def test_cbrt(device):
    input_tensor = generate_bfloat16_bits_in_range(-1e38, 1e38)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.cbrt)
    golden = golden_function(input_tensor.to(torch.float64)).to(torch.bfloat16)

    tt_result = ttnn.cbrt(tt_in)
    result = ttnn.to_torch(tt_result).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


# ─────────────────────────────────────────────────────────────────────────────
# Exponential functions (exp, exp2, expm1)
# exp:   overflow at ~88.5 (-> inf), underflow at ~-87 (-> 0)
# exp2:  overflow at 128 (-> inf), underflow at -126 (-> 0)
# expm1: same overflow as exp; underflow produces -1 for large negative inputs
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, low, high",
    [
        (ttnn.exp, -87.0, 88.5),
        (ttnn.exp2, -126.0, 127.0),
        (ttnn.expm1, -87.0, 88.5),
    ],
)
def test_exp_ops(device, ttnn_op, low, high):
    input_tensor = generate_bfloat16_bits_in_range(low, high)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


def test_exp_allclose(device):
    """exp underflow region (-89, -87) allclose check.

    The ULP sweep in test_exp_ops covers (-87.0, 88.5); this extends into the
    underflow tail where exp(x) approaches the smallest normals. 1020 of 1024
    bf16 values here are bit-exact; the remaining 4 are flushed to zero by the
    device (golden ~1e-38, device 0). atol is set just above the largest
    flushed golden value.
    """
    input_tensor = generate_bfloat16_bits_in_range(-89, -87)

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(input_tensor, device=device)

    tt_in = to_tt_tensor(input_tensor, device)

    tt_result = ttnn.exp(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_allclose(expected_result=golden, actual_result=result, atol=1.1e-38, rtol=0)


def test_exp2_allclose(device):
    """exp2 underflow region (-127, -126) allclose check.

    The ULP sweep in test_exp_ops covers (-126.0, 127.0); this extends into the
    underflow tail where exp2(x) approaches the smallest normals. 1022 of 1024
    bf16 values here are bit-exact; the remaining 2 are flushed to zero by the
    device (golden ~8.4e-39, device 0). atol is set just above the largest
    flushed golden value.
    """
    input_tensor = generate_bfloat16_bits_in_range(-127, -126)

    golden_function = ttnn.get_golden_function(ttnn.exp2)
    golden = golden_function(input_tensor, device=device)

    tt_in = to_tt_tensor(input_tensor, device)

    tt_result = ttnn.exp2(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_allclose(actual_result=result, expected_result=golden, atol=8.5e-39, rtol=0)


def test_exp2_special_values(device):
    """exp2 saturation and infinity handling, which the bfloat16 sweeps replace with 0 (see #44507).

    | Input                    | Expected                                   |
    |--------------------------|--------------------------------------------|
    | -inf                     | 0                                          |
    | -127                     | 0 (golden 2^-127 is subnormal; device FTZ) |
    | -126                     | 2^-126, smallest normal (ULP ≤ 1)          |
    | 0, ±1, ±10, 127          | ULP ≤ 1                                    |
    | 128, +inf                | +inf                                       |

    NaN is excluded: ttnn.bfloat16 host-side packing turns NaN into +inf before
    the tensor reaches the device, so NaN propagation can't be tested at this dtype.
    """
    input_tensor = torch.tensor(
        [0.0, 1.0, -1.0, 10.0, -10.0, 127.0, -126.0, float("inf"), -float("inf"), 128.0, -127.0],
        dtype=torch.bfloat16,
    )

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.exp2)
    golden = flush_subnormal_values_to_zero(golden_function(input_tensor, device=device))

    result = ttnn.to_torch(ttnn.exp2(tt_in))

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)


@pytest.mark.parametrize(
    "low, high, expected_atol, expected_rtol",
    [
        (-1.6 * 10**38, -0.28515625, 0, 0),
        (-0.28515625, 0.69140625, 0, 0),
        (0.69140625, 88.5, 0, 0),
    ],
)
def test_expm1_allclose(low, high, expected_atol, expected_rtol, device):
    """expm1 sub-range allclose check.

    The ULP sweep in test_exp_ops covers [-87.0, 88.5]; this test extends the
    negative tail to -1.6e38 and verifies allclose over three subdomains. All
    three are bit-exact on device (17408 + 32768 + 1024 = 51200 values, zero
    mismatches), so atol = rtol = 0.
    """
    input_tensor = generate_bfloat16_bits_in_range(low, high)

    golden_function = ttnn.get_golden_function(ttnn.expm1)
    golden = golden_function(input_tensor, device=device)

    tt_in = to_tt_tensor(input_tensor, device)

    tt_result = ttnn.expm1(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_allclose(actual_result=result, expected_result=golden, atol=expected_atol, rtol=expected_rtol)


# ─────────────────────────────────────────────────────────────────────────────
# digamma and multigammaln
# digamma: defined for x > 0, LUT kernel fitted on [0.01, 102], asymptotic for x > 102
# multigammaln: requires x > 1.5 (uses lgamma(x) and lgamma(x-0.5))
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, low, high, ulp",
    [
        (ttnn.digamma, 1.0, 102.0, 2),
        (ttnn.multigammaln, 1.6, 100.0, 3),
    ],
)
def test_digamma_multigammaln(device, ttnn_op, low, high, ulp):
    input_tensor = generate_bfloat16_bits_in_range(low, high)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp)


def test_digamma_large_x(device):
    """Regression guard for digamma at large x (issue #45520: "behaves bad for x>1000").

    The LUT kernel is fit on [0.01, 102]; beyond it a Bernoulli asymptotic branch
    (ln(x) - 1/2x - 1/12x^2 + ...) restores the (1, inf) support the pre-LUT composite
    op had. ``test_digamma`` only exercises [2, 102], so this covers the LUT->asymptotic
    crossover (102) and several decades past x=1000.
    """
    xs = torch.tensor(
        [[101.0, 102.0, 103.0, 150.0, 500.0, 1000.0, 5000.0, 1e4, 5e4, 1e5, 5e5, 1e6, 1e7, float("inf")]],
        dtype=torch.bfloat16,
    )
    golden = torch.digamma(xs.to(torch.float64)).to(torch.float32)
    input_tensor = ttnn.from_torch(xs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.digamma(input_tensor))
    assert_with_ulp(expected_result=golden, actual_result=output_tensor, ulp_threshold=2, allow_nonfinite=True)


def test_digamma_small_x(device):
    """Guard the steep near-pole region [0.01, 2): psi has a pole at 0 (psi(x) ~ -1/x),
    the steepest part of the fitted domain. test_digamma only exercises [2, 102].
    Sample avoids the zero-crossing at x~=1.4616 where ULP is ill-defined.
    """
    xs = torch.tensor(
        [[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.25, 1.75, 1.9, 1.99]],
        dtype=torch.bfloat16,
    )
    golden = torch.digamma(xs.to(torch.float64)).to(torch.float32)
    input_tensor = ttnn.from_torch(xs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.digamma(input_tensor))
    assert_with_ulp(expected_result=golden, actual_result=output_tensor, ulp_threshold=2)


# ─────────────────────────────────────────────────────────────────────────────
# lgamma: defined for all reals except poles at 0, -1, -2, ...
# ─────────────────────────────────────────────────────────────────────────────


def test_lgamma(device):
    input_tensor = generate_bfloat16_bits_in_range(-1000, 1000).flatten()
    input_tensor_f32 = input_tensor.to(torch.float32)
    # masking poles at 0, -1, -2, ...
    is_non_positive_int = (input_tensor_f32 <= 0) & (input_tensor_f32 == torch.floor(input_tensor_f32))
    input_tensor = input_tensor[~is_non_positive_int]

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.lgamma)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.lgamma(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_pcc(golden, result, 0.999)


# ─────────────────────────────────────────────────────────────────────────────
# Modified Bessel functions (i0, i1)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, low, high",
    [
        (ttnn.i0, -10.0, 10.0),
        (ttnn.i1, -10.0, 10.0),
    ],
)
def test_bessel_ops(device, ttnn_op, low, high):
    input_tensor = generate_bfloat16_bits_in_range(low, high)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    # device returns the smallest normal (2^-126) instead of exact zero.
    result = flush_to_zero(result)
    golden = flush_to_zero(golden)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


@pytest.mark.parametrize("generated_first", [True, False])
def test_erf_with_sqrt_in_sfpu_chain(device, generated_first):
    """erf in one SFPU chain with stock sqrt, both orders.

    Each chain step runs the kernel its own TT-NN op runs, and the second op stores exactly what it
    stores standalone on the first op's device result, which the chain keeps in DEST.
    """
    generated = (ttnn.UnaryWithParam(ttnn.UnaryOpType.ERF, 0.0), lambda tensor: ttnn.erf(tensor))
    stock = (ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT), lambda tensor: ttnn.sqrt(tensor))
    first, second = (generated, stock) if generated_first else (stock, generated)
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)
    tt_in = to_tt_tensor(input_tensor, device)

    def words(tensor):
        return ttnn.to_torch(tensor).view(torch.int16)

    for step, standalone in (generated, stock):
        assert torch.equal(
            words(ttnn.unary_chain(tt_in, [step])), words(standalone(tt_in))
        ), f"{step} is not its op's kernel"

    result = words(ttnn.unary_chain(tt_in, [first[0], second[0]]))
    middle = ttnn.to_torch(first[1](tt_in))
    expected = words(second[1](to_tt_tensor(middle, device)))
    # A finite normal first result sits in DEST exactly as stored, so both runs feed the second op the same input.
    checked = torch.isfinite(middle) & (middle.abs() >= SMALLEST_NORMAL_BF16)
    assert checked.sum() >= 64
    mismatched = checked & (result != expected)
    assert not mismatched.any(), (
        f"{int(mismatched.sum())} of {int(checked.sum())} lanes differ from the standalone second op, "
        f"first at its input {middle[mismatched][0].item()!r}"
    )


def test_erf_bf16_unflushed_zero_and_subnormal_inputs(device):
    """TT-NN's copy into DEST keeps -0 and subnormal BF16 inputs, which the LLK sweep's unpack delivers as +0.

    Each such input and +0 must store torch's result: its class as the BF16 pack stores it (NaN as
    +inf, a subnormal result as 0) and a finite result within 1 ULP of torch's float64 value.
    The pack writes a zero result as +0, so ±0 and both signs of subnormal store the word 0x0000.
    STOCK lists, per board, the inputs where the kernel stores its stock kernel's word instead,
    as {word: [(first input, last input), ...]} over the input bits.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.erf keeps its stock kernel on {board}")
    STOCK = {
        "blackhole": {0x0000: [(0x0071, 0x007F), (0x8071, 0x807F)]},
        "wormhole_b0": {0x0000: [(0x0071, 0x007F), (0x8071, 0x807F)]},
    }
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        t = torch.where(torch.isnan(t), torch.full_like(t, float("inf")), t)
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    device_out = ttnn.to_torch(ttnn.erf(tt_x))
    raw = device_out.reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    bits = words.repeat(4).to(torch.int32) & 0xFFFF
    assert torch.all(raw[bits == 0x0000] == 0x0000)
    assert torch.all(raw[bits == 0x8000] == 0x0000)
    assert torch.all(raw[(bits > 0) & (bits < 0x80)] == 0x0000)
    assert torch.all(raw[(bits > 0x8000) & (bits < 0x8080)] == 0x0000)

    actual = stored(device_out)
    x64 = x.flatten().to(torch.float64)
    reference = ttnn.get_golden_function(ttnn.erf)(x64).to(torch.float64).flatten()
    rounded = stored(reference)
    spacing = 2.0 ** (torch.floor(torch.log2(reference.abs().clamp(min=tiny))) - 7)
    close = torch.isfinite(reference) & (actual != 0) & ((actual - reference).abs() < spacing)
    ok = (actual == rounded) | close
    for word, ranges in STOCK.get(board, {}).items():
        value = stored(torch.tensor([word], dtype=torch.int32).to(torch.int16).view(torch.bfloat16))
        for first, last in ranges:
            ok |= (bits >= first) & (bits <= last) & (actual == value)
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )


def test_erf_bf16_bfloat8_b(device):
    """bfloat8_b in and out. The SFPU reads Bfp8_b data in DEST as BF16, so this runs the BF16 kernel, as the
    BF16 cases do. The result must correlate with torch's on the input as stored to PCC 0.999, as upstream's
    bfloat8_b unary cases require. Over every finite BF16 exponent, each output must also be within one bfloat8_b
    step of the BF16 call's for the same stored input, packed to bfloat8_b (two in a block that holds an infinity
    or NaN), and of its class (infinity, NaN, zero): the op packs DEST to Bfp8_b, and the typecast rounds and
    sets a block's exponent its own way.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.erf keeps its stock kernel on {board}")
    torch.manual_seed(0)
    x = torch.empty(1, 1, 64, 64).uniform_(-10.0, 10.0).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.float64).flatten()
    output = ttnn.erf(tt_x)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float64).flatten()
    golden = ttnn.get_golden_function(ttnn.erf)(stored).to(torch.float64)
    pcc = torch.corrcoef(torch.stack([actual, golden]))[0, 1].item()
    assert pcc >= 0.999, f"PCC {pcc:.6f} against torch"
    # Every finite BF16 exponent and +-0, one exponent to each 16-value face row, so Bfp8_b keeps each value
    # to 7 bits under its own exponent.
    words = torch.arange(1 << 16, dtype=torch.int32)
    exponent = (words >> 7) & 0xFF
    zeros = torch.tensor([0x0000] * 256 + [0x8000] * 256, dtype=torch.int32)
    words = torch.cat([words[(exponent > 0) & (exponent < 0xFF)], zeros])
    x = words.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 256, 256)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.bfloat16)
    tt_stored = ttnn.from_torch(stored, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.erf(tt_x)).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.erf(tt_stored), ttnn.bfloat8_b)).to(torch.float32).flatten()
    # One step of a Bfp8_b block: its 16 values share the largest one's exponent and keep 7 bits under it.
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    magnitude = torch.where(finite, torch.maximum(actual.abs(), expected.abs()), torch.zeros_like(actual))
    block = magnitude.reshape(-1, 16).amax(dim=1, keepdim=True).expand(-1, 16).reshape(-1)
    step = torch.exp2(torch.floor(torch.log2(block.clamp_min(torch.finfo(torch.float32).tiny))) - 6)
    # A block that also holds an infinity or NaN: the op's pack and the typecast set its exponent apart, and
    # its finite values land up to two steps apart (polygamma's overflow binade, BH job 142644).
    special = (~finite).reshape(-1, 16).any(dim=1, keepdim=True).expand(-1, 16).reshape(-1)
    step = torch.where(special, 2 * step, step)
    # Infinity, NaN and zero keep their class exactly; a finite value may round one step apart.
    same = (
        (actual == expected)
        | (torch.isnan(actual) & torch.isnan(expected))
        | (finite & ((actual == 0) == (expected == 0)) & ((actual - expected).abs() <= step))
    )
    assert same.all(), (
        f"{int((~same).sum())} bfloat8_b outputs are more than one bfloat8_b step from the BF16 call's on the same "
        "stored input; first at "
        f"{stored.flatten()[~same][0].item()!r}: {actual[~same][0].item()!r}, BF16 call {expected[~same][0].item()!r}"
    )


def test_erf_bf16_nonfinite_inputs(device):
    """+-Inf and +-NaN through TT-NN, where the kernel may return its stock kernel's word for some of them.

    Each must store torch's result as the BF16 pack stores it (NaN as an infinity), or the word STOCK
    lists for that input on the board, as {input bits: word}.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.erf keeps its stock kernel on {board}")
    STOCK = {"wormhole_b0": {0x7FC0: 0x3F80, 0xFFC0: 0xBF80}}
    bits = torch.tensor([0x7F80, 0xFF80, 0x7FC0, 0xFFC0], dtype=torch.int32).repeat(256)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def stored(t):
        return t.to(torch.bfloat16).to(torch.float64).flatten()

    actual = stored(ttnn.to_torch(ttnn.erf(tt_x)))
    reference = ttnn.get_golden_function(ttnn.erf)(x.flatten().to(torch.float64)).to(torch.float64)
    expected = stored(reference)
    for input_bits, word in STOCK.get(board, {}).items():
        expected[bits == input_bits] = stored(
            torch.tensor([word], dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        )
    spacing = 2.0 ** (torch.floor(torch.log2(expected.abs().clamp(min=torch.finfo(torch.bfloat16).tiny))) - 7)
    ok = (actual == expected) | (torch.isfinite(expected) & ((actual - expected).abs() < spacing))
    # A NaN result reaches L1 as a NaN or as an infinity of either sign.
    ok |= torch.isnan(expected) & (torch.isnan(actual) | torch.isinf(actual))
    assert (
        ok.all()
    ), f"input 0x{bits[~ok][0].item():04x} stored {actual[~ok][0].item()!r}, expected {expected[~ok][0].item()!r}"


def test_erf_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs through ttnn.erf, the call TT-NN users make.

    Each must store torch's float64 result (torch.special.erf(x)) as the BF16 pack stores it (NaN as an
    infinity, a result below the smallest normal, or within 2^-12 above it, as 0), a finite word within
    STEPS BF16 steps of it, or the word STOCK lists for that input on the board, as
    {word: [(first input, last input), ...]} over the input bits. SKIP lists the inputs this case does
    not judge, which the LLK sweep checks against its own golden: where torch's float64 formula misses
    the exact result (a cancellation), where the kernel departs from torch by its spec, and NaN inputs
    whose stock word varies with the payload. MODEL holds every other normal input whose torch result is finite to the
    kernel's own word: torch's, moved by MODEL's whole BF16 steps where MODEL lists the input, as
    {steps: "first-last first ..."} (hex input words) for each board.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.erf keeps its stock kernel on {board}")
    STOCK = {
        "blackhole": {0x0000: [(0x0071, 0x007F), (0x8071, 0x807F)]},
        "wormhole_b0": {
            0x0000: [(0x0071, 0x007F), (0x8071, 0x807F)],
            0x3F80: [(0x7F81, 0x7FFF)],
            0xBF80: [(0xFF81, 0xFFFF)],
        },
    }
    STEPS = 1
    SKIP = [(0x0002, 0x007F), (0x8002, 0x807F)]
    MODEL = {
        "blackhole": {
            -1: (
                "0081 0089 0091 0098-0099 00A0-00A1 00A8 00B0 00B8 00BF-00C0 00C7-00C8 "
                "00CF-00D0 00D7-00D8 00DE-00DF 00EB 00F2 00FB 0101 0109 0111 0118-0119 "
                "0120-0121 0128 0130 0138 013F-0140 0147-0148 014F-0150 0157-0158 015E-015F 016B "
                "0172 017B 0181 0189 0191 0198-0199 01A0-01A1 01A8 01B0 01B8 "
                "01BF-01C0 01C7-01C8 01CF-01D0 01D7-01D8 01DE-01DF 01EB 01F2 01FB 0201 0209 "
                "0211 0218-0219 0220-0221 0228 0230 0238 023F-0240 0247-0248 024F-0250 0257-0258 "
                "025E-025F 026B 0272 027B 0281 0289 0291 0298-0299 02A0-02A1 02A8 "
                "02B0 02B8 02BF-02C0 02C7-02C8 02CF-02D0 02D7-02D8 02DE-02DF 02EB 02F2 02FB "
                "0301 0309 0311 0318-0319 0320-0321 0328 0330 0338 033F-0340 0347-0348 "
                "034F-0350 0357-0358 035E-035F 036B 0372 037B 0381 0389 0391 0398-0399 "
                "03A0-03A1 03A8 03B0 03B8 03BF-03C0 03C7-03C8 03CF-03D0 03D7-03D8 03DE-03DF 03EB "
                "03F2 03FB 0401 0409 0411 0418-0419 0420-0421 0428 0430 0438 "
                "043F-0440 0447-0448 044F-0450 0457-0458 045E-045F 046B 0472 047B 0481 0489 "
                "0491 0498-0499 04A0-04A1 04A8 04B0 04B8 04BF-04C0 04C7-04C8 04CF-04D0 04D7-04D8 "
                "04DE-04DF 04EB 04F2 04FB 0501 0509 0511 0518-0519 0520-0521 0528 "
                "0530 0538 053F-0540 0547-0548 054F-0550 0557-0558 055E-055F 056B 0572 057B "
                "0581 0589 0591 0598-0599 05A0-05A1 05A8 05B0 05B8 05BF-05C0 05C7-05C8 "
                "05CF-05D0 05D7-05D8 05DE-05DF 05EB 05F2 05FB 0601 0609 0611 0618-0619 "
                "0620-0621 0628 0630 0638 063F-0640 0647-0648 064F-0650 0657-0658 065E-065F 066B "
                "0672 067B 0681 0689 0691 0698-0699 06A0-06A1 06A8 06B0 06B8 "
                "06BF-06C0 06C7-06C8 06CF-06D0 06D7-06D8 06DE-06DF 06EB 06F2 06FB 0701 0709 "
                "0711 0718-0719 0720-0721 0728 0730 0738 073F-0740 0747-0748 074F-0750 0757-0758 "
                "075E-075F 076B 0772 077B 0781 0789 0791 0798-0799 07A0-07A1 07A8 "
                "07B0 07B8 07BF-07C0 07C7-07C8 07CF-07D0 07D7-07D8 07DE-07DF 07EB 07F2 07FB "
                "0801 0809 0811 0818-0819 0820-0821 0828 0830 0838 083F-0840 0847-0848 "
                "084F-0850 0857-0858 085E-085F 086B 0872 087B 0881 0889 0891 0898-0899 "
                "08A0-08A1 08A8 08B0 08B8 08BF-08C0 08C7-08C8 08CF-08D0 08D7-08D8 08DE-08DF 08EB "
                "08F2 08FB 0901 0909 0911 0918-0919 0920-0921 0928 0930 0938 "
                "093F-0940 0947-0948 094F-0950 0957-0958 095E-095F 096B 0972 097B 0981 0989 "
                "0991 0998-0999 09A0-09A1 09A8 09B0 09B8 09BF-09C0 09C7-09C8 09CF-09D0 09D7-09D8 "
                "09DE-09DF 09EB 09F2 09FB 0A01 0A09 0A11 0A18-0A19 0A20-0A21 0A28 "
                "0A30 0A38 0A3F-0A40 0A47-0A48 0A4F-0A50 0A57-0A58 0A5E-0A5F 0A6B 0A72 0A7B "
                "0A81 0A89 0A91 0A98-0A99 0AA0-0AA1 0AA8 0AB0 0AB8 0ABF-0AC0 0AC7-0AC8 "
                "0ACF-0AD0 0AD7-0AD8 0ADE-0ADF 0AEB 0AF2 0AFB 0B01 0B09 0B11 0B18-0B19 "
                "0B20-0B21 0B28 0B30 0B38 0B3F-0B40 0B47-0B48 0B4F-0B50 0B57-0B58 0B5E-0B5F 0B6B "
                "0B72 0B7B 0B81 0B89 0B91 0B98-0B99 0BA0-0BA1 0BA8 0BB0 0BB8 "
                "0BBF-0BC0 0BC7-0BC8 0BCF-0BD0 0BD7-0BD8 0BDE-0BDF 0BEB 0BF2 0BFB 0C01 0C09 "
                "0C11 0C18-0C19 0C20-0C21 0C28 0C30 0C38 0C3F-0C40 0C47-0C48 0C4F-0C50 0C57-0C58 "
                "0C5E-0C5F 0C6B 0C72 0C7B 0C81 0C89 0C91 0C98-0C99 0CA0-0CA1 0CA8 "
                "0CB0 0CB8 0CBF-0CC0 0CC7-0CC8 0CCF-0CD0 0CD7-0CD8 0CDE-0CDF 0CEB 0CF2 0CFB "
                "0D01 0D09 0D11 0D18-0D19 0D20-0D21 0D28 0D30 0D38 0D3F-0D40 0D47-0D48 "
                "0D4F-0D50 0D57-0D58 0D5E-0D5F 0D6B 0D72 0D7B 0D81 0D89 0D91 0D98-0D99 "
                "0DA0-0DA1 0DA8 0DB0 0DB8 0DBF-0DC0 0DC7-0DC8 0DCF-0DD0 0DD7-0DD8 0DDE-0DDF 0DEB "
                "0DF2 0DFB 0E01 0E09 0E11 0E18-0E19 0E20-0E21 0E28 0E30 0E38 "
                "0E3F-0E40 0E47-0E48 0E4F-0E50 0E57-0E58 0E5E-0E5F 0E6B 0E72 0E7B 0E81 0E89 "
                "0E91 0E98-0E99 0EA0-0EA1 0EA8 0EB0 0EB8 0EBF-0EC0 0EC7-0EC8 0ECF-0ED0 0ED7-0ED8 "
                "0EDE-0EDF 0EEB 0EF2 0EFB 0F01 0F09 0F11 0F18-0F19 0F20-0F21 0F28 "
                "0F30 0F38 0F3F-0F40 0F47-0F48 0F4F-0F50 0F57-0F58 0F5E-0F5F 0F6B 0F72 0F7B "
                "0F81 0F89 0F91 0F98-0F99 0FA0-0FA1 0FA8 0FB0 0FB8 0FBF-0FC0 0FC7-0FC8 "
                "0FCF-0FD0 0FD7-0FD8 0FDE-0FDF 0FEB 0FF2 0FFB 1001 1009 1011 1018-1019 "
                "1020-1021 1028 1030 1038 103F-1040 1047-1048 104F-1050 1057-1058 105E-105F 106B "
                "1072 107B 1081 1089 1091 1098-1099 10A0-10A1 10A8 10B0 10B8 "
                "10BF-10C0 10C7-10C8 10CF-10D0 10D7-10D8 10DE-10DF 10EB 10F2 10FB 1101 1109 "
                "1111 1118-1119 1120-1121 1128 1130 1138 113F-1140 1147-1148 114F-1150 1157-1158 "
                "115E-115F 116B 1172 117B 1181 1189 1191 1198-1199 11A0-11A1 11A8 "
                "11B0 11B8 11BF-11C0 11C7-11C8 11CF-11D0 11D7-11D8 11DE-11DF 11EB 11F2 11FB "
                "1201 1209 1211 1218-1219 1220-1221 1228 1230 1238 123F-1240 1247-1248 "
                "124F-1250 1257-1258 125E-125F 126B 1272 127B 1281 1289 1291 1298-1299 "
                "12A0-12A1 12A8 12B0 12B8 12BF-12C0 12C7-12C8 12CF-12D0 12D7-12D8 12DE-12DF 12EB "
                "12F2 12FB 1301 1309 1311 1318-1319 1320-1321 1328 1330 1338 "
                "133F-1340 1347-1348 134F-1350 1357-1358 135E-135F 136B 1372 137B 1381 1389 "
                "1391 1398-1399 13A0-13A1 13A8 13B0 13B8 13BF-13C0 13C7-13C8 13CF-13D0 13D7-13D8 "
                "13DE-13DF 13EB 13F2 13FB 1401 1409 1411 1418-1419 1420-1421 1428 "
                "1430 1438 143F-1440 1447-1448 144F-1450 1457-1458 145E-145F 146B 1472 147B "
                "1481 1489 1491 1498-1499 14A0-14A1 14A8 14B0 14B8 14BF-14C0 14C7-14C8 "
                "14CF-14D0 14D7-14D8 14DE-14DF 14EB 14F2 14FB 1501 1509 1511 1518-1519 "
                "1520-1521 1528 1530 1538 153F-1540 1547-1548 154F-1550 1557-1558 155E-155F 156B "
                "1572 157B 1581 1589 1591 1598-1599 15A0-15A1 15A8 15B0 15B8 "
                "15BF-15C0 15C7-15C8 15CF-15D0 15D7-15D8 15DE-15DF 15EB 15F2 15FB 1601 1609 "
                "1611 1618-1619 1620-1621 1628 1630 1638 163F-1640 1647-1648 164F-1650 1657-1658 "
                "165E-165F 166B 1672 167B 1681 1689 1691 1698-1699 16A0-16A1 16A8 "
                "16B0 16B8 16BF-16C0 16C7-16C8 16CF-16D0 16D7-16D8 16DE-16DF 16EB 16F2 16FB "
                "1701 1709 1711 1718-1719 1720-1721 1728 1730 1738 173F-1740 1747-1748 "
                "174F-1750 1757-1758 175E-175F 176B 1772 177B 1781 1789 1791 1798-1799 "
                "17A0-17A1 17A8 17B0 17B8 17BF-17C0 17C7-17C8 17CF-17D0 17D7-17D8 17DE-17DF 17EB "
                "17F2 17FB 1801 1809 1811 1818-1819 1820-1821 1828 1830 1838 "
                "183F-1840 1847-1848 184F-1850 1857-1858 185E-185F 186B 1872 187B 1881 1889 "
                "1891 1898-1899 18A0-18A1 18A8 18B0 18B8 18BF-18C0 18C7-18C8 18CF-18D0 18D7-18D8 "
                "18DE-18DF 18EB 18F2 18FB 1901 1909 1911 1918-1919 1920-1921 1928 "
                "1930 1938 193F-1940 1947-1948 194F-1950 1957-1958 195E-195F 196B 1972 197B "
                "1981 1989 1991 1998-1999 19A0-19A1 19A8 19B0 19B8 19BF-19C0 19C7-19C8 "
                "19CF-19D0 19D7-19D8 19DE-19DF 19EB 19F2 19FB 1A01 1A09 1A11 1A18-1A19 "
                "1A20-1A21 1A28 1A30 1A38 1A3F-1A40 1A47-1A48 1A4F-1A50 1A57-1A58 1A5E-1A5F 1A6B "
                "1A72 1A7B 1A81 1A89 1A91 1A98-1A99 1AA0-1AA1 1AA8 1AB0 1AB8 "
                "1ABF-1AC0 1AC7-1AC8 1ACF-1AD0 1AD7-1AD8 1ADE-1ADF 1AEB 1AF2 1AFB 1B01 1B09 "
                "1B11 1B18-1B19 1B20-1B21 1B28 1B30 1B38 1B3F-1B40 1B47-1B48 1B4F-1B50 1B57-1B58 "
                "1B5E-1B5F 1B6B 1B72 1B7B 1B81 1B89 1B91 1B98-1B99 1BA0-1BA1 1BA8 "
                "1BB0 1BB8 1BBF-1BC0 1BC7-1BC8 1BCF-1BD0 1BD7-1BD8 1BDE-1BDF 1BEB 1BF2 1BFB "
                "1C01 1C09 1C11 1C18-1C19 1C20-1C21 1C28 1C30 1C38 1C3F-1C40 1C47-1C48 "
                "1C4F-1C50 1C57-1C58 1C5E-1C5F 1C6B 1C72 1C7B 1C81 1C89 1C91 1C98-1C99 "
                "1CA0-1CA1 1CA8 1CB0 1CB8 1CBF-1CC0 1CC7-1CC8 1CCF-1CD0 1CD7-1CD8 1CDE-1CDF 1CEB "
                "1CF2 1CFB 1D01 1D09 1D11 1D18-1D19 1D20-1D21 1D28 1D30 1D38 "
                "1D3F-1D40 1D47-1D48 1D4F-1D50 1D57-1D58 1D5E-1D5F 1D6B 1D72 1D7B 1D81 1D89 "
                "1D91 1D98-1D99 1DA0-1DA1 1DA8 1DB0 1DB8 1DBF-1DC0 1DC7-1DC8 1DCF-1DD0 1DD7-1DD8 "
                "1DDE-1DDF 1DEB 1DF2 1DFB 1E01 1E09 1E11 1E18-1E19 1E20-1E21 1E28 "
                "1E30 1E38 1E3F-1E40 1E47-1E48 1E4F-1E50 1E57-1E58 1E5E-1E5F 1E6B 1E72 1E7B "
                "1E81 1E89 1E91 1E98-1E99 1EA0-1EA1 1EA8 1EB0 1EB8 1EBF-1EC0 1EC7-1EC8 "
                "1ECF-1ED0 1ED7-1ED8 1EDE-1EDF 1EEB 1EF2 1EFB 1F01 1F09 1F11 1F18-1F19 "
                "1F20-1F21 1F28 1F30 1F38 1F3F-1F40 1F47-1F48 1F4F-1F50 1F57-1F58 1F5E-1F5F 1F6B "
                "1F72 1F7B 1F81 1F89 1F91 1F98-1F99 1FA0-1FA1 1FA8 1FB0 1FB8 "
                "1FBF-1FC0 1FC7-1FC8 1FCF-1FD0 1FD7-1FD8 1FDE-1FDF 1FEB 1FF2 1FFB 2001 2009 "
                "2011 2018-2019 2020-2021 2028 2030 2038 203F-2040 2047-2048 204F-2050 2057-2058 "
                "205E-205F 206B 2072 207B 2081 2089 2091 2098-2099 20A0-20A1 20A8 "
                "20B0 20B8 20BF-20C0 20C7-20C8 20CF-20D0 20D7-20D8 20DE-20DF 20EB 20F2 20FB "
                "2101 2109 2111 2118-2119 2120-2121 2128 2130 2138 213F-2140 2147-2148 "
                "214F-2150 2157-2158 215E-215F 216B 2172 217B 2181 2189 2191 2198-2199 "
                "21A0-21A1 21A8 21B0 21B8 21BF-21C0 21C7-21C8 21CF-21D0 21D7-21D8 21DE-21DF 21EB "
                "21F2 21FB 2201 2209 2211 2218-2219 2220-2221 2228 2230 2238 "
                "223F-2240 2247-2248 224F-2250 2257-2258 225E-225F 226B 2272 227B 2281 2289 "
                "2291 2298-2299 22A0-22A1 22A8 22B0 22B8 22BF-22C0 22C7-22C8 22CF-22D0 22D7-22D8 "
                "22DE-22DF 22EB 22F2 22FB 2301 2309 2311 2318-2319 2320-2321 2328 "
                "2330 2338 233F-2340 2347-2348 234F-2350 2357-2358 235E-235F 236B 2372 237B "
                "2381 2389 2391 2398-2399 23A0-23A1 23A8 23B0 23B8 23BF-23C0 23C7-23C8 "
                "23CF-23D0 23D7-23D8 23DE-23DF 23EB 23F2 23FB 2401 2409 2411 2418-2419 "
                "2420-2421 2428 2430 2438 243F-2440 2447-2448 244F-2450 2457-2458 245E-245F 246B "
                "2472 247B 2481 2489 2491 2498-2499 24A0-24A1 24A8 24B0 24B8 "
                "24BF-24C0 24C7-24C8 24CF-24D0 24D7-24D8 24DE-24DF 24EB 24F2 24FB 2501 2509 "
                "2511 2518-2519 2520-2521 2528 2530 2538 253F-2540 2547-2548 254F-2550 2557-2558 "
                "255E-255F 256B 2572 257B 2581 2589 2591 2598-2599 25A0-25A1 25A8 "
                "25B0 25B8 25BF-25C0 25C7-25C8 25CF-25D0 25D7-25D8 25DE-25DF 25EB 25F2 25FB "
                "2601 2609 2611 2618-2619 2620-2621 2628 2630 2638 263F-2640 2647-2648 "
                "264F-2650 2657-2658 265E-265F 266B 2672 267B 2681 2689 2691 2698-2699 "
                "26A0-26A1 26A8 26B0 26B8 26BF-26C0 26C7-26C8 26CF-26D0 26D7-26D8 26DE-26DF 26EB "
                "26F2 26FB 2701 2709 2711 2718-2719 2720-2721 2728 2730 2738 "
                "273F-2740 2747-2748 274F-2750 2757-2758 275E-275F 276B 2772 277B 2781 2789 "
                "2791 2798-2799 27A0-27A1 27A8 27B0 27B8 27BF-27C0 27C7-27C8 27CF-27D0 27D7-27D8 "
                "27DE-27DF 27EB 27F2 27FB 2801 2809 2811 2818-2819 2820-2821 2828 "
                "2830 2838 283F-2840 2847-2848 284F-2850 2857-2858 285E-285F 286B 2872 287B "
                "2881 2889 2891 2898-2899 28A0-28A1 28A8 28B0 28B8 28BF-28C0 28C7-28C8 "
                "28CF-28D0 28D7-28D8 28DE-28DF 28EB 28F2 28FB 2901 2909 2911 2918-2919 "
                "2920-2921 2928 2930 2938 293F-2940 2947-2948 294F-2950 2957-2958 295E-295F 296B "
                "2972 297B 2981 2989 2991 2998-2999 29A0-29A1 29A8 29B0 29B8 "
                "29BF-29C0 29C7-29C8 29CF-29D0 29D7-29D8 29DE-29DF 29EB 29F2 29FB 2A01 2A09 "
                "2A11 2A18-2A19 2A20-2A21 2A28 2A30 2A38 2A3F-2A40 2A47-2A48 2A4F-2A50 2A57-2A58 "
                "2A5E-2A5F 2A6B 2A72 2A7B 2A81 2A89 2A91 2A98-2A99 2AA0-2AA1 2AA8 "
                "2AB0 2AB8 2ABF-2AC0 2AC7-2AC8 2ACF-2AD0 2AD7-2AD8 2ADE-2ADF 2AEB 2AF2 2AFB "
                "2B01 2B09 2B11 2B18-2B19 2B20-2B21 2B28 2B30 2B38 2B3F-2B40 2B47-2B48 "
                "2B4F-2B50 2B57-2B58 2B5E-2B5F 2B6B 2B72 2B7B 2B81 2B89 2B91 2B98-2B99 "
                "2BA0-2BA1 2BA8 2BB0 2BB8 2BBF-2BC0 2BC7-2BC8 2BCF-2BD0 2BD7-2BD8 2BDE-2BDF 2BEB "
                "2BF2 2BFB 2C01 2C09 2C11 2C18-2C19 2C20-2C21 2C28 2C30 2C38 "
                "2C3F-2C40 2C47-2C48 2C4F-2C50 2C57-2C58 2C5E-2C5F 2C6B 2C72 2C7B 2C81 2C89 "
                "2C91 2C98-2C99 2CA0-2CA1 2CA8 2CB0 2CB8 2CBF-2CC0 2CC7-2CC8 2CCF-2CD0 2CD7-2CD8 "
                "2CDE-2CDF 2CEB 2CF2 2CFB 2D01 2D09 2D11 2D18-2D19 2D20-2D21 2D28 "
                "2D30 2D38 2D3F-2D40 2D47-2D48 2D4F-2D50 2D57-2D58 2D5E-2D5F 2D6B 2D72 2D7B "
                "2D81 2D89 2D91 2D98-2D99 2DA0-2DA1 2DA8 2DB0 2DB8 2DBF-2DC0 2DC7-2DC8 "
                "2DCF-2DD0 2DD7-2DD8 2DDE-2DDF 2DEB 2DF2 2DFB 2E01 2E09 2E11 2E18-2E19 "
                "2E20-2E21 2E28 2E30 2E38 2E3F-2E40 2E47-2E48 2E4F-2E50 2E57-2E58 2E5E-2E5F 2E6B "
                "2E72 2E7B 2E81 2E89 2E91 2E98-2E99 2EA0-2EA1 2EA8 2EB0 2EB8 "
                "2EBF-2EC0 2EC7-2EC8 2ECF-2ED0 2ED7-2ED8 2EDE-2EDF 2EEB 2EF2 2EFB 2F01 2F09 "
                "2F11 2F18-2F19 2F20-2F21 2F28 2F30 2F38 2F3F-2F40 2F47-2F48 2F4F-2F50 2F57-2F58 "
                "2F5E-2F5F 2F6B 2F72 2F7B 2F81 2F89 2F91 2F98-2F99 2FA0-2FA1 2FA8 "
                "2FB0 2FB8 2FBF-2FC0 2FC7-2FC8 2FCF-2FD0 2FD7-2FD8 2FDE-2FDF 2FEB 2FF2 2FFB "
                "3001 3009 3011 3018-3019 3020-3021 3028 3030 3038 303F-3040 3047-3048 "
                "304F-3050 3057-3058 305E-305F 306B 3072 307B 3081 3089 3091 3098-3099 "
                "30A0-30A1 30A8 30B0 30B8 30BF-30C0 30C7-30C8 30CF-30D0 30D7-30D8 30DE-30DF 30EB "
                "30F2 30FB 3101 3109 3111 3118-3119 3120-3121 3128 3130 3138 "
                "313F-3140 3147-3148 314F-3150 3157-3158 315E-315F 316B 3172 317B 3181 3189 "
                "3191 3198-3199 31A0-31A1 31A8 31B0 31B8 31BF-31C0 31C7-31C8 31CF-31D0 31D7-31D8 "
                "31DE-31DF 31EB 31F2 31FB 3201 3209 3211 3218-3219 3220-3221 3228 "
                "3230 3238 323F-3240 3247-3248 324F-3250 3257-3258 325E-325F 326B 3272 327B "
                "3281 3289 3291 3298-3299 32A0-32A1 32A8 32B0 32B8 32BF-32C0 32C7-32C8 "
                "32CF-32D0 32D7-32D8 32DE-32DF 32EB 32F2 32FB 3301 3309 3311 3318-3319 "
                "3320-3321 3328 3330 3338 333F-3340 3347-3348 334F-3350 3357-3358 335E-335F 336B "
                "3372 337B 3381 3389 3391 3398-3399 33A0-33A1 33A8 33B0 33B8 "
                "33BF-33C0 33C7-33C8 33CF-33D0 33D7-33D8 33DE-33DF 33EB 33F2 33FB 3401 3409 "
                "3411 3418-3419 3420-3421 3428 3430 3438 343F-3440 3447-3448 344F-3450 3457-3458 "
                "345E-345F 346B 3472 347B 3481 3489 3491 3498-3499 34A0-34A1 34A8 "
                "34B0 34B8 34BF-34C0 34C7-34C8 34CF-34D0 34D7-34D8 34DE-34DF 34EB 34F2 34FB "
                "3501 3509 3511 3518-3519 3520-3521 3528 3530 3538 353F-3540 3547-3548 "
                "354F-3550 3557-3558 355E-355F 356B 3572 357B 3581 3589 3591 3598-3599 "
                "35A0-35A1 35A8 35B0 35B8 35BF-35C0 35C7-35C8 35CF-35D0 35D7-35D8 35DE-35DF 35EB "
                "35F2 35FB 3601 3609 3611 3618-3619 3620-3621 3628 3630 3638 "
                "363F-3640 3647-3648 364F-3650 3657-3658 365E-365F 366B 3672 367B 3681 3689 "
                "3691 3698-3699 36A0-36A1 36A8 36B0 36B8 36BF-36C0 36C7-36C8 36CF-36D0 36D7-36D8 "
                "36DE-36DF 36EB 36F2 36FB 3701 3709 3711 3718-3719 3720-3721 3728 "
                "3730 3738 373F-3740 3747-3748 374F-3750 3757-3758 375E-375F 376B 3772 377B "
                "3781 3789 3791 3798-3799 37A0-37A1 37A8 37B0 37B8 37BF-37C0 37C7-37C8 "
                "37CF-37D0 37D7-37D8 37DE-37DF 37EB 37F2 37FB 3801 3809 3811 3818-3819 "
                "3820-3821 3828 3830 3838 383F-3840 3847-3848 384F-3850 3857-3858 385E-385F 386B "
                "3872 387B 3881 3889 3891 3898-3899 38A0-38A1 38A8 38B0 38B8 "
                "38BF-38C0 38C7-38C8 38CF-38D0 38D7-38D8 38DE-38DF 38EB 38F2 38FB 3901 3909 "
                "3911 3918-3919 3920-3921 3928 3930 3938 393F-3940 3947-3948 394F-3950 3957 "
                "395E-395F 396B 3972 397B 3981 3989 3991 3998-3999 39A0-39A1 39A8 "
                "39B0 39B8 39BF-39C0 39C7-39C8 39CF-39D0 39D7 39DE-39DF 39EB 39F2 39FB "
                "3A01 3A09 3A11 3A18-3A19 3A20-3A21 3A28 3A30 3A38 3A3F-3A40 3A47-3A48 "
                "3A4F-3A50 3A57 3A5E-3A5F 3A6B 3A72 3A7B 3A81 3A89 3A91 3A98-3A99 "
                "3AA0 3AA8 3AB0 3AB8 3ABF-3AC0 3AC7-3AC8 3ACF-3AD0 3AD7 3ADF 3AEB "
                "3AF2 3AFB 3B01 3B09 3B11 3B18-3B19 3B20 3B28 3B30 3B38 "
                "3B3F-3B40 3B47-3B48 3B4F-3B50 3B57 3B5F 3B6B 3B72 3B7B 3B81 3B89 "
                "3B91 3B98-3B99 3BA0 3BA8 3BB0 3BB8 3BBF-3BC0 3BC7-3BC8 3BCF 3BD7 "
                "3BDF 3BEB 3BF2 3C01 3C09 3C11 3C18 3C20 3C28 3C30 "
                "3C38 3C3F-3C40 3C47 3C4F 3C57 3C5F 3C6B 3C72 3C81 3C89 "
                "3CA0 3CA8 3CB0 3CB8 3CC7 3CCF 3CD7 3CDF 3D01 3D09 "
                "3EE0 3EEC 3F0C 3F17 3F20 3F24 3F28 3F2F 3F44 3F4C "
                "3F53 3F5A 3FD7 3FE0 3FEA-3FEC 3FFA-3FFE BD48 BD74 BD82 BD8A "
                "BD92 BD9A BDA2 BDB3 BDBB BDC3-BDC4 BDCC BDD4-BDD5 BDDD BDEA "
                "BDF3 BDFC BE07 BE10 BE19-BE1A BE23-BE24 BE2D-BE2E BE37-BE38 BE42-BE44 BE4E-BE4F "
                "BE5A-BE5C BE6F BE7E BE80 BE93-BE94 BF7D BF86 BF8A BF8D BF90 "
                "BF95 BF9C BF9E BFA7 BFAC BFAF BFB2 BFB5 BFC5"
            ),
            1: (
                "3D48 3D74 3D82 3D8A 3D92 3D9A 3DA2 3DB3 3DBB 3DC3-3DC4 "
                "3DCC 3DD4-3DD5 3DDD 3DEA 3DF3 3DFC 3E07 3E10 3E19-3E1A 3E23-3E24 "
                "3E2D-3E2E 3E37-3E38 3E42-3E44 3E4E-3E4F 3E5A-3E5C 3E6F 3E7E 3E80 3E93-3E94 3F7D "
                "3F86 3F8A 3F8D 3F90 3F95 3F9C 3F9E 3FA7 3FAC 3FAF "
                "3FB2 3FB5 3FC5 8081 8089 8091 8098-8099 80A0-80A1 80A8 80B0 "
                "80B8 80BF-80C0 80C7-80C8 80CF-80D0 80D7-80D8 80DE-80DF 80EB 80F2 80FB 8101 "
                "8109 8111 8118-8119 8120-8121 8128 8130 8138 813F-8140 8147-8148 814F-8150 "
                "8157-8158 815E-815F 816B 8172 817B 8181 8189 8191 8198-8199 81A0-81A1 "
                "81A8 81B0 81B8 81BF-81C0 81C7-81C8 81CF-81D0 81D7-81D8 81DE-81DF 81EB 81F2 "
                "81FB 8201 8209 8211 8218-8219 8220-8221 8228 8230 8238 823F-8240 "
                "8247-8248 824F-8250 8257-8258 825E-825F 826B 8272 827B 8281 8289 8291 "
                "8298-8299 82A0-82A1 82A8 82B0 82B8 82BF-82C0 82C7-82C8 82CF-82D0 82D7-82D8 82DE-82DF "
                "82EB 82F2 82FB 8301 8309 8311 8318-8319 8320-8321 8328 8330 "
                "8338 833F-8340 8347-8348 834F-8350 8357-8358 835E-835F 836B 8372 837B 8381 "
                "8389 8391 8398-8399 83A0-83A1 83A8 83B0 83B8 83BF-83C0 83C7-83C8 83CF-83D0 "
                "83D7-83D8 83DE-83DF 83EB 83F2 83FB 8401 8409 8411 8418-8419 8420-8421 "
                "8428 8430 8438 843F-8440 8447-8448 844F-8450 8457-8458 845E-845F 846B 8472 "
                "847B 8481 8489 8491 8498-8499 84A0-84A1 84A8 84B0 84B8 84BF-84C0 "
                "84C7-84C8 84CF-84D0 84D7-84D8 84DE-84DF 84EB 84F2 84FB 8501 8509 8511 "
                "8518-8519 8520-8521 8528 8530 8538 853F-8540 8547-8548 854F-8550 8557-8558 855E-855F "
                "856B 8572 857B 8581 8589 8591 8598-8599 85A0-85A1 85A8 85B0 "
                "85B8 85BF-85C0 85C7-85C8 85CF-85D0 85D7-85D8 85DE-85DF 85EB 85F2 85FB 8601 "
                "8609 8611 8618-8619 8620-8621 8628 8630 8638 863F-8640 8647-8648 864F-8650 "
                "8657-8658 865E-865F 866B 8672 867B 8681 8689 8691 8698-8699 86A0-86A1 "
                "86A8 86B0 86B8 86BF-86C0 86C7-86C8 86CF-86D0 86D7-86D8 86DE-86DF 86EB 86F2 "
                "86FB 8701 8709 8711 8718-8719 8720-8721 8728 8730 8738 873F-8740 "
                "8747-8748 874F-8750 8757-8758 875E-875F 876B 8772 877B 8781 8789 8791 "
                "8798-8799 87A0-87A1 87A8 87B0 87B8 87BF-87C0 87C7-87C8 87CF-87D0 87D7-87D8 87DE-87DF "
                "87EB 87F2 87FB 8801 8809 8811 8818-8819 8820-8821 8828 8830 "
                "8838 883F-8840 8847-8848 884F-8850 8857-8858 885E-885F 886B 8872 887B 8881 "
                "8889 8891 8898-8899 88A0-88A1 88A8 88B0 88B8 88BF-88C0 88C7-88C8 88CF-88D0 "
                "88D7-88D8 88DE-88DF 88EB 88F2 88FB 8901 8909 8911 8918-8919 8920-8921 "
                "8928 8930 8938 893F-8940 8947-8948 894F-8950 8957-8958 895E-895F 896B 8972 "
                "897B 8981 8989 8991 8998-8999 89A0-89A1 89A8 89B0 89B8 89BF-89C0 "
                "89C7-89C8 89CF-89D0 89D7-89D8 89DE-89DF 89EB 89F2 89FB 8A01 8A09 8A11 "
                "8A18-8A19 8A20-8A21 8A28 8A30 8A38 8A3F-8A40 8A47-8A48 8A4F-8A50 8A57-8A58 8A5E-8A5F "
                "8A6B 8A72 8A7B 8A81 8A89 8A91 8A98-8A99 8AA0-8AA1 8AA8 8AB0 "
                "8AB8 8ABF-8AC0 8AC7-8AC8 8ACF-8AD0 8AD7-8AD8 8ADE-8ADF 8AEB 8AF2 8AFB 8B01 "
                "8B09 8B11 8B18-8B19 8B20-8B21 8B28 8B30 8B38 8B3F-8B40 8B47-8B48 8B4F-8B50 "
                "8B57-8B58 8B5E-8B5F 8B6B 8B72 8B7B 8B81 8B89 8B91 8B98-8B99 8BA0-8BA1 "
                "8BA8 8BB0 8BB8 8BBF-8BC0 8BC7-8BC8 8BCF-8BD0 8BD7-8BD8 8BDE-8BDF 8BEB 8BF2 "
                "8BFB 8C01 8C09 8C11 8C18-8C19 8C20-8C21 8C28 8C30 8C38 8C3F-8C40 "
                "8C47-8C48 8C4F-8C50 8C57-8C58 8C5E-8C5F 8C6B 8C72 8C7B 8C81 8C89 8C91 "
                "8C98-8C99 8CA0-8CA1 8CA8 8CB0 8CB8 8CBF-8CC0 8CC7-8CC8 8CCF-8CD0 8CD7-8CD8 8CDE-8CDF "
                "8CEB 8CF2 8CFB 8D01 8D09 8D11 8D18-8D19 8D20-8D21 8D28 8D30 "
                "8D38 8D3F-8D40 8D47-8D48 8D4F-8D50 8D57-8D58 8D5E-8D5F 8D6B 8D72 8D7B 8D81 "
                "8D89 8D91 8D98-8D99 8DA0-8DA1 8DA8 8DB0 8DB8 8DBF-8DC0 8DC7-8DC8 8DCF-8DD0 "
                "8DD7-8DD8 8DDE-8DDF 8DEB 8DF2 8DFB 8E01 8E09 8E11 8E18-8E19 8E20-8E21 "
                "8E28 8E30 8E38 8E3F-8E40 8E47-8E48 8E4F-8E50 8E57-8E58 8E5E-8E5F 8E6B 8E72 "
                "8E7B 8E81 8E89 8E91 8E98-8E99 8EA0-8EA1 8EA8 8EB0 8EB8 8EBF-8EC0 "
                "8EC7-8EC8 8ECF-8ED0 8ED7-8ED8 8EDE-8EDF 8EEB 8EF2 8EFB 8F01 8F09 8F11 "
                "8F18-8F19 8F20-8F21 8F28 8F30 8F38 8F3F-8F40 8F47-8F48 8F4F-8F50 8F57-8F58 8F5E-8F5F "
                "8F6B 8F72 8F7B 8F81 8F89 8F91 8F98-8F99 8FA0-8FA1 8FA8 8FB0 "
                "8FB8 8FBF-8FC0 8FC7-8FC8 8FCF-8FD0 8FD7-8FD8 8FDE-8FDF 8FEB 8FF2 8FFB 9001 "
                "9009 9011 9018-9019 9020-9021 9028 9030 9038 903F-9040 9047-9048 904F-9050 "
                "9057-9058 905E-905F 906B 9072 907B 9081 9089 9091 9098-9099 90A0-90A1 "
                "90A8 90B0 90B8 90BF-90C0 90C7-90C8 90CF-90D0 90D7-90D8 90DE-90DF 90EB 90F2 "
                "90FB 9101 9109 9111 9118-9119 9120-9121 9128 9130 9138 913F-9140 "
                "9147-9148 914F-9150 9157-9158 915E-915F 916B 9172 917B 9181 9189 9191 "
                "9198-9199 91A0-91A1 91A8 91B0 91B8 91BF-91C0 91C7-91C8 91CF-91D0 91D7-91D8 91DE-91DF "
                "91EB 91F2 91FB 9201 9209 9211 9218-9219 9220-9221 9228 9230 "
                "9238 923F-9240 9247-9248 924F-9250 9257-9258 925E-925F 926B 9272 927B 9281 "
                "9289 9291 9298-9299 92A0-92A1 92A8 92B0 92B8 92BF-92C0 92C7-92C8 92CF-92D0 "
                "92D7-92D8 92DE-92DF 92EB 92F2 92FB 9301 9309 9311 9318-9319 9320-9321 "
                "9328 9330 9338 933F-9340 9347-9348 934F-9350 9357-9358 935E-935F 936B 9372 "
                "937B 9381 9389 9391 9398-9399 93A0-93A1 93A8 93B0 93B8 93BF-93C0 "
                "93C7-93C8 93CF-93D0 93D7-93D8 93DE-93DF 93EB 93F2 93FB 9401 9409 9411 "
                "9418-9419 9420-9421 9428 9430 9438 943F-9440 9447-9448 944F-9450 9457-9458 945E-945F "
                "946B 9472 947B 9481 9489 9491 9498-9499 94A0-94A1 94A8 94B0 "
                "94B8 94BF-94C0 94C7-94C8 94CF-94D0 94D7-94D8 94DE-94DF 94EB 94F2 94FB 9501 "
                "9509 9511 9518-9519 9520-9521 9528 9530 9538 953F-9540 9547-9548 954F-9550 "
                "9557-9558 955E-955F 956B 9572 957B 9581 9589 9591 9598-9599 95A0-95A1 "
                "95A8 95B0 95B8 95BF-95C0 95C7-95C8 95CF-95D0 95D7-95D8 95DE-95DF 95EB 95F2 "
                "95FB 9601 9609 9611 9618-9619 9620-9621 9628 9630 9638 963F-9640 "
                "9647-9648 964F-9650 9657-9658 965E-965F 966B 9672 967B 9681 9689 9691 "
                "9698-9699 96A0-96A1 96A8 96B0 96B8 96BF-96C0 96C7-96C8 96CF-96D0 96D7-96D8 96DE-96DF "
                "96EB 96F2 96FB 9701 9709 9711 9718-9719 9720-9721 9728 9730 "
                "9738 973F-9740 9747-9748 974F-9750 9757-9758 975E-975F 976B 9772 977B 9781 "
                "9789 9791 9798-9799 97A0-97A1 97A8 97B0 97B8 97BF-97C0 97C7-97C8 97CF-97D0 "
                "97D7-97D8 97DE-97DF 97EB 97F2 97FB 9801 9809 9811 9818-9819 9820-9821 "
                "9828 9830 9838 983F-9840 9847-9848 984F-9850 9857-9858 985E-985F 986B 9872 "
                "987B 9881 9889 9891 9898-9899 98A0-98A1 98A8 98B0 98B8 98BF-98C0 "
                "98C7-98C8 98CF-98D0 98D7-98D8 98DE-98DF 98EB 98F2 98FB 9901 9909 9911 "
                "9918-9919 9920-9921 9928 9930 9938 993F-9940 9947-9948 994F-9950 9957-9958 995E-995F "
                "996B 9972 997B 9981 9989 9991 9998-9999 99A0-99A1 99A8 99B0 "
                "99B8 99BF-99C0 99C7-99C8 99CF-99D0 99D7-99D8 99DE-99DF 99EB 99F2 99FB 9A01 "
                "9A09 9A11 9A18-9A19 9A20-9A21 9A28 9A30 9A38 9A3F-9A40 9A47-9A48 9A4F-9A50 "
                "9A57-9A58 9A5E-9A5F 9A6B 9A72 9A7B 9A81 9A89 9A91 9A98-9A99 9AA0-9AA1 "
                "9AA8 9AB0 9AB8 9ABF-9AC0 9AC7-9AC8 9ACF-9AD0 9AD7-9AD8 9ADE-9ADF 9AEB 9AF2 "
                "9AFB 9B01 9B09 9B11 9B18-9B19 9B20-9B21 9B28 9B30 9B38 9B3F-9B40 "
                "9B47-9B48 9B4F-9B50 9B57-9B58 9B5E-9B5F 9B6B 9B72 9B7B 9B81 9B89 9B91 "
                "9B98-9B99 9BA0-9BA1 9BA8 9BB0 9BB8 9BBF-9BC0 9BC7-9BC8 9BCF-9BD0 9BD7-9BD8 9BDE-9BDF "
                "9BEB 9BF2 9BFB 9C01 9C09 9C11 9C18-9C19 9C20-9C21 9C28 9C30 "
                "9C38 9C3F-9C40 9C47-9C48 9C4F-9C50 9C57-9C58 9C5E-9C5F 9C6B 9C72 9C7B 9C81 "
                "9C89 9C91 9C98-9C99 9CA0-9CA1 9CA8 9CB0 9CB8 9CBF-9CC0 9CC7-9CC8 9CCF-9CD0 "
                "9CD7-9CD8 9CDE-9CDF 9CEB 9CF2 9CFB 9D01 9D09 9D11 9D18-9D19 9D20-9D21 "
                "9D28 9D30 9D38 9D3F-9D40 9D47-9D48 9D4F-9D50 9D57-9D58 9D5E-9D5F 9D6B 9D72 "
                "9D7B 9D81 9D89 9D91 9D98-9D99 9DA0-9DA1 9DA8 9DB0 9DB8 9DBF-9DC0 "
                "9DC7-9DC8 9DCF-9DD0 9DD7-9DD8 9DDE-9DDF 9DEB 9DF2 9DFB 9E01 9E09 9E11 "
                "9E18-9E19 9E20-9E21 9E28 9E30 9E38 9E3F-9E40 9E47-9E48 9E4F-9E50 9E57-9E58 9E5E-9E5F "
                "9E6B 9E72 9E7B 9E81 9E89 9E91 9E98-9E99 9EA0-9EA1 9EA8 9EB0 "
                "9EB8 9EBF-9EC0 9EC7-9EC8 9ECF-9ED0 9ED7-9ED8 9EDE-9EDF 9EEB 9EF2 9EFB 9F01 "
                "9F09 9F11 9F18-9F19 9F20-9F21 9F28 9F30 9F38 9F3F-9F40 9F47-9F48 9F4F-9F50 "
                "9F57-9F58 9F5E-9F5F 9F6B 9F72 9F7B 9F81 9F89 9F91 9F98-9F99 9FA0-9FA1 "
                "9FA8 9FB0 9FB8 9FBF-9FC0 9FC7-9FC8 9FCF-9FD0 9FD7-9FD8 9FDE-9FDF 9FEB 9FF2 "
                "9FFB A001 A009 A011 A018-A019 A020-A021 A028 A030 A038 A03F-A040 "
                "A047-A048 A04F-A050 A057-A058 A05E-A05F A06B A072 A07B A081 A089 A091 "
                "A098-A099 A0A0-A0A1 A0A8 A0B0 A0B8 A0BF-A0C0 A0C7-A0C8 A0CF-A0D0 A0D7-A0D8 A0DE-A0DF "
                "A0EB A0F2 A0FB A101 A109 A111 A118-A119 A120-A121 A128 A130 "
                "A138 A13F-A140 A147-A148 A14F-A150 A157-A158 A15E-A15F A16B A172 A17B A181 "
                "A189 A191 A198-A199 A1A0-A1A1 A1A8 A1B0 A1B8 A1BF-A1C0 A1C7-A1C8 A1CF-A1D0 "
                "A1D7-A1D8 A1DE-A1DF A1EB A1F2 A1FB A201 A209 A211 A218-A219 A220-A221 "
                "A228 A230 A238 A23F-A240 A247-A248 A24F-A250 A257-A258 A25E-A25F A26B A272 "
                "A27B A281 A289 A291 A298-A299 A2A0-A2A1 A2A8 A2B0 A2B8 A2BF-A2C0 "
                "A2C7-A2C8 A2CF-A2D0 A2D7-A2D8 A2DE-A2DF A2EB A2F2 A2FB A301 A309 A311 "
                "A318-A319 A320-A321 A328 A330 A338 A33F-A340 A347-A348 A34F-A350 A357-A358 A35E-A35F "
                "A36B A372 A37B A381 A389 A391 A398-A399 A3A0-A3A1 A3A8 A3B0 "
                "A3B8 A3BF-A3C0 A3C7-A3C8 A3CF-A3D0 A3D7-A3D8 A3DE-A3DF A3EB A3F2 A3FB A401 "
                "A409 A411 A418-A419 A420-A421 A428 A430 A438 A43F-A440 A447-A448 A44F-A450 "
                "A457-A458 A45E-A45F A46B A472 A47B A481 A489 A491 A498-A499 A4A0-A4A1 "
                "A4A8 A4B0 A4B8 A4BF-A4C0 A4C7-A4C8 A4CF-A4D0 A4D7-A4D8 A4DE-A4DF A4EB A4F2 "
                "A4FB A501 A509 A511 A518-A519 A520-A521 A528 A530 A538 A53F-A540 "
                "A547-A548 A54F-A550 A557-A558 A55E-A55F A56B A572 A57B A581 A589 A591 "
                "A598-A599 A5A0-A5A1 A5A8 A5B0 A5B8 A5BF-A5C0 A5C7-A5C8 A5CF-A5D0 A5D7-A5D8 A5DE-A5DF "
                "A5EB A5F2 A5FB A601 A609 A611 A618-A619 A620-A621 A628 A630 "
                "A638 A63F-A640 A647-A648 A64F-A650 A657-A658 A65E-A65F A66B A672 A67B A681 "
                "A689 A691 A698-A699 A6A0-A6A1 A6A8 A6B0 A6B8 A6BF-A6C0 A6C7-A6C8 A6CF-A6D0 "
                "A6D7-A6D8 A6DE-A6DF A6EB A6F2 A6FB A701 A709 A711 A718-A719 A720-A721 "
                "A728 A730 A738 A73F-A740 A747-A748 A74F-A750 A757-A758 A75E-A75F A76B A772 "
                "A77B A781 A789 A791 A798-A799 A7A0-A7A1 A7A8 A7B0 A7B8 A7BF-A7C0 "
                "A7C7-A7C8 A7CF-A7D0 A7D7-A7D8 A7DE-A7DF A7EB A7F2 A7FB A801 A809 A811 "
                "A818-A819 A820-A821 A828 A830 A838 A83F-A840 A847-A848 A84F-A850 A857-A858 A85E-A85F "
                "A86B A872 A87B A881 A889 A891 A898-A899 A8A0-A8A1 A8A8 A8B0 "
                "A8B8 A8BF-A8C0 A8C7-A8C8 A8CF-A8D0 A8D7-A8D8 A8DE-A8DF A8EB A8F2 A8FB A901 "
                "A909 A911 A918-A919 A920-A921 A928 A930 A938 A93F-A940 A947-A948 A94F-A950 "
                "A957-A958 A95E-A95F A96B A972 A97B A981 A989 A991 A998-A999 A9A0-A9A1 "
                "A9A8 A9B0 A9B8 A9BF-A9C0 A9C7-A9C8 A9CF-A9D0 A9D7-A9D8 A9DE-A9DF A9EB A9F2 "
                "A9FB AA01 AA09 AA11 AA18-AA19 AA20-AA21 AA28 AA30 AA38 AA3F-AA40 "
                "AA47-AA48 AA4F-AA50 AA57-AA58 AA5E-AA5F AA6B AA72 AA7B AA81 AA89 AA91 "
                "AA98-AA99 AAA0-AAA1 AAA8 AAB0 AAB8 AABF-AAC0 AAC7-AAC8 AACF-AAD0 AAD7-AAD8 AADE-AADF "
                "AAEB AAF2 AAFB AB01 AB09 AB11 AB18-AB19 AB20-AB21 AB28 AB30 "
                "AB38 AB3F-AB40 AB47-AB48 AB4F-AB50 AB57-AB58 AB5E-AB5F AB6B AB72 AB7B AB81 "
                "AB89 AB91 AB98-AB99 ABA0-ABA1 ABA8 ABB0 ABB8 ABBF-ABC0 ABC7-ABC8 ABCF-ABD0 "
                "ABD7-ABD8 ABDE-ABDF ABEB ABF2 ABFB AC01 AC09 AC11 AC18-AC19 AC20-AC21 "
                "AC28 AC30 AC38 AC3F-AC40 AC47-AC48 AC4F-AC50 AC57-AC58 AC5E-AC5F AC6B AC72 "
                "AC7B AC81 AC89 AC91 AC98-AC99 ACA0-ACA1 ACA8 ACB0 ACB8 ACBF-ACC0 "
                "ACC7-ACC8 ACCF-ACD0 ACD7-ACD8 ACDE-ACDF ACEB ACF2 ACFB AD01 AD09 AD11 "
                "AD18-AD19 AD20-AD21 AD28 AD30 AD38 AD3F-AD40 AD47-AD48 AD4F-AD50 AD57-AD58 AD5E-AD5F "
                "AD6B AD72 AD7B AD81 AD89 AD91 AD98-AD99 ADA0-ADA1 ADA8 ADB0 "
                "ADB8 ADBF-ADC0 ADC7-ADC8 ADCF-ADD0 ADD7-ADD8 ADDE-ADDF ADEB ADF2 ADFB AE01 "
                "AE09 AE11 AE18-AE19 AE20-AE21 AE28 AE30 AE38 AE3F-AE40 AE47-AE48 AE4F-AE50 "
                "AE57-AE58 AE5E-AE5F AE6B AE72 AE7B AE81 AE89 AE91 AE98-AE99 AEA0-AEA1 "
                "AEA8 AEB0 AEB8 AEBF-AEC0 AEC7-AEC8 AECF-AED0 AED7-AED8 AEDE-AEDF AEEB AEF2 "
                "AEFB AF01 AF09 AF11 AF18-AF19 AF20-AF21 AF28 AF30 AF38 AF3F-AF40 "
                "AF47-AF48 AF4F-AF50 AF57-AF58 AF5E-AF5F AF6B AF72 AF7B AF81 AF89 AF91 "
                "AF98-AF99 AFA0-AFA1 AFA8 AFB0 AFB8 AFBF-AFC0 AFC7-AFC8 AFCF-AFD0 AFD7-AFD8 AFDE-AFDF "
                "AFEB AFF2 AFFB B001 B009 B011 B018-B019 B020-B021 B028 B030 "
                "B038 B03F-B040 B047-B048 B04F-B050 B057-B058 B05E-B05F B06B B072 B07B B081 "
                "B089 B091 B098-B099 B0A0-B0A1 B0A8 B0B0 B0B8 B0BF-B0C0 B0C7-B0C8 B0CF-B0D0 "
                "B0D7-B0D8 B0DE-B0DF B0EB B0F2 B0FB B101 B109 B111 B118-B119 B120-B121 "
                "B128 B130 B138 B13F-B140 B147-B148 B14F-B150 B157-B158 B15E-B15F B16B B172 "
                "B17B B181 B189 B191 B198-B199 B1A0-B1A1 B1A8 B1B0 B1B8 B1BF-B1C0 "
                "B1C7-B1C8 B1CF-B1D0 B1D7-B1D8 B1DE-B1DF B1EB B1F2 B1FB B201 B209 B211 "
                "B218-B219 B220-B221 B228 B230 B238 B23F-B240 B247-B248 B24F-B250 B257-B258 B25E-B25F "
                "B26B B272 B27B B281 B289 B291 B298-B299 B2A0-B2A1 B2A8 B2B0 "
                "B2B8 B2BF-B2C0 B2C7-B2C8 B2CF-B2D0 B2D7-B2D8 B2DE-B2DF B2EB B2F2 B2FB B301 "
                "B309 B311 B318-B319 B320-B321 B328 B330 B338 B33F-B340 B347-B348 B34F-B350 "
                "B357-B358 B35E-B35F B36B B372 B37B B381 B389 B391 B398-B399 B3A0-B3A1 "
                "B3A8 B3B0 B3B8 B3BF-B3C0 B3C7-B3C8 B3CF-B3D0 B3D7-B3D8 B3DE-B3DF B3EB B3F2 "
                "B3FB B401 B409 B411 B418-B419 B420-B421 B428 B430 B438 B43F-B440 "
                "B447-B448 B44F-B450 B457-B458 B45E-B45F B46B B472 B47B B481 B489 B491 "
                "B498-B499 B4A0-B4A1 B4A8 B4B0 B4B8 B4BF-B4C0 B4C7-B4C8 B4CF-B4D0 B4D7-B4D8 B4DE-B4DF "
                "B4EB B4F2 B4FB B501 B509 B511 B518-B519 B520-B521 B528 B530 "
                "B538 B53F-B540 B547-B548 B54F-B550 B557-B558 B55E-B55F B56B B572 B57B B581 "
                "B589 B591 B598-B599 B5A0-B5A1 B5A8 B5B0 B5B8 B5BF-B5C0 B5C7-B5C8 B5CF-B5D0 "
                "B5D7-B5D8 B5DE-B5DF B5EB B5F2 B5FB B601 B609 B611 B618-B619 B620-B621 "
                "B628 B630 B638 B63F-B640 B647-B648 B64F-B650 B657-B658 B65E-B65F B66B B672 "
                "B67B B681 B689 B691 B698-B699 B6A0-B6A1 B6A8 B6B0 B6B8 B6BF-B6C0 "
                "B6C7-B6C8 B6CF-B6D0 B6D7-B6D8 B6DE-B6DF B6EB B6F2 B6FB B701 B709 B711 "
                "B718-B719 B720-B721 B728 B730 B738 B73F-B740 B747-B748 B74F-B750 B757-B758 B75E-B75F "
                "B76B B772 B77B B781 B789 B791 B798-B799 B7A0-B7A1 B7A8 B7B0 "
                "B7B8 B7BF-B7C0 B7C7-B7C8 B7CF-B7D0 B7D7-B7D8 B7DE-B7DF B7EB B7F2 B7FB B801 "
                "B809 B811 B818-B819 B820-B821 B828 B830 B838 B83F-B840 B847-B848 B84F-B850 "
                "B857-B858 B85E-B85F B86B B872 B87B B881 B889 B891 B898-B899 B8A0-B8A1 "
                "B8A8 B8B0 B8B8 B8BF-B8C0 B8C7-B8C8 B8CF-B8D0 B8D7-B8D8 B8DE-B8DF B8EB B8F2 "
                "B8FB B901 B909 B911 B918-B919 B920-B921 B928 B930 B938 B93F-B940 "
                "B947-B948 B94F-B950 B957 B95E-B95F B96B B972 B97B B981 B989 B991 "
                "B998-B999 B9A0-B9A1 B9A8 B9B0 B9B8 B9BF-B9C0 B9C7-B9C8 B9CF-B9D0 B9D7 B9DE-B9DF "
                "B9EB B9F2 B9FB BA01 BA09 BA11 BA18-BA19 BA20-BA21 BA28 BA30 "
                "BA38 BA3F-BA40 BA47-BA48 BA4F-BA50 BA57 BA5E-BA5F BA6B BA72 BA7B BA81 "
                "BA89 BA91 BA98-BA99 BAA0 BAA8 BAB0 BAB8 BABF-BAC0 BAC7-BAC8 BACF-BAD0 "
                "BAD7 BADF BAEB BAF2 BAFB BB01 BB09 BB11 BB18-BB19 BB20 "
                "BB28 BB30 BB38 BB3F-BB40 BB47-BB48 BB4F-BB50 BB57 BB5F BB6B BB72 "
                "BB7B BB81 BB89 BB91 BB98-BB99 BBA0 BBA8 BBB0 BBB8 BBBF-BBC0 "
                "BBC7-BBC8 BBCF BBD7 BBDF BBEB BBF2 BC01 BC09 BC11 BC18 "
                "BC20 BC28 BC30 BC38 BC3F-BC40 BC47 BC4F BC57 BC5F BC6B "
                "BC72 BC81 BC89 BCA0 BCA8 BCB0 BCB8 BCC7 BCCF BCD7 "
                "BCDF BD01 BD09 BEE0 BEEC BF0C BF17 BF20 BF24 BF28 "
                "BF2F BF44 BF4C BF53 BF5A BFD7 BFE0 BFEA-BFEC BFFA-BFFE"
            ),
        },
        "wormhole_b0": {
            -1: (
                "0081 0089 0091 0098-0099 00A0-00A1 00A8 00B0 00B8 00BF-00C0 00C7-00C8 "
                "00CF-00D0 00D7-00D8 00DE-00DF 00EB 00F2 00FB 0101 0109 0111 0118-0119 "
                "0120-0121 0128 0130 0138 013F-0140 0147-0148 014F-0150 0157-0158 015E-015F 016B "
                "0172 017B 0181 0189 0191 0198-0199 01A0-01A1 01A8 01B0 01B8 "
                "01BF-01C0 01C7-01C8 01CF-01D0 01D7-01D8 01DE-01DF 01EB 01F2 01FB 0201 0209 "
                "0211 0218-0219 0220-0221 0228 0230 0238 023F-0240 0247-0248 024F-0250 0257-0258 "
                "025E-025F 026B 0272 027B 0281 0289 0291 0298-0299 02A0-02A1 02A8 "
                "02B0 02B8 02BF-02C0 02C7-02C8 02CF-02D0 02D7-02D8 02DE-02DF 02EB 02F2 02FB "
                "0301 0309 0311 0318-0319 0320-0321 0328 0330 0338 033F-0340 0347-0348 "
                "034F-0350 0357-0358 035E-035F 036B 0372 037B 0381 0389 0391 0398-0399 "
                "03A0-03A1 03A8 03B0 03B8 03BF-03C0 03C7-03C8 03CF-03D0 03D7-03D8 03DE-03DF 03EB "
                "03F2 03FB 0401 0409 0411 0418-0419 0420-0421 0428 0430 0438 "
                "043F-0440 0447-0448 044F-0450 0457-0458 045E-045F 046B 0472 047B 0481 0489 "
                "0491 0498-0499 04A0-04A1 04A8 04B0 04B8 04BF-04C0 04C7-04C8 04CF-04D0 04D7-04D8 "
                "04DE-04DF 04EB 04F2 04FB 0501 0509 0511 0518-0519 0520-0521 0528 "
                "0530 0538 053F-0540 0547-0548 054F-0550 0557-0558 055E-055F 056B 0572 057B "
                "0581 0589 0591 0598-0599 05A0-05A1 05A8 05B0 05B8 05BF-05C0 05C7-05C8 "
                "05CF-05D0 05D7-05D8 05DE-05DF 05EB 05F2 05FB 0601 0609 0611 0618-0619 "
                "0620-0621 0628 0630 0638 063F-0640 0647-0648 064F-0650 0657-0658 065E-065F 066B "
                "0672 067B 0681 0689 0691 0698-0699 06A0-06A1 06A8 06B0 06B8 "
                "06BF-06C0 06C7-06C8 06CF-06D0 06D7-06D8 06DE-06DF 06EB 06F2 06FB 0701 0709 "
                "0711 0718-0719 0720-0721 0728 0730 0738 073F-0740 0747-0748 074F-0750 0757-0758 "
                "075E-075F 076B 0772 077B 0781 0789 0791 0798-0799 07A0-07A1 07A8 "
                "07B0 07B8 07BF-07C0 07C7-07C8 07CF-07D0 07D7-07D8 07DE-07DF 07EB 07F2 07FB "
                "0801 0809 0811 0818-0819 0820-0821 0828 0830 0838 083F-0840 0847-0848 "
                "084F-0850 0857-0858 085E-085F 086B 0872 087B 0881 0889 0891 0898-0899 "
                "08A0-08A1 08A8 08B0 08B8 08BF-08C0 08C7-08C8 08CF-08D0 08D7-08D8 08DE-08DF 08EB "
                "08F2 08FB 0901 0909 0911 0918-0919 0920-0921 0928 0930 0938 "
                "093F-0940 0947-0948 094F-0950 0957-0958 095E-095F 096B 0972 097B 0981 0989 "
                "0991 0998-0999 09A0-09A1 09A8 09B0 09B8 09BF-09C0 09C7-09C8 09CF-09D0 09D7-09D8 "
                "09DE-09DF 09EB 09F2 09FB 0A01 0A09 0A11 0A18-0A19 0A20-0A21 0A28 "
                "0A30 0A38 0A3F-0A40 0A47-0A48 0A4F-0A50 0A57-0A58 0A5E-0A5F 0A6B 0A72 0A7B "
                "0A81 0A89 0A91 0A98-0A99 0AA0-0AA1 0AA8 0AB0 0AB8 0ABF-0AC0 0AC7-0AC8 "
                "0ACF-0AD0 0AD7-0AD8 0ADE-0ADF 0AEB 0AF2 0AFB 0B01 0B09 0B11 0B18-0B19 "
                "0B20-0B21 0B28 0B30 0B38 0B3F-0B40 0B47-0B48 0B4F-0B50 0B57-0B58 0B5E-0B5F 0B6B "
                "0B72 0B7B 0B81 0B89 0B91 0B98-0B99 0BA0-0BA1 0BA8 0BB0 0BB8 "
                "0BBF-0BC0 0BC7-0BC8 0BCF-0BD0 0BD7-0BD8 0BDE-0BDF 0BEB 0BF2 0BFB 0C01 0C09 "
                "0C11 0C18-0C19 0C20-0C21 0C28 0C30 0C38 0C3F-0C40 0C47-0C48 0C4F-0C50 0C57-0C58 "
                "0C5E-0C5F 0C6B 0C72 0C7B 0C81 0C89 0C91 0C98-0C99 0CA0-0CA1 0CA8 "
                "0CB0 0CB8 0CBF-0CC0 0CC7-0CC8 0CCF-0CD0 0CD7-0CD8 0CDE-0CDF 0CEB 0CF2 0CFB "
                "0D01 0D09 0D11 0D18-0D19 0D20-0D21 0D28 0D30 0D38 0D3F-0D40 0D47-0D48 "
                "0D4F-0D50 0D57-0D58 0D5E-0D5F 0D6B 0D72 0D7B 0D81 0D89 0D91 0D98-0D99 "
                "0DA0-0DA1 0DA8 0DB0 0DB8 0DBF-0DC0 0DC7-0DC8 0DCF-0DD0 0DD7-0DD8 0DDE-0DDF 0DEB "
                "0DF2 0DFB 0E01 0E09 0E11 0E18-0E19 0E20-0E21 0E28 0E30 0E38 "
                "0E3F-0E40 0E47-0E48 0E4F-0E50 0E57-0E58 0E5E-0E5F 0E6B 0E72 0E7B 0E81 0E89 "
                "0E91 0E98-0E99 0EA0-0EA1 0EA8 0EB0 0EB8 0EBF-0EC0 0EC7-0EC8 0ECF-0ED0 0ED7-0ED8 "
                "0EDE-0EDF 0EEB 0EF2 0EFB 0F01 0F09 0F11 0F18-0F19 0F20-0F21 0F28 "
                "0F30 0F38 0F3F-0F40 0F47-0F48 0F4F-0F50 0F57-0F58 0F5E-0F5F 0F6B 0F72 0F7B "
                "0F81 0F89 0F91 0F98-0F99 0FA0-0FA1 0FA8 0FB0 0FB8 0FBF-0FC0 0FC7-0FC8 "
                "0FCF-0FD0 0FD7-0FD8 0FDE-0FDF 0FEB 0FF2 0FFB 1001 1009 1011 1018-1019 "
                "1020-1021 1028 1030 1038 103F-1040 1047-1048 104F-1050 1057-1058 105E-105F 106B "
                "1072 107B 1081 1089 1091 1098-1099 10A0-10A1 10A8 10B0 10B8 "
                "10BF-10C0 10C7-10C8 10CF-10D0 10D7-10D8 10DE-10DF 10EB 10F2 10FB 1101 1109 "
                "1111 1118-1119 1120-1121 1128 1130 1138 113F-1140 1147-1148 114F-1150 1157-1158 "
                "115E-115F 116B 1172 117B 1181 1189 1191 1198-1199 11A0-11A1 11A8 "
                "11B0 11B8 11BF-11C0 11C7-11C8 11CF-11D0 11D7-11D8 11DE-11DF 11EB 11F2 11FB "
                "1201 1209 1211 1218-1219 1220-1221 1228 1230 1238 123F-1240 1247-1248 "
                "124F-1250 1257-1258 125E-125F 126B 1272 127B 1281 1289 1291 1298-1299 "
                "12A0-12A1 12A8 12B0 12B8 12BF-12C0 12C7-12C8 12CF-12D0 12D7-12D8 12DE-12DF 12EB "
                "12F2 12FB 1301 1309 1311 1318-1319 1320-1321 1328 1330 1338 "
                "133F-1340 1347-1348 134F-1350 1357-1358 135E-135F 136B 1372 137B 1381 1389 "
                "1391 1398-1399 13A0-13A1 13A8 13B0 13B8 13BF-13C0 13C7-13C8 13CF-13D0 13D7-13D8 "
                "13DE-13DF 13EB 13F2 13FB 1401 1409 1411 1418-1419 1420-1421 1428 "
                "1430 1438 143F-1440 1447-1448 144F-1450 1457-1458 145E-145F 146B 1472 147B "
                "1481 1489 1491 1498-1499 14A0-14A1 14A8 14B0 14B8 14BF-14C0 14C7-14C8 "
                "14CF-14D0 14D7-14D8 14DE-14DF 14EB 14F2 14FB 1501 1509 1511 1518-1519 "
                "1520-1521 1528 1530 1538 153F-1540 1547-1548 154F-1550 1557-1558 155E-155F 156B "
                "1572 157B 1581 1589 1591 1598-1599 15A0-15A1 15A8 15B0 15B8 "
                "15BF-15C0 15C7-15C8 15CF-15D0 15D7-15D8 15DE-15DF 15EB 15F2 15FB 1601 1609 "
                "1611 1618-1619 1620-1621 1628 1630 1638 163F-1640 1647-1648 164F-1650 1657-1658 "
                "165E-165F 166B 1672 167B 1681 1689 1691 1698-1699 16A0-16A1 16A8 "
                "16B0 16B8 16BF-16C0 16C7-16C8 16CF-16D0 16D7-16D8 16DE-16DF 16EB 16F2 16FB "
                "1701 1709 1711 1718-1719 1720-1721 1728 1730 1738 173F-1740 1747-1748 "
                "174F-1750 1757-1758 175E-175F 176B 1772 177B 1781 1789 1791 1798-1799 "
                "17A0-17A1 17A8 17B0 17B8 17BF-17C0 17C7-17C8 17CF-17D0 17D7-17D8 17DE-17DF 17EB "
                "17F2 17FB 1801 1809 1811 1818-1819 1820-1821 1828 1830 1838 "
                "183F-1840 1847-1848 184F-1850 1857-1858 185E-185F 186B 1872 187B 1881 1889 "
                "1891 1898-1899 18A0-18A1 18A8 18B0 18B8 18BF-18C0 18C7-18C8 18CF-18D0 18D7-18D8 "
                "18DE-18DF 18EB 18F2 18FB 1901 1909 1911 1918-1919 1920-1921 1928 "
                "1930 1938 193F-1940 1947-1948 194F-1950 1957-1958 195E-195F 196B 1972 197B "
                "1981 1989 1991 1998-1999 19A0-19A1 19A8 19B0 19B8 19BF-19C0 19C7-19C8 "
                "19CF-19D0 19D7-19D8 19DE-19DF 19EB 19F2 19FB 1A01 1A09 1A11 1A18-1A19 "
                "1A20-1A21 1A28 1A30 1A38 1A3F-1A40 1A47-1A48 1A4F-1A50 1A57-1A58 1A5E-1A5F 1A6B "
                "1A72 1A7B 1A81 1A89 1A91 1A98-1A99 1AA0-1AA1 1AA8 1AB0 1AB8 "
                "1ABF-1AC0 1AC7-1AC8 1ACF-1AD0 1AD7-1AD8 1ADE-1ADF 1AEB 1AF2 1AFB 1B01 1B09 "
                "1B11 1B18-1B19 1B20-1B21 1B28 1B30 1B38 1B3F-1B40 1B47-1B48 1B4F-1B50 1B57-1B58 "
                "1B5E-1B5F 1B6B 1B72 1B7B 1B81 1B89 1B91 1B98-1B99 1BA0-1BA1 1BA8 "
                "1BB0 1BB8 1BBF-1BC0 1BC7-1BC8 1BCF-1BD0 1BD7-1BD8 1BDE-1BDF 1BEB 1BF2 1BFB "
                "1C01 1C09 1C11 1C18-1C19 1C20-1C21 1C28 1C30 1C38 1C3F-1C40 1C47-1C48 "
                "1C4F-1C50 1C57-1C58 1C5E-1C5F 1C6B 1C72 1C7B 1C81 1C89 1C91 1C98-1C99 "
                "1CA0-1CA1 1CA8 1CB0 1CB8 1CBF-1CC0 1CC7-1CC8 1CCF-1CD0 1CD7-1CD8 1CDE-1CDF 1CEB "
                "1CF2 1CFB 1D01 1D09 1D11 1D18-1D19 1D20-1D21 1D28 1D30 1D38 "
                "1D3F-1D40 1D47-1D48 1D4F-1D50 1D57-1D58 1D5E-1D5F 1D6B 1D72 1D7B 1D81 1D89 "
                "1D91 1D98-1D99 1DA0-1DA1 1DA8 1DB0 1DB8 1DBF-1DC0 1DC7-1DC8 1DCF-1DD0 1DD7-1DD8 "
                "1DDE-1DDF 1DEB 1DF2 1DFB 1E01 1E09 1E11 1E18-1E19 1E20-1E21 1E28 "
                "1E30 1E38 1E3F-1E40 1E47-1E48 1E4F-1E50 1E57-1E58 1E5E-1E5F 1E6B 1E72 1E7B "
                "1E81 1E89 1E91 1E98-1E99 1EA0-1EA1 1EA8 1EB0 1EB8 1EBF-1EC0 1EC7-1EC8 "
                "1ECF-1ED0 1ED7-1ED8 1EDE-1EDF 1EEB 1EF2 1EFB 1F01 1F09 1F11 1F18-1F19 "
                "1F20-1F21 1F28 1F30 1F38 1F3F-1F40 1F47-1F48 1F4F-1F50 1F57-1F58 1F5E-1F5F 1F6B "
                "1F72 1F7B 1F81 1F89 1F91 1F98-1F99 1FA0-1FA1 1FA8 1FB0 1FB8 "
                "1FBF-1FC0 1FC7-1FC8 1FCF-1FD0 1FD7-1FD8 1FDE-1FDF 1FEB 1FF2 1FFB 2001 2009 "
                "2011 2018-2019 2020-2021 2028 2030 2038 203F-2040 2047-2048 204F-2050 2057-2058 "
                "205E-205F 206B 2072 207B 2081 2089 2091 2098-2099 20A0-20A1 20A8 "
                "20B0 20B8 20BF-20C0 20C7-20C8 20CF-20D0 20D7-20D8 20DE-20DF 20EB 20F2 20FB "
                "2101 2109 2111 2118-2119 2120-2121 2128 2130 2138 213F-2140 2147-2148 "
                "214F-2150 2157-2158 215E-215F 216B 2172 217B 2181 2189 2191 2198-2199 "
                "21A0-21A1 21A8 21B0 21B8 21BF-21C0 21C7-21C8 21CF-21D0 21D7-21D8 21DE-21DF 21EB "
                "21F2 21FB 2201 2209 2211 2218-2219 2220-2221 2228 2230 2238 "
                "223F-2240 2247-2248 224F-2250 2257-2258 225E-225F 226B 2272 227B 2281 2289 "
                "2291 2298-2299 22A0-22A1 22A8 22B0 22B8 22BF-22C0 22C7-22C8 22CF-22D0 22D7-22D8 "
                "22DE-22DF 22EB 22F2 22FB 2301 2309 2311 2318-2319 2320-2321 2328 "
                "2330 2338 233F-2340 2347-2348 234F-2350 2357-2358 235E-235F 236B 2372 237B "
                "2381 2389 2391 2398-2399 23A0-23A1 23A8 23B0 23B8 23BF-23C0 23C7-23C8 "
                "23CF-23D0 23D7-23D8 23DE-23DF 23EB 23F2 23FB 2401 2409 2411 2418-2419 "
                "2420-2421 2428 2430 2438 243F-2440 2447-2448 244F-2450 2457-2458 245E-245F 246B "
                "2472 247B 2481 2489 2491 2498-2499 24A0-24A1 24A8 24B0 24B8 "
                "24BF-24C0 24C7-24C8 24CF-24D0 24D7-24D8 24DE-24DF 24EB 24F2 24FB 2501 2509 "
                "2511 2518-2519 2520-2521 2528 2530 2538 253F-2540 2547-2548 254F-2550 2557-2558 "
                "255E-255F 256B 2572 257B 2581 2589 2591 2598-2599 25A0-25A1 25A8 "
                "25B0 25B8 25BF-25C0 25C7-25C8 25CF-25D0 25D7-25D8 25DE-25DF 25EB 25F2 25FB "
                "2601 2609 2611 2618-2619 2620-2621 2628 2630 2638 263F-2640 2647-2648 "
                "264F-2650 2657-2658 265E-265F 266B 2672 267B 2681 2689 2691 2698-2699 "
                "26A0-26A1 26A8 26B0 26B8 26BF-26C0 26C7-26C8 26CF-26D0 26D7-26D8 26DE-26DF 26EB "
                "26F2 26FB 2701 2709 2711 2718-2719 2720-2721 2728 2730 2738 "
                "273F-2740 2747-2748 274F-2750 2757-2758 275E-275F 276B 2772 277B 2781 2789 "
                "2791 2798-2799 27A0-27A1 27A8 27B0 27B8 27BF-27C0 27C7-27C8 27CF-27D0 27D7-27D8 "
                "27DE-27DF 27EB 27F2 27FB 2801 2809 2811 2818-2819 2820-2821 2828 "
                "2830 2838 283F-2840 2847-2848 284F-2850 2857-2858 285E-285F 286B 2872 287B "
                "2881 2889 2891 2898-2899 28A0-28A1 28A8 28B0 28B8 28BF-28C0 28C7-28C8 "
                "28CF-28D0 28D7-28D8 28DE-28DF 28EB 28F2 28FB 2901 2909 2911 2918-2919 "
                "2920-2921 2928 2930 2938 293F-2940 2947-2948 294F-2950 2957-2958 295E-295F 296B "
                "2972 297B 2981 2989 2991 2998-2999 29A0-29A1 29A8 29B0 29B8 "
                "29BF-29C0 29C7-29C8 29CF-29D0 29D7-29D8 29DE-29DF 29EB 29F2 29FB 2A01 2A09 "
                "2A11 2A18-2A19 2A20-2A21 2A28 2A30 2A38 2A3F-2A40 2A47-2A48 2A4F-2A50 2A57-2A58 "
                "2A5E-2A5F 2A6B 2A72 2A7B 2A81 2A89 2A91 2A98-2A99 2AA0-2AA1 2AA8 "
                "2AB0 2AB8 2ABF-2AC0 2AC7-2AC8 2ACF-2AD0 2AD7-2AD8 2ADE-2ADF 2AEB 2AF2 2AFB "
                "2B01 2B09 2B11 2B18-2B19 2B20-2B21 2B28 2B30 2B38 2B3F-2B40 2B47-2B48 "
                "2B4F-2B50 2B57-2B58 2B5E-2B5F 2B6B 2B72 2B7B 2B81 2B89 2B91 2B98-2B99 "
                "2BA0-2BA1 2BA8 2BB0 2BB8 2BBF-2BC0 2BC7-2BC8 2BCF-2BD0 2BD7-2BD8 2BDE-2BDF 2BEB "
                "2BF2 2BFB 2C01 2C09 2C11 2C18-2C19 2C20-2C21 2C28 2C30 2C38 "
                "2C3F-2C40 2C47-2C48 2C4F-2C50 2C57-2C58 2C5E-2C5F 2C6B 2C72 2C7B 2C81 2C89 "
                "2C91 2C98-2C99 2CA0-2CA1 2CA8 2CB0 2CB8 2CBF-2CC0 2CC7-2CC8 2CCF-2CD0 2CD7-2CD8 "
                "2CDE-2CDF 2CEB 2CF2 2CFB 2D01 2D09 2D11 2D18-2D19 2D20-2D21 2D28 "
                "2D30 2D38 2D3F-2D40 2D47-2D48 2D4F-2D50 2D57-2D58 2D5E-2D5F 2D6B 2D72 2D7B "
                "2D81 2D89 2D91 2D98-2D99 2DA0-2DA1 2DA8 2DB0 2DB8 2DBF-2DC0 2DC7-2DC8 "
                "2DCF-2DD0 2DD7-2DD8 2DDE-2DDF 2DEB 2DF2 2DFB 2E01 2E09 2E11 2E18-2E19 "
                "2E20-2E21 2E28 2E30 2E38 2E3F-2E40 2E47-2E48 2E4F-2E50 2E57-2E58 2E5E-2E5F 2E6B "
                "2E72 2E7B 2E81 2E89 2E91 2E98-2E99 2EA0-2EA1 2EA8 2EB0 2EB8 "
                "2EBF-2EC0 2EC7-2EC8 2ECF-2ED0 2ED7-2ED8 2EDE-2EDF 2EEB 2EF2 2EFB 2F01 2F09 "
                "2F11 2F18-2F19 2F20-2F21 2F28 2F30 2F38 2F3F-2F40 2F47-2F48 2F4F-2F50 2F57-2F58 "
                "2F5E-2F5F 2F6B 2F72 2F7B 2F81 2F89 2F91 2F98-2F99 2FA0-2FA1 2FA8 "
                "2FB0 2FB8 2FBF-2FC0 2FC7-2FC8 2FCF-2FD0 2FD7-2FD8 2FDE-2FDF 2FEB 2FF2 2FFB "
                "3001 3009 3011 3018-3019 3020-3021 3028 3030 3038 303F-3040 3047-3048 "
                "304F-3050 3057-3058 305E-305F 306B 3072 307B 3081 3089 3091 3098-3099 "
                "30A0-30A1 30A8 30B0 30B8 30BF-30C0 30C7-30C8 30CF-30D0 30D7-30D8 30DE-30DF 30EB "
                "30F2 30FB 3101 3109 3111 3118-3119 3120-3121 3128 3130 3138 "
                "313F-3140 3147-3148 314F-3150 3157-3158 315E-315F 316B 3172 317B 3181 3189 "
                "3191 3198-3199 31A0-31A1 31A8 31B0 31B8 31BF-31C0 31C7-31C8 31CF-31D0 31D7-31D8 "
                "31DE-31DF 31EB 31F2 31FB 3201 3209 3211 3218-3219 3220-3221 3228 "
                "3230 3238 323F-3240 3247-3248 324F-3250 3257-3258 325E-325F 326B 3272 327B "
                "3281 3289 3291 3298-3299 32A0-32A1 32A8 32B0 32B8 32BF-32C0 32C7-32C8 "
                "32CF-32D0 32D7-32D8 32DE-32DF 32EB 32F2 32FB 3301 3309 3311 3318-3319 "
                "3320-3321 3328 3330 3338 333F-3340 3347-3348 334F-3350 3357-3358 335E-335F 336B "
                "3372 337B 3381 3389 3391 3398-3399 33A0-33A1 33A8 33B0 33B8 "
                "33BF-33C0 33C7-33C8 33CF-33D0 33D7-33D8 33DE-33DF 33EB 33F2 33FB 3401 3409 "
                "3411 3418-3419 3420-3421 3428 3430 3438 343F-3440 3447-3448 344F-3450 3457-3458 "
                "345E-345F 346B 3472 347B 3481 3489 3491 3498-3499 34A0-34A1 34A8 "
                "34B0 34B8 34BF-34C0 34C7-34C8 34CF-34D0 34D7-34D8 34DE-34DF 34EB 34F2 34FB "
                "3501 3509 3511 3518-3519 3520-3521 3528 3530 3538 353F-3540 3547-3548 "
                "354F-3550 3557-3558 355E-355F 356B 3572 357B 3581 3589 3591 3598-3599 "
                "35A0-35A1 35A8 35B0 35B8 35BF-35C0 35C7-35C8 35CF-35D0 35D7-35D8 35DE-35DF 35EB "
                "35F2 35FB 3601 3609 3611 3618-3619 3620-3621 3628 3630 3638 "
                "363F-3640 3647-3648 364F-3650 3657-3658 365E-365F 366B 3672 367B 3681 3689 "
                "3691 3698-3699 36A0-36A1 36A8 36B0 36B8 36BF-36C0 36C7-36C8 36CF-36D0 36D7-36D8 "
                "36DE-36DF 36EB 36F2 36FB 3701 3709 3711 3718-3719 3720-3721 3728 "
                "3730 3738 373F-3740 3747-3748 374F-3750 3757-3758 375E-375F 376B 3772 377B "
                "3781 3789 3791 3798-3799 37A0-37A1 37A8 37B0 37B8 37BF-37C0 37C7-37C8 "
                "37CF-37D0 37D7-37D8 37DE-37DF 37EB 37F2 37FB 3801 3809 3811 3818-3819 "
                "3820-3821 3828 3830 3838 383F-3840 3847-3848 384F-3850 3857-3858 385E-385F 386B "
                "3872 387B 3881 3889 3891 3898-3899 38A0-38A1 38A8 38B0 38B8 "
                "38BF-38C0 38C7-38C8 38CF-38D0 38D7-38D8 38DE-38DF 38EB 38F2 38FB 3901 3909 "
                "3911 3918-3919 3920-3921 3928 3930 3938 393F-3940 3947-3948 394F-3950 3957 "
                "395E-395F 396B 3972 397B 3981 3989 3991 3998-3999 39A0-39A1 39A8 "
                "39B0 39B8 39BF-39C0 39C7-39C8 39CF-39D0 39D7 39DE-39DF 39EB 39F2 39FB "
                "3A01 3A09 3A11 3A18-3A19 3A20-3A21 3A28 3A30 3A38 3A3F-3A40 3A47-3A48 "
                "3A4F-3A50 3A57 3A5E-3A5F 3A6B 3A72 3A7B 3A81 3A89 3A91 3A98-3A99 "
                "3AA0 3AA8 3AB0 3AB8 3ABF-3AC0 3AC7-3AC8 3ACF-3AD0 3AD7 3ADF 3AEB "
                "3AF2 3AFB 3B01 3B09 3B11 3B18-3B19 3B20 3B28 3B30 3B38 "
                "3B3F-3B40 3B47-3B48 3B4F-3B50 3B57 3B5F 3B6B 3B72 3B7B 3B81 3B89 "
                "3B91 3B98-3B99 3BA0 3BA8 3BB0 3BB8 3BBF-3BC0 3BC7-3BC8 3BCF 3BD7 "
                "3BDF 3BEB 3BF2 3C01 3C09 3C11 3C18 3C20 3C28 3C30 "
                "3C38 3C3F-3C40 3C47 3C4F 3C57 3C5F 3C6B 3C72 3C81 3C89 "
                "3CA0 3CA8 3CB0 3CB8 3CC7 3CCF 3CD7 3CDF 3D01 3D09 "
                "3EE0 3EEC 3F0C 3F17 3F20 3F24 3F28 3F2F 3F44 3F4C "
                "3F53 3F5A 3FD7 3FE0 3FEA-3FEC 3FFA-3FFE BD48 BD74 BD82 BD8A "
                "BD92 BD9A BDA2 BDB3 BDBB BDC3-BDC4 BDCC BDD4-BDD5 BDDD BDEA "
                "BDF3 BDFC BE07 BE10 BE19-BE1A BE23-BE24 BE2D-BE2E BE37-BE38 BE42-BE44 BE4E-BE4F "
                "BE5A-BE5C BE6F BE7E BE80 BE93-BE94 BF7D BF86 BF8A BF8D BF90 "
                "BF95 BF9C BF9E BFA7 BFAC BFAF BFB2 BFB5 BFC5"
            ),
            1: (
                "3D48 3D74 3D82 3D8A 3D92 3D9A 3DA2 3DB3 3DBB 3DC3-3DC4 "
                "3DCC 3DD4-3DD5 3DDD 3DEA 3DF3 3DFC 3E07 3E10 3E19-3E1A 3E23-3E24 "
                "3E2D-3E2E 3E37-3E38 3E42-3E44 3E4E-3E4F 3E5A-3E5C 3E6F 3E7E 3E80 3E93-3E94 3F7D "
                "3F86 3F8A 3F8D 3F90 3F95 3F9C 3F9E 3FA7 3FAC 3FAF "
                "3FB2 3FB5 3FC5 8081 8089 8091 8098-8099 80A0-80A1 80A8 80B0 "
                "80B8 80BF-80C0 80C7-80C8 80CF-80D0 80D7-80D8 80DE-80DF 80EB 80F2 80FB 8101 "
                "8109 8111 8118-8119 8120-8121 8128 8130 8138 813F-8140 8147-8148 814F-8150 "
                "8157-8158 815E-815F 816B 8172 817B 8181 8189 8191 8198-8199 81A0-81A1 "
                "81A8 81B0 81B8 81BF-81C0 81C7-81C8 81CF-81D0 81D7-81D8 81DE-81DF 81EB 81F2 "
                "81FB 8201 8209 8211 8218-8219 8220-8221 8228 8230 8238 823F-8240 "
                "8247-8248 824F-8250 8257-8258 825E-825F 826B 8272 827B 8281 8289 8291 "
                "8298-8299 82A0-82A1 82A8 82B0 82B8 82BF-82C0 82C7-82C8 82CF-82D0 82D7-82D8 82DE-82DF "
                "82EB 82F2 82FB 8301 8309 8311 8318-8319 8320-8321 8328 8330 "
                "8338 833F-8340 8347-8348 834F-8350 8357-8358 835E-835F 836B 8372 837B 8381 "
                "8389 8391 8398-8399 83A0-83A1 83A8 83B0 83B8 83BF-83C0 83C7-83C8 83CF-83D0 "
                "83D7-83D8 83DE-83DF 83EB 83F2 83FB 8401 8409 8411 8418-8419 8420-8421 "
                "8428 8430 8438 843F-8440 8447-8448 844F-8450 8457-8458 845E-845F 846B 8472 "
                "847B 8481 8489 8491 8498-8499 84A0-84A1 84A8 84B0 84B8 84BF-84C0 "
                "84C7-84C8 84CF-84D0 84D7-84D8 84DE-84DF 84EB 84F2 84FB 8501 8509 8511 "
                "8518-8519 8520-8521 8528 8530 8538 853F-8540 8547-8548 854F-8550 8557-8558 855E-855F "
                "856B 8572 857B 8581 8589 8591 8598-8599 85A0-85A1 85A8 85B0 "
                "85B8 85BF-85C0 85C7-85C8 85CF-85D0 85D7-85D8 85DE-85DF 85EB 85F2 85FB 8601 "
                "8609 8611 8618-8619 8620-8621 8628 8630 8638 863F-8640 8647-8648 864F-8650 "
                "8657-8658 865E-865F 866B 8672 867B 8681 8689 8691 8698-8699 86A0-86A1 "
                "86A8 86B0 86B8 86BF-86C0 86C7-86C8 86CF-86D0 86D7-86D8 86DE-86DF 86EB 86F2 "
                "86FB 8701 8709 8711 8718-8719 8720-8721 8728 8730 8738 873F-8740 "
                "8747-8748 874F-8750 8757-8758 875E-875F 876B 8772 877B 8781 8789 8791 "
                "8798-8799 87A0-87A1 87A8 87B0 87B8 87BF-87C0 87C7-87C8 87CF-87D0 87D7-87D8 87DE-87DF "
                "87EB 87F2 87FB 8801 8809 8811 8818-8819 8820-8821 8828 8830 "
                "8838 883F-8840 8847-8848 884F-8850 8857-8858 885E-885F 886B 8872 887B 8881 "
                "8889 8891 8898-8899 88A0-88A1 88A8 88B0 88B8 88BF-88C0 88C7-88C8 88CF-88D0 "
                "88D7-88D8 88DE-88DF 88EB 88F2 88FB 8901 8909 8911 8918-8919 8920-8921 "
                "8928 8930 8938 893F-8940 8947-8948 894F-8950 8957-8958 895E-895F 896B 8972 "
                "897B 8981 8989 8991 8998-8999 89A0-89A1 89A8 89B0 89B8 89BF-89C0 "
                "89C7-89C8 89CF-89D0 89D7-89D8 89DE-89DF 89EB 89F2 89FB 8A01 8A09 8A11 "
                "8A18-8A19 8A20-8A21 8A28 8A30 8A38 8A3F-8A40 8A47-8A48 8A4F-8A50 8A57-8A58 8A5E-8A5F "
                "8A6B 8A72 8A7B 8A81 8A89 8A91 8A98-8A99 8AA0-8AA1 8AA8 8AB0 "
                "8AB8 8ABF-8AC0 8AC7-8AC8 8ACF-8AD0 8AD7-8AD8 8ADE-8ADF 8AEB 8AF2 8AFB 8B01 "
                "8B09 8B11 8B18-8B19 8B20-8B21 8B28 8B30 8B38 8B3F-8B40 8B47-8B48 8B4F-8B50 "
                "8B57-8B58 8B5E-8B5F 8B6B 8B72 8B7B 8B81 8B89 8B91 8B98-8B99 8BA0-8BA1 "
                "8BA8 8BB0 8BB8 8BBF-8BC0 8BC7-8BC8 8BCF-8BD0 8BD7-8BD8 8BDE-8BDF 8BEB 8BF2 "
                "8BFB 8C01 8C09 8C11 8C18-8C19 8C20-8C21 8C28 8C30 8C38 8C3F-8C40 "
                "8C47-8C48 8C4F-8C50 8C57-8C58 8C5E-8C5F 8C6B 8C72 8C7B 8C81 8C89 8C91 "
                "8C98-8C99 8CA0-8CA1 8CA8 8CB0 8CB8 8CBF-8CC0 8CC7-8CC8 8CCF-8CD0 8CD7-8CD8 8CDE-8CDF "
                "8CEB 8CF2 8CFB 8D01 8D09 8D11 8D18-8D19 8D20-8D21 8D28 8D30 "
                "8D38 8D3F-8D40 8D47-8D48 8D4F-8D50 8D57-8D58 8D5E-8D5F 8D6B 8D72 8D7B 8D81 "
                "8D89 8D91 8D98-8D99 8DA0-8DA1 8DA8 8DB0 8DB8 8DBF-8DC0 8DC7-8DC8 8DCF-8DD0 "
                "8DD7-8DD8 8DDE-8DDF 8DEB 8DF2 8DFB 8E01 8E09 8E11 8E18-8E19 8E20-8E21 "
                "8E28 8E30 8E38 8E3F-8E40 8E47-8E48 8E4F-8E50 8E57-8E58 8E5E-8E5F 8E6B 8E72 "
                "8E7B 8E81 8E89 8E91 8E98-8E99 8EA0-8EA1 8EA8 8EB0 8EB8 8EBF-8EC0 "
                "8EC7-8EC8 8ECF-8ED0 8ED7-8ED8 8EDE-8EDF 8EEB 8EF2 8EFB 8F01 8F09 8F11 "
                "8F18-8F19 8F20-8F21 8F28 8F30 8F38 8F3F-8F40 8F47-8F48 8F4F-8F50 8F57-8F58 8F5E-8F5F "
                "8F6B 8F72 8F7B 8F81 8F89 8F91 8F98-8F99 8FA0-8FA1 8FA8 8FB0 "
                "8FB8 8FBF-8FC0 8FC7-8FC8 8FCF-8FD0 8FD7-8FD8 8FDE-8FDF 8FEB 8FF2 8FFB 9001 "
                "9009 9011 9018-9019 9020-9021 9028 9030 9038 903F-9040 9047-9048 904F-9050 "
                "9057-9058 905E-905F 906B 9072 907B 9081 9089 9091 9098-9099 90A0-90A1 "
                "90A8 90B0 90B8 90BF-90C0 90C7-90C8 90CF-90D0 90D7-90D8 90DE-90DF 90EB 90F2 "
                "90FB 9101 9109 9111 9118-9119 9120-9121 9128 9130 9138 913F-9140 "
                "9147-9148 914F-9150 9157-9158 915E-915F 916B 9172 917B 9181 9189 9191 "
                "9198-9199 91A0-91A1 91A8 91B0 91B8 91BF-91C0 91C7-91C8 91CF-91D0 91D7-91D8 91DE-91DF "
                "91EB 91F2 91FB 9201 9209 9211 9218-9219 9220-9221 9228 9230 "
                "9238 923F-9240 9247-9248 924F-9250 9257-9258 925E-925F 926B 9272 927B 9281 "
                "9289 9291 9298-9299 92A0-92A1 92A8 92B0 92B8 92BF-92C0 92C7-92C8 92CF-92D0 "
                "92D7-92D8 92DE-92DF 92EB 92F2 92FB 9301 9309 9311 9318-9319 9320-9321 "
                "9328 9330 9338 933F-9340 9347-9348 934F-9350 9357-9358 935E-935F 936B 9372 "
                "937B 9381 9389 9391 9398-9399 93A0-93A1 93A8 93B0 93B8 93BF-93C0 "
                "93C7-93C8 93CF-93D0 93D7-93D8 93DE-93DF 93EB 93F2 93FB 9401 9409 9411 "
                "9418-9419 9420-9421 9428 9430 9438 943F-9440 9447-9448 944F-9450 9457-9458 945E-945F "
                "946B 9472 947B 9481 9489 9491 9498-9499 94A0-94A1 94A8 94B0 "
                "94B8 94BF-94C0 94C7-94C8 94CF-94D0 94D7-94D8 94DE-94DF 94EB 94F2 94FB 9501 "
                "9509 9511 9518-9519 9520-9521 9528 9530 9538 953F-9540 9547-9548 954F-9550 "
                "9557-9558 955E-955F 956B 9572 957B 9581 9589 9591 9598-9599 95A0-95A1 "
                "95A8 95B0 95B8 95BF-95C0 95C7-95C8 95CF-95D0 95D7-95D8 95DE-95DF 95EB 95F2 "
                "95FB 9601 9609 9611 9618-9619 9620-9621 9628 9630 9638 963F-9640 "
                "9647-9648 964F-9650 9657-9658 965E-965F 966B 9672 967B 9681 9689 9691 "
                "9698-9699 96A0-96A1 96A8 96B0 96B8 96BF-96C0 96C7-96C8 96CF-96D0 96D7-96D8 96DE-96DF "
                "96EB 96F2 96FB 9701 9709 9711 9718-9719 9720-9721 9728 9730 "
                "9738 973F-9740 9747-9748 974F-9750 9757-9758 975E-975F 976B 9772 977B 9781 "
                "9789 9791 9798-9799 97A0-97A1 97A8 97B0 97B8 97BF-97C0 97C7-97C8 97CF-97D0 "
                "97D7-97D8 97DE-97DF 97EB 97F2 97FB 9801 9809 9811 9818-9819 9820-9821 "
                "9828 9830 9838 983F-9840 9847-9848 984F-9850 9857-9858 985E-985F 986B 9872 "
                "987B 9881 9889 9891 9898-9899 98A0-98A1 98A8 98B0 98B8 98BF-98C0 "
                "98C7-98C8 98CF-98D0 98D7-98D8 98DE-98DF 98EB 98F2 98FB 9901 9909 9911 "
                "9918-9919 9920-9921 9928 9930 9938 993F-9940 9947-9948 994F-9950 9957-9958 995E-995F "
                "996B 9972 997B 9981 9989 9991 9998-9999 99A0-99A1 99A8 99B0 "
                "99B8 99BF-99C0 99C7-99C8 99CF-99D0 99D7-99D8 99DE-99DF 99EB 99F2 99FB 9A01 "
                "9A09 9A11 9A18-9A19 9A20-9A21 9A28 9A30 9A38 9A3F-9A40 9A47-9A48 9A4F-9A50 "
                "9A57-9A58 9A5E-9A5F 9A6B 9A72 9A7B 9A81 9A89 9A91 9A98-9A99 9AA0-9AA1 "
                "9AA8 9AB0 9AB8 9ABF-9AC0 9AC7-9AC8 9ACF-9AD0 9AD7-9AD8 9ADE-9ADF 9AEB 9AF2 "
                "9AFB 9B01 9B09 9B11 9B18-9B19 9B20-9B21 9B28 9B30 9B38 9B3F-9B40 "
                "9B47-9B48 9B4F-9B50 9B57-9B58 9B5E-9B5F 9B6B 9B72 9B7B 9B81 9B89 9B91 "
                "9B98-9B99 9BA0-9BA1 9BA8 9BB0 9BB8 9BBF-9BC0 9BC7-9BC8 9BCF-9BD0 9BD7-9BD8 9BDE-9BDF "
                "9BEB 9BF2 9BFB 9C01 9C09 9C11 9C18-9C19 9C20-9C21 9C28 9C30 "
                "9C38 9C3F-9C40 9C47-9C48 9C4F-9C50 9C57-9C58 9C5E-9C5F 9C6B 9C72 9C7B 9C81 "
                "9C89 9C91 9C98-9C99 9CA0-9CA1 9CA8 9CB0 9CB8 9CBF-9CC0 9CC7-9CC8 9CCF-9CD0 "
                "9CD7-9CD8 9CDE-9CDF 9CEB 9CF2 9CFB 9D01 9D09 9D11 9D18-9D19 9D20-9D21 "
                "9D28 9D30 9D38 9D3F-9D40 9D47-9D48 9D4F-9D50 9D57-9D58 9D5E-9D5F 9D6B 9D72 "
                "9D7B 9D81 9D89 9D91 9D98-9D99 9DA0-9DA1 9DA8 9DB0 9DB8 9DBF-9DC0 "
                "9DC7-9DC8 9DCF-9DD0 9DD7-9DD8 9DDE-9DDF 9DEB 9DF2 9DFB 9E01 9E09 9E11 "
                "9E18-9E19 9E20-9E21 9E28 9E30 9E38 9E3F-9E40 9E47-9E48 9E4F-9E50 9E57-9E58 9E5E-9E5F "
                "9E6B 9E72 9E7B 9E81 9E89 9E91 9E98-9E99 9EA0-9EA1 9EA8 9EB0 "
                "9EB8 9EBF-9EC0 9EC7-9EC8 9ECF-9ED0 9ED7-9ED8 9EDE-9EDF 9EEB 9EF2 9EFB 9F01 "
                "9F09 9F11 9F18-9F19 9F20-9F21 9F28 9F30 9F38 9F3F-9F40 9F47-9F48 9F4F-9F50 "
                "9F57-9F58 9F5E-9F5F 9F6B 9F72 9F7B 9F81 9F89 9F91 9F98-9F99 9FA0-9FA1 "
                "9FA8 9FB0 9FB8 9FBF-9FC0 9FC7-9FC8 9FCF-9FD0 9FD7-9FD8 9FDE-9FDF 9FEB 9FF2 "
                "9FFB A001 A009 A011 A018-A019 A020-A021 A028 A030 A038 A03F-A040 "
                "A047-A048 A04F-A050 A057-A058 A05E-A05F A06B A072 A07B A081 A089 A091 "
                "A098-A099 A0A0-A0A1 A0A8 A0B0 A0B8 A0BF-A0C0 A0C7-A0C8 A0CF-A0D0 A0D7-A0D8 A0DE-A0DF "
                "A0EB A0F2 A0FB A101 A109 A111 A118-A119 A120-A121 A128 A130 "
                "A138 A13F-A140 A147-A148 A14F-A150 A157-A158 A15E-A15F A16B A172 A17B A181 "
                "A189 A191 A198-A199 A1A0-A1A1 A1A8 A1B0 A1B8 A1BF-A1C0 A1C7-A1C8 A1CF-A1D0 "
                "A1D7-A1D8 A1DE-A1DF A1EB A1F2 A1FB A201 A209 A211 A218-A219 A220-A221 "
                "A228 A230 A238 A23F-A240 A247-A248 A24F-A250 A257-A258 A25E-A25F A26B A272 "
                "A27B A281 A289 A291 A298-A299 A2A0-A2A1 A2A8 A2B0 A2B8 A2BF-A2C0 "
                "A2C7-A2C8 A2CF-A2D0 A2D7-A2D8 A2DE-A2DF A2EB A2F2 A2FB A301 A309 A311 "
                "A318-A319 A320-A321 A328 A330 A338 A33F-A340 A347-A348 A34F-A350 A357-A358 A35E-A35F "
                "A36B A372 A37B A381 A389 A391 A398-A399 A3A0-A3A1 A3A8 A3B0 "
                "A3B8 A3BF-A3C0 A3C7-A3C8 A3CF-A3D0 A3D7-A3D8 A3DE-A3DF A3EB A3F2 A3FB A401 "
                "A409 A411 A418-A419 A420-A421 A428 A430 A438 A43F-A440 A447-A448 A44F-A450 "
                "A457-A458 A45E-A45F A46B A472 A47B A481 A489 A491 A498-A499 A4A0-A4A1 "
                "A4A8 A4B0 A4B8 A4BF-A4C0 A4C7-A4C8 A4CF-A4D0 A4D7-A4D8 A4DE-A4DF A4EB A4F2 "
                "A4FB A501 A509 A511 A518-A519 A520-A521 A528 A530 A538 A53F-A540 "
                "A547-A548 A54F-A550 A557-A558 A55E-A55F A56B A572 A57B A581 A589 A591 "
                "A598-A599 A5A0-A5A1 A5A8 A5B0 A5B8 A5BF-A5C0 A5C7-A5C8 A5CF-A5D0 A5D7-A5D8 A5DE-A5DF "
                "A5EB A5F2 A5FB A601 A609 A611 A618-A619 A620-A621 A628 A630 "
                "A638 A63F-A640 A647-A648 A64F-A650 A657-A658 A65E-A65F A66B A672 A67B A681 "
                "A689 A691 A698-A699 A6A0-A6A1 A6A8 A6B0 A6B8 A6BF-A6C0 A6C7-A6C8 A6CF-A6D0 "
                "A6D7-A6D8 A6DE-A6DF A6EB A6F2 A6FB A701 A709 A711 A718-A719 A720-A721 "
                "A728 A730 A738 A73F-A740 A747-A748 A74F-A750 A757-A758 A75E-A75F A76B A772 "
                "A77B A781 A789 A791 A798-A799 A7A0-A7A1 A7A8 A7B0 A7B8 A7BF-A7C0 "
                "A7C7-A7C8 A7CF-A7D0 A7D7-A7D8 A7DE-A7DF A7EB A7F2 A7FB A801 A809 A811 "
                "A818-A819 A820-A821 A828 A830 A838 A83F-A840 A847-A848 A84F-A850 A857-A858 A85E-A85F "
                "A86B A872 A87B A881 A889 A891 A898-A899 A8A0-A8A1 A8A8 A8B0 "
                "A8B8 A8BF-A8C0 A8C7-A8C8 A8CF-A8D0 A8D7-A8D8 A8DE-A8DF A8EB A8F2 A8FB A901 "
                "A909 A911 A918-A919 A920-A921 A928 A930 A938 A93F-A940 A947-A948 A94F-A950 "
                "A957-A958 A95E-A95F A96B A972 A97B A981 A989 A991 A998-A999 A9A0-A9A1 "
                "A9A8 A9B0 A9B8 A9BF-A9C0 A9C7-A9C8 A9CF-A9D0 A9D7-A9D8 A9DE-A9DF A9EB A9F2 "
                "A9FB AA01 AA09 AA11 AA18-AA19 AA20-AA21 AA28 AA30 AA38 AA3F-AA40 "
                "AA47-AA48 AA4F-AA50 AA57-AA58 AA5E-AA5F AA6B AA72 AA7B AA81 AA89 AA91 "
                "AA98-AA99 AAA0-AAA1 AAA8 AAB0 AAB8 AABF-AAC0 AAC7-AAC8 AACF-AAD0 AAD7-AAD8 AADE-AADF "
                "AAEB AAF2 AAFB AB01 AB09 AB11 AB18-AB19 AB20-AB21 AB28 AB30 "
                "AB38 AB3F-AB40 AB47-AB48 AB4F-AB50 AB57-AB58 AB5E-AB5F AB6B AB72 AB7B AB81 "
                "AB89 AB91 AB98-AB99 ABA0-ABA1 ABA8 ABB0 ABB8 ABBF-ABC0 ABC7-ABC8 ABCF-ABD0 "
                "ABD7-ABD8 ABDE-ABDF ABEB ABF2 ABFB AC01 AC09 AC11 AC18-AC19 AC20-AC21 "
                "AC28 AC30 AC38 AC3F-AC40 AC47-AC48 AC4F-AC50 AC57-AC58 AC5E-AC5F AC6B AC72 "
                "AC7B AC81 AC89 AC91 AC98-AC99 ACA0-ACA1 ACA8 ACB0 ACB8 ACBF-ACC0 "
                "ACC7-ACC8 ACCF-ACD0 ACD7-ACD8 ACDE-ACDF ACEB ACF2 ACFB AD01 AD09 AD11 "
                "AD18-AD19 AD20-AD21 AD28 AD30 AD38 AD3F-AD40 AD47-AD48 AD4F-AD50 AD57-AD58 AD5E-AD5F "
                "AD6B AD72 AD7B AD81 AD89 AD91 AD98-AD99 ADA0-ADA1 ADA8 ADB0 "
                "ADB8 ADBF-ADC0 ADC7-ADC8 ADCF-ADD0 ADD7-ADD8 ADDE-ADDF ADEB ADF2 ADFB AE01 "
                "AE09 AE11 AE18-AE19 AE20-AE21 AE28 AE30 AE38 AE3F-AE40 AE47-AE48 AE4F-AE50 "
                "AE57-AE58 AE5E-AE5F AE6B AE72 AE7B AE81 AE89 AE91 AE98-AE99 AEA0-AEA1 "
                "AEA8 AEB0 AEB8 AEBF-AEC0 AEC7-AEC8 AECF-AED0 AED7-AED8 AEDE-AEDF AEEB AEF2 "
                "AEFB AF01 AF09 AF11 AF18-AF19 AF20-AF21 AF28 AF30 AF38 AF3F-AF40 "
                "AF47-AF48 AF4F-AF50 AF57-AF58 AF5E-AF5F AF6B AF72 AF7B AF81 AF89 AF91 "
                "AF98-AF99 AFA0-AFA1 AFA8 AFB0 AFB8 AFBF-AFC0 AFC7-AFC8 AFCF-AFD0 AFD7-AFD8 AFDE-AFDF "
                "AFEB AFF2 AFFB B001 B009 B011 B018-B019 B020-B021 B028 B030 "
                "B038 B03F-B040 B047-B048 B04F-B050 B057-B058 B05E-B05F B06B B072 B07B B081 "
                "B089 B091 B098-B099 B0A0-B0A1 B0A8 B0B0 B0B8 B0BF-B0C0 B0C7-B0C8 B0CF-B0D0 "
                "B0D7-B0D8 B0DE-B0DF B0EB B0F2 B0FB B101 B109 B111 B118-B119 B120-B121 "
                "B128 B130 B138 B13F-B140 B147-B148 B14F-B150 B157-B158 B15E-B15F B16B B172 "
                "B17B B181 B189 B191 B198-B199 B1A0-B1A1 B1A8 B1B0 B1B8 B1BF-B1C0 "
                "B1C7-B1C8 B1CF-B1D0 B1D7-B1D8 B1DE-B1DF B1EB B1F2 B1FB B201 B209 B211 "
                "B218-B219 B220-B221 B228 B230 B238 B23F-B240 B247-B248 B24F-B250 B257-B258 B25E-B25F "
                "B26B B272 B27B B281 B289 B291 B298-B299 B2A0-B2A1 B2A8 B2B0 "
                "B2B8 B2BF-B2C0 B2C7-B2C8 B2CF-B2D0 B2D7-B2D8 B2DE-B2DF B2EB B2F2 B2FB B301 "
                "B309 B311 B318-B319 B320-B321 B328 B330 B338 B33F-B340 B347-B348 B34F-B350 "
                "B357-B358 B35E-B35F B36B B372 B37B B381 B389 B391 B398-B399 B3A0-B3A1 "
                "B3A8 B3B0 B3B8 B3BF-B3C0 B3C7-B3C8 B3CF-B3D0 B3D7-B3D8 B3DE-B3DF B3EB B3F2 "
                "B3FB B401 B409 B411 B418-B419 B420-B421 B428 B430 B438 B43F-B440 "
                "B447-B448 B44F-B450 B457-B458 B45E-B45F B46B B472 B47B B481 B489 B491 "
                "B498-B499 B4A0-B4A1 B4A8 B4B0 B4B8 B4BF-B4C0 B4C7-B4C8 B4CF-B4D0 B4D7-B4D8 B4DE-B4DF "
                "B4EB B4F2 B4FB B501 B509 B511 B518-B519 B520-B521 B528 B530 "
                "B538 B53F-B540 B547-B548 B54F-B550 B557-B558 B55E-B55F B56B B572 B57B B581 "
                "B589 B591 B598-B599 B5A0-B5A1 B5A8 B5B0 B5B8 B5BF-B5C0 B5C7-B5C8 B5CF-B5D0 "
                "B5D7-B5D8 B5DE-B5DF B5EB B5F2 B5FB B601 B609 B611 B618-B619 B620-B621 "
                "B628 B630 B638 B63F-B640 B647-B648 B64F-B650 B657-B658 B65E-B65F B66B B672 "
                "B67B B681 B689 B691 B698-B699 B6A0-B6A1 B6A8 B6B0 B6B8 B6BF-B6C0 "
                "B6C7-B6C8 B6CF-B6D0 B6D7-B6D8 B6DE-B6DF B6EB B6F2 B6FB B701 B709 B711 "
                "B718-B719 B720-B721 B728 B730 B738 B73F-B740 B747-B748 B74F-B750 B757-B758 B75E-B75F "
                "B76B B772 B77B B781 B789 B791 B798-B799 B7A0-B7A1 B7A8 B7B0 "
                "B7B8 B7BF-B7C0 B7C7-B7C8 B7CF-B7D0 B7D7-B7D8 B7DE-B7DF B7EB B7F2 B7FB B801 "
                "B809 B811 B818-B819 B820-B821 B828 B830 B838 B83F-B840 B847-B848 B84F-B850 "
                "B857-B858 B85E-B85F B86B B872 B87B B881 B889 B891 B898-B899 B8A0-B8A1 "
                "B8A8 B8B0 B8B8 B8BF-B8C0 B8C7-B8C8 B8CF-B8D0 B8D7-B8D8 B8DE-B8DF B8EB B8F2 "
                "B8FB B901 B909 B911 B918-B919 B920-B921 B928 B930 B938 B93F-B940 "
                "B947-B948 B94F-B950 B957 B95E-B95F B96B B972 B97B B981 B989 B991 "
                "B998-B999 B9A0-B9A1 B9A8 B9B0 B9B8 B9BF-B9C0 B9C7-B9C8 B9CF-B9D0 B9D7 B9DE-B9DF "
                "B9EB B9F2 B9FB BA01 BA09 BA11 BA18-BA19 BA20-BA21 BA28 BA30 "
                "BA38 BA3F-BA40 BA47-BA48 BA4F-BA50 BA57 BA5E-BA5F BA6B BA72 BA7B BA81 "
                "BA89 BA91 BA98-BA99 BAA0 BAA8 BAB0 BAB8 BABF-BAC0 BAC7-BAC8 BACF-BAD0 "
                "BAD7 BADF BAEB BAF2 BAFB BB01 BB09 BB11 BB18-BB19 BB20 "
                "BB28 BB30 BB38 BB3F-BB40 BB47-BB48 BB4F-BB50 BB57 BB5F BB6B BB72 "
                "BB7B BB81 BB89 BB91 BB98-BB99 BBA0 BBA8 BBB0 BBB8 BBBF-BBC0 "
                "BBC7-BBC8 BBCF BBD7 BBDF BBEB BBF2 BC01 BC09 BC11 BC18 "
                "BC20 BC28 BC30 BC38 BC3F-BC40 BC47 BC4F BC57 BC5F BC6B "
                "BC72 BC81 BC89 BCA0 BCA8 BCB0 BCB8 BCC7 BCCF BCD7 "
                "BCDF BD01 BD09 BEE0 BEEC BF0C BF17 BF20 BF24 BF28 "
                "BF2F BF44 BF4C BF53 BF5A BFD7 BFE0 BFEA-BFEC BFFA-BFFE"
            ),
        },
    }
    bits = torch.arange(1 << 16, dtype=torch.int32)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 256, 256)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    def ordered(t):
        # BF16 words in value order, so that adjacent finite values differ by one step.
        words = t.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        return torch.where(words < 0, -(words & 0x7FFF), words)

    actual = stored(ttnn.to_torch(ttnn.erf(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (torch.special.erf(x)).to(torch.float64).flatten()
    expected = stored(reference)
    finite = torch.isfinite(expected) & torch.isfinite(actual)
    ok = (actual == expected) | (finite & ((ordered(actual) - ordered(expected)).abs() <= STEPS))
    # A NaN result reaches L1 as a NaN or as an infinity of either sign.
    ok |= torch.isnan(reference) & (torch.isnan(actual) | torch.isinf(actual))
    # The SFPU flushes a result below the smallest normal, or within 2^-12 above it, to 0.
    ok |= (actual == 0) & (reference.abs() < tiny * (1 + 2.0**-12))
    listed = torch.zeros_like(ok)
    for word, ranges in STOCK.get(board, {}).items():
        value = stored(torch.tensor([word], dtype=torch.int32).to(torch.int16).view(torch.bfloat16))
        for first, last in ranges:
            lanes = (bits >= first) & (bits <= last)
            listed |= lanes
            ok |= lanes & ((actual == value) | (torch.isnan(actual) & torch.isnan(value)))
    for first, last in SKIP:
        listed |= (bits >= first) & (bits <= last)
        ok |= (bits >= first) & (bits <= last)
    shift = torch.zeros_like(bits)
    for offset, spans in MODEL.get(board, {}).items():
        for span in spans.split():
            first, _, last = span.partition("-")
            shift[(bits >= int(first, 16)) & (bits <= int(last or first, 16))] = offset
    moved = ordered(expected) + shift
    word = torch.where(moved < 0, (-moved) | 0x8000, moved)
    held = stored(word.to(torch.int16).view(torch.bfloat16))
    normal = ((bits & 0x7F80) != 0) & ((bits & 0x7F80) != 0x7F80)
    ok = torch.where(normal & ~listed & torch.isfinite(expected), actual == held, ok)
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )

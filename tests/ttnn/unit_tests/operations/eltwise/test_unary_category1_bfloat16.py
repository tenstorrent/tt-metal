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
# Logical NOT (-0.0) returns False while golden returns True
# ─────────────────────────────────────────────────────────────────────────────


def test_logical_not_ops(device):
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)
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
def test_mish_with_sqrt_in_sfpu_chain(device, generated_first):
    """mish in one SFPU chain with stock sqrt, both orders.

    Each chain step runs the kernel its own TT-NN op runs, and the second op stores exactly what it
    stores standalone on the first op's device result, which the chain keeps in DEST.
    """
    generated = (ttnn.UnaryWithParam(ttnn.UnaryOpType.MISH, 0.0), lambda tensor: ttnn.mish(tensor))
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


def test_mish_bf16_unflushed_zero_and_subnormal_inputs(device):
    """TT-NN's copy into DEST keeps -0 and subnormal BF16 inputs, which the LLK sweep's unpack delivers as +0.

    Each such input and +0 must store torch's result: its class as the BF16 pack stores it (NaN as
    +inf, a subnormal result as 0) and a finite result within 1 ULP of torch's float64 value.
    The pack writes a zero result as +0, so ±0 and both signs of subnormal store the word 0x0000.
    STOCK lists, per board, the inputs where the kernel stores its stock kernel's word instead,
    as {word: [(first input, last input), ...]} over the input bits.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.mish keeps its stock kernel on {board}")
    STOCK = {}
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        t = torch.where(torch.isnan(t), torch.full_like(t, float("inf")), t)
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    device_out = ttnn.to_torch(ttnn.mish(tt_x))
    raw = device_out.reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    bits = words.repeat(4).to(torch.int32) & 0xFFFF
    assert torch.all(raw[bits == 0x0000] == 0x0000)
    assert torch.all(raw[bits == 0x8000] == 0x0000)
    assert torch.all(raw[(bits > 0) & (bits < 0x80)] == 0x0000)
    assert torch.all(raw[(bits > 0x8000) & (bits < 0x8080)] == 0x0000)

    actual = stored(device_out)
    x64 = x.flatten().to(torch.float64)
    reference = ttnn.get_golden_function(ttnn.mish)(x64).to(torch.float64).flatten()
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


def test_mish_bf16_bfloat8_b(device):
    """bfloat8_b in and out. The SFPU reads Bfp8_b data in DEST as BF16, so this runs the BF16 kernel, as the
    BF16 cases do. The result must correlate with torch's on the input as stored to PCC 0.999, as upstream's
    bfloat8_b unary cases require. Over every finite BF16 exponent, each output must also be within one bfloat8_b
    step of the BF16 call's for the same stored input, packed to bfloat8_b (two in a block that holds an infinity
    or NaN), and of its class where that is infinity or NaN: the op packs DEST to Bfp8_b, and the typecast
    rounds and sets a block's exponent its own way.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.mish keeps its stock kernel on {board}")
    torch.manual_seed(0)
    x = torch.empty(1, 1, 64, 64).uniform_(-10.0, 10.0).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.float64).flatten()
    output = ttnn.mish(tt_x)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float64).flatten()
    golden = ttnn.get_golden_function(ttnn.mish)(stored).to(torch.float64)
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
    actual = ttnn.to_torch(ttnn.mish(tt_x)).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.mish(tt_stored), ttnn.bfloat8_b)).to(torch.float32).flatten()
    # One step of a Bfp8_b block: its 16 values share the largest one's exponent and keep 7 bits under it.
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    magnitude = torch.where(finite, torch.maximum(actual.abs(), expected.abs()), torch.zeros_like(actual))
    block = magnitude.reshape(-1, 16).amax(dim=1, keepdim=True).expand(-1, 16).reshape(-1)
    step = torch.exp2(torch.floor(torch.log2(block.clamp_min(torch.finfo(torch.float32).tiny))) - 6)
    # A block that also holds an infinity or NaN: the op's pack and the typecast set its exponent apart, and
    # its finite values land up to two steps apart (polygamma's overflow binade, BH job 142644).
    special = (~finite).reshape(-1, 16).any(dim=1, keepdim=True).expand(-1, 16).reshape(-1)
    step = torch.where(special, 2 * step, step)
    # Infinity and NaN keep their class exactly; a finite value, zero included, may round one step apart (a value
    # under one step of its block packs to zero on one side and not the other: gelu at -8.625, BH job 142674).
    same = (
        (actual == expected)
        | (torch.isnan(actual) & torch.isnan(expected))
        | (finite & ((actual - expected).abs() <= step))
    )
    assert same.all(), (
        f"{int((~same).sum())} bfloat8_b outputs are more than one bfloat8_b step from the BF16 call's on the same "
        "stored input; first at "
        f"{stored.flatten()[~same][0].item()!r}: {actual[~same][0].item()!r}, BF16 call {expected[~same][0].item()!r}"
    )


def test_mish_bf16_nonfinite_inputs(device):
    """+-Inf and +-NaN through TT-NN, where the kernel may return its stock kernel's word for some of them.

    Each must store torch's result as the BF16 pack stores it (NaN as an infinity), or the word STOCK
    lists for that input on the board, as {input bits: word}.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.mish keeps its stock kernel on {board}")
    STOCK = {"wormhole_b0": {0xFF80: 0xFF80, 0xFFC0: 0xFF80}}
    bits = torch.tensor([0x7F80, 0xFF80, 0x7FC0, 0xFFC0], dtype=torch.int32).repeat(256)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def stored(t):
        return t.to(torch.bfloat16).to(torch.float64).flatten()

    actual = stored(ttnn.to_torch(ttnn.mish(tt_x)))
    reference = ttnn.get_golden_function(ttnn.mish)(x.flatten().to(torch.float64)).to(torch.float64)
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


def test_mish_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs through ttnn.mish, the call TT-NN users make.

    Each must store torch's float64 result (torch.nn.functional.mish(x)) as the BF16 pack stores it (NaN as an
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
        pytest.skip(f"ttnn.mish keeps its stock kernel on {board}")
    STOCK = {"wormhole_b0": {0xFF80: [(0xFF80, 0xFFFF)]}}
    STEPS = 1
    SKIP = [(0x0003, 0x00D4), (0x8003, 0x80D4), (0xC2B8, 0xC2C0), (0xFF80, 0xFF80)]
    MODEL = {
        "blackhole": {
            -1: (
                "3DD0 3DD3 3DD6 3DD9 3DDC 3DDF 3DE2 3E0D 3E15 3E3A "
                "3E41 3F31 3F68 3FCA 80D6 80DB 80E0 80E5 80EA 80EF "
                "80F4 80F9 80FE 8104 8109 810E 8113 8118 811D 8121-8122 "
                "8126-8127 812B-812C 8130-8131 8135-8136 813A-813B 813F-8140 8144-8145 8149-814A 814E-814F 8153-8154 "
                "8156 815B 8160 8165 816A 816F 8174 8179 817E 8184 "
                "8189 818E 8193 8198 819D 81A1-81A2 81A6-81A7 81AB-81AC 81B0-81B1 81B5-81B6 "
                "81BA-81BB 81BF-81C0 81C4-81C5 81C9-81CA 81CE-81CF 81D3-81D4 81D6 81DB 81E0 81E5 "
                "81EA 81EF 81F4 81F9 81FE 8204 8209 820E 8213 8218 "
                "821D 8221-8222 8226-8227 822B-822C 8230-8231 8235-8236 823A-823B 823F-8240 8244-8245 8249-824A "
                "824E-824F 8253-8254 8256 825B 8260 8265 826A 826F 8274 8279 "
                "827E 8284 8289 828E 8293 8298 829D 82A1-82A2 82A6-82A7 82AB-82AC "
                "82B0-82B1 82B5-82B6 82BA-82BB 82BF-82C0 82C4-82C5 82C9-82CA 82CE-82CF 82D3-82D4 82D6 82DB "
                "82E0 82E5 82EA 82EF 82F4 82F9 82FE 8304 8309 830E "
                "8313 8318 831D 8321-8322 8326-8327 832B-832C 8330-8331 8335-8336 833A-833B 833F-8340 "
                "8344-8345 8349-834A 834E-834F 8353-8354 8356 835B 8360 8365 836A 836F "
                "8374 8379 837E 8384 8389 838E 8393 8398 839D 83A1-83A2 "
                "83A6-83A7 83AB-83AC 83B0-83B1 83B5-83B6 83BA-83BB 83BF-83C0 83C4-83C5 83C9-83CA 83CE-83CF 83D3-83D4 "
                "83D6 83DB 83E0 83E5 83EA 83EF 83F4 83F9 83FE 8404 "
                "8409 840E 8413 8418 841D 8421-8422 8426-8427 842B-842C 8430-8431 8435-8436 "
                "843A-843B 843F-8440 8444-8445 8449-844A 844E-844F 8453-8454 8456 845B 8460 8465 "
                "846A 846F 8474 8479 847E 8484 8489 848E 8493 8498 "
                "849D 84A1-84A2 84A6-84A7 84AB-84AC 84B0-84B1 84B5-84B6 84BA-84BB 84BF-84C0 84C4-84C5 84C9-84CA "
                "84CE-84CF 84D3-84D4 84D6 84DB 84E0 84E5 84EA 84EF 84F4 84F9 "
                "84FE 8504 8509 850E 8513 8518 851D 8521-8522 8526-8527 852B-852C "
                "8530-8531 8535-8536 853A-853B 853F-8540 8544-8545 8549-854A 854E-854F 8553-8554 8556 855B "
                "8560 8565 856A 856F 8574 8579 857E 8584 8589 858E "
                "8593 8598 859D 85A1-85A2 85A6-85A7 85AB-85AC 85B0-85B1 85B5-85B6 85BA-85BB 85BF-85C0 "
                "85C4-85C5 85C9-85CA 85CE-85CF 85D3-85D4 85D6 85DB 85E0 85E5 85EA 85EF "
                "85F4 85F9 85FE 8604 8609 860E 8613 8618 861D 8621-8622 "
                "8626-8627 862B-862C 8630-8631 8635-8636 863A-863B 863F-8640 8644-8645 8649-864A 864E-864F 8653-8654 "
                "8656 865B 8660 8665 866A 866F 8674 8679 867E 8684 "
                "8689 868E 8693 8698 869D 86A1-86A2 86A6-86A7 86AB-86AC 86B0-86B1 86B5-86B6 "
                "86BA-86BB 86BF-86C0 86C4-86C5 86C9-86CA 86CE-86CF 86D3-86D4 86D6 86DB 86E0 86E5 "
                "86EA 86EF 86F4 86F9 86FE 8704 8709 870E 8713 8718 "
                "871D 8721-8722 8726-8727 872B-872C 8730-8731 8735-8736 873A-873B 873F-8740 8744-8745 8749-874A "
                "874E-874F 8753-8754 8756 875B 8760 8765 876A 876F 8774 8779 "
                "877E 8784 8789 878E 8793 8798 879D 87A1-87A2 87A6-87A7 87AB-87AC "
                "87B0-87B1 87B5-87B6 87BA-87BB 87BF-87C0 87C4-87C5 87C9-87CA 87CE-87CF 87D3-87D4 87D6 87DB "
                "87E0 87E5 87EA 87EF 87F4 87F9 87FE 8804 8809 880E "
                "8813 8818 881D 8821-8822 8826-8827 882B-882C 8830-8831 8835-8836 883A-883B 883F-8840 "
                "8844-8845 8849-884A 884E-884F 8853-8854 8856 885B 8860 8865 886A 886F "
                "8874 8879 887E 8884 8889 888E 8893 8898 889D 88A1-88A2 "
                "88A6-88A7 88AB-88AC 88B0-88B1 88B5-88B6 88BA-88BB 88BF-88C0 88C4-88C5 88C9-88CA 88CE-88CF 88D3-88D4 "
                "88D6 88DB 88E0 88E5 88EA 88EF 88F4 88F9 88FE 8904 "
                "8909 890E 8913 8918 891D 8921-8922 8926-8927 892B-892C 8930-8931 8935-8936 "
                "893A-893B 893F-8940 8944-8945 8949-894A 894E-894F 8953-8954 8956 895B 8960 8965 "
                "896A 896F 8974 8979 897E 8984 8989 898E 8993 8998 "
                "899D 89A1-89A2 89A6-89A7 89AB-89AC 89B0-89B1 89B5-89B6 89BA-89BB 89BF-89C0 89C4-89C5 89C9-89CA "
                "89CE-89CF 89D3-89D4 89D6 89DB 89E0 89E5 89EA 89EF 89F4 89F9 "
                "89FE 8A04 8A09 8A0E 8A13 8A18 8A1D 8A21-8A22 8A26-8A27 8A2B-8A2C "
                "8A30-8A31 8A35-8A36 8A3A-8A3B 8A3F-8A40 8A44-8A45 8A49-8A4A 8A4E-8A4F 8A53-8A54 8A56 8A5B "
                "8A60 8A65 8A6A 8A6F 8A74 8A79 8A7E 8A84 8A89 8A8E "
                "8A93 8A98 8A9D 8AA1-8AA2 8AA6-8AA7 8AAB-8AAC 8AB0-8AB1 8AB5-8AB6 8ABA-8ABB 8ABF-8AC0 "
                "8AC4-8AC5 8AC9-8ACA 8ACE-8ACF 8AD3-8AD4 8AD6 8ADB 8AE0 8AE5 8AEA 8AEF "
                "8AF4 8AF9 8AFE 8B04 8B09 8B0E 8B13 8B18 8B1D 8B21-8B22 "
                "8B26-8B27 8B2B-8B2C 8B30-8B31 8B35-8B36 8B3A-8B3B 8B3F-8B40 8B44-8B45 8B49-8B4A 8B4E-8B4F 8B53-8B54 "
                "8B56 8B5B 8B60 8B65 8B6A 8B6F 8B74 8B79 8B7E 8B84 "
                "8B89 8B8E 8B93 8B98 8B9D 8BA1-8BA2 8BA6-8BA7 8BAB-8BAC 8BB0-8BB1 8BB5-8BB6 "
                "8BBA-8BBB 8BBF-8BC0 8BC4-8BC5 8BC9-8BCA 8BCE-8BCF 8BD3-8BD4 8BD6 8BDB 8BE0 8BE5 "
                "8BEA 8BEF 8BF4 8BF9 8BFE 8C04 8C09 8C0E 8C13 8C18 "
                "8C1D 8C21-8C22 8C26-8C27 8C2B-8C2C 8C30-8C31 8C35-8C36 8C3A-8C3B 8C3F-8C40 8C44-8C45 8C49-8C4A "
                "8C4E-8C4F 8C53-8C54 8C56 8C5B 8C60 8C65 8C6A 8C6F 8C74 8C79 "
                "8C7E 8C84 8C89 8C8E 8C93 8C98 8C9D 8CA1-8CA2 8CA6-8CA7 8CAB-8CAC "
                "8CB0-8CB1 8CB5-8CB6 8CBA-8CBB 8CBF-8CC0 8CC4-8CC5 8CC9-8CCA 8CCE-8CCF 8CD3-8CD4 8CD6 8CDB "
                "8CE0 8CE5 8CEA 8CEF 8CF4 8CF9 8CFE 8D04 8D09 8D0E "
                "8D13 8D18 8D1D 8D21-8D22 8D26-8D27 8D2B-8D2C 8D30-8D31 8D35-8D36 8D3A-8D3B 8D3F-8D40 "
                "8D44-8D45 8D49-8D4A 8D4E-8D4F 8D53-8D54 8D56 8D5B 8D60 8D65 8D6A 8D6F "
                "8D74 8D79 8D7E 8D84 8D89 8D8E 8D93 8D98 8D9D 8DA1-8DA2 "
                "8DA6-8DA7 8DAB-8DAC 8DB0-8DB1 8DB5-8DB6 8DBA-8DBB 8DBF-8DC0 8DC4-8DC5 8DC9-8DCA 8DCE-8DCF 8DD3-8DD4 "
                "8DD6 8DDB 8DE0 8DE5 8DEA 8DEF 8DF4 8DF9 8DFE 8E04 "
                "8E09 8E0E 8E13 8E18 8E1D 8E21-8E22 8E26-8E27 8E2B-8E2C 8E30-8E31 8E35-8E36 "
                "8E3A-8E3B 8E3F-8E40 8E44-8E45 8E49-8E4A 8E4E-8E4F 8E53-8E54 8E56 8E5B 8E60 8E65 "
                "8E6A 8E6F 8E74 8E79 8E7E 8E84 8E89 8E8E 8E93 8E98 "
                "8E9D 8EA1-8EA2 8EA6-8EA7 8EAB-8EAC 8EB0-8EB1 8EB5-8EB6 8EBA-8EBB 8EBF-8EC0 8EC4-8EC5 8EC9-8ECA "
                "8ECE-8ECF 8ED3-8ED4 8ED6 8EDB 8EE0 8EE5 8EEA 8EEF 8EF4 8EF9 "
                "8EFE 8F04 8F09 8F0E 8F13 8F18 8F1D 8F21-8F22 8F26-8F27 8F2B-8F2C "
                "8F30-8F31 8F35-8F36 8F3A-8F3B 8F3F-8F40 8F44-8F45 8F49-8F4A 8F4E-8F4F 8F53-8F54 8F56 8F5B "
                "8F60 8F65 8F6A 8F6F 8F74 8F79 8F7E 8F84 8F89 8F8E "
                "8F93 8F98 8F9D 8FA1-8FA2 8FA6-8FA7 8FAB-8FAC 8FB0-8FB1 8FB5-8FB6 8FBA-8FBB 8FBF-8FC0 "
                "8FC4-8FC5 8FC9-8FCA 8FCE-8FCF 8FD3-8FD4 8FD6 8FDB 8FE0 8FE5 8FEA 8FEF "
                "8FF4 8FF9 8FFE 9004 9009 900E 9013 9018 901D 9021-9022 "
                "9026-9027 902B-902C 9030-9031 9035-9036 903A-903B 903F-9040 9044-9045 9049-904A 904E-904F 9053-9054 "
                "9056 905B 9060 9065 906A 906F 9074 9079 907E 9084 "
                "9089 908E 9093 9098 909D 90A1-90A2 90A6-90A7 90AB-90AC 90B0-90B1 90B5-90B6 "
                "90BA-90BB 90BF-90C0 90C4-90C5 90C9-90CA 90CE-90CF 90D3-90D4 90D6 90DB 90E0 90E5 "
                "90EA 90EF 90F4 90F9 90FE 9104 9109 910E 9113 9118 "
                "911D 9121-9122 9126-9127 912B-912C 9130-9131 9135-9136 913A-913B 913F-9140 9144-9145 9149-914A "
                "914E-914F 9153-9154 9156 915B 9160 9165 916A 916F 9174 9179 "
                "917E 9184 9189 918E 9193 9198 919D 91A1-91A2 91A6-91A7 91AB-91AC "
                "91B0-91B1 91B5-91B6 91BA-91BB 91BF-91C0 91C4-91C5 91C9-91CA 91CE-91CF 91D3-91D4 91D6 91DB "
                "91E0 91E5 91EA 91EF 91F4 91F9 91FE 9204 9209 920E "
                "9213 9218 921D 9221-9222 9226-9227 922B-922C 9230-9231 9235-9236 923A-923B 923F-9240 "
                "9244-9245 9249-924A 924E-924F 9253-9254 9256 925B 9260 9265 926A 926F "
                "9274 9279 927E 9284 9289 928E 9293 9298 929D 92A1-92A2 "
                "92A6-92A7 92AB-92AC 92B0-92B1 92B5-92B6 92BA-92BB 92BF-92C0 92C4-92C5 92C9-92CA 92CE-92CF 92D3-92D4 "
                "92D6 92DB 92E0 92E5 92EA 92EF 92F4 92F9 92FE 9304 "
                "9309 930E 9313 9318 931D 9321-9322 9326-9327 932B-932C 9330-9331 9335-9336 "
                "933A-933B 933F-9340 9344-9345 9349-934A 934E-934F 9353-9354 9356 935B 9360 9365 "
                "936A 936F 9374 9379 937E 9384 9389 938E 9393 9398 "
                "939D 93A1-93A2 93A6-93A7 93AB-93AC 93B0-93B1 93B5-93B6 93BA-93BB 93BF-93C0 93C4-93C5 93C9-93CA "
                "93CE-93CF 93D3-93D4 93D6 93DB 93E0 93E5 93EA 93EF 93F4 93F9 "
                "93FE 9404 9409 940E 9413 9418 941D 9421-9422 9426-9427 942B-942C "
                "9430-9431 9435-9436 943A-943B 943F-9440 9444-9445 9449-944A 944E-944F 9453-9454 9456 945B "
                "9460 9465 946A 946F 9474 9479 947E 9484 9489 948E "
                "9493 9498 949D 94A1-94A2 94A6-94A7 94AB-94AC 94B0-94B1 94B5-94B6 94BA-94BB 94BF-94C0 "
                "94C4-94C5 94C9-94CA 94CE-94CF 94D3-94D4 94D6 94DB 94E0 94E5 94EA 94EF "
                "94F4 94F9 94FE 9504 9509 950E 9513 9518 951D 9521-9522 "
                "9526-9527 952B-952C 9530-9531 9535-9536 953A-953B 953F-9540 9544-9545 9549-954A 954E-954F 9553-9554 "
                "9556 955B 9560 9565 956A 956F 9574 9579 957E 9584 "
                "9589 958E 9593 9598 959D 95A1-95A2 95A6-95A7 95AB-95AC 95B0-95B1 95B5-95B6 "
                "95BA-95BB 95BF-95C0 95C4-95C5 95C9-95CA 95CE-95CF 95D3-95D4 95D6 95DB 95E0 95E5 "
                "95EA 95EF 95F4 95F9 95FE 9604 9609 960E 9613 9618 "
                "961D 9621-9622 9626-9627 962B-962C 9630-9631 9635-9636 963A-963B 963F-9640 9644-9645 9649-964A "
                "964E-964F 9653-9654 9656 965B 9660 9665 966A 966F 9674 9679 "
                "967E 9684 9689 968E 9693 9698 969D 96A1-96A2 96A6-96A7 96AB-96AC "
                "96B0-96B1 96B5-96B6 96BA-96BB 96BF-96C0 96C4-96C5 96C9-96CA 96CE-96CF 96D3-96D4 96D6 96DB "
                "96E0 96E5 96EA 96EF 96F4 96F9 96FE 9704 9709 970E "
                "9713 9718 971D 9721-9722 9726-9727 972B-972C 9730-9731 9735-9736 973A-973B 973F-9740 "
                "9744-9745 9749-974A 974E-974F 9753-9754 9756 975B 9760 9765 976A 976F "
                "9774 9779 977E 9784 9789 978E 9793 9798 979D 97A1-97A2 "
                "97A6-97A7 97AB-97AC 97B0-97B1 97B5-97B6 97BA-97BB 97BF-97C0 97C4-97C5 97C9-97CA 97CE-97CF 97D3-97D4 "
                "97D6 97DB 97E0 97E5 97EA 97EF 97F4 97F9 97FE 9804 "
                "9809 980E 9813 9818 981D 9821-9822 9826-9827 982B-982C 9830-9831 9835-9836 "
                "983A-983B 983F-9840 9844-9845 9849-984A 984E-984F 9853-9854 9856 985B 9860 9865 "
                "986A 986F 9874 9879 987E 9884 9889 988E 9893 9898 "
                "989D 98A1-98A2 98A6-98A7 98AB-98AC 98B0-98B1 98B5-98B6 98BA-98BB 98BF-98C0 98C4-98C5 98C9-98CA "
                "98CE-98CF 98D3-98D4 98D6 98DB 98E0 98E5 98EA 98EF 98F4 98F9 "
                "98FE 9904 9909 990E 9913 9918 991D 9921-9922 9926-9927 992B-992C "
                "9930-9931 9935-9936 993A-993B 993F-9940 9944-9945 9949-994A 994E-994F 9953-9954 9956 995B "
                "9960 9965 996A 996F 9974 9979 997E 9984 9989 998E "
                "9993 9998 999D 99A1-99A2 99A6-99A7 99AB-99AC 99B0-99B1 99B5-99B6 99BA-99BB 99BF-99C0 "
                "99C4-99C5 99C9-99CA 99CE-99CF 99D3-99D4 99D6 99DB 99E0 99E5 99EA 99EF "
                "99F4 99F9 99FE 9A04 9A09 9A0E 9A13 9A18 9A1D 9A21-9A22 "
                "9A26-9A27 9A2B-9A2C 9A30-9A31 9A35-9A36 9A3A-9A3B 9A3F-9A40 9A44-9A45 9A49-9A4A 9A4E-9A4F 9A53-9A54 "
                "9A56 9A5B 9A60 9A65 9A6A 9A6F 9A74 9A79 9A7E 9A84 "
                "9A89 9A8E 9A93 9A98 9A9D 9AA1-9AA2 9AA6-9AA7 9AAB-9AAC 9AB0-9AB1 9AB5-9AB6 "
                "9ABA-9ABB 9ABF-9AC0 9AC4-9AC5 9AC9-9ACA 9ACE-9ACF 9AD3-9AD4 9AD6 9ADB 9AE0 9AE5 "
                "9AEA 9AEF 9AF4 9AF9 9AFE 9B04 9B09 9B0E 9B13 9B18 "
                "9B1D 9B21-9B22 9B26-9B27 9B2B-9B2C 9B30-9B31 9B35-9B36 9B3A-9B3B 9B3F-9B40 9B44-9B45 9B49-9B4A "
                "9B4E-9B4F 9B53-9B54 9B56 9B5B 9B60 9B65 9B6A 9B6F 9B74 9B79 "
                "9B7E 9B84 9B89 9B8E 9B93 9B98 9B9D 9BA1-9BA2 9BA6-9BA7 9BAB-9BAC "
                "9BB0-9BB1 9BB5-9BB6 9BBA-9BBB 9BBF-9BC0 9BC4-9BC5 9BC9-9BCA 9BCE-9BCF 9BD3-9BD4 9BD6 9BDB "
                "9BE0 9BE5 9BEA 9BEF 9BF4 9BF9 9BFE 9C04 9C09 9C0E "
                "9C13 9C18 9C1D 9C21-9C22 9C26-9C27 9C2B-9C2C 9C30-9C31 9C35-9C36 9C3A-9C3B 9C3F-9C40 "
                "9C44-9C45 9C49-9C4A 9C4E-9C4F 9C53-9C54 9C56 9C5B 9C60 9C65 9C6A 9C6F "
                "9C74 9C79 9C7E 9C84 9C89 9C8E 9C93 9C98 9C9D 9CA1-9CA2 "
                "9CA6-9CA7 9CAB-9CAC 9CB0-9CB1 9CB5-9CB6 9CBA-9CBB 9CBF-9CC0 9CC4-9CC5 9CC9-9CCA 9CCE-9CCF 9CD3-9CD4 "
                "9CD6 9CDB 9CE0 9CE5 9CEA 9CEF 9CF4 9CF9 9CFE 9D04 "
                "9D09 9D0E 9D13 9D18 9D1D 9D21-9D22 9D26-9D27 9D2B-9D2C 9D30-9D31 9D35-9D36 "
                "9D3A-9D3B 9D3F-9D40 9D44-9D45 9D49-9D4A 9D4E-9D4F 9D53-9D54 9D56 9D5B 9D60 9D65 "
                "9D6A 9D6F 9D74 9D79 9D7E 9D84 9D89 9D8E 9D93 9D98 "
                "9D9D 9DA1-9DA2 9DA6-9DA7 9DAB-9DAC 9DB0-9DB1 9DB5-9DB6 9DBA-9DBB 9DBF-9DC0 9DC4-9DC5 9DC9-9DCA "
                "9DCE-9DCF 9DD3-9DD4 9DD6 9DDB 9DE0 9DE5 9DEA 9DEF 9DF4 9DF9 "
                "9DFE 9E04 9E09 9E0E 9E13 9E18 9E1D 9E21-9E22 9E26-9E27 9E2B-9E2C "
                "9E30-9E31 9E35-9E36 9E3A-9E3B 9E3F-9E40 9E44-9E45 9E49-9E4A 9E4E-9E4F 9E53-9E54 9E56 9E5B "
                "9E60 9E65 9E6A 9E6F 9E74 9E79 9E7E 9E84 9E89 9E8E "
                "9E93 9E98 9E9D 9EA1-9EA2 9EA6-9EA7 9EAB-9EAC 9EB0-9EB1 9EB5-9EB6 9EBA-9EBB 9EBF-9EC0 "
                "9EC4-9EC5 9EC9-9ECA 9ECE-9ECF 9ED3-9ED4 9ED6 9EDB 9EE0 9EE5 9EEA 9EEF "
                "9EF4 9EF9 9EFE 9F04 9F09 9F0E 9F13 9F18 9F1D 9F21-9F22 "
                "9F26-9F27 9F2B-9F2C 9F30-9F31 9F35-9F36 9F3A-9F3B 9F3F-9F40 9F44-9F45 9F49-9F4A 9F4E-9F4F 9F53-9F54 "
                "9F56 9F5B 9F60 9F65 9F6A 9F6F 9F74 9F79 9F7E 9F84 "
                "9F89 9F8E 9F93 9F98 9F9D 9FA1-9FA2 9FA6-9FA7 9FAB-9FAC 9FB0-9FB1 9FB5-9FB6 "
                "9FBA-9FBB 9FBF-9FC0 9FC4-9FC5 9FC9-9FCA 9FCE-9FCF 9FD3-9FD4 9FD6 9FDB 9FE0 9FE5 "
                "9FEA 9FEF 9FF4 9FF9 9FFE A004 A009 A00E A013 A018 "
                "A01D A021-A022 A026-A027 A02B-A02C A030-A031 A035-A036 A03A-A03B A03F-A040 A044-A045 A049-A04A "
                "A04E-A04F A053-A054 A056 A05B A060 A065 A06A A06F A074 A079 "
                "A07E A084 A089 A08E A093 A098 A09D A0A1-A0A2 A0A6-A0A7 A0AB-A0AC "
                "A0B0-A0B1 A0B5-A0B6 A0BA-A0BB A0BF-A0C0 A0C4-A0C5 A0C9-A0CA A0CE-A0CF A0D3-A0D4 A0D6 A0DB "
                "A0E0 A0E5 A0EA A0EF A0F4 A0F9 A0FE A104 A109 A10E "
                "A113 A118 A11D A121-A122 A126-A127 A12B-A12C A130-A131 A135-A136 A13A-A13B A13F-A140 "
                "A144-A145 A149-A14A A14E-A14F A153-A154 A156 A15B A160 A165 A16A A16F "
                "A174 A179 A17E A184 A189 A18E A193 A198 A19D A1A1-A1A2 "
                "A1A6-A1A7 A1AB-A1AC A1B0-A1B1 A1B5-A1B6 A1BA-A1BB A1BF-A1C0 A1C4-A1C5 A1C9-A1CA A1CE-A1CF A1D3-A1D4 "
                "A1D6 A1DB A1E0 A1E5 A1EA A1EF A1F4 A1F9 A1FE A204 "
                "A209 A20E A213 A218 A21D A221-A222 A226-A227 A22B-A22C A230-A231 A235-A236 "
                "A23A-A23B A23F-A240 A244-A245 A249-A24A A24E-A24F A253-A254 A256 A25B A260 A265 "
                "A26A A26F A274 A279 A27E A284 A289 A28E A293 A298 "
                "A29D A2A1-A2A2 A2A6-A2A7 A2AB-A2AC A2B0-A2B1 A2B5-A2B6 A2BA-A2BB A2BF-A2C0 A2C4-A2C5 A2C9-A2CA "
                "A2CE-A2CF A2D3-A2D4 A2D6 A2DB A2E0 A2E5 A2EA A2EF A2F4 A2F9 "
                "A2FE A304 A309 A30E A313 A318 A31D A321-A322 A326-A327 A32B-A32C "
                "A330-A331 A335-A336 A33A-A33B A33F-A340 A344-A345 A349-A34A A34E-A34F A353-A354 A356 A35B "
                "A360 A365 A36A A36F A374 A379 A37E A384 A389 A38E "
                "A393 A398 A39D A3A1-A3A2 A3A6-A3A7 A3AB-A3AC A3B0-A3B1 A3B5-A3B6 A3BA-A3BB A3BF-A3C0 "
                "A3C4-A3C5 A3C9-A3CA A3CE-A3CF A3D3-A3D4 A3D6 A3DB A3E0 A3E5 A3EA A3EF "
                "A3F4 A3F9 A3FE A404 A409 A40E A413 A418 A41D A421-A422 "
                "A426-A427 A42B-A42C A430-A431 A435-A436 A43A-A43B A43F-A440 A444-A445 A449-A44A A44E-A44F A453-A454 "
                "A456 A45B A460 A465 A46A A46F A474 A479 A47E A484 "
                "A489 A48E A493 A498 A49D A4A1-A4A2 A4A6-A4A7 A4AB-A4AC A4B0-A4B1 A4B5-A4B6 "
                "A4BA-A4BB A4BF-A4C0 A4C4-A4C5 A4C9-A4CA A4CE-A4CF A4D3-A4D4 A4D6 A4DB A4E0 A4E5 "
                "A4EA A4EF A4F4 A4F9 A4FE A504 A509 A50E A513 A518 "
                "A51D A521-A522 A526-A527 A52B-A52C A530-A531 A535-A536 A53A-A53B A53F-A540 A544-A545 A549-A54A "
                "A54E-A54F A553-A554 A556 A55B A560 A565 A56A A56F A574 A579 "
                "A57E A584 A589 A58E A593 A598 A59D A5A1-A5A2 A5A6-A5A7 A5AB-A5AC "
                "A5B0-A5B1 A5B5-A5B6 A5BA-A5BB A5BF-A5C0 A5C4-A5C5 A5C9-A5CA A5CE-A5CF A5D3-A5D4 A5D6 A5DB "
                "A5E0 A5E5 A5EA A5EF A5F4 A5F9 A5FE A604 A609 A60E "
                "A613 A618 A61D A621-A622 A626-A627 A62B-A62C A630-A631 A635-A636 A63A-A63B A63F-A640 "
                "A644-A645 A649-A64A A64E-A64F A653-A654 A656 A65B A660 A665 A66A A66F "
                "A674 A679 A67E A684 A689 A68E A693 A698 A69D A6A1-A6A2 "
                "A6A6-A6A7 A6AB-A6AC A6B0-A6B1 A6B5-A6B6 A6BA-A6BB A6BF-A6C0 A6C4-A6C5 A6C9-A6CA A6CE-A6CF A6D3-A6D4 "
                "A6D6 A6DB A6E0 A6E5 A6EA A6EF A6F4 A6F9 A6FE A704 "
                "A709 A70E A713 A718 A71D A721-A722 A726-A727 A72B-A72C A730-A731 A735-A736 "
                "A73A-A73B A73F-A740 A744-A745 A749-A74A A74E-A74F A753-A754 A756 A75B A760 A765 "
                "A76A A76F A774 A779 A77E A784 A789 A78E A793 A798 "
                "A79D A7A1-A7A2 A7A6-A7A7 A7AB-A7AC A7B0-A7B1 A7B5-A7B6 A7BA-A7BB A7BF-A7C0 A7C4-A7C5 A7C9-A7CA "
                "A7CE-A7CF A7D3-A7D4 A7D6 A7DB A7E0 A7E5 A7EA A7EF A7F4 A7F9 "
                "A7FE A804 A809 A80E A813 A818 A81D A821-A822 A826-A827 A82B-A82C "
                "A830-A831 A835-A836 A83A-A83B A83F-A840 A844-A845 A849-A84A A84E-A84F A853-A854 A856 A85B "
                "A860 A865 A86A A86F A874 A879 A87E A884 A889 A88E "
                "A893 A898 A89D A8A1-A8A2 A8A6-A8A7 A8AB-A8AC A8B0-A8B1 A8B5-A8B6 A8BA-A8BB A8BF-A8C0 "
                "A8C4-A8C5 A8C9-A8CA A8CE-A8CF A8D3-A8D4 A8D6 A8DB A8E0 A8E5 A8EA A8EF "
                "A8F4 A8F9 A8FE A904 A909 A90E A913 A918 A91D A921-A922 "
                "A926-A927 A92B-A92C A930-A931 A935-A936 A93A-A93B A93F-A940 A944-A945 A949-A94A A94E-A94F A953-A954 "
                "A956 A95B A960 A965 A96A A96F A974 A979 A97E A984 "
                "A989 A98E A993 A998 A99D A9A1-A9A2 A9A6-A9A7 A9AB-A9AC A9B0-A9B1 A9B5-A9B6 "
                "A9BA-A9BB A9BF-A9C0 A9C4-A9C5 A9C9-A9CA A9CE-A9CF A9D3-A9D4 A9D6 A9DB A9E0 A9E5 "
                "A9EA A9EF A9F4 A9F9 A9FE AA04 AA09 AA0E AA13 AA18 "
                "AA1D AA21-AA22 AA26-AA27 AA2B-AA2C AA30-AA31 AA35-AA36 AA3A-AA3B AA3F-AA40 AA44-AA45 AA49-AA4A "
                "AA4E-AA4F AA53-AA54 AA56 AA5B AA60 AA65 AA6A AA6F AA74 AA79 "
                "AA7E AA84 AA89 AA8E AA93 AA98 AA9D AAA1-AAA2 AAA6-AAA7 AAAB-AAAC "
                "AAB0-AAB1 AAB5-AAB6 AABA-AABB AABF-AAC0 AAC4-AAC5 AAC9-AACA AACE-AACF AAD3-AAD4 AAD6 AADB "
                "AAE0 AAE5 AAEA AAEF AAF4 AAF9 AAFE AB04 AB09 AB0E "
                "AB13 AB18 AB1D AB21-AB22 AB26-AB27 AB2B-AB2C AB30-AB31 AB35-AB36 AB3A-AB3B AB3F-AB40 "
                "AB44-AB45 AB49-AB4A AB4E-AB4F AB53-AB54 AB56 AB5B AB60 AB65 AB6A AB6F "
                "AB74 AB79 AB7E AB84 AB89 AB8E AB93 AB98 AB9D ABA1-ABA2 "
                "ABA6-ABA7 ABAB-ABAC ABB0-ABB1 ABB5-ABB6 ABBA-ABBB ABBF-ABC0 ABC4-ABC5 ABC9-ABCA ABCE-ABCF ABD3-ABD4 "
                "ABD6 ABDB ABE0 ABE5 ABEA ABEF ABF4 ABF9 ABFE AC04 "
                "AC09 AC0E AC13 AC18 AC1D AC21-AC22 AC26-AC27 AC2B-AC2C AC30-AC31 AC35-AC36 "
                "AC3A-AC3B AC3F-AC40 AC44-AC45 AC49-AC4A AC4E-AC4F AC53-AC54 AC56 AC5B AC60 AC65 "
                "AC6A AC6F AC74 AC79 AC7E AC84 AC89 AC8E AC93 AC98 "
                "AC9D ACA1-ACA2 ACA6-ACA7 ACAB-ACAC ACB0-ACB1 ACB5-ACB6 ACBA-ACBB ACBF-ACC0 ACC4-ACC5 ACC9-ACCA "
                "ACCE-ACCF ACD3-ACD4 ACD6 ACDB ACE0 ACE5 ACEA ACEF ACF4 ACF9 "
                "ACFE AD04 AD09 AD0E AD13 AD18 AD1D AD21-AD22 AD26-AD27 AD2B-AD2C "
                "AD30-AD31 AD35-AD36 AD3A-AD3B AD3F-AD40 AD44-AD45 AD49-AD4A AD4E-AD4F AD53-AD54 AD56 AD5B "
                "AD60 AD65 AD6A AD6F AD74 AD79 AD7E AD84 AD89 AD8E "
                "AD93 AD98 AD9D ADA1-ADA2 ADA6-ADA7 ADAB-ADAC ADB0-ADB1 ADB5-ADB6 ADBA-ADBB ADBF-ADC0 "
                "ADC4-ADC5 ADC9-ADCA ADCE-ADCF ADD3-ADD4 ADD6 ADDB ADE0 ADE5 ADEA ADEF "
                "ADF4 ADF9 ADFE AE04 AE09 AE0E AE13 AE18 AE1D AE21-AE22 "
                "AE26-AE27 AE2B-AE2C AE30-AE31 AE35-AE36 AE3A-AE3B AE3F-AE40 AE44-AE45 AE49-AE4A AE4E-AE4F AE53-AE54 "
                "AE56 AE5B AE60 AE65 AE6A AE6F AE74 AE79 AE7E AE84 "
                "AE89 AE8E AE93 AE98 AE9D AEA1-AEA2 AEA6-AEA7 AEAB-AEAC AEB0-AEB1 AEB5-AEB6 "
                "AEBA-AEBB AEBF-AEC0 AEC4-AEC5 AEC9-AECA AECE-AECF AED3-AED4 AED6 AEDB AEE0 AEE5 "
                "AEEA AEEF AEF4 AEF9 AEFE AF04 AF09 AF0E AF13 AF18 "
                "AF1D AF21-AF22 AF26-AF27 AF2B-AF2C AF30-AF31 AF35-AF36 AF3A-AF3B AF3F-AF40 AF44-AF45 AF49-AF4A "
                "AF4E-AF4F AF53-AF54 AF56 AF5B AF60 AF65 AF6A AF6F AF74 AF79 "
                "AF7E AF84 AF89 AF8E AF93 AF98 AF9D AFA1-AFA2 AFA6-AFA7 AFAB-AFAC "
                "AFB0-AFB1 AFB5-AFB6 AFBA-AFBB AFBF-AFC0 AFC4-AFC5 AFC9-AFCA AFCE-AFCF AFD3-AFD4 AFD6 AFDB "
                "AFE0 AFE5 AFEA AFEF AFF4 AFF9 AFFE B004 B009 B00E "
                "B013 B018 B01D B021-B022 B026-B027 B02B-B02C B030-B031 B035-B036 B03A-B03B B03F-B040 "
                "B044-B045 B049-B04A B04E-B04F B053-B054 B056 B05B B060 B065 B06A B06F "
                "B074 B079 B07E B084 B089 B08E B093 B098 B09D B0A1-B0A2 "
                "B0A6-B0A7 B0AB-B0AC B0B0-B0B1 B0B5-B0B6 B0BA-B0BB B0BF-B0C0 B0C4-B0C5 B0C9-B0CA B0CE-B0CF B0D3-B0D4 "
                "B0D6 B0DB B0E0 B0E5 B0EA B0EF B0F4 B0F9 B0FE B104 "
                "B109 B10E B113 B118 B11D B121-B122 B126-B127 B12B-B12C B130-B131 B135-B136 "
                "B13A-B13B B13F-B140 B144-B145 B149-B14A B14E-B14F B153-B154 B156 B15B B160 B165 "
                "B16A B16F B174 B179 B17E B184 B189 B18E B193 B198 "
                "B19D B1A1-B1A2 B1A6-B1A7 B1AB-B1AC B1B0-B1B1 B1B5-B1B6 B1BA-B1BB B1BF-B1C0 B1C4-B1C5 B1C9-B1CA "
                "B1CE-B1CF B1D3-B1D4 B1D6 B1DB B1E0 B1E5 B1EA B1EF B1F4 B1F9 "
                "B1FE B204 B209 B20E B213 B218 B21D B221-B222 B226-B227 B22B-B22C "
                "B230-B231 B235-B236 B23A-B23B B23F-B240 B244-B245 B249-B24A B24E-B24F B253-B254 B256 B25B "
                "B260 B265 B26A B26F B274 B279 B27E B284 B289 B28E "
                "B293 B298 B29D B2A1-B2A2 B2A6-B2A7 B2AB-B2AC B2B0-B2B1 B2B5-B2B6 B2BA-B2BB B2BF-B2C0 "
                "B2C4-B2C5 B2C9-B2CA B2CE-B2CF B2D3-B2D4 B2D6 B2DB B2E0 B2E5 B2EA B2EF "
                "B2F4 B2F9 B2FE B304 B309 B30E B313 B318 B31D B321-B322 "
                "B326-B327 B32B-B32C B330-B331 B335-B336 B33A-B33B B33F-B340 B344-B345 B349-B34A B34E-B34F B353-B354 "
                "B356 B35B B360 B365 B36A B36F B374 B379 B37E B384 "
                "B389 B38E B393 B398 B39D B3A1-B3A2 B3A6-B3A7 B3AB-B3AC B3B0-B3B1 B3B5-B3B6 "
                "B3BA-B3BB B3BF-B3C0 B3C4-B3C5 B3C9-B3CA B3CE-B3CF B3D3-B3D4 B3D6 B3DB B3E0 B3E5 "
                "B3EA B3EF B3F4 B3F9 B3FE B404 B409 B40E B413 B418 "
                "B41D B421-B422 B426-B427 B42B-B42C B430-B431 B435-B436 B43A-B43B B43F-B440 B444-B445 B449-B44A "
                "B44E-B44F B453-B454 B456 B45B B460 B465 B46A B46F B474 B479 "
                "B47E B484 B489 B48E B493 B498 B49D B4A1-B4A2 B4A6-B4A7 B4AB-B4AC "
                "B4B0-B4B1 B4B5-B4B6 B4BA-B4BB B4BF-B4C0 B4C4-B4C5 B4C9-B4CA B4CE-B4CF B4D3-B4D4 B4D6 B4DB "
                "B4E0 B4E5 B4EA B4EF B4F4 B4F9 B4FE B504 B509 B50E "
                "B513 B518 B51D B521-B522 B526-B527 B52B-B52C B530-B531 B535-B536 B53A-B53B B53F-B540 "
                "B544-B545 B549-B54A B54E-B54F B553-B554 B556 B55B B560 B565 B56A B56F "
                "B574 B579 B57E B584 B589 B58E B593 B598 B59D B5A1-B5A2 "
                "B5A6-B5A7 B5AB-B5AC B5B0-B5B1 B5B5-B5B6 B5BA-B5BB B5BF-B5C0 B5C4-B5C5 B5C9-B5CA B5CE-B5CF B5D3-B5D4 "
                "B5D6 B5DB B5E0 B5E5 B5EA B5EF B5F4 B5F9 B5FE B604 "
                "B609 B60E B613 B618 B61D B621-B622 B626-B627 B62B-B62C B630-B631 B635-B636 "
                "B63A-B63B B63F-B640 B644-B645 B649-B64A B64E-B64F B653-B654 B656 B65B B660 B665 "
                "B66A B66F B674 B679 B67E B684 B689 B68E B693 B698 "
                "B69D B6A1-B6A2 B6A6-B6A7 B6AB-B6AC B6B0-B6B1 BC81 BCBA BCF3 BD09 BD16 "
                "BD23 BD2A BD31 BD38 BD3F BD46 BD4D BD54-BD55 BD60 BD67 "
                "BD70 BD77 BD80 BD88 BD91 BD9A-BD9B BDA4-BDA5 BDAE-BDB0 BDBA-BDBB BDC6-BDC8 "
                "BDD3-BDD6 BDEB BDED BDEF BDF1 BE00-BE05 BE3A-BE42 BE50-BE54 BE5E-BE62 BE6A-BE6D "
                "BE74 BE7A BE85 BE8C BE92-BE93 BE98 BE9E BEA3 BEA7 BEAC "
                "BEB0 BEB4 BEB8 BEBC BECA BF2C BF31 BF46 BF4D BF54 "
                "BF58 BF5D BF61-BF62 BF66-BF67 BF6C BF72 BF78-BF79 BF80 BFBD BFC1 "
                "BFC5 BFC8 BFCF BFD2 BFDD C00C C011 C016 C038 C03A "
                "C05D C067-C068 C070-C071 C085 C08C C0A3 C0CF C109 C113 C115 "
                "C12A C133-C134 C13E C149 C14B C156 C160-C161 C16C-C16D C185 C192 "
                "C19E C1B3 C1BF C1C9 C1CD C1DA C1E0 C1E5 C1F6 C206 "
                "C20B C21E C229 C22D C24E C256 C259 C25F C262 C275 "
                "C27A C280 C292 C296 C299 C2A1 C2B6"
            ),
            1: (
                "00D6 00DB 00E0 00E5 00EA 00EF 00F4 00F9 00FE 0104 "
                "0109 010E 0113 0118 011D 0121-0122 0126-0127 012B-012C 0130-0131 0135-0136 "
                "013A-013B 013F-0140 0144-0145 0149-014A 014E-014F 0153-0154 0156 015B 0160 0165 "
                "016A 016F 0174 0179 017E 0184 0189 018E 0193 0198 "
                "019D 01A1-01A2 01A6-01A7 01AB-01AC 01B0-01B1 01B5-01B6 01BA-01BB 01BF-01C0 01C4-01C5 01C9-01CA "
                "01CE-01CF 01D3-01D4 01D6 01DB 01E0 01E5 01EA 01EF 01F4 01F9 "
                "01FE 0204 0209 020E 0213 0218 021D 0221-0222 0226-0227 022B-022C "
                "0230-0231 0235-0236 023A-023B 023F-0240 0244-0245 0249-024A 024E-024F 0253-0254 0256 025B "
                "0260 0265 026A 026F 0274 0279 027E 0284 0289 028E "
                "0293 0298 029D 02A1-02A2 02A6-02A7 02AB-02AC 02B0-02B1 02B5-02B6 02BA-02BB 02BF-02C0 "
                "02C4-02C5 02C9-02CA 02CE-02CF 02D3-02D4 02D6 02DB 02E0 02E5 02EA 02EF "
                "02F4 02F9 02FE 0304 0309 030E 0313 0318 031D 0321-0322 "
                "0326-0327 032B-032C 0330-0331 0335-0336 033A-033B 033F-0340 0344-0345 0349-034A 034E-034F 0353-0354 "
                "0356 035B 0360 0365 036A 036F 0374 0379 037E 0384 "
                "0389 038E 0393 0398 039D 03A1-03A2 03A6-03A7 03AB-03AC 03B0-03B1 03B5-03B6 "
                "03BA-03BB 03BF-03C0 03C4-03C5 03C9-03CA 03CE-03CF 03D3-03D4 03D6 03DB 03E0 03E5 "
                "03EA 03EF 03F4 03F9 03FE 0404 0409 040E 0413 0418 "
                "041D 0421-0422 0426-0427 042B-042C 0430-0431 0435-0436 043A-043B 043F-0440 0444-0445 0449-044A "
                "044E-044F 0453-0454 0456 045B 0460 0465 046A 046F 0474 0479 "
                "047E 0484 0489 048E 0493 0498 049D 04A1-04A2 04A6-04A7 04AB-04AC "
                "04B0-04B1 04B5-04B6 04BA-04BB 04BF-04C0 04C4-04C5 04C9-04CA 04CE-04CF 04D3-04D4 04D6 04DB "
                "04E0 04E5 04EA 04EF 04F4 04F9 04FE 0504 0509 050E "
                "0513 0518 051D 0521-0522 0526-0527 052B-052C 0530-0531 0535-0536 053A-053B 053F-0540 "
                "0544-0545 0549-054A 054E-054F 0553-0554 0556 055B 0560 0565 056A 056F "
                "0574 0579 057E 0584 0589 058E 0593 0598 059D 05A1-05A2 "
                "05A6-05A7 05AB-05AC 05B0-05B1 05B5-05B6 05BA-05BB 05BF-05C0 05C4-05C5 05C9-05CA 05CE-05CF 05D3-05D4 "
                "05D6 05DB 05E0 05E5 05EA 05EF 05F4 05F9 05FE 0604 "
                "0609 060E 0613 0618 061D 0621-0622 0626-0627 062B-062C 0630-0631 0635-0636 "
                "063A-063B 063F-0640 0644-0645 0649-064A 064E-064F 0653-0654 0656 065B 0660 0665 "
                "066A 066F 0674 0679 067E 0684 0689 068E 0693 0698 "
                "069D 06A1-06A2 06A6-06A7 06AB-06AC 06B0-06B1 06B5-06B6 06BA-06BB 06BF-06C0 06C4-06C5 06C9-06CA "
                "06CE-06CF 06D3-06D4 06D6 06DB 06E0 06E5 06EA 06EF 06F4 06F9 "
                "06FE 0704 0709 070E 0713 0718 071D 0721-0722 0726-0727 072B-072C "
                "0730-0731 0735-0736 073A-073B 073F-0740 0744-0745 0749-074A 074E-074F 0753-0754 0756 075B "
                "0760 0765 076A 076F 0774 0779 077E 0784 0789 078E "
                "0793 0798 079D 07A1-07A2 07A6-07A7 07AB-07AC 07B0-07B1 07B5-07B6 07BA-07BB 07BF-07C0 "
                "07C4-07C5 07C9-07CA 07CE-07CF 07D3-07D4 07D6 07DB 07E0 07E5 07EA 07EF "
                "07F4 07F9 07FE 0804 0809 080E 0813 0818 081D 0821-0822 "
                "0826-0827 082B-082C 0830-0831 0835-0836 083A-083B 083F-0840 0844-0845 0849-084A 084E-084F 0853-0854 "
                "0856 085B 0860 0865 086A 086F 0874 0879 087E 0884 "
                "0889 088E 0893 0898 089D 08A1-08A2 08A6-08A7 08AB-08AC 08B0-08B1 08B5-08B6 "
                "08BA-08BB 08BF-08C0 08C4-08C5 08C9-08CA 08CE-08CF 08D3-08D4 08D6 08DB 08E0 08E5 "
                "08EA 08EF 08F4 08F9 08FE 0904 0909 090E 0913 0918 "
                "091D 0921-0922 0926-0927 092B-092C 0930-0931 0935-0936 093A-093B 093F-0940 0944-0945 0949-094A "
                "094E-094F 0953-0954 0956 095B 0960 0965 096A 096F 0974 0979 "
                "097E 0984 0989 098E 0993 0998 099D 09A1-09A2 09A6-09A7 09AB-09AC "
                "09B0-09B1 09B5-09B6 09BA-09BB 09BF-09C0 09C4-09C5 09C9-09CA 09CE-09CF 09D3-09D4 09D6 09DB "
                "09E0 09E5 09EA 09EF 09F4 09F9 09FE 0A04 0A09 0A0E "
                "0A13 0A18 0A1D 0A21-0A22 0A26-0A27 0A2B-0A2C 0A30-0A31 0A35-0A36 0A3A-0A3B 0A3F-0A40 "
                "0A44-0A45 0A49-0A4A 0A4E-0A4F 0A53-0A54 0A56 0A5B 0A60 0A65 0A6A 0A6F "
                "0A74 0A79 0A7E 0A84 0A89 0A8E 0A93 0A98 0A9D 0AA1-0AA2 "
                "0AA6-0AA7 0AAB-0AAC 0AB0-0AB1 0AB5-0AB6 0ABA-0ABB 0ABF-0AC0 0AC4-0AC5 0AC9-0ACA 0ACE-0ACF 0AD3-0AD4 "
                "0AD6 0ADB 0AE0 0AE5 0AEA 0AEF 0AF4 0AF9 0AFE 0B04 "
                "0B09 0B0E 0B13 0B18 0B1D 0B21-0B22 0B26-0B27 0B2B-0B2C 0B30-0B31 0B35-0B36 "
                "0B3A-0B3B 0B3F-0B40 0B44-0B45 0B49-0B4A 0B4E-0B4F 0B53-0B54 0B56 0B5B 0B60 0B65 "
                "0B6A 0B6F 0B74 0B79 0B7E 0B84 0B89 0B8E 0B93 0B98 "
                "0B9D 0BA1-0BA2 0BA6-0BA7 0BAB-0BAC 0BB0-0BB1 0BB5-0BB6 0BBA-0BBB 0BBF-0BC0 0BC4-0BC5 0BC9-0BCA "
                "0BCE-0BCF 0BD3-0BD4 0BD6 0BDB 0BE0 0BE5 0BEA 0BEF 0BF4 0BF9 "
                "0BFE 0C04 0C09 0C0E 0C13 0C18 0C1D 0C21-0C22 0C26-0C27 0C2B-0C2C "
                "0C30-0C31 0C35-0C36 0C3A-0C3B 0C3F-0C40 0C44-0C45 0C49-0C4A 0C4E-0C4F 0C53-0C54 0C56 0C5B "
                "0C60 0C65 0C6A 0C6F 0C74 0C79 0C7E 0C84 0C89 0C8E "
                "0C93 0C98 0C9D 0CA1-0CA2 0CA6-0CA7 0CAB-0CAC 0CB0-0CB1 0CB5-0CB6 0CBA-0CBB 0CBF-0CC0 "
                "0CC4-0CC5 0CC9-0CCA 0CCE-0CCF 0CD3-0CD4 0CD6 0CDB 0CE0 0CE5 0CEA 0CEF "
                "0CF4 0CF9 0CFE 0D04 0D09 0D0E 0D13 0D18 0D1D 0D21-0D22 "
                "0D26-0D27 0D2B-0D2C 0D30-0D31 0D35-0D36 0D3A-0D3B 0D3F-0D40 0D44-0D45 0D49-0D4A 0D4E-0D4F 0D53-0D54 "
                "0D56 0D5B 0D60 0D65 0D6A 0D6F 0D74 0D79 0D7E 0D84 "
                "0D89 0D8E 0D93 0D98 0D9D 0DA1-0DA2 0DA6-0DA7 0DAB-0DAC 0DB0-0DB1 0DB5-0DB6 "
                "0DBA-0DBB 0DBF-0DC0 0DC4-0DC5 0DC9-0DCA 0DCE-0DCF 0DD3-0DD4 0DD6 0DDB 0DE0 0DE5 "
                "0DEA 0DEF 0DF4 0DF9 0DFE 0E04 0E09 0E0E 0E13 0E18 "
                "0E1D 0E21-0E22 0E26-0E27 0E2B-0E2C 0E30-0E31 0E35-0E36 0E3A-0E3B 0E3F-0E40 0E44-0E45 0E49-0E4A "
                "0E4E-0E4F 0E53-0E54 0E56 0E5B 0E60 0E65 0E6A 0E6F 0E74 0E79 "
                "0E7E 0E84 0E89 0E8E 0E93 0E98 0E9D 0EA1-0EA2 0EA6-0EA7 0EAB-0EAC "
                "0EB0-0EB1 0EB5-0EB6 0EBA-0EBB 0EBF-0EC0 0EC4-0EC5 0EC9-0ECA 0ECE-0ECF 0ED3-0ED4 0ED6 0EDB "
                "0EE0 0EE5 0EEA 0EEF 0EF4 0EF9 0EFE 0F04 0F09 0F0E "
                "0F13 0F18 0F1D 0F21-0F22 0F26-0F27 0F2B-0F2C 0F30-0F31 0F35-0F36 0F3A-0F3B 0F3F-0F40 "
                "0F44-0F45 0F49-0F4A 0F4E-0F4F 0F53-0F54 0F56 0F5B 0F60 0F65 0F6A 0F6F "
                "0F74 0F79 0F7E 0F84 0F89 0F8E 0F93 0F98 0F9D 0FA1-0FA2 "
                "0FA6-0FA7 0FAB-0FAC 0FB0-0FB1 0FB5-0FB6 0FBA-0FBB 0FBF-0FC0 0FC4-0FC5 0FC9-0FCA 0FCE-0FCF 0FD3-0FD4 "
                "0FD6 0FDB 0FE0 0FE5 0FEA 0FEF 0FF4 0FF9 0FFE 1004 "
                "1009 100E 1013 1018 101D 1021-1022 1026-1027 102B-102C 1030-1031 1035-1036 "
                "103A-103B 103F-1040 1044-1045 1049-104A 104E-104F 1053-1054 1056 105B 1060 1065 "
                "106A 106F 1074 1079 107E 1084 1089 108E 1093 1098 "
                "109D 10A1-10A2 10A6-10A7 10AB-10AC 10B0-10B1 10B5-10B6 10BA-10BB 10BF-10C0 10C4-10C5 10C9-10CA "
                "10CE-10CF 10D3-10D4 10D6 10DB 10E0 10E5 10EA 10EF 10F4 10F9 "
                "10FE 1104 1109 110E 1113 1118 111D 1121-1122 1126-1127 112B-112C "
                "1130-1131 1135-1136 113A-113B 113F-1140 1144-1145 1149-114A 114E-114F 1153-1154 1156 115B "
                "1160 1165 116A 116F 1174 1179 117E 1184 1189 118E "
                "1193 1198 119D 11A1-11A2 11A6-11A7 11AB-11AC 11B0-11B1 11B5-11B6 11BA-11BB 11BF-11C0 "
                "11C4-11C5 11C9-11CA 11CE-11CF 11D3-11D4 11D6 11DB 11E0 11E5 11EA 11EF "
                "11F4 11F9 11FE 1204 1209 120E 1213 1218 121D 1221-1222 "
                "1226-1227 122B-122C 1230-1231 1235-1236 123A-123B 123F-1240 1244-1245 1249-124A 124E-124F 1253-1254 "
                "1256 125B 1260 1265 126A 126F 1274 1279 127E 1284 "
                "1289 128E 1293 1298 129D 12A1-12A2 12A6-12A7 12AB-12AC 12B0-12B1 12B5-12B6 "
                "12BA-12BB 12BF-12C0 12C4-12C5 12C9-12CA 12CE-12CF 12D3-12D4 12D6 12DB 12E0 12E5 "
                "12EA 12EF 12F4 12F9 12FE 1304 1309 130E 1313 1318 "
                "131D 1321-1322 1326-1327 132B-132C 1330-1331 1335-1336 133A-133B 133F-1340 1344-1345 1349-134A "
                "134E-134F 1353-1354 1356 135B 1360 1365 136A 136F 1374 1379 "
                "137E 1384 1389 138E 1393 1398 139D 13A1-13A2 13A6-13A7 13AB-13AC "
                "13B0-13B1 13B5-13B6 13BA-13BB 13BF-13C0 13C4-13C5 13C9-13CA 13CE-13CF 13D3-13D4 13D6 13DB "
                "13E0 13E5 13EA 13EF 13F4 13F9 13FE 1404 1409 140E "
                "1413 1418 141D 1421-1422 1426-1427 142B-142C 1430-1431 1435-1436 143A-143B 143F-1440 "
                "1444-1445 1449-144A 144E-144F 1453-1454 1456 145B 1460 1465 146A 146F "
                "1474 1479 147E 1484 1489 148E 1493 1498 149D 14A1-14A2 "
                "14A6-14A7 14AB-14AC 14B0-14B1 14B5-14B6 14BA-14BB 14BF-14C0 14C4-14C5 14C9-14CA 14CE-14CF 14D3-14D4 "
                "14D6 14DB 14E0 14E5 14EA 14EF 14F4 14F9 14FE 1504 "
                "1509 150E 1513 1518 151D 1521-1522 1526-1527 152B-152C 1530-1531 1535-1536 "
                "153A-153B 153F-1540 1544-1545 1549-154A 154E-154F 1553-1554 1556 155B 1560 1565 "
                "156A 156F 1574 1579 157E 1584 1589 158E 1593 1598 "
                "159D 15A1-15A2 15A6-15A7 15AB-15AC 15B0-15B1 15B5-15B6 15BA-15BB 15BF-15C0 15C4-15C5 15C9-15CA "
                "15CE-15CF 15D3-15D4 15D6 15DB 15E0 15E5 15EA 15EF 15F4 15F9 "
                "15FE 1604 1609 160E 1613 1618 161D 1621-1622 1626-1627 162B-162C "
                "1630-1631 1635-1636 163A-163B 163F-1640 1644-1645 1649-164A 164E-164F 1653-1654 1656 165B "
                "1660 1665 166A 166F 1674 1679 167E 1684 1689 168E "
                "1693 1698 169D 16A1-16A2 16A6-16A7 16AB-16AC 16B0-16B1 16B5-16B6 16BA-16BB 16BF-16C0 "
                "16C4-16C5 16C9-16CA 16CE-16CF 16D3-16D4 16D6 16DB 16E0 16E5 16EA 16EF "
                "16F4 16F9 16FE 1704 1709 170E 1713 1718 171D 1721-1722 "
                "1726-1727 172B-172C 1730-1731 1735-1736 173A-173B 173F-1740 1744-1745 1749-174A 174E-174F 1753-1754 "
                "1756 175B 1760 1765 176A 176F 1774 1779 177E 1784 "
                "1789 178E 1793 1798 179D 17A1-17A2 17A6-17A7 17AB-17AC 17B0-17B1 17B5-17B6 "
                "17BA-17BB 17BF-17C0 17C4-17C5 17C9-17CA 17CE-17CF 17D3-17D4 17D6 17DB 17E0 17E5 "
                "17EA 17EF 17F4 17F9 17FE 1804 1809 180E 1813 1818 "
                "181D 1821-1822 1826-1827 182B-182C 1830-1831 1835-1836 183A-183B 183F-1840 1844-1845 1849-184A "
                "184E-184F 1853-1854 1856 185B 1860 1865 186A 186F 1874 1879 "
                "187E 1884 1889 188E 1893 1898 189D 18A1-18A2 18A6-18A7 18AB-18AC "
                "18B0-18B1 18B5-18B6 18BA-18BB 18BF-18C0 18C4-18C5 18C9-18CA 18CE-18CF 18D3-18D4 18D6 18DB "
                "18E0 18E5 18EA 18EF 18F4 18F9 18FE 1904 1909 190E "
                "1913 1918 191D 1921-1922 1926-1927 192B-192C 1930-1931 1935-1936 193A-193B 193F-1940 "
                "1944-1945 1949-194A 194E-194F 1953-1954 1956 195B 1960 1965 196A 196F "
                "1974 1979 197E 1984 1989 198E 1993 1998 199D 19A1-19A2 "
                "19A6-19A7 19AB-19AC 19B0-19B1 19B5-19B6 19BA-19BB 19BF-19C0 19C4-19C5 19C9-19CA 19CE-19CF 19D3-19D4 "
                "19D6 19DB 19E0 19E5 19EA 19EF 19F4 19F9 19FE 1A04 "
                "1A09 1A0E 1A13 1A18 1A1D 1A21-1A22 1A26-1A27 1A2B-1A2C 1A30-1A31 1A35-1A36 "
                "1A3A-1A3B 1A3F-1A40 1A44-1A45 1A49-1A4A 1A4E-1A4F 1A53-1A54 1A56 1A5B 1A60 1A65 "
                "1A6A 1A6F 1A74 1A79 1A7E 1A84 1A89 1A8E 1A93 1A98 "
                "1A9D 1AA1-1AA2 1AA6-1AA7 1AAB-1AAC 1AB0-1AB1 1AB5-1AB6 1ABA-1ABB 1ABF-1AC0 1AC4-1AC5 1AC9-1ACA "
                "1ACE-1ACF 1AD3-1AD4 1AD6 1ADB 1AE0 1AE5 1AEA 1AEF 1AF4 1AF9 "
                "1AFE 1B04 1B09 1B0E 1B13 1B18 1B1D 1B21-1B22 1B26-1B27 1B2B-1B2C "
                "1B30-1B31 1B35-1B36 1B3A-1B3B 1B3F-1B40 1B44-1B45 1B49-1B4A 1B4E-1B4F 1B53-1B54 1B56 1B5B "
                "1B60 1B65 1B6A 1B6F 1B74 1B79 1B7E 1B84 1B89 1B8E "
                "1B93 1B98 1B9D 1BA1-1BA2 1BA6-1BA7 1BAB-1BAC 1BB0-1BB1 1BB5-1BB6 1BBA-1BBB 1BBF-1BC0 "
                "1BC4-1BC5 1BC9-1BCA 1BCE-1BCF 1BD3-1BD4 1BD6 1BDB 1BE0 1BE5 1BEA 1BEF "
                "1BF4 1BF9 1BFE 1C04 1C09 1C0E 1C13 1C18 1C1D 1C21-1C22 "
                "1C26-1C27 1C2B-1C2C 1C30-1C31 1C35-1C36 1C3A-1C3B 1C3F-1C40 1C44-1C45 1C49-1C4A 1C4E-1C4F 1C53-1C54 "
                "1C56 1C5B 1C60 1C65 1C6A 1C6F 1C74 1C79 1C7E 1C84 "
                "1C89 1C8E 1C93 1C98 1C9D 1CA1-1CA2 1CA6-1CA7 1CAB-1CAC 1CB0-1CB1 1CB5-1CB6 "
                "1CBA-1CBB 1CBF-1CC0 1CC4-1CC5 1CC9-1CCA 1CCE-1CCF 1CD3-1CD4 1CD6 1CDB 1CE0 1CE5 "
                "1CEA 1CEF 1CF4 1CF9 1CFE 1D04 1D09 1D0E 1D13 1D18 "
                "1D1D 1D21-1D22 1D26-1D27 1D2B-1D2C 1D30-1D31 1D35-1D36 1D3A-1D3B 1D3F-1D40 1D44-1D45 1D49-1D4A "
                "1D4E-1D4F 1D53-1D54 1D56 1D5B 1D60 1D65 1D6A 1D6F 1D74 1D79 "
                "1D7E 1D84 1D89 1D8E 1D93 1D98 1D9D 1DA1-1DA2 1DA6-1DA7 1DAB-1DAC "
                "1DB0-1DB1 1DB5-1DB6 1DBA-1DBB 1DBF-1DC0 1DC4-1DC5 1DC9-1DCA 1DCE-1DCF 1DD3-1DD4 1DD6 1DDB "
                "1DE0 1DE5 1DEA 1DEF 1DF4 1DF9 1DFE 1E04 1E09 1E0E "
                "1E13 1E18 1E1D 1E21-1E22 1E26-1E27 1E2B-1E2C 1E30-1E31 1E35-1E36 1E3A-1E3B 1E3F-1E40 "
                "1E44-1E45 1E49-1E4A 1E4E-1E4F 1E53-1E54 1E56 1E5B 1E60 1E65 1E6A 1E6F "
                "1E74 1E79 1E7E 1E84 1E89 1E8E 1E93 1E98 1E9D 1EA1-1EA2 "
                "1EA6-1EA7 1EAB-1EAC 1EB0-1EB1 1EB5-1EB6 1EBA-1EBB 1EBF-1EC0 1EC4-1EC5 1EC9-1ECA 1ECE-1ECF 1ED3-1ED4 "
                "1ED6 1EDB 1EE0 1EE5 1EEA 1EEF 1EF4 1EF9 1EFE 1F04 "
                "1F09 1F0E 1F13 1F18 1F1D 1F21-1F22 1F26-1F27 1F2B-1F2C 1F30-1F31 1F35-1F36 "
                "1F3A-1F3B 1F3F-1F40 1F44-1F45 1F49-1F4A 1F4E-1F4F 1F53-1F54 1F56 1F5B 1F60 1F65 "
                "1F6A 1F6F 1F74 1F79 1F7E 1F84 1F89 1F8E 1F93 1F98 "
                "1F9D 1FA1-1FA2 1FA6-1FA7 1FAB-1FAC 1FB0-1FB1 1FB5-1FB6 1FBA-1FBB 1FBF-1FC0 1FC4-1FC5 1FC9-1FCA "
                "1FCE-1FCF 1FD3-1FD4 1FD6 1FDB 1FE0 1FE5 1FEA 1FEF 1FF4 1FF9 "
                "1FFE 2004 2009 200E 2013 2018 201D 2021-2022 2026-2027 202B-202C "
                "2030-2031 2035-2036 203A-203B 203F-2040 2044-2045 2049-204A 204E-204F 2053-2054 2056 205B "
                "2060 2065 206A 206F 2074 2079 207E 2084 2089 208E "
                "2093 2098 209D 20A1-20A2 20A6-20A7 20AB-20AC 20B0-20B1 20B5-20B6 20BA-20BB 20BF-20C0 "
                "20C4-20C5 20C9-20CA 20CE-20CF 20D3-20D4 20D6 20DB 20E0 20E5 20EA 20EF "
                "20F4 20F9 20FE 2104 2109 210E 2113 2118 211D 2121-2122 "
                "2126-2127 212B-212C 2130-2131 2135-2136 213A-213B 213F-2140 2144-2145 2149-214A 214E-214F 2153-2154 "
                "2156 215B 2160 2165 216A 216F 2174 2179 217E 2184 "
                "2189 218E 2193 2198 219D 21A1-21A2 21A6-21A7 21AB-21AC 21B0-21B1 21B5-21B6 "
                "21BA-21BB 21BF-21C0 21C4-21C5 21C9-21CA 21CE-21CF 21D3-21D4 21D6 21DB 21E0 21E5 "
                "21EA 21EF 21F4 21F9 21FE 2204 2209 220E 2213 2218 "
                "221D 2221-2222 2226-2227 222B-222C 2230-2231 2235-2236 223A-223B 223F-2240 2244-2245 2249-224A "
                "224E-224F 2253-2254 2256 225B 2260 2265 226A 226F 2274 2279 "
                "227E 2284 2289 228E 2293 2298 229D 22A1-22A2 22A6-22A7 22AB-22AC "
                "22B0-22B1 22B5-22B6 22BA-22BB 22BF-22C0 22C4-22C5 22C9-22CA 22CE-22CF 22D3-22D4 22D6 22DB "
                "22E0 22E5 22EA 22EF 22F4 22F9 22FE 2304 2309 230E "
                "2313 2318 231D 2321-2322 2326-2327 232B-232C 2330-2331 2335-2336 233A-233B 233F-2340 "
                "2344-2345 2349-234A 234E-234F 2353-2354 2356 235B 2360 2365 236A 236F "
                "2374 2379 237E 2384 2389 238E 2393 2398 239D 23A1-23A2 "
                "23A6-23A7 23AB-23AC 23B0-23B1 23B5-23B6 23BA-23BB 23BF-23C0 23C4-23C5 23C9-23CA 23CE-23CF 23D3-23D4 "
                "23D6 23DB 23E0 23E5 23EA 23EF 23F4 23F9 23FE 2404 "
                "2409 240E 2413 2418 241D 2421-2422 2426-2427 242B-242C 2430-2431 2435-2436 "
                "243A-243B 243F-2440 2444-2445 2449-244A 244E-244F 2453-2454 2456 245B 2460 2465 "
                "246A 246F 2474 2479 247E 2484 2489 248E 2493 2498 "
                "249D 24A1-24A2 24A6-24A7 24AB-24AC 24B0-24B1 24B5-24B6 24BA-24BB 24BF-24C0 24C4-24C5 24C9-24CA "
                "24CE-24CF 24D3-24D4 24D6 24DB 24E0 24E5 24EA 24EF 24F4 24F9 "
                "24FE 2504 2509 250E 2513 2518 251D 2521-2522 2526-2527 252B-252C "
                "2530-2531 2535-2536 253A-253B 253F-2540 2544-2545 2549-254A 254E-254F 2553-2554 2556 255B "
                "2560 2565 256A 256F 2574 2579 257E 2584 2589 258E "
                "2593 2598 259D 25A1-25A2 25A6-25A7 25AB-25AC 25B0-25B1 25B5-25B6 25BA-25BB 25BF-25C0 "
                "25C4-25C5 25C9-25CA 25CE-25CF 25D3-25D4 25D6 25DB 25E0 25E5 25EA 25EF "
                "25F4 25F9 25FE 2604 2609 260E 2613 2618 261D 2621-2622 "
                "2626-2627 262B-262C 2630-2631 2635-2636 263A-263B 263F-2640 2644-2645 2649-264A 264E-264F 2653-2654 "
                "2656 265B 2660 2665 266A 266F 2674 2679 267E 2684 "
                "2689 268E 2693 2698 269D 26A1-26A2 26A6-26A7 26AB-26AC 26B0-26B1 26B5-26B6 "
                "26BA-26BB 26BF-26C0 26C4-26C5 26C9-26CA 26CE-26CF 26D3-26D4 26D6 26DB 26E0 26E5 "
                "26EA 26EF 26F4 26F9 26FE 2704 2709 270E 2713 2718 "
                "271D 2721-2722 2726-2727 272B-272C 2730-2731 2735-2736 273A-273B 273F-2740 2744-2745 2749-274A "
                "274E-274F 2753-2754 2756 275B 2760 2765 276A 276F 2774 2779 "
                "277E 2784 2789 278E 2793 2798 279D 27A1-27A2 27A6-27A7 27AB-27AC "
                "27B0-27B1 27B5-27B6 27BA-27BB 27BF-27C0 27C4-27C5 27C9-27CA 27CE-27CF 27D3-27D4 27D6 27DB "
                "27E0 27E5 27EA 27EF 27F4 27F9 27FE 2804 2809 280E "
                "2813 2818 281D 2821-2822 2826-2827 282B-282C 2830-2831 2835-2836 283A-283B 283F-2840 "
                "2844-2845 2849-284A 284E-284F 2853-2854 2856 285B 2860 2865 286A 286F "
                "2874 2879 287E 2884 2889 288E 2893 2898 289D 28A1-28A2 "
                "28A6-28A7 28AB-28AC 28B0-28B1 28B5-28B6 28BA-28BB 28BF-28C0 28C4-28C5 28C9-28CA 28CE-28CF 28D3-28D4 "
                "28D6 28DB 28E0 28E5 28EA 28EF 28F4 28F9 28FE 2904 "
                "2909 290E 2913 2918 291D 2921-2922 2926-2927 292B-292C 2930-2931 2935-2936 "
                "293A-293B 293F-2940 2944-2945 2949-294A 294E-294F 2953-2954 2956 295B 2960 2965 "
                "296A 296F 2974 2979 297E 2984 2989 298E 2993 2998 "
                "299D 29A1-29A2 29A6-29A7 29AB-29AC 29B0-29B1 29B5-29B6 29BA-29BB 29BF-29C0 29C4-29C5 29C9-29CA "
                "29CE-29CF 29D3-29D4 29D6 29DB 29E0 29E5 29EA 29EF 29F4 29F9 "
                "29FE 2A04 2A09 2A0E 2A13 2A18 2A1D 2A21-2A22 2A26-2A27 2A2B-2A2C "
                "2A30-2A31 2A35-2A36 2A3A-2A3B 2A3F-2A40 2A44-2A45 2A49-2A4A 2A4E-2A4F 2A53-2A54 2A56 2A5B "
                "2A60 2A65 2A6A 2A6F 2A74 2A79 2A7E 2A84 2A89 2A8E "
                "2A93 2A98 2A9D 2AA1-2AA2 2AA6-2AA7 2AAB-2AAC 2AB0-2AB1 2AB5-2AB6 2ABA-2ABB 2ABF-2AC0 "
                "2AC4-2AC5 2AC9-2ACA 2ACE-2ACF 2AD3-2AD4 2AD6 2ADB 2AE0 2AE5 2AEA 2AEF "
                "2AF4 2AF9 2AFE 2B04 2B09 2B0E 2B13 2B18 2B1D 2B21-2B22 "
                "2B26-2B27 2B2B-2B2C 2B30-2B31 2B35-2B36 2B3A-2B3B 2B3F-2B40 2B44-2B45 2B49-2B4A 2B4E-2B4F 2B53-2B54 "
                "2B56 2B5B 2B60 2B65 2B6A 2B6F 2B74 2B79 2B7E 2B84 "
                "2B89 2B8E 2B93 2B98 2B9D 2BA1-2BA2 2BA6-2BA7 2BAB-2BAC 2BB0-2BB1 2BB5-2BB6 "
                "2BBA-2BBB 2BBF-2BC0 2BC4-2BC5 2BC9-2BCA 2BCE-2BCF 2BD3-2BD4 2BD6 2BDB 2BE0 2BE5 "
                "2BEA 2BEF 2BF4 2BF9 2BFE 2C04 2C09 2C0E 2C13 2C18 "
                "2C1D 2C21-2C22 2C26-2C27 2C2B-2C2C 2C30-2C31 2C35-2C36 2C3A-2C3B 2C3F-2C40 2C44-2C45 2C49-2C4A "
                "2C4E-2C4F 2C53-2C54 2C56 2C5B 2C60 2C65 2C6A 2C6F 2C74 2C79 "
                "2C7E 2C84 2C89 2C8E 2C93 2C98 2C9D 2CA1-2CA2 2CA6-2CA7 2CAB-2CAC "
                "2CB0-2CB1 2CB5-2CB6 2CBA-2CBB 2CBF-2CC0 2CC4-2CC5 2CC9-2CCA 2CCE-2CCF 2CD3-2CD4 2CD6 2CDB "
                "2CE0 2CE5 2CEA 2CEF 2CF4 2CF9 2CFE 2D04 2D09 2D0E "
                "2D13 2D18 2D1D 2D21-2D22 2D26-2D27 2D2B-2D2C 2D30-2D31 2D35-2D36 2D3A-2D3B 2D3F-2D40 "
                "2D44-2D45 2D49-2D4A 2D4E-2D4F 2D53-2D54 2D56 2D5B 2D60 2D65 2D6A 2D6F "
                "2D74 2D79 2D7E 2D84 2D89 2D8E 2D93 2D98 2D9D 2DA1-2DA2 "
                "2DA6-2DA7 2DAB-2DAC 2DB0-2DB1 2DB5-2DB6 2DBA-2DBB 2DBF-2DC0 2DC4-2DC5 2DC9-2DCA 2DCE-2DCF 2DD3-2DD4 "
                "2DD6 2DDB 2DE0 2DE5 2DEA 2DEF 2DF4 2DF9 2DFE 2E04 "
                "2E09 2E0E 2E13 2E18 2E1D 2E21-2E22 2E26-2E27 2E2B-2E2C 2E30-2E31 2E35-2E36 "
                "2E3A-2E3B 2E3F-2E40 2E44-2E45 2E49-2E4A 2E4E-2E4F 2E53-2E54 2E56 2E5B 2E60 2E65 "
                "2E6A 2E6F 2E74 2E79 2E7E 2E84 2E89 2E8E 2E93 2E98 "
                "2E9D 2EA1-2EA2 2EA6-2EA7 2EAB-2EAC 2EB0-2EB1 2EB5-2EB6 2EBA-2EBB 2EBF-2EC0 2EC4-2EC5 2EC9-2ECA "
                "2ECE-2ECF 2ED3-2ED4 2ED6 2EDB 2EE0 2EE5 2EEA 2EEF 2EF4 2EF9 "
                "2EFE 2F04 2F09 2F0E 2F13 2F18 2F1D 2F21-2F22 2F26-2F27 2F2B-2F2C "
                "2F30-2F31 2F35-2F36 2F3A-2F3B 2F3F-2F40 2F44-2F45 2F49-2F4A 2F4E-2F4F 2F53-2F54 2F56 2F5B "
                "2F60 2F65 2F6A 2F6F 2F74 2F79 2F7E 2F84 2F89 2F8E "
                "2F93 2F98 2F9D 2FA1-2FA2 2FA6-2FA7 2FAB-2FAC 2FB0-2FB1 2FB5-2FB6 2FBA-2FBB 2FBF-2FC0 "
                "2FC4-2FC5 2FC9-2FCA 2FCE-2FCF 2FD3-2FD4 2FD6 2FDB 2FE0 2FE5 2FEA 2FEF "
                "2FF4 2FF9 2FFE 3004 3009 300E 3013 3018 301D 3021-3022 "
                "3026-3027 302B-302C 3030-3031 3035-3036 303A-303B 303F-3040 3044-3045 3049-304A 304E-304F 3053-3054 "
                "3056 305B 3060 3065 306A 306F 3074 3079 307E 3084 "
                "3089 308E 3093 3098 309D 30A1-30A2 30A6-30A7 30AB-30AC 30B0-30B1 30B5-30B6 "
                "30BA-30BB 30BF-30C0 30C4-30C5 30C9-30CA 30CE-30CF 30D3-30D4 30D6 30DB 30E0 30E5 "
                "30EA 30EF 30F4 30F9 30FE 3104 3109 310E 3113 3118 "
                "311D 3121-3122 3126-3127 312B-312C 3130-3131 3135-3136 313A-313B 313F-3140 3144-3145 3149-314A "
                "314E-314F 3153-3154 3156 315B 3160 3165 316A 316F 3174 3179 "
                "317E 3184 3189 318E 3193 3198 319D 31A1-31A2 31A6-31A7 31AB-31AC "
                "31B0-31B1 31B5-31B6 31BA-31BB 31BF-31C0 31C4-31C5 31C9-31CA 31CE-31CF 31D3-31D4 31D6 31DB "
                "31E0 31E5 31EA 31EF 31F4 31F9 31FE 3204 3209 320E "
                "3213 3218 321D 3221-3222 3226-3227 322B-322C 3230-3231 3235-3236 323A-323B 323F-3240 "
                "3244-3245 3249-324A 324E-324F 3253-3254 3256 325B 3260 3265 326A 326F "
                "3274 3279 327E 3284 3289 328E 3293 3298 329D 32A1-32A2 "
                "32A6-32A7 32AB-32AC 32B0-32B1 32B5-32B6 32BA-32BB 32BF-32C0 32C4-32C5 32C9-32CA 32CE-32CF 32D3-32D4 "
                "32D6 32DB 32E0 32E5 32EA 32EF 32F4 32F9 32FE 3304 "
                "3309 330E 3313 3318 331D 3321-3322 3326-3327 332B-332C 3330-3331 3335-3336 "
                "333A-333B 333F-3340 3344-3345 3349-334A 334E-334F 3353-3354 3356 335B 3360 3365 "
                "336A 336F 3374 3379 337E 3384 3389 338E 3393 3398 "
                "339D 33A1-33A2 33A6-33A7 33AB-33AC 33B0-33B1 33B5-33B6 33BA-33BB 33BF-33C0 33C4-33C5 33C9-33CA "
                "33CE-33CF 33D3-33D4 33D6 33DB 33E0 33E5 33EA 33EF 33F4 33F9 "
                "33FE 3404 3409 340E 3413 3418 341D 3421-3422 3426-3427 342B-342C "
                "3430-3431 3435-3436 343A-343B 343F-3440 3444-3445 3449-344A 344E-344F 3453-3454 3456 345B "
                "3460 3465 346A 346F 3474 3479 347E 3484 3489 348E "
                "3493 3498 349D 34A1-34A2 34A6-34A7 34AB-34AC 34B0-34B1 34B5-34B6 34BA-34BB 34BF-34C0 "
                "34C4-34C5 34C9-34CA 34CE-34CF 34D3-34D4 34D6 34DB 34E0 34E5 34EA 34EF "
                "34F4 34F9 34FE 3504 3509 350E 3513 3518 351D 3521-3522 "
                "3526-3527 352B-352C 3530-3531 3535-3536 353A-353B 353F-3540 3544-3545 3549-354A 354E-354F 3553-3554 "
                "3556 355B 3560 3565 356A 356F 3574 3579 357E 3584 "
                "3589 358E 3593 3598 359D 35A1-35A2 35A6-35A7 35AB-35AC 35B0-35B1 35B5-35B6 "
                "35BA-35BB 35BF-35C0 35C4-35C5 35C9-35CA 35CE-35CF 35D3-35D4 35D6 35DB 35E0 35E5 "
                "35EA 35EF 35F4 35F9 35FE 3604 3609 360E 3613 3618 "
                "361D 3621-3622 3626-3627 362B-362C 3630-3631 3635-3636 363A-363B 363F-3640 3644-3645 3649-364A "
                "364E-364F 3653-3654 3656 365B 3660 3665 366A 366F 3674 3679 "
                "367E 3684 3689 368E 3693 3698 369D 36A1-36A2 36A6-36A7 36AB-36AC "
                "36B0-36B1 36B5-36B6 36BA-36BB 36BF-36C0 36C4-36C5 36C9-36CA 36CE-36CF 36D3-36D4 36D6 36DB "
                "36E0 36E5 36EA 36EF 36F4 36F9 36FE 3704 3709 370E "
                "3713 3718 371D 3721-3722 3726-3727 372B-372C 3730-3731 3735-3736 373A-373B 373F-3740 "
                "3744-3745 3749-374A 374E-374F 3753-3754 3756 375B 3760 3765 376A 376F "
                "3774 3779 377E 3784 3789 378E 3793 3798 379D 37A1-37A2 "
                "37A6-37A7 37AB-37AC 37B0-37B1 37B5-37B6 37BA-37BB 37BF-37C0 37C4-37C5 37C9-37CA 37CE-37CF 37D3-37D4 "
                "37D6 37DB 37E0 37E5 37EA 37EF 37F4 37F9 37FE 3804 "
                "3809 380E 3813 3818 381C-381D 3821-3822 3826-3827 382B-382C 3830-3831 3835-3836 "
                "383A-383B 383F-3840 3844-3845 3849-384A 384E-384F 3853-3854 3856 385B 3860 3865 "
                "386A 386F 3874 3879 387E 3884 3889 388E 3893 3898 "
                "389C-389D 38A1-38A2 38A6-38A7 38AB-38AC 38B0-38B1 38B5-38B6 38BA-38BB 38BF-38C0 38C4-38C5 38C9-38CA "
                "38CE-38CF 38D3-38D4 38D6 38DB 38E0 38E5 38EA 38EF 38F4 38F9 "
                "38FE 3904 3909 390E 3913 3917-3918 391C-391D 3921-3922 3926-3927 392B-392C "
                "3930-3931 3935-3936 393A-393B 393F-3940 3944-3945 3949-394A 394E-394F 3953-3954 3956 395B "
                "3960 3965 396A 396F 3974 3979 397E 3984 3989 398E "
                "3992-3993 3997-3998 399C-399D 39A1-39A2 39A6-39A7 39AB-39AC 39B0-39B1 39B5-39B6 39BA-39BB 39BF-39C0 "
                "39C4-39C5 39C9-39CA 39CE-39CF 39D3-39D4 39D6 39DB 39E0 39E5 39EA 39EF "
                "39F4 39F9 39FE 3A04 3A08-3A09 3A0D-3A0E 3A12-3A13 3A17-3A18 3A1C-3A1D 3A21-3A22 "
                "3A26-3A27 3A2B-3A2C 3A30-3A31 3A35-3A36 3A3A-3A3B 3A3F-3A40 3A44-3A45 3A49-3A4A 3A4E 3A52-3A53 "
                "3A56 3A5B 3A60 3A65 3A6A 3A6F 3A74 3A77 3A79 3A7C "
                "3A7E 3A83-3A84 3A88-3A89 3A8D-3A8E 3A92 3A97 3A9C 3AA1 3AA6 3AAB "
                "3AB0 3AB5 3AB9-3ABA 3ABE-3ABF 3AC3-3AC4 3AC8-3AC9 3ACD-3ACE 3AD2-3AD3 3AD9 3ADE "
                "3AE3 3AE8 3AED 3AF2 3AF7 3AFC 3B03 3B08 3B0D 3B12 "
                "3B17 3B1B-3B1C 3B20-3B21 3B25-3B26 3B2A-3B2B 3B2F 3B34 3B39 3B3E 3B42-3B43 "
                "3B47-3B48 3B4C-3B4D 3B51-3B52 3B59 3B5E 3B63 3B68 3B6D 3B72 3B75 "
                "3B77 3B7A 3B7F 3B82 3B87 3B8C 3B91 3B96 3B9A-3B9B 3B9F-3BA0 "
                "3BA4 3BA9 3BAE 3BB2-3BB3 3BB7-3BB8 3BBC-3BBD 3BC1 3BC6 3BCA-3BCB 3BCF-3BD0 "
                "3BD4 3BD7 3BDC 3BE1 3BE9 3BEE 3BF3 3BF8 3BFD 3C01 "
                "3C05 3C0A 3C0F 3C14 3C18 3C1D 3C22 3C26-3C27 3C2B 3C30 "
                "3C34-3C35 3C39 3C3E 3C42-3C43 3C47 3C4B-3C4C 3C50 3C60 3C65 3C6A "
                "3C72 3C77 3C7C 3C82 3C87 3C90 3C94 3C99 3C9D 3CA2 "
                "3CA6 3CAB 3CAF 3CB3 3CB8 3CBC 3CC0 3CC5 3CC9 3CCD "
                "3CD2 3CD8 3CE5 3CED 3CF2 3CFA 3D01 3D1A 3D1E 3D22 "
                "3D26 3D41 3D45 3D49 3D65 3D70 3E98 3E9F 3EAE 3EB3 "
                "3EBA 3EC5 3ECB 3ED8 3EDF 3EE6 3EED 3EF5 3EFD-3EFE 3F03 "
                "3F08 3F0D 3F13 3F1A 3F32-3F34 3F89 3F9B 3FB3 B9C1 B9C6 "
                "B9CB B9D0 B9D5 BA19 BA1E BA23 BA28 BA2D BA32 BA37 "
                "BA3C BA41 BA46 BA6C BA71 BA76 BA7B BA80 BA85 BA8A "
                "BA8F BB24 BB29 BB2E BB6E BB73 BB9B BBBA BBD4 BBE1 "
                "BEE8 BEF5 BEFA BF04 BF37 BFA4-BFA5 BFB3 BFF1 BFF6 BFFD "
                "C020 C024 C028 C02B C04D C04F C052 C054 C05A C05F "
                "C07C C080 C097-C099 C0AB C0AF C0B2 C0C5 C0DE C0EB C0EF "
                "C0F4 C102 C10E C11A C125 C130 C139 C13D C14F C151 "
                "C159 C169 C174 C17C C18F C1AA C1B7 C1BB C1C6 C1D1-C1D2 "
                "C1D7 C1EE C1F4 C1F7 C202 C20A C21B C223 C237 C244 "
                "C247 C255 C258 C273 C281 C286 C289 C29F C2A2 C2AD "
                "C2B4"
            ),
        },
        "wormhole_b0": {
            -1: (
                "3DD3 3DD6 3DD9 3DDC 3DDF 3E0D 3E15 3E3A 3E41 3F31 "
                "3F68 3FCA 80D6 80DB 80E0 80E5 80EA 80EF 80F4 80F9 "
                "80FE 8104 8109 810E 8113 8118 811D 8122 8127 812B-812C "
                "8130-8131 8135-8136 813A-813B 813F-8140 8144-8145 8149-814A 814E-814F 8153-8154 8156 815B "
                "8160 8165 816A 816F 8174 8179 817E 8184 8189 818E "
                "8193 8198 819D 81A2 81A7 81AB-81AC 81B0-81B1 81B5-81B6 81BA-81BB 81BF-81C0 "
                "81C4-81C5 81C9-81CA 81CE-81CF 81D3-81D4 81D6 81DB 81E0 81E5 81EA 81EF "
                "81F4 81F9 81FE 8204 8209 820E 8213 8218 821D 8222 "
                "8227 822B-822C 8230-8231 8235-8236 823A-823B 823F-8240 8244-8245 8249-824A 824E-824F 8253-8254 "
                "8256 825B 8260 8265 826A 826F 8274 8279 827E 8284 "
                "8289 828E 8293 8298 829D 82A2 82A7 82AB-82AC 82B0-82B1 82B5-82B6 "
                "82BA-82BB 82BF-82C0 82C4-82C5 82C9-82CA 82CE-82CF 82D3-82D4 82D6 82DB 82E0 82E5 "
                "82EA 82EF 82F4 82F9 82FE 8304 8309 830E 8313 8318 "
                "831D 8322 8327 832B-832C 8330-8331 8335-8336 833A-833B 833F-8340 8344-8345 8349-834A "
                "834E-834F 8353-8354 8356 835B 8360 8365 836A 836F 8374 8379 "
                "837E 8384 8389 838E 8393 8398 839D 83A2 83A7 83AB-83AC "
                "83B0-83B1 83B5-83B6 83BA-83BB 83BF-83C0 83C4-83C5 83C9-83CA 83CE-83CF 83D3-83D4 83D6 83DB "
                "83E0 83E5 83EA 83EF 83F4 83F9 83FE 8404 8409 840E "
                "8413 8418 841D 8422 8427 842B-842C 8430-8431 8435-8436 843A-843B 843F-8440 "
                "8444-8445 8449-844A 844E-844F 8453-8454 8456 845B 8460 8465 846A 846F "
                "8474 8479 847E 8484 8489 848E 8493 8498 849D 84A2 "
                "84A7 84AB-84AC 84B0-84B1 84B5-84B6 84BA-84BB 84BF-84C0 84C4-84C5 84C9-84CA 84CE-84CF 84D3-84D4 "
                "84D6 84DB 84E0 84E5 84EA 84EF 84F4 84F9 84FE 8504 "
                "8509 850E 8513 8518 851D 8522 8527 852B-852C 8530-8531 8535-8536 "
                "853A-853B 853F-8540 8544-8545 8549-854A 854E-854F 8553-8554 8556 855B 8560 8565 "
                "856A 856F 8574 8579 857E 8584 8589 858E 8593 8598 "
                "859D 85A2 85A7 85AB-85AC 85B0-85B1 85B5-85B6 85BA-85BB 85BF-85C0 85C4-85C5 85C9-85CA "
                "85CE-85CF 85D3-85D4 85D6 85DB 85E0 85E5 85EA 85EF 85F4 85F9 "
                "85FE 8604 8609 860E 8613 8618 861D 8622 8627 862B-862C "
                "8630-8631 8635-8636 863A-863B 863F-8640 8644-8645 8649-864A 864E-864F 8653-8654 8656 865B "
                "8660 8665 866A 866F 8674 8679 867E 8684 8689 868E "
                "8693 8698 869D 86A2 86A7 86AB-86AC 86B0-86B1 86B5-86B6 86BA-86BB 86BF-86C0 "
                "86C4-86C5 86C9-86CA 86CE-86CF 86D3-86D4 86D6 86DB 86E0 86E5 86EA 86EF "
                "86F4 86F9 86FE 8704 8709 870E 8713 8718 871D 8722 "
                "8727 872B-872C 8730-8731 8735-8736 873A-873B 873F-8740 8744-8745 8749-874A 874E-874F 8753-8754 "
                "8756 875B 8760 8765 876A 876F 8774 8779 877E 8784 "
                "8789 878E 8793 8798 879D 87A2 87A7 87AB-87AC 87B0-87B1 87B5-87B6 "
                "87BA-87BB 87BF-87C0 87C4-87C5 87C9-87CA 87CE-87CF 87D3-87D4 87D6 87DB 87E0 87E5 "
                "87EA 87EF 87F4 87F9 87FE 8804 8809 880E 8813 8818 "
                "881D 8822 8827 882B-882C 8830-8831 8835-8836 883A-883B 883F-8840 8844-8845 8849-884A "
                "884E-884F 8853-8854 8856 885B 8860 8865 886A 886F 8874 8879 "
                "887E 8884 8889 888E 8893 8898 889D 88A2 88A7 88AB-88AC "
                "88B0-88B1 88B5-88B6 88BA-88BB 88BF-88C0 88C4-88C5 88C9-88CA 88CE-88CF 88D3-88D4 88D6 88DB "
                "88E0 88E5 88EA 88EF 88F4 88F9 88FE 8904 8909 890E "
                "8913 8918 891D 8922 8927 892B-892C 8930-8931 8935-8936 893A-893B 893F-8940 "
                "8944-8945 8949-894A 894E-894F 8953-8954 8956 895B 8960 8965 896A 896F "
                "8974 8979 897E 8984 8989 898E 8993 8998 899D 89A2 "
                "89A7 89AB-89AC 89B0-89B1 89B5-89B6 89BA-89BB 89BF-89C0 89C4-89C5 89C9-89CA 89CE-89CF 89D3-89D4 "
                "89D6 89DB 89E0 89E5 89EA 89EF 89F4 89F9 89FE 8A04 "
                "8A09 8A0E 8A13 8A18 8A1D 8A22 8A27 8A2B-8A2C 8A30-8A31 8A35-8A36 "
                "8A3A-8A3B 8A3F-8A40 8A44-8A45 8A49-8A4A 8A4E-8A4F 8A53-8A54 8A56 8A5B 8A60 8A65 "
                "8A6A 8A6F 8A74 8A79 8A7E 8A84 8A89 8A8E 8A93 8A98 "
                "8A9D 8AA2 8AA7 8AAB-8AAC 8AB0-8AB1 8AB5-8AB6 8ABA-8ABB 8ABF-8AC0 8AC4-8AC5 8AC9-8ACA "
                "8ACE-8ACF 8AD3-8AD4 8AD6 8ADB 8AE0 8AE5 8AEA 8AEF 8AF4 8AF9 "
                "8AFE 8B04 8B09 8B0E 8B13 8B18 8B1D 8B22 8B27 8B2B-8B2C "
                "8B30-8B31 8B35-8B36 8B3A-8B3B 8B3F-8B40 8B44-8B45 8B49-8B4A 8B4E-8B4F 8B53-8B54 8B56 8B5B "
                "8B60 8B65 8B6A 8B6F 8B74 8B79 8B7E 8B84 8B89 8B8E "
                "8B93 8B98 8B9D 8BA2 8BA7 8BAB-8BAC 8BB0-8BB1 8BB5-8BB6 8BBA-8BBB 8BBF-8BC0 "
                "8BC4-8BC5 8BC9-8BCA 8BCE-8BCF 8BD3-8BD4 8BD6 8BDB 8BE0 8BE5 8BEA 8BEF "
                "8BF4 8BF9 8BFE 8C04 8C09 8C0E 8C13 8C18 8C1D 8C22 "
                "8C27 8C2B-8C2C 8C30-8C31 8C35-8C36 8C3A-8C3B 8C3F-8C40 8C44-8C45 8C49-8C4A 8C4E-8C4F 8C53-8C54 "
                "8C56 8C5B 8C60 8C65 8C6A 8C6F 8C74 8C79 8C7E 8C84 "
                "8C89 8C8E 8C93 8C98 8C9D 8CA2 8CA7 8CAB-8CAC 8CB0-8CB1 8CB5-8CB6 "
                "8CBA-8CBB 8CBF-8CC0 8CC4-8CC5 8CC9-8CCA 8CCE-8CCF 8CD3-8CD4 8CD6 8CDB 8CE0 8CE5 "
                "8CEA 8CEF 8CF4 8CF9 8CFE 8D04 8D09 8D0E 8D13 8D18 "
                "8D1D 8D22 8D27 8D2B-8D2C 8D30-8D31 8D35-8D36 8D3A-8D3B 8D3F-8D40 8D44-8D45 8D49-8D4A "
                "8D4E-8D4F 8D53-8D54 8D56 8D5B 8D60 8D65 8D6A 8D6F 8D74 8D79 "
                "8D7E 8D84 8D89 8D8E 8D93 8D98 8D9D 8DA2 8DA7 8DAB-8DAC "
                "8DB0-8DB1 8DB5-8DB6 8DBA-8DBB 8DBF-8DC0 8DC4-8DC5 8DC9-8DCA 8DCE-8DCF 8DD3-8DD4 8DD6 8DDB "
                "8DE0 8DE5 8DEA 8DEF 8DF4 8DF9 8DFE 8E04 8E09 8E0E "
                "8E13 8E18 8E1D 8E22 8E27 8E2B-8E2C 8E30-8E31 8E35-8E36 8E3A-8E3B 8E3F-8E40 "
                "8E44-8E45 8E49-8E4A 8E4E-8E4F 8E53-8E54 8E56 8E5B 8E60 8E65 8E6A 8E6F "
                "8E74 8E79 8E7E 8E84 8E89 8E8E 8E93 8E98 8E9D 8EA2 "
                "8EA7 8EAB-8EAC 8EB0-8EB1 8EB5-8EB6 8EBA-8EBB 8EBF-8EC0 8EC4-8EC5 8EC9-8ECA 8ECE-8ECF 8ED3-8ED4 "
                "8ED6 8EDB 8EE0 8EE5 8EEA 8EEF 8EF4 8EF9 8EFE 8F04 "
                "8F09 8F0E 8F13 8F18 8F1D 8F22 8F27 8F2B-8F2C 8F30-8F31 8F35-8F36 "
                "8F3A-8F3B 8F3F-8F40 8F44-8F45 8F49-8F4A 8F4E-8F4F 8F53-8F54 8F56 8F5B 8F60 8F65 "
                "8F6A 8F6F 8F74 8F79 8F7E 8F84 8F89 8F8E 8F93 8F98 "
                "8F9D 8FA2 8FA7 8FAB-8FAC 8FB0-8FB1 8FB5-8FB6 8FBA-8FBB 8FBF-8FC0 8FC4-8FC5 8FC9-8FCA "
                "8FCE-8FCF 8FD3-8FD4 8FD6 8FDB 8FE0 8FE5 8FEA 8FEF 8FF4 8FF9 "
                "8FFE 9004 9009 900E 9013 9018 901D 9022 9027 902B-902C "
                "9030-9031 9035-9036 903A-903B 903F-9040 9044-9045 9049-904A 904E-904F 9053-9054 9056 905B "
                "9060 9065 906A 906F 9074 9079 907E 9084 9089 908E "
                "9093 9098 909D 90A2 90A7 90AB-90AC 90B0-90B1 90B5-90B6 90BA-90BB 90BF-90C0 "
                "90C4-90C5 90C9-90CA 90CE-90CF 90D3-90D4 90D6 90DB 90E0 90E5 90EA 90EF "
                "90F4 90F9 90FE 9104 9109 910E 9113 9118 911D 9122 "
                "9127 912B-912C 9130-9131 9135-9136 913A-913B 913F-9140 9144-9145 9149-914A 914E-914F 9153-9154 "
                "9156 915B 9160 9165 916A 916F 9174 9179 917E 9184 "
                "9189 918E 9193 9198 919D 91A2 91A7 91AB-91AC 91B0-91B1 91B5-91B6 "
                "91BA-91BB 91BF-91C0 91C4-91C5 91C9-91CA 91CE-91CF 91D3-91D4 91D6 91DB 91E0 91E5 "
                "91EA 91EF 91F4 91F9 91FE 9204 9209 920E 9213 9218 "
                "921D 9222 9227 922B-922C 9230-9231 9235-9236 923A-923B 923F-9240 9244-9245 9249-924A "
                "924E-924F 9253-9254 9256 925B 9260 9265 926A 926F 9274 9279 "
                "927E 9284 9289 928E 9293 9298 929D 92A2 92A7 92AB-92AC "
                "92B0-92B1 92B5-92B6 92BA-92BB 92BF-92C0 92C4-92C5 92C9-92CA 92CE-92CF 92D3-92D4 92D6 92DB "
                "92E0 92E5 92EA 92EF 92F4 92F9 92FE 9304 9309 930E "
                "9313 9318 931D 9322 9327 932B-932C 9330-9331 9335-9336 933A-933B 933F-9340 "
                "9344-9345 9349-934A 934E-934F 9353-9354 9356 935B 9360 9365 936A 936F "
                "9374 9379 937E 9384 9389 938E 9393 9398 939D 93A2 "
                "93A7 93AB-93AC 93B0-93B1 93B5-93B6 93BA-93BB 93BF-93C0 93C4-93C5 93C9-93CA 93CE-93CF 93D3-93D4 "
                "93D6 93DB 93E0 93E5 93EA 93EF 93F4 93F9 93FE 9404 "
                "9409 940E 9413 9418 941D 9422 9427 942B-942C 9430-9431 9435-9436 "
                "943A-943B 943F-9440 9444-9445 9449-944A 944E-944F 9453-9454 9456 945B 9460 9465 "
                "946A 946F 9474 9479 947E 9484 9489 948E 9493 9498 "
                "949D 94A2 94A7 94AB-94AC 94B0-94B1 94B5-94B6 94BA-94BB 94BF-94C0 94C4-94C5 94C9-94CA "
                "94CE-94CF 94D3-94D4 94D6 94DB 94E0 94E5 94EA 94EF 94F4 94F9 "
                "94FE 9504 9509 950E 9513 9518 951D 9522 9527 952B-952C "
                "9530-9531 9535-9536 953A-953B 953F-9540 9544-9545 9549-954A 954E-954F 9553-9554 9556 955B "
                "9560 9565 956A 956F 9574 9579 957E 9584 9589 958E "
                "9593 9598 959D 95A2 95A7 95AB-95AC 95B0-95B1 95B5-95B6 95BA-95BB 95BF-95C0 "
                "95C4-95C5 95C9-95CA 95CE-95CF 95D3-95D4 95D6 95DB 95E0 95E5 95EA 95EF "
                "95F4 95F9 95FE 9604 9609 960E 9613 9618 961D 9622 "
                "9627 962B-962C 9630-9631 9635-9636 963A-963B 963F-9640 9644-9645 9649-964A 964E-964F 9653-9654 "
                "9656 965B 9660 9665 966A 966F 9674 9679 967E 9684 "
                "9689 968E 9693 9698 969D 96A2 96A7 96AB-96AC 96B0-96B1 96B5-96B6 "
                "96BA-96BB 96BF-96C0 96C4-96C5 96C9-96CA 96CE-96CF 96D3-96D4 96D6 96DB 96E0 96E5 "
                "96EA 96EF 96F4 96F9 96FE 9704 9709 970E 9713 9718 "
                "971D 9722 9727 972B-972C 9730-9731 9735-9736 973A-973B 973F-9740 9744-9745 9749-974A "
                "974E-974F 9753-9754 9756 975B 9760 9765 976A 976F 9774 9779 "
                "977E 9784 9789 978E 9793 9798 979D 97A2 97A7 97AB-97AC "
                "97B0-97B1 97B5-97B6 97BA-97BB 97BF-97C0 97C4-97C5 97C9-97CA 97CE-97CF 97D3-97D4 97D6 97DB "
                "97E0 97E5 97EA 97EF 97F4 97F9 97FE 9804 9809 980E "
                "9813 9818 981D 9822 9827 982B-982C 9830-9831 9835-9836 983A-983B 983F-9840 "
                "9844-9845 9849-984A 984E-984F 9853-9854 9856 985B 9860 9865 986A 986F "
                "9874 9879 987E 9884 9889 988E 9893 9898 989D 98A2 "
                "98A7 98AB-98AC 98B0-98B1 98B5-98B6 98BA-98BB 98BF-98C0 98C4-98C5 98C9-98CA 98CE-98CF 98D3-98D4 "
                "98D6 98DB 98E0 98E5 98EA 98EF 98F4 98F9 98FE 9904 "
                "9909 990E 9913 9918 991D 9922 9927 992B-992C 9930-9931 9935-9936 "
                "993A-993B 993F-9940 9944-9945 9949-994A 994E-994F 9953-9954 9956 995B 9960 9965 "
                "996A 996F 9974 9979 997E 9984 9989 998E 9993 9998 "
                "999D 99A2 99A7 99AB-99AC 99B0-99B1 99B5-99B6 99BA-99BB 99BF-99C0 99C4-99C5 99C9-99CA "
                "99CE-99CF 99D3-99D4 99D6 99DB 99E0 99E5 99EA 99EF 99F4 99F9 "
                "99FE 9A04 9A09 9A0E 9A13 9A18 9A1D 9A22 9A27 9A2B-9A2C "
                "9A30-9A31 9A35-9A36 9A3A-9A3B 9A3F-9A40 9A44-9A45 9A49-9A4A 9A4E-9A4F 9A53-9A54 9A56 9A5B "
                "9A60 9A65 9A6A 9A6F 9A74 9A79 9A7E 9A84 9A89 9A8E "
                "9A93 9A98 9A9D 9AA2 9AA7 9AAB-9AAC 9AB0-9AB1 9AB5-9AB6 9ABA-9ABB 9ABF-9AC0 "
                "9AC4-9AC5 9AC9-9ACA 9ACE-9ACF 9AD3-9AD4 9AD6 9ADB 9AE0 9AE5 9AEA 9AEF "
                "9AF4 9AF9 9AFE 9B04 9B09 9B0E 9B13 9B18 9B1D 9B22 "
                "9B27 9B2B-9B2C 9B30-9B31 9B35-9B36 9B3A-9B3B 9B3F-9B40 9B44-9B45 9B49-9B4A 9B4E-9B4F 9B53-9B54 "
                "9B56 9B5B 9B60 9B65 9B6A 9B6F 9B74 9B79 9B7E 9B84 "
                "9B89 9B8E 9B93 9B98 9B9D 9BA2 9BA7 9BAB-9BAC 9BB0-9BB1 9BB5-9BB6 "
                "9BBA-9BBB 9BBF-9BC0 9BC4-9BC5 9BC9-9BCA 9BCE-9BCF 9BD3-9BD4 9BD6 9BDB 9BE0 9BE5 "
                "9BEA 9BEF 9BF4 9BF9 9BFE 9C04 9C09 9C0E 9C13 9C18 "
                "9C1D 9C22 9C27 9C2B-9C2C 9C30-9C31 9C35-9C36 9C3A-9C3B 9C3F-9C40 9C44-9C45 9C49-9C4A "
                "9C4E-9C4F 9C53-9C54 9C56 9C5B 9C60 9C65 9C6A 9C6F 9C74 9C79 "
                "9C7E 9C84 9C89 9C8E 9C93 9C98 9C9D 9CA2 9CA7 9CAB-9CAC "
                "9CB0-9CB1 9CB5-9CB6 9CBA-9CBB 9CBF-9CC0 9CC4-9CC5 9CC9-9CCA 9CCE-9CCF 9CD3-9CD4 9CD6 9CDB "
                "9CE0 9CE5 9CEA 9CEF 9CF4 9CF9 9CFE 9D04 9D09 9D0E "
                "9D13 9D18 9D1D 9D22 9D27 9D2B-9D2C 9D30-9D31 9D35-9D36 9D3A-9D3B 9D3F-9D40 "
                "9D44-9D45 9D49-9D4A 9D4E-9D4F 9D53-9D54 9D56 9D5B 9D60 9D65 9D6A 9D6F "
                "9D74 9D79 9D7E 9D84 9D89 9D8E 9D93 9D98 9D9D 9DA2 "
                "9DA7 9DAB-9DAC 9DB0-9DB1 9DB5-9DB6 9DBA-9DBB 9DBF-9DC0 9DC4-9DC5 9DC9-9DCA 9DCE-9DCF 9DD3-9DD4 "
                "9DD6 9DDB 9DE0 9DE5 9DEA 9DEF 9DF4 9DF9 9DFE 9E04 "
                "9E09 9E0E 9E13 9E18 9E1D 9E22 9E27 9E2B-9E2C 9E30-9E31 9E35-9E36 "
                "9E3A-9E3B 9E3F-9E40 9E44-9E45 9E49-9E4A 9E4E-9E4F 9E53-9E54 9E56 9E5B 9E60 9E65 "
                "9E6A 9E6F 9E74 9E79 9E7E 9E84 9E89 9E8E 9E93 9E98 "
                "9E9D 9EA2 9EA7 9EAB-9EAC 9EB0-9EB1 9EB5-9EB6 9EBA-9EBB 9EBF-9EC0 9EC4-9EC5 9EC9-9ECA "
                "9ECE-9ECF 9ED3-9ED4 9ED6 9EDB 9EE0 9EE5 9EEA 9EEF 9EF4 9EF9 "
                "9EFE 9F04 9F09 9F0E 9F13 9F18 9F1D 9F22 9F27 9F2B-9F2C "
                "9F30-9F31 9F35-9F36 9F3A-9F3B 9F3F-9F40 9F44-9F45 9F49-9F4A 9F4E-9F4F 9F53-9F54 9F56 9F5B "
                "9F60 9F65 9F6A 9F6F 9F74 9F79 9F7E 9F84 9F89 9F8E "
                "9F93 9F98 9F9D 9FA2 9FA7 9FAB-9FAC 9FB0-9FB1 9FB5-9FB6 9FBA-9FBB 9FBF-9FC0 "
                "9FC4-9FC5 9FC9-9FCA 9FCE-9FCF 9FD3-9FD4 9FD6 9FDB 9FE0 9FE5 9FEA 9FEF "
                "9FF4 9FF9 9FFE A004 A009 A00E A013 A018 A01D A022 "
                "A027 A02B-A02C A030-A031 A035-A036 A03A-A03B A03F-A040 A044-A045 A049-A04A A04E-A04F A053-A054 "
                "A056 A05B A060 A065 A06A A06F A074 A079 A07E A084 "
                "A089 A08E A093 A098 A09D A0A2 A0A7 A0AB-A0AC A0B0-A0B1 A0B5-A0B6 "
                "A0BA-A0BB A0BF-A0C0 A0C4-A0C5 A0C9-A0CA A0CE-A0CF A0D3-A0D4 A0D6 A0DB A0E0 A0E5 "
                "A0EA A0EF A0F4 A0F9 A0FE A104 A109 A10E A113 A118 "
                "A11D A122 A127 A12B-A12C A130-A131 A135-A136 A13A-A13B A13F-A140 A144-A145 A149-A14A "
                "A14E-A14F A153-A154 A156 A15B A160 A165 A16A A16F A174 A179 "
                "A17E A184 A189 A18E A193 A198 A19D A1A2 A1A7 A1AB-A1AC "
                "A1B0-A1B1 A1B5-A1B6 A1BA-A1BB A1BF-A1C0 A1C4-A1C5 A1C9-A1CA A1CE-A1CF A1D3-A1D4 A1D6 A1DB "
                "A1E0 A1E5 A1EA A1EF A1F4 A1F9 A1FE A204 A209 A20E "
                "A213 A218 A21D A222 A227 A22B-A22C A230-A231 A235-A236 A23A-A23B A23F-A240 "
                "A244-A245 A249-A24A A24E-A24F A253-A254 A256 A25B A260 A265 A26A A26F "
                "A274 A279 A27E A284 A289 A28E A293 A298 A29D A2A2 "
                "A2A7 A2AB-A2AC A2B0-A2B1 A2B5-A2B6 A2BA-A2BB A2BF-A2C0 A2C4-A2C5 A2C9-A2CA A2CE-A2CF A2D3-A2D4 "
                "A2D6 A2DB A2E0 A2E5 A2EA A2EF A2F4 A2F9 A2FE A304 "
                "A309 A30E A313 A318 A31D A322 A327 A32B-A32C A330-A331 A335-A336 "
                "A33A-A33B A33F-A340 A344-A345 A349-A34A A34E-A34F A353-A354 A356 A35B A360 A365 "
                "A36A A36F A374 A379 A37E A384 A389 A38E A393 A398 "
                "A39D A3A2 A3A7 A3AB-A3AC A3B0-A3B1 A3B5-A3B6 A3BA-A3BB A3BF-A3C0 A3C4-A3C5 A3C9-A3CA "
                "A3CE-A3CF A3D3-A3D4 A3D6 A3DB A3E0 A3E5 A3EA A3EF A3F4 A3F9 "
                "A3FE A404 A409 A40E A413 A418 A41D A422 A427 A42B-A42C "
                "A430-A431 A435-A436 A43A-A43B A43F-A440 A444-A445 A449-A44A A44E-A44F A453-A454 A456 A45B "
                "A460 A465 A46A A46F A474 A479 A47E A484 A489 A48E "
                "A493 A498 A49D A4A2 A4A7 A4AB-A4AC A4B0-A4B1 A4B5-A4B6 A4BA-A4BB A4BF-A4C0 "
                "A4C4-A4C5 A4C9-A4CA A4CE-A4CF A4D3-A4D4 A4D6 A4DB A4E0 A4E5 A4EA A4EF "
                "A4F4 A4F9 A4FE A504 A509 A50E A513 A518 A51D A522 "
                "A527 A52B-A52C A530-A531 A535-A536 A53A-A53B A53F-A540 A544-A545 A549-A54A A54E-A54F A553-A554 "
                "A556 A55B A560 A565 A56A A56F A574 A579 A57E A584 "
                "A589 A58E A593 A598 A59D A5A2 A5A7 A5AB-A5AC A5B0-A5B1 A5B5-A5B6 "
                "A5BA-A5BB A5BF-A5C0 A5C4-A5C5 A5C9-A5CA A5CE-A5CF A5D3-A5D4 A5D6 A5DB A5E0 A5E5 "
                "A5EA A5EF A5F4 A5F9 A5FE A604 A609 A60E A613 A618 "
                "A61D A622 A627 A62B-A62C A630-A631 A635-A636 A63A-A63B A63F-A640 A644-A645 A649-A64A "
                "A64E-A64F A653-A654 A656 A65B A660 A665 A66A A66F A674 A679 "
                "A67E A684 A689 A68E A693 A698 A69D A6A2 A6A7 A6AB-A6AC "
                "A6B0-A6B1 A6B5-A6B6 A6BA-A6BB A6BF-A6C0 A6C4-A6C5 A6C9-A6CA A6CE-A6CF A6D3-A6D4 A6D6 A6DB "
                "A6E0 A6E5 A6EA A6EF A6F4 A6F9 A6FE A704 A709 A70E "
                "A713 A718 A71D A722 A727 A72B-A72C A730-A731 A735-A736 A73A-A73B A73F-A740 "
                "A744-A745 A749-A74A A74E-A74F A753-A754 A756 A75B A760 A765 A76A A76F "
                "A774 A779 A77E A784 A789 A78E A793 A798 A79D A7A2 "
                "A7A7 A7AB-A7AC A7B0-A7B1 A7B5-A7B6 A7BA-A7BB A7BF-A7C0 A7C4-A7C5 A7C9-A7CA A7CE-A7CF A7D3-A7D4 "
                "A7D6 A7DB A7E0 A7E5 A7EA A7EF A7F4 A7F9 A7FE A804 "
                "A809 A80E A813 A818 A81D A822 A827 A82B-A82C A830-A831 A835-A836 "
                "A83A-A83B A83F-A840 A844-A845 A849-A84A A84E-A84F A853-A854 A856 A85B A860 A865 "
                "A86A A86F A874 A879 A87E A884 A889 A88E A893 A898 "
                "A89D A8A2 A8A7 A8AB-A8AC A8B0-A8B1 A8B5-A8B6 A8BA-A8BB A8BF-A8C0 A8C4-A8C5 A8C9-A8CA "
                "A8CE-A8CF A8D3-A8D4 A8D6 A8DB A8E0 A8E5 A8EA A8EF A8F4 A8F9 "
                "A8FE A904 A909 A90E A913 A918 A91D A922 A927 A92B-A92C "
                "A930-A931 A935-A936 A93A-A93B A93F-A940 A944-A945 A949-A94A A94E-A94F A953-A954 A956 A95B "
                "A960 A965 A96A A96F A974 A979 A97E A984 A989 A98E "
                "A993 A998 A99D A9A2 A9A7 A9AB-A9AC A9B0-A9B1 A9B5-A9B6 A9BA-A9BB A9BF-A9C0 "
                "A9C4-A9C5 A9C9-A9CA A9CE-A9CF A9D3-A9D4 A9D6 A9DB A9E0 A9E5 A9EA A9EF "
                "A9F4 A9F9 A9FE AA04 AA09 AA0E AA13 AA18 AA1D AA22 "
                "AA27 AA2B-AA2C AA30-AA31 AA35-AA36 AA3A-AA3B AA3F-AA40 AA44-AA45 AA49-AA4A AA4E-AA4F AA53-AA54 "
                "AA56 AA5B AA60 AA65 AA6A AA6F AA74 AA79 AA7E AA84 "
                "AA89 AA8E AA93 AA98 AA9D AAA2 AAA7 AAAB-AAAC AAB0-AAB1 AAB5-AAB6 "
                "AABA-AABB AABF-AAC0 AAC4-AAC5 AAC9-AACA AACE-AACF AAD3-AAD4 AAD6 AADB AAE0 AAE5 "
                "AAEA AAEF AAF4 AAF9 AAFE AB04 AB09 AB0E AB13 AB18 "
                "AB1D AB22 AB27 AB2B-AB2C AB30-AB31 AB35-AB36 AB3A-AB3B AB3F-AB40 AB44-AB45 AB49-AB4A "
                "AB4E-AB4F AB53-AB54 AB56 AB5B AB60 AB65 AB6A AB6F AB74 AB79 "
                "AB7E AB84 AB89 AB8E AB93 AB98 AB9D ABA2 ABA7 ABAB-ABAC "
                "ABB0-ABB1 ABB5-ABB6 ABBA-ABBB ABBF-ABC0 ABC4-ABC5 ABC9-ABCA ABCE-ABCF ABD3-ABD4 ABD6 ABDB "
                "ABE0 ABE5 ABEA ABEF ABF4 ABF9 ABFE AC04 AC09 AC0E "
                "AC13 AC18 AC1D AC22 AC27 AC2B-AC2C AC30-AC31 AC35-AC36 AC3A-AC3B AC3F-AC40 "
                "AC44-AC45 AC49-AC4A AC4E-AC4F AC53-AC54 AC56 AC5B AC60 AC65 AC6A AC6F "
                "AC74 AC79 AC7E AC84 AC89 AC8E AC93 AC98 AC9D ACA2 "
                "ACA7 ACAB-ACAC ACB0-ACB1 ACB5-ACB6 ACBA-ACBB ACBF-ACC0 ACC4-ACC5 ACC9-ACCA ACCE-ACCF ACD3-ACD4 "
                "ACD6 ACDB ACE0 ACE5 ACEA ACEF ACF4 ACF9 ACFE AD04 "
                "AD09 AD0E AD13 AD18 AD1D AD22 AD27 AD2B-AD2C AD30-AD31 AD35-AD36 "
                "AD3A-AD3B AD3F-AD40 AD44-AD45 AD49-AD4A AD4E-AD4F AD53-AD54 AD56 AD5B AD60 AD65 "
                "AD6A AD6F AD74 AD79 AD7E AD84 AD89 AD8E AD93 AD98 "
                "AD9D ADA2 ADA7 ADAB-ADAC ADB0-ADB1 ADB5-ADB6 ADBA-ADBB ADBF-ADC0 ADC4-ADC5 ADC9-ADCA "
                "ADCE-ADCF ADD3-ADD4 ADD6 ADDB ADE0 ADE5 ADEA ADEF ADF4 ADF9 "
                "ADFE AE04 AE09 AE0E AE13 AE18 AE1D AE22 AE27 AE2B-AE2C "
                "AE30-AE31 AE35-AE36 AE3A-AE3B AE3F-AE40 AE44-AE45 AE49-AE4A AE4E-AE4F AE53-AE54 AE56 AE5B "
                "AE60 AE65 AE6A AE6F AE74 AE79 AE7E AE84 AE89 AE8E "
                "AE93 AE98 AE9D AEA2 AEA7 AEAB-AEAC AEB0-AEB1 AEB5-AEB6 AEBA-AEBB AEBF-AEC0 "
                "AEC4-AEC5 AEC9-AECA AECE-AECF AED3-AED4 AED6 AEDB AEE0 AEE5 AEEA AEEF "
                "AEF4 AEF9 AEFE AF04 AF09 AF0E AF13 AF18 AF1D AF22 "
                "AF27 AF2B-AF2C AF30-AF31 AF35-AF36 AF3A-AF3B AF3F-AF40 AF44-AF45 AF49-AF4A AF4E-AF4F AF53-AF54 "
                "AF56 AF5B AF60 AF65 AF6A AF6F AF74 AF79 AF7E AF84 "
                "AF89 AF8E AF93 AF98 AF9D AFA2 AFA7 AFAB-AFAC AFB0-AFB1 AFB5-AFB6 "
                "AFBA-AFBB AFBF-AFC0 AFC4-AFC5 AFC9-AFCA AFCE-AFCF AFD3-AFD4 AFD6 AFDB AFE0 AFE5 "
                "AFEA AFEF AFF4 AFF9 AFFE B004 B009 B00E B013 B018 "
                "B01D B022 B027 B02B-B02C B030-B031 B035-B036 B03A-B03B B03F-B040 B044-B045 B049-B04A "
                "B04E-B04F B053-B054 B056 B05B B060 B065 B06A B06F B074 B079 "
                "B07E B084 B089 B08E B093 B098 B09D B0A2 B0A7 B0AB-B0AC "
                "B0B0-B0B1 B0B5-B0B6 B0BA-B0BB B0BF-B0C0 B0C4-B0C5 B0C9-B0CA B0CE-B0CF B0D3-B0D4 B0D6 B0DB "
                "B0E0 B0E5 B0EA B0EF B0F4 B0F9 B0FE B104 B109 B10E "
                "B113 B118 B11D B122 B127 B12B-B12C B130-B131 B135-B136 B13A-B13B B13F-B140 "
                "B144-B145 B149-B14A B14E-B14F B153-B154 B156 B15B B160 B165 B16A B16F "
                "B174 B179 B17E B184 B189 B18E B193 B198 B19D B1A2 "
                "B1A7 B1AB-B1AC B1B0-B1B1 B1B5-B1B6 B1BA-B1BB B1BF-B1C0 B1C4-B1C5 B1C9-B1CA B1CE-B1CF B1D3-B1D4 "
                "B1D6 B1DB B1E0 B1E5 B1EA B1EF B1F4 B1F9 B1FE B204 "
                "B209 B20E B213 B218 B21D B222 B227 B22B-B22C B230-B231 B235-B236 "
                "B23A-B23B B23F-B240 B244-B245 B249-B24A B24E-B24F B253-B254 B256 B25B B260 B265 "
                "B26A B26F B274 B279 B27E B284 B289 B28E B293 B298 "
                "B29D B2A2 B2A7 B2AB-B2AC B2B0-B2B1 B2B5-B2B6 B2BA-B2BB B2BF-B2C0 B2C4-B2C5 B2C9-B2CA "
                "B2CE-B2CF B2D3-B2D4 B2D6 B2DB B2E0 B2E5 B2EA B2EF B2F4 B2F9 "
                "B2FE B304 B309 B30E B313 B318 B31D B322 B327 B32B-B32C "
                "B330-B331 B335-B336 B33A-B33B B33F-B340 B344-B345 B349-B34A B34E-B34F B353-B354 B356 B35B "
                "B360 B365 B36A B36F B374 B379 B37E B384 B389 B38E "
                "B393 B398 B39D B3A2 B3A7 B3AB-B3AC B3B0-B3B1 B3B5-B3B6 B3BA-B3BB B3BF-B3C0 "
                "B3C4-B3C5 B3C9-B3CA B3CE-B3CF B3D3-B3D4 B3D6 B3DB B3E0 B3E5 B3EA B3EF "
                "B3F4 B3F9 B3FE B404 B409 B40E B413 B418 B41D B422 "
                "B427 B42B-B42C B430-B431 B435-B436 B43A-B43B B43F-B440 B444-B445 B449-B44A B44E-B44F B453-B454 "
                "B456 B45B B460 B465 B46A B46F B474 B479 B47E B484 "
                "B489 B48E B493 B498 B49D B4A2 B4A7 B4AB-B4AC B4B0-B4B1 B4B5-B4B6 "
                "B4BA-B4BB B4BF-B4C0 B4C4-B4C5 B4C9-B4CA B4CE-B4CF B4D3-B4D4 B4D6 B4DB B4E0 B4E5 "
                "B4EA B4EF B4F4 B4F9 B4FE B504 B509 B50E B513 B518 "
                "B51D B522 B527 B52B-B52C B530-B531 B535-B536 B53A-B53B B53F-B540 B544-B545 B549-B54A "
                "B54E-B54F B553-B554 B556 B55B B560 B565 B56A B56F B574 B579 "
                "B57E B584 B589 B58E B593 B598 B59D B5A2 B5A7 B5AB-B5AC "
                "B5B0-B5B1 B5B5-B5B6 B5BA-B5BB B5BF-B5C0 B5C4-B5C5 B5C9-B5CA B5CE-B5CF B5D3-B5D4 B5D6 B5DB "
                "B5E0 B5E5 B5EA B5EF B5F4 B5F9 B5FE B604 B609 B60E "
                "B613 B618 B61D B622 B627 B62B-B62C B630-B631 B635-B636 B63A-B63B B63F-B640 "
                "B644-B645 B649-B64A B64E-B64F B653-B654 B656 B65B B660 B665 B66A B66F "
                "B674 B679 B67E B684 B689 B68E B693 B698 B69D B6A2 "
                "B6A7 B6AB-B6AC B6B0-B6B1 BCBA BCF3 BD09 BD16 BD23 BD2A BD31 "
                "BD38 BD3F BD46 BD4D BD55 BD60 BD70 BD77 BD80 BD88 "
                "BD91 BD9A-BD9B BDA4-BDA5 BDAF-BDB0 BDBA-BDBB BDC6-BDC8 BDD3-BDD6 BDEB BDED BDEF "
                "BDF1 BE00-BE05 BE3A-BE42 BE50-BE54 BE5E-BE62 BE6A-BE6D BE74 BE7A BE85 BE8C "
                "BE92-BE93 BE98 BEA7 BEAC BEB0 BEB4 BEB8 BEBC BECA BF2C "
                "BF31 BF46 BF4D BF54 BF58 BF5D BF61-BF62 BF66-BF67 BF6C BF72 "
                "BF78-BF79 BF80 BFC1 BFC5 BFC8 BFCF BFD2 BFDD C00C C011 "
                "C016 C038 C03A C05D C067-C068 C070-C071 C085 C08C C0A3 C0CF "
                "C109 C113 C115 C12A C133-C134 C13E C149 C14B C156 C160-C161 "
                "C16C-C16D C185 C192 C19E C1B3 C1C9 C1CD C1DA C1E0 C1E5 "
                "C1F6 C20B C21E C229 C22D C24E C256 C259 C25F C27A "
                "C280 C292 C296 C299 C2A1 C2B6"
            ),
            1: (
                "00D6 00DB 00E0 00E5 00EA 00EF 00F4 00F9 00FE 0104 "
                "0109 010E 0113 0118 011C-011D 0121-0122 0126-0127 012B-012C 0130-0131 0135-0136 "
                "013A-013B 013F-0140 0144-0145 0149-014A 014E-014F 0153-0154 0156 015B 0160 0165 "
                "016A 016F 0174 0179 017E 0184 0189 018E 0193 0198 "
                "019C-019D 01A1-01A2 01A6-01A7 01AB-01AC 01B0-01B1 01B5-01B6 01BA-01BB 01BF-01C0 01C4-01C5 01C9-01CA "
                "01CE-01CF 01D3-01D4 01D6 01DB 01E0 01E5 01EA 01EF 01F4 01F9 "
                "01FE 0204 0209 020E 0213 0218 021C-021D 0221-0222 0226-0227 022B-022C "
                "0230-0231 0235-0236 023A-023B 023F-0240 0244-0245 0249-024A 024E-024F 0253-0254 0256 025B "
                "0260 0265 026A 026F 0274 0279 027E 0284 0289 028E "
                "0293 0298 029C-029D 02A1-02A2 02A6-02A7 02AB-02AC 02B0-02B1 02B5-02B6 02BA-02BB 02BF-02C0 "
                "02C4-02C5 02C9-02CA 02CE-02CF 02D3-02D4 02D6 02DB 02E0 02E5 02EA 02EF "
                "02F4 02F9 02FE 0304 0309 030E 0313 0318 031C-031D 0321-0322 "
                "0326-0327 032B-032C 0330-0331 0335-0336 033A-033B 033F-0340 0344-0345 0349-034A 034E-034F 0353-0354 "
                "0356 035B 0360 0365 036A 036F 0374 0379 037E 0384 "
                "0389 038E 0393 0398 039C-039D 03A1-03A2 03A6-03A7 03AB-03AC 03B0-03B1 03B5-03B6 "
                "03BA-03BB 03BF-03C0 03C4-03C5 03C9-03CA 03CE-03CF 03D3-03D4 03D6 03DB 03E0 03E5 "
                "03EA 03EF 03F4 03F9 03FE 0404 0409 040E 0413 0418 "
                "041C-041D 0421-0422 0426-0427 042B-042C 0430-0431 0435-0436 043A-043B 043F-0440 0444-0445 0449-044A "
                "044E-044F 0453-0454 0456 045B 0460 0465 046A 046F 0474 0479 "
                "047E 0484 0489 048E 0493 0498 049C-049D 04A1-04A2 04A6-04A7 04AB-04AC "
                "04B0-04B1 04B5-04B6 04BA-04BB 04BF-04C0 04C4-04C5 04C9-04CA 04CE-04CF 04D3-04D4 04D6 04DB "
                "04E0 04E5 04EA 04EF 04F4 04F9 04FE 0504 0509 050E "
                "0513 0518 051C-051D 0521-0522 0526-0527 052B-052C 0530-0531 0535-0536 053A-053B 053F-0540 "
                "0544-0545 0549-054A 054E-054F 0553-0554 0556 055B 0560 0565 056A 056F "
                "0574 0579 057E 0584 0589 058E 0593 0598 059C-059D 05A1-05A2 "
                "05A6-05A7 05AB-05AC 05B0-05B1 05B5-05B6 05BA-05BB 05BF-05C0 05C4-05C5 05C9-05CA 05CE-05CF 05D3-05D4 "
                "05D6 05DB 05E0 05E5 05EA 05EF 05F4 05F9 05FE 0604 "
                "0609 060E 0613 0618 061C-061D 0621-0622 0626-0627 062B-062C 0630-0631 0635-0636 "
                "063A-063B 063F-0640 0644-0645 0649-064A 064E-064F 0653-0654 0656 065B 0660 0665 "
                "066A 066F 0674 0679 067E 0684 0689 068E 0693 0698 "
                "069C-069D 06A1-06A2 06A6-06A7 06AB-06AC 06B0-06B1 06B5-06B6 06BA-06BB 06BF-06C0 06C4-06C5 06C9-06CA "
                "06CE-06CF 06D3-06D4 06D6 06DB 06E0 06E5 06EA 06EF 06F4 06F9 "
                "06FE 0704 0709 070E 0713 0718 071C-071D 0721-0722 0726-0727 072B-072C "
                "0730-0731 0735-0736 073A-073B 073F-0740 0744-0745 0749-074A 074E-074F 0753-0754 0756 075B "
                "0760 0765 076A 076F 0774 0779 077E 0784 0789 078E "
                "0793 0798 079C-079D 07A1-07A2 07A6-07A7 07AB-07AC 07B0-07B1 07B5-07B6 07BA-07BB 07BF-07C0 "
                "07C4-07C5 07C9-07CA 07CE-07CF 07D3-07D4 07D6 07DB 07E0 07E5 07EA 07EF "
                "07F4 07F9 07FE 0804 0809 080E 0813 0818 081C-081D 0821-0822 "
                "0826-0827 082B-082C 0830-0831 0835-0836 083A-083B 083F-0840 0844-0845 0849-084A 084E-084F 0853-0854 "
                "0856 085B 0860 0865 086A 086F 0874 0879 087E 0884 "
                "0889 088E 0893 0898 089C-089D 08A1-08A2 08A6-08A7 08AB-08AC 08B0-08B1 08B5-08B6 "
                "08BA-08BB 08BF-08C0 08C4-08C5 08C9-08CA 08CE-08CF 08D3-08D4 08D6 08DB 08E0 08E5 "
                "08EA 08EF 08F4 08F9 08FE 0904 0909 090E 0913 0918 "
                "091C-091D 0921-0922 0926-0927 092B-092C 0930-0931 0935-0936 093A-093B 093F-0940 0944-0945 0949-094A "
                "094E-094F 0953-0954 0956 095B 0960 0965 096A 096F 0974 0979 "
                "097E 0984 0989 098E 0993 0998 099C-099D 09A1-09A2 09A6-09A7 09AB-09AC "
                "09B0-09B1 09B5-09B6 09BA-09BB 09BF-09C0 09C4-09C5 09C9-09CA 09CE-09CF 09D3-09D4 09D6 09DB "
                "09E0 09E5 09EA 09EF 09F4 09F9 09FE 0A04 0A09 0A0E "
                "0A13 0A18 0A1C-0A1D 0A21-0A22 0A26-0A27 0A2B-0A2C 0A30-0A31 0A35-0A36 0A3A-0A3B 0A3F-0A40 "
                "0A44-0A45 0A49-0A4A 0A4E-0A4F 0A53-0A54 0A56 0A5B 0A60 0A65 0A6A 0A6F "
                "0A74 0A79 0A7E 0A84 0A89 0A8E 0A93 0A98 0A9C-0A9D 0AA1-0AA2 "
                "0AA6-0AA7 0AAB-0AAC 0AB0-0AB1 0AB5-0AB6 0ABA-0ABB 0ABF-0AC0 0AC4-0AC5 0AC9-0ACA 0ACE-0ACF 0AD3-0AD4 "
                "0AD6 0ADB 0AE0 0AE5 0AEA 0AEF 0AF4 0AF9 0AFE 0B04 "
                "0B09 0B0E 0B13 0B18 0B1C-0B1D 0B21-0B22 0B26-0B27 0B2B-0B2C 0B30-0B31 0B35-0B36 "
                "0B3A-0B3B 0B3F-0B40 0B44-0B45 0B49-0B4A 0B4E-0B4F 0B53-0B54 0B56 0B5B 0B60 0B65 "
                "0B6A 0B6F 0B74 0B79 0B7E 0B84 0B89 0B8E 0B93 0B98 "
                "0B9C-0B9D 0BA1-0BA2 0BA6-0BA7 0BAB-0BAC 0BB0-0BB1 0BB5-0BB6 0BBA-0BBB 0BBF-0BC0 0BC4-0BC5 0BC9-0BCA "
                "0BCE-0BCF 0BD3-0BD4 0BD6 0BDB 0BE0 0BE5 0BEA 0BEF 0BF4 0BF9 "
                "0BFE 0C04 0C09 0C0E 0C13 0C18 0C1C-0C1D 0C21-0C22 0C26-0C27 0C2B-0C2C "
                "0C30-0C31 0C35-0C36 0C3A-0C3B 0C3F-0C40 0C44-0C45 0C49-0C4A 0C4E-0C4F 0C53-0C54 0C56 0C5B "
                "0C60 0C65 0C6A 0C6F 0C74 0C79 0C7E 0C84 0C89 0C8E "
                "0C93 0C98 0C9C-0C9D 0CA1-0CA2 0CA6-0CA7 0CAB-0CAC 0CB0-0CB1 0CB5-0CB6 0CBA-0CBB 0CBF-0CC0 "
                "0CC4-0CC5 0CC9-0CCA 0CCE-0CCF 0CD3-0CD4 0CD6 0CDB 0CE0 0CE5 0CEA 0CEF "
                "0CF4 0CF9 0CFE 0D04 0D09 0D0E 0D13 0D18 0D1C-0D1D 0D21-0D22 "
                "0D26-0D27 0D2B-0D2C 0D30-0D31 0D35-0D36 0D3A-0D3B 0D3F-0D40 0D44-0D45 0D49-0D4A 0D4E-0D4F 0D53-0D54 "
                "0D56 0D5B 0D60 0D65 0D6A 0D6F 0D74 0D79 0D7E 0D84 "
                "0D89 0D8E 0D93 0D98 0D9C-0D9D 0DA1-0DA2 0DA6-0DA7 0DAB-0DAC 0DB0-0DB1 0DB5-0DB6 "
                "0DBA-0DBB 0DBF-0DC0 0DC4-0DC5 0DC9-0DCA 0DCE-0DCF 0DD3-0DD4 0DD6 0DDB 0DE0 0DE5 "
                "0DEA 0DEF 0DF4 0DF9 0DFE 0E04 0E09 0E0E 0E13 0E18 "
                "0E1C-0E1D 0E21-0E22 0E26-0E27 0E2B-0E2C 0E30-0E31 0E35-0E36 0E3A-0E3B 0E3F-0E40 0E44-0E45 0E49-0E4A "
                "0E4E-0E4F 0E53-0E54 0E56 0E5B 0E60 0E65 0E6A 0E6F 0E74 0E79 "
                "0E7E 0E84 0E89 0E8E 0E93 0E98 0E9C-0E9D 0EA1-0EA2 0EA6-0EA7 0EAB-0EAC "
                "0EB0-0EB1 0EB5-0EB6 0EBA-0EBB 0EBF-0EC0 0EC4-0EC5 0EC9-0ECA 0ECE-0ECF 0ED3-0ED4 0ED6 0EDB "
                "0EE0 0EE5 0EEA 0EEF 0EF4 0EF9 0EFE 0F04 0F09 0F0E "
                "0F13 0F18 0F1C-0F1D 0F21-0F22 0F26-0F27 0F2B-0F2C 0F30-0F31 0F35-0F36 0F3A-0F3B 0F3F-0F40 "
                "0F44-0F45 0F49-0F4A 0F4E-0F4F 0F53-0F54 0F56 0F5B 0F60 0F65 0F6A 0F6F "
                "0F74 0F79 0F7E 0F84 0F89 0F8E 0F93 0F98 0F9C-0F9D 0FA1-0FA2 "
                "0FA6-0FA7 0FAB-0FAC 0FB0-0FB1 0FB5-0FB6 0FBA-0FBB 0FBF-0FC0 0FC4-0FC5 0FC9-0FCA 0FCE-0FCF 0FD3-0FD4 "
                "0FD6 0FDB 0FE0 0FE5 0FEA 0FEF 0FF4 0FF9 0FFE 1004 "
                "1009 100E 1013 1018 101C-101D 1021-1022 1026-1027 102B-102C 1030-1031 1035-1036 "
                "103A-103B 103F-1040 1044-1045 1049-104A 104E-104F 1053-1054 1056 105B 1060 1065 "
                "106A 106F 1074 1079 107E 1084 1089 108E 1093 1098 "
                "109C-109D 10A1-10A2 10A6-10A7 10AB-10AC 10B0-10B1 10B5-10B6 10BA-10BB 10BF-10C0 10C4-10C5 10C9-10CA "
                "10CE-10CF 10D3-10D4 10D6 10DB 10E0 10E5 10EA 10EF 10F4 10F9 "
                "10FE 1104 1109 110E 1113 1118 111C-111D 1121-1122 1126-1127 112B-112C "
                "1130-1131 1135-1136 113A-113B 113F-1140 1144-1145 1149-114A 114E-114F 1153-1154 1156 115B "
                "1160 1165 116A 116F 1174 1179 117E 1184 1189 118E "
                "1193 1198 119C-119D 11A1-11A2 11A6-11A7 11AB-11AC 11B0-11B1 11B5-11B6 11BA-11BB 11BF-11C0 "
                "11C4-11C5 11C9-11CA 11CE-11CF 11D3-11D4 11D6 11DB 11E0 11E5 11EA 11EF "
                "11F4 11F9 11FE 1204 1209 120E 1213 1218 121C-121D 1221-1222 "
                "1226-1227 122B-122C 1230-1231 1235-1236 123A-123B 123F-1240 1244-1245 1249-124A 124E-124F 1253-1254 "
                "1256 125B 1260 1265 126A 126F 1274 1279 127E 1284 "
                "1289 128E 1293 1298 129C-129D 12A1-12A2 12A6-12A7 12AB-12AC 12B0-12B1 12B5-12B6 "
                "12BA-12BB 12BF-12C0 12C4-12C5 12C9-12CA 12CE-12CF 12D3-12D4 12D6 12DB 12E0 12E5 "
                "12EA 12EF 12F4 12F9 12FE 1304 1309 130E 1313 1318 "
                "131C-131D 1321-1322 1326-1327 132B-132C 1330-1331 1335-1336 133A-133B 133F-1340 1344-1345 1349-134A "
                "134E-134F 1353-1354 1356 135B 1360 1365 136A 136F 1374 1379 "
                "137E 1384 1389 138E 1393 1398 139C-139D 13A1-13A2 13A6-13A7 13AB-13AC "
                "13B0-13B1 13B5-13B6 13BA-13BB 13BF-13C0 13C4-13C5 13C9-13CA 13CE-13CF 13D3-13D4 13D6 13DB "
                "13E0 13E5 13EA 13EF 13F4 13F9 13FE 1404 1409 140E "
                "1413 1418 141C-141D 1421-1422 1426-1427 142B-142C 1430-1431 1435-1436 143A-143B 143F-1440 "
                "1444-1445 1449-144A 144E-144F 1453-1454 1456 145B 1460 1465 146A 146F "
                "1474 1479 147E 1484 1489 148E 1493 1498 149C-149D 14A1-14A2 "
                "14A6-14A7 14AB-14AC 14B0-14B1 14B5-14B6 14BA-14BB 14BF-14C0 14C4-14C5 14C9-14CA 14CE-14CF 14D3-14D4 "
                "14D6 14DB 14E0 14E5 14EA 14EF 14F4 14F9 14FE 1504 "
                "1509 150E 1513 1518 151C-151D 1521-1522 1526-1527 152B-152C 1530-1531 1535-1536 "
                "153A-153B 153F-1540 1544-1545 1549-154A 154E-154F 1553-1554 1556 155B 1560 1565 "
                "156A 156F 1574 1579 157E 1584 1589 158E 1593 1598 "
                "159C-159D 15A1-15A2 15A6-15A7 15AB-15AC 15B0-15B1 15B5-15B6 15BA-15BB 15BF-15C0 15C4-15C5 15C9-15CA "
                "15CE-15CF 15D3-15D4 15D6 15DB 15E0 15E5 15EA 15EF 15F4 15F9 "
                "15FE 1604 1609 160E 1613 1618 161C-161D 1621-1622 1626-1627 162B-162C "
                "1630-1631 1635-1636 163A-163B 163F-1640 1644-1645 1649-164A 164E-164F 1653-1654 1656 165B "
                "1660 1665 166A 166F 1674 1679 167E 1684 1689 168E "
                "1693 1698 169C-169D 16A1-16A2 16A6-16A7 16AB-16AC 16B0-16B1 16B5-16B6 16BA-16BB 16BF-16C0 "
                "16C4-16C5 16C9-16CA 16CE-16CF 16D3-16D4 16D6 16DB 16E0 16E5 16EA 16EF "
                "16F4 16F9 16FE 1704 1709 170E 1713 1718 171C-171D 1721-1722 "
                "1726-1727 172B-172C 1730-1731 1735-1736 173A-173B 173F-1740 1744-1745 1749-174A 174E-174F 1753-1754 "
                "1756 175B 1760 1765 176A 176F 1774 1779 177E 1784 "
                "1789 178E 1793 1798 179C-179D 17A1-17A2 17A6-17A7 17AB-17AC 17B0-17B1 17B5-17B6 "
                "17BA-17BB 17BF-17C0 17C4-17C5 17C9-17CA 17CE-17CF 17D3-17D4 17D6 17DB 17E0 17E5 "
                "17EA 17EF 17F4 17F9 17FE 1804 1809 180E 1813 1818 "
                "181C-181D 1821-1822 1826-1827 182B-182C 1830-1831 1835-1836 183A-183B 183F-1840 1844-1845 1849-184A "
                "184E-184F 1853-1854 1856 185B 1860 1865 186A 186F 1874 1879 "
                "187E 1884 1889 188E 1893 1898 189C-189D 18A1-18A2 18A6-18A7 18AB-18AC "
                "18B0-18B1 18B5-18B6 18BA-18BB 18BF-18C0 18C4-18C5 18C9-18CA 18CE-18CF 18D3-18D4 18D6 18DB "
                "18E0 18E5 18EA 18EF 18F4 18F9 18FE 1904 1909 190E "
                "1913 1918 191C-191D 1921-1922 1926-1927 192B-192C 1930-1931 1935-1936 193A-193B 193F-1940 "
                "1944-1945 1949-194A 194E-194F 1953-1954 1956 195B 1960 1965 196A 196F "
                "1974 1979 197E 1984 1989 198E 1993 1998 199C-199D 19A1-19A2 "
                "19A6-19A7 19AB-19AC 19B0-19B1 19B5-19B6 19BA-19BB 19BF-19C0 19C4-19C5 19C9-19CA 19CE-19CF 19D3-19D4 "
                "19D6 19DB 19E0 19E5 19EA 19EF 19F4 19F9 19FE 1A04 "
                "1A09 1A0E 1A13 1A18 1A1C-1A1D 1A21-1A22 1A26-1A27 1A2B-1A2C 1A30-1A31 1A35-1A36 "
                "1A3A-1A3B 1A3F-1A40 1A44-1A45 1A49-1A4A 1A4E-1A4F 1A53-1A54 1A56 1A5B 1A60 1A65 "
                "1A6A 1A6F 1A74 1A79 1A7E 1A84 1A89 1A8E 1A93 1A98 "
                "1A9C-1A9D 1AA1-1AA2 1AA6-1AA7 1AAB-1AAC 1AB0-1AB1 1AB5-1AB6 1ABA-1ABB 1ABF-1AC0 1AC4-1AC5 1AC9-1ACA "
                "1ACE-1ACF 1AD3-1AD4 1AD6 1ADB 1AE0 1AE5 1AEA 1AEF 1AF4 1AF9 "
                "1AFE 1B04 1B09 1B0E 1B13 1B18 1B1C-1B1D 1B21-1B22 1B26-1B27 1B2B-1B2C "
                "1B30-1B31 1B35-1B36 1B3A-1B3B 1B3F-1B40 1B44-1B45 1B49-1B4A 1B4E-1B4F 1B53-1B54 1B56 1B5B "
                "1B60 1B65 1B6A 1B6F 1B74 1B79 1B7E 1B84 1B89 1B8E "
                "1B93 1B98 1B9C-1B9D 1BA1-1BA2 1BA6-1BA7 1BAB-1BAC 1BB0-1BB1 1BB5-1BB6 1BBA-1BBB 1BBF-1BC0 "
                "1BC4-1BC5 1BC9-1BCA 1BCE-1BCF 1BD3-1BD4 1BD6 1BDB 1BE0 1BE5 1BEA 1BEF "
                "1BF4 1BF9 1BFE 1C04 1C09 1C0E 1C13 1C18 1C1C-1C1D 1C21-1C22 "
                "1C26-1C27 1C2B-1C2C 1C30-1C31 1C35-1C36 1C3A-1C3B 1C3F-1C40 1C44-1C45 1C49-1C4A 1C4E-1C4F 1C53-1C54 "
                "1C56 1C5B 1C60 1C65 1C6A 1C6F 1C74 1C79 1C7E 1C84 "
                "1C89 1C8E 1C93 1C98 1C9C-1C9D 1CA1-1CA2 1CA6-1CA7 1CAB-1CAC 1CB0-1CB1 1CB5-1CB6 "
                "1CBA-1CBB 1CBF-1CC0 1CC4-1CC5 1CC9-1CCA 1CCE-1CCF 1CD3-1CD4 1CD6 1CDB 1CE0 1CE5 "
                "1CEA 1CEF 1CF4 1CF9 1CFE 1D04 1D09 1D0E 1D13 1D18 "
                "1D1C-1D1D 1D21-1D22 1D26-1D27 1D2B-1D2C 1D30-1D31 1D35-1D36 1D3A-1D3B 1D3F-1D40 1D44-1D45 1D49-1D4A "
                "1D4E-1D4F 1D53-1D54 1D56 1D5B 1D60 1D65 1D6A 1D6F 1D74 1D79 "
                "1D7E 1D84 1D89 1D8E 1D93 1D98 1D9C-1D9D 1DA1-1DA2 1DA6-1DA7 1DAB-1DAC "
                "1DB0-1DB1 1DB5-1DB6 1DBA-1DBB 1DBF-1DC0 1DC4-1DC5 1DC9-1DCA 1DCE-1DCF 1DD3-1DD4 1DD6 1DDB "
                "1DE0 1DE5 1DEA 1DEF 1DF4 1DF9 1DFE 1E04 1E09 1E0E "
                "1E13 1E18 1E1C-1E1D 1E21-1E22 1E26-1E27 1E2B-1E2C 1E30-1E31 1E35-1E36 1E3A-1E3B 1E3F-1E40 "
                "1E44-1E45 1E49-1E4A 1E4E-1E4F 1E53-1E54 1E56 1E5B 1E60 1E65 1E6A 1E6F "
                "1E74 1E79 1E7E 1E84 1E89 1E8E 1E93 1E98 1E9C-1E9D 1EA1-1EA2 "
                "1EA6-1EA7 1EAB-1EAC 1EB0-1EB1 1EB5-1EB6 1EBA-1EBB 1EBF-1EC0 1EC4-1EC5 1EC9-1ECA 1ECE-1ECF 1ED3-1ED4 "
                "1ED6 1EDB 1EE0 1EE5 1EEA 1EEF 1EF4 1EF9 1EFE 1F04 "
                "1F09 1F0E 1F13 1F18 1F1C-1F1D 1F21-1F22 1F26-1F27 1F2B-1F2C 1F30-1F31 1F35-1F36 "
                "1F3A-1F3B 1F3F-1F40 1F44-1F45 1F49-1F4A 1F4E-1F4F 1F53-1F54 1F56 1F5B 1F60 1F65 "
                "1F6A 1F6F 1F74 1F79 1F7E 1F84 1F89 1F8E 1F93 1F98 "
                "1F9C-1F9D 1FA1-1FA2 1FA6-1FA7 1FAB-1FAC 1FB0-1FB1 1FB5-1FB6 1FBA-1FBB 1FBF-1FC0 1FC4-1FC5 1FC9-1FCA "
                "1FCE-1FCF 1FD3-1FD4 1FD6 1FDB 1FE0 1FE5 1FEA 1FEF 1FF4 1FF9 "
                "1FFE 2004 2009 200E 2013 2018 201C-201D 2021-2022 2026-2027 202B-202C "
                "2030-2031 2035-2036 203A-203B 203F-2040 2044-2045 2049-204A 204E-204F 2053-2054 2056 205B "
                "2060 2065 206A 206F 2074 2079 207E 2084 2089 208E "
                "2093 2098 209C-209D 20A1-20A2 20A6-20A7 20AB-20AC 20B0-20B1 20B5-20B6 20BA-20BB 20BF-20C0 "
                "20C4-20C5 20C9-20CA 20CE-20CF 20D3-20D4 20D6 20DB 20E0 20E5 20EA 20EF "
                "20F4 20F9 20FE 2104 2109 210E 2113 2118 211C-211D 2121-2122 "
                "2126-2127 212B-212C 2130-2131 2135-2136 213A-213B 213F-2140 2144-2145 2149-214A 214E-214F 2153-2154 "
                "2156 215B 2160 2165 216A 216F 2174 2179 217E 2184 "
                "2189 218E 2193 2198 219C-219D 21A1-21A2 21A6-21A7 21AB-21AC 21B0-21B1 21B5-21B6 "
                "21BA-21BB 21BF-21C0 21C4-21C5 21C9-21CA 21CE-21CF 21D3-21D4 21D6 21DB 21E0 21E5 "
                "21EA 21EF 21F4 21F9 21FE 2204 2209 220E 2213 2218 "
                "221C-221D 2221-2222 2226-2227 222B-222C 2230-2231 2235-2236 223A-223B 223F-2240 2244-2245 2249-224A "
                "224E-224F 2253-2254 2256 225B 2260 2265 226A 226F 2274 2279 "
                "227E 2284 2289 228E 2293 2298 229C-229D 22A1-22A2 22A6-22A7 22AB-22AC "
                "22B0-22B1 22B5-22B6 22BA-22BB 22BF-22C0 22C4-22C5 22C9-22CA 22CE-22CF 22D3-22D4 22D6 22DB "
                "22E0 22E5 22EA 22EF 22F4 22F9 22FE 2304 2309 230E "
                "2313 2318 231C-231D 2321-2322 2326-2327 232B-232C 2330-2331 2335-2336 233A-233B 233F-2340 "
                "2344-2345 2349-234A 234E-234F 2353-2354 2356 235B 2360 2365 236A 236F "
                "2374 2379 237E 2384 2389 238E 2393 2398 239C-239D 23A1-23A2 "
                "23A6-23A7 23AB-23AC 23B0-23B1 23B5-23B6 23BA-23BB 23BF-23C0 23C4-23C5 23C9-23CA 23CE-23CF 23D3-23D4 "
                "23D6 23DB 23E0 23E5 23EA 23EF 23F4 23F9 23FE 2404 "
                "2409 240E 2413 2418 241C-241D 2421-2422 2426-2427 242B-242C 2430-2431 2435-2436 "
                "243A-243B 243F-2440 2444-2445 2449-244A 244E-244F 2453-2454 2456 245B 2460 2465 "
                "246A 246F 2474 2479 247E 2484 2489 248E 2493 2498 "
                "249C-249D 24A1-24A2 24A6-24A7 24AB-24AC 24B0-24B1 24B5-24B6 24BA-24BB 24BF-24C0 24C4-24C5 24C9-24CA "
                "24CE-24CF 24D3-24D4 24D6 24DB 24E0 24E5 24EA 24EF 24F4 24F9 "
                "24FE 2504 2509 250E 2513 2518 251C-251D 2521-2522 2526-2527 252B-252C "
                "2530-2531 2535-2536 253A-253B 253F-2540 2544-2545 2549-254A 254E-254F 2553-2554 2556 255B "
                "2560 2565 256A 256F 2574 2579 257E 2584 2589 258E "
                "2593 2598 259C-259D 25A1-25A2 25A6-25A7 25AB-25AC 25B0-25B1 25B5-25B6 25BA-25BB 25BF-25C0 "
                "25C4-25C5 25C9-25CA 25CE-25CF 25D3-25D4 25D6 25DB 25E0 25E5 25EA 25EF "
                "25F4 25F9 25FE 2604 2609 260E 2613 2618 261C-261D 2621-2622 "
                "2626-2627 262B-262C 2630-2631 2635-2636 263A-263B 263F-2640 2644-2645 2649-264A 264E-264F 2653-2654 "
                "2656 265B 2660 2665 266A 266F 2674 2679 267E 2684 "
                "2689 268E 2693 2698 269C-269D 26A1-26A2 26A6-26A7 26AB-26AC 26B0-26B1 26B5-26B6 "
                "26BA-26BB 26BF-26C0 26C4-26C5 26C9-26CA 26CE-26CF 26D3-26D4 26D6 26DB 26E0 26E5 "
                "26EA 26EF 26F4 26F9 26FE 2704 2709 270E 2713 2718 "
                "271C-271D 2721-2722 2726-2727 272B-272C 2730-2731 2735-2736 273A-273B 273F-2740 2744-2745 2749-274A "
                "274E-274F 2753-2754 2756 275B 2760 2765 276A 276F 2774 2779 "
                "277E 2784 2789 278E 2793 2798 279C-279D 27A1-27A2 27A6-27A7 27AB-27AC "
                "27B0-27B1 27B5-27B6 27BA-27BB 27BF-27C0 27C4-27C5 27C9-27CA 27CE-27CF 27D3-27D4 27D6 27DB "
                "27E0 27E5 27EA 27EF 27F4 27F9 27FE 2804 2809 280E "
                "2813 2818 281C-281D 2821-2822 2826-2827 282B-282C 2830-2831 2835-2836 283A-283B 283F-2840 "
                "2844-2845 2849-284A 284E-284F 2853-2854 2856 285B 2860 2865 286A 286F "
                "2874 2879 287E 2884 2889 288E 2893 2898 289C-289D 28A1-28A2 "
                "28A6-28A7 28AB-28AC 28B0-28B1 28B5-28B6 28BA-28BB 28BF-28C0 28C4-28C5 28C9-28CA 28CE-28CF 28D3-28D4 "
                "28D6 28DB 28E0 28E5 28EA 28EF 28F4 28F9 28FE 2904 "
                "2909 290E 2913 2918 291C-291D 2921-2922 2926-2927 292B-292C 2930-2931 2935-2936 "
                "293A-293B 293F-2940 2944-2945 2949-294A 294E-294F 2953-2954 2956 295B 2960 2965 "
                "296A 296F 2974 2979 297E 2984 2989 298E 2993 2998 "
                "299C-299D 29A1-29A2 29A6-29A7 29AB-29AC 29B0-29B1 29B5-29B6 29BA-29BB 29BF-29C0 29C4-29C5 29C9-29CA "
                "29CE-29CF 29D3-29D4 29D6 29DB 29E0 29E5 29EA 29EF 29F4 29F9 "
                "29FE 2A04 2A09 2A0E 2A13 2A18 2A1C-2A1D 2A21-2A22 2A26-2A27 2A2B-2A2C "
                "2A30-2A31 2A35-2A36 2A3A-2A3B 2A3F-2A40 2A44-2A45 2A49-2A4A 2A4E-2A4F 2A53-2A54 2A56 2A5B "
                "2A60 2A65 2A6A 2A6F 2A74 2A79 2A7E 2A84 2A89 2A8E "
                "2A93 2A98 2A9C-2A9D 2AA1-2AA2 2AA6-2AA7 2AAB-2AAC 2AB0-2AB1 2AB5-2AB6 2ABA-2ABB 2ABF-2AC0 "
                "2AC4-2AC5 2AC9-2ACA 2ACE-2ACF 2AD3-2AD4 2AD6 2ADB 2AE0 2AE5 2AEA 2AEF "
                "2AF4 2AF9 2AFE 2B04 2B09 2B0E 2B13 2B18 2B1C-2B1D 2B21-2B22 "
                "2B26-2B27 2B2B-2B2C 2B30-2B31 2B35-2B36 2B3A-2B3B 2B3F-2B40 2B44-2B45 2B49-2B4A 2B4E-2B4F 2B53-2B54 "
                "2B56 2B5B 2B60 2B65 2B6A 2B6F 2B74 2B79 2B7E 2B84 "
                "2B89 2B8E 2B93 2B98 2B9C-2B9D 2BA1-2BA2 2BA6-2BA7 2BAB-2BAC 2BB0-2BB1 2BB5-2BB6 "
                "2BBA-2BBB 2BBF-2BC0 2BC4-2BC5 2BC9-2BCA 2BCE-2BCF 2BD3-2BD4 2BD6 2BDB 2BE0 2BE5 "
                "2BEA 2BEF 2BF4 2BF9 2BFE 2C04 2C09 2C0E 2C13 2C18 "
                "2C1C-2C1D 2C21-2C22 2C26-2C27 2C2B-2C2C 2C30-2C31 2C35-2C36 2C3A-2C3B 2C3F-2C40 2C44-2C45 2C49-2C4A "
                "2C4E-2C4F 2C53-2C54 2C56 2C5B 2C60 2C65 2C6A 2C6F 2C74 2C79 "
                "2C7E 2C84 2C89 2C8E 2C93 2C98 2C9C-2C9D 2CA1-2CA2 2CA6-2CA7 2CAB-2CAC "
                "2CB0-2CB1 2CB5-2CB6 2CBA-2CBB 2CBF-2CC0 2CC4-2CC5 2CC9-2CCA 2CCE-2CCF 2CD3-2CD4 2CD6 2CDB "
                "2CE0 2CE5 2CEA 2CEF 2CF4 2CF9 2CFE 2D04 2D09 2D0E "
                "2D13 2D18 2D1C-2D1D 2D21-2D22 2D26-2D27 2D2B-2D2C 2D30-2D31 2D35-2D36 2D3A-2D3B 2D3F-2D40 "
                "2D44-2D45 2D49-2D4A 2D4E-2D4F 2D53-2D54 2D56 2D5B 2D60 2D65 2D6A 2D6F "
                "2D74 2D79 2D7E 2D84 2D89 2D8E 2D93 2D98 2D9C-2D9D 2DA1-2DA2 "
                "2DA6-2DA7 2DAB-2DAC 2DB0-2DB1 2DB5-2DB6 2DBA-2DBB 2DBF-2DC0 2DC4-2DC5 2DC9-2DCA 2DCE-2DCF 2DD3-2DD4 "
                "2DD6 2DDB 2DE0 2DE5 2DEA 2DEF 2DF4 2DF9 2DFE 2E04 "
                "2E09 2E0E 2E13 2E18 2E1C-2E1D 2E21-2E22 2E26-2E27 2E2B-2E2C 2E30-2E31 2E35-2E36 "
                "2E3A-2E3B 2E3F-2E40 2E44-2E45 2E49-2E4A 2E4E-2E4F 2E53-2E54 2E56 2E5B 2E60 2E65 "
                "2E6A 2E6F 2E74 2E79 2E7E 2E84 2E89 2E8E 2E93 2E98 "
                "2E9C-2E9D 2EA1-2EA2 2EA6-2EA7 2EAB-2EAC 2EB0-2EB1 2EB5-2EB6 2EBA-2EBB 2EBF-2EC0 2EC4-2EC5 2EC9-2ECA "
                "2ECE-2ECF 2ED3-2ED4 2ED6 2EDB 2EE0 2EE5 2EEA 2EEF 2EF4 2EF9 "
                "2EFE 2F04 2F09 2F0E 2F13 2F18 2F1C-2F1D 2F21-2F22 2F26-2F27 2F2B-2F2C "
                "2F30-2F31 2F35-2F36 2F3A-2F3B 2F3F-2F40 2F44-2F45 2F49-2F4A 2F4E-2F4F 2F53-2F54 2F56 2F5B "
                "2F60 2F65 2F6A 2F6F 2F74 2F79 2F7E 2F84 2F89 2F8E "
                "2F93 2F98 2F9C-2F9D 2FA1-2FA2 2FA6-2FA7 2FAB-2FAC 2FB0-2FB1 2FB5-2FB6 2FBA-2FBB 2FBF-2FC0 "
                "2FC4-2FC5 2FC9-2FCA 2FCE-2FCF 2FD3-2FD4 2FD6 2FDB 2FE0 2FE5 2FEA 2FEF "
                "2FF4 2FF9 2FFE 3004 3009 300E 3013 3018 301C-301D 3021-3022 "
                "3026-3027 302B-302C 3030-3031 3035-3036 303A-303B 303F-3040 3044-3045 3049-304A 304E-304F 3053-3054 "
                "3056 305B 3060 3065 306A 306F 3074 3079 307E 3084 "
                "3089 308E 3093 3098 309C-309D 30A1-30A2 30A6-30A7 30AB-30AC 30B0-30B1 30B5-30B6 "
                "30BA-30BB 30BF-30C0 30C4-30C5 30C9-30CA 30CE-30CF 30D3-30D4 30D6 30DB 30E0 30E5 "
                "30EA 30EF 30F4 30F9 30FE 3104 3109 310E 3113 3118 "
                "311C-311D 3121-3122 3126-3127 312B-312C 3130-3131 3135-3136 313A-313B 313F-3140 3144-3145 3149-314A "
                "314E-314F 3153-3154 3156 315B 3160 3165 316A 316F 3174 3179 "
                "317E 3184 3189 318E 3193 3198 319C-319D 31A1-31A2 31A6-31A7 31AB-31AC "
                "31B0-31B1 31B5-31B6 31BA-31BB 31BF-31C0 31C4-31C5 31C9-31CA 31CE-31CF 31D3-31D4 31D6 31DB "
                "31E0 31E5 31EA 31EF 31F4 31F9 31FE 3204 3209 320E "
                "3213 3218 321C-321D 3221-3222 3226-3227 322B-322C 3230-3231 3235-3236 323A-323B 323F-3240 "
                "3244-3245 3249-324A 324E-324F 3253-3254 3256 325B 3260 3265 326A 326F "
                "3274 3279 327E 3284 3289 328E 3293 3298 329C-329D 32A1-32A2 "
                "32A6-32A7 32AB-32AC 32B0-32B1 32B5-32B6 32BA-32BB 32BF-32C0 32C4-32C5 32C9-32CA 32CE-32CF 32D3-32D4 "
                "32D6 32DB 32E0 32E5 32EA 32EF 32F4 32F9 32FE 3304 "
                "3309 330E 3313 3318 331C-331D 3321-3322 3326-3327 332B-332C 3330-3331 3335-3336 "
                "333A-333B 333F-3340 3344-3345 3349-334A 334E-334F 3353-3354 3356 335B 3360 3365 "
                "336A 336F 3374 3379 337E 3384 3389 338E 3393 3398 "
                "339C-339D 33A1-33A2 33A6-33A7 33AB-33AC 33B0-33B1 33B5-33B6 33BA-33BB 33BF-33C0 33C4-33C5 33C9-33CA "
                "33CE-33CF 33D3-33D4 33D6 33DB 33E0 33E5 33EA 33EF 33F4 33F9 "
                "33FE 3404 3409 340E 3413 3418 341C-341D 3421-3422 3426-3427 342B-342C "
                "3430-3431 3435-3436 343A-343B 343F-3440 3444-3445 3449-344A 344E-344F 3453-3454 3456 345B "
                "3460 3465 346A 346F 3474 3479 347E 3484 3489 348E "
                "3493 3498 349C-349D 34A1-34A2 34A6-34A7 34AB-34AC 34B0-34B1 34B5-34B6 34BA-34BB 34BF-34C0 "
                "34C4-34C5 34C9-34CA 34CE-34CF 34D3-34D4 34D6 34DB 34E0 34E5 34EA 34EF "
                "34F4 34F9 34FE 3504 3509 350E 3513 3518 351C-351D 3521-3522 "
                "3526-3527 352B-352C 3530-3531 3535-3536 353A-353B 353F-3540 3544-3545 3549-354A 354E-354F 3553-3554 "
                "3556 355B 3560 3565 356A 356F 3574 3579 357E 3584 "
                "3589 358E 3593 3598 359C-359D 35A1-35A2 35A6-35A7 35AB-35AC 35B0-35B1 35B5-35B6 "
                "35BA-35BB 35BF-35C0 35C4-35C5 35C9-35CA 35CE-35CF 35D3-35D4 35D6 35DB 35E0 35E5 "
                "35EA 35EF 35F4 35F9 35FE 3604 3609 360E 3613 3618 "
                "361C-361D 3621-3622 3626-3627 362B-362C 3630-3631 3635-3636 363A-363B 363F-3640 3644-3645 3649-364A "
                "364E-364F 3653-3654 3656 365B 3660 3665 366A 366F 3674 3679 "
                "367E 3684 3689 368E 3693 3698 369C-369D 36A1-36A2 36A6-36A7 36AB-36AC "
                "36B0-36B1 36B5-36B6 36BA-36BB 36BF-36C0 36C4-36C5 36C9-36CA 36CE-36CF 36D3-36D4 36D6 36DB "
                "36E0 36E5 36EA 36EF 36F4 36F9 36FE 3704 3709 370E "
                "3713 3717-3718 371C-371D 3721-3722 3726-3727 372B-372C 3730-3731 3735-3736 373A-373B 373F-3740 "
                "3744-3745 3749-374A 374E-374F 3753-3754 3756 375B 3760 3765 376A 376F "
                "3774 3779 377E 3784 3789 378E 3793 3797-3798 379C-379D 37A1-37A2 "
                "37A6-37A7 37AB-37AC 37B0-37B1 37B5-37B6 37BA-37BB 37BF-37C0 37C4-37C5 37C9-37CA 37CE-37CF 37D3-37D4 "
                "37D6 37DB 37E0 37E5 37EA 37EF 37F4 37F9 37FE 3804 "
                "3809 380E 3813 3817-3818 381C-381D 3821-3822 3826-3827 382B-382C 3830-3831 3835-3836 "
                "383A-383B 383F-3840 3844-3845 3849-384A 384E-384F 3853-3854 3856 385B 3860 3865 "
                "386A 386F 3874 3879 387E 3884 3889 388E 3893 3897-3898 "
                "389C-389D 38A1-38A2 38A6-38A7 38AB-38AC 38B0-38B1 38B5-38B6 38BA-38BB 38BF-38C0 38C4-38C5 38C9-38CA "
                "38CE-38CF 38D3-38D4 38D6 38DB 38E0 38E5 38EA 38EF 38F4 38F9 "
                "38FE 3904 3909 390E 3912-3913 3917-3918 391C-391D 3921-3922 3926-3927 392B-392C "
                "3930-3931 3935-3936 393A-393B 393F-3940 3944-3945 3949-394A 394E-394F 3953-3954 3956 395B "
                "3960 3965 396A 396F 3974 3979 397E 3984 3989 398D-398E "
                "3992-3993 3997-3998 399C-399D 39A1-39A2 39A6-39A7 39AB-39AC 39B0-39B1 39B5-39B6 39BA-39BB 39BF-39C0 "
                "39C4-39C5 39C9-39CA 39CE-39CF 39D3-39D4 39D6 39DB 39E0 39E5 39EA 39EF "
                "39F4 39F9 39FE 3A04 3A08-3A09 3A0D-3A0E 3A12-3A13 3A17-3A18 3A1C-3A1D 3A21-3A22 "
                "3A26-3A27 3A2B-3A2C 3A30-3A31 3A35-3A36 3A3A-3A3B 3A3F-3A40 3A44-3A45 3A49-3A4A 3A4D-3A4E 3A52-3A53 "
                "3A56 3A5B 3A60 3A65 3A6A 3A6F 3A72 3A74 3A77 3A79 "
                "3A7C 3A7E 3A83-3A84 3A88-3A89 3A8D-3A8E 3A92 3A97 3A9C 3AA1 3AA6 "
                "3AAB 3AB0 3AB4-3AB5 3AB9-3ABA 3ABE-3ABF 3AC3-3AC4 3AC8-3AC9 3ACD-3ACE 3AD2-3AD3 3AD9 "
                "3ADE 3AE3 3AE8 3AED 3AF2 3AF7 3AFC 3B03 3B08 3B0D "
                "3B12 3B16-3B17 3B1B-3B1C 3B20-3B21 3B25-3B26 3B2A-3B2B 3B2F 3B34 3B39 3B3D-3B3E "
                "3B42-3B43 3B47-3B48 3B4C-3B4D 3B51-3B52 3B59 3B5E 3B63 3B68 3B6D 3B70 "
                "3B72 3B75 3B77 3B7A 3B7F 3B82 3B87 3B8C 3B91 3B95-3B96 "
                "3B9A-3B9B 3B9F-3BA0 3BA4 3BA9 3BAE 3BB2-3BB3 3BB7-3BB8 3BBC-3BBD 3BC1 3BC5-3BC6 "
                "3BCA-3BCB 3BCF-3BD0 3BD4 3BD7 3BDC 3BE1 3BE4 3BE9 3BEE 3BF3 "
                "3BF8 3BFD 3C01 3C05 3C0A 3C0F 3C14 3C18 3C1D 3C22 "
                "3C26-3C27 3C2B 3C30 3C34-3C35 3C39 3C3E 3C42-3C43 3C47 3C4B-3C4C 3C50 "
                "3C5B 3C60 3C65 3C6A 3C72 3C77 3C7C 3C82 3C87 3C8B "
                "3C90 3C94 3C99 3C9D 3CA2 3CA6 3CAB 3CAF 3CB3 3CB8 "
                "3CBC 3CC0 3CC5 3CC9 3CCD 3CD2 3CD8 3CE5 3CED 3CF2 "
                "3CFA 3D01 3D1A 3D1E 3D22 3D26 3D3D 3D41 3D45 3D49 "
                "3D65 3D70 3D9B 3E98 3E9F 3EA4 3EAE 3EB3 3EBA 3EC5 "
                "3ECB 3ED8 3EDF 3EE6 3EED 3EF5 3EFD-3EFE 3F03 3F08 3F0D "
                "3F13 3F1A 3F32-3F34 3F89 3F9B 3FB3 B8D0 B8D5 B93C B941 "
                "B946 B94B B950 B955 B9A3 B9A8 B9AD B9B2 B9B7 B9BC "
                "B9C1 B9C6 B9CB B9D0 B9D5 BA0A BA0F BA14 BA19 BA1E "
                "BA23 BA28 BA2D BA32 BA37 BA3C BA41 BA46 BA58 BA5D "
                "BA62 BA67 BA6C BA71 BA76 BA7B BA80 BA85 BA8A BA8F "
                "BB1F BB24 BB29 BB2E BB52 BB69 BB6E BB73 BB9B BBBA "
                "BBD4 BBDC BBE1 BC03 BC43 BC6C BEE8 BEF5 BEFA BF04 "
                "BF37 BFA4-BFA5 BFB3 BFF1 BFF6 BFFD C020 C024 C028 C02B "
                "C04D C04F C052 C054 C05A C05F C07C C080 C097-C099 C0AB "
                "C0AF C0B2 C0C5 C0DE C0EB C0EF C0F4 C102 C10E C11A "
                "C125 C12E C130 C139 C13D C14F C151 C159 C169 C174 "
                "C17C C18F C1A6 C1AA C1B0 C1B7 C1BB C1C6 C1D1-C1D2 C1D7 "
                "C1DE C1EE C1F4 C1F7 C202 C20A C21B C221 C223 C237 "
                "C244 C247 C255 C258 C25E C273 C281 C286 C289 C29F "
                "C2A2 C2AD C2B4"
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

    actual = stored(ttnn.to_torch(ttnn.mish(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (torch.nn.functional.mish(x)).to(torch.float64).flatten()
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

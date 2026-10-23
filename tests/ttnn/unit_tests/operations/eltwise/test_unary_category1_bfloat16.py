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
def test_acosh_with_sqrt_in_sfpu_chain(device, generated_first):
    """acosh in one SFPU chain with stock sqrt, both orders.

    Each chain step runs the kernel its own TT-NN op runs, and the second op stores exactly what it
    stores standalone on the first op's device result, which the chain keeps in DEST.
    """
    generated = (ttnn.UnaryWithParam(ttnn.UnaryOpType.ACOSH), lambda tensor: ttnn.acosh(tensor))
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


def test_acosh_bf16_unflushed_zero_and_subnormal_inputs(device):
    """TT-NN's copy into DEST keeps -0 and subnormal BF16 inputs, which the LLK sweep's unpack delivers as +0.

    Each such input and +0 must store torch's result: its class as the BF16 pack stores it (NaN as
    +inf, -0 as +0, a subnormal result as 0) and a finite result within 1 ULP of torch's float64
    value. STOCK lists, per board, the inputs where the kernel stores its stock kernel's word
    instead, as {word: [(first input, last input), ...]} over the input bits.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.acosh keeps its stock kernel on {board}")
    STOCK = {}
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        t = torch.where(torch.isnan(t), torch.full_like(t, float("inf")), t)
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    actual = stored(ttnn.to_torch(ttnn.acosh(tt_x)))
    x64 = x.flatten().to(torch.float64)
    reference = ttnn.get_golden_function(ttnn.acosh)(x64).to(torch.float64).flatten()
    rounded = stored(reference)
    spacing = 2.0 ** (torch.floor(torch.log2(reference.abs().clamp(min=tiny))) - 7)
    close = torch.isfinite(reference) & (actual != 0) & ((actual - reference).abs() < spacing)
    ok = (actual == rounded) | close
    bits = words.repeat(4).to(torch.int32) & 0xFFFF
    for word, ranges in STOCK.get(board, {}).items():
        value = stored(torch.tensor([word], dtype=torch.int32).to(torch.int16).view(torch.bfloat16))
        for first, last in ranges:
            ok |= (bits >= first) & (bits <= last) & (actual == value)
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )


def test_acosh_bf16_bfloat8_b(device):
    """bfloat8_b in and out. The SFPU reads Bfp8_b data in DEST as BF16, so this runs the BF16 kernel, as the
    BF16 cases do. The result must correlate with torch's on the input as stored to PCC 0.999, as upstream's
    bfloat8_b unary cases require. Over every finite BF16 exponent, each output must also be within one bfloat8_b
    step of the BF16 call's for the same stored input, packed to bfloat8_b (two in a block that holds an infinity
    or NaN), and of its class where that is infinity or NaN: the op packs DEST to Bfp8_b, and the typecast
    rounds and sets a block's exponent its own way.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.acosh keeps its stock kernel on {board}")
    torch.manual_seed(0)
    x = torch.empty(1, 1, 64, 64).uniform_(1.01, 100.0).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.float64).flatten()
    output = ttnn.acosh(tt_x)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float64).flatten()
    golden = ttnn.get_golden_function(ttnn.acosh)(stored).to(torch.float64)
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
    actual = ttnn.to_torch(ttnn.acosh(tt_x)).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.acosh(tt_stored), ttnn.bfloat8_b)).to(torch.float32).flatten()
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


def test_acosh_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs through ttnn.acosh, the call TT-NN users make.

    Each must store torch's float64 result (torch.acosh(x)) as the BF16 pack stores it (NaN as an
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
        pytest.skip(f"ttnn.acosh keeps its stock kernel on {board}")
    STOCK = {}
    STEPS = 1
    SKIP = []
    MODEL = {
        "blackhole": {
            -1: (
                "3F82 3F8D 3F8F 3F91 3F99-3F9A 3FF1 4000 4009-4015 401E-4020 4025-4027 "
                "402B-402C 4030-4031 4034-4035 4038-4039 403C 4040 4043 4046 404A 4052 "
                "4055 4058 405D 4062 4070 414A 4157 415E 4165 4169 "
                "416C 4170 4174 4178 417C 4180 4182 4184 4186 4188 "
                "418A 418F 4191 4193 4198 419A 419D 419F 41A2 41A4 "
                "41A7 41A9-41AA 41AC 41AF 41B1-41B2 41B4 41B7 41BA 41BD 41C0 "
                "41C3 41C6 41C9 41CC 41CF-41D0 41D3 41D6 41D9-41DA 41DE-41DF 41E5-41E6 "
                "41ED 41F4 41FC 4202 4206 4213 42B6 42BC 42C2-42C3 42C8 "
                "430D 4388 4403 4407 448F 450C 4583 4602 4709 4780 "
                "4788 4810 4906 4985 4B07 4D04 4E8A 4F01 5087 5284 "
                "5481 5607 590B 5993 5B08 5D05 5F02 640B 6608 6805 "
                "6A02 6F82 7710 7885 7995"
            ),
            1: (
                "3F84 3FA0 3FA8 3FAB 3FAF-3FB2 3FB8-3FBC 3FC0-3FC1 3FC4-3FC6 3FCC 408B "
                "408D 4096 4098 409D 40A4 40A9 40AC 40B1 40B4 40B7 "
                "40BF 40C2 40C5 40C8 40CF 40D2 40D5 40D8 40DC 40DF "
                "40E3 40E6 40EA 40ED-40EE 40F1 40F5 40F9 40FD 410D 410F "
                "4116 411D 4122 412A 4135 4231 423C 4242 4248 424E-424F "
                "4255 425B-425C 4262-4263 4269-426A 4271-4272 4278-4279 4280 4284-4285 4289 428D "
                "4292 4296 429B 42A0 42CD-42CE 42D4 42DA-42DB 42E1-42E2 42E8-42E9 42EF-42F0 "
                "42F7-42F8 42FF 432F 433A 4340 4346 434C 4352-4353 4359-435A 4360-4361 "
                "4367-4368 436E-436F 4375-4377 437D-437E 43AE 43B9 43BF 43C5 43CB 43D1-43D2 "
                "43D8 43DE-43DF 43E5-43E6 43ED-43EE 43F4-43F5 43FC-43FD 442D 4438 443E 4444 "
                "444A 4450-4451 4457 445D-445E 4464-4465 446B-446C 4473-4474 447A-447C 44A7 44AC "
                "44B7 44C0 44CC 44D9 44E6-44E7 44F5-44F6 4533 453F 454B 4557-4558 "
                "4565-4566 4574-4575 45B2 45BE 45CA 45D6-45D7 45E4-45E5 45F2-45F4 4631 4648-4649 "
                "4655-4656 4663 4671-4672 46B0 46BB 46C7-46C8 46D4 46E1-46E2 46F0-46F1 46FF "
                "472F 473A 4746 4753 4760-4761 476E-476F 477E-477F 479A 47AE 47B9 "
                "47C5 47D2 47DF-47E0 47ED-47EE 47FC-47FD 4823 4838 4844 4850-4851 485E "
                "486C-486D 487B-487C 48A2 48B7 48C3 48CF-48D0 48DC-48DD 48EA-48EB 48F9-48FB 4921 "
                "4936 4942 494E-494F 495B-495C 4969-496A 4978-4979 49B5 49C1 49CD 49DA-49DB "
                "49E8-49E9 49F7-49F8 4A34 4A40 4A4C 4A59 4A67 4A75-4A76 4AB9 4AD1-4AD2 "
                "4AEC-4AED 4B2D 4B44 4B5D-4B5E 4B7A-4B7B 4BB7 4BCF 4BEA-4BEB 4C2B 4C41 "
                "4C5B 4C77-4C79 4CB5 4CCC-4CCD 4CE7-4CE8 4D29 4D3F 4D58-4D59 4D75-4D76 4D9E "
                "4DB3 4DCA-4DCB 4DE5-4DE6 4E27 4E3D 4E56 4E72-4E73 4EB1 4EC8 4EE2-4EE3 "
                "4F3B 4F53-4F54 4F6F-4F70 4FAF 4FC6 4FE0 4FFD-4FFE 5039 5051-5052 506D-506E "
                "50AD 50C4 50DD-50DE 50FA-50FB 5137 514F 516A-516B 51AB 51C1-51C2 51DB "
                "51F7-51F9 5235 524D 5267-5268 52A9 52BF 52D8-52D9 52F5-52F6 531E 5333 "
                "534A-534B 5365-5366 53BD 53D6 53F2-53F3 5431 5448 5462-5463 54BB 54D3-54D4 "
                "54EF-54F0 552F 5546 5560 557D-557E 55B9 55D1-55D2 55ED-55EE 5650 56AC "
                "56DC-56DD 5736 5769-576A 57C0-57C1 57F6-57F7 581F 584B-584C 58A8 58D7-58D8 5932 "
                "5964 59BC 59F1-59F2 5A47 5A7F 5AD2-5AD3 5B2E 5B5F 5BB8 5BEB-5BEC "
                "5C43 5C79-5C7A 5CCE 5D2A 5D5A 5DB4 5DE6-5DE7 5E3E 5E74-5E75 5EC9-5ECA "
                "5F55 5FB0 5FE1-5FE2 603A 606E-606F 60C5 60FC-60FD 6150-6151 61AC 61DC-61DD "
                "6236 6269-626A 62C1 62F6-62F8 634C 63D7-63D8 6432 6464-6465 64BC 64F1-64F2 "
                "6547 657F 65D3 662E 665F 66B8 66EC-66ED 6743 6779-677A 67A1 "
                "67CE 685A 68B4 68E6-68E7 693F 6974-6975 69C9-69CA 6A55-6A56 6AB0 6AE1-6AE2 "
                "6B3A 6B6E-6B6F 6BC5 6BFC-6BFD 6C23 6C50-6C51 6CAC 6CDC-6CDD 6D4E 6E66-6E67 "
                "6EBF 6FD5-6FD6 7030 70EE-70EF 7145 71A3 725C-725D 72B6 7377-7378 73CC "
                "74E4-74E5 753D 75FF 7653 776C-776D 77C3 78DA-78DB 7934 79F4-79F5 7A4A "
                "7B62 7BBB 7C7C-7C7E 7CD1 7DE9-7DEA 7E41 7F58"
            ),
        },
        "wormhole_b0": {
            -1: (
                "3F82 3F8D 3F8F 3F91 3F99-3F9A 3FF1 4000 4009-4015 401E-4020 4025-4027 "
                "402B-402C 4030-4031 4034-4035 4038-4039 403C 4040 4043 4046 404A 4052 "
                "4055 4058 405D 4062 4070 414A 4157 415E 4165 4169 "
                "416C 4170 4174 4178 417C 4180 4182 4184 4186 4188 "
                "418A 418F 4191 4193 4198 419A 419D 419F 41A2 41A4 "
                "41A7 41A9-41AA 41AC 41AF 41B1-41B2 41B4 41B7 41BA 41BD 41C0 "
                "41C3 41C6 41C9 41CC 41CF-41D0 41D3 41D6 41D9-41DA 41DE-41DF 41E5-41E6 "
                "41ED 41F4 41FC 4202 4206 4213 42B6 42BC 42C2-42C3 42C8 "
                "430D 4388 4403 4407 448F 450C 4583 4602 4709 4780 "
                "4788 4810 4906 4985 4B07 4D04 4E8A 4F01 5087 5284 "
                "5481 5607 590B 5993 5B08 5D05 5F02 640B 6608 6805 "
                "6A02 6F82 7710 7885 7995"
            ),
            1: (
                "3F84 3FA0 3FA8 3FAB 3FAF-3FB2 3FB8-3FBC 3FC0-3FC1 3FC4-3FC6 3FCC 408B "
                "408D 4096 4098 409D 40A4 40A9 40AC 40B1 40B4 40B7 "
                "40BF 40C2 40C5 40C8 40CF 40D2 40D5 40D8 40DC 40DF "
                "40E3 40E6 40EA 40ED-40EE 40F1 40F5 40F9 40FD 410D 410F "
                "4116 411D 4122 412A 4135 4231 423C 4242 4248 424E-424F "
                "4255 425B-425C 4262-4263 4269-426A 4271-4272 4278-4279 4280 4284-4285 4289 428D "
                "4292 4296 429B 42A0 42CD-42CE 42D4 42DA-42DB 42E1-42E2 42E8-42E9 42EF-42F0 "
                "42F7-42F8 42FF 432F 433A 4340 4346 434C 4352-4353 4359-435A 4360-4361 "
                "4367-4368 436E-436F 4375-4377 437D-437E 43AE 43B9 43BF 43C5 43CB 43D1-43D2 "
                "43D8 43DE-43DF 43E5-43E6 43ED-43EE 43F4-43F5 43FC-43FD 442D 4438 443E 4444 "
                "444A 4450-4451 4457 445D-445E 4464-4465 446B-446C 4473-4474 447A-447C 44A7 44AC "
                "44B7 44C0 44CC 44D9 44E6-44E7 44F5-44F6 4533 453F 454B 4557-4558 "
                "4565-4566 4574-4575 45B2 45BE 45CA 45D6-45D7 45E4-45E5 45F2-45F4 4631 4648-4649 "
                "4655-4656 4663 4671-4672 46B0 46BB 46C7-46C8 46D4 46E1-46E2 46F0-46F1 46FF "
                "472F 473A 4746 4753 4760-4761 476E-476F 477E-477F 479A 47AE 47B9 "
                "47C5 47D2 47DF-47E0 47ED-47EE 47FC-47FD 4823 4838 4844 4850-4851 485E "
                "486C-486D 487B-487C 48A2 48B7 48C3 48CF-48D0 48DC-48DD 48EA-48EB 48F9-48FB 4921 "
                "4936 4942 494E-494F 495B-495C 4969-496A 4978-4979 49B5 49C1 49CD 49DA-49DB "
                "49E8-49E9 49F7-49F8 4A34 4A40 4A4C 4A59 4A67 4A75-4A76 4AB9 4AD1-4AD2 "
                "4AEC-4AED 4B2D 4B44 4B5D-4B5E 4B7A-4B7B 4BB7 4BCF 4BEA-4BEB 4C2B 4C41 "
                "4C5B 4C77-4C79 4CB5 4CCC-4CCD 4CE7-4CE8 4D29 4D3F 4D58-4D59 4D75-4D76 4D9E "
                "4DB3 4DCA-4DCB 4DE5-4DE6 4E27 4E3D 4E56 4E72-4E73 4EB1 4EC8 4EE2-4EE3 "
                "4F3B 4F53-4F54 4F6F-4F70 4FAF 4FC6 4FE0 4FFD-4FFE 5039 5051-5052 506D-506E "
                "50AD 50C4 50DD-50DE 50FA-50FB 5137 514F 516A-516B 51AB 51C1-51C2 51DB "
                "51F7-51F9 5235 524D 5267-5268 52A9 52BF 52D8-52D9 52F5-52F6 531E 5333 "
                "534A-534B 5365-5366 53BD 53D6 53F2-53F3 5431 5448 5462-5463 54BB 54D3-54D4 "
                "54EF-54F0 552F 5546 5560 557D-557E 55B9 55D1-55D2 55ED-55EE 5650 56AC "
                "56DC-56DD 5736 5769-576A 57C0-57C1 57F6-57F7 581F 584B-584C 58A8 58D7-58D8 5932 "
                "5964 59BC 59F1-59F2 5A47 5A7F 5AD2-5AD3 5B2E 5B5F 5BB8 5BEB-5BEC "
                "5C43 5C79-5C7A 5CCE 5D2A 5D5A 5DB4 5DE6-5DE7 5E3E 5E74-5E75 5EC9-5ECA "
                "5F55 5FB0 5FE1-5FE2 603A 606E-606F 60C5 60FC-60FD 6150-6151 61AC 61DC-61DD "
                "6236 6269-626A 62C1 62F6-62F8 634C 63D7-63D8 6432 6464-6465 64BC 64F1-64F2 "
                "6547 657F 65D3 662E 665F 66B8 66EC-66ED 6743 6779-677A 67A1 "
                "67CE 685A 68B4 68E6-68E7 693F 6974-6975 69C9-69CA 6A55-6A56 6AB0 6AE1-6AE2 "
                "6B3A 6B6E-6B6F 6BC5 6BFC-6BFD 6C23 6C50-6C51 6CAC 6CDC-6CDD 6D4E 6E66-6E67 "
                "6EBF 6FD5-6FD6 7030 70EE-70EF 7145 71A3 725C-725D 72B6 7377-7378 73CC "
                "74E4-74E5 753D 75FF 7653 776C-776D 77C3 78DA-78DB 7934 79F4-79F5 7A4A "
                "7B62 7BBB 7C7C-7C7E 7CD1 7DE9-7DEA 7E41 7F58"
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

    actual = stored(ttnn.to_torch(ttnn.acosh(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (torch.acosh(x)).to(torch.float64).flatten()
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

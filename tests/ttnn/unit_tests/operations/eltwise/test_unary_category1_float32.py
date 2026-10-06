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
    generate_float32_bits,
    generate_float32_bits_in_range,
    flush_to_zero,
    to_tt_tensor as _upload,
    MAX_BF16,
    SMALLEST_NORMAL_BF16,
)

pytestmark = pytest.mark.use_module_device


def to_tt_tensor(input_tensor, device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Upload as ttnn.float32. The bfloat16 category files use the bfloat16 default of _upload."""
    return _upload(
        input_tensor,
        device,
        dtype=ttnn.float32,
        layout=layout,
        memory_config=memory_config,
    )


"""
Float32 accuracy sweep of the same ops and input ranges as
test_unary_category1_bfloat16.py.

Inputs are the bfloat16 lattice stored as float32
(generate_float32_bits / generate_float32_bits_in_range): all 65,536
bfloat16 encodings, promoted without rounding. The op runs at ttnn.float32
and ULP thresholds below are float32 ULPs.


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
    input_tensor = generate_float32_bits(include_spl_values=True)

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
    input_tensor = generate_float32_bits(include_spl_values=True)

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
    input_tensor = generate_float32_bits()
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
    input_tensor = generate_float32_bits(include_spl_values=include_spl_values)

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
    input_tensor = generate_float32_bits()

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_equal(result, golden)


# ─────────────────────────────────────────────────────────────────────────────
# Trigonometric and hyperbolic ops
# Each op is tested with its valid input domain using generate_float32_bits_in_range
# ─────────────────────────────────────────────────────────────────────────────


# Bounds sit just above the measured max |err| and max |err|/|device|.
# Relative error is about 1.2e-7 for every op. Absolute error follows the
# output magnitude, so sinh and cosh near ±9 need a larger atol.
@pytest.mark.parametrize(
    "ttnn_op, low, high, atol, rtol",
    [
        (ttnn.acos, -1.0, 1.0, 2.4e-7, 1.2e-7),
        (ttnn.cos, -10.0, 10.0, 1.2e-7, 1.2e-7),
        (ttnn.acosh, 1.0, 100.0, 4.8e-7, 1.2e-7),
        (ttnn.asinh, -100.0, 100.0, 4.8e-7, 1.3e-7),
        (ttnn.sin, -10.0, 10.0, 6.0e-8, 1.2e-7),
        (ttnn.sinh, -9.0, 9.0, 2.5e-4, 1.3e-7),
        (ttnn.cosh, -9.0, 9.0, 2.5e-4, 1.2e-7),
        (ttnn.tan, -1.45, 1.45, 4.8e-7, 1.2e-7),
    ],
)
def test_trig_ops(device, ttnn_op, low, high, atol, rtol):
    input_tensor = generate_float32_bits_in_range(low, high)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_allclose(expected_result=golden, actual_result=result, atol=atol, rtol=rtol)


# ─────────────────────────────────────────────────────────────────────────────
# Trig ops with output FTZ handling (asin, atan, atanh)
# These ops produce near-zero outputs for small inputs where the device returns
# the smallest normal (2^-126) instead of exact zero. We zero out the device
# result where golden is 0 and device produced the smallest normal.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, low, high, ulp_threshold",
    [
        (ttnn.asin, -1.0, 1.0, 1),
        # 10 values land on exactly 2 float32 ULPs. Worst is x=±0.7109375
        # (0.618028998 vs 0.618028879). Nothing is above 2.
        (ttnn.atan, -100.0, 100.0, 2),
    ],
)
def test_trig_ops_out_ftz(device, ttnn_op, low, high, ulp_threshold):
    input_tensor = generate_float32_bits_in_range(low, high)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    # Device SFPU cannot produce exact zero — it returns the smallest normal instead.
    # Flush both outputs: anything at or below smallest normal becomes zero.
    result = flush_to_zero(result)
    golden = flush_to_zero(golden)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp_threshold)


# ─────────────────────────────────────────────────────────────────────────────
# atanh - uses PCC due to inherently higher SFPU error
# ─────────────────────────────────────────────────────────────────────────────


def test_atanh(device):
    input_tensor = generate_float32_bits_in_range(-100, 100)

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
    input_tensor = generate_float32_bits_in_range(low, high)

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
    input_tensor = generate_float32_bits_in_range(low, high)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_pcc(golden, result, 0.999)


# ─────────────────────────────────────────────────────────────────────────────
# reciprocal: 1/x, swept over the bfloat16 lattice stored as float32
#
# Float32 keeps 1/2^126, which is the smallest normal. The bfloat16 sweep
# flushed that value to zero. Measured on wormhole:
#   |x| < 2^126   within 1 float32 ULP of torch
#   |x| = 2^126   exact signed smallest normal
#   |x| > 2^126   signed zero (the reciprocal is subnormal and is flushed)
# ─────────────────────────────────────────────────────────────────────────────

RECIPROCAL_FTZ_INPUT = 2.0**126
RECIPROCAL_MAX_INPUT = RECIPROCAL_FTZ_INPUT * (1 - 2.0**-8)  # largest bfloat16 below it


@pytest.mark.parametrize(
    "low, high",
    [
        (SMALLEST_NORMAL_BF16, RECIPROCAL_MAX_INPUT),
        (-RECIPROCAL_MAX_INPUT, -SMALLEST_NORMAL_BF16),
    ],
    ids=["positive", "negative"],
)
def test_reciprocal(device, low, high):
    """Bfloat16-lattice inputs of one sign with |x| < 2^126.

    1/x stays a normal float32 on this side of the cutoff, from 2^126 at the
    small-input end down to just above the smallest normal.
    """
    input_tensor = generate_float32_bits_in_range(low, high)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.reciprocal)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.reciprocal(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


@pytest.mark.parametrize(
    "low, high",
    [
        (RECIPROCAL_FTZ_INPUT, MAX_BF16),
        (-MAX_BF16, -RECIPROCAL_FTZ_INPUT),
    ],
    ids=["positive", "negative"],
)
def test_reciprocal_flushes_to_zero(device, low, high):
    """|x| >= 2^126. One input per sign is the boundary; the rest flush.

    1/2^126 is the smallest normal, and float32 returns it with the input's
    sign. Every larger magnitude on this lattice has a subnormal reciprocal,
    which the device flushes to zero while keeping the sign, so the negative
    half is -0.
    """
    input_tensor = generate_float32_bits_in_range(low, high)
    golden = ttnn.get_golden_function(ttnn.reciprocal)(input_tensor, device=device)
    result = ttnn.to_torch(ttnn.reciprocal(to_tt_tensor(input_tensor, device)))

    assert (golden != 0).all(), "expected every reciprocal in this range to be nonzero in torch"

    at_boundary = input_tensor.abs() == RECIPROCAL_FTZ_INPUT
    flushed = input_tensor.abs() > RECIPROCAL_FTZ_INPUT
    assert at_boundary.any() and flushed.any()

    assert_equal(golden[at_boundary], result[at_boundary])

    flushed_result = result[flushed]
    assert (flushed_result == 0).all(), "subnormal reciprocals must flush to zero"
    assert torch.equal(
        torch.signbit(flushed_result), torch.signbit(input_tensor[flushed])
    ), "flushed reciprocal keeps the input sign, including -0"


def test_reciprocal_zero_and_nonfinite(device):
    """Zeros and infinities match torch, including the sign of 1/-inf.

    NaN still becomes +0. That is the device's current behavior, not an
    endorsement of it. The bfloat16 sweep also lost the sign of 1/-inf; float32
    keeps -0.

        input    | torch (IEEE 754) | device
        ---------+------------------+----------------
        +0       | +inf             | +inf
        -0       | -inf             | -inf
        +inf     | +0               | +0
        -inf     | -0               | -0
        NaN      | NaN              | +0
    """
    special = [0.0, -0.0, float("inf"), float("-inf"), float("nan")]
    input_tensor = torch.ones(32, 32, dtype=torch.float32)
    input_tensor.view(-1)[: len(special)] = torch.tensor(special, dtype=torch.float32)

    result = ttnn.to_torch(ttnn.reciprocal(to_tt_tensor(input_tensor, device))).view(-1)
    pos_zero, neg_zero, pos_inf, neg_inf, nan = (result[i] for i in range(len(special)))

    assert pos_zero == float("inf") and not torch.signbit(pos_zero), "1/+0 must be +inf"
    assert neg_zero == float("-inf") and torch.signbit(neg_zero), "1/-0 must be -inf"

    assert pos_inf == 0.0 and not torch.signbit(pos_inf), "1/+inf is +0"
    assert neg_inf == 0.0 and torch.signbit(neg_inf), "1/-inf is -0"
    assert nan == 0.0 and not torch.signbit(nan), "the device returns +0 for NaN"


# ─────────────────────────────────────────────────────────────────────────────
# square: x^2, overflow when |x| > ~1.84e19; use (-1e19, 1e19)
# ─────────────────────────────────────────────────────────────────────────────


def test_square(device):
    input_tensor = generate_float32_bits_in_range(-1e19, 1e19)

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
# Golden is evaluated in float64 and cast back to float32. On the bfloat16
# lattice, 840 values are exactly 2 float32 ULPs (worst x=±1.414e-38) and
# none are worse.
# ─────────────────────────────────────────────────────────────────────────────


def test_cbrt(device):
    input_tensor = generate_float32_bits_in_range(-1e38, 1e38)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.cbrt)
    golden = golden_function(input_tensor.to(torch.float64)).to(torch.float32)

    tt_result = ttnn.cbrt(tt_in)
    result = ttnn.to_torch(tt_result).to(torch.float32)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2)


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
    input_tensor = generate_float32_bits_in_range(low, high)

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
    input_tensor = generate_float32_bits_in_range(-89, -87)

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
    input_tensor = generate_float32_bits_in_range(-127, -126)

    golden_function = ttnn.get_golden_function(ttnn.exp2)
    golden = golden_function(input_tensor, device=device)

    tt_in = to_tt_tensor(input_tensor, device)

    tt_result = ttnn.exp2(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_allclose(actual_result=result, expected_result=golden, atol=8.5e-39, rtol=0)


@pytest.mark.parametrize(
    "low, high, expected_atol, expected_rtol",
    [
        (-1.6 * 10**38, -0.28515625, 0, 1.2e-7),
        (-0.28515625, 0.69140625, 0, 1.2e-7),
        (0.69140625, 88.5, 0, 1.2e-7),
    ],
)
def test_expm1_allclose(low, high, expected_atol, expected_rtol, device):
    """expm1 sub-range allclose check.

    The ULP sweep in test_exp_ops covers [-87.0, 88.5]; this test extends the
    negative tail to -1.6e38 and checks three subdomains. Each is within 1
    float32 ULP. Max |err|/|device| is 1.19e-7, so rtol sits just above that
    and atol stays 0. Exact zeros in the middle range make the reported
    relative delta NaN and still pass.
    """
    input_tensor = generate_float32_bits_in_range(low, high)

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
    "ttnn_op, low, high, atol, rtol",
    [
        # ULP is unstable where digamma crosses zero (x ≈ 1.461). Measured on the
        # bfloat16 lattice in float32: max |err| = 1.43e-6, max |err|/|device| = 1.48e-4.
        (ttnn.digamma, 1.0, 102.0, 1.5e-6, 1.5e-4),
        # Worst point is x = 2.015625 (3.854 vs 3.807): max |err| = 4.76e-2,
        # max |err|/|device| = 1.25e-2.
        (ttnn.multigammaln, 1.6, 100.0, 5e-2, 1.3e-2),
    ],
)
def test_digamma_multigammaln(device, ttnn_op, low, high, atol, rtol):
    input_tensor = generate_float32_bits_in_range(low, high)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_allclose(expected_result=golden, actual_result=result, atol=atol, rtol=rtol)


def test_digamma_large_x(device):
    """Regression guard for digamma at large x (issue #45520: "behaves bad for x>1000").

    The LUT kernel is fit on [0.01, 102]; beyond it a Bernoulli asymptotic branch
    (ln(x) - 1/2x - 1/12x^2 + ...) restores the (1, inf) support the pre-LUT composite
    op had. ``test_digamma`` only exercises [2, 102], so this covers the LUT->asymptotic
    crossover (102) and several decades past x=1000.
    """
    xs = torch.tensor(
        [[101.0, 102.0, 103.0, 150.0, 500.0, 1000.0, 5000.0, 1e4, 5e4, 1e5, 5e5, 1e6, 1e7, float("inf")]],
        dtype=torch.float32,
    )
    golden = torch.digamma(xs.to(torch.float64)).to(torch.float32)
    input_tensor = ttnn.from_torch(xs, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.digamma(input_tensor))
    # Asymptotic branch from x=103: max |err| = 5.88e-3 at x=1e7, max |err|/|device| = 5.56e-4 at x=103.
    assert_allclose(expected_result=golden, actual_result=output_tensor, atol=6e-3, rtol=6e-4)


def test_digamma_small_x(device):
    """Guard the steep near-pole region [0.01, 2): psi has a pole at 0 (psi(x) ~ -1/x),
    the steepest part of the fitted domain. test_digamma only exercises [2, 102].
    Sample avoids the zero-crossing at x~=1.4616 where ULP is ill-defined.
    """
    xs = torch.tensor(
        [[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.25, 1.75, 1.9, 1.99]],
        dtype=torch.float32,
    )
    golden = torch.digamma(xs.to(torch.float64)).to(torch.float32)
    input_tensor = ttnn.from_torch(xs, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.digamma(input_tensor))
    # Worst point is x=0.01: max |err| = 6.03e-4, max |err|/|device| = 6.0e-6.
    assert_allclose(expected_result=golden, actual_result=output_tensor, atol=6.1e-4, rtol=6.1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# lgamma: defined for all reals except poles at 0, -1, -2, ...
# ─────────────────────────────────────────────────────────────────────────────


def test_lgamma(device):
    input_tensor = generate_float32_bits_in_range(-1000, 1000).flatten()
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
    "ttnn_op, low, high, atol, rtol",
    [
        # Worst x=10: max |err| = 0.630, max |err|/|device| = 2.24e-4.
        (ttnn.i0, -10.0, 10.0, 0.64, 2.3e-4),
        # After the flush below, worst abs is x=9.9375 (4.88e-4) and worst rel
        # is x=8 (1.07e-6). Unflushed, x≈2.3e-38 returns 0 against golden ≈1.17e-38.
        (ttnn.i1, -10.0, 10.0, 4.9e-4, 1.1e-6),
    ],
)
def test_bessel_ops(device, ttnn_op, low, high, atol, rtol):
    input_tensor = generate_float32_bits_in_range(low, high)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    # device returns the smallest normal (2^-126) instead of exact zero.
    result = flush_to_zero(result)
    golden = flush_to_zero(golden)

    assert_allclose(expected_result=golden, actual_result=result, atol=atol, rtol=rtol)

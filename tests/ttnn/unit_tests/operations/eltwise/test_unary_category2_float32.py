# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from mpmath import mp, mpf, tanh as mp_tanh
from tt_py_test_utils_common.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import (
    assert_allclose,
    assert_equal,
    assert_with_ulp,
    generate_all_bfloat16_bitpatterns,
)
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    assert_ftz_band,
    generate_float32_bits,
    generate_float32_bits_in_range,
    to_float32_tt_tensor,
    SMALLEST_NORMAL_BF16,
)

pytestmark = pytest.mark.use_module_device


"""
Float32 accuracy sweep of the same ops and input ranges as
test_unary_category2_bfloat16.py.

Inputs are the bfloat16 lattice stored as float32
(generate_float32_bits / generate_float32_bits_in_range): all 65,536
bfloat16 encodings, promoted without rounding. The op runs at ttnn.float32
and ULP thresholds below are float32 ULPs.


Category 2: basic_unary_activation (no extra parameters)
Neural network activation functions

| #  | Op               | Description                 |
|----|------------------|-----------------------------|
| 1  | ttnn.relu        | Rectified linear unit       |
| 2  | ttnn.relu6       | ReLU capped at 6            |
| 3  | ttnn.silu        | SiLU (Sigmoid Linear Unit)  |
| 4  | ttnn.swish       | Swish activation            |
| 5  | ttnn.hardmish    | Hard Mish activation        |
| 6  | ttnn.hardsigmoid | Hard sigmoid                |
| 7  | ttnn.hardswish   | Hard swish                  |
| 8  | ttnn.softsign    | Softsign                    |
| 9  | ttnn.log_sigmoid | Log sigmoid                 |
| 10 | ttnn.tanhshrink  | Tanh shrink                 |

Accuracy criteria (FTZ = flush-to-zero, SNB = smallest normal bfloat16)

| Op           | Input range            | Check                    | Reason                                     |
|--------------|------------------------|--------------------------|--------------------------------------------|
| relu, relu6  | all                    | exact                    | comparison + select, no rounding           |
| hardmish     | all                    | ULP ≤ 1                  | same fp32 formula as golden (measured 0)   |
|              | ±inf, NaN, ±0          | matches golden by value  | -0 returns +0 (sign dropped)               |
|              | subnormals             | exactly 0                | device flushes subnormal inputs            |
| hardsigmoid  | -3 < x < -2.25         | abs ≤ 2^-24              | x*(1/6) + 0.5 cancels near x = -3          |
|              | elsewhere              | ULP ≤ 1                  |                                            |
| hardswish    | x > ~5.65e37           | device returns x         | torch golden overflows to inf              |
|              | |x| < 2*SNB            | exactly 0                | x/2 underflow FTZ                          |
|              | -3 < x < -2.25         | abs ≤ 2^-22              | inherits hardsigmoid cancellation          |
|              | elsewhere              | ULP ≤ 2                  |                                            |
| silu, swish  | |x| < 2*SNB            | exactly 0                | x/2 underflow FTZ                          |
|              | -88.5 <= x <= -87.5    | exactly 0                | device sigmoid FTZ before torch's          |
|              | elsewhere              | ULP ≤ 3                  | fp32 sigmoid exp/reciprocal error          |
| softsign     | |x| > 1/SNB - 1        | exactly ±1               | saturates; reciprocal(1 + |x|) < SNB       |
|              | all                    | ULP ≤ 2                  | reciprocal + multiply rounding             |
| log_sigmoid  | x <= -4                | device returns x         | identity branch                            |
|              | x <= -13               | ULP ≤ 2                  | identity within 2 ULP of golden            |
|              | -4 < x <= 0            | abs ≤ 0.02               | degree-8 polynomial                        |
|              | 0 < x <= 4             | abs ≤ 0.005              | degree-8 polynomial                        |
|              | x > 4, golden normal   | rel ≤ 0.04               | fast-approx exp                            |
|              | golden < SNB           | exactly 0                | -exp(-x) underflow FTZ                     |
| tanhshrink   | |x| < 1, result < SNB  | exactly 0                | x^3/3 underflow FTZ (mpmath reference)     |
|              | |x| < 1, elsewhere     | ULP ≤ 3 vs mpmath        | torch golden cancels; mpmath doesn't       |
|              | 1 <= |x| < 1.625       | ULP ≤ 8                  | partial cancellation in x - tanh(x)        |
|              | |x| >= 1.625           | ULP ≤ 3                  | tanh via 2*sigmoid - 1 (BH max 3, WH 2)    |

Thresholds are tighter in absolute terms than the bfloat16 file but, being
float32 ULPs (2^16 times finer), expose kernel error the bfloat16 output
rounding hides.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Exact piecewise ops (relu, relu6) — comparison + select only, no rounding
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.relu,
        ttnn.relu6,
    ],
)
def test_exact_piecewise_ops(device, ttnn_op):
    """Float32 sweep over every normal bfloat16 value for relu = max(0, x) and relu6 = min(max(0, x), 6).

    | Input range | Check | Reason                                          |
    |-------------|-------|-------------------------------------------------|
    | all         | exact | every output is x, 0 or 6; no rounding involved |
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_equal(golden, result)


# ─────────────────────────────────────────────────────────────────────────────
# Piecewise-linear-with-division ops (hardmish, hardsigmoid)
# ─────────────────────────────────────────────────────────────────────────────

# hardsigmoid's x*(1/6) + 0.5 cancels as x -> -3, where the output tends to 0.
# Measured on the exhaustive sweep, ULP > 1 occurs only for -2.984 <= x <= -2.266.
HARDSIGMOID_CANCELLATION_LO = -3.0
HARDSIGMOID_CANCELLATION_HI = -2.25


def hardsigmoid_cancellation_band(input_tensor):
    x = input_tensor.float()
    return (x > HARDSIGMOID_CANCELLATION_LO) & (x < HARDSIGMOID_CANCELLATION_HI)


def test_hardmish(device):
    """Float32 sweep over every normal bfloat16 value for hardmish.

    | Input range | Check   | Reason                                                        |
    |-------------|---------|---------------------------------------------------------------|
    | all         | ULP ≤ 1 | for fp32 input the golden is plain x * clamp(x/2 + 1, 0, 1)   |
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.hardmish)
    golden = golden_function(input_tensor, device=device)

    result = ttnn.to_torch(ttnn.hardmish(tt_in))

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


def test_hardmish_special_values(device):
    """hardmish on the inputs the exhaustive sweep replaces with 0: ±inf, NaN, ±0 and subnormals.

    | Input               | Device returns | Golden returns | Check                                 |
    |---------------------|----------------|----------------|---------------------------------------|
    | +inf                | +inf           | +inf           | equal to golden                       |
    | -inf                | NaN            | NaN            | both NaN (-inf * clamp(...) = -inf*0) |
    | NaN                 | NaN            | NaN            | both NaN                              |
    | +0, -0              | +0             | +0, -0         | equal by value; device drops -0 sign  |
    | 254 bf16 subnormals | 0              | x              | exactly 0 (device flushes the input)  |
    """
    special = torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0, -0.0])
    all_bf16 = generate_all_bfloat16_bitpatterns(torch.float32).flatten()
    subnormals = all_bf16[(all_bf16 != 0) & (all_bf16.abs() < SMALLEST_NORMAL_BF16)]
    input_tensor = torch.cat([special, subnormals]).reshape(1, -1)

    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.hardmish)
    golden = golden_function(input_tensor, device=device).flatten()

    result = ttnn.to_torch(ttnn.hardmish(tt_in)).flatten()

    n_special = special.numel()
    golden_special, result_special = golden[:n_special], result[:n_special]
    assert torch.equal(torch.isnan(result_special), torch.isnan(golden_special)), f"NaN mismatch: {result_special}"
    not_nan = ~torch.isnan(golden_special)
    assert torch.equal(result_special[not_nan], golden_special[not_nan]), f"special-value mismatch: {result_special}"

    result_subnormal = result[n_special:]
    assert subnormals.numel() == 254, "expected all 254 non-zero bf16 subnormals"
    assert (result_subnormal == 0).all(), "device must flush subnormal inputs to 0"


def test_hardsigmoid(device):
    """Float32 sweep over every normal bfloat16 value for hardsigmoid = clip(x/6 + 0.5, 0, 1).

    The kernel evaluates x * fp32(1/6) + 0.5.

    | Input range    | Check       | Reason                                                  |
    |----------------|-------------|---------------------------------------------------------|
    | -3 < x < -2.25 | abs ≤ 2^-24 | x/6 ≈ -0.5 carries ≤ ULP(0.5) = 2^-24 error; adding 0.5 |
    |                |             | cancels to an output near 0 (75 ULP at x = -2.984375)   |
    | elsewhere      | ULP ≤ 1     | single rounding of the multiply-add                     |
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.hardsigmoid)
    golden = golden_function(input_tensor, device=device)

    result = ttnn.to_torch(ttnn.hardsigmoid(tt_in))

    band = hardsigmoid_cancellation_band(input_tensor)
    assert band.any(), "expected hardsigmoid cancellation band to be non-empty for this exhaustive sweep"
    assert_allclose(golden[band], result[band], rtol=0, atol=2**-24)
    assert_with_ulp(expected_result=golden[~band], actual_result=result[~band], ulp_threshold=1)


# ─────────────────────────────────────────────────────────────────────────────
# hardswish — x * hardsigmoid(x), ULP ≤ 2
# ─────────────────────────────────────────────────────────────────────────────


def test_hardswish(device):
    """Float32 sweep over every normal bfloat16 value for hardswish = x * hardsigmoid(x).

    Known artifacts are verified explicitly rather than excluded.

    | Input range    | Check                  | Reason                                                 |
    |----------------|------------------------|--------------------------------------------------------|
    | x > ~5.65e37   | device returns x       | torch's golden formula overflows to +inf (golden bug)  |
    | |x| < 2*SNB    | exactly 0              | output ≈ x/2 is subnormal and FTZ'd; at exactly 2*SNB  |
    |                |                        | the device already returns a normal value              |
    | -3 < x < -2.25 | abs ≤ 2^-22            | hardsigmoid's 2^-24 cancellation error scaled by       |
    |                |                        | |x| < 3 plus rounding (112 ULP at x = -2.984375)       |
    | elsewhere      | ULP ≤ 2                | multiply of two rounded values                         |
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.hardswish)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.hardswish(tt_in)
    result = ttnn.to_torch(tt_result)

    # (1) Golden-formula overflow: device must return x exactly and finite.
    golden_overflow = torch.isinf(golden)
    assert golden_overflow.any(), "expected golden-overflow band to be non-empty for this exhaustive sweep"
    assert torch.isfinite(result[golden_overflow]).all(), "device diverged to inf/nan in the golden-overflow band"
    assert_equal(input_tensor[golden_overflow], result[golden_overflow])

    # (2) near-zero output FTZ: |x/2| < smallest normal, i.e. |x| < 2*SNB.
    near_zero_ftz = (input_tensor.abs().float() < 2 * SMALLEST_NORMAL_BF16) & ~golden_overflow
    assert_ftz_band(result, near_zero_ftz, "near-zero FTZ band")

    band = hardsigmoid_cancellation_band(input_tensor)
    assert band.any(), "expected hardsigmoid cancellation band to be non-empty for this exhaustive sweep"
    assert_allclose(golden[band], result[band], rtol=0, atol=2**-22)

    remaining = ~golden_overflow & ~near_zero_ftz & ~band
    assert_with_ulp(expected_result=golden[remaining], actual_result=result[remaining], ulp_threshold=2)


# ─────────────────────────────────────────────────────────────────────────────
# silu, swish — x * sigmoid(x), ULP ≤ 3 (SFPU FTZ for large negative inputs)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.silu,
        ttnn.swish,
    ],
)
def test_silu_swish_ops(device, ttnn_op):
    """Float32 sweep over every normal bfloat16 value for silu/swish = x * sigmoid(x).

    swish is the same SILU kernel, so both share one set of bounds.

    | Input range         | Check     | Reason                                                   |
    |---------------------|-----------|----------------------------------------------------------|
    | |x| < 2*SNB         | exactly 0 | output ≈ x/2 is subnormal and FTZ'd; unlike bfloat16, at |
    |                     |           | exactly 2*SNB the fp32 result x/2 = SNB is normal        |
    | -88.5 <= x <= -87.5 | exactly 0 | device sigmoid FTZs at x <= -87.5; torch's golden only   |
    |                     |           | reaches 0 at x <= -89 (exp(-x) overflow near -88.7)      |
    | elsewhere           | ULP ≤ 3   | fp32 sigmoid exp/reciprocal error (3 ULP at x = -11.5)   |

    The -88.5..-87.5 boundaries come from a full-negative-domain hardware
    sweep, not the analytic exp-underflow limit (~-104).
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    # (1) near-zero output FTZ: |x/2| < smallest normal, i.e. |x| < 2*SNB.
    near_zero_ftz = input_tensor.abs().float() < 2 * SMALLEST_NORMAL_BF16
    assert_ftz_band(result, near_zero_ftz, "near-zero FTZ band")

    # (2) negative sigmoid-underflow sliver: device must FTZ to exactly 0.
    neg_ftz_band = (input_tensor.float() >= -88.5) & (input_tensor.float() <= -87.5) & ~near_zero_ftz
    assert_ftz_band(result, neg_ftz_band, "negative sigmoid-underflow sliver")

    remaining = ~near_zero_ftz & ~neg_ftz_band
    assert_with_ulp(expected_result=golden[remaining], actual_result=result[remaining], ulp_threshold=3)


# ─────────────────────────────────────────────────────────────────────────────
# softsign — x / (1 + |x|), ULP ≤ 2 (division/reciprocal approximation path)
# ─────────────────────────────────────────────────────────────────────────────


def test_softsign(device):
    """Float32 sweep over every normal bfloat16 value for softsign = x / (1 + |x|).

    The SFPU computes this via reciprocal(1 + |x|).

    | Input range               | Check      | Reason                                               |
    |---------------------------|------------|------------------------------------------------------|
    | |x| > 1/SNB - 1 (~8.5e37) | exactly ±1 | reciprocal is below SNB; the kernel saturates to ±1  |
    |                           |            | (within 1 ULP on Wormhole)                           |
    | all                       | ULP ≤ 2    | reciprocal + multiply rounding                       |

    ULP rather than whole-domain PCC: the ~46% of inputs already saturating
    to exactly ±1 (|x| >= 512) would satisfy PCC ≥ 0.999 for a badly broken
    kernel.
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.softsign)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.softsign(tt_in)
    result = ttnn.to_torch(tt_result)

    near_max = input_tensor.abs().float() > 1.0 / SMALLEST_NORMAL_BF16 - 1.0
    assert near_max.any(), "expected the sweep to reach the near-bf16_max magnitudes"
    saturated = torch.sign(input_tensor[near_max]).to(result.dtype)
    if is_wormhole_b0():
        # Wormhole lands within 1 ULP of ±1 here, as it does for 2**24 <= |x| < 2**126.
        assert_with_ulp(expected_result=saturated, actual_result=result[near_max], ulp_threshold=1)
    else:
        assert torch.equal(result[near_max], saturated), "softsign must saturate to exactly ±1 near bf16_max"

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2)


# ─────────────────────────────────────────────────────────────────────────────
# log_sigmoid — log(sigmoid(x)), per-branch checks, restricted domain
# ─────────────────────────────────────────────────────────────────────────────


def test_log_sigmoid(device):
    """Float32 sweep over every normal bfloat16 value for log_sigmoid = log(sigmoid(x)).

    Kernel branches (ckernel_sfpu_logsigmoid.h branches on -x):

    | Input range | Kernel computes                   | Error vs golden                         |
    |-------------|-----------------------------------|-----------------------------------------|
    | x <= -4     | x                                 | log1p(exp(x)): 0.018 at x = -4,         |
    |             |                                   | 2.5e-3 at -6, 4.6e-5 at -10             |
    | -4 < x <= 4 | degree-8 polynomial, softplus(-x) | max abs 0.016 (at x ≈ -0.54)            |
    | x > 4       | -exp(-x), fast-approx exp         | max rel 3.6% (at x = 4.5)               |

    Checks:

    | Input range          | Check            | Reason                                          |
    |----------------------|------------------|-------------------------------------------------|
    | x <= -4              | device returns x | pins the identity branch exactly                |
    | x <= -13             | ULP ≤ 2          | identity is within 2 ULP of golden from here on |
    | -4 < x <= 0          | abs ≤ 0.02       | polynomial (max 0.0163)                         |
    | 0 < x <= 4           | abs ≤ 0.005      | polynomial (max 0.0020)                         |
    | x > 4, golden normal | rel ≤ 0.04       | fast-approx exp; abs is vacuous once            |
    |                      |                  | |golden| < 0.005 (x > ~5.3)                     |
    | golden < SNB         | exactly 0        | -exp(-x) underflows; device FTZs (x >= 87.5)    |

    Known kernel bug (#55457): for x > ~172 the
    large-positive branch diverges and eventually returns -inf (e.g. x=266
    -> -inf) instead of ~0, so 170 is the tested upper bound.
    """
    negative_domain = generate_float32_bits_in_range(-torch.finfo(torch.float32).max, 0.0)
    positive_domain = generate_float32_bits_in_range(0.0, 170.0)

    tt_neg = to_float32_tt_tensor(negative_domain, device)
    tt_pos = to_float32_tt_tensor(positive_domain, device)

    golden_function = ttnn.get_golden_function(ttnn.log_sigmoid)
    golden_neg = golden_function(negative_domain, device=device)
    golden_pos = golden_function(positive_domain, device=device)

    result_neg = ttnn.to_torch(ttnn.log_sigmoid(tt_neg))
    result_pos = ttnn.to_torch(ttnn.log_sigmoid(tt_pos))

    identity_branch = negative_domain <= -4.0
    assert_equal(negative_domain[identity_branch], result_neg[identity_branch])

    identity_within_ulp = negative_domain <= -13.0
    assert_with_ulp(
        expected_result=golden_neg[identity_within_ulp],
        actual_result=result_neg[identity_within_ulp],
        ulp_threshold=2,
    )
    assert_allclose(golden_neg[~identity_branch], result_neg[~identity_branch], rtol=0, atol=0.02)

    polynomial_pos = positive_domain <= 4.0
    assert_allclose(golden_pos[polynomial_pos], result_pos[polynomial_pos], rtol=0, atol=0.005)

    exp_underflow = ~polynomial_pos & (golden_pos.abs() < SMALLEST_NORMAL_BF16)
    assert_ftz_band(result_pos, exp_underflow, "exp-branch underflow band")

    exp_branch = ~polynomial_pos & ~exp_underflow
    assert_allclose(golden_pos[exp_branch], result_pos[exp_branch], rtol=0.04, atol=0)


# ─────────────────────────────────────────────────────────────────────────────
# tanhshrink — x - tanh(x), mpmath reference for |x| < 1
# ─────────────────────────────────────────────────────────────────────────────


def test_tanhshrink(device):
    """Float32 sweep over every normal bfloat16 value for tanhshrink = x - tanh(x).

    For |x| <= 1 the kernel uses x^3 * Q(x^2); for |x| > 1, |x| - tanh(|x|).

    | Input range        | Reference | Check     | Reason                                             |
    |--------------------|-----------|-----------|----------------------------------------------------|
    | |x| < 1, reference | mpmath    | exactly 0 | true result x^3/3 is below SNB                     |
    | below SNB          |           |           | (|x| < ~3.3e-13); device FTZs                      |
    | |x| < 1, elsewhere | mpmath    | ULP ≤ 3   | x^3 * Q(x^2) has no cancellation (max 2.1 ULP)     |
    | 1 <= |x| < 1.625   | golden    | ULP ≤ 8   | partial cancellation magnifies tanh's ~1 ULP       |
    |                    |           |           | error (x = 1.0078: tanh ≈ 0.765, result ≈ 0.243)   |
    | |x| >= 1.625       | golden    | ULP ≤ 3   | tanh = 2*sigmoid(2|x|) - 1 doubles the sigmoid's   |
    |                    |           |           | rounding (max: WH 2 ULP, BH 3 ULP at x ≈ 1.80)     |

    For |x| < 1 the torch fp32 golden x - tanh(x) cancels (up to ~1.7e7 ULP
    off), and so does float64 for |x| < ~1.7e-8, so the reference there is
    computed with mpmath at 200-bit precision.
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.tanhshrink)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.tanhshrink(tt_in)
    result = ttnn.to_torch(tt_result)

    abs_input = input_tensor.abs().float()
    small = abs_input < 1.0
    partial_cancellation = (abs_input >= 1.0) & (abs_input < 1.625)
    remaining = abs_input >= 1.625

    with mp.workprec(200):
        reference_small = torch.tensor(
            [float(mpf(v) - mp_tanh(mpf(v))) for v in input_tensor[small].double().tolist()], dtype=torch.float64
        ).float()
    result_small = result[small]

    underflow = reference_small.abs() < SMALLEST_NORMAL_BF16
    assert_ftz_band(result_small, underflow, "small-|x| underflow band")
    assert_with_ulp(
        expected_result=reference_small[~underflow], actual_result=result_small[~underflow], ulp_threshold=3
    )

    assert_with_ulp(
        expected_result=golden[partial_cancellation], actual_result=result[partial_cancellation], ulp_threshold=8
    )
    assert_with_ulp(expected_result=golden[remaining], actual_result=result[remaining], ulp_threshold=3)

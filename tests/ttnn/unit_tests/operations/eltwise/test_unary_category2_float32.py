# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_allclose, assert_equal, assert_with_ulp, assert_with_pcc
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
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
| hardmish     | all                    | ULP ≤ 1                  | golden emulates SFPSTORE truncation        |
| hardsigmoid  | -3 < x < -2.25         | abs ≤ 2^-24              | x*(1/6) + 0.5 cancels near x = -3          |
|              | elsewhere              | ULP ≤ 1                  |                                            |
| hardswish    | x > ~5.65e37           | device returns x         | torch golden overflows to inf              |
|              | |x| < 2*SNB            | exactly 0                | x/2 underflow FTZ                          |
|              | -3 < x < -2.25         | abs ≤ 2^-22              | inherits hardsigmoid cancellation          |
|              | elsewhere              | ULP ≤ 2                  |                                            |
| silu, swish  | |x| < 2*SNB            | exactly 0                | x/2 underflow FTZ                          |
|              | -88.5 <= x <= -87.5    | exactly 0                | device sigmoid FTZ before torch's          |
|              | elsewhere              | ULP ≤ 3                  | fp32 sigmoid exp/reciprocal error          |
| softsign     | |x| > 1/SNB - 1        | exactly 0                | reciprocal(1 + |x|) FTZ                    |
|              | |x| == 1/SNB - 1       | 0 or golden ±1           | arch-dependent rounding                    |
|              | elsewhere              | ULP ≤ 2                  |                                            |
| log_sigmoid  | x <= -13               | ULP ≤ 2                  | kernel returns x; exact here               |
|              | -13 < x <= 0           | abs ≤ 0.02               | x-identity tail + polynomial               |
|              | 0 < x <= 170           | PCC ≥ 0.999, abs ≤ 0.005 | output -> 0, ULP not meaningful            |
| tanhshrink   | |x| < 1                | PCC ≥ 0.999              | golden loses precision to cancellation     |
|              | 1 <= |x| < 1.625       | ULP ≤ 8                  | partial cancellation in x - tanh(x)        |
|              | |x| >= 1.625           | ULP ≤ 3                  | tanh via 2*sigmoid - 1 (BH max 3, WH 2)    |

Thresholds are tighter in absolute terms than the bfloat16 file but, being
float32 ULPs (2^16 times finer), expose kernel error the bfloat16 output
rounding hides.
"""


def assert_ftz_band(result, band_mask, band_desc):
    """Assert a known flush-to-zero band is non-empty and device-flushed to 0.

    The band is a deterministic slice of the exhaustive sweep, so the
    non-emptiness assertion guards against a future golden/kernel change that
    shifts a boundary and silently empties the band (which would otherwise let
    the test pass without ever checking the documented FTZ behavior).
    """
    assert band_mask.any(), f"expected {band_desc} to be non-empty for this exhaustive sweep"
    assert_equal(torch.zeros_like(result[band_mask]), result[band_mask])


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
    """Exhaustive normal bfloat16 coverage for relu = max(0, x) and relu6 = min(max(0, x), 6).

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
    """Exhaustive normal bfloat16 coverage for hardmish.

    | Input range | Check   | Reason                                                      |
    |-------------|---------|-------------------------------------------------------------|
    | all         | ULP ≤ 1 | golden emulates SFPSTORE truncation (torch_hardmish in      |
    |             |         | ttnn/ttnn/operations/unary.py); residual SFPU rounding only |
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.hardmish)
    golden = golden_function(input_tensor, device=device)

    result = ttnn.to_torch(ttnn.hardmish(tt_in))

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


def test_hardsigmoid(device):
    """Exhaustive normal bfloat16 coverage for hardsigmoid = clip(x/6 + 0.5, 0, 1).

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
    """Exhaustive normal bfloat16 coverage for hardswish = x * hardsigmoid(x).

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
# silu, swish — x * sigmoid(x), PCC (SFPU FTZ for large negative inputs)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.silu,
        ttnn.swish,
    ],
)
def test_silu_swish_ops(device, ttnn_op):
    """Exhaustive normal bfloat16 coverage for silu/swish = x * sigmoid(x).

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
# softsign — x / (1 + |x|), PCC (division/reciprocal approximation path)
# ─────────────────────────────────────────────────────────────────────────────


def test_softsign(device):
    """Exhaustive normal bfloat16 coverage for softsign = x / (1 + |x|).

    The SFPU computes this via reciprocal(1 + |x|).

    | Input range               | Check          | Reason                                           |
    |---------------------------|----------------|--------------------------------------------------|
    | |x| > 1/SNB - 1 (~8.5e37) | exactly 0      | reciprocal underflows and is FTZ'd instead of ±1 |
    | |x| == 1/SNB - 1          | 0 or golden ±1 | arch-dependent rounding (WH: ±1, BH: 0)          |
    | elsewhere                 | ULP ≤ 2        | reciprocal + multiply rounding                   |

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

    ftz_threshold = 1.0 / SMALLEST_NORMAL_BF16 - 1.0
    abs_input = input_tensor.abs().float()
    # Deep-subnormal band: unambiguous FTZ to 0 on every architecture.
    near_max = abs_input > ftz_threshold
    assert_ftz_band(result, near_max, "near-bf16_max FTZ band")

    # Boundary magnitude (|x| == ftz_threshold exactly): normal/subnormal
    # rounding of the reciprocal is architecture-dependent, so either the
    # FTZ'd 0 or the mathematically-correct golden (±1) is accepted.
    boundary = abs_input == ftz_threshold
    assert boundary.any(), "expected near-bf16_max boundary magnitude to be non-empty for this exhaustive sweep"
    boundary_ok = (result[boundary] == 0) | (result[boundary] == golden[boundary])
    assert boundary_ok.all(), "boundary magnitude must be either FTZ'd to 0 or exactly the golden ±1"

    ftz_safe = ~near_max & ~boundary
    assert_with_ulp(expected_result=golden[ftz_safe], actual_result=result[ftz_safe], ulp_threshold=2)


# ─────────────────────────────────────────────────────────────────────────────
# log_sigmoid — log(sigmoid(x)), PCC, restricted domain
# ─────────────────────────────────────────────────────────────────────────────


def test_log_sigmoid(device):
    """Exhaustive normal bfloat16 coverage for log_sigmoid = log(sigmoid(x)).

    Kernel branches:

    | Input range | Kernel computes                   | Error                                 |
    |-------------|-----------------------------------|---------------------------------------|
    | x < -4      | x                                 | drops -log1p(exp(x)): 0.018 at x = -4 |
    | -4 <= x < 4 | degree-8 polynomial, softplus(-x) | max abs ~0.016 (at x ≈ -0.54)         |
    | x >= 4      | -exp(-x), fast-approx exp         |                                       |

    Checks:

    | Input range  | Check                    | Reason                                       |
    |--------------|--------------------------|----------------------------------------------|
    | x <= -13     | ULP ≤ 2                  | exp(x) falls under 2 fp32 ULP of x           |
    | -13 < x <= 0 | abs ≤ 0.02               | x-identity tail + polynomial (max 0.0181)    |
    | 0 < x <= 170 | PCC ≥ 0.999, abs ≤ 0.005 | output -> 0, ULP not meaningful (max ~0.002) |

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

    identity_exact = negative_domain.float() <= -13.0
    assert_with_ulp(
        expected_result=golden_neg[identity_exact], actual_result=result_neg[identity_exact], ulp_threshold=2
    )
    assert_allclose(golden_neg[~identity_exact], result_neg[~identity_exact], rtol=0, atol=0.02)

    assert_with_pcc(golden_pos, result_pos, pcc=0.999)
    assert_allclose(golden_pos, result_pos, rtol=0, atol=0.005)


# ─────────────────────────────────────────────────────────────────────────────
# tanhshrink — x - tanh(x), PCC
# ─────────────────────────────────────────────────────────────────────────────


def test_tanhshrink(device):
    """Exhaustive normal bfloat16 coverage for tanhshrink = x - tanh(x).

    For |x| <= 1 the kernel uses x^3 * Q(x^2); for |x| > 1, |x| - tanh(|x|).

    | Input range      | Check       | Reason                                                    |
    |------------------|-------------|-----------------------------------------------------------|
    | |x| < 1          | PCC ≥ 0.999 | torch's fp32 golden itself loses precision to x - tanh(x) |
    |                  |             | cancellation (mpmath ULP check: issue #45520,             |
    |                  |             | test_activation.py::test_tanhshrink_ulp)                  |
    | 1 <= |x| < 1.625 | ULP ≤ 8     | partial cancellation magnifies tanh's ~1 ULP error        |
    |                  |             | (x = 1.0078: tanh ≈ 0.765, result ≈ 0.243)                |
    | |x| >= 1.625     | ULP ≤ 3     | tanh = 2*sigmoid(2|x|) - 1 doubles the sigmoid's rounding |
    |                  |             | (max: WH 2 ULP, BH 3 ULP at x ≈ 1.80)                     |

    Why not one PCC check over all inputs: tanh(x) never exceeds 1, so for
    most inputs the output is almost exactly x. A broken kernel that just
    returned x would still pass a whole-range PCC check.
    """
    input_tensor = generate_float32_bits()

    tt_in = to_float32_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.tanhshrink)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.tanhshrink(tt_in)
    result = ttnn.to_torch(tt_result)

    abs_input = input_tensor.abs().float()
    cancellation_band = abs_input < 1.0
    partial_cancellation = (abs_input >= 1.0) & (abs_input < 1.625)
    remaining = abs_input >= 1.625

    assert_with_ulp(
        expected_result=golden[partial_cancellation], actual_result=result[partial_cancellation], ulp_threshold=8
    )
    assert_with_ulp(expected_result=golden[remaining], actual_result=result[remaining], ulp_threshold=3)
    assert_with_pcc(golden[cancellation_band], result[cancellation_band], pcc=0.999)

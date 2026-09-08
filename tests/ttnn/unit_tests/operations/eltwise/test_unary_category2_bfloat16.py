# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_equal, assert_with_ulp, assert_with_pcc
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    generate_bfloat16_bits_in_range,
    to_tt_tensor,
    SMALLEST_NORMAL_BF16,
)

pytestmark = pytest.mark.use_module_device

"""
Category 2: basic_unary_activation (no extra parameters)
Neural network activation functions

 1. ttnn.relu             - Rectified linear unit
 2. ttnn.relu6            - ReLU capped at 6
 3. ttnn.silu             - SiLU (Sigmoid Linear Unit)
 4. ttnn.swish            - Swish activation
 5. ttnn.hardmish         - Hard Mish activation
 6. ttnn.hardsigmoid      - Hard sigmoid
 7. ttnn.hardswish        - Hard swish
 8. ttnn.softsign         - Softsign
 9. ttnn.log_sigmoid      - Log sigmoid
10. ttnn.tanhshrink       - Tanh shrink

Accuracy criteria
─────────────────
  relu, relu6    : exact  (comparison + select, no rounding introduced)
  hardsigmoid    : ULP ≤ 1  (clip(x/6 + 0.5, 0, 1); /6 division rounds ≤ 1 ULP)
  hardmish       : ULP ≤ 1  (golden emulates hardware's SFPSTORE truncation)
  hardswish      : ULP ≤ 2  (two known artifacts verified explicitly, see
                              test_hardswish)
  silu, swish    : ULP ≤ 2  (near-zero and negative-sigmoid FTZ bands
                              verified explicitly; see test_silu_swish_ops)
  softsign       : ULP ≤ 2  (near-bf16_max FTZ band verified explicitly;
                              see test_softsign)
  log_sigmoid    : ULP ≤ 2 for x <= 0; PCC ≥ 0.999 for 0 < x <= 170; see
                              test_log_sigmoid for a known kernel bug above 170
  tanhshrink     : ULP ≤ 2 for |x| >= 1; PCC ≥ 0.999 for |x| < 1; see
                              test_tanhshrink
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
    """Exhaustive normal bfloat16 coverage for relu and relu6.

    Both are exact piecewise-linear functions (max(0, x) and
    min(max(0, x), 6)): every output is the input value, 0, or 6 — all
    exactly representable in bfloat16, so exact equality is asserted.
    """
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_equal(golden, result)


# ─────────────────────────────────────────────────────────────────────────────
# Piecewise-linear-with-division ops (hardsigmoid, hardmish) — ULP ≤ 1
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.hardsigmoid,
        ttnn.hardmish,
    ],
)
def test_piecewise_division_ops(device, ttnn_op):
    """Exhaustive normal bfloat16 coverage for hardsigmoid and hardmish.

    hardsigmoid = clip(x/6 + 0.5, 0, 1): division/clip/add round at most
    1 ULP total. hardmish's golden already emulates hardware's SFPSTORE
    truncation (see torch_hardmish in ttnn/ttnn/operations/unary.py), so
    both agree to within 1 ULP of residual SFPU rounding.
    """
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(golden, result, 1)


# ─────────────────────────────────────────────────────────────────────────────
# hardswish — x * hardsigmoid(x), ULP ≤ 2
# ─────────────────────────────────────────────────────────────────────────────


def test_hardswish(device):
    """Exhaustive normal bfloat16 coverage for hardswish = x * hardsigmoid(x).

    Two known artifacts, verified explicitly rather than excluded:
    1. x > ~5.65e37: torch's golden formula overflows to +inf (a golden bug,
       not a device bug); device is asserted to return x exactly.
    2. near-zero output FTZ: hardswish(x) = x*relu6(x+3)/6 ≈ x/2 for tiny x,
       and the device flushes that subnormal result to 0 for |x| < 2*SNB
       (strict; at exactly 2*SNB the device already returns a normal value).
       Device is asserted to return exactly 0 there.
    ULP ≤ 2 covers everything else.
    """
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)

    tt_in = to_tt_tensor(input_tensor, device)

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

    remaining = ~golden_overflow & ~near_zero_ftz
    assert_with_ulp(golden[remaining], result[remaining], 2)


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

    Two known FTZ artifacts, both verified explicitly (device returns exactly
    0), then ULP ≤ 2 for everything else:
    1. near-zero output FTZ: silu(x) = x*sigmoid(x) ≈ x/2 for tiny x, whose
       subnormal result the device flushes to 0 for |x| <= 2*SNB. This is one
       bf16 step wider than hardswish's identical x/2 underflow (<, exclusive)
       because silu flushes the boundary value 2*SNB too.
    2. negative sigmoid-underflow sliver x in [-88.5, -87.5]: the device's
       internal sigmoid(x) flushes to 0 at x <= -87.5, whereas torch's float32
       golden sigmoid = 1/(1+exp(-x)) only reaches exact 0 at x <= -89, where
       exp(-x) overflows float32 (near x=-88.7). The 3 bf16 values in between
       are the only ones where device (0) and golden (~1e-37) disagree; for
       x <= -89 both are 0 and match. Boundaries are from a full-negative-
       domain hardware sweep, not the analytic exp-underflow limit (~-104)
       that torch's overflow-based sigmoid never actually reaches.
    """
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    # (1) near-zero output FTZ: |x/2| <= smallest normal, i.e. |x| <= 2*SNB.
    near_zero_ftz = input_tensor.abs().float() <= 2 * SMALLEST_NORMAL_BF16
    assert_ftz_band(result, near_zero_ftz, "near-zero FTZ band")

    # (2) negative sigmoid-underflow sliver: device must FTZ to exactly 0.
    neg_ftz_band = (input_tensor.float() >= -88.5) & (input_tensor.float() <= -87.5) & ~near_zero_ftz
    assert_ftz_band(result, neg_ftz_band, "negative sigmoid-underflow sliver")

    remaining = ~near_zero_ftz & ~neg_ftz_band
    assert_with_ulp(golden[remaining], result[remaining], 2)


# ─────────────────────────────────────────────────────────────────────────────
# softsign — x / (1 + |x|), PCC (division/reciprocal approximation path)
# ─────────────────────────────────────────────────────────────────────────────


def test_softsign(device):
    """Exhaustive normal bfloat16 coverage for softsign = x / (1 + |x|).

    The SFPU computes this via reciprocal(1 + |x|); near bf16_max
    (|x| > ~8.5e37) that intermediate underflows and flushes to 0 instead
    of the correct ±1 saturation. At the exact threshold, that rounding is
    architecture-dependent (WH: correct ±1, BH: flushed 0), so only that
    one boundary magnitude accepts either outcome; everything beyond it
    must FTZ to 0. Both are verified explicitly.

    ULP ≤ 2 covers the remaining "FTZ-safe" domain. A whole-domain PCC
    would not constrain the reciprocal path at all here, since the ~46%
    of that domain already saturating to exactly ±1 (|x| >= 512) is
    enough alone to satisfy PCC ≥ 0.999 for a badly broken kernel.
    """
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)

    tt_in = to_tt_tensor(input_tensor, device)

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
    assert_with_ulp(golden[ftz_safe], result[ftz_safe], 2)


# ─────────────────────────────────────────────────────────────────────────────
# log_sigmoid — log(sigmoid(x)), PCC, restricted domain
# ─────────────────────────────────────────────────────────────────────────────


def test_log_sigmoid(device):
    """Exhaustive normal bfloat16 coverage for log_sigmoid = log(sigmoid(x)).

    For x <= -4 the kernel uses the stable identity log_sigmoid(x) ≈ x
    directly; exact for every negative value, so checked with ULP ≤ 2 over
    the full negative domain. For 0 < x <= 170 a polynomial/exp approximation
    is used; PCC ≥ 0.999 covers its compound approximation error (max abs
    diff stays ~0.004 throughout this range).

    Known kernel bug (tenstorrent/tt-metal#55457): for x > ~172 the
    large-positive branch diverges and eventually returns -inf (e.g. x=266
    -> -inf) instead of ~0. 170 is excluded as the tested upper bound since
    it's the last point before that divergence begins.
    """
    negative_domain = generate_bfloat16_bits_in_range(-torch.finfo(torch.bfloat16).max, 0.0)
    positive_domain = generate_bfloat16_bits_in_range(0.0, 170.0)

    tt_neg = to_tt_tensor(negative_domain, device)
    tt_pos = to_tt_tensor(positive_domain, device)

    golden_function = ttnn.get_golden_function(ttnn.log_sigmoid)
    golden_neg = golden_function(negative_domain, device=device)
    golden_pos = golden_function(positive_domain, device=device)

    result_neg = ttnn.to_torch(ttnn.log_sigmoid(tt_neg))
    result_pos = ttnn.to_torch(ttnn.log_sigmoid(tt_pos))

    assert_with_ulp(golden_neg, result_neg, 2)
    assert_with_pcc(golden_pos, result_pos, pcc=0.999)


# ─────────────────────────────────────────────────────────────────────────────
# tanhshrink — x - tanh(x), PCC
# ─────────────────────────────────────────────────────────────────────────────


def test_tanhshrink(device):
    """Exhaustive normal bfloat16 coverage for tanhshrink = x - tanh(x).

    ULP ≤ 2 for |x| >= 1, where the subtraction doesn't cancel. For |x| < 1,
    x and tanh(x) nearly cancel and torch's float32 golden rounds this
    differently than bfloat16 hardware, so PCC ≥ 0.999 is used there instead
    (a dedicated mpmath-based ULP regression for this region, issue #45520,
    lives in test_activation.py::test_tanhshrink_ulp).

    Splitting by magnitude matters: a single full-range PCC ≥ 0.999 would not
    validate the |x| >= 1 majority, since tanh(x) is bounded by 1 and barely
    perturbs the aggregate correlation once x spans up to ~3.4e38 — a kernel
    that always returned x unchanged would still pass it.
    """
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.tanhshrink)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.tanhshrink(tt_in)
    result = ttnn.to_torch(tt_result)

    cancellation_band = input_tensor.abs().float() < 1.0
    remaining = ~cancellation_band

    assert_with_ulp(golden[remaining], result[remaining], 2)
    assert_with_pcc(golden[cancellation_band], result[cancellation_band], pcc=0.999)

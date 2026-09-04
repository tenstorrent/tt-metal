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
  hardswish      : ULP ≤ 2  (x * hardsigmoid(x)); two known artifacts are
                              verified explicitly, see test_hardswish
  silu, swish    : PCC ≥ 0.999  (x * sigmoid(x); SFPU FTZ for large negative x)
  softsign       : PCC ≥ 0.999  (x / (1 + |x|)); near-bf16_max FTZ artifact
                              verified explicitly, see test_softsign
  log_sigmoid    : PCC ≥ 0.999  (log(sigmoid(x)), restricted to [-80, 80])
  tanhshrink     : PCC ≥ 0.999  (x - tanh(x); dedicated mpmath-based ULP
                              regression for the cancellation region lives in
                              test_activation.py::test_tanhshrink_ulp)
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
    """Exhaustive normal bfloat16 coverage for relu and relu6.

    Both ops are exact piecewise-linear functions (max(0, x) and
    min(max(0, x), 6)): every output is either the input value, 0, or 6 —
    all exactly representable in bfloat16, so no rounding is introduced by
    the op itself. Exact equality is asserted.
    """
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_equal(result, golden)


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

    hardsigmoid = clip(x/6 + 0.5, 0, 1): the division by 6 rounds at most
    1 ULP; the clip and add are exact or round at most 1 ULP as well.
    hardmish's golden function already emulates the hardware's SFPSTORE
    truncation behaviour for bfloat16 inputs (see torch_hardmish in
    ttnn/ttnn/operations/unary.py), so device and golden should agree to
    within 1 ULP of residual SFPU rounding.
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
    """Exhaustive normal bfloat16 coverage for hardswish.

    hardswish(x) = x * hardsigmoid(x); ULP ≤ 2. Two known artifacts are
    verified explicitly rather than silently excluded:

    1. For x > ~bf16_max/6 (~5.65e37), torch's own golden formula
       (x * relu6(x+3) / 6) overflows its intermediate product to +inf,
       even though the true hardswish(x) = x is finite there. This is a
       bug in torch's reference formula, not the device: the device is
       asserted to return x exactly for every such input.
    2. For x with bf16 exponent field == 1 (|x| in [tiny, 2*tiny)),
       hardswish(x) ≈ x/2 underflows to a subnormal magnitude, which
       hardware flushes to exact zero (FTZ). The device is asserted to
       return exactly 0 for every such input.
    """
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.hardswish)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.hardswish(tt_in)
    result = ttnn.to_torch(tt_result)

    # (1) Golden-formula overflow: device must return x exactly.
    golden_overflow = torch.isinf(golden) & torch.isfinite(result)
    if golden_overflow.any():
        assert_equal(result[golden_overflow], input_tensor[golden_overflow])

    # (2) exponent-field == 1 band: device must FTZ to exactly 0.
    exp1_band = input_tensor.abs().float() < 2 * SMALLEST_NORMAL_BF16
    if (exp1_band & ~golden_overflow).any():
        flushed = exp1_band & ~golden_overflow
        assert_equal(result[flushed], torch.zeros_like(result[flushed]))

    remaining = ~golden_overflow & ~exp1_band
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
    """Exhaustive normal bfloat16 coverage for silu and swish.

    silu(x) = swish(x) = x * sigmoid(x). PCC ≥ 0.999 is used because the
    SFPU flushes sigmoid subnormals to zero for large negative x, causing
    small legitimate differences from the CPU reference — the same
    rationale as swiglu's gate function in test_unary_category5_bfloat16.py.
    """
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn_op(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_pcc(golden, result, pcc=0.999)


# ─────────────────────────────────────────────────────────────────────────────
# softsign — x / (1 + |x|), PCC (division/reciprocal approximation path)
# ─────────────────────────────────────────────────────────────────────────────


def test_softsign(device):
    """Exhaustive normal bfloat16 coverage for softsign.

    softsign(x) = x / (1 + |x|); PCC ≥ 0.999. The SFPU computes this as
    x * reciprocal(1 + |x|); once |x| exceeds ~1/tiny - 1 (~8.5e37, near
    bf16_max), that reciprocal intermediate underflows and hardware
    flushes it to zero, producing 0 instead of the correct ±1 saturation
    value. The device is asserted to return exactly 0 there; PCC is then
    checked over the remaining "FTZ-safe" domain.
    """
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.softsign)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.softsign(tt_in)
    result = ttnn.to_torch(tt_result)

    ftz_threshold = 1.0 / SMALLEST_NORMAL_BF16 - 1.0
    near_max = input_tensor.abs().float() > ftz_threshold

    # Verified FTZ: device returns exactly 0 here.
    if near_max.any():
        assert_equal(result[near_max], torch.zeros_like(result[near_max]))

    ftz_safe = ~near_max
    assert_with_pcc(golden[ftz_safe], result[ftz_safe], pcc=0.999)


# ─────────────────────────────────────────────────────────────────────────────
# log_sigmoid — log(sigmoid(x)), PCC, restricted domain
# ─────────────────────────────────────────────────────────────────────────────


def test_log_sigmoid(device):
    """Exhaustive normal bfloat16 coverage for log_sigmoid over [-80, 80].

    log_sigmoid(x) = log(sigmoid(x)) is a compound log+sigmoid approximation.
    The domain is restricted to [-80, 80]: for x below roughly -88, sigmoid(x)
    legitimately underflows to exact 0 in bfloat16/float32, and log(0) = -inf
    would incorrectly diverge from the true near-linear value of log_sigmoid
    there. PCC ≥ 0.999 is used for the compound approximation error.
    """
    input_tensor = generate_bfloat16_bits_in_range(-80.0, 80.0)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.log_sigmoid)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.log_sigmoid(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_pcc(golden, result, pcc=0.999)


# ─────────────────────────────────────────────────────────────────────────────
# tanhshrink — x - tanh(x), PCC
# ─────────────────────────────────────────────────────────────────────────────


def test_tanhshrink(device):
    """Exhaustive normal bfloat16 coverage for tanhshrink.

    tanhshrink(x) = x - tanh(x). The torch golden cancels the same way the
    bfloat16 hardware does for small |x|, so PCC ≥ 0.999 is the appropriate
    aggregate check for full-range exhaustive coverage. A dedicated mpmath-based
    ULP regression for the cancellation region (issue #45520) lives separately
    in test_activation.py::test_tanhshrink_ulp.
    """
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)

    tt_in = to_tt_tensor(input_tensor, device)

    golden_function = ttnn.get_golden_function(ttnn.tanhshrink)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.tanhshrink(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_pcc(golden, result, pcc=0.999)

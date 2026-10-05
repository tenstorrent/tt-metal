# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_equal,
    assert_with_ulp,
    assert_with_pcc,
    generate_all_bfloat16_bitpatterns,
)
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

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


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
    assert_with_ulp(expected_result=golden[remaining], actual_result=result[remaining], ulp_threshold=2)


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
    assert_with_ulp(expected_result=golden[ftz_safe], actual_result=result[ftz_safe], ulp_threshold=2)


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

    assert_with_ulp(expected_result=golden_neg, actual_result=result_neg, ulp_threshold=2)
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

    assert_with_ulp(expected_result=golden[remaining], actual_result=result[remaining], ulp_threshold=2)
    assert_with_pcc(golden[cancellation_band], result[cancellation_band], pcc=0.999)


# Exhaustive BF16 accuracy of the unary ops that run in the packer's ReLU stage.
#
# The packer serves sharded tensors only, so the inputs are height-sharded. The reference is torch
# on the operand as BF16 DEST holds it, stored as a BF16 tile holds the result. The ops are exact,
# so every output must equal it bit for bit. Each op logs one ULP line: the largest ULP error
# against the reference and the outputs of another class, for the packer and for the SFPU kernel it
# replaces, run on the same input as a chain of the op and IDENTITY (see the device-perf test).
PACK_RELU_SMALLEST_NORMAL = 2.0**-126
# The 256 x 256 input, one row of tiles per core.
PACK_RELU_SHARDED = ttnn.create_sharded_memory_config(
    shape=(32, 256),
    core_grid=ttnn.CoreGrid(y=1, x=8),
    strategy=ttnn.ShardStrategy.HEIGHT,
    use_height_and_width_as_shard_shape=True,
)

PACK_RELU_CLASSES = {
    "pos_nan": lambda t: torch.isnan(t) & ~torch.signbit(t),
    "neg_nan": lambda t: torch.isnan(t) & torch.signbit(t),
    "neg_zero": lambda t: (t == 0) & torch.signbit(t),
    "pos_subnormal": lambda t: (t > 0) & (t < PACK_RELU_SMALLEST_NORMAL),
    "neg_subnormal": lambda t: (t < 0) & (t > -PACK_RELU_SMALLEST_NORMAL),
}
PACK_RELU_VALUES = {
    "pos_inf": float("inf"),
    "neg_inf": float("-inf"),
    "pos_zero": 0.0,
}

# (raw class, class it is held as) pairs: the operand DEST holds and the result a tile stores.
PACK_RELU_DEST_INPUT = (
    ("neg_nan", "neg_inf"),
    ("neg_subnormal", "pos_zero"),
    ("neg_zero", "pos_zero"),
    ("pos_nan", "pos_inf"),
    ("pos_subnormal", "pos_zero"),
)
PACK_RELU_STORED_RESULT = (("neg_zero", "pos_zero"),)

# op: (TT-NN call, torch reference, operand transport, result transport)
PACK_RELU_OPS = {
    "relu": (
        lambda x: ttnn.relu(x),
        lambda x: torch.nn.functional.relu(x),
        PACK_RELU_DEST_INPUT,
        PACK_RELU_STORED_RESULT,
    ),
    "relu_min": (
        lambda x: ttnn.relu_min(x, 0.0),
        lambda x: torch.clamp_min(x, min=0.0),
        PACK_RELU_DEST_INPUT,
        PACK_RELU_STORED_RESULT,
    ),
    "threshold": (
        lambda x: ttnn.threshold(x, 0.0, 0.0),
        lambda x: torch.nn.functional.threshold(x, 0.0, 0.0),
        PACK_RELU_DEST_INPUT,
        PACK_RELU_STORED_RESULT,
    ),
    "relu6": (
        lambda x: ttnn.relu6(x),
        lambda x: torch.nn.functional.relu6(x),
        PACK_RELU_DEST_INPUT,
        PACK_RELU_STORED_RESULT,
    ),
    "relu_max": (
        lambda x: ttnn.relu_max(x, 6.0),
        lambda x: torch.clamp(x, min=0.0, max=6.0),
        PACK_RELU_DEST_INPUT,
        PACK_RELU_STORED_RESULT,
    ),
}

# op: TT-NN's SFPU kernel for the op, which the packer replaces.
PACK_RELU_OLD = {
    "relu": lambda x: ttnn.unary_chain(
        x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
    ),
    "relu_min": lambda x: ttnn.unary_chain(
        x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MIN, 0.0), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
    ),
    "threshold": lambda x: ttnn.unary_chain(
        x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.THRESHOLD, 0.0, 0.0), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
    ),
    "relu6": lambda x: ttnn.unary_chain(
        x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
    ),
    "relu_max": lambda x: ttnn.unary_chain(
        x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MAX, 6.0), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
    ),
}


def _pack_relu_transport(t, source, pairs):
    for raw, held in pairs:
        t = torch.where(PACK_RELU_CLASSES[raw](source), torch.full_like(t, PACK_RELU_VALUES[held]), t)
    return t


def _pack_relu_stored_classes(t):
    """0 +inf (or NaN), 1 -inf, 2 zero of either sign, 3 finite nonzero."""
    t = t.to(torch.float64)
    classes = torch.full(t.shape, 3, dtype=torch.int8)
    classes[t == 0] = 2
    classes[(t == float("inf")) | torch.isnan(t)] = 0
    classes[t == float("-inf")] = 1
    return classes


def _pack_relu_versus(expected, output):
    """The largest ULP error against ``expected`` over outputs of its class, and the outputs of another class."""
    e, o = expected.to(torch.float64), output.to(torch.float64)
    same = _pack_relu_stored_classes(e) == _pack_relu_stored_classes(o)
    finite = same & torch.isfinite(e) & (e != 0)
    exponent = torch.floor(torch.log2(torch.where(finite, e.abs(), torch.ones_like(e))))
    ulp = torch.where(finite, (e - o).abs() / 2.0 ** (exponent.clamp(min=-126) - 7), torch.zeros_like(e))
    return ulp.max().item(), int((~same).sum())


@pytest.mark.parametrize("op", list(PACK_RELU_OPS))
def test_pack_relu_exhaustive_bfloat16(op, device):
    run, reference, operand, result = PACK_RELU_OPS[op]
    x = generate_all_bfloat16_bitpatterns(torch.bfloat16).reshape(256, 256)
    expected = reference(_pack_relu_transport(x.clone(), x, operand))
    expected = _pack_relu_transport(expected, expected, result)

    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=PACK_RELU_SHARDED
    )
    actual = ttnn.to_torch(run(tt_x))
    stock = ttnn.to_torch(PACK_RELU_OLD[op](tt_x))

    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    (ours, ours_classes), (old, old_classes) = _pack_relu_versus(expected, actual), _pack_relu_versus(expected, stock)
    print(
        f"ULP {op} {board} ours={ours:.3f} stock={old:.3f} "
        f"ours_class_mismatches={ours_classes} stock_class_mismatches={old_classes}"
    )
    mismatch = actual.view(torch.int16) != expected.view(torch.int16)
    assert not mismatch.any(), (
        f"{op}: {mismatch.sum().item()} mismatches, first at x={x[mismatch][0].item()}: "
        f"expected {expected[mismatch][0].item()}, got {actual[mismatch][0].item()}"
    )

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
from models.common.utility_functions import run_for_blackhole, run_for_wormhole_b0

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


# Exhaustive BF16 accuracy of ttnn.log_sigmoid.
#
# Every BF16 bit pattern is an input. The reference is torch.nn.functional.logsigmoid(x) in float64,
# rounded once to BF16 with subnormal results flushed to zero. The SFPU may read a subnormal input
# as zero, so torch is evaluated both at the input and at the input with subnormals flushed, and an
# output may match either. The BF16 pack stores NaN as +inf and -0 as +0, so classes are compared as
# stored. Each output must have the reference's class and a pure ULP error, |reference - output| /
# ulp(rounded reference), below 1.
# The inputs in DECLARED are stored as the class given there, the first row that holds; where
# that differs from torch, it is the class the TT-NN op this kernel replaces stores.
# Each run logs one ULP line: the largest pure ULP error against torch and the outputs of
# another class, for this program and for the path it replaces on the same input.
LOG_SIGMOID_SMALLEST_NORMAL = 2.0**-126
LOG_SIGMOID_CLASS_CODES = {"inf": 0, "-inf": 1, "zero": 2, "finite": 3}

# Per board: (inputs, as a torch expression of the input x or of daz, the input as the SFPU reads it
# with subnormals as zero, and the class their outputs are stored as).
LOG_SIGMOID_DECLARED = {
    "blackhole": [
        ("torch.isfinite(daz) & (daz > 3.3895313892515355e+38)", "zero"),  # finite x > 3.389531e+38
    ],
}


def _log_sigmoid_flush(t):
    return torch.where(t.abs() < LOG_SIGMOID_SMALLEST_NORMAL, torch.zeros_like(t), t)


def _log_sigmoid_reference(x):
    return torch.nn.functional.logsigmoid(x)


def _log_sigmoid_round_to_bfloat16(t):
    """Round float64 to BF16 once (round-to-odd into float32, then nearest-even), then flush."""
    f32 = t.to(torch.float32)
    back = f32.to(torch.float64)
    inexact = torch.isfinite(t) & (back != t)
    bits = f32.view(torch.int32) - (inexact & (back.abs() > t.abs())).to(torch.int32)
    bits = bits | inexact.to(torch.int32)
    return _log_sigmoid_flush(bits.view(torch.float32).to(torch.bfloat16))


def _log_sigmoid_stored_classes(t):
    """The CLASS_CODES of each value as stored: NaN as +inf, either zero as zero."""
    t = t.to(torch.float64)
    classes = torch.full(t.shape, LOG_SIGMOID_CLASS_CODES["finite"], dtype=torch.int8)
    classes[t == 0] = LOG_SIGMOID_CLASS_CODES["zero"]
    classes[(t == float("inf")) | torch.isnan(t)] = LOG_SIGMOID_CLASS_CODES["inf"]
    classes[t == float("-inf")] = LOG_SIGMOID_CLASS_CODES["-inf"]
    return classes


def _log_sigmoid_pure_ulp(reference, actual):
    """Pure ULP error, infinite where the stored class differs."""
    rounded = _log_sigmoid_round_to_bfloat16(reference).to(torch.float64)
    magnitude = rounded.abs()
    exponent = torch.floor(torch.log2(torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))))
    spacing = torch.where(magnitude > 0, 2.0 ** (exponent.clamp(min=-126) - 7), torch.full_like(magnitude, 2.0**-133))
    # The numerator is flushed only where the correctly rounded result is zero (post-round flush).
    golden = torch.where(magnitude == 0, torch.zeros_like(reference), reference)
    ulp = ((golden - actual.to(torch.float64)).abs().to(torch.float32) / spacing.to(torch.float32)).to(torch.float64)
    same_class = _log_sigmoid_stored_classes(rounded) == _log_sigmoid_stored_classes(actual)
    ulp = torch.where(torch.isfinite(rounded), ulp, torch.zeros_like(ulp))
    return torch.where(same_class, ulp, torch.full_like(ulp, float("inf")))


def _log_sigmoid_versus_torch(x64, output):
    """The largest pure ULP error against torch over outputs of torch's stored class, and the number
    of outputs of another class."""
    ulp = torch.minimum(
        _log_sigmoid_pure_ulp(_log_sigmoid_reference(x64), output),
        _log_sigmoid_pure_ulp(_log_sigmoid_reference(_log_sigmoid_flush(x64)), output),
    )
    mismatched = torch.isinf(ulp)
    return (ulp[~mismatched].max().item() if (~mismatched).any() else 0.0), int(mismatched.sum())


@run_for_blackhole("the generated kernel runs on Blackhole only")
def test_log_sigmoid_exhaustive_bfloat16(device):
    x = generate_all_bfloat16_bitpatterns(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.log_sigmoid(tt_x)).to(torch.bfloat16)
    # The path this program replaces, on the same input (see the device-perf test).
    stock = ttnn.to_torch(
        ttnn.unary_chain(
            tt_x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.LOGSIGMOID), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
        )
    ).to(torch.bfloat16)

    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    x64 = x.to(torch.float64)
    (ours, ours_classes), (old, old_classes) = _log_sigmoid_versus_torch(x64, actual), _log_sigmoid_versus_torch(
        x64, stock
    )
    print(
        f"ULP log_sigmoid {board} ours={ours:.3f} stock={old:.3f} "
        f"ours_class_mismatches={ours_classes} stock_class_mismatches={old_classes}"
    )
    expected = torch.full(x.shape, -1, dtype=torch.int8)
    for inputs, stored in LOG_SIGMOID_DECLARED[board]:
        lanes = eval(
            inputs,
            {"torch": torch, "x": x64, "daz": _log_sigmoid_flush(x64), "SMALLEST_NORMAL": LOG_SIGMOID_SMALLEST_NORMAL},
        )
        expected = torch.where(lanes & (expected < 0), LOG_SIGMOID_CLASS_CODES[stored], expected)
    declared = expected >= 0
    wrong = declared & (_log_sigmoid_stored_classes(actual) != expected)
    assert not wrong.any(), (
        f"{wrong.sum().item()} declared inputs of the wrong class; first x={x[wrong][0].item()}, "
        f"got {actual[wrong][0].item()}"
    )

    ulp = torch.minimum(
        _log_sigmoid_pure_ulp(_log_sigmoid_reference(x64), actual),
        _log_sigmoid_pure_ulp(_log_sigmoid_reference(_log_sigmoid_flush(x64)), actual),
    )
    ulp = torch.where(declared, torch.zeros_like(ulp), ulp)
    worst = ulp.argmax()
    assert ulp.max().item() < 1.0, (
        f"{(ulp >= 1.0).sum().item()} outputs at or beyond 1 ulp or of the wrong class; worst at "
        f"x={x.flatten()[worst].item()}: expected {_log_sigmoid_reference(x64).flatten()[worst].item()}, "
        f"got {actual.flatten()[worst].item()}"
    )


# Inputs well inside the range the generated kernel is fitted on, for the calls below.
LOG_SIGMOID_LOW, LOG_SIGMOID_HIGH = -10.0, 10.0


def _log_sigmoid_inputs(shape):
    generator = torch.Generator().manual_seed(0)
    return (
        LOG_SIGMOID_LOW + (LOG_SIGMOID_HIGH - LOG_SIGMOID_LOW) * (0.05 + 0.9 * torch.rand(shape, generator=generator))
    ).to(torch.bfloat16)


def _log_sigmoid_on_device(x, device, **kwargs):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, **kwargs)


@run_for_blackhole("the generated kernel runs on Blackhole only")
@pytest.mark.parametrize("placement", ["row_major", "height_sharded", "unaligned", "cached"])
def test_log_sigmoid_every_placement_matches_interleaved_tiles(device, placement):
    """The generated kernel computes each element alone, so placement must not change a result."""
    x = _log_sigmoid_inputs((1, 1, 64, 96))
    expected = ttnn.to_torch(ttnn.log_sigmoid(_log_sigmoid_on_device(x, device)))
    if placement == "row_major":
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    elif placement == "height_sharded":
        shard = ttnn.create_sharded_memory_config(
            x.shape, core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.HEIGHT
        )
        tt_x = _log_sigmoid_on_device(x, device, memory_config=shard)
    elif placement == "unaligned":
        x, expected = x[..., :33, :65].contiguous(), expected[..., :33, :65]
        tt_x = _log_sigmoid_on_device(x, device)
    else:
        tt_x = _log_sigmoid_on_device(x, device)
        ttnn.log_sigmoid(tt_x)
    actual = ttnn.to_torch(ttnn.log_sigmoid(tt_x))
    assert torch.equal(actual, expected)


@run_for_blackhole("the generated kernel runs on Blackhole only")
@pytest.mark.parametrize("call", ["float32", "bfloat16_to_float32"])
def test_log_sigmoid_keeps_its_own_path_elsewhere(device, call):
    """A call the generated kernel does not serve runs the op's existing path and matches its golden."""
    x = _log_sigmoid_inputs((1, 1, 64, 64))
    golden = ttnn.get_golden_function(ttnn.log_sigmoid)
    if call == "float32":
        actual = ttnn.log_sigmoid(
            ttnn.from_torch(x.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        )
        expected = golden(x.float())
    elif call == "bfloat16_to_float32":
        # Only the route is checked: the generated kernel does not compile with FP32 DEST, so any
        # result shows that the op's own kernel ran.
        output = ttnn.from_torch(torch.zeros(x.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        assert ttnn.log_sigmoid(_log_sigmoid_on_device(x, device), output_tensor=output).dtype == ttnn.float32
        return
    else:
        # Any value other than the parameter's default keeps the op's own kernel.
        actual = ttnn.log_sigmoid(_log_sigmoid_on_device(x, device), **{call: 0.125})
        expected = golden(x.float(), **{call: 0.125})
    assert_with_pcc(expected, ttnn.to_torch(actual).float(), 0.999)


# Per board that keeps TT-NN's own path: (input, that path's output, the generated kernel's output)
# BF16 words at inputs where the two differ, as measured on the board.
LOG_SIGMOID_KEPT = {
    "wormhole_b0": [
        (0x4092, 0xBC30, 0xBC2A),
        (0x40A8, 0xBBB1, 0xBBAB),
        (0x4091, 0xBC35, 0xBC2F),
        (0x4131, 0xB77F, 0xB784),
    ],
}


@run_for_wormhole_b0("TT-NN's own path is kept on this board")
def test_log_sigmoid_keeps_its_own_path(device):
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    words = torch.tensor([word for word, _, _ in LOG_SIGMOID_KEPT[board]], dtype=torch.int32).to(torch.int16)
    x = words.view(torch.bfloat16).reshape(1, -1).expand(32, -1).contiguous()
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.log_sigmoid(tt_x)).to(torch.bfloat16)[0].view(torch.int16).to(torch.int32) & 0xFFFF
    expected = torch.tensor([word for _, word, _ in LOG_SIGMOID_KEPT[board]], dtype=torch.int32)
    assert torch.equal(actual, expected), f"expected TT-NN's own path {expected.tolist()}, got {actual.tolist()}"

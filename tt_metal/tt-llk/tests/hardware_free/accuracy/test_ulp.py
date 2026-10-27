# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the integer ULP metric in ``helpers/ulp.py``: unit spacing, the
zero neighbourhood, the non-finites, the elementwise verdict and its floor, the reported
step, and what the metric refuses. No device: the metric is bit arithmetic."""

import math

import numpy as np
import pytest
import torch
from conftest import refuses
from helpers.accuracy_metrics import local_ulp
from helpers.format_config import DataFormat
from helpers.pack import float_to_bfp8_block
from helpers.stimuli_generator.strategies.structured import _enumerate_fp32_in_range
from helpers.ulp import (
    NEAR_ZERO_FRACTION,
    ULP_FORMATS,
    UNMEASURABLE,
    flushes_subnormals,
    has_ulp_gate,
    local_step,
    mantissa_bits,
    max_meaningful_ulp,
    nonfinite_mismatches,
    ulp_distance,
    ulp_dtype,
    ulp_elementwise_valid,
    ulp_stats,
    ulp_verdict_message,
)

FLOAT_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
MANTISSA_BITS = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23}
FORMAT_OF = {
    torch.bfloat16: DataFormat.Float16_b,
    torch.float16: DataFormat.Float16,
    torch.float32: DataFormat.Float32,
}
# The two bf16 gaps either side of 1.0; which one a message reports is the direction.
BELOW_ONE = 2.0**-8
ABOVE_ONE = 2.0**-7
# `pytest.approx` defaults to abs=1e-12, larger than any subnormal: comparisons at that
# scale pass abs=0.


def _t(values, dtype):
    return torch.tensor(values, dtype=dtype)


def _step_up(value, dtype, *, steps=1):
    out = _t([value], dtype)
    for _ in range(steps):
        out = torch.nextafter(out, _t([float("inf")], dtype))
    return out


def _step_down(value, dtype, *, steps=1):
    out = _t([value], dtype)
    for _ in range(steps):
        out = torch.nextafter(out, _t([float("-inf")], dtype))
    return out


def _message(golden, result, fmt=None, **kwargs):
    return ulp_verdict_message(
        golden, result, ulp_distance(golden, result), fmt, **kwargs
    )


# ── Unit spacing ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_the_value_order_is_unit_spaced_over_a_whole_16bit_format(dtype):
    """Exhaustive over 2**16 patterns, powers of two included; the span pins
    monotonicity, which the per-step bound alone does not."""
    patterns = torch.arange(-32768, 32768, dtype=torch.int16).view(dtype)
    finite = patterns[torch.isfinite(patterns)].to(torch.float64)
    if flushes_subnormals(dtype):
        tiny = torch.finfo(dtype).tiny
        finite = torch.where(finite.abs() < tiny, torch.zeros_like(finite), finite)
    ordered = torch.unique(finite).to(dtype)
    steps = ulp_distance(ordered[:-1], ordered[1:])
    assert int(steps.min()) == int(steps.max()) == 1
    assert int(ulp_distance(ordered[:1], ordered[-1:])[0]) == ordered.numel() - 1


def test_the_value_order_agrees_with_the_sweep_enumerators_key():
    """The stimuli walk float32 by a twos-complement key; unflushed, the metric's
    sign-and-magnitude rank is that same total order, +0 and -0 collapsed."""
    info = torch.finfo(torch.float32)
    probes = _t(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0**-149,
            -(2.0**-149),
            info.tiny,
            -info.tiny,
            info.max,
            -info.max,
            float("inf"),
            float("-inf"),
            3.14159265,
            -1e-30,
        ],
        torch.float32,
    )
    bits = probes.view(torch.int32).to(torch.int64)
    sweep_key = torch.where(bits < 0, -(2**31) - bits, bits)
    to_zero = ulp_distance(probes, torch.zeros_like(probes), flush_subnormals=False)
    assert to_zero.tolist() == sweep_key.abs().tolist()
    run = _enumerate_fp32_in_range(1.0, 2.0, 64)
    assert ulp_distance(run[:-1], run[1:], flush_subnormals=False).tolist() == [1] * 63


# ── The zero neighbourhood ───────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_signed_zeros_coincide(dtype):
    zeros = _t([0.0, -0.0, 0.0, -0.0], dtype)
    other = _t([-0.0, 0.0, 0.0, -0.0], dtype)
    for flush in (True, False):
        assert int(ulp_distance(zeros, other, flush_subnormals=flush).max()) == 0


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_flushed_the_smallest_normal_is_one_step_from_zero_and_the_band_is_zero(dtype):
    """Both directions: the ported ttnn index mis-based its negative half."""
    tiny = torch.finfo(dtype).tiny
    flush = {"flush_subnormals": True}
    zero, up, down = _t([0.0], dtype), _t([tiny], dtype), _t([-tiny], dtype)
    assert int(ulp_distance(zero, up, **flush)[0]) == 1
    assert int(ulp_distance(zero, down, **flush)[0]) == 1
    assert int(ulp_distance(down, up, **flush)[0]) == 2
    assert int(ulp_distance(down, _step_up(tiny, dtype), **flush)[0]) == 3
    band = _t([tiny / 2, tiny / 4, -tiny / 2, tiny / 8], dtype)
    assert bool((band != 0).all())
    assert int(ulp_distance(torch.zeros_like(band), band, **flush).max()) == 0


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_without_the_flush_the_subnormal_band_is_counted(dtype):
    tiny = torch.finfo(dtype).tiny
    zero, unflushed = _t([0.0], dtype), {"flush_subnormals": False}
    assert int(ulp_distance(zero, _t([tiny], dtype), **unflushed)[0]) == (
        1 << MANTISSA_BITS[dtype]
    )
    assert int(ulp_distance(zero, _t([tiny / 2], dtype), **unflushed)[0]) > 0


def test_the_flush_default_follows_the_harness_ftz_model():
    """bf16 and fp32 flush below their smallest normal; an fp16 golden keeps its band,
    so the default must not hide 1022 steps of fp16 error."""
    assert flushes_subnormals(torch.bfloat16) is True
    assert flushes_subnormals(torch.float32) is True
    assert flushes_subnormals(torch.float16) is False
    smallest = 2.0**-24
    golden, result = _t([smallest], torch.float16), _t([1023 * smallest], torch.float16)
    assert int(ulp_distance(golden, result)[0]) == 1022
    assert int(ulp_distance(golden, result, flush_subnormals=True)[0]) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_flushing_is_a_no_op_for_same_sign_pairs_outside_the_band(dtype):
    """The compaction shifts each magnitude rank, so it cancels in a same-sign
    subtraction and survives a sign-crossing one."""
    torch.manual_seed(0)
    values = torch.randn(256, dtype=torch.float32).to(dtype)
    other = torch.nextafter(values, torch.full_like(values, float("inf")))
    assert bool((values.abs() >= torch.finfo(dtype).tiny).all())
    assert torch.equal(
        ulp_distance(values, other, flush_subnormals=True),
        ulp_distance(values, other, flush_subnormals=False),
    )
    one, minus_one = _t([1.0], dtype), _t([-1.0], dtype)
    flushed = int(ulp_distance(one, minus_one, flush_subnormals=True)[0])
    unflushed = int(ulp_distance(one, minus_one, flush_subnormals=False)[0])
    assert unflushed - flushed == 2 * (2 ** MANTISSA_BITS[dtype] - 1)


# ── Non-finites ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_inf_is_one_step_past_the_largest_finite(dtype):
    largest = torch.finfo(dtype).max
    assert int(ulp_distance(_t([largest], dtype), _t([float("inf")], dtype))[0]) == 1
    assert int(ulp_distance(_t([-largest], dtype), _t([float("-inf")], dtype))[0]) == 1


def test_nan_lanes_are_unmeasurable():
    nan = float("nan")
    golden = _t([1.0, nan, 2.0, nan], torch.float32)
    result = _t([1.0, 3.0, nan, nan], torch.float32)
    assert ulp_distance(golden, result).tolist() == [0] + [UNMEASURABLE] * 3


def test_nonfinite_mismatches_is_positional_only():
    nan, inf, largest = float("nan"), float("inf"), torch.finfo(torch.float32).max
    golden = _t([nan, nan, inf, inf, inf, 1.0], torch.float32)
    result = _t([nan, 1.0, inf, -inf, largest, nan], torch.float32)
    #     both NaN  one NaN  both +inf  opposite  inf vs finite  one NaN
    expected = [False, True, False, True, True, True]
    assert nonfinite_mismatches(golden, result).tolist() == expected


def test_a_nan_sign_is_not_judged_here():
    """-NaN folds to the other operand on Wormhole; sfpu_domains owns the NaN-sign
    rule, so the metric must not hold a second opinion."""
    positive = _t([float("nan")], torch.float32)
    assert bool(torch.signbit(-positive)[0])
    assert nonfinite_mismatches(positive, -positive).tolist() == [False]


# ── The elementwise verdict ──────────────────────────────────────────────────


def test_elementwise_valid_marks_only_the_out_of_budget_lanes():
    dtype = torch.bfloat16
    golden = torch.ones(4, dtype=dtype)
    result = torch.cat(
        [
            _t([1.0], dtype),
            _step_up(1.0, dtype),
            _step_up(1.0, dtype, steps=2),
            _step_down(1.0, dtype, steps=3),
        ]
    )
    is_valid, distance, _ = ulp_elementwise_valid(golden, result, 1)
    assert distance.tolist() == [0, 1, 2, 3]
    assert is_valid.tolist() == [True, True, False, False]


def test_elementwise_valid_accepts_both_nan_and_rejects_a_missing_one():
    nan = float("nan")
    golden = _t([nan, nan, 1.0], torch.float32)
    result = _t([nan, 1.0, nan], torch.float32)
    assert ulp_elementwise_valid(golden, result, 0)[0].tolist() == [True, False, False]


def test_elementwise_valid_rejects_an_overflow_even_though_it_ranks_one_step():
    golden = _t([torch.finfo(torch.float32).max], torch.float32)
    result = _t([float("inf")], torch.float32)
    budget = max_meaningful_ulp(torch.float32)
    is_valid, distance, _ = ulp_elementwise_valid(golden, result, budget)
    assert int(distance[0]) == 1 and is_valid.tolist() == [False]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_the_verdict_resolves_the_flush_default_per_dtype_like_the_metric(dtype):
    """fp16 keeps its band here: a 0-step budget must not accept its 1022 steps."""
    smallest = float(torch.finfo(dtype).tiny) * 2.0 ** -MANTISSA_BITS[dtype]
    band_steps = (1 << MANTISSA_BITS[dtype]) - 1
    golden, result = _t([smallest], dtype), _t([band_steps * smallest], dtype)
    is_valid, distance, _ = ulp_elementwise_valid(golden, result, 0)
    if dtype is torch.float16:
        assert int(distance[0]) == band_steps - 1 and not bool(is_valid[0])
    else:
        assert int(distance[0]) == 0 and bool(is_valid[0])
    assert bool(ulp_elementwise_valid(golden, result, 0, flush_subnormals=True)[0][0])


# ── The near-zero floor ──────────────────────────────────────────────────────


def test_the_floor_rescues_a_cancellation_lane():
    golden, result = _t([100.0, 1e-8], torch.float32), _t([100.0, 2e-8], torch.float32)
    is_valid, distance, _ = ulp_elementwise_valid(golden, result, 4)
    assert int(distance[1]) > 1000
    assert is_valid.tolist() == [True, False]
    is_valid, _, rescued = ulp_elementwise_valid(golden, result, 4, near_zero_atol=1e-7)
    assert is_valid.tolist() == [True, True] and rescued.tolist() == [False, True]


def test_the_floor_does_not_loosen_the_large_magnitude_lanes():
    golden, result = _t([100.0, 1e-8], torch.float32), _t([100.5, 1e-8], torch.float32)
    is_valid, _, _ = ulp_elementwise_valid(golden, result, 0, near_zero_atol=1.0)
    assert is_valid.tolist() == [False, True]


def test_the_near_zero_cut_is_a_fraction_of_the_tensors_dynamic_range():
    dynamic_range = 100.0
    below = dynamic_range * NEAR_ZERO_FRACTION * 0.9
    above = dynamic_range * NEAR_ZERO_FRACTION * 1.1
    golden = _t([dynamic_range, below, above], torch.float32)
    result = golden + _t([0.0, 0.5, 0.5], torch.float32)
    is_valid, _, _ = ulp_elementwise_valid(golden, result, 0, near_zero_atol=1.0)
    assert is_valid.tolist() == [True, True, False]


def test_the_floor_covers_every_lane_of_an_all_zero_golden():
    golden = torch.zeros(3, dtype=torch.float32)
    result = _t([0.0, 1e-9, 1.0], torch.float32)
    is_valid, _, _ = ulp_elementwise_valid(golden, result, 0, near_zero_atol=1e-8)
    assert is_valid.tolist() == [True, True, False]


def test_every_floor_covered_lane_is_reported_covered_even_within_budget():
    """The third return is every lane the floor judges, not only the ones it had to
    save: the emitter's residual is the maximum over the lanes outside it."""
    golden = _t([100.0, 1e-8, 50.0], torch.float32)
    result = golden.clone()
    result[2] = 50.00001  # outside the band, 3 steps
    _, _, covered = ulp_elementwise_valid(golden, result, 4, near_zero_atol=1e-7)
    assert covered.tolist() == [False, True, False]


def test_the_near_zero_band_is_bounded_absolutely_as_well_as_relatively():
    """One large golden would otherwise widen the relative band across the tile."""
    near_zero_atol = 1e-6
    golden = _t([1000.0, 1.0, 1e-9], torch.float32)
    result = _t([1000.0, 1.0000008, 1e-9 + 5e-7], torch.float32)
    assert 1.0 < NEAR_ZERO_FRACTION * 1000.0  # the relative rule alone would rescue it
    assert 1.0 > near_zero_atol / NEAR_ZERO_FRACTION  # the absolute rule refuses it
    is_valid, _, rescued = ulp_elementwise_valid(
        golden, result, 1, near_zero_atol=near_zero_atol
    )
    assert is_valid.tolist() == [True, False, True]
    assert rescued.tolist() == [False, False, True]
    assert not ulp_elementwise_valid(golden, result, 1, near_zero_atol=1e-9)[0][2]


def test_the_near_zero_band_is_scoped_to_the_lanes_under_judgement():
    """`dynamic_range` is the one verdict input that is not elementwise: a large golden
    in a masked-out lane must not widen the band for the judged ones."""
    golden, result = _t([1e6, 1.0], torch.float32), _t([1e6, 1.02], torch.float32)
    assert int(ulp_distance(golden, result)[1]) > 100000
    judged = torch.tensor([False, True])
    scoped, _, _ = ulp_elementwise_valid(
        golden, result, 0, near_zero_atol=0.05, selected=judged
    )
    assert not bool(scoped[1])
    is_valid, _, rescued = ulp_elementwise_valid(golden, result, 0, near_zero_atol=0.05)
    assert bool(is_valid.all()) and bool(rescued[1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_the_band_cuts_are_compared_in_float32_not_the_tensor_dtype(dtype):
    """Both cuts are Python floats; a 16-bit ``golden.abs()`` would round them onto the
    tensor's lattice, the relative ``<`` edge down and the absolute ``<=`` edge up."""

    def verdict(golden, result, atol):
        valid, _, _ = ulp_elementwise_valid(golden, result, 0, near_zero_atol=atol)
        return bool(valid.all())

    dynamic_range = 11.0  # its 1% rounds down in both dtypes
    cut = NEAR_ZERO_FRACTION * dynamic_range
    rounded = float(torch.tensor(cut, dtype=dtype))
    assert rounded < cut
    golden = _t([dynamic_range, (cut + rounded) / 2.0], dtype)
    assert float(golden[1]) == rounded
    result = golden.clone()
    result[1] = float(_step_up(float(golden[1]), dtype, steps=4)[0])
    assert verdict(golden, result, 1.0) == verdict(golden.float(), result.float(), 1.0)

    near_zero_atol = 0.012  # its absolute cut, 1.2, rounds up in both dtypes
    rounded = float(torch.tensor(near_zero_atol / NEAR_ZERO_FRACTION, dtype=dtype))
    assert rounded > near_zero_atol / NEAR_ZERO_FRACTION
    golden = _t([1000.0, rounded], dtype)
    result = golden.clone()
    result[1] = float(_step_up(rounded, dtype, steps=1)[0])
    assert int(ulp_distance(golden, result)[1]) == 1
    assert float(result[1]) - rounded <= near_zero_atol
    wide = verdict(golden.float(), result.float(), near_zero_atol)
    assert verdict(golden, result, near_zero_atol) == wide and not wide


# ── Statistics and the reported step ─────────────────────────────────────────


def test_ulp_stats_keeps_unmeasurable_lanes_out_of_the_aggregates():
    stats = ulp_stats(_t([0, 2, UNMEASURABLE, 4, UNMEASURABLE], torch.int64))
    assert (stats["lanes"], stats["unmeasurable"], stats["max"]) == (3, 2, 4)
    assert stats["worst_index"] == 3
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["exact_frac"] == pytest.approx(1 / 3)
    empty = ulp_stats(torch.full((4,), UNMEASURABLE, dtype=torch.int64))
    assert (empty["lanes"], empty["unmeasurable"], empty["max"]) == (0, 4, 0)
    assert empty["worst_index"] is None
    assert all(math.isnan(empty[key]) for key in ("mean", "p95", "p99", "exact_frac"))


def test_the_quantiles_fall_back_to_the_max_below_their_thresholds():
    stats = ulp_stats(_t([0, 0, 0, 9, 0], torch.int64))
    assert stats["p95"] == stats["p99"] == pytest.approx(9.0)
    wide = torch.zeros(200, dtype=torch.int64)
    wide[-1] = 9
    stats = ulp_stats(wide)
    assert stats["p95"] == pytest.approx(0.0) and stats["p99"] < 9.0
    middle = torch.zeros(50, dtype=torch.int64)  # a real p95, p99 still falling back
    middle[-1] = 9
    stats = ulp_stats(middle)
    assert stats["p95"] == pytest.approx(0.0) and stats["p99"] == pytest.approx(9.0)


def test_a_non_contiguous_input_is_measured_correctly():
    """The flatten inside the metric stays in step with the caller's flat indexing."""
    golden = torch.arange(24).reshape(4, 6).to(torch.bfloat16).t()
    assert not golden.is_contiguous()
    steps, position = 5, (3, 1)
    result = golden.clone()
    result[position] = _step_up(float(golden[position]), torch.bfloat16, steps=steps)[0]
    stats = ulp_stats(ulp_distance(golden, result))
    assert stats["max"] == steps
    assert stats["worst_index"] == position[0] * golden.shape[1] + position[1]


def test_the_distance_is_symmetric_and_shape_preserving():
    torch.manual_seed(0)
    golden = torch.randn(3, 5, 7, dtype=torch.float32).to(torch.bfloat16)
    result = golden.clone()
    flat = result.reshape(-1)
    flat[::3] = torch.nextafter(flat[::3], torch.full_like(flat[::3], float("inf")))
    forward = ulp_distance(golden, result)
    assert forward.shape == golden.shape and forward.dtype == torch.int64
    assert torch.equal(forward, ulp_distance(result, golden))
    assert int(forward.max()) == 1
    assert int(ulp_distance(golden, golden.clone()).max()) == 0


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_local_step_agrees_with_the_sweep_metric(dtype):
    largest = float(torch.finfo(dtype).max)
    for value in (1.0, -1.0, 3.75, 1024.0, largest):
        expected = float(local_ulp(np.array([value]), FORMAT_OF[dtype])[0])
        assert math.isfinite(expected)
        assert local_step(value, dtype) == pytest.approx(expected)
    assert math.isnan(local_step(float("inf"), dtype))


def test_the_reported_step_takes_the_direction_the_result_moved():
    """The gap below a power of two is half the gap above it; direction is the signed
    delta, so 1.0 -> -2.0 is a step down."""
    for toward, expected in (
        (None, ABOVE_ONE),
        (2.0, ABOVE_ONE),
        (0.5, BELOW_ONE),
        (0.0, BELOW_ONE),
        (-0.5, BELOW_ONE),
        (-2.0, BELOW_ONE),
    ):
        assert local_step(1.0, torch.bfloat16, toward=toward) == pytest.approx(
            expected
        ), toward
    assert local_step(-1.0, torch.bfloat16, toward=2.0) == pytest.approx(BELOW_ONE)
    assert local_step(-1.0, torch.bfloat16, toward=-2.0) == pytest.approx(ABOVE_ONE)
    golden, fmt = _t([1.0], torch.bfloat16), DataFormat.Float16_b
    down = _message(golden, _step_down(1.0, torch.bfloat16), fmt)
    assert f"{BELOW_ONE:.6e}" in down and f"{ABOVE_ONE:.6e}" not in down
    up = _message(golden, _step_up(1.0, torch.bfloat16, steps=3), fmt)
    assert f"{ABOVE_ONE:.6e}" in up and "max 3 ULP @ [0]" in up


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_the_reported_step_across_a_flushed_band_is_the_jump_it_makes(dtype):
    """Flushed, one counted step out of zero is `tiny` either way; the raw nextafter gap
    there is the smallest subnormal, 2**mantissa_bits smaller."""
    tiny = float(torch.finfo(dtype).tiny)
    raw_gap = tiny * 2.0 ** -MANTISSA_BITS[dtype]
    assert int(ulp_distance(_t([tiny], dtype), _t([0.0], dtype))[0]) == 1
    assert local_step(0.0, dtype) == pytest.approx(tiny, abs=0)
    assert local_step(tiny / 4, dtype) == pytest.approx(tiny, abs=0)
    assert local_step(tiny, dtype, toward=0.0) == pytest.approx(tiny, abs=0)
    assert local_step(tiny, dtype, toward=2.0 * tiny) == pytest.approx(raw_gap, abs=0)
    assert local_step(tiny, dtype, toward=0.0, flush_subnormals=False) == pytest.approx(
        raw_gap, abs=0
    )


def test_the_step_at_zero_has_no_downward_direction():
    """`copysign(1.0, 0.0)` is +1.0 and `nextafter(+0.0, 0.0)` is +0.0: the step at zero
    must not come back 0.0. fp16 reaches it, since it does not flush."""
    smallest = 2.0**-24
    for toward in (-1.0, -smallest, 1.0, None, 0.0):
        assert local_step(0.0, torch.float16, toward=toward) == pytest.approx(
            smallest, abs=0
        ), toward
    assert local_step(-0.0, torch.float16, toward=1.0) == pytest.approx(smallest, abs=0)
    for dtype in (torch.bfloat16, torch.float32):
        tiny = float(torch.finfo(dtype).tiny)
        assert local_step(0.0, dtype, toward=-1.0) == pytest.approx(tiny, abs=0)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_the_step_at_the_top_of_the_range_stays_finite(dtype):
    """Upward from the largest finite is Inf, so the step reported there is the downward
    one; a saturated pair reports no step rather than a NaN one."""
    largest = float(torch.finfo(dtype).max)
    expected = largest - float(_step_down(largest, dtype))
    for toward in (None, float("inf"), -largest):
        assert local_step(largest, dtype, toward=toward) == pytest.approx(
            expected
        ), toward
    assert "1 ULP = inf" not in _message(_t([largest], dtype), _t([-largest], dtype))
    saturated = _t([float("inf")], dtype)
    assert "nan" not in _message(saturated, saturated.clone())


# ── What the metric refuses ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fmt, expected",
    [
        (DataFormat.Float32, torch.float32),
        (DataFormat.Float16, torch.float16),
        (DataFormat.Float16_b, torch.bfloat16),
    ],
    ids=lambda x: getattr(x, "name", str(x)),
)
def test_ulp_dtype_maps_the_float_formats(fmt, expected):
    assert fmt in ULP_FORMATS and has_ulp_gate(fmt)
    assert ulp_dtype(fmt) == expected


@pytest.mark.parametrize(
    "fmt", [DataFormat.Bfp4_b, DataFormat.Tf32, DataFormat.Int32], ids=lambda f: f.name
)
def test_a_format_without_a_per_element_ulp_is_refused(fmt):
    """Bfp4_b leaves 2 fractional bits, Tf32 sits in an fp32 container that is not its
    lattice, and an integer wants bit equality."""
    assert not has_ulp_gate(fmt)
    with refuses("no per-element ULP"):
        ulp_dtype(fmt)


def test_bfp8_b_is_measured_in_bf16_proxy_space():
    """Its 7 magnitude bits include an explicit leading 1, so one Bfp8_b step is two
    bf16 steps: a budget in bf16 steps buys half as many format steps."""
    assert DataFormat.Bfp8_b not in ULP_FORMATS and has_ulp_gate(DataFormat.Bfp8_b)
    assert ulp_dtype(DataFormat.Bfp8_b) == torch.bfloat16
    block = [1.0] * 16
    _, baseline = float_to_bfp8_block(block)
    encoded = []
    for steps in range(3):
        block[0] = float(_step_up(1.0, torch.bfloat16, steps=steps)[0])
        encoded.append(float_to_bfp8_block(block)[1][0])
    assert encoded[0] == encoded[1] == baseline[0]
    assert encoded[2] == baseline[0] + 1
    values = np.array([1.0, 2.0])  # the plot harness measures the proxy too
    assert not np.isnan(local_ulp(values, DataFormat.Bfp8_b)).any()
    assert np.isnan(local_ulp(values, DataFormat.Bfp4_b)).all()


@pytest.mark.parametrize(
    "golden_dtype, result_dtype, shape, match",
    [
        (torch.float32, torch.bfloat16, (2, 2), "dtype mismatch"),
        (torch.float32, torch.float32, (2, 3), "shape mismatch"),
        (torch.float64, torch.float64, (2, 2), "unsupported dtype"),
        (torch.int32, torch.int32, (2, 2), "unsupported dtype"),
    ],
    ids=["dtype", "shape", "float64", "int32"],
)
def test_ulp_distance_refuses_what_it_cannot_measure(
    golden_dtype, result_dtype, shape, match
):
    golden = torch.zeros(shape[0], dtype=golden_dtype)
    result = torch.zeros(shape[1], dtype=result_dtype)
    with refuses(match):
        ulp_distance(golden, result)


def test_a_wrong_shaped_or_non_boolean_mask_is_refused():
    """(1,) would broadcast through and judge every lane or none; an integer mask is
    combined bitwise, where a truthy 2 drops the lane it meant to select."""
    golden = torch.ones(2, 6, dtype=torch.bfloat16)
    distance = ulp_distance(golden, golden.clone())
    with refuses("mask shape"):
        ulp_stats(distance, torch.ones(1, dtype=torch.bool))
    with refuses("mask must be bool"):
        ulp_stats(distance, torch.full((2, 6), 2, dtype=torch.int64))


@pytest.mark.parametrize(
    "dtype",
    [torch.float64, torch.int16, torch.int32, torch.int64, torch.bool],
    ids=str,
)
def test_every_entry_point_refuses_an_unmeasured_dtype(dtype):
    """Including the bit containers: reading a bfloat16 through int16 must not make an
    int16 tensor measurable. `local_step` guards on its own, since an explicit
    flush_subnormals= skips the lookup."""
    values = torch.ones(4, dtype=dtype)
    with refuses("unsupported dtype"):
        ulp_distance(values, values.clone())
    with refuses("unsupported dtype"):
        flushes_subnormals(dtype)
    with refuses("unsupported dtype"):
        local_step(1.0, dtype, flush_subnormals=True)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_the_mantissa_width_and_the_meaningful_ceiling_follow_the_dtype(dtype):
    assert mantissa_bits(dtype) == MANTISSA_BITS[dtype]
    assert max_meaningful_ulp(dtype) == 1 << MANTISSA_BITS[dtype]

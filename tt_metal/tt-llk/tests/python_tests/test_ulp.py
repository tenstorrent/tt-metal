# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the integer ULP metric in ``helpers/ulp.py``.

No kernel, no device: the metric is pure bit arithmetic, and a future accuracy budget
will be compared against it, so the properties the gate rests on are pinned here.

Grouped by the property each test defends: unit spacing (one representable step reads as
1 everywhere, including across a power of two), the zero neighbourhood (DAZ+FTZ removes
the subnormal band, and both signed zeros are one value), and the non-finites (``Inf``
has a rank, ``NaN`` does not, and the sentinel for "no rank" is negative -- so a caller
comparing it against a budget would pass).
"""

import math

import pytest
import torch
from helpers.accuracy_metrics import local_ulp
from helpers.format_config import DataFormat
from helpers.ulp import (
    _MIN_LANES_FOR_P95,
    _MIN_LANES_FOR_P99,
    _ULP_DTYPES,
    MAX_MEANINGFUL_ULP,
    ULP_FORMATS,
    UNMEASURABLE,
    _value_order_index,
    flushes_subnormals,
    local_step,
    nonfinite_mismatches,
    ulp_distance,
    ulp_dtype,
    ulp_failure_message,
    ulp_stats,
    warn_if_threshold_unmeaningful,
    within_ulp,
)

FLOAT_DTYPES = [torch.bfloat16, torch.float16, torch.float32]

# Mantissa bits after the implicit leading 1 -- the size of the band the flush compacts.
MANTISSA_BITS = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23}

# The two bf16 gaps either side of 1.0, bound once: the whole point of the direction
# handling is which of the two gets reported, and a -8/-7 transposition in any assertion
# that uses them would assert the wrong direction instead of failing.
BELOW_ONE = 2.0**-8
ABOVE_ONE = 2.0**-7

# `pytest.approx` defaults to `abs=1e-12`, which is larger than every value in a subnormal
# band, so a bare `approx(tiny)` accepts any subnormal-scale number at all. Comparisons at
# that scale below pass `abs=0`, leaving the relative tolerance as the only one in play.


def _t(values, dtype):
    return torch.tensor(values, dtype=dtype)


def _step_up(value, dtype, *, steps=1):
    """*value* moved *steps* representable values toward +inf. Keyword-only, so a bare
    third argument cannot read as a coordinate."""
    out = _t([value], dtype)
    for _ in range(steps):
        out = torch.nextafter(out, _t([float("inf")], dtype))
    return out


def _step_down(value, dtype, *, steps=1):
    out = _t([value], dtype)
    for _ in range(steps):
        out = torch.nextafter(out, _t([float("-inf")], dtype))
    return out


def _refuses(match):
    """The suite's ``expect_error`` fixture needs a device; these are host-only tests."""
    return pytest.raises(ValueError, match=match)  # allow-pytest.raises: host-only test


# ─────────────────────────────────────────────────────────────────────────────
# Unit spacing
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("value", [1.0, 3.5, 100.0, -1.0, -7.25, 0.125])
def test_adjacent_representable_values_are_one_step(dtype, value):
    golden = _t([value], dtype)
    assert int(ulp_distance(golden, _step_up(value, dtype))[0]) == 1
    assert int(ulp_distance(golden, _step_down(value, dtype))[0]) == 1


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("steps", [2, 3, 17])
def test_n_representable_steps_read_as_n(dtype, steps):
    golden = _t([1.0], dtype)
    assert int(ulp_distance(golden, _step_up(1.0, dtype, steps=steps))[0]) == steps


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("boundary", [0.5, 1.0, 2.0, 256.0])
def test_a_power_of_two_boundary_is_one_step_in_both_directions(dtype, boundary):
    """The whole reason the gate counts steps instead of dividing by ``ulp(golden)``."""
    golden = _t([boundary], dtype)
    assert int(ulp_distance(golden, _step_down(boundary, dtype))[0]) == 1
    assert int(ulp_distance(golden, _step_up(boundary, dtype))[0]) == 1


def test_the_fractional_metric_disagrees_at_a_power_of_two():
    """The motivation, asserted: one step reads 0.5 below the boundary and 1.0 above it,
    so a fractional "1 ULP" budget buys two steps on one side and one on the other."""
    one = _t([1.0], torch.bfloat16)
    below, above = _step_down(1.0, torch.bfloat16), _step_up(1.0, torch.bfloat16)
    step_at_one = local_ulp(one.to(torch.float64).numpy(), DataFormat.Float16_b)[0]

    assert abs(float(below) - 1.0) / step_at_one == pytest.approx(0.5)
    assert abs(float(above) - 1.0) / step_at_one == pytest.approx(1.0)

    # The integer metric calls both of them one step, and the pair two.
    assert int(ulp_distance(one, below)[0]) == 1
    assert int(ulp_distance(one, above)[0]) == 1
    assert int(ulp_distance(below, above)[0]) == 2


def test_the_bf16_value_order_is_unit_spaced_over_the_whole_format():
    """Exhaustive over 2**16 patterns, so an ordering defect anywhere in the range fails,
    not only at the handful of points a parametrized test names."""
    patterns = torch.arange(-32768, 32768, dtype=torch.int16).view(torch.bfloat16)
    finite = patterns[torch.isfinite(patterns)].to(torch.float64)
    tiny = torch.finfo(torch.bfloat16).tiny
    flushed = torch.where(finite.abs() < tiny, torch.zeros_like(finite), finite)
    ordered = torch.unique(flushed).to(torch.bfloat16)

    assert int(ulp_distance(ordered[:-1], ordered[1:]).min()) == 1
    assert int(ulp_distance(ordered[:-1], ordered[1:]).max()) == 1
    # The per-step bound alone does not force monotonicity -- `ulp_distance` takes
    # `.abs()`, so a rank walk that oscillates satisfies it. The span pins the rest.
    assert int(ulp_distance(ordered[:1], ordered[-1:])[0]) == ordered.numel() - 1


def test_the_value_order_agrees_with_the_sweep_enumerators_key():
    """The stimuli and the metric have to mean the same thing by "one representable step".

    ``UlpSweepStrategy`` walks the float32 line by a twos-complement key; unflushed, this
    module's sign-and-magnitude rank is that same total order, ``+0``/``-0`` collapse
    included. Claimed for the unflushed index only -- the compaction is this module's own.
    """
    from helpers.stimuli_generator.strategies.structured import _enumerate_fp32_in_range

    info = torch.finfo(torch.float32)
    probes = _t(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0**-149,  # the smallest subnormal, which the flush would compact
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
    ours = _value_order_index(
        probes, _ULP_DTYPES[torch.float32], flush_subnormals=False
    )
    assert ours.tolist() == sweep_key.tolist()

    # ...and end to end, so "the Nth value in the sweep" and "N steps away" cannot drift.
    run = _enumerate_fp32_in_range(1.0, 2.0, 64)
    assert ulp_distance(run[:-1], run[1:], flush_subnormals=False).tolist() == [1] * 63


# ─────────────────────────────────────────────────────────────────────────────
# The zero neighbourhood: signed zeros, subnormals, the flush
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("flush", [True, False])
def test_signed_zeros_coincide(dtype, flush):
    """``-0.0`` dies on unpack and the pack path canonicalises it again, so the one step a
    raw bit ordering reports is an encoding artefact. True with or without the flush."""
    zeros = _t([0.0, -0.0, 0.0, -0.0], dtype)
    other = _t([-0.0, 0.0, 0.0, -0.0], dtype)
    assert int(ulp_distance(zeros, other, flush_subnormals=flush).max()) == 0


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_flush_makes_the_smallest_normal_one_step_from_zero(dtype):
    """The crossing the ported ttnn index gets wrong in its negative half: a base one
    subnormal-band too far out keeps same-sign distances correct and inflates every one of
    these, which is why both directions are asserted."""
    tiny = torch.finfo(dtype).tiny
    flush = {"flush_subnormals": True}  # fp16 does not flush by default
    zero, up, down = _t([0.0], dtype), _t([tiny], dtype), _t([-tiny], dtype)
    assert int(ulp_distance(zero, up, **flush)[0]) == 1
    assert int(ulp_distance(zero, down, **flush)[0]) == 1
    assert int(ulp_distance(down, up, **flush)[0]) == 2
    assert int(ulp_distance(down, _step_up(tiny, dtype), **flush)[0]) == 3


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_every_subnormal_is_zero_steps_from_zero_when_flushed(dtype):
    """A flushed result next to a subnormal golden is a 0-step agreement in the SFPU's own
    number system, not a ``2**mantissa_bits - 1`` error."""
    tiny = torch.finfo(dtype).tiny
    band = _t([tiny / 2, tiny / 4, -tiny / 2, tiny / 8], dtype)
    assert bool((band != 0).all()) and bool((band.abs() < tiny).all())
    zeros = torch.zeros_like(band)
    assert int(ulp_distance(zeros, band, flush_subnormals=True).max()) == 0


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_without_the_flush_the_subnormal_band_is_counted(dtype):
    """``flush_subnormals=False`` is the IEEE ordering, kept for anything that is not the
    SFPU: there the band is real and zero is a whole band below the smallest normal."""
    tiny = torch.finfo(dtype).tiny
    zero, unflushed = _t([0.0], dtype), {"flush_subnormals": False}
    assert int(ulp_distance(zero, _t([tiny], dtype), **unflushed)[0]) == (
        1 << MANTISSA_BITS[dtype]
    )
    assert int(ulp_distance(zero, _t([tiny / 2], dtype), **unflushed)[0]) > 0


def test_the_flush_default_follows_the_harness_ftz_model():
    """bf16 and fp32 flush below their smallest normal, so the collapse is a no-op there.
    fp16's ``golden_generators._FTZ_THRESHOLD`` is the smallest fp16 *subnormal*, so an
    fp16 golden legitimately carries the whole band."""
    assert flushes_subnormals(torch.bfloat16) is True
    assert flushes_subnormals(torch.float32) is True
    assert flushes_subnormals(torch.float16) is False


def test_fp16_subnormals_are_measured_not_collapsed():
    """The blind spot a blanket flush would open: ``2**-24`` against ``1023 * 2**-24`` is
    1022 steps and a ~1000x relative error, and a 1-step fp16 gate would see none of it.
    """
    smallest = 2.0**-24
    golden, result = _t([smallest], torch.float16), _t([1023 * smallest], torch.float16)
    assert int(ulp_distance(golden, result)[0]) == 1022
    # The blanket behaviour stays available for a caller that knows the Dest flushed.
    assert int(ulp_distance(golden, result, flush_subnormals=True)[0]) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_flushing_is_a_no_op_for_the_formats_that_already_flush(dtype):
    """Why the default is safe -- but only for *same-sign pairs outside the band*, which
    is every pair these two formats can produce here. The compaction shifts each magnitude
    rank, so it cancels in a same-sign subtraction and survives a sign-crossing one."""
    torch.manual_seed(0)
    values = torch.randn(256, dtype=torch.float32).to(dtype)
    other = torch.nextafter(values, torch.full_like(values, float("inf")))
    assert bool((values.abs() >= torch.finfo(dtype).tiny).all())
    assert torch.equal(
        ulp_distance(values, other, flush_subnormals=True),
        ulp_distance(values, other, flush_subnormals=False),
    )

    # ...and the crossing where they do not agree, so the narrowed claim is tested too.
    one, minus_one = _t([1.0], dtype), _t([-1.0], dtype)
    flushed = int(ulp_distance(one, minus_one, flush_subnormals=True)[0])
    unflushed = int(ulp_distance(one, minus_one, flush_subnormals=False)[0])
    assert unflushed - flushed == 2 * (2 ** MANTISSA_BITS[dtype] - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Non-finites
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_inf_is_one_step_past_the_largest_finite(dtype):
    largest = torch.finfo(dtype).max
    assert int(ulp_distance(_t([largest], dtype), _t([float("inf")], dtype))[0]) == 1
    assert int(ulp_distance(_t([-largest], dtype), _t([float("-inf")], dtype))[0]) == 1


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_a_spurious_inf_away_from_the_top_is_a_large_distance(dtype):
    distance = int(ulp_distance(_t([1.0], dtype), _t([float("inf")], dtype))[0])
    assert distance > MAX_MEANINGFUL_ULP[dtype]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_nan_lanes_are_unmeasurable(dtype):
    nan = float("nan")
    golden, result = _t([1.0, nan, 2.0, nan], dtype), _t([1.0, 3.0, nan, nan], dtype)
    assert ulp_distance(golden, result).tolist() == [0] + [UNMEASURABLE] * 3


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_nonfinite_mismatches_is_positional_only(dtype):
    nan, inf, largest = float("nan"), float("inf"), torch.finfo(dtype).max
    golden = _t([nan, nan, inf, inf, inf, 1.0], dtype)
    result = _t([nan, 1.0, inf, -inf, largest, nan], dtype)
    #     both NaN  one NaN  both +inf  opposite  inf vs finite  one NaN
    expected = [False, True, False, True, True, True]
    assert nonfinite_mismatches(golden, result).tolist() == expected


def test_a_nan_sign_is_not_judged_here():
    """``-NaN`` folds to the other operand on Wormhole and ``sfpu_domains`` owns when a
    NaN's sign may be asserted, so the metric must not hold a second opinion."""
    positive = _t([float("nan")], torch.float32)
    assert bool(torch.signbit(-positive)[0])
    assert nonfinite_mismatches(positive, -positive).tolist() == [False]


# ─────────────────────────────────────────────────────────────────────────────
# The composite verdict
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_the_budget_boundary_is_inclusive(dtype):
    budget = 3  # one name for both, so this cannot drift into 2 <= 3 and stay green
    golden = _t([1.0], dtype)
    assert within_ulp(golden, _step_up(1.0, dtype, steps=budget), max_ulp=budget)[0]
    assert not within_ulp(
        golden, _step_up(1.0, dtype, steps=budget + 1), max_ulp=budget
    )[0]


def test_the_verdict_does_not_pass_on_the_unmeasurable_sentinel():
    """The footgun the composite exists to remove: ``-1 <= max_ulp`` for every budget, so
    a caller gating on the raw distance would call a missing NaN a pass."""
    golden = _t([1.0, float("nan")], torch.float32)
    result = _t([1.0, 2.0], torch.float32)
    assert bool((ulp_distance(golden, result) <= 0).all())  # the naive gate would pass
    ok, message = within_ulp(golden, result, max_ulp=0)
    assert not ok
    assert "non-finite disagreement" in message


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_the_verdict_refuses_an_overflow_no_budget_can_buy(dtype):
    """The distance and the verdict part company here, deliberately: ``Inf`` ranks one
    past the largest finite, but an overflow is a different kind of answer from an inexact
    one, so ``within_ulp`` rejects it positionally -- the line
    ``utils.py::_bfp_block_aware_compare`` already takes for the block floats."""
    largest = float(torch.finfo(dtype).max)
    golden, result = _t([largest], dtype), _t([float("inf")], dtype)
    assert int(ulp_distance(golden, result)[0]) == 1  # measured, it is one step

    for budget in (0, 1, MAX_MEANINGFUL_ULP[dtype]):
        assert not within_ulp(golden, result, max_ulp=budget)[0], budget
    assert not within_ulp(result, golden, max_ulp=1)[0]  # and the other direction
    # Same-sign Inf against Inf is the only Inf lane that reaches the distance: 0 steps.
    assert within_ulp(result, result.clone(), max_ulp=0)[0]


def test_the_verdict_passes_when_both_sides_are_nan():
    values = _t([1.0, float("nan")], torch.float32)
    ok, message = within_ulp(values, values.clone(), max_ulp=0)
    assert ok
    assert "1 unmeasurable" in message


def test_a_mask_excludes_lanes_an_op_has_already_settled():
    golden, result = _t([1.0, 1.0], torch.float32), _t([1.0, 1000.0], torch.float32)
    assert not within_ulp(golden, result, max_ulp=1)[0]
    assert within_ulp(golden, result, max_ulp=1, mask=torch.tensor([True, False]))[0]


def test_the_verdict_passes_when_every_lane_is_masked_out():
    golden, result = _t([1.0, 1.0], torch.float32), _t([5.0, 1000.0], torch.float32)
    mask = torch.tensor([False, False])
    ok, message = within_ulp(golden, result, max_ulp=0, mask=mask)
    assert ok
    assert "no measurable lane" in message


def test_a_shape_mismatch_is_reported_rather_than_raised():
    ok, message = within_ulp(
        torch.zeros(4, dtype=torch.float32),
        torch.zeros(5, dtype=torch.float32),
        max_ulp=1,
    )
    assert not ok
    assert "shape mismatch" in message


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def test_ulp_stats_keeps_unmeasurable_lanes_out_of_the_aggregates():
    stats = ulp_stats(_t([0, 2, UNMEASURABLE, 4, UNMEASURABLE], torch.int64))
    assert stats["lanes"] == 3
    assert stats["unmeasurable"] == 2
    assert stats["max"] == 4
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["exact_frac"] == pytest.approx(1 / 3)
    assert stats["worst_index"] == 3


def test_the_worst_index_is_a_flat_index_into_the_input():
    golden = torch.ones(3, 4, dtype=torch.bfloat16)
    result = golden.clone()
    result[2, 1] = _step_up(1.0, torch.bfloat16, steps=5)[0]
    stats = ulp_stats(ulp_distance(golden, result))
    assert stats["max"] == 5
    assert stats["worst_index"] == 2 * 4 + 1


def test_the_quantiles_fall_back_to_the_max_below_their_thresholds():
    """Borrowed from ttnn: a quantile over 5 elements is not a percentile."""
    stats = ulp_stats(_t([0, 0, 0, 9, 0], torch.int64))
    assert stats["p95"] == stats["p99"] == pytest.approx(9.0)

    wide = torch.zeros(200, dtype=torch.int64)
    wide[-1] = 9
    stats = ulp_stats(wide)
    assert stats["max"] == 9
    assert stats["p95"] == pytest.approx(0.0)
    assert stats["p99"] < 9.0

    # 5 lanes is below both thresholds and 200 above both, so neither tells the two
    # constants apart. 50 sits between them: a real p95, and p99 still falling back.
    assert _MIN_LANES_FOR_P95 < 50 < _MIN_LANES_FOR_P99
    middle = torch.zeros(50, dtype=torch.int64)
    middle[-1] = 9
    stats = ulp_stats(middle)
    assert stats["p95"] == pytest.approx(0.0)
    assert stats["p99"] == pytest.approx(9.0)


def test_ulp_stats_on_an_all_unmeasurable_tensor():
    stats = ulp_stats(torch.full((4,), UNMEASURABLE, dtype=torch.int64))
    assert stats == {**stats, "lanes": 0, "unmeasurable": 4, "worst_index": None}


def test_the_failure_message_locates_the_lane_and_sizes_its_step():
    """A ULP verdict is only useful if it names the point: which lane, what the hardware
    produced there, and what one step is worth at that value."""
    golden = torch.ones(8, dtype=torch.bfloat16)
    result = golden.clone()
    result[5] = _step_up(1.0, torch.bfloat16, steps=7)[0]
    message = ulp_failure_message(
        golden, result, ulp_distance(golden, result), DataFormat.Float16_b, max_ulp=3
    )
    assert "max 7 ULP @ [5]" in message
    assert repr(float(result[5])) in message
    # The step is the *upward* gap at 1.0, since the result moved up. Against the constant
    # rather than against local_step(), which is the function the message already called.
    assert f"{ABOVE_ONE:.6e}" in message


def test_the_message_builder_can_reuse_stats_it_was_given():
    """The verdict already computed them; building the message must not pay again."""
    golden = torch.ones(8, dtype=torch.bfloat16)
    result = golden.clone()
    result[3] = _step_up(1.0, torch.bfloat16, steps=4)[0]
    distance = ulp_distance(golden, result)
    assert ulp_failure_message(
        golden, result, distance, DataFormat.Float16_b, stats=ulp_stats(distance)
    ) == ulp_failure_message(golden, result, distance, DataFormat.Float16_b)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_local_step_agrees_with_the_sweep_metric_in_the_normal_range(dtype):
    fmt = {
        torch.bfloat16: DataFormat.Float16_b,
        torch.float16: DataFormat.Float16,
        torch.float32: DataFormat.Float32,
    }[dtype]
    for value in (1.0, -1.0, 3.75, 1024.0):
        expected = float(
            local_ulp(_t([value], dtype).to(torch.float64).numpy(), fmt)[0]
        )
        assert local_step(value, dtype) == pytest.approx(expected)


def test_local_step_of_a_nonfinite_is_not_a_number():
    assert math.isnan(local_step(float("inf"), torch.float32))
    assert math.isnan(local_step(float("nan"), torch.float32))


# ─────────────────────────────────────────────────────────────────────────────
# The reported step has to agree with the counted one
# ─────────────────────────────────────────────────────────────────────────────


def test_the_reported_step_takes_the_direction_the_result_moved():
    """The gap below a power of two is half the gap above it, so the message has to report
    the one the counted step crossed. Direction is the *signed* delta: a magnitude
    comparison calls ``1.0 -> -2.0`` upward, but the first step out of ``1.0`` is down.
    """
    for toward, expected in (
        (None, ABOVE_ONE),
        (2.0, ABOVE_ONE),
        (0.5, BELOW_ONE),
        (0.0, BELOW_ONE),
        (-0.5, BELOW_ONE),  # straddling zero: judged on where the path starts
        (-2.0, BELOW_ONE),
    ):
        assert local_step(1.0, torch.bfloat16, toward=toward) == pytest.approx(
            expected
        ), toward
    # A negative value mirrors it: "toward zero" means increasing, not decreasing.
    assert local_step(-1.0, torch.bfloat16, toward=2.0) == pytest.approx(BELOW_ONE)
    assert local_step(-1.0, torch.bfloat16, toward=-2.0) == pytest.approx(ABOVE_ONE)


def test_the_verdict_prints_the_gap_the_result_crossed():
    """The same asymmetry, end to end through the message the gate logs."""
    golden = _t([1.0], torch.bfloat16)
    fmt = DataFormat.Float16_b
    _, down = within_ulp(golden, _step_down(1.0, torch.bfloat16), max_ulp=0, fmt=fmt)
    assert f"{BELOW_ONE:.6e}" in down and f"{ABOVE_ONE:.6e}" not in down
    up_result = _step_up(1.0, torch.bfloat16, steps=3)
    _, up = within_ulp(golden, up_result, max_ulp=0, fmt=fmt)
    assert f"{ABOVE_ONE:.6e}" in up


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_the_reported_step_across_a_flushed_band_is_the_jump_it_makes(dtype):
    """With the band collapsed, zero and every subnormal share rank 0 and the smallest
    normal is rank 1, so one counted step is ``tiny`` in either direction. The raw
    ``nextafter`` gap there is the smallest *subnormal*, which understates it by
    ``2**mantissa_bits`` and contradicts the count the same message prints."""
    tiny = float(torch.finfo(dtype).tiny)
    raw_gap = tiny * 2.0 ** -MANTISSA_BITS[dtype]

    assert int(ulp_distance(_t([tiny], dtype), _t([0.0], dtype))[0]) == 1
    assert local_step(0.0, dtype) == pytest.approx(tiny, abs=0)
    assert local_step(tiny / 4, dtype) == pytest.approx(tiny, abs=0)
    # The boundary the band check used to sit just outside of: at exactly `tiny`, heading
    # down lands on zero and is worth the whole jump.
    assert local_step(tiny, dtype, toward=0.0) == pytest.approx(tiny, abs=0)
    # Upward from the smallest normal is an ordinary normal-range step.
    assert local_step(tiny, dtype, toward=2.0 * tiny) == pytest.approx(raw_gap, abs=0)
    # Unflushed, it is the true gap throughout.
    assert local_step(tiny, dtype, toward=0.0, flush_subnormals=False) == pytest.approx(
        raw_gap, abs=0
    )


def test_the_reported_step_for_fp16_near_zero_is_the_true_gap():
    """fp16 does not flush, so there is nothing to compact."""
    smallest = 2.0**-24
    assert local_step(smallest, torch.float16) == pytest.approx(smallest, abs=0)


def test_the_step_at_zero_has_no_downward_direction():
    """``copysign(1.0, 0.0)`` is ``+1.0``, so a bare signed-delta check called every
    negative *toward* downward at zero -- and ``nextafter(+0.0, 0.0)`` is ``+0.0``, so the
    step came back ``0.0`` and printed ``1 ULP = 0.000000e+00`` on the zero-crossing lane
    (``sin``/``tanh``/``erf`` at 0) it exists to explain. fp16 is the case that reached it;
    the flushing dtypes exit through the band branch instead."""
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
    """Upward from the largest finite is ``Inf``, so the raw difference is infinite and a
    failure there would read "1 ULP = inf". The binade is the same downward.

    Two cases need the ``info.max`` disjunct: no ``toward`` at all, and ``toward=+inf``,
    which ``math.isfinite`` rejects so the signed-delta check cannot fire. ``+max`` against
    ``-max`` is not one of them -- the delta already picks the downward gap there -- and is
    checked only end to end.
    """
    largest = float(torch.finfo(dtype).max)
    expected = largest - float(_step_down(largest, dtype))
    for toward in (None, float("inf"), -largest):
        assert local_step(largest, dtype, toward=toward) == pytest.approx(
            expected
        ), toward

    ok, message = within_ulp(_t([largest], dtype), _t([-largest], dtype), max_ulp=0)
    assert not ok
    assert "1 ULP = inf" not in message


# ─────────────────────────────────────────────────────────────────────────────
# What the metric refuses to measure
# ─────────────────────────────────────────────────────────────────────────────


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
    assert fmt in ULP_FORMATS
    assert ulp_dtype(fmt) == expected


@pytest.mark.parametrize(
    "fmt",
    [
        DataFormat.Bfp8_b,
        DataFormat.Bfp4_b,
        DataFormat.MxFp8P,
        DataFormat.MxFp4,
        DataFormat.Int32,
        DataFormat.Tf32,
    ],
    ids=lambda f: f.name,
)
def test_ulp_dtype_rejects_formats_without_a_per_element_ulp(fmt):
    """Rejected, not silently redirected: a block float's spacing is set by an exponent
    shared across 16 elements, and ``Tf32`` sits in an fp32 container whose lattice is not
    its own."""
    with _refuses("no per-element ULP"):
        ulp_dtype(fmt)


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
    with _refuses(match):
        ulp_distance(golden, result)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32, torch.bool], ids=str)
def test_the_step_and_the_flush_default_refuse_an_unmeasured_dtype(dtype):
    """``.get(dtype, True)`` invented an answer for a dtype ``ulp_distance`` refuses, on
    the error-hiding polarity: ``local_step(1.0, torch.float64)`` handed back ``2**-52``.
    ``local_step`` needs its own guard, since an explicit ``flush_subnormals=`` skips the
    lookup."""
    with _refuses("unsupported dtype"):
        flushes_subnormals(dtype)
    with _refuses("unsupported dtype"):
        local_step(1.0, dtype)
    with _refuses("unsupported dtype"):
        local_step(1.0, dtype, flush_subnormals=True)


@pytest.mark.parametrize(
    "fmt",
    [DataFormat.Bfp8_b, DataFormat.Bfp4_b, DataFormat.MxFp8P, DataFormat.Tf32],
    ids=lambda f: f.name,
)
def test_the_verdict_refuses_a_format_with_no_per_element_ulp(fmt):
    """``fmt`` is not only a label: ``format_dict`` collapses every block float onto
    ``torch.bfloat16``, so a verdict labelled ``Bfp8_b`` would come back measured in
    bfloat16 steps -- the measurement ``ulp_dtype`` exists to refuse."""
    values = torch.ones(4, dtype=torch.bfloat16)
    with _refuses("no per-element ULP"):
        within_ulp(values, values.clone(), max_ulp=1, fmt=fmt)


@pytest.mark.parametrize(
    "fmt, wrong",
    [
        (DataFormat.Float16_b, torch.float32),
        (DataFormat.Float32, torch.bfloat16),
        (DataFormat.Float16, torch.float32),
    ],
    ids=lambda x: getattr(x, "name", str(x)),
)
def test_the_verdict_refuses_a_format_that_disagrees_with_the_dtype(fmt, wrong):
    """Allowlisting the format is not the whole check: two float32 tensors labelled
    ``Float16_b`` would be measured in float32 steps under a Float16_b verdict. The label
    and the lattice have to be the same claim."""
    values = torch.ones(4, dtype=wrong)
    with _refuses("but the tensors are"):
        within_ulp(values, values.clone(), max_ulp=1, fmt=fmt)


@pytest.mark.parametrize("shape", [(1,), (4,), (3, 4)], ids=str)
def test_the_verdict_refuses_a_mask_it_would_otherwise_broadcast(shape):
    """A per-op budget computes the mask dynamically, so a wrong-shaped one has to be a
    named error. ``(1,)`` is the dangerous case: it broadcasts through and silently judges
    every lane or none."""
    golden = torch.ones(2, 6, dtype=torch.bfloat16)
    mask = torch.ones(shape, dtype=torch.bool)
    with _refuses("mask shape"):
        within_ulp(golden, golden.clone(), max_ulp=1, mask=mask)
    with _refuses("mask shape"):
        ulp_stats(ulp_distance(golden, golden.clone()), mask)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_the_threshold_warning_fires_past_one_binade(dtype):
    assert MAX_MEANINGFUL_ULP[dtype] == 1 << MANTISSA_BITS[dtype]
    assert not warn_if_threshold_unmeaningful(MAX_MEANINGFUL_ULP[dtype], dtype)
    assert warn_if_threshold_unmeaningful(MAX_MEANINGFUL_ULP[dtype] + 1, dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Shape, symmetry and layout
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_the_distance_is_symmetric_and_shape_preserving(dtype):
    torch.manual_seed(0)
    golden = torch.randn(3, 5, 7, dtype=torch.float32).to(dtype)
    result = golden.clone()
    flat = result.reshape(-1)
    flat[::3] = torch.nextafter(flat[::3], torch.full_like(flat[::3], float("inf")))

    forward = ulp_distance(golden, result)
    assert forward.shape == golden.shape
    assert forward.dtype == torch.int64
    assert torch.equal(forward, ulp_distance(result, golden))
    assert int(forward.max()) == 1


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_identical_tensors_are_zero_steps_apart(dtype):
    torch.manual_seed(0)
    values = torch.randn(64, dtype=torch.float32).to(dtype)
    assert int(ulp_distance(values, values.clone()).max()) == 0
    assert within_ulp(values, values.clone(), max_ulp=0)[0]


def test_a_non_contiguous_input_is_measured_correctly():
    """A transposed tile is the ordinary case in this harness, so the flatten inside the
    metric has to stay in step with the caller's own flat indexing. Distinct values and a
    known perturbation, not a constant tensor, over which ``max() == 0`` would hold under
    any ordering."""
    golden = torch.arange(24).reshape(4, 6).to(torch.bfloat16).t()
    assert not golden.is_contiguous()

    steps, position = 5, (3, 1)  # a position whose flat index moves when transposed
    result = golden.clone()
    result[position] = _step_up(float(golden[position]), torch.bfloat16, steps=steps)[0]

    stats = ulp_stats(ulp_distance(golden, result))
    assert stats["max"] == steps
    assert stats["worst_index"] == position[0] * golden.shape[1] + position[1]

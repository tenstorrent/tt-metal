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

import numpy as np
import pytest
import torch
from helpers.accuracy_metrics import local_ulp
from helpers.format_config import DataFormat
from helpers.pack import float_to_bfp8_block
from helpers.ulp import (
    _MIN_LANES_FOR_P95,
    _MIN_LANES_FOR_P99,
    _ULP_DTYPES,
    INTEGER_FORMATS,
    MAX_MEANINGFUL_ULP,
    NEAR_ZERO_FRACTION,
    ULP_FORMATS,
    UNMEASURABLE,
    _value_order_index,
    flushes_subnormals,
    has_ulp_gate,
    local_step,
    nonfinite_disagreement_summary,
    nonfinite_mismatches,
    ulp_distance,
    ulp_dtype,
    ulp_elementwise_valid,
    ulp_stats,
    ulp_verdict_message,
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
    """...and says the mask excluded them, not that there was nothing to compare. Both
    lanes here are measurable, and the two cases have different remediations."""
    golden, result = _t([1.0, 1.0], torch.float32), _t([5.0, 1000.0], torch.float32)
    mask = torch.tensor([False, False])
    ok, message = within_ulp(golden, result, max_ulp=0, mask=mask)
    assert ok
    assert (
        "no lane under judgement" in message and "selected none of 2 lanes" in message
    )

    # The unmeasurable case still says what it always said.
    nan = _t([float("nan")] * 2, torch.float32)
    ok, message = within_ulp(nan, nan.clone(), max_ulp=0)
    assert ok and "no measurable lane (2 unmeasurable" in message


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
    assert stats["lanes"] == 0
    assert stats["unmeasurable"] == 4
    assert stats["worst_index"] is None
    assert stats["max"] == 0
    assert all(math.isnan(stats[key]) for key in ("mean", "p95", "p99", "exact_frac"))


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


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_a_saturated_lane_does_not_print_a_nan_step(dtype):
    """``local_step`` has no answer at ``Inf`` and returns NaN; an agreeing overflow is a
    pass, so that NaN would reach the log as "1 ULP = nan"."""
    saturated = _t([float("inf")], dtype)
    ok, message = within_ulp(saturated, saturated.clone(), max_ulp=0)
    assert ok
    assert "nan" not in message


# ─────────────────────────────────────────────────────────────────────────────
# What the metric refuses to measure
# ─────────────────────────────────────────────────────────────────────────────


def test_within_ulp_refuses_a_golden_and_result_of_different_dtypes():
    """A rank is a position on one lattice. Uncast, an overflowed lane reads as a kernel
    overflow on the non-finite path instead, and is labelled with the golden's dtype."""
    with _refuses("golden is torch.float32 but result is"):
        within_ulp(
            _t([70000.0], torch.float32), _t([float("inf")], torch.float16), max_ulp=0
        )


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
        DataFormat.Bfp4_b,
        DataFormat.Bfp2_b,
        DataFormat.MxFp8P,
        DataFormat.MxFp4,
        DataFormat.Int32,
        DataFormat.Tf32,
    ],
    ids=lambda f: f.name,
)
def test_ulp_dtype_rejects_formats_without_a_per_element_ulp(fmt):
    """Rejected, not silently redirected: Bfp4_b leaves 2 fractional bits and Bfp2_b 0,
    against bfloat16's 7, so a bf16 step count would read every legal quantization as a
    32- or 128-step error. ``Tf32`` sits in an fp32 container whose lattice is not its
    own. ``Bfp8_b`` is the one exception -- see the proxy test below."""
    with _refuses("no per-element ULP"):
        ulp_dtype(fmt)


def test_bfp8_b_is_measured_in_bf16_proxy_space():
    """Close enough to bfloat16 to be gated in its step space -- ttnn makes the same
    choice, and ``passed_test`` has already cast the tensor -- but not equal to it: its 7
    magnitude bits *include* an explicit leading 1, so it has 6 fractional bits against
    bfloat16's 7 and one Bfp8_b step is two bf16 steps. A budget denominated in bf16 steps
    buys half as many format steps, which is what an enrolling caller has to know."""
    assert DataFormat.Bfp8_b not in ULP_FORMATS
    assert ulp_dtype(DataFormat.Bfp8_b) == torch.bfloat16

    # Two bf16 steps up from 1.0 is the first to change the encoded Bfp8_b mantissa.
    block = [1.0] * 16
    _, baseline = float_to_bfp8_block(block)
    encoded = []
    for steps in range(3):
        block[0] = float(_step_up(1.0, torch.bfloat16, steps=steps)[0])
        encoded.append(float_to_bfp8_block(block)[1][0])
    assert encoded[0] == encoded[1] == baseline[0]
    assert encoded[2] == baseline[0] + 1


def test_the_verdict_refuses_a_non_boolean_mask():
    """The selection is combined with ``&``. An integer mask makes that bitwise, where a
    truthy ``2`` becomes ``2 & 1 == 0`` and drops the lane it was meant to select."""
    golden = torch.ones(2, 6, dtype=torch.bfloat16)
    mask = torch.full((2, 6), 2, dtype=torch.int64)
    with _refuses("mask must be bool"):
        within_ulp(golden, golden.clone(), max_ulp=1, mask=mask)
    with _refuses("mask must be bool"):
        ulp_stats(ulp_distance(golden, golden.clone()), mask)


def test_the_verdict_accepts_bfp8_b_in_its_proxy_space():
    values = torch.ones(4, dtype=torch.bfloat16)
    ok, message = within_ulp(values, values.clone(), max_ulp=0, fmt=DataFormat.Bfp8_b)
    assert ok and "Bfp8_b" in message


def test_the_sweep_metric_covers_the_proxy_formats_the_gate_judges():
    """``local_ulp`` probed a private copy of the native format tuple, so the sweep wrote
    NaN ``signed_ulp_error`` for exactly the format the gate can judge."""
    values = np.array([1.0, 2.0])
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
    with _refuses(match):
        ulp_distance(golden, result)


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


# ─────────────────────────────────────────────────────────────────────────────
# The elementwise verdict the gate consumes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_elementwise_valid_marks_only_the_out_of_budget_lanes(dtype):
    golden = torch.ones(4, dtype=dtype)
    result = torch.cat(
        [
            _t([1.0], dtype),
            _step_up(1.0, dtype, steps=1),
            _step_up(1.0, dtype, steps=2),
            _step_down(1.0, dtype, steps=3),
        ]
    )
    is_valid, distance, _ = ulp_elementwise_valid(golden, result, 1)
    assert distance.tolist() == [0, 1, 2, 3]
    assert is_valid.tolist() == [True, True, False, False]


def test_elementwise_valid_accepts_both_nan_and_rejects_a_missing_one():
    nan = float("nan")
    golden, result = _t([nan, nan, 1.0], torch.float32), _t(
        [nan, 1.0, nan], torch.float32
    )
    assert ulp_elementwise_valid(golden, result, 0)[0].tolist() == [True, False, False]


def test_elementwise_valid_rejects_an_overflow_even_though_it_ranks_one_step():
    """The gate is stricter than the metric here, deliberately."""
    golden = _t([torch.finfo(torch.float32).max], torch.float32)
    result = _t([float("inf")], torch.float32)
    is_valid, distance, _ = ulp_elementwise_valid(golden, result, 1)
    assert int(distance[0]) == 1
    assert is_valid.tolist() == [False]


def test_elementwise_valid_without_a_floor_is_the_plain_budget():
    torch.manual_seed(0)
    golden = torch.randn(128, dtype=torch.float32)
    result = torch.nextafter(golden, torch.full_like(golden, float("inf")))
    assert bool(ulp_elementwise_valid(golden, result, 1)[0].all())
    assert not bool(ulp_elementwise_valid(golden, result, 0)[0].any())


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_the_gate_resolves_the_flush_default_per_dtype_like_the_metric(dtype):
    """``ulp_elementwise_valid`` hardcoded ``flush_subnormals=True``, overriding the
    per-dtype default for every caller including ``passed_test``. For fp16, which keeps
    its subnormals here, that collapsed up to 1023 steps of error onto 0."""
    smallest = float(torch.finfo(dtype).tiny) * 2.0 ** -MANTISSA_BITS[dtype]
    band_steps = (1 << MANTISSA_BITS[dtype]) - 1
    golden, result = _t([smallest], dtype), _t([band_steps * smallest], dtype)

    is_valid, distance, _ = ulp_elementwise_valid(golden, result, 0)
    if dtype is torch.float16:
        assert int(distance[0]) == band_steps - 1 and not bool(is_valid[0])
    else:  # bf16 and fp32 flush the band, so both sides really are zero here
        assert int(distance[0]) == 0 and bool(is_valid[0])
    # The blanket behaviour stays reachable for a caller that knows the Dest flushed.
    assert bool(ulp_elementwise_valid(golden, result, 0, flush_subnormals=True)[0][0])


# ─────────────────────────────────────────────────────────────────────────────
# The near-zero floor
# ─────────────────────────────────────────────────────────────────────────────


def test_the_floor_rescues_a_cancellation_lane():
    """What the floor is for: a golden that crosses zero, where ``ulp(golden)`` collapses
    and a tiny absolute error becomes an enormous step count."""
    golden, result = _t([100.0, 1e-8], torch.float32), _t([100.0, 2e-8], torch.float32)
    is_valid, distance, _ = ulp_elementwise_valid(golden, result, 4)
    assert int(distance[1]) > 1000  # meaningless as a kernel verdict
    assert is_valid.tolist() == [True, False]

    is_valid, _, rescued = ulp_elementwise_valid(golden, result, 4, near_zero_atol=1e-7)
    assert is_valid.tolist() == [True, True]
    assert rescued.tolist() == [False, True]


def test_the_floor_does_not_loosen_the_large_magnitude_lanes():
    """An atol applied at every magnitude is the magnitude-blind gate a step count
    replaces, so the floor stays below the near-zero cut."""
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


def test_no_lane_is_reported_as_rescued_when_it_was_in_budget_anyway():
    golden = _t([100.0, 1e-8], torch.float32)
    _, _, rescued = ulp_elementwise_valid(
        golden, golden.clone(), 4, near_zero_atol=1e-7
    )
    assert not bool(rescued.any())


def test_the_near_zero_band_is_bounded_absolutely_as_well_as_relatively():
    """The relative band alone is unbounded in absolute terms, so one large golden widens
    it across the tile: over a golden spanning [0, 1000] the cut lands at 10.0 and a lane
    at 1.0, seven steps out, was rescued against a 1-step budget. The absolute cut is the
    magnitude at which the forgiven error is exactly ``near_zero_fraction`` of the
    reference."""
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
    # Tightening the atol is not a substitute: the step count it admits grows as
    # 1/|golden|, so any atol loose enough for the cancellation lane is loose enough for
    # the mid-band one under the relative rule alone.
    assert not ulp_elementwise_valid(golden, result, 1, near_zero_atol=1e-9)[0][2]


def test_the_near_zero_band_is_scoped_to_the_lanes_under_judgement():
    """``dynamic_range`` is the only verdict input that is not elementwise, so an unscoped
    max let a large golden in a lane the caller masked *out* widen the band applied to the
    lanes it masked *in* -- the first place an excluded lane could change a judged
    verdict."""
    golden, result = _t([1e6, 1.0], torch.float32), _t([1e6, 1.02], torch.float32)
    assert int(ulp_distance(golden, result)[1]) > 100000

    ok, _ = within_ulp(
        golden, result, max_ulp=0, near_zero_atol=0.05, mask=torch.tensor([False, True])
    )
    assert not ok, "the excluded lane must not widen the band for the judged one"
    # Unmasked, the same tensors legitimately reach the wide band: the fix is the scoping,
    # not a change to the rule.
    is_valid, _, rescued = ulp_elementwise_valid(golden, result, 0, near_zero_atol=0.05)
    assert bool(is_valid.all()) and bool(rescued[1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_the_relative_cut_is_compared_in_float32_not_the_tensor_dtype(dtype):
    """Both cuts are Python floats, so a 16-bit ``golden.abs()`` promoted them onto the
    tensor's lattice and rounded each edge -- narrowing the relative ``<`` one. Probing at
    +/-10% cannot see it; the lane has to sit on the edge, and the cut has to round
    *down*, or nothing separates the two compares. 13.0 rounds down for bf16 but up for
    fp16, so that parameter was green either way; 11.0 rounds down in both."""
    dynamic_range = 11.0
    cut = NEAR_ZERO_FRACTION * dynamic_range
    rounded = float(torch.tensor(cut, dtype=dtype))
    assert rounded < cut, "the cut must round down, or the lane cannot separate the two"

    golden = _t([dynamic_range, (cut + rounded) / 2.0], dtype)
    assert float(golden[1]) == rounded, "the lane has to sit on the rounded edge"
    result = golden.clone()
    result[1] = float(_step_up(float(golden[1]), dtype, steps=4)[0])

    narrow = within_ulp(golden, result, max_ulp=0, near_zero_atol=1.0)[0]
    wide = within_ulp(
        golden.to(torch.float32),
        result.to(torch.float32),
        max_ulp=0,
        near_zero_atol=1.0,
    )[0]
    assert narrow == wide, f"{dtype} disagreed with float32 on the rounded band edge"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_the_absolute_cut_is_compared_in_float32_not_the_tensor_dtype(dtype):
    """The other half, which the relative case cannot reach: the absolute edge is ``<=``,
    so rounding *widens* the band -- the opposite direction. The test above uses
    ``near_zero_atol=1.0``, whose cut is exactly 100.0 and representable in both, so that
    edge went unprobed. 0.012 puts the cut at 1.2, which both dtypes round up."""
    near_zero_atol = 0.012
    absolute_cut = near_zero_atol / NEAR_ZERO_FRACTION
    rounded = float(torch.tensor(absolute_cut, dtype=dtype))
    assert rounded > absolute_cut, "the cut must round up, or the <= edge does not move"

    # Far below the relative cut (1% of 1000.0), so the absolute edge alone decides.
    golden = _t([1000.0, rounded], dtype)
    result = golden.clone()
    result[1] = float(_step_up(rounded, dtype, steps=1)[0])
    assert int(ulp_distance(golden, result)[1]) == 1  # one step out of budget...
    assert float(result[1]) - rounded <= near_zero_atol  # ...and well inside the floor

    narrow = within_ulp(golden, result, max_ulp=0, near_zero_atol=near_zero_atol)[0]
    wide = within_ulp(
        golden.to(torch.float32),
        result.to(torch.float32),
        max_ulp=0,
        near_zero_atol=near_zero_atol,
    )[0]
    assert narrow == wide, f"{dtype} disagreed with float32 on the rounded absolute cut"
    assert not wide  # outside the band on both sides, so the step fails


def test_the_verdict_reports_what_the_floor_carried():
    """A floor-carried pass and a budget-carried one are not the same result, and the
    rescued lanes are exactly the ones the summary keeps out of its ranking."""
    golden = _t([100.0, 1e-4, 2e-4], torch.float32)
    result = _t([100.0, 1.1e-4, 2.1e-4], torch.float32)

    ok, message = within_ulp(golden, result, max_ulp=0, fmt=DataFormat.Float32)
    assert not ok and "held by the floor" not in message

    ok, message = within_ulp(
        golden, result, max_ulp=0, fmt=DataFormat.Float32, near_zero_atol=1e-4
    )
    assert ok and "2 held by the floor" in message
    # ...and the ranking excludes them, so the summary describes the lane that was
    # judged rather than the ones the floor carried, which hold the largest counts.
    assert "max 0 ULP" in message and "over 1 lanes" in message


def test_the_verdict_reproduces_every_gate_verdict_including_the_floor():
    """Verdict parity, not just message parity: without the ``near_zero_atol``
    passthrough there were gate verdicts ``within_ulp`` could not reach at any
    argument."""
    golden = torch.linspace(1.0, 100.0, 64, dtype=torch.float32)
    golden[-1] = 1e-8
    result = golden.clone()
    result[-1] = 2e-8

    assert not within_ulp(golden, result, max_ulp=2, fmt=DataFormat.Float32)[0]
    assert within_ulp(
        golden, result, max_ulp=2, fmt=DataFormat.Float32, near_zero_atol=1e-7
    )[0]
    is_valid, _, rescued = ulp_elementwise_valid(golden, result, 2, near_zero_atol=1e-7)
    assert bool(is_valid.all()) and bool(rescued[-1])


# ─────────────────────────────────────────────────────────────────────────────
# What a verdict says when the step count cannot describe the failure
# ─────────────────────────────────────────────────────────────────────────────


def test_a_missing_nan_is_named_rather_than_reported_as_zero_steps():
    """A NaN lane is UNMEASURABLE and drops out of the statistics, so a verdict that
    failed only on a missing NaN reported "max 0 ULP (budget 0)" -- true, and useless.
    """
    golden = _t([1.0, float("nan"), 2.0], torch.float32)
    result = _t([1.0, 1.0, 2.0], torch.float32)
    distance = ulp_distance(golden, result)
    assert ulp_stats(distance)["max"] == 0  # the statistics genuinely see nothing

    summary = nonfinite_disagreement_summary(golden, result, DataFormat.Float32)
    assert "non-finite disagreement @ [1]" in summary and "1 such lane(s)" in summary

    message = ulp_verdict_message(
        golden, result, distance, DataFormat.Float32, max_ulp=0
    )
    assert message.startswith("non-finite disagreement @ [1]")
    assert "max 0 ULP" in message  # the step summary is still there, as detail


def test_a_failure_at_the_top_of_the_range_reports_a_usable_step():
    """Both at once: the overflow names itself, and the step it is measured against is a
    number rather than infinity."""
    largest = float(torch.finfo(torch.bfloat16).max)
    golden, result = _t([largest], torch.bfloat16), _t([float("inf")], torch.bfloat16)
    ok, message = within_ulp(golden, result, max_ulp=4, fmt=DataFormat.Float16_b)
    assert not ok
    assert message.startswith("non-finite disagreement")
    assert "1 ULP = inf" not in message


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_the_sweep_metric_matches_local_step_at_the_top_of_the_range(dtype):
    """``test_local_step_agrees_with_the_sweep_metric`` never probed ``finfo.max``, so the
    ``nextafter`` saturation there went uncaught in both functions."""
    fmt = {
        torch.bfloat16: DataFormat.Float16_b,
        torch.float16: DataFormat.Float16,
        torch.float32: DataFormat.Float32,
    }[dtype]
    largest = float(torch.finfo(dtype).max)
    swept = float(local_ulp(np.array([largest]), fmt)[0])
    assert math.isfinite(swept)
    assert swept == pytest.approx(local_step(largest, dtype))


# ── Integers are not ULP territory ──────────────────────────────────────────
#
# A step count says "how many representable values apart". For an integer format that is
# always the arithmetic difference, the values are exact, and the only sensible verdict is
# bit equality -- so ULP is meaningless there, not merely weaker, and every entry point
# must refuse rather than compute something plausible.

#: The torch dtypes an integer format lands in, including the containers the float bit
#: arithmetic borrows: ``torch.int16`` is how a bfloat16's bits are read.
TORCH_INT_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
)


@pytest.mark.parametrize("fmt", INTEGER_FORMATS, ids=lambda f: f.name)
def test_no_integer_format_is_ulp_gateable(fmt):
    assert not has_ulp_gate(fmt) and fmt not in ULP_FORMATS
    with _refuses("no per-element ULP"):
        ulp_dtype(fmt)


@pytest.mark.parametrize("dtype", TORCH_INT_DTYPES, ids=str)
def test_the_metric_refuses_every_integer_tensor_dtype(dtype):
    """Including the bit containers: reading a bfloat16 through ``torch.int16`` must not
    make an int16 *tensor* measurable."""
    values = torch.ones(4, dtype=dtype)
    with _refuses("unsupported dtype"):
        ulp_distance(values, values.clone())
    assert dtype not in _ULP_DTYPES  # keyed on the float dtypes only


def test_the_integer_format_list_comes_from_the_enum_not_from_format_dict():
    """``format_dict`` omits ``Bfp8`` and both ``MxFp4_2x`` variants and gives the
    ``MxInt*`` formats a bfloat16 proxy, so deriving the integer set through it would
    silently miss a format added without an entry, or given a float proxy.

    Pinned as the explicit six rather than against ``is_integer()``, which is
    ``INTEGER_FORMATS``' own defining expression and so cannot fail. The gap that
    motivates it is asserted directly: every integer format is in ``format_dict`` today,
    so a ``format_dict``-derived list would yield the same set and stay green too.
    """
    from helpers.llk_params import format_dict

    assert set(INTEGER_FORMATS) == {
        DataFormat.Int32,
        DataFormat.Int16,
        DataFormat.Int8,
        DataFormat.UInt32,
        DataFormat.UInt16,
        DataFormat.UInt8,
    }
    missing = {f for f in DataFormat if f not in format_dict}
    assert {DataFormat.Bfp8, DataFormat.MxFp4_2x_A, DataFormat.MxFp4_2x_B} <= missing
    for fmt in (DataFormat.MxInt8, DataFormat.MxInt4, DataFormat.MxInt2):
        assert format_dict[fmt] is torch.bfloat16  # a float proxy, not an integer one

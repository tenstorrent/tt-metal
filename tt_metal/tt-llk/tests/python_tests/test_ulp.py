# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the integer ULP metric in ``helpers/ulp.py``.

No kernel, no device: the metric is pure bit arithmetic, and it is the thing a future
accuracy budget will be compared against, so every property the gate relies on is pinned
here rather than discovered on hardware.

The tests are grouped by the property they defend, because each one exists to stop a
specific way the metric could look right and be wrong:

* *unit spacing* — one representable step must read as exactly 1 everywhere, including
  across a power of two. ``local_ulp`` divides by the *upward* gap, so the fractional
  ``|err| / ulp(golden)`` form reads one step below a boundary as 0.5 and one above it as
  1.0: a "1 ULP" fractional budget admits two representable steps below a power of two and
  only one above. That contrast is asserted directly.
* *the zero neighbourhood* — under the SFPU's DAZ+FTZ the subnormal band is not there, so
  the smallest normal is one step from zero and both signed zeros are the same value. Get
  the compaction wrong in one half and every sign-crossing distance is inflated by
  ``2**mantissa_bits`` — the missing compaction plus the base being one lower — while
  same-sign distances stay correct, which is exactly the shape of bug that survives a
  casual read.
* *the non-finites* — ``Inf`` has a rank and ``NaN`` does not. The sentinel for "no rank"
  is negative, so a caller comparing it against a budget would pass; the ``within_ulp``
  tests exist to prove the composite verdict does not.
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
    ulp_failure_message,
    ulp_stats,
    ulp_verdict_message,
    warn_if_threshold_unmeaningful,
    within_ulp,
)

FLOAT_DTYPES = [torch.bfloat16, torch.float16, torch.float32]

# The two bf16 gaps either side of 1.0. Bound once because the whole point of the
# direction handling is which of the two gets reported, and a -8/-7 transposition in any
# of the dozen assertions that use them asserts the wrong direction instead of failing.
BELOW_ONE = 2.0**-8  # 1.0 down to the previous bf16 value
ABOVE_ONE = 2.0**-7  # 1.0 up to the next one
assert float(torch.tensor(1.0, dtype=torch.bfloat16)) - BELOW_ONE == float(
    torch.nextafter(
        torch.tensor(1.0, dtype=torch.bfloat16), torch.tensor(0.0, dtype=torch.bfloat16)
    )
)

TORCH_INT_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
)

# INTEGER_FORMATS comes from helpers.ulp, which derives it from DataFormat.is_integer()
# rather than from format_dict: that mapping omits Bfp8 and both MxFp4_2x variants and
# gives MxInt8/MxInt4/MxInt2 a bfloat16 proxy, so deriving through it would silently miss
# an integer format. ULP is meaningless for these anyway -- the values are exact, adjacent
# integers are one apart by definition, and "correct" is bit equality.

# Mantissa bits after the implicit leading 1, i.e. what sets the size of the subnormal
# band that the flush has to compact away.
MANTISSA_BITS = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23}


def _t(values, dtype):
    return torch.tensor(values, dtype=dtype)


def _step_up(value, dtype, *, steps=1):
    """*value* moved *steps* representable values toward +inf, in *dtype*.

    *steps* is keyword-only: a bare ``_step_up(1.0, torch.bfloat16, steps=7)`` reads as a third
    coordinate rather than as a count, and every assertion below is written against the
    count."""
    out = _t([value], dtype)
    for _ in range(steps):
        out = torch.nextafter(out, _t([float("inf")], dtype))
    return out


def _step_down(value, dtype, *, steps=1):
    out = _t([value], dtype)
    for _ in range(steps):
        out = torch.nextafter(out, _t([float("-inf")], dtype))
    return out


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
def test_power_of_two_boundary_is_one_step_in_both_directions(dtype, boundary):
    """The whole reason the gate counts steps instead of dividing by ``ulp(golden)``."""
    golden = _t([boundary], dtype)
    assert int(ulp_distance(golden, _step_down(boundary, dtype))[0]) == 1
    assert int(ulp_distance(golden, _step_up(boundary, dtype))[0]) == 1


def test_fractional_metric_disagrees_at_a_power_of_two():
    """Pins the motivation: one step reads 0.5 below the boundary and 1.0 above it, so a
    fractional "1 ULP" budget admits two steps on one side and one on the other."""
    below = _step_down(1.0, torch.bfloat16)
    above = _step_up(1.0, torch.bfloat16)
    one = _t([1.0], torch.bfloat16)

    step_at_one = local_ulp(one.to(torch.float64).numpy(), DataFormat.Float16_b)[0]
    fractional_below = abs(float(below) - 1.0) / step_at_one
    fractional_above = abs(float(above) - 1.0) / step_at_one

    assert fractional_below == pytest.approx(0.5)
    assert fractional_above == pytest.approx(1.0)
    # ...and measured the other way round, from the golden that sits below the boundary:
    step_below = local_ulp(below.to(torch.float64).numpy(), DataFormat.Float16_b)[0]
    assert abs(1.0 - float(below)) / step_below == pytest.approx(1.0)
    assert abs(float(above) - float(below)) / step_below == pytest.approx(3.0)

    # The integer metric calls all of these what they are.
    assert int(ulp_distance(one, below)[0]) == 1
    assert int(ulp_distance(one, above)[0]) == 1
    assert int(ulp_distance(below, above)[0]) == 2


def test_bf16_value_order_is_unit_spaced_over_the_whole_format():
    """Exhaustive: every adjacent pair of distinct flushed bf16 values is 1 step apart.

    Cheap at 2**16 patterns, and it catches an ordering defect anywhere in the range
    rather than at the handful of points a parametrized test happens to name. This is an
    ordering check, not an error aggregate, so it is not subject to the usual caution
    about uniform bf16 sweeps.
    """
    patterns = torch.arange(-32768, 32768, dtype=torch.int16).view(torch.bfloat16)
    finite = patterns[torch.isfinite(patterns)].to(torch.float64)
    # Collapse the subnormals the way the metric does, then take the distinct values in
    # increasing order.
    tiny = torch.finfo(torch.bfloat16).tiny
    flushed = torch.where(finite.abs() < tiny, torch.zeros_like(finite), finite)
    ordered = torch.unique(flushed).to(torch.bfloat16)

    distances = ulp_distance(ordered[:-1], ordered[1:])
    assert int(distances.min()) == 1
    assert int(distances.max()) == 1
    # The per-step bound alone does not force the walk to be monotonic: `ulp_distance`
    # takes `.abs()`, so a rank sequence that oscillates (..., 0, 1, 0, 1, ...) satisfies
    # it. Measuring the two endpoints against each other forces every one of those steps
    # to be +1 -- the span has to equal the number of values in it.
    span = int(ulp_distance(ordered[:1], ordered[-1:])[0])
    assert span == ordered.numel() - 1


def test_the_value_order_agrees_with_the_sweep_enumerators_key():
    """The stimuli and the metric have to mean the same thing by "one representable step".

    ``stimuli_generator.strategies.structured`` walks the float32 line by a twos-complement
    key (``bits`` for the positive half, ``INT_MIN - bits`` for the negative), which is
    what ``UlpSweepStrategy`` and ``ulp_sweep_value_count`` are built on. Unflushed, this
    module's sign-and-magnitude rank is that same total order, including the ``+0``/``-0``
    collapse -- so a sweep that says it stepped N values and a metric that measures N steps
    are making one claim, not two. Claimed for the unflushed index only: the compaction is
    this module's own, and the sweep enumerates the subnormal band.
    """
    from helpers.stimuli_generator.strategies.structured import _enumerate_fp32_in_range

    INT_MIN = -(2**31)
    probes = torch.tensor(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0**-149,  # smallest subnormal, the band the flush would compact
            -(2.0**-149),
            float(torch.finfo(torch.float32).tiny),
            -float(torch.finfo(torch.float32).tiny),
            float(torch.finfo(torch.float32).max),
            -float(torch.finfo(torch.float32).max),
            float("inf"),
            float("-inf"),
            3.14159265,
            -1e-30,
        ],
        dtype=torch.float32,
    )
    bits = probes.view(torch.int32).to(torch.int64)
    sweep_key = torch.where(bits < 0, INT_MIN - bits, bits)
    spec = _ULP_DTYPES[torch.float32]
    ours = _value_order_index(probes, spec, flush_subnormals=False)
    assert ours.tolist() == sweep_key.tolist()

    # ...and end to end: the sweep's own enumeration is unit-spaced under this metric, so
    # "the Nth value in the sweep" and "N steps away" cannot drift apart.
    run = _enumerate_fp32_in_range(1.0, 2.0, 64)
    assert run.numel() == 64
    assert ulp_distance(run[:-1], run[1:], flush_subnormals=False).tolist() == [1] * 63
    assert int(ulp_distance(run[:1], run[-1:], flush_subnormals=False)[0]) == 63


# ─────────────────────────────────────────────────────────────────────────────
# The zero neighbourhood: signed zeros, subnormals, the flush
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("flush", [True, False])
def test_signed_zeros_coincide(dtype, flush):
    """``-0.0`` dies on unpack and the pack path canonicalises it again, so the one step a
    raw bit ordering would report is an encoding artefact. True with or without the flush:
    the index is a signed rank of the magnitude, and both zeros rank 0."""
    zeros = _t([0.0, -0.0, 0.0, -0.0], dtype)
    other = _t([-0.0, 0.0, 0.0, -0.0], dtype)
    assert ulp_distance(zeros, other, flush_subnormals=flush).tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_flush_makes_smallest_normal_one_step_from_zero(dtype):
    """The crossing the ported ttnn index gets wrong in its negative half.

    Under DAZ+FTZ the smallest normal is the first value away from zero, on both sides,
    and the two smallest normals are 2 steps apart across zero. A negative half based one
    subnormal-band too far out keeps the same-sign distances correct and inflates every
    one of these, which is why it is worth asserting in both directions.
    """
    tiny = torch.finfo(dtype).tiny
    zero = _t([0.0], dtype)
    flush = {"flush_subnormals": True}  # fp16 does not flush by default; see below
    assert int(ulp_distance(zero, _t([tiny], dtype), **flush)[0]) == 1
    assert int(ulp_distance(zero, _t([-tiny], dtype), **flush)[0]) == 1
    assert int(ulp_distance(_t([-tiny], dtype), _t([tiny], dtype), **flush)[0]) == 2
    assert int(ulp_distance(_t([-tiny], dtype), _step_up(tiny, dtype), **flush)[0]) == 3


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_every_subnormal_is_zero_steps_from_zero_when_flushed(dtype):
    """A flushed hardware result next to a subnormal golden is a 0-step agreement in the
    SFPU's own number system, not a ``2**mantissa_bits - 1`` error.

    Asked for explicitly, because fp16 keeps its subnormals in this harness and so does
    not collapse them by default."""
    tiny = torch.finfo(dtype).tiny
    subnormals = _t(
        [tiny / 2, tiny / 4, -tiny / 2, torch.finfo(dtype).smallest_normal / 8], dtype
    )
    assert torch.all(subnormals.abs() < tiny)  # still subnormal after the cast
    assert torch.all(subnormals != 0)
    zeros = torch.zeros_like(subnormals)
    assert ulp_distance(zeros, subnormals, flush_subnormals=True).tolist() == [
        0,
        0,
        0,
        0,
    ]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_without_flush_the_subnormal_band_is_counted(dtype):
    """``flush_subnormals=False`` is the IEEE ordering, kept for anything that is not the
    SFPU: there the band is real and zero is ``2**mantissa_bits`` steps below the smallest
    normal."""
    tiny = torch.finfo(dtype).tiny
    zero = _t([0.0], dtype)
    expected = 1 << MANTISSA_BITS[dtype]
    assert (
        int(ulp_distance(zero, _t([tiny], dtype), flush_subnormals=False)[0])
        == expected
    )
    assert int(ulp_distance(zero, _t([tiny / 2], dtype), flush_subnormals=False)[0]) > 0


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
    golden = _t([1.0, float("nan"), 2.0, float("nan")], dtype)
    result = _t([1.0, 3.0, float("nan"), float("nan")], dtype)
    assert ulp_distance(golden, result).tolist() == [
        0,
        UNMEASURABLE,
        UNMEASURABLE,
        UNMEASURABLE,
    ]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_nonfinite_mismatches_is_positional_only(dtype):
    nan = float("nan")
    golden = _t([nan, nan, float("inf"), float("inf"), float("inf"), 1.0], dtype)
    result = _t(
        [nan, 1.0, float("inf"), float("-inf"), torch.finfo(dtype).max, nan], dtype
    )
    #            both NaN  one NaN  both +inf  opposite  inf vs finite  one NaN
    assert nonfinite_mismatches(golden, result).tolist() == [
        False,
        True,
        False,
        True,
        True,
        True,
    ]


def test_nan_sign_is_not_judged_here():
    """``-NaN`` folds to the other operand on Wormhole and ``sfpu_domains`` owns the rule
    for when a NaN's sign may be asserted, so the metric must not hold a second opinion.
    """
    positive = torch.tensor([float("nan")], dtype=torch.float32)
    negative = -positive
    assert bool(torch.signbit(negative)[0]) and not bool(torch.signbit(positive)[0])
    assert nonfinite_mismatches(positive, negative).tolist() == [False]


# ─────────────────────────────────────────────────────────────────────────────
# The composite verdict
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_within_ulp_boundary_is_inclusive(dtype):
    # One name for both, so the test cannot drift into 2 <= 3 and stay green while no
    # longer testing the inclusive boundary it is named for.
    budget = 3
    golden = _t([1.0], dtype)
    assert within_ulp(golden, _step_up(1.0, dtype, steps=budget), max_ulp=budget)[0]
    assert not within_ulp(
        golden, _step_up(1.0, dtype, steps=budget + 1), max_ulp=budget
    )[0]


def test_within_ulp_does_not_pass_on_the_unmeasurable_sentinel():
    """The footgun the composite exists to remove: ``-1 <= max_ulp`` for every budget, so
    a caller gating on the raw distance would call a missing NaN a pass."""
    golden = torch.tensor([1.0, float("nan")], dtype=torch.float32)
    result = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assert ulp_distance(golden, result)[1] == UNMEASURABLE
    assert bool((ulp_distance(golden, result) <= 0).all())  # the naive gate would pass
    ok, message = within_ulp(golden, result, max_ulp=0)
    assert not ok
    assert "non-finite disagreement" in message


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_the_verdict_refuses_an_overflow_no_budget_can_buy(dtype):
    """``finfo.max`` against ``+Inf`` is one step in the value order and still a hard fail.

    The distance and the verdict deliberately part company here.
    :func:`ulp_distance` ranks ``Inf`` one past the largest finite, which is the honest
    reading of an overflow at the very top of the range -- but
    :func:`within_ulp` rejects every finite-against-``Inf`` lane positionally, before any
    step is counted, so no budget buys it. An overflow is a different kind of answer from
    an inexact one, and it is the line ``utils.py::_bfp_block_aware_compare`` already
    takes for the block floats.

    Pinned because neither half of that had a test: no ``within_ulp`` call in this file
    had an ``Inf`` on either side, which is how the docstring and the gate could have
    drifted.
    """
    largest = float(torch.finfo(dtype).max)
    golden, result = _t([largest], dtype), _t([float("inf")], dtype)
    assert int(ulp_distance(golden, result)[0]) == 1  # ...measured, it is one step

    for budget in (0, 1, MAX_MEANINGFUL_ULP[dtype]):
        ok, message = within_ulp(golden, result, max_ulp=budget, fmt=None)
        assert not ok, budget
        assert "non-finite disagreement" in message
    # ...and the other direction, an Inf golden the kernel answered with a finite.
    ok, _ = within_ulp(result, golden, max_ulp=1)
    assert not ok
    # Same-sign Inf against Inf is the only Inf lane that reaches the distance: 0 steps.
    ok, _ = within_ulp(result, result.clone(), max_ulp=0)
    assert ok


def test_within_ulp_passes_when_both_sides_are_nan():
    golden = torch.tensor([1.0, float("nan")], dtype=torch.float32)
    result = torch.tensor([1.0, float("nan")], dtype=torch.float32)
    ok, message = within_ulp(golden, result, max_ulp=0)
    assert ok
    assert "1 unmeasurable" in message


def test_within_ulp_mask_excludes_lanes_already_settled():
    budget = 1
    golden = torch.tensor([1.0, 1.0], dtype=torch.float32)
    result = torch.tensor([1.0, 1000.0], dtype=torch.float32)
    assert not within_ulp(golden, result, max_ulp=budget)[0]
    mask = torch.tensor([True, False])
    assert within_ulp(golden, result, max_ulp=budget, mask=mask)[0]


def test_within_ulp_passes_when_every_lane_is_masked_out():
    """...and says the mask excluded them, not that there was nothing to compare.

    Both lanes here are perfectly measurable. "no measurable lane (0 unmeasurable)" --
    what this printed before -- reads as "nothing to compare" for a tile where every lane
    was comparable and the mask took them all out, and the two have different
    remediations. Reachable on a pass path through a near-zero floor that rescues the
    whole tile.
    """
    golden = torch.tensor([1.0, 1.0], dtype=torch.float32)
    result = torch.tensor([5.0, 1000.0], dtype=torch.float32)
    ok, message = within_ulp(
        golden, result, max_ulp=0, mask=torch.tensor([False, False])
    )
    assert ok
    assert "no lane under judgement" in message
    assert "selected none of 2 lanes" in message
    assert "no measurable lane" not in message

    # ...and the other one still says what it always said.
    nan = float("nan")
    ok, message = within_ulp(
        torch.tensor([nan, nan], dtype=torch.float32),
        torch.tensor([nan, nan], dtype=torch.float32),
        max_ulp=0,
    )
    assert ok
    assert "no measurable lane (2 unmeasurable" in message


def test_the_verdict_reports_what_the_floor_carried():
    """A floor-carried pass and a budget-carried one are not the same result.

    The rescued lanes are exactly the ones `ranked` keeps out of the summary -- they hold
    the largest step counts by construction -- so without this a DEBUG export cannot tell
    which of the two it is looking at. Reported whenever a floor is configured, on the
    pass path as well, and left out entirely when there is none, so the unfloored
    verdicts do not grow a "0 held by the floor" that says nothing.
    """
    golden = torch.tensor([100.0, 1e-4, 2e-4], dtype=torch.float32)
    result = torch.tensor([100.0, 1.1e-4, 2.1e-4], dtype=torch.float32)

    ok, message = within_ulp(golden, result, max_ulp=0, fmt=DataFormat.Float32)
    assert not ok
    assert "held by the floor" not in message

    ok, message = within_ulp(
        golden, result, max_ulp=0, fmt=DataFormat.Float32, near_zero_atol=1e-4
    )
    assert ok
    assert "2 held by the floor" in message


def test_within_ulp_reports_a_shape_mismatch_instead_of_raising():
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
    distance = torch.tensor([0, 2, UNMEASURABLE, 4, UNMEASURABLE], dtype=torch.int64)
    stats = ulp_stats(distance)
    assert stats["lanes"] == 3
    assert stats["unmeasurable"] == 2
    assert stats["max"] == 4
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["exact_frac"] == pytest.approx(1 / 3)
    assert stats["worst_index"] == 3


def test_ulp_stats_worst_index_is_a_flat_index_into_the_input():
    golden = torch.ones(3, 4, dtype=torch.bfloat16)
    result = golden.clone()
    result[2, 1] = _step_up(1.0, torch.bfloat16, steps=5)[0]
    stats = ulp_stats(ulp_distance(golden, result))
    assert stats["max"] == 5
    assert stats["worst_index"] == 2 * 4 + 1
    assert float(result.reshape(-1)[stats["worst_index"]]) == float(result[2, 1])


def test_ulp_stats_falls_back_to_max_below_the_quantile_thresholds():
    """Borrowed from ttnn: a quantile over 5 elements is not a percentile."""
    small = torch.tensor([0, 0, 0, 9, 0], dtype=torch.int64)
    stats = ulp_stats(small)
    assert stats["p95"] == pytest.approx(9.0)
    assert stats["p99"] == pytest.approx(9.0)

    wide = torch.zeros(200, dtype=torch.int64)
    wide[-1] = 9
    stats = ulp_stats(wide)
    assert stats["p95"] == pytest.approx(0.0)
    assert stats["p99"] < 9.0
    assert stats["max"] == 9

    # 5 lanes is below both thresholds and 200 is above both, so neither tells the two
    # constants apart -- collapsing either onto the other keeps them green. 50 lanes sits
    # between them: p95 is a real quantile, p99 still falls back to the max.
    assert _MIN_LANES_FOR_P95 < 50 < _MIN_LANES_FOR_P99
    middle = torch.zeros(50, dtype=torch.int64)
    middle[-1] = 9
    stats = ulp_stats(middle)
    assert stats["max"] == 9
    assert stats["p95"] == pytest.approx(0.0)
    assert stats["p99"] == pytest.approx(9.0)


def test_ulp_stats_on_an_all_unmeasurable_tensor():
    stats = ulp_stats(torch.full((4,), UNMEASURABLE, dtype=torch.int64))
    assert stats["lanes"] == 0
    assert stats["unmeasurable"] == 4
    assert stats["worst_index"] is None


def test_ulp_failure_message_names_the_point_and_the_step():
    golden = torch.ones(8, dtype=torch.bfloat16)
    result = golden.clone()
    result[5] = _step_up(1.0, torch.bfloat16, steps=7)[0]
    distance = ulp_distance(golden, result)
    message = ulp_failure_message(
        golden, result, distance, DataFormat.Float16_b, max_ulp=3
    )
    assert "max 7 ULP @ [5]" in message
    assert "budget 3" in message
    assert "Float16_b" in message
    assert f"{local_step(1.0, torch.bfloat16):.6e}" in message
    assert "87.5% exact" in message


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_local_step_agrees_with_the_sweep_metric(dtype):
    fmt = {
        torch.bfloat16: DataFormat.Float16_b,
        torch.float16: DataFormat.Float16,
        torch.float32: DataFormat.Float32,
    }[dtype]
    for value in (1.0, -1.0, 3.75, 1024.0):
        assert local_step(value, dtype) == pytest.approx(
            float(local_ulp(_t([value], dtype).to(torch.float64).numpy(), fmt)[0])
        )


def test_local_step_of_a_nonfinite_is_not_a_number():
    assert math.isnan(local_step(float("inf"), torch.float32))
    assert math.isnan(local_step(float("nan"), torch.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Format and threshold gating
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fmt, expected",
    [
        (DataFormat.Float32, torch.float32),
        (DataFormat.Float16, torch.float16),
        (DataFormat.Float16_b, torch.bfloat16),
    ],
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
        DataFormat.Tf32,
    ]
    + list(INTEGER_FORMATS),
    ids=lambda f: f.name,
)
def test_ulp_dtype_rejects_formats_without_a_per_element_ulp(fmt):
    """Rejected, not silently redirected: Bfp4_b's 3 magnitude bits leave 2 fractional and
    Bfp2_b's 1 leaves 0, against bfloat16's 7, so a bf16 step count would read every legal
    quantization as a 32- or 128-step error. ``Tf32`` is held in an fp32 container whose
    lattice is not its own. A caller must not be able to think it has a gate it does not
    have."""
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="no per-element ULP"
    ):
        ulp_dtype(fmt)


def test_bfp8_b_is_measured_in_bf16_space():
    """Bfp8_b is close enough to bfloat16 to be gated in its step space -- ttnn makes the
    same choice, and ``passed_test`` has already cast the tensor to bfloat16 -- but not
    equal to it: its 7 magnitude bits *include* an explicit leading 1, so it has 6
    fractional bits against bfloat16's 7 and one Bfp8_b step is two bf16 steps. A budget
    denominated in bf16 steps therefore buys half as many format steps, which is the thing
    an enrolling caller has to know."""
    assert DataFormat.Bfp8_b not in ULP_FORMATS
    assert ulp_dtype(DataFormat.Bfp8_b) == torch.bfloat16

    # Two bf16 steps up from 1.0 is the first to change the encoded Bfp8_b mantissa.
    block = [1.0] * 16
    _, baseline = float_to_bfp8_block(block)
    encoded = []
    for steps in range(3):
        block[0] = float(_step_up(1.0, torch.bfloat16, steps=steps)[0])
        _, mantissas = float_to_bfp8_block(block)
        encoded.append(mantissas[0])
    assert encoded[0] == encoded[1] == baseline[0]
    assert encoded[2] == baseline[0] + 1


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_threshold_sanity_ceiling_is_two_to_the_mantissa_bits(dtype):
    assert MAX_MEANINGFUL_ULP[dtype] == 1 << MANTISSA_BITS[dtype]
    assert not warn_if_threshold_unmeaningful(MAX_MEANINGFUL_ULP[dtype], dtype)
    assert warn_if_threshold_unmeaningful(MAX_MEANINGFUL_ULP[dtype] + 1, dtype)


def test_ulp_distance_rejects_mismatched_and_unsupported_inputs():
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="dtype mismatch"
    ):
        ulp_distance(
            torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.bfloat16)
        )
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="shape mismatch"
    ):
        ulp_distance(
            torch.zeros(2, dtype=torch.float32), torch.zeros(3, dtype=torch.float32)
        )
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="unsupported dtype"
    ):
        ulp_distance(
            torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
        )
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="unsupported dtype"
    ):
        ulp_distance(
            torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)
        )


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_distance_is_symmetric_and_shape_preserving(dtype):
    torch.manual_seed(0)
    golden = torch.randn(3, 5, 7, dtype=torch.float32).to(dtype)
    result = golden.clone()
    result.reshape(-1)[::3] = torch.nextafter(
        result.reshape(-1)[::3], torch.full_like(result.reshape(-1)[::3], float("inf"))
    )
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
    ok, message = within_ulp(values, values.clone(), max_ulp=0)
    assert ok
    assert "100.0% exact" in message


def test_a_non_contiguous_input_is_measured_correctly():
    """A transposed tile is the ordinary case in this harness, so the flattening inside
    the metric has to stay in step with the caller's own flat indexing.

    Distinct values and a known perturbation, not a constant tensor: over a constant
    tensor ``distance.max() == 0`` holds under any ordering, so the test could not fail.
    Contiguity itself is not what is load-bearing today -- an equal-itemsize ``view()``
    carries strides through -- but it becomes so for a future different-width
    ``bits_dtype`` (fp8), and this is the assertion that would catch a desynchronised
    flatten either way."""
    golden = torch.arange(24).reshape(4, 6).to(torch.bfloat16).t()
    assert not golden.is_contiguous()
    assert len(set(golden.reshape(-1).tolist())) == 24

    steps = 5
    position = (
        3,
        1,
    )  # a position whose flat index differs before and after transposing
    result = golden.clone()
    result[position] = _step_up(float(golden[position]), torch.bfloat16, steps=steps)[0]

    distance = ulp_distance(golden, result)
    assert int(distance.max()) == steps
    stats = ulp_stats(distance)
    assert stats["max"] == steps
    # The flat index has to point back at the perturbed lane of the transposed view.
    expected_flat = position[0] * golden.shape[1] + position[1]
    assert stats["worst_index"] == expected_flat
    assert float(golden.reshape(-1)[stats["worst_index"]]) == float(golden[position])


# ─────────────────────────────────────────────────────────────────────────────
# The flush default is per dtype, because fp16 keeps its subnormals
# ─────────────────────────────────────────────────────────────────────────────


def test_the_flush_default_follows_the_harness_ftz_model():
    """bf16 and fp32 flush below their smallest normal, so collapsing the band is a no-op
    there. fp16's threshold in ``golden_generators._FTZ_THRESHOLD`` is ``2**-24``, the
    smallest fp16 *subnormal*, so an fp16 golden legitimately carries the whole band."""
    assert flushes_subnormals(torch.bfloat16) is True
    assert flushes_subnormals(torch.float32) is True
    assert flushes_subnormals(torch.float16) is False


def test_fp16_subnormals_are_measured_not_collapsed():
    """The blind spot this guards: with a blanket flush, every fp16 pair inside the band
    read as 0 steps. ``2**-24`` against ``1023 * 2**-24`` is 1022 representable steps and
    a ~1000x relative error -- almost the whole ``MAX_MEANINGFUL_ULP[float16]`` range --
    and a 1-step fp16 gate would have seen none of it."""
    smallest = 2.0**-24
    golden = _t([smallest], torch.float16)
    result = _t([1023 * smallest], torch.float16)
    assert float(golden) != 0.0 and float(result) != 0.0

    assert int(ulp_distance(golden, result)[0]) == 1022
    # The old blanket behaviour, kept available for a caller that knows the Dest flushed.
    assert int(ulp_distance(golden, result, flush_subnormals=True)[0]) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_flushing_is_a_no_op_for_the_formats_that_already_flush(dtype):
    """Which is why the default is safe: for bf16 and fp32 the golden carries no
    subnormals, so the two settings agree on every *same-sign pair outside the band* --
    which is every pair the harness can produce for these two formats.

    Deliberately not the wider claim that the settings agree on any value at all. The
    compaction shifts each magnitude rank by ``2**mantissa_bits - 1``, so it cancels in a
    same-sign subtraction and survives a sign-crossing one: bf16 ``1.0`` against ``-1.0``
    is 32258 steps flushed and 32512 unflushed. What the compaction itself is worth is
    pinned by ``test_flush_makes_smallest_normal_one_step_from_zero``; this test is only
    about the default being inert where it is on by default.
    """
    torch.manual_seed(0)
    values = torch.randn(256, dtype=torch.float32).to(dtype)
    other = torch.nextafter(values, torch.full_like(values, float("inf")))
    assert torch.equal(torch.signbit(values), torch.signbit(other))
    assert bool((values.abs() >= torch.finfo(dtype).tiny).all())
    assert torch.equal(
        ulp_distance(values, other, flush_subnormals=True),
        ulp_distance(values, other, flush_subnormals=False),
    )

    # ...and the crossing where they do not agree, so the narrowed claim is the tested one
    # rather than an untested caveat in a docstring.
    mantissa_bits = MANTISSA_BITS[dtype]
    one, minus_one = _t([1.0], dtype), _t([-1.0], dtype)
    flushed = int(ulp_distance(one, minus_one, flush_subnormals=True)[0])
    unflushed = int(ulp_distance(one, minus_one, flush_subnormals=False)[0])
    assert unflushed - flushed == 2 * (2**mantissa_bits - 1)


# ─────────────────────────────────────────────────────────────────────────────
# The reported step has to agree with the counted one
# ─────────────────────────────────────────────────────────────────────────────


# ``pytest.approx`` defaults to ``abs=1e-12``, which is larger than every value in a
# subnormal band -- so a bare ``approx(tiny)`` accepts any subnormal-scale number at all
# and cannot fail. Every comparison at that scale below passes ``abs=0`` so the relative
# tolerance is the only one in play; without it the flush-boundary test one section down
# stayed green with its fix reverted.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_the_reported_step_inside_a_flushed_band_is_the_jump_to_the_normal(dtype):
    """With the band collapsed, one counted step out of zero lands on the smallest
    normal. Reporting the raw ``nextafter`` gap there understates it by
    ``2**mantissa_bits`` and contradicts the distance the same message prints."""
    tiny = float(torch.finfo(dtype).tiny)
    assert local_step(0.0, dtype, flush_subnormals=True) == pytest.approx(tiny, abs=0)
    assert local_step(tiny / 4, dtype, flush_subnormals=True) == pytest.approx(
        tiny, abs=0
    )
    # Without the collapse it is the true gap, which is far smaller.
    assert local_step(tiny / 4, dtype, flush_subnormals=False) < tiny


def test_the_reported_step_for_fp16_near_zero_is_the_true_gap():
    """fp16 does not flush, so there is nothing to compact and the raw gap is correct."""
    smallest = 2.0**-24
    assert local_step(smallest, torch.float16) == pytest.approx(smallest, abs=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_the_step_into_a_flushed_band_is_the_jump_it_makes(dtype):
    """The boundary the band check used to sit just outside of.

    With the band collapsed, zero and every subnormal share rank 0 and the smallest normal
    is rank 1 -- so one counted step at ``tiny`` heading *down* lands on zero and is worth
    ``tiny``. The raw downward gap there is the smallest *subnormal*, which understates the
    step by ``2**mantissa_bits`` and disagrees with the integer ranking that
    ``test_flush_makes_smallest_normal_one_step_from_zero`` pins.
    """
    tiny = float(torch.finfo(dtype).tiny)
    mantissa_bits = MANTISSA_BITS[dtype]

    # Down from the smallest normal: one step, and it is the whole jump to zero.
    assert int(ulp_distance(_t([tiny], dtype), _t([0.0], dtype))[0]) == 1
    assert local_step(tiny, dtype, toward=0.0) == pytest.approx(tiny, abs=0)
    # The raw gap, which is what it used to report.
    assert local_step(tiny, dtype, toward=0.0, flush_subnormals=False) == pytest.approx(
        tiny * 2.0**-mantissa_bits, abs=0
    )
    # Upward from the smallest normal is an ordinary normal-range step.
    assert local_step(tiny, dtype, toward=2.0 * tiny) == pytest.approx(
        tiny * 2.0**-mantissa_bits, abs=0
    )
    # And inside the band it stays the jump out, in either direction.
    assert local_step(tiny / 4, dtype, toward=0.0) == pytest.approx(tiny, abs=0)
    assert local_step(tiny / 4, dtype, toward=1.0) == pytest.approx(tiny, abs=0)


def test_the_reported_step_is_taken_from_the_signed_direction_not_the_magnitude():
    """``abs(toward) < magnitude`` loses the direction whenever the pair straddles zero.

    ``1.0 -> -2.0`` has the larger magnitude on the far side, so a magnitude comparison
    calls it upward -- but the first step out of ``1.0`` is downward, and the gap below a
    power of two is half the one above it, so that names the wrong side of the boundary.
    """
    # Crossing zero: the path leaves 1.0 heading down, whatever |toward| is.
    assert local_step(1.0, torch.bfloat16, toward=-2.0) == pytest.approx(BELOW_ONE)
    assert local_step(1.0, torch.bfloat16, toward=-0.5) == pytest.approx(BELOW_ONE)
    # A negative value is symmetric: "toward zero" means increasing, not decreasing.
    assert local_step(-1.0, torch.bfloat16, toward=2.0) == pytest.approx(BELOW_ONE)
    assert local_step(-1.0, torch.bfloat16, toward=-2.0) == pytest.approx(ABOVE_ONE)
    # Same sign, no crossing: unchanged.
    assert local_step(1.0, torch.bfloat16, toward=0.5) == pytest.approx(BELOW_ONE)
    assert local_step(1.0, torch.bfloat16, toward=2.0) == pytest.approx(ABOVE_ONE)


def test_the_step_at_zero_has_no_downward_direction():
    """``copysign(1.0, 0.0)`` is ``+1.0``, so a bare signed-delta check calls every
    negative *toward* downward at ``value == 0`` -- and ``nextafter(+0.0, 0.0)`` is
    ``+0.0``, so the step came back ``0.0`` and the message printed
    ``1 ULP = 0.000000e+00`` beside a nonzero count, on exactly the zero-crossing lane
    (``sin``/``tanh``/``erf`` at 0) it exists to explain.

    fp16 is the case that reached it: ranks ``-1`` and ``+1`` both sit at the smallest
    subnormal, so the step out of rank 0 is ``2**-24`` whichever way *toward* points. The
    flushing dtypes exit through the band branch above instead, which is why the existing
    zero coverage did not catch this.
    """
    smallest_fp16_subnormal = 2.0**-24
    for toward in (-1.0, -smallest_fp16_subnormal, 1.0, None, 0.0):
        step = local_step(0.0, torch.float16, toward=toward)
        assert step == pytest.approx(smallest_fp16_subnormal, abs=0), toward
    # -0.0 mirrors it, and the band branch of a flushing dtype is unaffected either way.
    assert local_step(-0.0, torch.float16, toward=1.0) == pytest.approx(
        smallest_fp16_subnormal, abs=0
    )
    for dtype in (torch.bfloat16, torch.float32):
        tiny = float(torch.finfo(dtype).tiny)
        assert local_step(0.0, dtype, toward=-1.0) == pytest.approx(tiny, abs=0)
        assert local_step(0.0, dtype, toward=1.0) == pytest.approx(tiny, abs=0)


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.int32, torch.bool], ids=lambda d: str(d)
)
def test_the_step_and_the_flush_default_refuse_a_dtype_the_metric_does_not_measure(
    dtype,
):
    """``.get(dtype, True)`` invented an answer for a dtype ``ulp_distance`` refuses, and
    ``True`` is the error-hiding polarity the table's comment warns about:
    ``local_step(1.0, torch.float64)`` handed back ``2**-52``. ``local_step`` needs its own
    guard as well, because an explicit ``flush_subnormals=`` skips the lookup entirely.
    """
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="unsupported dtype"
    ):
        flushes_subnormals(dtype)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="unsupported dtype"
    ):
        local_step(1.0, dtype)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="unsupported dtype"
    ):
        local_step(1.0, dtype, flush_subnormals=True)


def test_a_verdict_at_the_top_of_the_range_reports_a_finite_step():
    golden = _t([float(torch.finfo(torch.bfloat16).max)], torch.bfloat16)
    result = _step_down(float(torch.finfo(torch.bfloat16).max), torch.bfloat16)
    ok, message = within_ulp(golden, result, max_ulp=0, fmt=DataFormat.Float16_b)
    assert not ok
    assert "1 ULP = inf" not in message


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_the_finfo_max_fixup_carries_its_own_weight(dtype):
    """The test above does not reach the ``float(scalar) == info.max`` branch: its result
    is smaller in magnitude, so the *direction* check already sends it downward and
    deleting the ``info.max`` disjunct leaves it green.

    Two cases that need the disjunct, over every dtype rather than bf16 only:

    * no ``toward`` at all, where nothing else can choose a direction;
    * ``toward=+inf``, which ``math.isfinite`` rejects, so the signed-delta check cannot
      fire and the disjunct is the only thing keeping the gap to ``Inf`` out of the
      report.

    ``+max`` against ``-max`` is *not* one of them, and is checked below only end to end:
    the signed delta there is ``-2 * max``, so ``heads_toward_zero`` already picks the
    downward gap and deleting the disjunct leaves it green.
    """
    largest = float(torch.finfo(dtype).max)
    assert math.isfinite(local_step(largest, dtype))
    assert math.isfinite(local_step(largest, dtype, toward=float("inf")))
    assert math.isfinite(local_step(largest, dtype, toward=-largest))
    # Same binade downward, so it is exactly the step below the largest finite.
    expected = largest - float(_step_down(largest, dtype))
    assert local_step(largest, dtype) == pytest.approx(expected)
    assert local_step(largest, dtype, toward=float("inf")) == pytest.approx(expected)
    assert local_step(largest, dtype, toward=-largest) == pytest.approx(expected)

    golden = _t([largest], dtype)
    result = _t([-largest], dtype)
    ok, message = within_ulp(golden, result, max_ulp=0)
    assert not ok
    assert "1 ULP = inf" not in message


# ─────────────────────────────────────────────────────────────────────────────
# within_ulp validates the format it labels the verdict with
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fmt",
    [DataFormat.Bfp4_b, DataFormat.Bfp2_b, DataFormat.MxFp8P, DataFormat.Tf32],
    ids=lambda f: f.name,
)
def test_within_ulp_refuses_a_format_with_no_per_element_ulp(fmt):
    """``fmt`` is not only a label. ``format_dict`` collapses every block float onto
    ``torch.bfloat16`` and ``Tf32`` onto ``torch.float32``, so the dtype check inside
    ``ulp_distance`` cannot tell them apart -- a verdict labelled ``Bfp2_b`` would come
    back measured in bfloat16 steps, the measurement ``ulp_dtype`` exists to refuse.
    ``Bfp8_b`` is the one block float that is *not* on this list: it is gated in bf16 step
    space on purpose, which the test fifteen lines down pins."""
    values = torch.ones(4, dtype=torch.bfloat16)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="no per-element ULP"
    ):
        within_ulp(values, values.clone(), max_ulp=1, fmt=fmt)


@pytest.mark.parametrize(
    "fmt,wrong",
    [
        (DataFormat.Float16_b, torch.float32),
        (DataFormat.Float32, torch.bfloat16),
        (DataFormat.Float16, torch.float32),
    ],
    ids=lambda x: getattr(x, "name", str(x)),
)
def test_within_ulp_refuses_a_supported_format_that_disagrees_with_the_dtype(
    fmt, wrong
):
    """Allowlisting the format is not the whole check. Two float32 tensors labelled
    ``Float16_b`` would be measured in float32 steps under a Float16_b verdict -- the
    wrong lattice for the gate, and a wrong pass/fail, with nothing in the message to say
    so. The label and the lattice have to be the same claim."""
    values = torch.ones(4, dtype=wrong)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="but the tensors are"
    ):
        within_ulp(values, values.clone(), max_ulp=1, fmt=fmt)


@pytest.mark.parametrize("shape", [(1,), (4,), (3, 4)], ids=str)
def test_the_verdict_refuses_a_mask_it_would_otherwise_broadcast(shape):
    """``mask`` is the input a per-op budget registry computes dynamically, so a
    wrong-shaped one has to be a named error. A ``(1,)`` mask is the dangerous case: it
    broadcasts all the way through and silently judges every lane or none."""
    golden = torch.ones(2, 6, dtype=torch.bfloat16)
    mask = torch.ones(shape, dtype=torch.bool)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="mask shape"
    ):
        within_ulp(golden, golden.clone(), max_ulp=1, mask=mask)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="mask shape"
    ):
        ulp_stats(ulp_distance(golden, golden.clone()), mask)


def test_the_reported_step_below_a_power_of_two_is_the_gap_it_crossed():
    """The gap is half as large below a boundary as above it. Reporting the upward gap for
    a result that stepped *down* across ``1.0`` would print ``1 ULP = 7.8125e-3`` for a
    step of ``3.90625e-3`` -- the asymmetry the integer count exists to avoid, reappearing
    in the diagnostic that sits next to it."""
    assert local_step(1.0, torch.bfloat16, toward=0.0) == pytest.approx(BELOW_ONE)
    assert local_step(1.0, torch.bfloat16, toward=2.0) == pytest.approx(ABOVE_ONE)
    assert local_step(1.0, torch.bfloat16) == pytest.approx(ABOVE_ONE)

    golden = _t([1.0], torch.bfloat16)
    result = _step_down(1.0, torch.bfloat16)
    ok, message = within_ulp(golden, result, max_ulp=0, fmt=DataFormat.Float16_b)
    assert not ok
    assert "max 1 ULP" in message
    assert f"{BELOW_ONE:.6e}" in message
    assert f"{ABOVE_ONE:.6e}" not in message


def test_the_reported_step_above_a_power_of_two_is_unchanged():
    """The upward gap stays the default, so nothing about an ordinary failure moves."""
    golden = _t([1.0], torch.bfloat16)
    result = _step_up(1.0, torch.bfloat16, steps=3)
    ok, message = within_ulp(golden, result, max_ulp=0, fmt=DataFormat.Float16_b)
    assert not ok
    assert f"{ABOVE_ONE:.6e}" in message


def test_within_ulp_accepts_bfp8_b_in_its_proxy_space():
    """``Bfp8_b`` is gated in bfloat16 step space, so unlike the coarser block floats it
    is a format ``within_ulp`` may legitimately label a verdict with."""
    values = torch.ones(4, dtype=torch.bfloat16)
    ok, message = within_ulp(values, values.clone(), max_ulp=0, fmt=DataFormat.Bfp8_b)
    assert ok and "Bfp8_b" in message


def test_within_ulp_still_works_without_a_format():
    values = torch.ones(4, dtype=torch.bfloat16)
    ok, message = within_ulp(values, values.clone(), max_ulp=0)
    assert ok and "bfloat16" in message


def test_the_message_builder_can_reuse_stats_it_was_given():
    """The verdict already computed them; building the message must not pay again."""
    golden = torch.ones(8, dtype=torch.bfloat16)
    result = golden.clone()
    result[3] = _step_up(1.0, torch.bfloat16, steps=4)[0]
    distance = ulp_distance(golden, result)
    stats = ulp_stats(distance)
    assert ulp_failure_message(
        golden, result, distance, DataFormat.Float16_b, stats=stats
    ) == ulp_failure_message(golden, result, distance, DataFormat.Float16_b)


# The elementwise verdict the gate consumes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_elementwise_valid_marks_only_the_out_of_budget_lanes(dtype):
    golden = torch.ones(4, dtype=dtype)
    result = torch.stack(
        [
            _t([1.0], dtype)[0],
            _step_up(1.0, dtype, steps=1)[0],
            _step_up(1.0, dtype, steps=2)[0],
            _step_down(1.0, dtype, steps=3)[0],
        ]
    )
    is_valid, distance, _rescued = ulp_elementwise_valid(golden, result, 1)
    assert distance.tolist() == [0, 1, 2, 3]
    assert is_valid.tolist() == [True, True, False, False]


def test_elementwise_valid_accepts_both_nan_and_rejects_a_missing_one():
    nan = float("nan")
    golden = torch.tensor([nan, nan, 1.0], dtype=torch.float32)
    result = torch.tensor([nan, 1.0, nan], dtype=torch.float32)
    is_valid, _, _rescued = ulp_elementwise_valid(golden, result, 0)
    assert is_valid.tolist() == [True, False, False]


def test_elementwise_valid_rejects_an_overflow_to_inf_even_though_it_ranks_one_step():
    """The gate is stricter than the metric here, deliberately: ``ulp_distance`` ranks Inf
    one step past the largest finite, but an overflow where the reference is finite is a
    different kind of wrong, and the tolerance gate this replaces rejects it too."""
    golden = torch.tensor([torch.finfo(torch.float32).max], dtype=torch.float32)
    result = torch.tensor([float("inf")], dtype=torch.float32)
    is_valid, distance, _rescued = ulp_elementwise_valid(golden, result, 1)
    assert int(distance[0]) == 1
    assert is_valid.tolist() == [False]


def test_near_zero_atol_rescues_a_cancellation_lane():
    """The case the floor exists for: a golden that crosses zero, where ``ulp(golden)``
    collapses and a tiny absolute error becomes an enormous step count."""
    golden = torch.tensor([100.0, 1e-8], dtype=torch.float32)
    result = torch.tensor([100.0, 2e-8], dtype=torch.float32)

    is_valid, distance, _rescued = ulp_elementwise_valid(golden, result, 4)
    assert int(distance[1]) > 1000  # meaningless as a kernel verdict
    assert is_valid.tolist() == [True, False]

    is_valid, _, _rescued = ulp_elementwise_valid(
        golden, result, 4, near_zero_atol=1e-7
    )
    assert is_valid.tolist() == [True, True]


def test_near_zero_atol_does_not_loosen_the_large_magnitude_lanes():
    """An atol applied at every magnitude is the format- and magnitude-blind gate the step
    count replaces, so the floor must stay below the near-zero cut."""
    golden = torch.tensor([100.0, 1e-8], dtype=torch.float32)
    result = torch.tensor([100.5, 1e-8], dtype=torch.float32)
    is_valid, _, _rescued = ulp_elementwise_valid(golden, result, 0, near_zero_atol=1.0)
    assert is_valid.tolist() == [False, True]


def test_near_zero_cut_is_a_fraction_of_the_tensors_dynamic_range():
    dynamic_range = 100.0
    just_below = dynamic_range * NEAR_ZERO_FRACTION * 0.9
    just_above = dynamic_range * NEAR_ZERO_FRACTION * 1.1
    golden = torch.tensor([dynamic_range, just_below, just_above], dtype=torch.float32)
    result = golden.clone()
    result[1] += 0.5
    result[2] += 0.5
    is_valid, _, _rescued = ulp_elementwise_valid(golden, result, 0, near_zero_atol=1.0)
    assert is_valid.tolist() == [True, True, False]


def test_near_zero_atol_covers_every_lane_of_an_all_zero_golden():
    golden = torch.zeros(3, dtype=torch.float32)
    result = torch.tensor([0.0, 1e-9, 1.0], dtype=torch.float32)
    is_valid, _, _rescued = ulp_elementwise_valid(
        golden, result, 0, near_zero_atol=1e-8
    )
    assert is_valid.tolist() == [True, True, False]


def test_elementwise_valid_without_the_floor_is_the_plain_budget():
    torch.manual_seed(0)
    golden = torch.randn(128, dtype=torch.float32)
    result = torch.nextafter(golden, torch.full_like(golden, float("inf")))
    assert torch.all(ulp_elementwise_valid(golden, result, 1)[0])
    assert not torch.any(ulp_elementwise_valid(golden, result, 0)[0])


# ─────────────────────────────────────────────────────────────────────────────
# What a verdict says when the step count cannot describe the failure
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_the_step_at_the_largest_finite_is_not_infinite(dtype):
    """``nextafter`` from the largest finite goes to ``Inf``, so the naive difference is
    infinite and a message about a failure at the top of the range reads "1 ULP = inf".
    The binade is the same downward, so the step is the same size measured that way.
    ttnn's ``ulp()`` carries the same ``finfo.max`` fixup."""
    largest = float(torch.finfo(dtype).max)
    step = local_step(largest, dtype)
    assert math.isfinite(step)
    assert step > 0
    # Same binade, so it is exactly the step below the largest finite.
    below = _step_down(largest, dtype)
    assert step == pytest.approx(largest - float(below))


def test_a_missing_nan_is_named_rather_than_reported_as_zero_steps():
    """The defect this guards: a NaN lane is UNMEASURABLE and drops out of the statistics,
    so a verdict that failed only on a missing NaN used to report "max 0 ULP (budget 0)"
    — true, and useless to whoever has to fix it."""
    golden = torch.tensor([1.0, float("nan"), 2.0], dtype=torch.float32)
    result = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32)
    distance = ulp_distance(golden, result)

    assert ulp_stats(distance)["max"] == 0  # the statistics genuinely see nothing
    summary = nonfinite_disagreement_summary(golden, result, DataFormat.Float32)
    assert summary is not None
    assert "non-finite disagreement @ [1]" in summary
    assert "1 such lane(s)" in summary

    message = ulp_verdict_message(
        golden, result, distance, DataFormat.Float32, max_ulp=0
    )
    assert message.startswith("non-finite disagreement @ [1]")
    assert "max 0 ULP" in message  # the step summary is still there, as detail


def test_an_agreeing_verdict_has_no_disagreement_line():
    golden = torch.tensor([1.0, float("nan"), float("inf")], dtype=torch.float32)
    result = golden.clone()
    assert nonfinite_disagreement_summary(golden, result, DataFormat.Float32) is None
    message = ulp_verdict_message(
        golden, result, ulp_distance(golden, result), DataFormat.Float32, max_ulp=0
    )
    assert "non-finite disagreement" not in message


def test_the_disagreement_count_covers_every_bad_lane():
    nan = float("nan")
    golden = torch.tensor([nan, nan, nan, 1.0], dtype=torch.float32)
    result = torch.tensor([1.0, 2.0, nan, 1.0], dtype=torch.float32)
    summary = nonfinite_disagreement_summary(golden, result, DataFormat.Float32)
    assert "@ [0]" in summary and "2 such lane(s)" in summary


def test_within_ulp_reproduces_every_gate_verdict_including_the_floor():
    """Verdict parity, not just message parity. Without a ``near_zero_atol`` passthrough
    there were gate verdicts ``within_ulp`` could not reach at any argument: the same
    tensor and budget that the floor flips to a pass had no corresponding call here."""
    golden = torch.linspace(1.0, 100.0, 64, dtype=torch.float32)
    golden[-1] = 1e-8
    result = golden.clone()
    result[-1] = 2e-8

    assert not within_ulp(golden, result, max_ulp=2, fmt=DataFormat.Float32)[0]
    ok, _ = within_ulp(
        golden, result, max_ulp=2, fmt=DataFormat.Float32, near_zero_atol=1e-7
    )
    assert ok
    # Same verdict the gate reaches, for the same reason.
    is_valid, _, rescued = ulp_elementwise_valid(golden, result, 2, near_zero_atol=1e-7)
    assert bool(torch.all(is_valid)) and bool(rescued[-1])


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_the_gate_resolves_the_flush_default_per_dtype_like_the_metric(dtype):
    """``ulp_elementwise_valid`` used to hardcode ``flush_subnormals=True``, which
    overrode the per-dtype default for every caller including ``passed_test``. For fp16,
    which keeps its subnormals in this harness, that collapsed up to 1023 representable
    steps of error near zero onto 0 and a 0-step budget accepted it."""
    assert flushes_subnormals(dtype) is (dtype is not torch.float16)
    # Two distinct values inside *this* dtype's subnormal band: the smallest subnormal and
    # the largest one, which are 2**mantissa_bits - 1 representable steps apart.
    smallest = float(torch.finfo(dtype).tiny) * 2.0 ** -MANTISSA_BITS[dtype]
    band_steps = (1 << MANTISSA_BITS[dtype]) - 1
    golden = _t([smallest], dtype)
    result = _t([band_steps * smallest], dtype)
    assert float(golden) != 0.0 and float(result) != float(golden)

    is_valid, distance, _ = ulp_elementwise_valid(golden, result, 0)
    if dtype is torch.float16:
        assert int(distance[0]) == band_steps - 1
        assert not bool(is_valid[0])
    else:
        # bf16 and fp32 flush their band, so both sides really are zero to this harness.
        assert int(distance[0]) == 0
        assert bool(is_valid[0])

    # The old blanket behaviour stays reachable for a caller that knows the Dest flushed.
    forced, _, _ = ulp_elementwise_valid(golden, result, 0, flush_subnormals=True)
    assert bool(forced[0])


def test_the_near_zero_band_is_bounded_absolutely_as_well_as_relatively():
    """The relative band alone is unbounded in absolute terms, so one large golden widens
    it across the tile and mid-range lanes get the magnitude-blind gate a step count
    exists to replace. With fp32, ``max_ulp=1`` and ``near_zero_atol=1e-6`` over a golden
    spanning [0, 1000] the relative cut lands at 10.0, and ``1.0`` vs ``1.0000008`` --
    7 representable steps -- was rescued against a 1-step budget.

    The absolute cut is ``near_zero_atol / near_zero_fraction``, the magnitude at which the
    forgiven error is exactly ``near_zero_fraction`` of the reference. Above it the
    relative error is under 1% and the reference has not collapsed."""
    near_zero_atol = 1e-6
    golden = torch.tensor([1000.0, 1.0, 1e-9], dtype=torch.float32)
    result = golden.clone()
    result[1] = 1.0000008  # 7 fp32 steps, inside the relative cut of 10.0
    result[2] = 1e-9 + 5e-7  # a genuine cancellation lane, inside the atol

    assert int(ulp_distance(golden, result)[1]) > 1
    assert 1.0 < NEAR_ZERO_FRACTION * 1000.0  # the relative rule alone would rescue it
    assert 1.0 > near_zero_atol / NEAR_ZERO_FRACTION  # the absolute rule refuses it

    is_valid, _, rescued = ulp_elementwise_valid(
        golden, result, 1, near_zero_atol=near_zero_atol
    )
    assert is_valid.tolist() == [True, False, True]
    assert rescued.tolist() == [False, False, True]

    # Tightening the atol cannot substitute for the bound: the step count it admits grows
    # as 1/|golden| below the cut, so any atol loose enough for the cancellation lane is
    # loose enough for the mid-band one under the relative rule alone.
    assert not ulp_elementwise_valid(golden, result, 1, near_zero_atol=1e-9)[0][2]


def test_within_ulp_and_the_gate_describe_a_verdict_the_same_way():
    """One message builder for both, so the two cannot drift into describing the same
    failure differently."""
    golden = torch.tensor([1.0, float("nan")], dtype=torch.float32)
    result = torch.tensor([1.0, 1.0], dtype=torch.float32)
    _, from_within_ulp = within_ulp(golden, result, max_ulp=0, fmt=DataFormat.Float32)
    from_builder = ulp_verdict_message(
        golden,
        result,
        ulp_distance(golden, result),
        DataFormat.Float32,
        mask=torch.ones_like(golden, dtype=torch.bool),
        max_ulp=0,
    )
    assert from_within_ulp == from_builder


def test_a_failure_at_the_top_of_the_range_reports_a_usable_step():
    """Both fixes at once: the overflow names itself, and the step it is measured against
    is a number rather than infinity."""
    largest = float(torch.finfo(torch.bfloat16).max)
    golden = torch.tensor([largest], dtype=torch.bfloat16)
    result = torch.tensor([float("inf")], dtype=torch.bfloat16)
    ok, message = within_ulp(golden, result, max_ulp=4, fmt=DataFormat.Float16_b)
    assert not ok
    assert message.startswith("non-finite disagreement")
    assert "1 ULP = inf" not in message


def test_elementwise_valid_reports_which_lanes_the_floor_rescued():
    """The third return value. Those lanes hold the largest step counts in the tensor by
    construction, so a reporting caller that ranks every lane names a lane that *passed*
    and never mentions the one that failed."""
    golden = torch.tensor([100.0, 1e-8], dtype=torch.float32)
    result = torch.tensor([100.0, 2e-8], dtype=torch.float32)
    is_valid, distance, rescued = ulp_elementwise_valid(
        golden, result, 4, near_zero_atol=1e-7
    )
    assert is_valid.tolist() == [True, True]
    assert rescued.tolist() == [False, True]
    # Ranked without the rescued lane, the summary names nothing; with it, it names the
    # lane that was never a failure.
    assert ulp_stats(distance, ~rescued)["max"] == 0
    assert ulp_stats(distance)["max"] > 1000


def test_no_lane_is_reported_as_rescued_when_it_was_in_budget_anyway():
    golden = torch.tensor([100.0, 1e-8], dtype=torch.float32)
    result = golden.clone()
    _, _, rescued = ulp_elementwise_valid(golden, result, 4, near_zero_atol=1e-7)
    assert not bool(rescued.any())


def test_the_sweep_metric_covers_the_proxy_formats_the_gate_judges():
    """``local_ulp`` probed a private copy of the native format tuple, so the sweep wrote
    NaN ``signed_ulp_error`` for exactly the format the gate can now judge. It asks
    ``has_ulp_gate`` now, making the proxy table the single source of truth."""
    values = np.array([1.0, 2.0])
    assert not np.isnan(local_ulp(values, DataFormat.Bfp8_b)).any()
    assert np.isnan(local_ulp(values, DataFormat.Bfp4_b)).all()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_the_sweep_metric_matches_local_step_at_the_top_of_the_range(dtype):
    """The gap the reviewer pointed at: ``test_local_step_agrees_with_the_sweep_metric``
    never probed ``finfo.max``, so the ``nextafter`` saturation there went uncaught in
    both functions and the shared-definition claim quietly stopped holding."""
    fmt = {
        torch.bfloat16: DataFormat.Float16_b,
        torch.float16: DataFormat.Float16,
        torch.float32: DataFormat.Float32,
    }[dtype]
    largest = float(torch.finfo(dtype).max)
    swept = float(local_ulp(np.array([largest]), fmt)[0])
    assert math.isfinite(swept)
    assert swept == pytest.approx(local_step(largest, dtype))


def test_the_near_zero_band_is_scoped_to_the_lanes_under_judgement():
    """``dynamic_range`` is the only verdict input that is not elementwise, so an unscoped
    max let a large golden in a lane the caller masked *out* widen the band applied to the
    lanes it masked *in* -- the first place an excluded lane could change a judged lane's
    verdict.

    The measured case: the excluded lane's ``1e6`` puts the relative cut at ``1e4``, so a
    judged lane at ``1.0`` that is 167772 steps over budget gets rescued by the floor.
    Scoped to the judged lane the cut is ``0.01`` and it fails, which is the right answer.
    ``mask`` plus ``near_zero_atol`` is what the budget registry wants, and nothing
    combined the two before."""
    golden = torch.tensor([1e6, 1.0], dtype=torch.float32)
    result = torch.tensor([1e6, 1.02], dtype=torch.float32)
    mask = torch.tensor([False, True])

    assert int(ulp_distance(golden, result)[1]) > 100000
    ok, _ = within_ulp(golden, result, max_ulp=0, near_zero_atol=0.05, mask=mask)
    assert not ok, "the excluded lane must not widen the band for the judged one"

    # Unmasked, the same tensors legitimately reach the wide band -- so the fix is the
    # scoping, not a change to the rule.
    is_valid, _, rescued = ulp_elementwise_valid(golden, result, 0, near_zero_atol=0.05)
    assert bool(is_valid.all()) and bool(rescued[1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_the_relative_near_zero_cut_is_compared_in_float32_not_the_tensor_dtype(dtype):
    """Both cuts are Python floats, so a 16-bit ``golden.abs()`` promoted them onto the
    tensor's own lattice and rounded each edge to a representable value -- narrowing the
    relative ``<`` edge and widening the absolute ``<=`` one. A lane sitting exactly on a
    rounded edge then got a different verdict than the unrounded cut, and than the
    identical call on fp32.

    Probing at plus or minus 10% of the cut cannot see this; the lane has to sit on the
    edge. So this picks a dynamic range whose 1% cut is *not* representable in *dtype*,
    puts a lane at the value that rounding would move across it, and requires the verdict
    to match fp32's.

    The cut has to round **down** for the lane to separate the two compares: nothing
    representable lies strictly between ``cut`` and ``rounded``, so the lane collapses
    onto ``rounded`` itself, and only ``rounded < cut`` makes the fp32 compare (in band)
    and the narrow one (``rounded < rounded``, out of band) disagree. ``dynamic_range =
    13.0`` satisfied that for bf16 but not for fp16, which rounds 0.13 *up* -- so the
    fp16 parameter was green either way. 11.0 rounds down in both, and the assertion
    below pins it rather than leaving it to be rediscovered.
    """
    # 1% of this is 0.11, which is not a bf16/fp16 value; the neighbours straddle it.
    dynamic_range = 11.0
    cut = NEAR_ZERO_FRACTION * dynamic_range
    rounded = float(torch.tensor(cut, dtype=dtype))
    assert rounded != cut, "pick a cut that the dtype cannot represent"
    assert rounded < cut, "the cut must round down, or the lane cannot separate the two"

    # A lane between the true cut and the rounded one: judged differently iff the compare
    # happens in the narrow dtype.
    edge = (cut + rounded) / 2.0
    golden = torch.tensor([dynamic_range, edge], dtype=dtype)
    result = golden.clone()
    result[1] = float(_step_up(float(golden[1]), dtype, steps=4)[0])
    assert float(golden[1]) == rounded, "the lane has to sit on the rounded edge"

    narrow = within_ulp(golden, result, max_ulp=0, near_zero_atol=1.0)[0]
    wide = within_ulp(
        golden.to(torch.float32),
        result.to(torch.float32),
        max_ulp=0,
        near_zero_atol=1.0,
    )[0]
    assert narrow == wide, (
        f"{dtype} disagreed with float32 for a lane on the rounded band edge "
        f"(cut {cut!r}, rounded {rounded!r}, lane {edge!r})"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_the_absolute_near_zero_cut_is_compared_in_float32_not_the_tensor_dtype(dtype):
    """The other half of the same rule, and the one the relative case cannot reach.

    ``absolute_cut = near_zero_atol / near_zero_fraction`` bounds the band from above so
    a floor cannot follow a wide dynamic range out to a magnitude it was never measured
    at. Its edge is ``<=``, so rounding it *widens* the band -- the opposite direction
    from the relative ``<`` edge, which is why it needs its own parameters: the test
    above uses ``near_zero_atol=1.0``, whose cut is exactly 100.0 and representable in
    both dtypes, so the ``<=`` edge was never probed.

    ``near_zero_atol=0.012`` puts the cut at 1.2, which both dtypes round *up*. A lane at
    that rounded value is inside the band for a narrow compare (``rounded <= rounded``)
    and outside it for fp32 (``1.203125 <= 1.2`` is false), so the floor rescues it in
    one and not the other.
    """
    near_zero_atol = 0.012
    absolute_cut = near_zero_atol / NEAR_ZERO_FRACTION
    rounded = float(torch.tensor(absolute_cut, dtype=dtype))
    assert rounded > absolute_cut, "the cut must round up, or the <= edge does not move"

    # Far enough below the relative cut (1% of 1000.0 = 10.0) that the absolute edge is
    # the only one deciding, and 1000.0 is exact in both dtypes so it moves nothing.
    dynamic_range = 1000.0
    assert rounded < NEAR_ZERO_FRACTION * dynamic_range
    golden = torch.tensor([dynamic_range, rounded], dtype=dtype)
    result = golden.clone()
    # One step out of budget, and well inside the floor -- so band membership alone
    # decides the verdict.
    result[1] = float(_step_up(rounded, dtype, steps=1)[0])
    assert float(result[1]) - rounded <= near_zero_atol
    assert int(ulp_distance(golden, result)[1]) == 1

    narrow = within_ulp(golden, result, max_ulp=0, near_zero_atol=near_zero_atol)[0]
    wide = within_ulp(
        golden.to(torch.float32),
        result.to(torch.float32),
        max_ulp=0,
        near_zero_atol=near_zero_atol,
    )[0]
    assert narrow == wide, (
        f"{dtype} disagreed with float32 for a lane on the rounded absolute cut "
        f"(cut {absolute_cut!r}, rounded {rounded!r})"
    )
    # The lane is outside the band on both sides of the comparison, so the step fails.
    assert not wide
# ─────────────────────────────────────────────────────────────────────────────
# Integers are not ULP territory
#
# A step count says "how many representable values apart". For an integer format the
# answer is always the arithmetic difference, the values are exact, and the only sensible
# verdict is bit equality — so ULP is not a weaker gate there, it is a meaningless one.
# Every entry point must refuse rather than compute something plausible.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fmt", INTEGER_FORMATS, ids=lambda f: f.name)
def test_no_integer_format_is_ulp_gateable(fmt):
    assert not has_ulp_gate(fmt)
    assert fmt not in ULP_FORMATS
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="no per-element ULP"
    ):
        ulp_dtype(fmt)


@pytest.mark.parametrize("dtype", TORCH_INT_DTYPES, ids=str)
def test_the_metric_refuses_every_integer_tensor_dtype(dtype):
    """Including the containers the float bit arithmetic uses internally: ``torch.int16``
    is how a bfloat16's bits are read, which must not make an int16 *tensor* measurable.
    """
    values = torch.ones(4, dtype=dtype)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="unsupported dtype"
    ):
        ulp_distance(values, values.clone())


def test_the_bit_containers_are_not_measurable_dtypes():
    """``_ULP_DTYPES`` is keyed on the float dtypes only. Its *values* name integer dtypes
    because that is how the bits are viewed; a lookup by one of those must miss."""
    from helpers.ulp import _ULP_DTYPES

    assert set(_ULP_DTYPES) == set(FLOAT_DTYPES)
    for dtype in TORCH_INT_DTYPES:
        assert dtype not in _ULP_DTYPES


def test_the_integer_format_list_comes_from_the_enum_not_from_format_dict():
    """``format_dict`` omits ``Bfp8`` and both ``MxFp4_2x`` variants and gives the
    ``MxInt*`` formats a bfloat16 proxy, so deriving the integer set through it would
    silently miss a format added without an entry — or given a float proxy.
    ``DataFormat.is_integer()`` is the authority, and this pins the two agreeing today so
    a divergence is a test failure rather than a quiet coverage hole."""
    assert set(INTEGER_FORMATS) == {f for f in DataFormat if f.is_integer()}
    assert (
        INTEGER_FORMATS
    ), "the derivation has gone empty; every test using it is vacuous"

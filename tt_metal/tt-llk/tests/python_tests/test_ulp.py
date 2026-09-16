# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the integer ULP metric in ``helpers/ulp.py``.

No kernel, no device: the metric is pure bit arithmetic, and it is the thing a future
accuracy budget will be compared against, so every property the gate relies on is pinned
here rather than discovered on hardware.

The tests are grouped by the property they defend, because each one exists to stop a
specific way the metric could look right and be wrong:

* *unit spacing* — one representable step must read as exactly 1 everywhere, including
  across a power of two, where the fractional ``|err| / ulp(golden)`` form returns 0.5 or
  2.0 and a "1 ULP" budget becomes ill-defined. That contrast is asserted directly.
* *the zero neighbourhood* — under the SFPU's DAZ+FTZ the subnormal band is not there, so
  the smallest normal is one step from zero and both signed zeros are the same value. Get
  the compaction wrong in one half and every sign-crossing distance is inflated by
  ``2**mantissa_bits - 1`` while same-sign distances stay correct, which is exactly the
  shape of bug that survives a casual read.
* *the non-finites* — ``Inf`` has a rank and ``NaN`` does not. The sentinel for "no rank"
  is negative, so a caller comparing it against a budget would pass; the ``within_ulp``
  tests exist to prove the composite verdict does not.
"""

import math

import pytest
import torch
from helpers.accuracy_metrics import local_ulp
from helpers.format_config import DataFormat
from helpers.ulp import (
    MAX_MEANINGFUL_ULP,
    ULP_FORMATS,
    UNMEASURABLE,
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

# Mantissa bits after the implicit leading 1, i.e. what sets the size of the subnormal
# band that the flush has to compact away.
MANTISSA_BITS = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23}


def _t(values, dtype):
    return torch.tensor(values, dtype=dtype)


def _step_up(value, dtype, steps=1):
    """*value* moved *steps* representable values toward +inf, in *dtype*."""
    out = _t([value], dtype)
    for _ in range(steps):
        out = torch.nextafter(out, _t([float("inf")], dtype))
    return out


def _step_down(value, dtype, steps=1):
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
    assert int(ulp_distance(golden, _step_up(1.0, dtype, steps))[0]) == steps


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("boundary", [0.5, 1.0, 2.0, 256.0])
def test_power_of_two_boundary_is_one_step_in_both_directions(dtype, boundary):
    """The whole reason the gate counts steps instead of dividing by ``ulp(golden)``."""
    golden = _t([boundary], dtype)
    assert int(ulp_distance(golden, _step_down(boundary, dtype))[0]) == 1
    assert int(ulp_distance(golden, _step_up(boundary, dtype))[0]) == 1


def test_fractional_metric_disagrees_at_a_power_of_two():
    """Pins the motivation: the same single step is 0.5 and 2.0 under the old metric."""
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
    assert int(ulp_distance(zero, _t([tiny], dtype))[0]) == 1
    assert int(ulp_distance(zero, _t([-tiny], dtype))[0]) == 1
    assert int(ulp_distance(_t([-tiny], dtype), _t([tiny], dtype))[0]) == 2
    assert int(ulp_distance(_t([-tiny], dtype), _step_up(tiny, dtype))[0]) == 3


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
def test_every_subnormal_is_zero_steps_from_zero_when_flushed(dtype):
    """A flushed hardware result next to a subnormal golden is a 0-step agreement in the
    SFPU's own number system, not a ``2**mantissa_bits`` error."""
    tiny = torch.finfo(dtype).tiny
    subnormals = _t(
        [tiny / 2, tiny / 4, -tiny / 2, torch.finfo(dtype).smallest_normal / 8], dtype
    )
    assert torch.all(subnormals.abs() < tiny)  # still subnormal after the cast
    assert torch.all(subnormals != 0)
    zeros = torch.zeros_like(subnormals)
    assert ulp_distance(zeros, subnormals).tolist() == [0, 0, 0, 0]


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
    golden = _t([1.0], dtype)
    at_budget = _step_up(1.0, dtype, 3)
    over_budget = _step_up(1.0, dtype, 4)
    assert within_ulp(golden, at_budget, 3)[0]
    assert not within_ulp(golden, over_budget, 3)[0]


def test_within_ulp_does_not_pass_on_the_unmeasurable_sentinel():
    """The footgun the composite exists to remove: ``-1 <= max_ulp`` for every budget, so
    a caller gating on the raw distance would call a missing NaN a pass."""
    golden = torch.tensor([1.0, float("nan")], dtype=torch.float32)
    result = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assert ulp_distance(golden, result)[1] == UNMEASURABLE
    assert bool((ulp_distance(golden, result) <= 0).all())  # the naive gate would pass
    ok, message = within_ulp(golden, result, 0)
    assert not ok
    assert "non-finite disagreement" in message


def test_within_ulp_passes_when_both_sides_are_nan():
    golden = torch.tensor([1.0, float("nan")], dtype=torch.float32)
    result = torch.tensor([1.0, float("nan")], dtype=torch.float32)
    ok, message = within_ulp(golden, result, 0)
    assert ok
    assert "1 unmeasurable" in message


def test_within_ulp_mask_excludes_lanes_already_settled():
    golden = torch.tensor([1.0, 1.0], dtype=torch.float32)
    result = torch.tensor([1.0, 1000.0], dtype=torch.float32)
    assert not within_ulp(golden, result, 1)[0]
    mask = torch.tensor([True, False])
    assert within_ulp(golden, result, 1, mask=mask)[0]


def test_within_ulp_passes_when_every_lane_is_masked_out():
    golden = torch.tensor([1.0, 1.0], dtype=torch.float32)
    result = torch.tensor([5.0, 1000.0], dtype=torch.float32)
    ok, message = within_ulp(golden, result, 0, mask=torch.tensor([False, False]))
    assert ok
    assert "no measurable lane" in message


def test_within_ulp_reports_a_shape_mismatch_instead_of_raising():
    ok, message = within_ulp(
        torch.zeros(4, dtype=torch.float32), torch.zeros(5, dtype=torch.float32), 1
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
    result[2, 1] = _step_up(1.0, torch.bfloat16, 5)[0]
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


def test_ulp_stats_on_an_all_unmeasurable_tensor():
    stats = ulp_stats(torch.full((4,), UNMEASURABLE, dtype=torch.int64))
    assert stats["lanes"] == 0
    assert stats["unmeasurable"] == 4
    assert stats["worst_index"] is None


def test_ulp_failure_message_names_the_point_and_the_step():
    golden = torch.ones(8, dtype=torch.bfloat16)
    result = golden.clone()
    result[5] = _step_up(1.0, torch.bfloat16, 7)[0]
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
        DataFormat.Bfp8_b,
        DataFormat.Bfp4_b,
        DataFormat.MxFp8P,
        DataFormat.MxFp4,
        DataFormat.Int32,
        DataFormat.Tf32,
    ],
)
def test_ulp_dtype_rejects_formats_without_a_per_element_ulp(fmt):
    """Rejected, not silently redirected: a block float's spacing is set by an exponent
    shared across 16 elements, and ``Tf32`` is held in an fp32 container whose lattice is
    not its own. A caller must not be able to think it has a gate it does not have."""
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="no per-element ULP"
    ):
        ulp_dtype(fmt)


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
    ok, message = within_ulp(values, values.clone(), 0)
    assert ok
    assert "100.0% exact" in message


def test_a_non_contiguous_input_is_measured_correctly():
    """``view()`` on the bit pattern needs a contiguous buffer; a transposed tile is the
    ordinary case in this harness, so the copy must happen inside the metric."""
    golden = torch.ones(4, 6, dtype=torch.bfloat16).t()
    result = golden.clone()
    assert not golden.is_contiguous()
    assert int(ulp_distance(golden, result).max()) == 0

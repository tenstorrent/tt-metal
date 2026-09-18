# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for accuracy/emit_budget.py.

No kernel, no device: the script reads a DataFrame and prints registry entries, so every
rule in it is testable against a few synthetic rows. That matters more here than for most
tooling, because the output of this script is *checked in as the gate*. A bug in the budget
formula does not produce a crash, it produces a number that looks measured and gates the
wrong thing — and the comment next to it will say it came from hardware.

The rules under test are the four refusals in the module docstring, plus the arithmetic
that turns a distribution into one integer.
"""

import math

import pandas as pd
import pytest
import torch
from accuracy.emit_budget import (
    DEFAULT_HEADROOM,
    DEFAULT_PERCENTILE,
    MANTISSA_BITS,
    NEAR_ZERO_MAX_SHARE,
    CellMeasurement,
    EmittedKey,
    _collapse,
    _to_enum_flag,
    agreement_bits,
    measure_all,
    measure_cell,
    render,
    render_skipped,
)
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
)
from helpers.sfpu_accuracy_budget import (
    DEFAULT,
    AccuracyContract,
    BudgetKey,
    Metric,
    budget_table,
)
from helpers.ulp import MAX_MEANINGFUL_ULP, NEAR_ZERO_FRACTION, ulp_dtype


def _cell(**kwargs) -> CellMeasurement:
    """A measurement with the all-lane view mirroring the bulk view by default.

    A test that cares about the two diverging sets ``all_max_ulp`` explicitly; everything
    else describes a cell with no near-zero lanes, where they are the same thing.
    """
    defaults = dict(
        op=MathOperation.Tanh,
        input_format=DataFormat.Float32,
        output_format=DataFormat.Float16_b,
        approx_mode=ApproximationMode.No,
        dest_acc=DestAccumulation.No,
        # Both modes by default: Tanh is not fast-mode capable, but a test that swaps in
        # Rsqrt/Sqrt needs both present or the cell is ungateable by design.
        fast_modes=(FastMode.No, FastMode.Yes),
        points=1000,
        max_ulp=1,
        percentile_ulp=1.0,
        exact_fraction=0.9,
        nonfinite_disagreements=0,
        unmeasurable=0,
        near_zero_points=0,
        near_zero_max_ulp=0,
        near_zero_max_abs_err=0.0,
        all_max_ulp=None,
        all_percentile_ulp=None,
        all_exact_fraction=None,
    )
    defaults.update(kwargs)
    if defaults["all_max_ulp"] is None:
        defaults["all_max_ulp"] = defaults["max_ulp"]
    if defaults["all_percentile_ulp"] is None:
        defaults["all_percentile_ulp"] = defaults["percentile_ulp"]
    if defaults["all_exact_fraction"] is None:
        defaults["all_exact_fraction"] = defaults["exact_fraction"]
    return CellMeasurement(**defaults)


def _rows(golden, hardware) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "golden_result": list(golden),
            "hardware_result": list(hardware),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Making two formats comparable
# ─────────────────────────────────────────────────────────────────────────────


def test_the_same_accuracy_reads_the_same_in_any_format():
    """The figure that stops a five-figure float32 budget from reading as a loose gate.

    One bfloat16 step and 65536 float32 steps are the same physical accuracy, because
    float32 counts 2**16 finer steps over the same interval.
    """
    assert agreement_bits(1, DataFormat.Float16_b) == pytest.approx(7.0)
    assert agreement_bits(65536, DataFormat.Float32) == pytest.approx(7.0)
    assert agreement_bits(1, DataFormat.Float16) == pytest.approx(10.0)


def test_an_exact_result_agrees_to_the_whole_mantissa():
    for fmt in (DataFormat.Float32, DataFormat.Float16, DataFormat.Float16_b):
        expected = math.log2(
            MAX_MEANINGFUL_ULP[
                {
                    DataFormat.Float32: torch.float32,
                    DataFormat.Float16: torch.float16,
                    DataFormat.Float16_b: torch.bfloat16,
                }[fmt]
            ]
        )
        assert agreement_bits(0, fmt) == pytest.approx(expected)


def test_a_disagreement_past_the_whole_mantissa_goes_negative():
    """Which is the signal that the op belongs on the tolerance metric, not a budget."""
    assert agreement_bits(1 << 20, DataFormat.Float16_b) < 0


# ─────────────────────────────────────────────────────────────────────────────
# The budget formula
# ─────────────────────────────────────────────────────────────────────────────


def test_a_budget_is_never_below_the_measured_maximum():
    """The gate is a max over lanes, so a budget under the measured max fails on the
    first run. The percentile is a floor, not the answer.

    On Float32, whose usable ceiling is wide enough that the arithmetic rather than the
    ceiling is what is under test here."""
    cell = _cell(output_format=DataFormat.Float32, max_ulp=40, percentile_ulp=4.0)
    assert cell.budget(headroom=1.25) == 40


def test_the_percentile_floor_lifts_a_tight_distribution():
    """Where the distribution is tight, the budget gets headroom above it rather than
    being pinned to the single worst observed lane."""
    cell = _cell(output_format=DataFormat.Float32, max_ulp=8, percentile_ulp=8.0)
    assert cell.budget(headroom=1.25) == 10


def test_headroom_of_one_pins_the_budget_to_the_measurement():
    cell = _cell(output_format=DataFormat.Float32, max_ulp=8, percentile_ulp=8.0)
    assert cell.budget(headroom=1.0) == 8


# ─────────────────────────────────────────────────────────────────────────────
# The near-zero floor, and when it must not be emitted
# ─────────────────────────────────────────────────────────────────────────────


def test_no_floor_when_the_near_zero_lanes_are_already_in_budget():
    """A floor for lanes the budget already covers would be inert, so `_floor` returns
    early on `near_zero_max_ulp <= _budget(...)`.

    The cell has to be built so that rule is the *only* thing producing the ``None``.
    With `_cell`'s defaults it was not: `near_zero_max_abs_err` stays 0.0, so deleting
    the rule still returned ``None`` three lines later at `if floor <= 0`, and the
    default `Float16_b` output makes `_budget(8, 8.0, 1.25) = 10` exceed
    `usable_budget_ceiling(Float16_b) = 6.4`, so the whole resolution came back
    `(None, None)` on the ceiling branch instead. Float32 and a positive
    `near_zero_max_abs_err` remove both, and the mutation case below -- the same cell
    with the near-zero lanes *out* of budget -- shows the rule is what decides.
    """
    common = dict(
        output_format=DataFormat.Float32,
        max_ulp=8,
        percentile_ulp=8.0,
        near_zero_points=10,
        near_zero_max_abs_err=1e-6,
    )
    in_budget = _cell(near_zero_max_ulp=3, **common)
    assert in_budget.budget(DEFAULT_HEADROOM) is not None  # not a ceiling refusal
    assert in_budget.near_zero_atol(DEFAULT_HEADROOM) is None

    out_of_budget = _cell(near_zero_max_ulp=99999, **common)
    assert out_of_budget.near_zero_atol(DEFAULT_HEADROOM) == pytest.approx(
        1e-6 * DEFAULT_HEADROOM, rel=1e-2
    )


def test_a_floor_appears_when_a_few_near_zero_lanes_blow_the_budget():
    """The gelu(-4.18) / erfinv(0.0005) case: the hardware returns exactly 0 against a
    small non-zero reference, which is a five-figure step count and says nothing about the
    kernel elsewhere."""
    cell = _cell(
        max_ulp=1,
        percentile_ulp=1.0,
        points=1000,
        near_zero_points=20,
        near_zero_max_ulp=14337,
        near_zero_max_abs_err=6.1e-05,
    )
    floor = cell.near_zero_atol(headroom=1.25)
    assert floor is not None
    assert floor == pytest.approx(6.1e-05 * 1.25, rel=1e-2)


def test_no_floor_when_it_would_cover_most_of_the_lanes():
    """The exp case. "Below 1% of the dynamic range" assumes a roughly linear-scale
    tensor; over a domain whose output spans decades it swallows almost every lane, and a
    floor covering the majority is not a floor, it is the gate — the magnitude-blind gate
    the step count exists to replace."""
    cell = _cell(
        max_ulp=1,
        percentile_ulp=1.0,
        points=6144,
        near_zero_points=6036,
        near_zero_max_ulp=32767,
        near_zero_max_abs_err=6.4e29,
        all_max_ulp=32767,
        all_percentile_ulp=32767.0,
    )
    assert cell.near_zero_points / cell.measurable_points > NEAR_ZERO_MAX_SHARE
    assert cell.near_zero_atol(DEFAULT_HEADROOM) is None


def test_the_share_guard_is_a_boundary_not_a_cliff():
    below = _cell(
        points=1000,
        near_zero_points=int(1000 * NEAR_ZERO_MAX_SHARE) - 1,
        near_zero_max_ulp=99999,
        near_zero_max_abs_err=1e-4,
    )
    above = _cell(
        points=1000,
        near_zero_points=int(1000 * NEAR_ZERO_MAX_SHARE) + 1,
        near_zero_max_ulp=99999,
        near_zero_max_abs_err=1e-4,
    )
    assert below.near_zero_atol(DEFAULT_HEADROOM) is not None
    assert above.near_zero_atol(DEFAULT_HEADROOM) is None


def test_unmeasurable_lanes_do_not_count_toward_the_share():
    """The denominator is the *measurable* lanes, so a cell full of NaN goldens cannot
    hide a near-zero majority behind the lanes that were never compared.

    That direction, not the other way round: ``measurable_points <= points`` always, so
    swapping in ``points`` can only *shrink* the computed share -- 60/1000 = 6% against
    60/100 = 60% here -- and would make a genuine majority look like a minority, which
    is the reading that suppresses the floor's own guard. The share guard is what keeps
    an absolute floor from becoming the gate, so the failure that matters is a majority
    read as a minority.
    """
    cell = _cell(
        points=1000,
        unmeasurable=900,
        near_zero_points=60,
        near_zero_max_ulp=99999,
        near_zero_max_abs_err=1e-6,
        output_format=DataFormat.Float32,
    )
    assert cell.measurable_points == 100
    assert cell.near_zero_points / cell.measurable_points > NEAR_ZERO_MAX_SHARE
    # ...and the wrong denominator would read the same cell as a 6% minority.
    assert cell.near_zero_points / cell.points < NEAR_ZERO_MAX_SHARE
    # The guard the ratio feeds, exercised rather than re-derived: these near-zero lanes
    # are far out of budget and would otherwise earn a floor.
    assert cell.near_zero_atol(DEFAULT_HEADROOM) is None


# ─────────────────────────────────────────────────────────────────────────────
# Measuring a cell: the gate's metric, not the sweep's diagnostic
# ─────────────────────────────────────────────────────────────────────────────


def _measure(
    golden,
    hardware,
    fmt=DataFormat.Float16_b,
    percentile=DEFAULT_PERCENTILE,
    fraction=NEAR_ZERO_FRACTION,
):
    return measure_cell(
        _rows(golden, hardware),
        MathOperation.Tanh,
        DataFormat.Float32,
        fmt,
        ApproximationMode.No,
        DestAccumulation.No,
        FastMode.No,
        percentile,
        fraction,
    )


def test_a_cell_of_identical_values_measures_zero():
    values = [1.0, 2.0, 4.0, 8.0] * 64
    cell = _measure(values, values)
    assert cell.max_ulp == 0
    assert cell.exact_fraction == pytest.approx(1.0)
    assert cell.gateable


def test_the_metric_is_the_integer_step_count_not_the_fractional_one():
    """One representable step below a power of two is 1, where the fractional
    ``|err| / ulp(golden)`` the sweep also records would call it 0.5."""
    golden = [1.0] * 128
    # 1 - 2**-8, the bfloat16 value one step below 1.0 and exactly representable.
    below = 0.99609375
    assert float(torch.tensor(below, dtype=torch.bfloat16)) == below
    cell = _measure(golden, [below] * 128)
    assert cell.max_ulp == 1


def test_a_non_finite_disagreement_makes_a_cell_ungateable():
    """No budget can pass such a cell — the gate requires non-finites to agree
    positionally before it looks at any step count — so emitting a number for it would be
    emitting a lie."""
    cell = _measure([1.0, 2.0, 3.0], [1.0, float("inf"), 3.0])
    assert cell.nonfinite_disagreements == 1
    assert not cell.gateable


def test_matching_non_finites_are_unmeasurable_but_not_a_disagreement():
    cell = _measure([1.0, float("nan")], [1.0, float("nan")])
    assert cell.nonfinite_disagreements == 0
    assert cell.unmeasurable == 1
    assert cell.gateable


def test_the_near_zero_split_keeps_a_tiny_golden_out_of_the_bulk():
    """A lane at 1e-8 against a dynamic range of 100 is below the 1% cut, so its enormous
    step count lands in the near-zero bucket and does not set the budget."""
    golden = [100.0] * 99 + [1e-8]
    hardware = [100.0] * 99 + [0.0]
    cell = _measure(golden, hardware, fmt=DataFormat.Float32)
    assert cell.max_ulp == 0  # the bulk is exact
    assert cell.near_zero_points == 1
    assert cell.near_zero_max_ulp > 1000
    assert cell.near_zero_max_abs_err == pytest.approx(1e-8)


def test_an_all_zero_golden_puts_every_lane_in_the_near_zero_bucket():
    cell = _measure([0.0] * 8, [0.0] * 8)
    assert cell.near_zero_points == 8


# ─────────────────────────────────────────────────────────────────────────────
# Collapsing keys
# ─────────────────────────────────────────────────────────────────────────────


def test_dest_acc_collapses_when_both_settings_agree():
    cells = [
        _cell(dest_acc=DestAccumulation.No, max_ulp=1, percentile_ulp=1.0),
        _cell(dest_acc=DestAccumulation.Yes, max_ulp=1, percentile_ulp=1.0),
    ]
    keys = _collapse(cells, DEFAULT_HEADROOM, DataFormat.Float32)
    assert len(keys) == 1
    assert keys[0].dest_acc is None
    assert "dest_acc" not in keys[0].key_source()


def test_dest_acc_stays_in_the_key_when_the_settings_disagree():
    """A key that keeps a dimension is itself the statement that the dimension mattered."""
    cells = [
        _cell(dest_acc=DestAccumulation.No, max_ulp=1, percentile_ulp=1.0),
        _cell(dest_acc=DestAccumulation.Yes, max_ulp=40, percentile_ulp=40.0),
    ]
    keys = _collapse(cells, DEFAULT_HEADROOM, DataFormat.Float32)
    assert len(keys) == 2
    assert all(k.dest_acc is not None for k in keys)


def test_neither_format_dimension_collapses_even_when_every_cell_agrees():
    """Both format dimensions stay pinned, for the same reason, and it is not symmetry.

    A key without the **input** format would extend the budget to input paths the sweep
    never measured -- the functional suite runs ``Bfp8_b`` inputs and the accuracy sweep
    does not -- and a much coarser path silently inheriting a budget is the exact failure
    this dimension was added to stop: 36 of 44 functional failures under the first cut of
    the table were ``Bfp8_b``-input variants.

    A key without the **output** format has the same hole in the other direction. A
    default run measures fp32 and bf16 only, so collapsing on those two agreeing would
    produce a wildcard matching the unmeasured Float16 and Bfp8_b outputs, and the whole
    enrolment model rests on absent formats falling back to tolerance on their own. It
    also left the agreement-bits figure in the comment picking whichever output format
    came first out of a ``sort=False`` groupby, so a merged fp32+bf16 key printed either
    ~23 or ~7 mantissa bits by row order.

    approx_mode and dest_acc do still collapse: they are measured on every run.
    """
    cells = [
        _cell(
            output_format=fmt,
            approx_mode=approx,
            dest_acc=dest,
            max_ulp=0,
            percentile_ulp=0.0,
        )
        for fmt in (DataFormat.Float32, DataFormat.Float16_b)
        for approx in ApproximationMode
        for dest in DestAccumulation
    ]
    keys = _collapse(cells, DEFAULT_HEADROOM, DataFormat.Float32)
    # One key per output format, not one key overall.
    assert len(keys) == 2
    sources = [k.key_source() for k in keys]
    assert sources == [
        "BudgetKey(input_format=DataFormat.Float32, output_format=DataFormat.Float32)",
        "BudgetKey(input_format=DataFormat.Float32, "
        "output_format=DataFormat.Float16_b)",
    ]
    for source in sources:
        for collapsed in ("approx_mode", "dest_acc"):
            assert collapsed not in source
    # And each key reports its own format's bits, not an arbitrary one.
    for key in keys:
        assert len({c.output_format for c in key.cells}) == 1


def test_a_budget_past_the_ceiling_is_emitted_as_a_tolerance_contract():
    """The guard against "the sweep said 15616, so the budget is 15616"."""
    cells = [
        _cell(
            output_format=DataFormat.Float16_b,
            max_ulp=20000,
            percentile_ulp=20000.0,
            dest_acc=dest,
        )
        for dest in DestAccumulation
    ]
    keys = _collapse(cells, DEFAULT_HEADROOM, DataFormat.Float32)
    assert all(k.budget is None for k in keys)
    assert all("Metric.TOLERANCE" in k.contract_source() for k in keys)


def test_an_ungateable_cell_gets_no_budget():
    cells = [
        _cell(nonfinite_disagreements=3, dest_acc=dest) for dest in DestAccumulation
    ]
    keys = _collapse(cells, DEFAULT_HEADROOM, DataFormat.Float32)
    assert all(k.budget is None for k in keys)


def test_an_emitted_contract_carries_the_floor_when_there_is_one():
    cells = [
        _cell(
            dest_acc=dest,
            max_ulp=1,
            percentile_ulp=1.0,
            points=1000,
            near_zero_points=20,
            near_zero_max_ulp=99999,
            near_zero_max_abs_err=1e-4,
        )
        for dest in DestAccumulation
    ]
    keys = _collapse(cells, DEFAULT_HEADROOM, DataFormat.Float32)
    assert len(keys) == 1
    source = keys[0].contract_source()
    assert "max_ulp=" in source and "near_zero_atol=" in source


def test_a_cell_with_a_floor_does_not_collapse_into_one_without():
    cells = [
        _cell(dest_acc=DestAccumulation.No, max_ulp=1, percentile_ulp=1.0),
        _cell(
            dest_acc=DestAccumulation.Yes,
            max_ulp=1,
            percentile_ulp=1.0,
            points=1000,
            near_zero_points=20,
            near_zero_max_ulp=99999,
            near_zero_max_abs_err=1e-4,
        ),
    ]
    keys = _collapse(cells, DEFAULT_HEADROOM, DataFormat.Float32)
    assert len(keys) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Reading the sweep's own encoding
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0", ApproximationMode.No),
        ("1", ApproximationMode.Yes),
        (0, ApproximationMode.No),
        (1, ApproximationMode.Yes),
        ("false", ApproximationMode.No),
        ("True", ApproximationMode.Yes),
        (ApproximationMode.Yes, ApproximationMode.Yes),
    ],
)
def test_the_harness_flag_encoding_round_trips(raw, expected):
    assert _to_enum_flag(raw, ApproximationMode) is expected


def test_an_unreadable_flag_raises_rather_than_guessing():
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="cannot read"
    ):
        _to_enum_flag("maybe", DestAccumulation)


def test_every_op_name_in_the_sweep_maps_back_to_a_math_operation():
    """The join between the sweep's lowercase op names and the registry's enum. A miss
    here silently drops an op from enrolment."""
    by_lower = {op.name.lower(): op for op in MathOperation}
    swept = [
        "acosh",
        "asinh",
        "atanh",
        "celu",
        "cos",
        "elu",
        "erfinv",
        "exp",
        "exp2",
        "gelu",
        "hardsigmoid",
        "log",
        "log1p",
        "reciprocal",
        "rsqrt",
        "silu",
        "sin",
        "sqrt",
        "tanh",
    ]
    assert [n for n in swept if n not in by_lower] == []


# ─────────────────────────────────────────────────────────────────────────────
# Budget and floor are decided together
#
# The subtle one, and the reason it is worth a section: a budget measured over the bulk
# lanes is only valid while something else holds the near-zero ones.
# ─────────────────────────────────────────────────────────────────────────────


def test_suppressing_the_floor_widens_the_budget_to_cover_every_lane():
    """The exp/bfloat16/dest_acc=Yes case that caught this on hardware.

    6036 of 6144 points sit under 1% of a dynamic range of 5.5e34, so the share guard
    refuses a floor. The bulk lanes are all exact, and emitting the bulk budget alone gave
    ``max_ulp=0`` with nothing holding the rest — the gate then saw all 6144 lanes, found
    one a step out, and failed a budget the sweep had apparently justified.
    """
    cell = _cell(
        max_ulp=0,
        percentile_ulp=0.0,
        points=6144,
        near_zero_points=6036,
        near_zero_max_ulp=1,
        near_zero_max_abs_err=1e-40,
        all_max_ulp=1,
        all_percentile_ulp=1.0,
    )
    budget, floor = cell.resolve(DEFAULT_HEADROOM)
    assert floor is None, "the share guard should refuse a floor here"
    assert budget >= cell.all_max_ulp, (
        "with no floor the budget has to cover the near-zero lanes too; a bulk-only "
        "budget here was 0 and the gate failed on hardware"
    )
    assert budget > cell.max_ulp, "the bulk view alone would have emitted 0"


def test_keeping_the_floor_keeps_the_budget_on_the_bulk_lanes():
    """The other side of the same decision: a floor that *is* emitted is what lets the
    budget stay tight, which is the entire point of having one."""
    cell = _cell(
        max_ulp=1,
        percentile_ulp=1.0,
        points=1000,
        near_zero_points=20,
        near_zero_max_ulp=14337,
        near_zero_max_abs_err=6.1e-05,
        all_max_ulp=14337,
        all_percentile_ulp=1.0,
    )
    budget, floor = cell.resolve(DEFAULT_HEADROOM)
    assert floor is not None
    assert budget == 2, "the 14337-step near-zero lane must not set the budget"


def test_a_cell_with_no_near_zero_lanes_resolves_the_same_either_way():
    cell = _cell(output_format=DataFormat.Float32, max_ulp=6, percentile_ulp=6.0)
    budget, floor = cell.resolve(DEFAULT_HEADROOM)
    assert floor is None
    assert budget == 8


def test_the_ceiling_is_applied_to_the_resolved_budget_not_the_bulk_one():
    """A cell whose bulk is tiny but whose all-lane view is hopeless still goes to
    tolerance, rather than emitting the flattering bulk number."""
    cell = _cell(
        output_format=DataFormat.Float16_b,
        max_ulp=1,
        percentile_ulp=1.0,
        points=6144,
        near_zero_points=6000,
        near_zero_max_ulp=99999,
        near_zero_max_abs_err=1.0,
        all_max_ulp=99999,
        all_percentile_ulp=99999.0,
    )
    budget, floor = cell.resolve(DEFAULT_HEADROOM)
    assert (budget, floor) == (None, None)


def test_measure_cell_records_both_views():
    golden = [100.0] * 99 + [1e-8]
    hardware = [100.0] * 99 + [0.0]
    cell = _measure(golden, hardware, fmt=DataFormat.Float32)
    assert cell.max_ulp == 0, "the bulk lanes are exact"
    assert cell.all_max_ulp > 1000, "the all-lane view sees the near-zero blow-up"


# ─────────────────────────────────────────────────────────────────────────────
# The path that actually produces the checked-in table
#
# The tests above cover the pure-logic helpers against synthetic cells. These drive
# measure_all() and render() end to end over a synthetic sweep frame, which is the code
# that turns sweep rows into the lines pasted into the registry. It cannot validate the
# real numbers — sweep data is far too large to check in — but it makes a regeneration
# diffable and catches the structural mistakes, which is where this file's bugs have been.
# ─────────────────────────────────────────────────────────────────────────────


def _sweep_frame(rows):
    """A frame shaped like the parquet the accuracy harness writes."""
    return pd.DataFrame(
        [
            {
                "op": op,
                "input_format": in_fmt,
                "output_format": out_fmt,
                "approx_mode": approx,
                "fast_mode": fast,
                "dest_acc": dest,
                "golden_result": golden,
                "hardware_result": hardware,
            }
            for op, in_fmt, out_fmt, approx, fast, dest, golden, hardware in rows
        ]
    )


def _one_cell(
    op="tanh", in_fmt="fp32", out_fmt="bf16", approx="0", dest="0", pairs=None
):
    pairs = pairs or [(1.0, 1.0)] * 32
    return [
        (op, in_fmt, out_fmt, approx, fast, dest, g, h)
        for fast in ("0", "1")
        for g, h in pairs
    ]


def test_render_emits_a_key_that_pins_the_input_format():
    """Never collapsed away, even for a single input format, because a key without it
    would cover input paths the sweep never measured."""
    df = _sweep_frame(_one_cell())
    cells, notes = measure_all(df, None, DEFAULT_PERCENTILE, NEAR_ZERO_FRACTION)
    assert notes == []
    text = render(cells, "wh", DEFAULT_HEADROOM, "2026-01-01")
    assert "MathOperation.Tanh" in text
    assert "input_format=DataFormat.Float32" in text
    assert text.count("MathOperation.") == 1


def test_render_does_not_collapse_a_single_dest_acc_group():
    """A single-``dest_acc`` group makes the equality check trivially true, so collapsing
    on it would drop a pin the sweep never justified removing — and groups really are
    single-dest, since ``main`` passes only gateable cells through."""
    df = _sweep_frame(_one_cell(dest="1"))
    cells, _ = measure_all(df, None, DEFAULT_PERCENTILE, NEAR_ZERO_FRACTION)
    text = render(cells, "wh", DEFAULT_HEADROOM, "2026-01-01")
    assert "dest_acc=DestAccumulation.Yes" in text


def test_render_keeps_an_approx_pin_when_collapsing_the_output_format():
    """Collapsing output format on key *count* alone dropped a pin the approximation stage
    had deliberately kept, which would then gate approx=Yes and the unswept output formats
    on a number measured for neither."""
    rows = _one_cell(out_fmt="bf16", approx="0") + _one_cell(out_fmt="fp32", approx="0")
    cells, _ = measure_all(
        _sweep_frame(rows), None, DEFAULT_PERCENTILE, NEAR_ZERO_FRACTION
    )
    text = render(cells, "wh", DEFAULT_HEADROOM, "2026-01-01")
    assert "approx_mode=ApproximationMode.No" in text


def test_render_refuses_a_budget_looser_than_the_tolerance_it_replaces():
    """``MAX_MEANINGFUL_ULP`` is ~100% relative error, so emitting against it admits
    budgets that gate nothing — and because the gate returns on the ULP verdict, such a
    budget *is* the whole gate. Measured: approximate tanh on fp32 came out at 2,949,120
    steps, ~35% relative error, on an op bounded in (-1, 1)."""
    loose = [(1.0, 1.35)] * 32
    df = _sweep_frame(_one_cell(out_fmt="fp32", pairs=loose))
    cells, _ = measure_all(df, None, DEFAULT_PERCENTILE, NEAR_ZERO_FRACTION)
    text = render(cells, "wh", DEFAULT_HEADROOM, "2026-01-01")
    assert "Metric.TOLERANCE" in text
    assert "stops being tighter than the tolerance it replaces" in text


def test_render_says_when_a_measured_zero_was_floored():
    df = _sweep_frame(_one_cell())
    cells, _ = measure_all(df, None, DEFAULT_PERCENTILE, NEAR_ZERO_FRACTION)
    text = render(cells, "wh", DEFAULT_HEADROOM, "2026-01-01")
    assert "max_ulp=1" in text
    assert "measured 0, floored to 1" in text


def test_render_skipped_names_the_cell_it_refused():
    """A non-finite disagreement gets no budget, and the report has to say which cell."""
    rows = _one_cell(pairs=[(1.0, float("inf"))] * 32)
    cells, _ = measure_all(
        _sweep_frame(rows), None, DEFAULT_PERCENTILE, NEAR_ZERO_FRACTION
    )
    lines = render_skipped(cells)
    assert lines and "Tanh Float32->Float16_b" in lines[0]
    assert "non-finite disagreement" in lines[0]
    # main() passes only the gateable cells to render, which is none of them here.
    assert [c for c in cells if c.gateable] == []


def test_measure_all_reports_an_unknown_op_instead_of_dropping_it():
    df = _sweep_frame(_one_cell(op="not_a_real_op"))
    cells, notes = measure_all(df, None, DEFAULT_PERCENTILE, NEAR_ZERO_FRACTION)
    assert cells == []
    assert any("no MathOperation" in n for n in notes)


def test_render_output_parses_as_python_and_rebuilds_the_contracts():
    """The strongest cheap check on the generated text: it is pasted into a module, so it
    has to be valid Python that evaluates back to the contracts it describes."""
    rows = _one_cell(out_fmt="bf16") + _one_cell(out_fmt="fp32")
    cells, _ = measure_all(
        _sweep_frame(rows), None, DEFAULT_PERCENTILE, NEAR_ZERO_FRACTION
    )
    text = render(cells, "wh", DEFAULT_HEADROOM, "2026-01-01")
    namespace = {
        "MathOperation": MathOperation,
        "DataFormat": DataFormat,
        "ApproximationMode": ApproximationMode,
        "DestAccumulation": DestAccumulation,
        "BudgetKey": BudgetKey,
        "AccuracyContract": AccuracyContract,
        "DEFAULT": DEFAULT,
        "Metric": Metric,
        # The emitter calls budget_table() rather than writing a dict literal, so the
        # generated entries get the duplicate-key refusal too. Evaluating the real one
        # here means this test would fail if the emitter ever produced a repeat.
        "budget_table": budget_table,
    }
    table = eval("{" + text + "}", namespace)  # noqa: S307 - generated, not user input
    assert MathOperation.Tanh in table
    assert "budget_table(" in text, "generated tables must go through the guard"
    for key, contract in table[MathOperation.Tanh].items():
        assert isinstance(key, BudgetKey)
        assert isinstance(contract, AccuracyContract)
        assert key.input_format is DataFormat.Float32


def test_a_budget_looser_than_its_formats_tolerance_is_refused_outright():
    """The same bound as ``usable_budget_ceiling``, applied through ``resolve``: bf16's
    rtol half is worth about 6 steps, so a 40-step bf16 budget is looser than the gate it
    would replace and must not be emitted at all."""
    cell = _cell(output_format=DataFormat.Float16_b, max_ulp=40, percentile_ulp=40.0)
    assert cell.resolve(DEFAULT_HEADROOM) == (None, None)
    # The same measurement on Float32 is comfortably inside its ceiling.
    wide = _cell(output_format=DataFormat.Float32, max_ulp=40, percentile_ulp=40.0)
    assert wide.resolve(DEFAULT_HEADROOM)[0] == 50


def test_the_usable_ceiling_is_the_rtol_half_not_the_whole_mantissa():
    from accuracy.emit_budget import usable_budget_ceiling

    for fmt in (DataFormat.Float32, DataFormat.Float16, DataFormat.Float16_b):
        assert usable_budget_ceiling(fmt) < MAX_MEANINGFUL_ULP[ulp_dtype(fmt)]
    assert usable_budget_ceiling(DataFormat.Float16_b) == pytest.approx(6.4)


# ─────────────────────────────────────────────────────────────────────────────
# Refusals: a cell with no evidence, and a sweep that measured half a dimension
# ─────────────────────────────────────────────────────────────────────────────


def test_a_cell_with_no_measurable_lane_gets_no_budget():
    """A cell of matching NaNs has no non-finite *disagreement*, so it used to read as
    gateable: both maxima came back 0, ``_budget`` floored them to
    ``MIN_MEASURED_BUDGET``, and the table claimed a measured 1-step contract over a cell
    that supplied no finite evidence at all. ``measurable_points``' ``max(..., 1)`` masked
    the state rather than rejecting it."""
    nan = float("nan")
    cell = _measure([nan] * 64, [nan] * 64)
    assert cell.unmeasurable == cell.points
    assert cell.nonfinite_disagreements == 0  # matching NaNs agree positionally
    assert not cell.gateable
    assert "no finite lane was measured" in cell.ungateable_reason
    assert "unmeasurable" in "\n".join(render_skipped([cell]))


def test_a_partial_fast_mode_sweep_is_refused_rather_than_halved():
    """``BudgetKey`` has no fast-mode dimension, so a key derived from one mode would gate
    the other on a number measured for neither. For an op that runs in both modes that is
    a reason to re-run the sweep, not to emit half a measurement."""
    one_mode = _cell(op=MathOperation.Rsqrt, fast_modes=(FastMode.No,))
    assert not one_mode.gateable
    assert "measured only No" in one_mode.ungateable_reason

    both = _cell(op=MathOperation.Rsqrt, fast_modes=(FastMode.No, FastMode.Yes))
    assert both.gateable
    # An op that does not run in both modes is unaffected.
    assert _cell(op=MathOperation.Tanh, fast_modes=(FastMode.No,)).gateable


def test_fast_modes_combine_by_the_worst_of_each_statistic():
    """Not by pooling the rows. A percentile over pooled rows is *not* the max of the
    per-mode percentiles -- the slower mode's tail is diluted by the other mode's rows --
    so the floor would come out under the value one mode needs, and the budget with it.
    """
    from accuracy.emit_budget import _combine_fast_modes

    slow = _cell(
        fast_modes=(FastMode.No,),
        max_ulp=40,
        percentile_ulp=40.0,
        exact_fraction=0.1,
        points=500,
        output_format=DataFormat.Float32,
    )
    fast = _cell(
        fast_modes=(FastMode.Yes,),
        max_ulp=2,
        percentile_ulp=2.0,
        exact_fraction=0.9,
        points=500,
        output_format=DataFormat.Float32,
    )
    combined = _combine_fast_modes([slow, fast])
    assert combined.fast_modes == (FastMode.No, FastMode.Yes)
    assert combined.max_ulp == 40
    assert combined.percentile_ulp == 40.0  # not the pooled ~21
    assert combined.exact_fraction == pytest.approx(0.1)  # the worse of the two
    assert combined.points == 1000
    # And the resolved budget is the one the slow mode needs.
    assert combined.resolve(DEFAULT_HEADROOM)[0] == slow.resolve(DEFAULT_HEADROOM)[0]


# ─────────────────────────────────────────────────────────────────────────────
# The audit trail beside each budget
# ─────────────────────────────────────────────────────────────────────────────


def test_a_floored_budget_keeps_both_of_its_notes():
    """``note`` was assigned rather than appended, so the near-zero clause overwrote the
    "measured 0, floored to 1" one -- and the two are not exclusive. Whenever a floor is
    emitted the reported max is the *bulk* max, so a cell whose bulk lanes are all exact
    gives ``max 0 ULP`` beside ``max_ulp=1`` with no explanation. That is the reading the
    floored-to-1 note was added to fix, and it was dead for every floored key."""
    cell = _cell(
        max_ulp=0,
        percentile_ulp=0.0,
        exact_fraction=1.0,
        all_max_ulp=9000,
        all_percentile_ulp=9000.0,
        near_zero_points=10,
        near_zero_max_ulp=9000,
        near_zero_max_abs_err=1e-4,
    )
    budget, floor = cell.resolve(DEFAULT_HEADROOM)
    assert budget == 1 and floor is not None
    key = EmittedKey(
        input_format=DataFormat.Float32,
        output_format=DataFormat.Float16_b,
        approx_mode=None,
        dest_acc=None,
        budget=budget,
        near_zero_atol=floor,
        cells=(cell,),
    )
    comment = key.comment("wh", "2026-09-17", DEFAULT_PERCENTILE)
    assert "measured 0, floored to 1" in comment
    assert "near-zero pts reach 9000 steps" in comment


def test_the_comment_reports_the_percentile_it_was_given():
    """``--percentile`` controls the budget, so labelling the figure ``p99.9`` regardless
    emitted valid Python beside a false measurement claim."""
    cell = _cell(max_ulp=3, percentile_ulp=3.0)
    key = EmittedKey(
        input_format=DataFormat.Float32,
        output_format=DataFormat.Float16_b,
        approx_mode=None,
        dest_acc=None,
        budget=4,
        near_zero_atol=None,
        cells=(cell,),
    )
    assert "p95 " in key.comment("wh", "2026-09-17", 95.0)
    assert "p99.9 " in key.comment("wh", "2026-09-17", 99.9)


def test_the_exact_fraction_comes_from_the_same_lanes_as_the_maximum():
    """With no floor emitted the reported max is the all-lane one, so the exact fraction
    has to be too. Reporting the bulk fraction beside it produced comments like Exp2's
    "max N ULP, 100% exact"."""
    cell = _cell(
        max_ulp=0,
        percentile_ulp=0.0,
        exact_fraction=1.0,  # the bulk lanes really are all exact
        all_max_ulp=5,
        all_percentile_ulp=5.0,
        all_exact_fraction=0.4,
        near_zero_points=0,  # so no floor is emitted and the all-lane view is used
    )
    key = EmittedKey(
        input_format=DataFormat.Float32,
        output_format=DataFormat.Float16_b,
        approx_mode=None,
        dest_acc=None,
        budget=6,
        near_zero_atol=None,
        cells=(cell,),
    )
    comment = key.comment("wh", "2026-09-17", DEFAULT_PERCENTILE)
    assert "max 5 ULP" in comment
    assert "40% exact" in comment
    assert "100% exact" not in comment


def test_the_floor_never_rounds_below_the_error_it_was_measured_from():
    """The gate rescues a lane only when ``absolute_error <= near_zero_atol``, so a floor
    rounded down by ``.3g`` rejects the very lane it came from. ``_budget`` makes the same
    refusal with ``max()`` and ``ceil()``."""
    measured = 1.2345678e-4
    cell = _cell(
        max_ulp=1,
        percentile_ulp=1.0,
        near_zero_points=10,
        near_zero_max_ulp=99999,
        near_zero_max_abs_err=measured,
    )
    floor = cell.resolve(1.0)[1]  # headroom 1.0 is where rounding could bite
    assert floor is not None
    assert floor >= measured


# ─────────────────────────────────────────────────────────────────────────────
# Provenance the emitter must not fake
# ─────────────────────────────────────────────────────────────────────────────


def test_a_mixed_parquet_and_csv_directory_is_refused(tmp_path):
    """``merge_shards()`` rewrites only the ops from the current run and leaves older
    per-op files in place, so preferring parquet let one stale file hide every fresh csv
    -- and the emitted budgets would carry today's stamp over another run's measurement.
    There is no run-level manifest to tell them apart."""
    from accuracy.emit_budget import load_sweep

    arch_dir = tmp_path / "wh"
    arch_dir.mkdir()
    _rows([1.0], [1.0]).to_parquet(arch_dir / "tanh.parquet")
    _rows([1.0], [1.0]).to_csv(arch_dir / "gelu.csv", index=False)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        SystemExit, match="both parquet and csv"
    ):
        load_sweep(tmp_path, "wh")


def test_only_the_measured_architecture_can_be_emitted():
    """``EmittedKey`` has no arch dimension and ``accuracy_contract()`` downgrades every
    architecture but ``MEASURED_ARCH`` before it resolves a key, so text emitted from
    another arch's sweep would be plausible, measured and silently inert."""
    from accuracy.emit_budget import ARCH_ABBR, EMITTABLE_ARCH, main
    from helpers.sfpu_accuracy_budget import MEASURED_ARCH

    # Against the registry's own constant, not against a literal: `EMITTABLE_ARCH == "wh"`
    # was asserting the value against itself, and the two were hand-maintained in
    # different naming schemes with nothing tying them together.
    assert EMITTABLE_ARCH == ARCH_ABBR[MEASURED_ARCH]
    assert set(ARCH_ABBR) == set(ChipArchitecture), "every arch needs a sweep directory"
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        SystemExit, match="cannot be emitted"
    ):
        main(["--arch", "bh"])


def test_a_non_default_near_zero_fraction_is_rejected_or_stamped(capsys, tmp_path):
    """The gate always splits at ``NEAR_ZERO_FRACTION``, and ``AccuracyContract`` has
    nowhere to carry a different one, so a regenerated table would silently omit lanes
    the gate still charges against its ``max_ulp``.

    Out of range is refused; in range but non-default is *stamped*, and the banner is
    the only thing standing between a ``--near-zero-fraction 0.005`` regeneration and a
    table of ``max_ulp`` values quietly measured against a different split. The
    "_or_stamped" half of this name went unexercised -- along with both fixtures -- until
    the second half below.
    """
    from accuracy.emit_budget import main

    for bad in ("0", "1", "-0.5"):
        with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
            SystemExit, match="must be in"
        ):
            main(["--near-zero-fraction", bad])

    arch_dir = tmp_path / "wh"
    arch_dir.mkdir()
    _sweep_frame(_one_cell()).to_parquet(arch_dir / "tanh.parquet")

    assert main(["--source", str(tmp_path), "--stamp", "2026-01-01"]) == 0
    assert "WARNING" not in capsys.readouterr().out, "the default must not warn"

    non_default = NEAR_ZERO_FRACTION / 2
    assert (
        main(
            [
                "--source",
                str(tmp_path),
                "--stamp",
                "2026-01-01",
                "--near-zero-fraction",
                str(non_default),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "WARNING" in printed
    assert f"the gate splits near-zero lanes at {NEAR_ZERO_FRACTION:g}" in printed
    assert f"near_zero_fraction={non_default:g}" in printed, "and it is stamped"


def test_two_input_formats_stay_split_end_to_end():
    """The ``input_format`` dimension this PR promotes to a first-class ``BudgetKey``
    field, driven through ``measure_all``/``render`` the way the output-format and
    dest_acc collapse tests already drive theirs.

    ``_one_cell`` takes ``in_fmt`` and no call passed it, so every end-to-end test here
    ran at ``Float32`` and nothing proved two *different* input formats stay split:
    ``test_render_emits_a_key_that_pins_the_input_format`` only asserts the single
    default appears, and the no-collapse test leaves every cell at ``Float32``. A
    collapse over the input axis would have passed both.
    """
    rows = _one_cell(in_fmt="fp32", pairs=[(1.0, 1.0)] * 32) + _one_cell(
        in_fmt="bf16", pairs=[(1.0, 1.0)] * 32
    )
    cells, notes = measure_all(
        _sweep_frame(rows), None, DEFAULT_PERCENTILE, NEAR_ZERO_FRACTION
    )
    assert notes == []
    assert {c.input_format for c in cells} == {
        DataFormat.Float32,
        DataFormat.Float16_b,
    }
    text = render(cells, "wh", DEFAULT_HEADROOM, "2026-01-01")
    # Two keys, one per input path, even though both measure identically -- which is the
    # point: a shared key would cover input paths the sweep never measured.
    assert text.count("input_format=DataFormat.Float32") == 1
    assert text.count("input_format=DataFormat.Float16_b") == 1
    assert text.count("BudgetKey(") == 2


def test_a_ceiling_refusal_reports_the_lane_set_the_ceiling_compared():
    """On the refusal path ``resolve()`` discards the floor, so the key-level
    ``near_zero_atol is not None`` was always false there -- and the statistics printed
    beside "budget would be N" came from all the lanes, while ``_resolve_measurement``
    had rejected a number computed from the *bulk* ones whenever a floor existed.

    The verdict was never wrong (the all-lane budget is the larger, so also past the
    ceiling); the magnitude was. Reachable on bf16, whose ceiling is 6.4, so a bulk
    budget of 7 is enough -- and floors do fire on bf16 outputs in the shipped table.
    """
    cell = _cell(
        output_format=DataFormat.Float16_b,
        max_ulp=9,  # bulk: _budget(9, 9.0, 1.25) = 12, past the 6.4 ceiling
        percentile_ulp=9.0,
        points=1000,
        near_zero_points=10,
        near_zero_max_ulp=99999,
        near_zero_max_abs_err=1e-4,
        all_max_ulp=99999,
        all_percentile_ulp=99999.0,
    )
    assert cell._floor(DEFAULT_HEADROOM) is not None, "a floor has to exist to matter"
    assert cell.resolve(DEFAULT_HEADROOM) == (None, None)

    key = EmittedKey(
        input_format=DataFormat.Float32,
        output_format=DataFormat.Float16_b,
        approx_mode=None,
        dest_acc=None,
        budget=None,
        near_zero_atol=None,
        cells=(cell,),
    )
    comment = key.comment("wh", "2026-09-17", DEFAULT_PERCENTILE, DEFAULT_HEADROOM)
    assert "budget would be 12" in comment, "the value the ceiling actually compared"
    # ...from the bulk lanes, which is where that 12 came from. The all-lane view says
    # 99999, and printing it here would describe a number the ceiling never saw.
    assert "max 9 ULP" in comment
    assert "99999" not in comment


def test_a_marker_and_a_ceiling_refusal_on_one_key_both_get_reported():
    """``budget_of`` gives ``(None, None)`` for a marker cell and for a ceiling refusal
    alike, so the collapse stage merges the two into one key -- and the comment used to
    ``return`` on the marker reason, dropping the ceiling clause *and* the whole
    measurement line with it.

    Two shipped entries were wrong because of it: both ``approx_mode=Yes`` at 4096 pts,
    carrying a ``dest_acc=Yes`` marker reason while the ``dest_acc=No`` half's ceiling
    refusal and its measurement appeared nowhere in the table. They were the only
    tolerance-demoted entries with no measurement line at all.
    """
    marker = _cell(
        op=MathOperation.Gelu,
        input_format=DataFormat.Float32,
        output_format=DataFormat.Float32,
        dest_acc=DestAccumulation.Yes,  # a real _NOT_PREDICTED_BY_SWEEP entry
        max_ulp=3,
        percentile_ulp=3.0,
    )
    refused = _cell(
        op=MathOperation.Gelu,
        input_format=DataFormat.Float32,
        output_format=DataFormat.Float16_b,
        dest_acc=DestAccumulation.No,  # not a marker; refused by the bf16 ceiling
        max_ulp=9,
        percentile_ulp=9.0,
    )
    assert refused.budget(DEFAULT_HEADROOM) is None, "must be a ceiling refusal"

    key = EmittedKey(
        input_format=DataFormat.Float32,
        output_format=None,
        approx_mode=ApproximationMode.Yes,
        dest_acc=None,
        budget=None,
        near_zero_atol=None,
        cells=(marker, refused),
    )
    comment = key.comment("wh", "2026-09-17", DEFAULT_PERCENTILE, DEFAULT_HEADROOM)
    assert "not enrolled -- near-zero tail" in comment, "the marker reason survives"
    assert (
        "stops being tighter than the tolerance it replaces" in comment
    ), "and so does the ceiling clause"
    assert "max 9 ULP" in comment, "and the measurement line is not dropped"


def test_the_marker_lookup_is_scoped_to_the_variant_it_was_measured_on():
    """A marker forced every approximation and destination variant of a format pair onto
    tolerance, although the divergence is variant-specific: Gelu fp32->fp32 diverges at
    ``dest_acc=Yes`` and Log fp32->fp32 at ``dest_acc=No``. The predicted cells were
    losing their gate -- 8192 Gelu points merged into one wildcard contract."""
    from accuracy.emit_budget import not_predicted_reason

    def gelu(dest):
        return _cell(
            op=MathOperation.Gelu,
            input_format=DataFormat.Float32,
            output_format=DataFormat.Float32,
            dest_acc=dest,
        )

    assert not_predicted_reason(gelu(DestAccumulation.Yes)) is not None
    assert not_predicted_reason(gelu(DestAccumulation.No)) is None

    def log(dest):
        return _cell(
            op=MathOperation.Log,
            input_format=DataFormat.Float32,
            output_format=DataFormat.Float32,
            dest_acc=dest,
        )

    assert not_predicted_reason(log(DestAccumulation.No)) is not None
    assert not_predicted_reason(log(DestAccumulation.Yes)) is None

    # An unscoped marker (dest_acc=None) still covers both settings.
    def log1p_bf16(dest):
        return _cell(
            op=MathOperation.Log1p,
            input_format=DataFormat.Float16,
            output_format=DataFormat.Float16_b,
            dest_acc=dest,
        )

    assert all(
        not_predicted_reason(log1p_bf16(dest)) is not None for dest in DestAccumulation
    )

    # And an unmarked op is predicted on every variant.
    assert not_predicted_reason(_cell(op=MathOperation.Tanh)) is None


def test_the_emitter_measures_the_flush_policy_the_gate_applies():
    """The emitter recomputes the step count with ``ulp_distance``'s per-dtype default.
    The gate reaches the same metric through ``ulp_elementwise_valid``, which used to
    hardcode ``flush_subnormals=True`` -- so on a Float16 output (``--formats all``) the
    emitter counted subnormal-band steps the gate collapsed to zero, and could derive a
    much larger budget than the verdict it claims to gate.

    Pinned as a property rather than as a comment, since the two live in different
    modules: whatever the gate resolves for a dtype, the metric the emitter uses must
    resolve the same."""
    from helpers.ulp import flushes_subnormals, ulp_distance, ulp_elementwise_valid

    for fmt in (DataFormat.Float32, DataFormat.Float16_b, DataFormat.Float16):
        dtype = ulp_dtype(fmt)
        # Two distinct values inside this dtype's subnormal band.
        smallest = float(torch.finfo(dtype).tiny) * 2.0 ** -MANTISSA_BITS[dtype]
        band_steps = (1 << MANTISSA_BITS[dtype]) - 1
        golden = torch.tensor([smallest], dtype=dtype)
        result = torch.tensor([band_steps * smallest], dtype=dtype)

        emitter_view = int(ulp_distance(golden, result)[0])
        _, gate_view, _ = ulp_elementwise_valid(golden, result, 0)
        assert emitter_view == int(gate_view[0]), fmt.name
        # And the band is only collapsed where the harness's datapath model flushes it.
        assert (emitter_view == 0) is flushes_subnormals(dtype), fmt.name


# ─────────────────────────────────────────────────────────────────────────────
# The emitter must model the gate it derives budgets for
# ─────────────────────────────────────────────────────────────────────────────


def _gate_accepts(cell, golden, hardware, headroom=DEFAULT_HEADROOM):
    """Feed a resolved contract back to the real gate over the data it came from."""
    from helpers.ulp import ulp_elementwise_valid

    budget, floor = cell.resolve(headroom)
    assert budget is not None, "cell resolved to tolerance; nothing to check"
    dtype = ulp_dtype(cell.output_format)
    g = torch.tensor(golden, dtype=torch.float64).to(dtype)
    h = torch.tensor(hardware, dtype=torch.float64).to(dtype)
    is_valid, distance, _ = ulp_elementwise_valid(g, h, budget, near_zero_atol=floor)
    return bool(is_valid.all()), budget, floor, int(distance.max())


def test_an_emitted_budget_is_accepted_by_the_gate_on_its_own_measurement():
    """The invariant the emitter exists to guarantee, asserted end to end rather than
    reasoned about: whatever ``(max_ulp, near_zero_atol)`` pair comes out must make
    ``ulp_elementwise_valid`` accept the very rows it was measured from.

    This is the regression test for a real defect. The gate bounds "near zero" both
    relatively (a fraction of the tensor's dynamic range) *and* absolutely
    (``near_zero_atol / near_zero_fraction``); the emitter modelled only the relative
    half, so it derived a floor from a lane the gate then refused to rescue. Measured on
    WH: hardsigmoid fp32->fp32 ``dest_acc=Yes`` emitted ``max_ulp=10`` with
    ``near_zero_atol=9.31e-09`` from a lane at ``|golden|=7.7e-4`` whose 7.45e-9 error is
    inside that atol but whose magnitude is 800x the absolute cut — so the functional
    suite charged it 128 steps against a 10-step budget.

    The shape below reproduces that: a wide dynamic range, bulk lanes that are nearly
    exact, and one small-magnitude lane whose absolute error is tiny but whose step count
    is enormous.
    """
    golden = [1.0, 2.0, 4.0, 8.0] * 16 + [7.7e-4]
    hardware = list(golden)
    hardware[-1] = 7.7e-4 + 7.45e-9

    cell = _measure(golden, hardware, fmt=DataFormat.Float32)
    accepted, budget, floor, worst = _gate_accepts(cell, golden, hardware)
    assert accepted, (
        f"the gate rejected the emitter's own contract: max_ulp={budget}, "
        f"near_zero_atol={floor}, worst measured {worst} steps"
    )


@pytest.mark.parametrize(
    "fmt", [DataFormat.Float32, DataFormat.Float16_b], ids=lambda f: f.name
)
@pytest.mark.parametrize(
    "headroom", [1.0, DEFAULT_HEADROOM, 2.0], ids=lambda h: f"h{h}"
)
def test_the_gate_accepts_the_emitted_contract_across_headrooms(fmt, headroom):
    """``headroom`` scales the floor, which moves the absolute bound, which moves the lane
    split — so the emitter has to be given the same headroom it will be resolved at, and
    the invariant has to hold at each one. ``main()`` passes one value to both."""
    from accuracy.emit_budget import measure_cell

    golden = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0] * 8 + [1e-3, 5e-4, 1e-4]
    hardware = list(golden)
    hardware[-3] = 1e-3 + 3e-9
    hardware[-2] = 5e-4 + 2e-9
    hardware[-1] = 1e-4 + 1e-9

    cell = measure_cell(
        _rows(golden, hardware),
        MathOperation.Gelu,
        DataFormat.Float32,
        fmt,
        ApproximationMode.No,
        DestAccumulation.Yes,
        FastMode.No,
        DEFAULT_PERCENTILE,
        NEAR_ZERO_FRACTION,
        headroom,
    )
    if cell.resolve(headroom)[0] is None:
        pytest.skip("cell resolved to tolerance at this headroom")
    accepted, budget, floor, worst = _gate_accepts(cell, golden, hardware, headroom)
    assert accepted, (
        f"{fmt.name} at headroom {headroom}: gate rejected max_ulp={budget}, "
        f"near_zero_atol={floor} over its own rows (worst {worst} steps)"
    )


def test_the_near_zero_split_converges_when_the_bound_shrinks_it():
    """The absolute bound depends on the floor, which is derived from the lanes the bound
    selects, so the split is solved by iteration. It terminates because the set only ever
    shrinks — dropping a lane can only lower the max error, which lowers the floor, which
    lowers the cut. This is the case where it actually iterates, twice.

    Reaching it needs care, and the previous shape did not. With every near-zero lane at
    a relative error of exactly ``near_zero_fraction`` the first cut lands at
    ``1.25 * max|golden_nz|`` -- above every near-zero magnitude -- so ``narrowed ==
    near_zero`` on the first pass and the loop broke having narrowed nothing. (The lane
    at 1e-2 was not even in the band: the relative cut is ``<``, and 0.01 < 0.01 is
    false.) The other two tests that reach the loop collapse the set straight to empty,
    so the converged-floor path -- where the emitted floor must equal the converged atol
    -- was untested repo-wide, and with it the ``.3g`` rounding and the loop's own
    termination.

    Here each lane's error is chosen so the cut it produces lands *inside* the band:
    round 1 evicts 9e-3 (cut 125 * 5e-5 = 6.25e-3), round 2 evicts 5e-3 (cut 2.5e-3),
    round 3 finds 2.25e-3 >= 2e-3 and stops. The set therefore shrinks 4 -> 3 -> 2, and
    the assertions below pin the shrink, the converged floor, and the gate agreeing.
    """
    near_zero_magnitudes = [9e-3, 5e-3, 2e-3, 1e-3]
    absolute_errors = [5e-5, 2e-5, 1.8e-5, 1e-6]
    golden = [1.0] * 32 + near_zero_magnitudes
    hardware = [1.0] * 32 + [
        m + e for m, e in zip(near_zero_magnitudes, absolute_errors)
    ]
    # Every one of them is inside the *relative* bound, so the narrowing is the absolute
    # bound's doing and not the relative cut's.
    assert all(m < NEAR_ZERO_FRACTION * 1.0 for m in near_zero_magnitudes)

    cell = _measure(golden, hardware, fmt=DataFormat.Float32)
    assert cell.near_zero_points == 2, "the iteration must actually narrow the set"

    budget, floor = cell.resolve(DEFAULT_HEADROOM)
    assert budget is not None
    # The converged atol, rounded up to three significant figures: the surviving lanes'
    # worst absolute error is the 2e-3 lane's 1.8e-5.
    assert floor == pytest.approx(1.8e-5 * DEFAULT_HEADROOM, rel=1e-3)
    # ...and that floor really does define the band it was measured over: a lane at the
    # top of the surviving set is inside `floor / NEAR_ZERO_FRACTION`, the evicted one is
    # not.
    assert 2e-3 <= floor / NEAR_ZERO_FRACTION < 5e-3

    accepted, _, _, worst = _gate_accepts(cell, golden, hardware)
    assert accepted, f"gate rejected max_ulp={budget}, floor={floor}, worst {worst}"

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for ``helpers/ulp_budget_diff.py``: what a change to the table does
to the gates, and how the ``--ulp-measure`` records are judged against it. Every table
and record here is built from the typed rows and measurements, never from text."""

import math

import pytest
import yaml
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, MathOperation
from helpers.ulp_budget_diff import (
    _KIND_TEXT,
    _MAX_ROWS,
    Kind,
    Verdict,
    compare,
    draw,
    judge,
    main,
    read_cells,
    render_budget_diff,
)
from helpers.ulp_domains import (
    BINADE_OFFSET,
    DOMAIN_MIN_BIN_LANES,
    Domain,
    Problem,
    binade,
)
from helpers.ulp_table import (
    MEASURED_ARCH,
    BudgetKey,
    Measurement,
    Metric,
    Reading,
    Row,
    write_table_file,
)
from helpers.ulp_testing import TANH_DOMAINS, refused

INF = math.inf
F16B, F16, F32 = DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32
NO, YES = ApproximationMode.No, ApproximationMode.Yes
D_NO, D_YES = DestAccumulation.No, DestAccumulation.Yes
ABS, TANH = MathOperation.Abs, MathOperation.Tanh
TOL = Metric.TOLERANCE


def _table(*rows, op=ABS):
    return {op: tuple(rows)}


_BASE_ROWS = (
    Row(input=F16B, output=F16B, measured=1),
    Row(input=F16, output=F16B, measured=4),
    Row(input=F16, output=F16, metric=TOL, measured=393),
    Row(input=F32, output=F32, measured=2),
)
_BASE = _table(*_BASE_ROWS)


def _edited(index, row):
    """The base table with one row replaced."""
    rows = list(_BASE_ROWS)
    rows[index] = row
    return _table(*rows)


def _kinds(base, head):
    return {change.cell[1]: change.kind for change in compare(base, head)}


def _pair(input_format, output_format):
    return BudgetKey(input_format=input_format, output_format=output_format)


# ── What counts as a regression ───────────────────────────────────────────────


def test_a_raised_budget_is_a_regression():
    (change,) = compare(_BASE, _edited(0, Row(input=F16B, output=F16B, measured=9)))
    assert change.kind is Kind.RAISED and change.is_regression


def test_losing_the_gate_entirely_is_a_regression():
    """Demotion to tolerance: the cell stops being judged on a step budget at all."""
    head = _edited(0, Row(input=F16B, output=F16B, metric=TOL, measured=14337))
    assert _kinds(_BASE, head)[_pair(F16B, F16B)] is Kind.UNGATED


def test_deleting_a_gated_row_is_a_regression_and_a_figureless_tolerance_row_is_not():
    assert _kinds(_BASE, _table(*_BASE_ROWS[1:]))[_pair(F16B, F16B)] is Kind.REMOVED
    bare = Row(input=F16, output=F16, metric=TOL)
    assert compare(_table(_BASE_ROWS[0], bare), _table(_BASE_ROWS[0])) == []


def test_tightening_and_newly_gating_are_not_regressions():
    head = _table(
        Row(input=F16B, output=F16B, measured=0),
        _BASE_ROWS[1],
        Row(input=F16, output=F16, measured=3),
        _BASE_ROWS[3],
        Row(input=F16, output=F32, measured=7),
    )
    changes = compare(_BASE, head)
    assert not any(c.is_regression for c in changes)
    assert {c.kind for c in changes} == {Kind.TIGHTENED, Kind.GATED}


def test_a_row_the_sweep_collapses_is_not_a_loss_while_its_cells_keep_their_budget():
    """The emitter drops a key field when both values measure alike. Nothing a measured
    cell resolves to changes; only a query with `approx` unset newly matches."""
    base = _table(
        Row(input=F16B, output=F32, approx=NO, dest=D_YES, measured=36018),
        Row(input=F16B, output=F32, approx=YES, dest=D_YES, measured=36018),
    )
    same = _table(Row(input=F16B, output=F32, dest=D_YES, measured=36018))
    assert [(c.kind, c.variants) for c in compare(base, same)] == [(Kind.GATED, 1)]
    raised = _table(Row(input=F16B, output=F32, dest=D_YES, measured=36030))
    (change,) = [c for c in compare(base, raised) if c.is_regression]
    assert change.kind is Kind.RAISED and change.variants == 2


def test_a_more_specific_row_that_loosens_a_cell_is_a_raise_not_an_addition():
    """No existing row changes, yet every Float32 query now resolves to 1000 steps."""
    base = _table(Row(measured=2))
    (change,) = compare(base, _table(Row(measured=2), Row(input=F32, measured=1000)))
    assert (change.kind, change.cell) == (
        Kind.RAISED,
        (ABS, BudgetKey(input_format=F32)),
    )
    demoted = _table(Row(measured=2), Row(input=F32, metric=TOL))
    assert [c.kind for c in compare(base, demoted)] == [Kind.UNGATED]


def test_a_tie_between_equally_specific_rows_is_refused_rather_than_ordered():
    tied = _table(Row(input=F16B, measured=1), Row(output=F16B, measured=9))
    with refused(Problem.AMBIGUOUS):
        compare(_BASE, tied)


# ── Tolerance cells: what the headroom check holds them to ────────────────────


@pytest.mark.parametrize(
    "head_row, kind",
    [
        (Row(input=F16, output=F16, metric=TOL, measured=500), Kind.BASELINE_RAISED),
        (
            Row(input=F16, output=F16, metric=TOL, measured=393, nonfinite=3),
            Kind.BASELINE_RAISED,
        ),
        (Row(input=F16, output=F16, metric=TOL), Kind.BASELINE_DROPPED),
        (Row(input=F16, output=F16, metric=TOL, measured=12), None),
    ],
    ids=["max-raised", "nonfinite-appears", "figure-deleted", "max-lowered"],
)
def test_a_tolerance_rows_recorded_measurement_is_a_baseline(head_row, kind):
    changes = compare(_BASE, _edited(2, head_row))
    assert [c.kind for c in changes] == ([kind] if kind else [])
    assert all(c.is_regression for c in changes)


def test_a_re_emit_that_splits_a_tolerance_row_drops_no_baseline():
    """A query leaving `dest` unset matches nothing after the split, but no measured
    cell loses its figure. Raising either half still reads as a raise."""
    one = _table(Row(input=F16B, output=F32, approx=YES, metric=TOL, measured=9))

    def split(no, yes):
        return _table(
            Row(input=F16B, output=F32, approx=YES, dest=D_NO, metric=TOL, measured=no),
            Row(
                input=F16B, output=F32, approx=YES, dest=D_YES, metric=TOL, measured=yes
            ),
        )

    assert not [c for c in compare(one, split(9, 8)) if c.is_regression]
    (raised,) = [c for c in compare(one, split(9, 12)) if c.is_regression]
    assert raised.kind is Kind.BASELINE_RAISED and raised.cell[1].dest_acc is D_YES


def test_deleting_a_tolerance_row_with_a_baseline_drops_it():
    head = _table(_BASE_ROWS[0], _BASE_ROWS[1], _BASE_ROWS[3])
    assert _kinds(_BASE, head)[_pair(F16, F16)] is Kind.BASELINE_DROPPED


@pytest.mark.parametrize("flag", ["approx", "dest"])
def test_deleting_a_parent_row_is_seen_on_the_flag_value_no_row_pins(flag):
    """`approx: Yes` alone pins one value; the `No` cells fell back to the broad row,
    and deleting it leaves them on nothing."""
    pinned = Row(
        input=F16B,
        output=F32,
        metric=TOL,
        measured=9,
        **{flag: YES if flag == "approx" else D_YES},
    )
    base = _table(Row(input=F16B, output=F32, metric=TOL, measured=9), pinned)
    assert {c.kind for c in compare(base, _table(pinned))} == {Kind.BASELINE_DROPPED}


def test_deleting_a_row_that_records_only_a_nonfinite_count_drops_a_baseline():
    base = _table(Row(input=F32, output=F16, metric=TOL, nonfinite=3))
    head = _table(Row(input=F16B, output=F16B, measured=1))
    assert Kind.BASELINE_DROPPED in {c.kind for c in compare(base, head)}


@pytest.mark.parametrize(
    "base_row, head_row",
    [
        (Row(metric=TOL, atol=0.13, rtol=0.05), Row(metric=TOL, atol=0.2, rtol=0.05)),
        (Row(metric=TOL, atol=0.13, rtol=0.05), Row(metric=TOL, atol=0.13)),
        (Row(metric=TOL, atol=0.13), Row(metric=TOL, atol=0.13, rtol=0.9)),
        (Row(metric=TOL), Row(metric=TOL, atol=100)),
    ],
    ids=["widened", "bound-dropped", "bound-appears", "bound-replaces-default"],
)
def test_a_declared_tolerance_that_may_loosen_is_a_regression(base_row, head_row):
    """A bound that appears replaces a per-format default this tool cannot see."""
    kinds = {c.kind for c in compare(_table(base_row), _table(head_row))}
    assert kinds == {Kind.TOLERANCE_WIDENED}


# ── Where a budget binds ──────────────────────────────────────────────────────


def test_an_unkeyed_budget_binds_only_on_the_measured_arch():
    """Elsewhere the registry hands the cell its tolerance, so a raise there loosens
    nothing; a row naming the arch binds there."""
    assert MEASURED_ARCH is not ChipArchitecture.BLACKHOLE
    unkeyed = Row(input=F16B, output=F16B, measured=3)
    bh = Row(input=F16B, output=F16B, arch=ChipArchitecture.BLACKHOLE, measured=5)
    changes = compare(_table(unkeyed), _table(unkeyed, bh))
    assert [(c.kind, c.cell[1].arch) for c in changes] == [
        (Kind.GATED, ChipArchitecture.BLACKHOLE)
    ]


@pytest.mark.parametrize(
    "base_floor, base_figure, head_floor, kind",
    [
        (1e-7, 2, 5e-5, Kind.FLOOR_WIDENED),
        (None, 2, 5e-5, Kind.FLOOR_WIDENED),
        # A smaller budget must not mask a wider floor: it can more than pay for it.
        (1e-7, 8, 5e-2, Kind.FLOOR_WIDENED),
        (5e-5, 2, 1e-7, None),
    ],
    ids=["widened", "appears", "masked-by-a-smaller-budget", "narrowed"],
)
def test_the_near_zero_floor_is_part_of_the_gate(
    base_floor, base_figure, head_floor, kind
):
    base = Row(input=F16B, output=F16B, near_zero_atol=base_floor, measured=base_figure)
    head = Row(input=F16B, output=F16B, near_zero_atol=head_floor, measured=2)
    kinds = [c.kind for c in compare(_table(base), _table(head))]
    assert kinds == ([kind] if kind else [])


# ── The diff report and command ───────────────────────────────────────────────


def test_an_unchanged_table_changes_nothing():
    assert compare(_BASE, _BASE) == []


def test_the_report_lists_every_loosened_cell_and_caps_only_the_improvements():
    extra = 5
    n = _MAX_ROWS + extra
    ops = sorted(MathOperation, key=lambda op: op.name)[: 2 * n]

    def table(raised, tightened):
        return {
            op: (Row(input=F16B, output=F16B, measured=raised if i < n else tightened),)
            for i, op in enumerate(ops)
        }

    lines = render_budget_diff(compare(table(5, 5), table(9, 1))).splitlines()

    def shown(kind):
        return sum(f"| {_KIND_TEXT[kind]} |" in line for line in lines)

    assert (shown(Kind.RAISED), shown(Kind.TIGHTENED)) == (n, _MAX_ROWS)


_RAISED_HEAD = _edited(0, Row(input=F16B, output=F16B, measured=9))


@pytest.mark.parametrize(
    "head, flags, status",
    [(_BASE, [], 0), (_RAISED_HEAD, [], 1), (_RAISED_HEAD, ["--allow-raises"], 0)],
    ids=["unchanged", "raised", "raised-advisory"],
)
def test_diff_exit_status(tmp_path, head, flags, status):
    base_path, head_path = tmp_path / "base.yaml", tmp_path / "head.yaml"
    write_table_file(_BASE, base_path)
    write_table_file(head, head_path)
    argv = ["diff", "--base", str(base_path), "--head", str(head_path), *flags]
    assert main(argv) == status


def _diff_main(tmp_path, head_text, *flags):
    """The guard's `diff` of the typed `_BASE` against *head_text* as written; its exit
    status."""
    base_path, head_path = tmp_path / "base.yaml", tmp_path / "head.yaml"
    write_table_file(_BASE, base_path)
    head_path.write_text(head_text, encoding="utf-8")
    return main(["diff", "--base", str(base_path), "--head", str(head_path), *flags])


@pytest.mark.parametrize(
    "head",
    [
        yaml.safe_dump({"Abs": [{"input": "Bfp8_b", "measured": 1}]}),
        "<<<<<<< HEAD\nAbs: [\n",
        "Abs:\n- {{measured: 1}}\n",
        "Abs:\n- {input: Float16_b, output: Float16_b, domains: [{low: -.inf}]}\n",
    ],
    ids=["block-float", "conflict-marker", "doubled-braces", "entry-without-high"],
)
def test_a_table_the_registry_refuses_gets_no_verdict(tmp_path, head):
    """Exit 2, "cannot load", even under --allow-raises: the flag admits a loosened
    gate, not a table no gate can be read from. Text PyYAML cannot read and a record
    missing a field used to escape as tracebacks with the regression exit status."""
    assert _diff_main(tmp_path, head, "--allow-raises") == 2


# ── Rows with a `domains` list ────────────────────────────────────────────────


def _tanh(domains=TANH_DOMAINS, **fields):
    row = Row(input=F16B, output=F16B, dest=D_NO, domains=domains, **fields)
    return _table(row, op=TANH)


def _segments(base, head):
    return [(c.kind, c.segment) for c in compare(base, head)]


def test_a_plain_row_turned_into_a_list_is_judged_segment_by_segment():
    """Gated at 3 outside [-1, 1); bf16's ceiling leaves the middle a tolerance cell."""
    plain = _table(
        Row(input=F16B, output=F16B, dest=D_NO, metric=TOL, measured=10), op=TANH
    )
    assert _segments(plain, _tanh()) == [
        (Kind.GATED, (-INF, -1.0)),
        (Kind.GATED, (1.0, INF)),
    ]


def test_an_entry_raised_is_a_raise_for_its_lanes_only():
    raised = (Domain(-INF, -1, 4), *TANH_DOMAINS[1:])
    assert _segments(_tanh(), _tanh(raised)) == [(Kind.RAISED, (-INF, -1.0))]


def test_a_list_redrawn_with_the_same_figures_is_no_change():
    split = (TANH_DOMAINS[0], Domain(-1, 0, 10), Domain(0, 1, 10), TANH_DOMAINS[2])
    assert compare(_tanh(), _tanh(split)) == []


@pytest.mark.parametrize(
    "last", [Domain(1, INF, 9), Domain(1, INF)], ids=["past-the-ceiling", "unranked"]
)
def test_a_list_replaced_by_a_looser_one_loses_the_gate_where_it_loosens(last):
    assert _segments(_tanh(), _tanh((*TANH_DOMAINS[:2], last))) == [
        (Kind.UNGATED, (1.0, INF))
    ]


def test_a_raise_in_a_large_negative_tail_is_seen():
    """`-2**60 - 1.0 == -2**60`: probing below an edge that far out landed on the edge,
    which belongs to the next entry, and the raise read as no change."""
    edge = -(2.0**60)

    def table(tail):
        domains = (Domain(-INF, edge, tail), Domain(edge, INF, 1))
        return _table(Row(input=F32, output=F32, dest=D_NO, domains=domains))

    assert _segments(table(1), table(2)) == [(Kind.RAISED, (-INF, edge))]


def test_a_widened_floor_beside_an_unchanged_list_is_a_regression():
    wider = (*TANH_DOMAINS[:2], Domain(1, INF, 2))
    changes = compare(_tanh(near_zero_atol=1e-7), _tanh(wider, near_zero_atol=1e-3))
    # The middle entry is past bf16's ceiling: not gated, but read outside the floor.
    assert [(c.kind, c.segment) for c in changes] == [
        (Kind.FLOOR_WIDENED, (-INF, -1.0)),
        (Kind.FLOOR_WIDENED, (-1.0, 1.0)),
        (Kind.FLOOR_WIDENED, (1.0, INF)),
    ]


def test_a_widened_floor_beside_an_all_past_list_is_a_regression():
    """No segment is gated, but the headroom report takes each entry's reading outside
    the floor, so a wider floor excuses more lanes there too."""
    past = (Domain(-INF, 0, 9), Domain(0, INF, 10))
    changes = compare(
        _tanh(past, near_zero_atol=1e-7), _tanh(past, near_zero_atol=1e-3)
    )
    assert [(c.kind, c.segment) for c in changes] == [
        (Kind.FLOOR_WIDENED, (-INF, 0.0)),
        (Kind.FLOOR_WIDENED, (0.0, INF)),
    ]


def test_a_widened_floor_where_an_entry_comes_within_the_ceiling_is_a_regression():
    """7 steps past bf16's ceiling to 6 within it reads as newly gated, but both figures
    are measured outside the floor: widened from 1e-7 to 1e-3, it can forgive far more
    than the one step the entry gained."""
    past = (TANH_DOMAINS[0], Domain(-1, 1, 7), TANH_DOMAINS[2])
    gated = (TANH_DOMAINS[0], Domain(-1, 1, 6), TANH_DOMAINS[2])
    changes = compare(
        _tanh(past, near_zero_atol=1e-7), _tanh(gated, near_zero_atol=1e-3)
    )
    assert [(c.kind, c.segment) for c in changes] == [
        (Kind.FLOOR_WIDENED, (-INF, -1.0)),
        (Kind.FLOOR_WIDENED, (-1.0, 1.0)),
        (Kind.FLOOR_WIDENED, (1.0, INF)),
    ]


def test_a_list_re_emitted_as_an_unmeasurable_row_raises_the_baseline():
    """Past the ceiling nothing is gated, and the plain row's figures sit at or below
    the list's -- but it lets the headroom report excuse lanes the list did not."""
    past = (Domain(-INF, 0, 9), Domain(0, INF, 10))
    plain = _table(
        Row(input=F16B, output=F16B, dest=D_NO, metric=TOL, measured=9, nonfinite=3),
        op=TANH,
    )
    changes = compare(_tanh(past), plain)
    assert [(c.kind, c.segment) for c in changes] == [(Kind.BASELINE_RAISED, None)]
    assert changes[0].is_regression
    assert compare(plain, plain) == []


@pytest.mark.parametrize(
    "base_extra, head_row",
    [
        (
            (),
            Row(input=F16B, output=F16B, dest=D_NO, metric=TOL, measured=10, atol=1.0),
        ),
        (
            (Row(metric=TOL, atol=0.05),),
            Row(input=F16B, output=F16B, dest=D_NO, domains=(Domain(-INF, INF, 10),)),
        ),
    ],
    ids=["list-to-plain", "broader-bound-widened"],
)
def test_a_declared_tolerance_that_may_loosen_beside_a_list_is_a_regression(
    base_extra, head_row
):
    """Past bf16's ceiling the list is no whole-cell gate, so the functional drivers
    compare on the op's declared bounds; segment by segment nothing changed."""
    past = Row(input=F16B, output=F16B, dest=D_NO, domains=(Domain(-INF, INF, 10),))
    head_extra = tuple(Row(metric=TOL, atol=1.0) for _ in base_extra)
    changes = compare(
        _table(past, *base_extra, op=TANH), _table(head_row, *head_extra, op=TANH)
    )
    assert (Kind.TOLERANCE_WIDENED, None) in [(c.kind, c.segment) for c in changes]
    assert all(c.is_regression for c in changes)
    assert compare(_table(past, op=TANH), _table(past, op=TANH)) == []


def test_a_re_emit_that_splits_a_list_row_drops_no_baseline():
    past = (Domain(-INF, -1, 9), Domain(-1, 1, 10), Domain(1, INF, 9))
    one = _table(Row(input=F16B, output=F16B, domains=past), op=TANH)
    split = _table(
        Row(input=F16B, output=F16B, dest=D_NO, domains=past),
        Row(input=F16B, output=F16B, dest=D_YES, domains=past),
        op=TANH,
    )
    assert not [c for c in compare(one, split) if c.is_regression]


@pytest.mark.parametrize(
    "output, kind",
    [(F16, Kind.RAISED), (F16B, Kind.UNGATED)],
    ids=["fits-the-ceiling", "past-the-ceiling"],
)
def test_a_null_entry_that_gains_a_figure_is_judged_against_the_largest_it_was_held_to(
    output, kind
):
    """An entry with no `ulp` is not ungated: the sweep holds its strays to the row's
    largest figure, and the emitter gives it a figure of its own only when they beat
    every other. So a re-emit that turns `null` into 40 loosens a gate of 2: a raise
    where 40 still fits the output's ceiling (fp16, 51), a lost gate where the cell
    falls to tolerance (bf16, 6). Read as ungated, both exited 0."""

    def table(low_ulp):
        domains = (Domain(-INF, 0, low_ulp), Domain(0, INF, 2))
        return _table(
            Row(input=F16, output=output, dest=D_NO, domains=domains), op=TANH
        )

    assert _segments(table(None), table(40)) == [(kind, (-INF, 0.0))]
    assert _segments(table(None), table(1)) == [(Kind.TIGHTENED, (-INF, 0.0))]
    assert compare(table(None), table(2)) == []


# ── The headroom check ────────────────────────────────────────────────────────

_CELL = BudgetKey(
    input_format=F16B,
    output_format=F16B,
    approx_mode=NO,
    dest_acc=D_NO,
    arch=MEASURED_ARCH,
)


def _measured(worst, *, op=ABS, lanes=65025, **fields):
    return Measurement(op=op, key=_CELL, worst=worst, lanes=lanes, **fields)


def _verdicts(table, *measurements):
    return [f.verdict for f in judge(table, measurements).findings]


def test_several_measurements_of_one_cell_keep_the_worst():
    """A driver enumerates axes the budget key does not, so one cell is recorded more
    than once. Keeping the last would hide the worst of them."""
    cells = read_cells([_measured(3), _measured(11), _measured(5)])
    assert [c.worst for c in cells.values()] == [11]


@pytest.mark.parametrize(
    "budget, worst, verdicts",
    [(1, 9, [Verdict.OVER_BUDGET]), (8, 0, [Verdict.TIGHTENS]), (3, 3, [])],
    ids=["over", "better", "exact"],
)
def test_a_gated_cell_is_judged_against_its_figure(budget, worst, verdicts):
    table = _table(Row(input=F16B, output=F16B, measured=budget))
    assert _verdicts(table, _measured(worst)) == verdicts


def test_a_tolerance_cell_is_judged_against_its_recorded_measurement():
    table = _table(Row(input=F16B, output=F16B, metric=TOL, measured=393))
    assert _verdicts(table, _measured(393)) == []
    (finding,) = judge(table, [_measured(500)]).findings
    assert (finding.verdict, finding.seen, finding.reference) == (
        Verdict.REGRESSED,
        500,
        393,
    )


def test_a_tolerance_cell_with_no_recorded_figure_is_counted_not_passed_silently():
    headroom = judge(
        _table(Row(input=F16B, output=F16B, metric=TOL)), [_measured(10**6)]
    )
    assert (headroom.findings, headroom.unjudged, headroom.regressions) == ([], 1, 0)


def test_a_measurement_resolves_against_the_most_specific_row():
    table = _table(
        Row(output=F16B, measured=100), Row(input=F16B, output=F16B, measured=1)
    )
    assert _verdicts(table, _measured(50)) == [Verdict.OVER_BUDGET]


def test_an_overflow_the_row_does_not_account_for_fails():
    table = _table(Row(input=F16B, output=F16B, metric=TOL, measured=0, nonfinite=2))
    assert _verdicts(table, _measured(0, nonfinite=2)) == []
    assert _verdicts(table, _measured(0, nonfinite=3)) == [Verdict.NONFINITE]


def test_an_empty_comparison_is_not_a_pass():
    assert judge(_BASE, []).regressions == 1


def test_a_cell_that_ranked_no_lane_regresses_against_its_figure():
    table = _table(Row(input=F16B, output=F16B, metric=TOL, measured=393))
    assert _verdicts(table, _measured(0, lanes=0)) == [Verdict.NO_LANE]


def test_a_measured_cell_no_row_covers_is_counted_not_dropped():
    headroom = judge(_table(Row(input=F32, measured=1)), [_measured(10**6)])
    assert (headroom.uncovered, headroom.regressions) == (1, 0)


def _tanh_measured(readings, bins=(), **fields):
    return _measured(10, op=TANH, readings=tuple(readings), bins=bins, **fields)


def test_each_entry_of_a_list_is_judged_on_its_own_lanes():
    table = _tanh((*TANH_DOMAINS[:2], Domain(1, INF)))
    readings = [
        Reading(-INF, -1.0, 5, 16000),
        Reading(-1.0, 1.0, 9, 33000),
        Reading(1.0, INF, 2, 70),
    ]
    findings = judge(table, [_tanh_measured(readings)]).findings
    assert [(f.verdict, f.entry) for f in findings] == [
        (Verdict.OVER_BUDGET, TANH_DOMAINS[0]),
        (Verdict.TIGHTENS, TANH_DOMAINS[1]),
        (Verdict.NOW_ANSWERS, Domain(1, INF)),
    ]


@pytest.mark.parametrize(
    "lanes, bins, answers",
    [
        (7, (), False),
        (70, (), True),
        # With the binade profile the rule is exact: 14 per binade ranks none of them.
        (70, tuple((BINADE_OFFSET + k, 128, 14, 0) for k in range(5)), False),
        (70, ((BINADE_OFFSET + 2, 128, DOMAIN_MIN_BIN_LANES, 0),), True),
    ],
    ids=["a-few-strays", "enough-lanes", "spread-thin", "one-full-binade"],
)
def test_a_null_entry_answers_once_a_binade_holds_enough_ranked_lanes(
    lanes, bins, answers
):
    table = _tanh((*TANH_DOMAINS[:2], Domain(1, INF)))
    readings = [
        Reading(-INF, -1.0, 3, 16000),
        Reading(-1.0, 1.0, 10, 33000),
        Reading(1.0, INF, 2, lanes),
    ]
    verdicts = _verdicts(table, _tanh_measured(readings, bins))
    assert verdicts == ([Verdict.NOW_ANSWERS] if answers else [])


@pytest.mark.parametrize(
    "middle, verdict",
    [(10, Verdict.REGRESSED), (3, Verdict.OVER_BUDGET)],
    ids=["largest-past-the-ceiling", "largest-gated"],
)
def test_ranked_lanes_in_a_null_entry_are_held_to_the_largest_figure(middle, verdict):
    table = _tanh((TANH_DOMAINS[0], Domain(-1, 1, middle), Domain(1, INF)))
    readings = [Reading(-INF, -1.0, 3, 16000), Reading(-1.0, 1.0, middle, 33000)]
    stray = Reading(1.0, INF, 1000, 5)
    assert _verdicts(table, _tanh_measured([*readings, stray])) == [verdict]
    fits = Reading(1.0, INF, 2, 5)
    assert _verdicts(table, _tanh_measured([*readings, fits])) == []


def test_a_sweep_record_drawn_over_another_list_fails_rather_than_going_unjudged():
    """One reading over the whole input against a three-entry list: the record was made
    against another table, and no entry of this one was read -- the past-ceiling middle
    one has no other check. A functional driver's record, with no readings or profile at
    all, is only not judged."""
    whole = [Reading(-INF, INF, 1000, 65025)]
    headroom = judge(_tanh(), [_tanh_measured(whole)])
    assert [(f.verdict, f.entry) for f in headroom.findings] == [
        (Verdict.NOT_READ, entry) for entry in TANH_DOMAINS
    ]
    assert headroom.regressions == 3
    functional = judge(_tanh(), [_tanh_measured([])])
    assert (functional.findings, functional.unjudged) == ([], 3)


def test_a_list_cell_that_lost_every_lane_is_a_regression():
    table = _tanh((Domain(-INF, -1, 9), TANH_DOMAINS[1], Domain(1, INF, 9)))
    assert _verdicts(table, _tanh_measured([], lanes=0)) == [Verdict.NO_LANE]


def test_headroom_exit_status(tmp_path):
    table, measured = tmp_path / "table.yaml", tmp_path / "measured.jsonl"
    write_table_file(_table(Row(input=F16B, output=F16B, measured=1)), table)
    argv = ["headroom", "--table", str(table), "--measured", str(measured)]
    for worst, status in ((1, 0), (9, 1)):
        measured.write_text(_measured(worst).to_json() + "\n")
        assert main(argv) == status
    measured.write_text("")
    assert main(argv) == 1
    for unreadable in ("{\n", '{"op": "Abs"}\n'):  # not JSON; a record missing fields
        measured.write_text(unreadable)
        assert main(argv) == 2, unreadable


def test_the_domains_command_reports_the_asked_op_and_writes_the_report_out(
    tmp_path, capsys
):
    """`--op` keeps one op, a cell the table holds a list for shows it beside the drawn
    one, and `--out` gets exactly what was printed."""
    table, measured = tmp_path / "table.yaml", tmp_path / "measured.jsonl"
    out = tmp_path / "domains.md"
    write_table_file(
        {**_tanh(), ABS: (Row(input=F16B, output=F16B, measured=1),)}, table
    )
    tanh = _measured(
        10,
        op=TANH,
        bins=tuple(
            (sign * (k + BINADE_OFFSET), 128, 128, 10 if k < 0 else 3)
            for k in range(-126, 128)
            for sign in (1, -1)
        ),
    )
    abs_ = _measured(1, bins=((BINADE_OFFSET, 128, 128, 1),))
    measured.write_text("\n".join(m.to_json() for m in (tanh, abs_)) + "\n")
    argv = ["domains", "--measured", str(measured), "--table", str(table)]
    assert main([*argv, "--op", "Tanh", "--out", str(out)]) == 0
    report = capsys.readouterr().out
    assert out.read_text(encoding="utf-8") == report
    assert "Tanh" in report and "table:" in report and "Abs" not in report


def test_an_unknown_op_is_a_usage_error_not_a_traceback(tmp_path):
    """argparse converts only an `ArgumentTypeError` into its exit-2 usage error; the
    `KeyError` a bare `MathOperation[name]` raised escaped with the regression status.
    """
    argv = ["domains", "--measured", str(tmp_path / "m.jsonl"), "--op", "cos"]
    with pytest.raises(SystemExit) as caught:  # allow-pytest.raises: host-only test
        main(argv)
    assert caught.value.code == 2


def test_the_guard_imports_no_module_that_loads_the_checked_out_table():
    """`helpers.sfpu_accuracy_budget` loads the table at import. The guard judges tables
    by path, and a PR head whose table the loader refuses is exactly its case: that
    refusal must reach `main` as "cannot load" (exit 2), not kill the process at import
    with the regression status. So the guard takes the resolution rule from
    `ulp_table`, and the registry stays out of its import chain."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys, helpers.ulp_budget_diff; "
        "sys.exit('helpers.sfpu_accuracy_budget' in sys.modules)"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


# ── The list the emitter's rule draws on a recorded profile ───────────────────


def _drawn(bins):
    measurement = _measured(max(w for *_, w in bins), op=TANH, bins=tuple(bins))
    return draw(read_cells([measurement])[(TANH, _CELL)], _CELL)


def test_the_rule_redraws_a_list_from_the_recorded_profile():
    bins = [
        (sign * (k + BINADE_OFFSET), 128, 128, 10 if k < 0 else 3)
        for k in range(-126, 128)
        for sign in (1, -1)
    ]
    assert _drawn([*bins, (0, 1, 1, 0)]) == TANH_DOMAINS


def test_a_negative_power_of_two_sits_where_the_list_puts_it():
    """-1 is the low edge of [-1, -0.5) under `low <= x < high`, so its binade must be
    that one: grouping it with (-2, -1) put its 9 steps on the wrong side of a split."""
    bins = [
        (binade(x), 2048, 2048, worst) for x, worst in ((-1.5, 1), (-1.0, 9), (-0.5, 9))
    ]
    assert _drawn(sorted(bins)) == (Domain(-INF, -1, 1), Domain(-1, INF, 9))


def test_a_stray_worse_than_every_figure_takes_a_figure_of_its_own():
    strays = [(BINADE_OFFSET + k, 128, 128, 2) for k in range(10)]  # [1, 1024): 2 steps
    strays.append((-BINADE_OFFSET, 128, 3, 40))  # [-2, -1): three lanes at 40
    assert _drawn(strays) == (Domain(-INF, 0, 40), Domain(0, INF, 2))
    unranked = [(BINADE_OFFSET + k, 128, 3, 1) for k in range(10)]
    assert _drawn(unranked) == ()

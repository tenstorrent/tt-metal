# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for ``helpers/ulp_sweep.py``: which lanes the sweep ranks, and the
emitter that folds a sweep back into the table. No device: the emitter's table is a
typed :data:`~helpers.ulp_table.Table`, read back through ``load_table``.
"""

import math
from dataclasses import replace

import pytest
import torch
from helpers import ulp_sweep
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    format_dict,
)
from helpers.ulp import has_ulp_gate, ulp_distance, ulp_stats
from helpers.ulp_domains import Domain, usable_budget_ceiling
from helpers.ulp_sweep import (
    DOMAINS,
    MEASURED,
    AmbiguousCollapse,
    Floored,
    IncompleteGrid,
    KeptRowCollides,
    SessionNotClean,
    Unmeasurable,
    WrongArch,
    _known_lanes,
    _verdict,
    cell_key,
    export_measured,
    finish_emit,
    golden_input,
    measurable_mask,
    merge_measured,
    nonfinite_failures,
    nonfinite_verdict,
    padding_lanes,
    ranked_mask,
    record,
    record_unmeasurable,
    stale_excuses,
    sweep_cells,
    sweep_spec,
    swept_value_count,
    write_table,
)
from helpers.ulp_table import (
    BudgetKey,
    Measurement,
    Metric,
    Reading,
    Row,
    load_table,
    write_table_file,
)

#: An op defined everywhere, so the non-finite tests below turn on the exclusion they
#: mean to test rather than on a domain.
_OP = MathOperation.Abs


@pytest.fixture(autouse=True)
def _host_only_arch(monkeypatch):
    """``sweep_cells`` asks the chip which cells promote to a 32-bit Dest, and without
    ``CHIP_ARCH`` that opens a device. These tests run on hosts without one, and judge
    the Wormhole table whatever ``CHIP_ARCH`` the host sets: under ``quasar`` the
    default-arch grid has cells a Wormhole emit never records."""
    import helpers.chip_architecture as chip

    monkeypatch.setenv("CHIP_ARCH", "wormhole")
    monkeypatch.setattr(chip, "_cached_chip_architecture", None)


@pytest.fixture(autouse=True)
def _no_reading_outlives_its_test():
    """`record` folds into the module-level `MEASURED` and `DOMAINS`, and keeps the worst
    lane: a reading one test left behind would be the next test's figure, and whether
    that showed depended on file order."""
    MEASURED.clear()
    DOMAINS.clear()
    yield
    MEASURED.clear()
    DOMAINS.clear()


# ── The emitter ───────────────────────────────────────────────────────────────

F16, F16B, F32 = DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32
NO, YES = ApproximationMode.No, ApproximationMode.Yes
D_NO, D_YES = DestAccumulation.No, DestAccumulation.Yes
GELU = MathOperation.Gelu

#: The key of a cell whose coordinates the test is not about.
_CELL = cell_key(F16, F16, NO, D_NO)

#: A floored reading: the worst lane over every lane, and ``(residual, atol)`` for the
#: lanes outside the floor.
_WORST, _FLOOR = 14337, (2, 2.4e-7)

_INF = math.inf
_ONE = (Domain(-_INF, _INF, 5),)
_TWO = (Domain(-_INF, 0, 5), Domain(0, _INF, 2))

#: A table with a row of each kind write_table has to tell apart.
_TABLE = {
    GELU: (
        Row(input=F16, output=F16, measured=7, note="superseded"),
        Row(input=F16B, output=F32, measured=9, note="a pair not swept here"),
        Row(input=F16B, output=F16B, arch=ChipArchitecture.BLACKHOLE, measured=44),
        Row(output=F32, measured=0, note="a wildcard"),
    ),
    MathOperation.Log1p: (Row(input=F16, output=F16B, metric=Metric.TOLERANCE),),
}


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "budget.yaml"
    write_table_file(_TABLE, path)
    return path


def _rows(path, op=GELU):
    """*op*'s rows in the table at *path*, by key."""
    return {row.key: row for row in load_table(path)[op]}


def _pair(input_format, output_format, **key):
    return BudgetKey(input_format=input_format, output_format=output_format, **key)


def _record_grid(op, in_fmt, out_fmt, value, **kw):
    """Every approx x dest cell of one pair, so the emitter collapses them."""
    for approx in ApproximationMode:
        for dest in DestAccumulation:
            record(op, cell_key(in_fmt, out_fmt, approx, dest), value, **kw)


def test_only_the_pairs_this_run_measured_are_replaced(table):
    """A `-k` run, an interrupt or a driver skip must leave every pair it did not
    measure alone: the other pair, the other arch's row, the wildcard, the other op."""
    record(GELU, _CELL, 5)
    report = write_table(table)
    assert set(report.written) == {GELU}
    after = _rows(table)
    assert after[_pair(F16, F16)] == Row(input=F16, output=F16, measured=5)
    for row in _TABLE[GELU][1:]:
        assert after[row.key] == row
    assert load_table(table)[MathOperation.Log1p] == _TABLE[MathOperation.Log1p]


def test_a_narrower_re_emit_keeps_the_rows_it_did_not_measure(table):
    _record_grid(GELU, F16, F16, 5)
    write_table(table)
    MEASURED.clear()
    _record_grid(GELU, F16B, F16B, 1)
    write_table(table)
    after = _rows(table)
    assert after[_pair(F16, F16)].measured == 5
    assert after[_pair(F16B, F16B)].measured == 1
    assert after[BudgetKey(output_format=F32)] == _TABLE[GELU][3]


def test_a_demotion_records_the_measurement_past_the_ceiling(table):
    _record_grid(GELU, F16B, F16B, 100)  # bf16's ceiling is 6
    write_table(table)
    row = _rows(table)[_pair(F16B, F16B)]
    assert (row.metric, row.measured) == (Metric.TOLERANCE, 100)


def test_a_floor_row_is_regenerated_from_the_measurement(tmp_path):
    """`near_zero_atol` is a field the emitter writes, so a re-measurement replaces a
    floored row, without the floor when the cell no longer needs one."""
    path = tmp_path / "budget.yaml"
    floored = Row(input=F16, output=F16, near_zero_atol=5.59e-07, measured=2)
    write_table_file({GELU: (floored,)}, path)
    _record_grid(GELU, F16, F16, 5)
    write_table(path)
    assert _rows(path) == {floored.key: Row(input=F16, output=F16, measured=5)}


def test_a_declared_tolerance_row_survives_and_is_not_shadowed(tmp_path):
    """An op-wide `atol` row and a pinned one are judgements no measurement re-derives:
    both are kept, and the emitted numberless tolerance rows fall back to the op-wide
    `atol` in the registry rather than retracting it. The emitted rows of the pinned
    pair are more specific than it, so they hold its cells."""
    path = tmp_path / "budget.yaml"
    wide = Row(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05)
    pinned = Row(input=F16, output=F16, metric=Metric.TOLERANCE, atol=0.2)
    write_table_file({GELU: (wide, pinned)}, path)
    _record_grid(GELU, F16B, F16B, 100)
    for approx in ApproximationMode:  # two figures: the pair's cells do not collapse
        for dest in DestAccumulation:
            record(GELU, cell_key(F16, F16, approx, dest), 3 if approx is NO else 4)
    write_table(path)
    after = _rows(path)
    assert after[wide.key] == wide and after[pinned.key] == pinned
    # `approx` splits F16 -> F16, so it stays pinned on every pair of the op.
    assert after[_pair(F16B, F16B, approx_mode=NO)].metric is Metric.TOLERANCE
    assert after[_pair(F16, F16, approx_mode=NO)].measured == 3


def test_a_kept_row_with_the_key_an_emit_writes_is_refused(tmp_path):
    """The pair's cells all agree and collapse onto the pinned row's own key: one row
    would silently replace the other, and only whether the cells happened to agree
    decided which. Nothing is written."""
    path = tmp_path / "budget.yaml"
    pinned = Row(input=F16, output=F16, metric=Metric.TOLERANCE, atol=0.2)
    write_table_file({GELU: (pinned,)}, path)
    _record_grid(GELU, F16, F16, 3)
    before = path.read_text()
    with pytest.raises(KeptRowCollides) as e:  # allow-pytest.raises: host-only test
        write_table(path)
    assert (e.value.op, e.value.key) == (GELU, pinned.key)
    assert path.read_text() == before


def test_an_op_not_yet_in_the_table_is_written(table):
    record(MathOperation.Sqrt, _CELL, 1)
    write_table(table)
    assert _rows(table, MathOperation.Sqrt) == {
        _pair(F16, F16): Row(input=F16, output=F16, measured=1)
    }


def test_a_cell_recorded_twice_keeps_the_worst_lane():
    """Last-write-wins is the polarity that hides error."""
    record(GELU, _CELL, 12)
    record(GELU, _CELL, 3)
    assert MEASURED[GELU][_CELL] == 12


def test_an_anti_diagonal_grid_is_refused_rather_than_collapsed(table):
    """`approx` and `dest` droppability is decided per field. On an anti-diagonal each
    field sees only singletons, both would be dropped, and one measurement would
    overwrite the other."""
    record(GELU, _CELL, 4)
    record(GELU, cell_key(F16, F16, YES, D_YES), 9)
    before = table.read_text()
    with pytest.raises(AmbiguousCollapse) as e:  # allow-pytest.raises: host-only test
        write_table(table)
    assert e.value.key == _pair(F16, F16)
    assert {d.measured for d in e.value.decisions} == {4, 9}
    assert table.read_text() == before


def test_the_emitted_figure_is_the_measurement():
    """No headroom and no cap: a gate while it fits the output's ceiling, tolerance with
    the same figure past it."""
    assert _verdict(100, F32) is Metric.ULP
    assert _verdict(6, F16B) is Metric.ULP  # at the ceiling: still a gate
    assert _verdict(7, F16B) is Metric.TOLERANCE
    assert _verdict(419430, F32) is Metric.ULP
    assert _verdict(419431, F32) is Metric.TOLERANCE


def test_xdist_workers_measurements_merge_worst_lane_first():
    """Under ``-n`` each worker fills its own ``MEASURED``; the controller merges the
    exports, keeping the worse reading of a cell two workers measured."""
    record(GELU, _CELL, 5)
    worker_a = export_measured()
    MEASURED.clear()
    other = cell_key(F16, F16, YES, D_NO)
    record(GELU, _CELL, 9)
    record(GELU, other, 2)
    worker_b = export_measured()
    MEASURED.clear()
    merge_measured(worker_a)
    merge_measured(worker_b)
    merge_measured(None)  # a worker that recorded nothing
    assert MEASURED == {GELU: {_CELL: 9, other: 2}}


@pytest.mark.parametrize(
    "verdict_first", [True, False], ids=["verdict,int", "int,verdict"]
)
def test_an_unmeasurable_cell_survives_the_xdist_merge_in_either_order(verdict_first):
    """A number must not rescue an overflow; the finite-lane figure is the worse."""
    verdict = Unmeasurable(why="lanes disagreeing", nonfinite=3, measured=2)
    record_unmeasurable(GELU, _CELL, verdict)
    with_verdict = export_measured()
    MEASURED.clear()
    record(GELU, _CELL, 4)
    with_number = export_measured()
    MEASURED.clear()
    for blob in [with_verdict, with_number][:: 1 if verdict_first else -1]:
        merge_measured(blob)
    assert MEASURED == {GELU: {_CELL: replace(verdict, measured=4)}}


@pytest.mark.parametrize("order", [1, -1], ids=["a,b", "b,a"])
def test_two_unmeasurable_readings_merge_to_the_worst_of_both(order):
    a = Unmeasurable(why="2 lanes", nonfinite=2, measured=9)
    b = Unmeasurable(why="5 lanes", nonfinite=5, measured=3)
    for verdict in (a, b)[::order]:
        record_unmeasurable(GELU, _CELL, verdict)
    assert MEASURED == {
        GELU: {_CELL: Unmeasurable(why="5 lanes", nonfinite=5, measured=9)}
    }


def test_a_cell_that_ranked_no_lane_is_written_without_a_figure_when_a_number_arrives(
    table,
):
    """A no-lane verdict (`nonfinite == 0`) given a later reading's figure was written as
    `{metric: TOLERANCE, measured: N}` with no `nonfinite`: an ordinary demotion to every
    reader, held to N by the headroom report. Its row stays not measurable, with no
    figure; the figure is kept underneath, for a non-finite verdict arriving later."""
    for order in (1, -1):
        MEASURED.clear()
        steps = [
            lambda: record_unmeasurable(GELU, _CELL, Unmeasurable(why="no lane")),
            lambda: record(GELU, _CELL, 5),
        ]
        for step in steps[::order]:
            step()
        assert MEASURED[GELU][_CELL].nonfinite == 0, order
        write_table(table)
        assert _rows(table)[_pair(F16, F16)] == Row(
            input=F16,
            output=F16,
            metric=Metric.TOLERANCE,
            note="not measurable: no lane",
        ), order


def test_three_readings_of_one_cell_merge_alike_in_any_order():
    """9, then a no-lane verdict, then an overflow verdict with a figure of its own, 2:
    the no-lane verdict dropped the 9, so the overflow won with its 2, while in the
    other order the cell kept 9. The order the workers' readings arrived in picked the
    baseline. Every order now gives the overflow verdict with the worst figure, read in
    one process or each in a worker of its own and merged by the controller."""
    from itertools import permutations

    readings = (
        lambda: record(GELU, _CELL, 9),
        lambda: record_unmeasurable(GELU, _CELL, Unmeasurable(why="no lane")),
        lambda: record_unmeasurable(
            GELU, _CELL, Unmeasurable(why="overflow", nonfinite=3, measured=2)
        ),
    )
    worst = Unmeasurable(why="overflow", nonfinite=3, measured=9)
    for order in permutations(range(len(readings))):
        MEASURED.clear()
        for i in order:
            readings[i]()
        assert MEASURED[GELU][_CELL] == worst, order
        exported = []
        for i in order:
            MEASURED.clear()
            readings[i]()
            exported.append(export_measured())
        MEASURED.clear()
        for blob in exported:
            merge_measured(blob)
        assert MEASURED[GELU][_CELL] == worst, order


@pytest.mark.parametrize(
    "arch, failed, exitstatus, refusal",
    [
        (ChipArchitecture.BLACKHOLE, 0, pytest.ExitCode.OK, WrongArch),
        (ChipArchitecture.WORMHOLE, 3, pytest.ExitCode.TESTS_FAILED, SessionNotClean),
        (ChipArchitecture.WORMHOLE, 0, pytest.ExitCode.INTERRUPTED, SessionNotClean),
        (ChipArchitecture.WORMHOLE, 0, 17, SessionNotClean),
    ],
    ids=[
        "other-arch",
        "failed-session",
        "interrupted-session",
        "pytest-exit-returncode",
    ],
)
def test_emit_refuses_what_the_session_cannot_vouch_for(
    table, arch, failed, exitstatus, refusal
):
    """Each case as pytest hands it to ``pytest_sessionfinish``; nothing is written."""
    _record_grid(GELU, F16, F16, 5)
    before = table.read_text()
    with pytest.raises(refusal) as caught:  # allow-pytest.raises: host-only test
        finish_emit(arch, failed, table, exitstatus=exitstatus)
    if refusal is WrongArch:
        assert caught.value.arch is arch
    else:
        assert (caught.value.failures, caught.value.exitstatus) == (failed, exitstatus)
    assert table.read_text() == before


def test_emit_refuses_a_partly_measured_grid(table):
    """A write replaces every row of a touched pair; a run narrowed inside one would
    drop the rows of the cells it never reached."""
    record(GELU, _CELL, 5)
    with pytest.raises(IncompleteGrid) as caught:  # allow-pytest.raises: host-only test
        finish_emit(ChipArchitecture.WORMHOLE, 0, table)
    assert caught.value.gaps == [(GELU, F16, F16, 3)]


def test_emit_writes_on_a_clean_wormhole_session(table):
    _record_grid(GELU, F16, F16, 5)
    report = finish_emit(
        ChipArchitecture.WORMHOLE, 0, table, exitstatus=pytest.ExitCode.OK
    )
    assert Row(input=F16, output=F16, measured=5) in report.written[GELU]
    assert report.summary() == "--ulp-emit: rewrote 1 op(s) in budget.yaml"


def test_a_whole_table_emit_over_the_real_table_loads(tmp_path, monkeypatch):
    """Every op the emit sweeps, measured on every cell, written into a copy of the
    checked-in table, reads back."""
    import test_unary_sfpu_ulp as sweep

    path = tmp_path / "budget.yaml"
    write_table_file(load_table(), path)
    monkeypatch.setattr(ulp_sweep, "EMIT", True)
    ops = sweep._sweep_ops()
    for op in ops:
        for cell in sweep_cells(ChipArchitecture.WORMHOLE):
            record(op, cell_key(*cell), 1)
    report = finish_emit(ChipArchitecture.WORMHOLE, 0, path)
    assert set(report.written) == set(ops)
    load_table(path)


def test_the_plugin_prints_the_emit_summary_on_a_clean_session(
    table, monkeypatch, capsys
):
    from types import SimpleNamespace

    from helpers import llk_pytest_plugin

    monkeypatch.setattr(ulp_sweep, "EMIT", True)
    monkeypatch.setattr(ulp_sweep, "TABLE_PATH", table)
    monkeypatch.setattr(
        llk_pytest_plugin, "get_chip_architecture", lambda: ChipArchitecture.WORMHOLE
    )
    _record_grid(GELU, F16, F16, 5)
    session = SimpleNamespace(
        config=SimpleNamespace(
            option=SimpleNamespace(collectonly=False),
            pluginmanager=SimpleNamespace(get_plugin=lambda name: None),
        ),
        testsfailed=0,
        exitstatus=pytest.ExitCode.OK,
    )
    llk_pytest_plugin._finish_ulp_emit(session)
    assert capsys.readouterr().out == "--ulp-emit: rewrote 1 op(s) in budget.yaml\n"
    assert session.exitstatus == pytest.ExitCode.OK
    assert _rows(table)[_pair(F16, F16)].measured == 5


def test_a_not_measurable_verdict_is_written_with_both_figures_and_why(table):
    """The lanes that park the cell, the worst of the rest, and why -- the last as the
    row's note, which no code reads."""
    src = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    golden = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    result = torch.tensor([1.0, float("inf"), 3.015625, 4.0], dtype=torch.bfloat16)
    overflowed = torch.tensor([False, True, False, False])
    stats = ulp_stats(ulp_distance(golden, result), ~overflowed)
    verdict = nonfinite_verdict(overflowed, src, golden, result, stats)
    assert (verdict.nonfinite, verdict.measured) == (1, 1)  # 3.0 -> 3.015625: 1 step
    record(GELU, _CELL, 5)
    parked_key = cell_key(F16, F16, YES, D_NO)
    record_unmeasurable(GELU, parked_key, verdict)
    write_table(table)
    rows = _rows(table)
    parked = rows[replace(parked_key, dest_acc=None)]
    assert (parked.metric, parked.measured, parked.nonfinite) == (
        Metric.TOLERANCE,
        1,
        1,
    )
    assert parked.unmeasurable and parked.note
    assert rows[replace(_CELL, dest_acc=None)].measured == 5


def test_a_floored_cell_is_written_with_its_floor(table):
    _record_grid(GELU, F16, F16, _WORST, floor=_FLOOR)
    write_table(table)
    row = _rows(table)[_pair(F16, F16)]
    assert (row.measured, row.near_zero_atol) == (2, 2.4e-7)
    assert row.contract.max_ulp == 2


def test_a_floor_short_of_the_ceiling_demotes_the_cell(table):
    _record_grid(GELU, F16, F16, _WORST, floor=(100, 1e-3))
    write_table(table)
    row = _rows(table)[_pair(F16, F16)]
    assert (row.metric, row.measured, row.near_zero_atol) == (
        Metric.TOLERANCE,
        _WORST,
        None,
    )


def test_a_tied_reading_resolves_the_same_whichever_arrives_first():
    """A floor rescues lanes, so on an equal worst lane the reading without one wins."""
    outcomes = []
    for order in ((None, _FLOOR), (_FLOOR, None)):
        MEASURED.clear()
        for floor in order:
            record(GELU, _CELL, _WORST, floor)
        outcomes.append(MEASURED[GELU][_CELL])
    assert outcomes == [_WORST, _WORST]


def test_a_floored_measurement_survives_the_xdist_merge():
    record(GELU, _CELL, _WORST, floor=_FLOOR)
    blob = export_measured()
    MEASURED.clear()
    merge_measured(blob)
    merge_measured(blob)
    assert MEASURED[GELU][_CELL] == Floored(_WORST, *_FLOOR)


@pytest.mark.parametrize(
    "first, second",
    [
        ((_WORST, _FLOOR), (5, None)),
        ((5, None), (_WORST, _FLOOR)),
        ((_WORST, _FLOOR), (_WORST, (3, 1e-7))),
        ((5, (1, 1e-7)), (_WORST, _FLOOR)),
    ],
)
def test_differing_readings_of_a_floored_cell_drop_the_floor(first, second):
    """The floor came from one reading's lanes; either order keeps the worst lane
    unfloored."""
    for max_ulp, floor in (first, second):
        record(GELU, _CELL, max_ulp, floor=floor)
    assert MEASURED[GELU][_CELL] == _WORST


def test_a_cell_with_a_list_is_written_as_the_list_and_nothing_else(table):
    _record_grid(GELU, F16, F16, 5, domains=_ONE)
    write_table(table)
    row = _rows(table)[_pair(F16, F16)]
    assert row == Row(input=F16, output=F16, domains=_ONE)
    assert row.contract.max_ulp == 5


def test_a_floor_is_written_beside_the_list_whatever_the_whole_cells_verdict(table):
    """The entries were measured outside the floor, so it is part of each entry's gate
    even when the cell as a whole is past the ceiling."""
    past = (Domain(-_INF, 0, 14337), Domain(0, _INF, 100))
    _record_grid(GELU, F16, F16, _WORST, floor=(100, 1e-3), domains=past)
    write_table(table)
    row = _rows(table)[_pair(F16, F16)]
    assert row == Row(input=F16, output=F16, near_zero_atol=1e-3, domains=past)


def test_lists_are_part_of_what_a_collapse_must_agree_on(table):
    """Two cells with one figure and different lists are two rows."""
    for approx in ApproximationMode:
        for dest in DestAccumulation:
            record(
                GELU,
                cell_key(F16, F16, approx, dest),
                5,
                domains=_ONE if approx is NO else _TWO,
            )
    write_table(table)
    rows = _rows(table)
    assert rows[_pair(F16, F16, approx_mode=NO)].domains == _ONE
    assert rows[_pair(F16, F16, approx_mode=YES)].domains == _TWO


def test_a_list_survives_the_xdist_merge_and_a_differing_reading_drops_it():
    record(GELU, _CELL, 5, domains=_TWO)
    blob = export_measured()
    MEASURED.clear()
    DOMAINS.clear()
    merge_measured(blob)
    assert DOMAINS[GELU][_CELL] == _TWO
    record(GELU, _CELL, 9, floor=(3, 1e-6), domains=_ONE)
    assert MEASURED[GELU][_CELL] == 9 and _CELL not in DOMAINS[GELU]
    MEASURED.clear()
    DOMAINS.clear()
    # The worse reading's list would hold only its own worst lanes: 5 below zero, from
    # the first reading, is not in _ONE's figure there. And a list once dropped stays
    # dropped, whatever reading arrives next.
    record(GELU, _CELL, 5, domains=_TWO)
    record(GELU, _CELL, 9, domains=_ONE)
    record(GELU, _CELL, 2, domains=_TWO)
    assert MEASURED[GELU][_CELL] == 9 and _CELL not in DOMAINS[GELU]


@pytest.mark.parametrize("order", [1, -1], ids=["a,b", "b,a"])
def test_two_readings_with_one_maximum_and_different_lists_keep_no_list(order):
    """Both read 5 at worst, in different entries: either list alone would hold the
    other reading's entry to a figure it measured over. Which list survived used to
    depend on the order the workers' readings arrived in."""
    a = (Domain(-_INF, 0, 3), Domain(0, _INF, 5))
    b = (Domain(-_INF, 0, 5), Domain(0, _INF, 1))
    for domains in (a, b)[::order]:
        record(GELU, _CELL, 5, domains=domains)
    assert MEASURED[GELU][_CELL] == 5 and _CELL not in DOMAINS[GELU]
    record(GELU, _CELL, 5, domains=a)
    assert _CELL not in DOMAINS[GELU]


def test_a_nonfinite_disagreement_is_reported_rather_than_only_masked_out():
    """`measurable_mask` drops it because a step count cannot describe it, and the sweep
    driver ranks a distance rather than calling `passed_test` — so without a separate
    report a hardware overflow leaves the statistics clean and passes."""
    src = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    result = torch.tensor([1.0, float("inf"), 3.0], dtype=torch.bfloat16)
    fmt = DataFormat.Float16_b

    measurable = measurable_mask(src, golden, result, fmt)
    assert measurable.tolist() == [True, False, True]
    assert nonfinite_failures(_OP, src, golden, result, fmt, fmt).tolist() == [
        False,
        True,
        False,
    ]


def test_a_flushed_subnormal_input_is_not_a_nonfinite_failure():
    """The unpack path flushes it and the golden does not, so a disagreement there is
    the flush, not the op — the same exclusion `measurable_mask` makes."""
    tiny = float(torch.finfo(torch.bfloat16).smallest_normal) / 2
    src = torch.tensor([tiny], dtype=torch.bfloat16)
    golden = torch.tensor([1.0], dtype=torch.bfloat16)
    result = torch.tensor([float("inf")], dtype=torch.bfloat16)
    assert not nonfinite_failures(
        _OP, src, golden, result, DataFormat.Float16_b, DataFormat.Float16_b
    ).any()


def test_a_finite_answer_to_an_infinite_golden_is_a_nonfinite_failure():
    """Only a *saturated* store is excused past the output range. `exp` past overflow
    returning the largest finite bf16 where the golden is `+inf`, or an infinity of the
    wrong sign, is the kernel being wrong -- and the ranking mask drops the lane too, so
    this is the only place it can fail."""
    fmt = DataFormat.Float16_b
    src = torch.tensor([5.0, 5.0, 5.0], dtype=torch.bfloat16)
    golden = torch.full((3,), float("inf"), dtype=torch.bfloat16)
    result = torch.tensor([3.39e38, float("-inf"), float("inf")], dtype=torch.bfloat16)
    assert nonfinite_failures(_OP, src, golden, result, fmt, fmt).tolist() == [
        True,
        True,
        False,
    ]


def test_an_infinite_golden_on_the_singularity_point_is_not_a_nonfinite_failure():
    """`rsqrt(-0)` answers `+inf` against `-inf` because the unpack drops the sign, and
    a 16-bit fp16 Dest answers `log(0)` with -130560 because it has no infinity. The
    value at a pole is a limit, so only the point itself is excused -- a finite answer
    one step off it, where the golden is also infinite in the output format, fails."""
    fmt = DataFormat.Float16_b
    src = torch.tensor([-0.0, 0.0, 1e-30], dtype=torch.bfloat16)
    golden = torch.tensor(
        [float("-inf"), float("inf"), float("inf")], dtype=torch.bfloat16
    )
    result = torch.tensor([float("inf"), -130560.0, 3.0e38], dtype=torch.bfloat16)
    assert nonfinite_failures(
        MathOperation.Rsqrt, src, golden, result, fmt, fmt
    ).tolist() == [False, False, True]


def test_the_unpack_format_bounds_which_inputs_count():
    """A Float32 input into a Float16 output at dest_acc=No unpacks into a Float16 Dest:
    below 2**-14 it is flushed, past 65504 it saturates. Neither lane is the op's, so
    both masks drop them there -- and only there: at dest_acc=Yes the Dest is 32-bit.
    A bfloat16 Dest saturates nothing: it has fp32's exponent range, so the walk's
    largest sample (~3.392e38, past bf16's largest) reaches it as bf16's largest."""
    fp32, fp16 = DataFormat.Float32, DataFormat.Float16
    src = torch.tensor([1e-6, 1.0, 1e6], dtype=torch.float32)
    golden = torch.tensor([1.0, 1.0, float("inf")], dtype=torch.float16)
    result = torch.tensor([0.0, 1.0, 11.8], dtype=torch.float16)
    no, yes = DestAccumulation.No, DestAccumulation.Yes
    assert measurable_mask(src, golden, result, fp32, fp16, no).tolist() == [
        False,
        True,
        False,
    ]
    assert measurable_mask(src, golden, result, fp32, fp16, yes)[0]
    assert not nonfinite_failures(
        _OP, src, golden, result, fp32, fp16, dest_acc=no
    ).any()
    assert nonfinite_failures(_OP, src, golden, result, fp32, fp16, dest_acc=yes)[2]

    bf16 = DataFormat.Float16_b
    largest = torch.tensor(
        [float.fromhex("0x1.fe55920000000p+127")], dtype=torch.float32
    )
    assert float(largest[0]) > torch.finfo(torch.bfloat16).max
    same = largest.to(torch.bfloat16)
    assert measurable_mask(largest, same, same, fp32, bf16, no).tolist() == [True]


def test_an_fp16_pack_clamp_is_a_saturated_store():
    """The fp16 pack clamps an out-of-range value to +-65504 of its sign. A bfloat16
    pack has the Dest's range and cannot, so a largest-finite answer there against an
    infinite golden is still the kernel's."""
    src = torch.tensor([5.0, 5.0], dtype=torch.bfloat16)
    fp16 = DataFormat.Float16
    golden = torch.tensor([float("-inf"), float("-inf")], dtype=torch.float16)
    result = torch.tensor([-65504.0, 65504.0], dtype=torch.float16)
    assert nonfinite_failures(
        _OP, src, golden, result, DataFormat.Float16_b, fp16
    ).tolist() == [False, True]


def test_a_golden_past_the_output_range_is_not_a_nonfinite_failure():
    """A full-range sweep of a bf16 input reaches magnitudes a Float16 output cannot
    hold, and `relu_min` passes most of them straight through -- 14,334 lanes of it.
    Saturating there is the store doing what it must, not the kernel being wrong.

    The input stays small and only the golden is large, so nothing but the output-range
    clause can excuse the lane: the same golden into a Float16_b output fails."""
    src = torch.tensor([5.0], dtype=torch.bfloat16)
    golden = torch.tensor([1e5], dtype=torch.bfloat16)
    result = torch.tensor([float("nan")], dtype=torch.bfloat16)
    fmt = DataFormat.Float16_b
    assert not nonfinite_failures(
        _OP, src, golden, result, fmt, DataFormat.Float16
    ).any()
    assert nonfinite_failures(_OP, src, golden, result, fmt, fmt).any()


def test_the_sweeps_own_zero_padding_is_not_data():
    """`generate_full_tensor` pads to the tile count, so the last 511 bf16 lanes are
    zeros the sweep never chose to feed. They inflate every lane count, and on an op
    singular at zero they land on the pole -- 511 copies of one value nobody chose. Both
    masks drop them by position, for every op, whatever its singularities say.

    By position, not by value: one legitimate `0.0` is swept, in the middle.
    """

    fmt = DataFormat.Float16_b
    swept = swept_value_count(fmt)
    # the 254 subnormals are not fed, nor the inf/NaN patterns
    pad_lanes = 2**16 - _BFLOAT16_SWEPT_VALUES
    src = torch.zeros(swept + pad_lanes, dtype=torch.bfloat16)
    pad = padding_lanes(src, fmt)
    assert int(pad.sum()) == pad_lanes
    assert not bool(pad[:swept].any()) and bool(pad[swept:].all())

    # And it reaches both masks: a padded lane is neither measurable nor a failure.
    golden = torch.full_like(src, 1.0)
    result = torch.full_like(src, float("inf"))
    assert not measurable_mask(src, golden, result, fmt)[swept:].any()
    assert not nonfinite_failures(MathOperation.Abs, src, golden, result, fmt, fmt)[
        swept:
    ].any()


def test_an_input_past_a_claim_limit_is_not_a_nonfinite_failure():
    """`Sin` and `Cos` disagree on bf16 lanes past +/-65504 -- an argument reduction
    giving up, not a regression -- and those lanes leave the claim, so they are neither
    ranked nor a non-finite failure."""
    fmt = DataFormat.Float16_b
    far = torch.tensor([2.6e28], dtype=torch.bfloat16)
    golden = torch.tensor([-1.0], dtype=torch.bfloat16)
    result = torch.tensor([float("inf")], dtype=torch.bfloat16)
    assert not nonfinite_failures(
        MathOperation.Sin, far, golden, result, fmt, fmt
    ).any()
    # Inside +/-65504 the same disagreement is a failure.
    assert nonfinite_failures(
        MathOperation.Sin,
        torch.tensor([1.0], dtype=torch.bfloat16),
        golden,
        result,
        fmt,
        fmt,
    ).any()


@pytest.mark.parametrize(
    "op", [MathOperation.Sin, MathOperation.Cos], ids=lambda o: o.name
)
def test_the_claim_limit_follows_what_the_stimuli_format_can_reach(op):
    """`sin(2.6e28)` is a bfloat16 value. float16 ends at 65504, inside the range the
    kernel reduces -- Sin and Cos read 1-4 steps over the whole fp16 format -- so an
    fp16 input carries no limit, and a non-finite answer at its very top is a failure.
    The bf16/fp32 limit is 65504, where the reduction really breaks."""
    fmt = DataFormat.Float16
    top = torch.tensor([65504.0, 4.0], dtype=torch.float16)
    golden = torch.tensor([-1.0, -1.0], dtype=torch.float16)
    result = torch.tensor([float("inf"), float("inf")], dtype=torch.float16)
    assert nonfinite_failures(op, top, golden, result, fmt, fmt).tolist() == [
        True,
        True,
    ]
    # From a bfloat16 input: 65504 rounds to 65536, past the limit and excused; 4.0 is
    # claimed, so an inf there is a failure.
    assert nonfinite_failures(
        op,
        top.to(torch.bfloat16),
        golden.to(torch.bfloat16),
        result.to(torch.bfloat16),
        DataFormat.Float16_b,
        DataFormat.Float16_b,
    ).tolist() == [False, True]
    # Float32 into a Float16 output: 4.0 is claimed at either Dest width.
    wide = top.to(torch.float32)
    for dest in DestAccumulation:
        assert nonfinite_failures(
            op,
            wide,
            golden,
            result,
            DataFormat.Float32,
            DataFormat.Float16,
            dest_acc=dest,
        ).tolist()[1], dest


@pytest.mark.parametrize(
    "fmt, dest_acc, keeps_sign",
    [
        # The unpack drops the sign of -0.0, so the golden is handed +0.0 too.
        (DataFormat.Float16_b, DestAccumulation.No, False),
        (DataFormat.Float16_b, DestAccumulation.Yes, False),
        (DataFormat.Float16, DestAccumulation.Yes, False),
        # Unpack-to-dest keeps it: a 32-bit input at dest_acc=Yes.
        (DataFormat.Float32, DestAccumulation.Yes, True),
        (DataFormat.Float32, DestAccumulation.No, False),
    ],
    ids=lambda v: getattr(v, "name", str(v)),
)
def test_the_golden_sees_the_zero_the_kernel_receives(fmt, dest_acc, keeps_sign):
    """`signbit(-0.0)` is 1.0 against the kernel's 0.0 wherever the unpack drops the
    sign: 16129 bf16 steps on one lane that is the unpack's, not the op's. Only the
    zero moves; every other value reaches the golden as generated."""
    dtype = torch.float32 if fmt.is_32_bit() else torch.bfloat16
    src = torch.tensor([-0.0, 0.0, -1.5, 2.0**-100], dtype=dtype)
    received = golden_input(src, fmt, dest_acc)
    assert bool(torch.signbit(received[0])) is keeps_sign
    assert not torch.signbit(received[1])
    assert torch.equal(received, src)  # -0.0 == 0.0: the values themselves are kept
    assert torch.equal(received[2:], src[2:])


def test_the_unary_golden_keeps_a_float32_output_at_fp32_on_a_16bit_input():
    """#58590: a bfloat16 input's Float32 golden at a 32-bit Dest was rounded to
    bfloat16. 1/3 is inexact in both formats, so the golden must be fp32's 1/3 exactly,
    not bf16's; x*x on a bf16 x is fp32-exact and could not tell them apart."""
    from helpers.golden_generators import UnarySFPUGolden, get_golden_generator

    src = torch.full((1024,), 3.0, dtype=torch.bfloat16)
    golden = get_golden_generator(UnarySFPUGolden)(
        MathOperation.Reciprocal,
        src,
        DataFormat.Float32,
        DestAccumulation.Yes,
        DataFormat.Float16_b,
        [32, 32],
    )
    assert golden.dtype == torch.float32
    fp32 = float(torch.tensor(1 / 3, dtype=torch.float32))
    assert float(golden[0]) == fp32
    assert fp32 != float(torch.tensor(1 / 3, dtype=torch.bfloat16))


def test_the_unary_golden_does_not_overflow_a_float16_input_into_a_float32_output():
    """cosh(-89.375) is 3.27e38: finite in fp32, past fp16. A tile held in the Float16
    input format read inf against the kernel's finite answer."""
    from helpers.golden_generators import UnarySFPUGolden, get_golden_generator

    src = torch.full((1024,), -89.375, dtype=torch.float16)
    golden = get_golden_generator(UnarySFPUGolden)(
        MathOperation.Cosh,
        src,
        DataFormat.Float32,
        DestAccumulation.Yes,
        DataFormat.Float16,
        [32, 32],
    )
    assert golden.dtype == torch.float32
    assert math.isclose(float(golden[0]), math.cosh(89.375), rel_tol=1e-6)


def _binary_golden(a: float, b: float, output_format, dest_acc):
    from helpers.golden_generators import BinarySFPUGolden, get_golden_generator

    pair = torch.tensor([a] * 1024 + [b] * 1024, dtype=torch.bfloat16)
    return get_golden_generator(BinarySFPUGolden)(
        MathOperation.SfpuElwmul,
        pair,
        0,
        1,
        0,
        32,
        [64, 32],
        DataFormat.Float16_b,
        dest_acc=dest_acc,
        output_format=output_format,
    ).flatten()


def test_the_binary_golden_keeps_a_float32_output_at_fp32_on_a_16bit_input():
    """The bf16 x bf16 product is fp32-exact and not bf16-exact, so a tile held in the
    operand format would round it."""
    x = 1.0 + 2.0**-7
    golden = _binary_golden(x, x, DataFormat.Float32, DestAccumulation.Yes)
    assert golden.dtype == torch.float32
    assert float(golden[0]) == x * x != float(torch.tensor(x * x).to(torch.bfloat16))


def test_the_binary_golden_hands_a_16bit_output_back_in_the_operand_dtype():
    """Bit-exact callers compare the golden to the packed result, dtype included."""
    x = 1.0 + 2.0**-7
    for dest_acc in (DestAccumulation.No, DestAccumulation.Yes):
        golden = _binary_golden(x, x, DataFormat.Float16_b, dest_acc)
        assert golden.dtype == torch.bfloat16
        assert float(golden[0]) == float(torch.tensor(x * x).to(torch.bfloat16))


def test_widening_an_fp16_nan_keeps_its_sign():
    """torch's fp16 -> fp32 cast drops a NaN's sign; the goldens' permutations must not."""

    from helpers.golden_generators import widen_to_float32

    half = torch.tensor([-math.nan, math.nan, -1.5], dtype=torch.float16)
    wide = widen_to_float32(half)
    assert torch.isnan(wide[:2]).all() and float(wide[2]) == -1.5
    assert torch.signbit(wide).tolist() == [True, False, True]


def test_an_fp16_dest_saturates_an_infinity_of_either_sign():
    """An fp16 Dest has no infinity; both overflows pack out as its top magnitude."""

    from helpers.golden_generators import convert_inf_to_value

    values = torch.tensor([-math.inf, math.inf, -2.0])
    assert convert_inf_to_value(values, 130560.0).tolist() == [
        -130560.0,
        130560.0,
        -2.0,
    ]
    assert convert_inf_to_value([-math.inf, 1.0], 130560.0) == [-130560.0, 1.0]


def test_a_known_nonfinite_lane_is_excused_on_its_cell_and_nowhere_else():
    """Celu returns inf for x in 65408..65504 on a 16-bit Float16 Dest (#58607). The
    entry names those inputs on that cell. The lane below the interval is still a
    failure, so is the same lane on a 32-bit Dest, and a named lane that agrees is
    ranked like any other: it is the disagreement that is excused, not the lane."""
    fmt = DataFormat.Float16
    src = torch.tensor([65376.0, 65408.0, 65504.0], dtype=torch.float16)
    golden = src.clone()
    result = torch.full_like(src, float("inf"))
    cell = dict(approx_mode=ApproximationMode.Yes, dest_acc=DestAccumulation.No)
    assert nonfinite_failures(
        MathOperation.Celu, src, golden, result, fmt, fmt, **cell
    ).tolist() == [True, False, False]
    assert nonfinite_failures(
        MathOperation.Celu,
        src,
        golden,
        result,
        fmt,
        fmt,
        approx_mode=ApproximationMode.Yes,
        dest_acc=DestAccumulation.Yes,
    ).all()
    # An op with no entry, and a caller that does not name the cell, get no excuse.
    assert nonfinite_failures(_OP, src, golden, result, fmt, fmt, **cell).all()
    assert nonfinite_failures(MathOperation.Celu, src, golden, result, fmt, fmt).all()
    assert measurable_mask(src, golden, golden.clone(), fmt).all()

    (entry,) = _known_lanes()[MathOperation.Celu]
    assert entry.applies_to(fmt, fmt, ApproximationMode.No, DestAccumulation.No)
    assert not entry.applies_to(
        DataFormat.Float16_b, fmt, ApproximationMode.No, DestAccumulation.No
    )
    assert not entry.applies_to(
        fmt, DataFormat.Float16_b, ApproximationMode.No, DestAccumulation.No
    )


def test_an_entry_no_lane_of_which_disagrees_is_stale():
    """The day the defect is fixed its lanes agree, and the gate has to say so rather
    than keep excusing them."""
    fmt = DataFormat.Float16
    src = torch.tensor([65408.0, 65504.0], dtype=torch.float16)
    golden = src.clone()
    cell = dict(approx_mode=ApproximationMode.No, dest_acc=DestAccumulation.No)
    still_broken = torch.tensor([float("inf"), 65504.0], dtype=torch.float16)
    assert (
        stale_excuses(MathOperation.Celu, src, golden, still_broken, fmt, fmt, **cell)
        == []
    )
    fixed = golden.clone()
    assert [
        e.issue
        for e in stale_excuses(MathOperation.Celu, src, golden, fixed, fmt, fmt, **cell)
    ] == ["#58607"]
    # On a cell the entry does not name there is nothing to go stale.
    other = dict(approx_mode=ApproximationMode.No, dest_acc=DestAccumulation.Yes)
    assert (
        stale_excuses(MathOperation.Celu, src, golden, fixed, fmt, fmt, **other) == []
    )
    # An entry another exclusion already covers is stale too: against a NaN golden the
    # same inf answers are excused before the entry is consulted, so it buys nothing.
    nan_golden = torch.full_like(golden, float("nan"))
    assert stale_excuses(
        MathOperation.Celu, src, nan_golden, still_broken, fmt, fmt, **cell
    ) == [_known_lanes()[MathOperation.Celu][0]]


def test_every_known_lane_entry_names_an_issue_and_a_gateable_cell():
    for op, entries in _known_lanes().items():
        for entry in entries:
            where = f"{op.name}: {entry}"
            assert entry.issue[:1] == "#" and entry.issue[1:].isdigit(), where
            assert entry.inputs and entry.low <= entry.high and entry.why, where
            # A block output is never step-gated, so there would be nothing to keep gated.
            assert has_ulp_gate(entry.output), where


@pytest.mark.parametrize(
    "exitstatus, expected",
    [
        (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED),
        (pytest.ExitCode.TESTS_FAILED, pytest.ExitCode.TESTS_FAILED),
        (pytest.ExitCode.INTERRUPTED, pytest.ExitCode.INTERRUPTED),
        (pytest.ExitCode.INTERNAL_ERROR, pytest.ExitCode.INTERNAL_ERROR),
        (17, 17),
    ],
    ids=["ok", "failed", "interrupted", "internal-error", "pytest-exit-returncode"],
)
def test_an_emit_refusal_fails_only_a_clean_session(monkeypatch, exitstatus, expected):
    """A refused emit must end non-zero, but a session that already ended non-OK keeps
    its own status: a Ctrl-C read as exit 1 is an ordinary test failure to a script."""
    from types import SimpleNamespace

    from helpers import llk_pytest_plugin

    monkeypatch.setattr(ulp_sweep, "EMIT", True)
    # MEASURED is empty: refused for having measured nothing.
    session = SimpleNamespace(
        config=SimpleNamespace(
            option=SimpleNamespace(collectonly=False),
            pluginmanager=SimpleNamespace(get_plugin=lambda name: None),
        ),
        testsfailed=0,
        exitstatus=exitstatus,
    )
    llk_pytest_plugin._finish_ulp_emit(session)
    assert session.exitstatus == expected


@pytest.mark.parametrize(
    "arch",
    [ChipArchitecture.WORMHOLE, ChipArchitecture.BLACKHOLE],
    ids=lambda a: a.value,
)
def test_the_sweep_cells_are_the_ones_testconfig_builds_as_asked(arch, monkeypatch):
    """A cell is swept exactly when the TestConfig built for it runs the Dest it asks
    for: `sweep_cells` leaves out the cells TestConfig would run as another, or a
    measurement would be keyed on a kernel that never ran. The promoted exponent-B ->
    Float16 `No` cells are the ones left out, and their `Yes` twins stay."""
    from helpers.test_config import TestConfig
    from helpers.ulp_sweep import SWEEP_FORMATS

    monkeypatch.setattr(TestConfig, "CHIP_ARCH", arch)
    built_as_asked = [
        (i, o, a, d)
        for i in SWEEP_FORMATS
        for o in SWEEP_FORMATS
        for a in ApproximationMode
        for d in DestAccumulation
        if TestConfig(
            "sources/eltwise_unary_sfpu_test.cpp",
            InputOutputFormat(i, o),
            dest_acc=d,
            skip_build_header=True,
        ).dest_acc
        == d
    ]
    cells = sweep_cells(arch)
    assert cells == built_as_asked
    promoted = {(i, o) for i, o, _, d in cells if d == DestAccumulation.No} ^ {
        (i, o) for i, o, _, d in cells if d == DestAccumulation.Yes
    }
    # The exponent-B input; Float32 unpacks to Dest and is not an outlier.
    assert promoted == {(DataFormat.Float16_b, DataFormat.Float16)}


def test_no_step_budget_names_an_arch_sweep_cells_does_not_filter_for():
    """`sweep_cells` applies only the Dest promotion off Wormhole, not Blackhole's or
    Quasar's format support. A ULP row naming another `arch` would bind on that arch and
    send those unsupported cells to the device: add the filters with the row."""
    from helpers.sfpu_accuracy_budget import _SFPU_ACCURACY_BUDGET, Metric
    from helpers.ulp_table import MEASURED_ARCH

    elsewhere = [
        (op.name, key)
        for op, rows in _SFPU_ACCURACY_BUDGET.items()
        for key, contract in rows.items()
        if contract.metric is Metric.ULP and key.arch not in (None, MEASURED_ARCH)
    ]
    assert elsewhere == []


def test_the_sweep_tile_count_holds_every_swept_value():
    """The exhaustive claim rests on `SWEEP_TILE_COUNT`: a count too small for the
    format's values keeps only the lowest-sorted of them, and nothing else notices."""
    import test_unary_sfpu_ulp as sweep
    from helpers.golden_generators import TILE_DIMENSIONS
    from helpers.ulp_sweep import SWEEP_FORMATS

    lanes = sweep.SWEEP_TILE_COUNT * TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1]
    assert sweep.SWEEP_DIMENSIONS[0] * sweep.SWEEP_DIMENSIONS[1] == lanes
    for fmt in SWEEP_FORMATS:
        assert swept_value_count(fmt) <= lanes, fmt.name


#: The Float32 walk's sample count, written down independently of the code that makes
#: it: one sample per normal bfloat16 bit pattern, 2**16 less the 256 inf/NaN cells and
#: the 256 subnormal ones, which the sweep does not feed. (65,025 distinct bf16 *values*,
#: zero included: -0.0 and +0.0 are two cells but one value.)
_FLOAT32_WALK_SAMPLES = 65024

#: The Float16_b sweep's value count, likewise by hand: the 65,024 nonzero normal bit
#: patterns and one zero (+0.0 and -0.0 are one value); no subnormal.
_BFLOAT16_SWEPT_VALUES = 65025

#: The Float16 sweep's value count, likewise: 2 signs x 30 normal binades x 1,024
#: mantissas, and one zero; no subnormal.
_FLOAT16_SWEPT_VALUES = 61441


def test_the_float32_walk_is_pinned_end_to_end():
    """`swept_value_count` enumerates at most `_SWEEP_TENSOR` lanes, so the check above
    cannot fail for Float32 whatever `_FP32_STRIDE` is: a stride of 2**15 would stop the
    walk near +1.2e-38, having covered only the negatives, and stay green. So the count,
    the ends, the tensor size and what the in-cell phase buys are pinned here: one
    sample per bfloat16 cell, the most negative one included, none of them a bfloat16
    value, and both halves of the cell reached in every binade (a stride of 2**16 + 1
    put a whole binade at one point of its cells)."""
    import test_unary_sfpu_ulp as sweep
    from helpers.golden_generators import TILE_DIMENSIONS
    from helpers.stimuli_generator.strategies.structured import (
        _enumerate_representable,
    )
    from helpers.ulp_sweep import _FP32_STRIDE, _SWEEP_TENSOR

    lanes = sweep.SWEEP_TILE_COUNT * TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1]
    assert _SWEEP_TENSOR == lanes
    walk = _enumerate_representable(
        DataFormat.Float32,
        -math.inf,
        math.inf,
        _SWEEP_TENSOR,
        stride=_FP32_STRIDE,
        normal_only=True,  # as sweep_spec walks it: no subnormal is fed
    )
    assert walk.numel() == _FLOAT32_WALK_SAMPLES
    assert not bool(
        ((walk != 0) & (walk.abs() < torch.finfo(torch.float32).tiny)).any()
    )
    assert swept_value_count(DataFormat.Float32) == walk.numel()
    assert float(walk.min()) == -torch.finfo(torch.float32).max
    assert float(walk.max()) > 3.39e38

    bits = walk.view(torch.int32)
    low = bits & 0xFFFF
    assert bool((low != 0).all()), "a sample is a bfloat16 value"
    assert (bits >> 16).unique().numel() == walk.numel(), "two samples share a cell"
    # Per sign and binade, the share of samples in the lower half of their cell.
    normal = torch.isfinite(walk) & (walk.abs() >= torch.finfo(torch.float32).tiny)
    binade = ((bits >> 23) & 0x1FF)[normal]  # sign and exponent
    lower = (low < 0x8000)[normal].to(torch.float64)
    for b in binade.unique().tolist():
        share = float(lower[binade == b].mean())
        assert 0.3 < share < 0.7, (hex(b), share)


@pytest.mark.parametrize(
    "fmt, count",
    [
        (F16B, _BFLOAT16_SWEPT_VALUES),
        (F16, _FLOAT16_SWEPT_VALUES),
        (F32, _FLOAT32_WALK_SAMPLES),
    ],
    ids=lambda v: v.name if isinstance(v, DataFormat) else str(v),
)
def test_the_sweep_feeds_every_normal_value_and_zero_and_nothing_else(fmt, count):
    """What `sweep_spec` makes `generate_stimuli` feed, end to end on the host, pinned by
    number rather than by `swept_value_count` (which `padding_lanes` reads too, so a
    check through it could not fail): every normal value of the format once and its
    zero -- no subnormal, which the unpack would flush -- then a zero tail that is
    exactly the padding. Drop `normal_only` at either end and the subnormals come back
    while the top real values are masked as padding."""
    import test_unary_sfpu_ulp as sweep
    from helpers.stimuli_generator import generate_stimuli

    src, *_ = generate_stimuli(
        stimuli_format_A=fmt,
        input_dimensions_A=sweep.SWEEP_DIMENSIONS,
        stimuli_format_B=fmt,
        input_dimensions_B=sweep.SWEEP_DIMENSIONS,
        spec_A=sweep_spec(fmt),
    )
    flat = src.flatten().to(torch.float64)
    assert flat.numel() == 2**16
    data, tail = flat[:count], flat[count:]
    finfo = torch.finfo(format_dict[fmt])
    assert not bool(((data != 0) & (data.abs() < finfo.tiny)).any()), "a subnormal"
    assert data.unique().numel() == count, "a value fed twice, or one missing"
    assert int((data == 0).sum()) == (0 if fmt is F32 else 1)
    assert float(data.abs().max()) == finfo.max
    assert bool((tail == 0).all()) and tail.numel() == 2**16 - count
    assert swept_value_count(fmt) == count
    assert int(padding_lanes(src, fmt).sum()) == tail.numel()


@pytest.mark.parametrize(
    "arch, promoted",
    [
        (ChipArchitecture.WORMHOLE, DestAccumulation.Yes),
        (ChipArchitecture.BLACKHOLE, DestAccumulation.Yes),
        (ChipArchitecture.QUASAR, DestAccumulation.No),
    ],
    ids=lambda v: getattr(v, "value", None) or v.name,
)
def test_effective_dest_acc_promotes_an_exponent_b_input_packed_to_float16(
    arch, promoted
):
    """Every arch but Quasar runs an exponent-B input packed to Float16 on a 32-bit Dest;
    Quasar keeps the 16-bit Dest it was asked for. An exponent-A input is never
    promoted."""
    from helpers.data_format_inference import effective_dest_acc

    No = DestAccumulation.No
    assert effective_dest_acc(DataFormat.Float16_b, DataFormat.Float16, No, arch) == (
        promoted
    )
    assert effective_dest_acc(DataFormat.Float16, DataFormat.Float16, No, arch) == No


def test_effective_dest_acc_asks_for_the_chip_only_for_an_outlier(monkeypatch):
    """Block sizing calls it for every variant; an ordinary combination cannot be
    promoted, so it must not need a chip context to say so."""
    import helpers.data_format_inference as inference

    def no_chip():
        raise AssertionError("looked up the chip for a combination it cannot promote")

    monkeypatch.setattr(inference, "get_chip_architecture", no_chip)
    for dest_acc in DestAccumulation:
        assert (
            inference.effective_dest_acc(
                DataFormat.Float16, DataFormat.Float16, dest_acc
            )
            == dest_acc
        )
    assert (
        inference.effective_dest_acc(
            DataFormat.Float16_b, DataFormat.Float16, DestAccumulation.Yes
        )
        == DestAccumulation.Yes
    )
    with pytest.raises(AssertionError) as caught:  # allow-pytest.raises: host-only test
        inference.effective_dest_acc(
            DataFormat.Float16_b, DataFormat.Float16, DestAccumulation.No
        )
    assert "looked up the chip" in str(caught.value)


def test_the_sweep_collects_every_keyed_op_whether_gating_or_emitting(monkeypatch):
    """The key line is the enrolment, and the sweep collects only the ops the table
    names, so both runs take exactly the unary ops that have one. Gating used to take
    only the ops with a step budget somewhere, which left every op on tolerance
    everywhere (Erfc, SigmoidAppx, ...) with recorded figures that no gate run ever
    re-measured."""
    import test_unary_sfpu_ulp as sweep
    from helpers.sfpu_accuracy_budget import _SFPU_ACCURACY_BUDGET, Metric
    from helpers.sfpu_domains import _UNARY_OPS_NOT_SWEPT, sfpu_unary_ops

    monkeypatch.setattr(ulp_sweep, "EMIT", True)
    emitted = set(sweep._sweep_ops())
    monkeypatch.setattr(ulp_sweep, "EMIT", False)
    gated = set(sweep._sweep_ops())

    keyed = {op for op in _SFPU_ACCURACY_BUDGET if op in sfpu_unary_ops()}
    assert emitted == keyed - set(_UNARY_OPS_NOT_SWEPT)
    assert gated == emitted
    tolerance_only = {
        op
        for op in keyed
        if not any(c.metric == Metric.ULP for c in _SFPU_ACCURACY_BUDGET[op].values())
    }
    assert tolerance_only <= gated, "an op on tolerance everywhere is still measured"


def test_the_claim_is_the_whole_format_not_the_drivers_sampling_window():
    """Abs is sampled on (-10, 10) by the functional driver and defined everywhere, so a
    hardware inf at 1e20 against a finite golden is a failure, not an excused lane."""
    src = torch.tensor([1e20], dtype=torch.bfloat16)
    result = torch.tensor([float("inf")], dtype=torch.bfloat16)
    fmt = DataFormat.Float16_b
    assert nonfinite_failures(_OP, src, src.clone(), result, fmt, fmt).any()


def test_an_interval_domain_is_not_clipped_to_the_specs_default_bounds():
    """Reciprocal registers its domain as ``intervals``; ANDing the spec's default
    [0, 1] in as well shrank it to [0.1, 1], so a hardware inf on most of the domain was
    neither counted nor reported."""
    fmt = DataFormat.Float16_b
    src = torch.tensor([-2.0, 0.5, 5.0], dtype=torch.bfloat16)
    golden = 1.0 / src
    result = torch.full_like(src, float("inf"))
    op = MathOperation.Reciprocal
    assert nonfinite_failures(op, src, golden, result, fmt, DataFormat.Float16_b).all()
    # Its pole at zero still excuses a lane.
    zero = torch.tensor([0.0], dtype=torch.bfloat16)
    one = torch.tensor([1.0], dtype=torch.bfloat16)
    assert not nonfinite_failures(op, zero, one, result[:1], fmt, fmt).any()


@pytest.mark.parametrize(
    "op, inside, outside",
    [
        (MathOperation.Reciprocal, [1e-7, -1e-7, 1e-30], []),
        (MathOperation.Log, [1e-7, 1e-30], [-1e-7, -2.0]),
        (MathOperation.Rsqrt, [1e-7], [-1e-7]),
        (MathOperation.Asin, [-1.0, 0.5, 1.0], [-1.5, 2.0]),
    ],
    ids=lambda v: getattr(v, "name", None) or "",
)
def test_the_claim_is_the_singularity_not_the_sampling_guard_band(op, inside, outside):
    """``_SFPU_UNDEFINED_RANGES`` keeps a random draw 1e-6 off Reciprocal's pole, and
    that band was the claim: a bf16 ``1/1e-7`` returning inf was excused in both the
    emitted measurement and the gate. The claim is ``_OP_SINGULARITIES`` -- the point,
    and the side the op is undefined on."""
    fmt = DataFormat.Float16_b
    finite = torch.tensor([1.0], dtype=torch.bfloat16)
    inf = torch.tensor([float("inf")], dtype=torch.bfloat16)
    for value, claimed in [(v, True) for v in inside] + [(v, False) for v in outside]:
        src = torch.tensor([value], dtype=torch.bfloat16)
        assert (
            bool(nonfinite_failures(op, src, finite, inf, fmt, fmt).any()) is claimed
        ), value


def test_the_claim_excludes_the_poles_of_the_gamma_family():
    """Digamma, Lgamma and Polygamma have a pole at every non-positive integer; the
    claim leaves those lanes out (torch's trigamma answers a finite 4e15 at -6.0, the
    kernel inf, and neither is a measurement). -0.0 is the pole at 0; -2.5 is not a
    pole; an op without the lattice keeps every lane."""
    from helpers.ulp_sweep import claimed_lanes

    src = torch.tensor([-3.0, -2.5, -0.0, 0.0, 0.5, 2.0], dtype=torch.float32)
    for op in (MathOperation.Digamma, MathOperation.Lgamma, MathOperation.Polygamma):
        assert claimed_lanes(op, src, DataFormat.Float32).tolist() == [
            False,
            True,
            False,
            False,
            True,
            True,
        ], op.name
    assert claimed_lanes(MathOperation.Abs, src, DataFormat.Float32).all()


def test_the_claim_judges_a_float32_input_as_the_16bit_dest_truncates_it():
    """-8326436.5 truncates to the bf16 pole -8323072 at ``dest_acc=No``; a 32-bit
    Dest keeps it off the lattice."""
    from helpers.ulp_sweep import claimed_lanes

    src = torch.tensor([-8326436.5, -2.5], dtype=torch.float32)
    op, fmt = MathOperation.Polygamma, DataFormat.Float32
    for out in (DataFormat.Float16_b, DataFormat.Float32):
        assert claimed_lanes(op, src, fmt, out, DestAccumulation.No).tolist() == [
            False,
            True,
        ], out.name
        assert claimed_lanes(op, src, fmt, out, DestAccumulation.Yes).all(), out.name


def test_the_polygamma_golden_is_the_pole_on_every_nonpositive_integer():
    """+inf for an odd order (every term of the series is positive), NaN for an even one
    (no limit); torch answers a large finite number there. Off the lattice, torch."""
    from helpers.golden_generators import UnarySFPUGolden

    golden = UnarySFPUGolden()
    golden.data_format = DataFormat.Float32
    for order in (1, 2, 3):
        golden._polygamma_order = order
        for pole in (-6.0, 0.0, -0.0, -1048576.0):
            value = golden._polygamma(pole)
            assert (value == math.inf) if order % 2 else math.isnan(value), (
                order,
                pole,
            )
        want = torch.polygamma(order, torch.tensor(-2.5, dtype=torch.float64)).item()
        assert golden._polygamma(-2.5) == want, order


def test_a_lane_the_dest_lost_is_neither_ranked_nor_a_failure():
    """A Float16 input runs on an fp16 Dest. Under a Float32 output a golden past 65504
    answered with its sign in the Dest's top binade (65536, inf, or I0's 65440 for a true
    65772), or with NaN, is the Dest's overflow, not the op's; the golden's finite 131008
    for an fp16 infinity is the overflow in disguise. So is a flush across 2**-14, either way round. Under a
    Float16 output the Dest is as wide as the output and nothing is excluded, and a
    bf16 Dest has fp32's range."""
    from helpers.ulp_sweep import dest_holds

    tiny = 2.0**-14
    golden = torch.tensor(
        [70000.0, 131008.0, -131008.0, 131008.0, tiny, 0.0, 1.0, 65504.0],
        dtype=torch.float32,
    )
    result = torch.tensor(
        [65536.0, math.inf, math.nan, 65440.0, 0.0, tiny, 1.0, 65504.0],
        dtype=torch.float32,
    )
    lost = [False] * 6 + [True, True]
    fp16, fp32, no = DataFormat.Float16, DataFormat.Float32, DestAccumulation.No
    assert dest_holds(golden, result, fp16, fp32, no).tolist() == lost
    assert dest_holds(golden, result, fp16, fp16, no).all()
    assert dest_holds(golden, result, DataFormat.Float16_b, fp32, no).all()
    assert dest_holds(golden, result, fp16).all()  # variant not named

    src = torch.full((8,), 2.0, dtype=torch.float16)
    assert measurable_mask(src, golden, result, fp16, fp32, no).tolist() == lost
    # The overflow lanes disagree about being non-finite, and the Dest excuses them.
    assert not nonfinite_failures(
        MathOperation.Exp, src, golden, result, fp16, fp32, dest_acc=no
    ).any()


def test_an_answer_the_dest_did_not_saturate_to_stays_judged():
    """Only an answer at the Dest's edge is excused. A wrong-signed answer against an
    overflowed golden (Expm1Cw's #54770: 131008 against -1 or -131008), a same-signed
    one more than 2x short of the edge, an answer 0 where the golden is a normal 1e-4,
    and a pair that are both nonzero are the op's, and ranked; an inf against a golden of 1.0, or of the wrong
    sign against 131008, is a non-finite failure."""
    from helpers.ulp_sweep import dest_holds

    tiny = 2.0**-14
    golden = torch.tensor(
        [131008.0, 131008.0, 70000.0, 1e-4, tiny, 1.0, 131008.0], dtype=torch.float32
    )
    result = torch.tensor(
        [-1.0, -131008.0, 30000.0, 0.0, 3 * tiny, math.inf, -math.inf],
        dtype=torch.float32,
    )
    fp16, fp32, no = DataFormat.Float16, DataFormat.Float32, DestAccumulation.No
    assert dest_holds(golden, result, fp16, fp32, no).all()

    src = torch.full((7,), 2.0, dtype=torch.float16)
    ranked = measurable_mask(src, golden, result, fp16, fp32, no)
    assert ranked.tolist() == [True] * 5 + [False, False]
    assert nonfinite_failures(
        MathOperation.Exp, src, golden, result, fp16, fp32, dest_acc=no
    ).tolist() == [False] * 5 + [True, True]


def test_the_near_zero_floor_is_derived_from_the_band_and_capped():
    """A cell demoted only by near-zero lanes gets the floor that covers them -- exactly
    the band's largest absolute error, no margin -- and the budget of the lanes outside
    it. A cell that fits gets none; one whose band error is past the cap
    stays demoted; a lane outside the band (|golden| >= 1% of the cell's maximum) is
    never rescued."""
    from helpers.ulp_sweep import EMIT_MAX_NEAR_ZERO_ATOL, near_zero_floor

    fmt = DataFormat.Float16_b
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)
    golden[30], golden[31] = 1e-6, 0.5
    result = golden.clone()
    result[1] = 1.5078125  # one bf16 step above 1.5
    result[30] = 0.0  # the hardware flushed a 1e-6 answer: ~14,000 steps, 1e-6 absolute
    mask = torch.ones(32, dtype=torch.bool)

    def floor(res):

        return near_zero_floor(
            golden, res, ulp_distance(golden, res, flush_subnormals=True), mask, fmt
        )

    residual, atol = floor(result)
    assert residual == 1
    # Exactly the band's largest absolute error: bf16's 1e-6, flushed to 0.
    assert atol == float(torch.tensor(1e-6, dtype=torch.bfloat16))

    fits = golden.clone()
    fits[1] = 1.5078125
    assert floor(fits) is None  # under the ceiling without a floor

    past_cap = golden.clone()
    past_cap[30] = 0.01  # 0.01 absolute, past EMIT_MAX_NEAR_ZERO_ATOL
    assert 0.01 > EMIT_MAX_NEAR_ZERO_ATOL
    # The capped floor is still returned, with the lanes it leaves past the ceiling, so
    # the demotion note can name the floor that was tried.
    residual, atol = floor(past_cap)
    assert atol == EMIT_MAX_NEAR_ZERO_ATOL and residual > 6

    outside_band = golden.clone()
    outside_band[31] = 0.0  # |golden| = 0.5 is not under 1% of 1.5
    assert floor(outside_band) is None


def test_a_32bit_or_bf16_dest_holds_every_answer_the_output_can():
    """Only an fp16 Dest at dest_acc=No narrows. At dest_acc=Yes the Dest is 32-bit,
    and a bf16 Dest has fp32's exponent range although bfloat16.max < float32.max: an
    infinite golden there is the kernel's answer to judge, not the Dest's overflow."""
    from helpers.ulp_sweep import dest_holds

    golden = torch.tensor([70000.0, math.inf, 3.4e38, 1e-5], dtype=torch.float32)
    result = torch.zeros_like(golden)  # what an fp16 Dest's flush would hand back
    for in_fmt, dest in (
        (DataFormat.Float16, DestAccumulation.Yes),
        (DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float32, DestAccumulation.No),
    ):
        assert dest_holds(golden, result, in_fmt, DataFormat.Float32, dest).all(), (
            in_fmt,
            dest,
        )


def test_a_floor_short_of_the_ceiling_is_returned_for_the_demotion_note():
    """near_zero_floor hands back the floor it tried even when the lanes outside it stay
    past the ceiling, so _decide can demote the cell and say what it would have needed.
    """
    from helpers.ulp_sweep import near_zero_floor

    golden = torch.tensor([1.0, 1e-6, 2.0], dtype=torch.float32)
    result = torch.tensor([1.0, 0.0, 2.5], dtype=torch.float32)
    distance = torch.tensor([0, 14000, 9000])
    mask = torch.ones(3, dtype=torch.bool)
    residual, atol = near_zero_floor(
        golden, result, distance, mask, DataFormat.Float16_b
    )
    assert residual == 9000 and atol > 0


def test_the_trig_claim_reaches_65504_on_a_wide_input():
    """Sin/Cos/Tan are accurate out to 65504, so the ranking mask keeps those lanes;
    past it the range reduction breaks and the lanes leave the claim."""
    from helpers.ulp_sweep import claimed_lanes

    src = torch.tensor([3.0, 4.0, 60000.0, 70000.0], dtype=torch.bfloat16)
    claimed = claimed_lanes(
        MathOperation.Sin,
        src,
        DataFormat.Float16_b,
        DataFormat.Float16_b,
        DestAccumulation.No,
    )
    assert claimed.tolist() == [True, True, True, False]
    # ranked_mask is the measurable lanes within that claim -- what the sweep ranks.
    golden = torch.zeros(4, dtype=torch.bfloat16)
    ranked = ranked_mask(
        MathOperation.Sin,
        src,
        golden,
        golden.clone(),
        DataFormat.Float16_b,
        DataFormat.Float16_b,
        DestAccumulation.No,
    )
    assert ranked.tolist() == [True, True, True, False]


def test_a_cell_whose_every_claimed_lane_is_non_finite_records_no_figure():
    """Over no ranked lane `ulp_stats` reads `max: 0`, which written as `measured: 0`
    would claim the rest of the cell bit-exact; there is no rest."""

    src = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    golden = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    result = torch.tensor([float("inf"), float("inf")], dtype=torch.bfloat16)
    overflowed = torch.tensor([True, True])
    stats = ulp_stats(ulp_distance(golden, result), ~overflowed)
    verdict = nonfinite_verdict(overflowed, src, golden, result, stats)
    assert (verdict.nonfinite, verdict.measured) == (2, None)


# ── A cell's `domains` list ──────────────────────────────────────────────────


def test_the_tensor_side_draws_the_rule_on_the_lanes_and_measures_each_entry(
    monkeypatch,
):
    """`binade_maxima` profiles the received inputs, `domain_figures` measures each
    drawn span over exactly its lanes, and `entry_readings` is what the recorder
    writes. The knobs are lowered so a tiny tensor can split."""
    import helpers.ulp_domains as rule
    from helpers.ulp_sweep import (
        domain_figures,
        entry_readings,
        floor_covered,
        floored_profile,
    )

    monkeypatch.setattr(rule, "DOMAIN_MIN_BIN_LANES", 1)
    monkeypatch.setattr(rule, "DOMAIN_MIN_LANES", 1)
    x = torch.tensor(
        [-3.0, -2.5, -1.5, -1.25, 0.5, 0.75, 1.5, 2.0, 3.0, 6.0], dtype=torch.float64
    )
    distance = torch.tensor([1, 2, 1, 1, 0, 0, 9, 8, 1, 1])
    ranked = torch.ones(10, dtype=torch.bool)
    present = torch.ones(10, dtype=torch.bool)
    # -1.25 is ranked, and a floor covers it -- a tiny answer off by less than the
    # floor -- so it is held at 0 steps.
    golden = x.to(torch.float32)
    golden[3] = 1e-7
    result = golden.clone()
    result[3] = 0.0
    judged, bins = floored_profile(golden, result, x, distance, ranked, present, 1e-6)
    assert judged.tolist() == [i != 3 for i in range(10)]
    assert bins[rule.binade(-3.0)] == (2, 2, 2) and bins[rule.binade(1.5)] == (1, 1, 9)
    assert bins[rule.binade(-1.25)] == (2, 2, 1)  # -1.5 shares the binade
    spans = rule.partition(bins, usable_budget_ceiling(F16B))
    # 1.5 (9 steps) and 2.0 (8 steps) are both past bf16's ceiling of 6: one bad range.
    assert spans == ((-_INF, 1.0, True), (1.0, 4.0, True), (4.0, _INF, True))
    drawn = domain_figures(spans, x, distance, ranked, judged)
    assert [d.ulp for d in drawn] == [2, 9, 1]
    assert entry_readings(drawn, x, distance, ranked, judged) == (
        # six ranked lanes; the figure is over the five outside the floor
        Reading(-_INF, 1.0, 2, 6),
        Reading(1.0, 4.0, 9, 3),
        Reading(4.0, _INF, 1, 1),
    )
    # An entry the floor covers entirely reads 0, not null: its lanes are held.
    none_outside = torch.zeros(10, dtype=torch.bool)
    assert domain_figures(spans, x, distance, ranked, none_outside)[0].ulp == 0
    golden = torch.tensor([1.0, 1e-6, 2.0], dtype=torch.float32)
    result = torch.tensor([1.0, 0.0, 2.5], dtype=torch.float32)
    assert floor_covered(
        golden, result, torch.ones(3, dtype=torch.bool), 1e-6
    ).tolist() == [False, True, False]
    assert not floor_covered(
        golden, result, torch.ones(3, dtype=torch.bool), None
    ).any()


def test_the_tensor_binade_is_the_scalar_one():
    """`binade_maxima` indexes lanes with `_binade_index`; the list's edges come from
    `ulp_domains.binade`. A negative power of two is where they could part: it is the
    low edge of the binade above it."""
    import helpers.ulp_domains as rule
    from helpers.ulp_sweep import _binade_index

    values = [0.0, 2.0**-149, -(2.0**-149), 2.0**-126, -(2.0**-126), -0.5, -0.75]
    values += [-1.0, -1.5, -2.0, 1.0, 1.5, 2.0, 65504.0, -65504.0, -(2.0**127)]
    x = torch.tensor(values, dtype=torch.float64)
    assert _binade_index(x).tolist() == [rule.binade(v) for v in values]


def test_a_worst_lane_on_a_negative_edge_is_figured_on_the_side_its_entry_holds():
    """-1 belongs to the entry that opens at -1, and its binade is that entry's. When it
    shared (-2, -1]'s, its 9 steps classed that binade past the ceiling, and the
    offline view, which figures an entry from its binades, read the entry below -1 as
    9 where the emitter measures 1. The two now draw the same list."""
    import helpers.ulp_domains as rule
    from helpers.ulp_budget_diff import draw, read_cells
    from helpers.ulp_sweep import binade_maxima, domain_figures

    n = 2048  # each run of lanes is a domain of its own at DOMAIN_MIN_LANES
    x = torch.tensor([-1.5] * n + [-1.0] * n + [-0.75] * n, dtype=torch.float64)
    distance = torch.tensor([1] * n + [9] * n + [9] * n)
    every = torch.ones(3 * n, dtype=torch.bool)
    bins = binade_maxima(x, distance, every, every)
    drawn = domain_figures(
        rule.partition(bins, usable_budget_ceiling(F16B)), x, distance, every, every
    )
    assert drawn == (Domain(-_INF, -1, 1), Domain(-1, _INF, 9))
    key = cell_key(F16B, F16B, NO, D_NO)
    record_of = Measurement(
        op=MathOperation.Tanh,
        key=key,
        worst=9,
        lanes=3 * n,
        bins=tuple((b, *counts) for b, counts in sorted(bins.items())),
    )
    (reading,) = read_cells([record_of]).values()
    assert draw(reading, key) == drawn


def test_a_cell_whose_strays_beat_every_figure_writes_a_list_its_gate_passes(
    monkeypatch,
):
    """A null entry's few ranked lanes are held to the row's largest figure, so one
    worse than every figure takes a figure of its own; and with no ranked span at all
    the cell gets no list rather than one the registry refuses."""
    import helpers.ulp_domains as rule
    from helpers.ulp_sweep import binade_maxima, domain_figures

    monkeypatch.setattr(rule, "DOMAIN_MIN_LANES", 1)
    good = [1.0 + i / 64 for i in range(64)]  # [1, 2): 64 ranked lanes at 1 step
    x = torch.tensor([-1.5] * 64 + good, dtype=torch.float64)
    distance = torch.tensor([30] * 64 + [1] * 64)
    ranked = torch.tensor([i < 3 for i in range(64)] + [True] * 64)  # 3 strays at 30
    present = torch.ones(128, dtype=torch.bool)
    bins = binade_maxima(x, distance, ranked, present)
    spans = rule.partition(bins, usable_budget_ceiling(F16B))
    assert [r for _, _, r in spans] == [False, True]
    drawn = domain_figures(spans, x, distance, ranked, ranked)
    assert [d.ulp for d in drawn] == [30, 1]
    few = torch.tensor([i < 3 for i in range(128)])
    nothing = rule.partition(
        binade_maxima(x, distance, few, present), usable_budget_ceiling(F16B)
    )
    assert domain_figures(nothing, x, distance, few, few) == ()

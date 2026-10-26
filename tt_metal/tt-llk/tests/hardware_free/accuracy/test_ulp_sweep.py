# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for ``helpers/ulp_sweep.py``: which lanes the sweep ranks, the
lists it draws, and the emitter that folds a sweep back into a typed table."""

import math
from dataclasses import replace
from types import SimpleNamespace

import helpers.ulp_domains as rule
import pytest
import torch
from conftest import refuses
from helpers import llk_pytest_plugin, ulp_sweep
from helpers.chip_architecture import ChipArchitecture
from helpers.data_format_inference import effective_dest_acc
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    BinarySFPUGolden,
    UnarySFPUGolden,
    convert_inf_to_value,
    get_golden_generator,
    widen_to_float32,
)
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    format_dict,
)
from helpers.stimuli_generator import StimuliSpec
from helpers.stimuli_generator.strategies.structured import _enumerate_representable
from helpers.test_config import TestConfig
from helpers.ulp import has_ulp_gate, ulp_distance, ulp_stats
from helpers.ulp_domains import Domain
from helpers.ulp_sweep import (
    _FP32_STRIDE,
    _KNOWN_NONFINITE_LANES,
    _SWEEP_TENSOR,
    DOMAINS,
    EMIT_MAX_NEAR_ZERO_ATOL,
    MEASURED,
    SWEEP_FORMATS,
    EmitRefused,
    Floored,
    Unmeasurable,
    _binade_index,
    _verdict,
    binade_maxima,
    cell_key,
    claimed_lanes,
    dest_holds,
    domain_figures,
    entry_readings,
    export_measured,
    finish_emit,
    floor_covered,
    floored_profile,
    golden_input,
    measurable_mask,
    merge_measured,
    near_zero_floor,
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

F16, F16B, F32 = DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32
NO, YES = ApproximationMode.No, ApproximationMode.Yes
D_NO, D_YES = DestAccumulation.No, DestAccumulation.Yes
GELU = MathOperation.Gelu
# An op defined everywhere, so the non-finite tests turn on the exclusion they mean.
_OP = MathOperation.Abs
# The key of a cell whose coordinates a test is not about.
_CELL = cell_key(F16, F16, NO, D_NO)
# A floored reading: the worst lane over every lane, and (residual, atol) outside it.
_WORST, _FLOOR = 14337, (2, 2.4e-7)
_INF = math.inf
_ONE = (Domain(-_INF, _INF, 5),)
_TWO = (Domain(-_INF, 0, 5), Domain(0, _INF, 2))


@pytest.fixture(autouse=True)
def _host_only_arch(monkeypatch):
    """`sweep_cells` asks the chip which cells promote; judge the Wormhole table
    whatever CHIP_ARCH the host sets."""
    import helpers.chip_architecture as chip

    monkeypatch.setenv("CHIP_ARCH", "wormhole")
    monkeypatch.setattr(chip, "_cached_chip_architecture", None)


@pytest.fixture(autouse=True)
def _no_records():
    MEASURED.clear()
    DOMAINS.clear()
    yield
    MEASURED.clear()
    DOMAINS.clear()


def _failures(op, src, golden, result, in_fmt, out_fmt, approx=NO, dest=D_YES, **kw):
    return nonfinite_failures(
        op,
        src,
        golden,
        result,
        in_fmt,
        out_fmt,
        approx_mode=approx,
        dest_acc=dest,
        **kw,
    )


# ── The emitter ───────────────────────────────────────────────────────────────

# A table with a row of each kind write_table has to tell apart.
_TABLE = {
    GELU: (
        Row(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05, note="an op-wide anchor"),
        Row(input=F16, output=F16, measured=7, note="superseded"),
        Row(input=F16B, output=F32, measured=9, note="a pair not swept here"),
        Row(input=F16B, output=F16B, arch=ChipArchitecture.BLACKHOLE, measured=44),
        Row(output=F32, measured=0, note="a wildcard"),
        Row(
            input=F16,
            output=F16,
            approx=NO,
            dest=D_YES,
            metric=Metric.TOLERANCE,
            atol=0.2,
        ),
    ),
    MathOperation.Log1p: (Row(input=F16, output=F16B, metric=Metric.TOLERANCE),),
}


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "budget.yaml"
    write_table_file(_TABLE, path)
    return path


def _rows(path, op=GELU):
    return {row.key: row for row in load_table(path)[op]}


def _pair(input_format, output_format, **key):
    return BudgetKey(input_format=input_format, output_format=output_format, **key)


def _record_grid(op, in_fmt, out_fmt, value, **kw):
    """Every approx x dest cell of one pair, so the emitter collapses them."""
    for approx in ApproximationMode:
        for dest in DestAccumulation:
            record(op, cell_key(in_fmt, out_fmt, approx, dest), value, **kw)


def _session(exitstatus, testsfailed=0):
    return SimpleNamespace(
        config=SimpleNamespace(
            option=SimpleNamespace(collectonly=False),
            pluginmanager=SimpleNamespace(get_plugin=lambda name: None),
        ),
        testsfailed=testsfailed,
        exitstatus=exitstatus,
    )


def test_only_the_pairs_this_run_measured_are_replaced(table):
    """Every row a run did not measure stays: the other pair, the other arch's row, the
    wildcard, the op-wide atol anchor (which the emitted tolerance rows fall back to
    rather than retract), the pinned `atol` row on the very pair measured (a judgement
    no measurement re-derives), the other op; an op not yet in the table is added."""
    record(GELU, _CELL, 5)
    record(MathOperation.Sqrt, _CELL, 1)
    _record_grid(GELU, F16B, F16B, 100)  # past bf16's ceiling: a numberless row
    assert set(write_table(table)) == {GELU, MathOperation.Sqrt}
    after = _rows(table)
    assert after[_pair(F16, F16)] == Row(input=F16, output=F16, measured=5)
    assert after[_pair(F16B, F16B)] == Row(
        input=F16B, output=F16B, metric=Metric.TOLERANCE, measured=100
    )
    for row in _TABLE[GELU][:1] + _TABLE[GELU][2:]:
        assert after[row.key] == row
    assert load_table(table)[MathOperation.Log1p] == _TABLE[MathOperation.Log1p]
    assert _rows(table, MathOperation.Sqrt) == {
        _pair(F16, F16): Row(input=F16, output=F16, measured=1)
    }


def test_a_cell_recorded_twice_keeps_the_worst_lane():
    """Last-write-wins is the polarity that hides error."""
    record(GELU, _CELL, 12)
    record(GELU, _CELL, 3)
    assert MEASURED[GELU][_CELL] == 12


def test_an_anti_diagonal_grid_is_refused_rather_than_collapsed(table):
    """Each field's droppability is decided alone; on an anti-diagonal both would be
    dropped and one measurement would overwrite the other."""
    record(GELU, _CELL, 4)
    record(GELU, cell_key(F16, F16, YES, D_YES), 9)
    before = table.read_text()
    with pytest.raises(EmitRefused):  # allow-pytest.raises: host-only test
        write_table(table)
    assert table.read_text() == before


def test_a_kept_row_with_the_key_an_emit_writes_is_refused(tmp_path):
    """The pair's cells all agree and collapse onto the pinned `atol` row's own key: one
    row would silently replace the other, and only whether the cells happened to agree
    decided which. Nothing is written."""
    path = tmp_path / "budget.yaml"
    pinned = Row(input=F16, output=F16, metric=Metric.TOLERANCE, atol=0.2)
    write_table_file({GELU: (pinned,)}, path)
    _record_grid(GELU, F16, F16, 3)
    before = path.read_text()
    with pytest.raises(EmitRefused):  # allow-pytest.raises: host-only test
        write_table(path)
    assert path.read_text() == before


def test_the_emitted_figure_is_the_measurement():
    """No headroom and no cap: a gate while it fits the ceiling, tolerance past it."""
    assert _verdict(6, F16B) is Metric.ULP
    assert _verdict(7, F16B) is Metric.TOLERANCE
    assert _verdict(419431, F32) is Metric.TOLERANCE


def test_xdist_workers_measurements_merge_worst_lane_first():
    """The controller merges the workers' exports, keeping the worse reading of a cell
    two workers measured; a floored reading survives the merge, and the same blob
    merged twice is one reading, not a differing pair that drops its floor."""
    record(GELU, _CELL, 5)
    worker_a = export_measured()
    MEASURED.clear()
    other, floored = cell_key(F16, F16, YES, D_NO), cell_key(F16, F16, YES, D_YES)
    record(GELU, _CELL, 9)
    record(GELU, other, 2)
    record(GELU, floored, _WORST, floor=_FLOOR)
    worker_b = export_measured()
    MEASURED.clear()
    merge_measured(worker_a)
    merge_measured(worker_b)
    merge_measured(worker_b)  # the same reading again, as two workers can send it
    merge_measured(None)  # a worker that recorded nothing
    assert MEASURED == {GELU: {_CELL: 9, other: 2, floored: Floored(_WORST, *_FLOOR)}}


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
    "arch, clean",
    [(ChipArchitecture.BLACKHOLE, True), (ChipArchitecture.WORMHOLE, False)],
    ids=["other-arch", "unclean-session"],
)
def test_emit_refuses_what_the_session_cannot_vouch_for(table, arch, clean):
    _record_grid(GELU, F16, F16, 5)
    before = table.read_text()
    with pytest.raises(EmitRefused):  # allow-pytest.raises: host-only test
        finish_emit(arch, clean, table)
    assert table.read_text() == before


def test_emit_refuses_a_partly_measured_grid(table):
    """A write replaces every row of a touched pair; a run narrowed inside one would
    drop the rows of the cells it never reached."""
    record(GELU, _CELL, 5)
    before = table.read_text()
    with pytest.raises(EmitRefused):  # allow-pytest.raises: host-only test
        finish_emit(ChipArchitecture.WORMHOLE, True, table)
    assert table.read_text() == before


def test_a_whole_table_emit_over_the_real_table_loads(tmp_path):
    """Every op the emit sweeps, measured on every cell, written into a copy of the
    checked-in table, reads back."""
    import test_unary_sfpu_ulp as sweep

    path = tmp_path / "budget.yaml"
    write_table_file(load_table(), path)
    ops = sweep._sweep_ops()
    for op in ops:
        for cell in sweep_cells(ChipArchitecture.WORMHOLE):
            record(op, cell_key(*cell), 1)
    written = finish_emit(ChipArchitecture.WORMHOLE, True, path)
    assert set(written) == set(ops)
    load_table(path)


def test_the_plugin_prints_the_emit_summary_on_a_clean_session(
    table, monkeypatch, capsys
):
    monkeypatch.setattr(ulp_sweep, "EMIT", True)
    monkeypatch.setattr("helpers.ulp_table.TABLE_PATH", table)
    monkeypatch.setattr(
        llk_pytest_plugin, "get_chip_architecture", lambda: ChipArchitecture.WORMHOLE
    )
    _record_grid(GELU, F16, F16, 5)
    session = _session(pytest.ExitCode.OK)
    llk_pytest_plugin._finish_ulp_emit(session)
    assert capsys.readouterr().out == "--ulp-emit: rewrote 1 op(s) in budget.yaml\n"
    assert session.exitstatus == pytest.ExitCode.OK
    assert _rows(table)[_pair(F16, F16)].measured == 5


@pytest.mark.parametrize(
    "exitstatus, expected",
    [
        (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED),
        (pytest.ExitCode.INTERRUPTED, pytest.ExitCode.INTERRUPTED),
    ],
    ids=["ok", "interrupted"],
)
def test_an_emit_refusal_fails_only_a_clean_session(monkeypatch, exitstatus, expected):
    """A refused emit ends non-zero, but a session that already ended non-OK keeps its
    own status: a Ctrl-C must not read as an ordinary test failure to a script."""
    monkeypatch.setattr(ulp_sweep, "EMIT", True)
    session = _session(exitstatus)  # refused for having measured nothing
    llk_pytest_plugin._finish_ulp_emit(session)
    assert session.exitstatus == expected


@pytest.mark.parametrize(
    "testsfailed, exitstatus",
    [
        (3, pytest.ExitCode.TESTS_FAILED),
        # The hook can see the count before pytest has set the status from it.
        (3, pytest.ExitCode.OK),
        (0, pytest.ExitCode.INTERRUPTED),
        (0, 17),  # pytest.exit(returncode=17)
    ],
    ids=[
        "failed",
        "failed-status-not-yet-set",
        "interrupted",
        "pytest-exit-returncode",
    ],
)
def test_a_session_that_measured_its_grids_but_did_not_end_clean_writes_nothing(
    table, monkeypatch, testsfailed, exitstatus
):
    """Every grid complete, so only the session's own status stands between it and
    the checked-in table: a failure elsewhere, a Ctrl-C or a `pytest.exit` refuses the
    write, and the session ends non-zero."""
    monkeypatch.setattr(ulp_sweep, "EMIT", True)
    monkeypatch.setattr("helpers.ulp_table.TABLE_PATH", table)
    monkeypatch.setattr(
        llk_pytest_plugin, "get_chip_architecture", lambda: ChipArchitecture.WORMHOLE
    )
    _record_grid(GELU, F16, F16, 5)
    before = table.read_text()
    session = _session(exitstatus, testsfailed)
    llk_pytest_plugin._finish_ulp_emit(session)
    assert table.read_text() == before
    assert session.exitstatus != pytest.ExitCode.OK
    if exitstatus != pytest.ExitCode.OK:
        assert session.exitstatus == exitstatus


def test_a_not_measurable_verdict_is_written_with_both_figures_and_why(table):
    """The lanes that park the cell, the worst of the rest, and why as the row's note;
    a cell with no rankable lane at all records no figure."""
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
    every = torch.tensor([True, True, True, True])
    no_lane = nonfinite_verdict(
        every, src, golden, result, ulp_stats(ulp_distance(golden, result), ~every)
    )
    assert (no_lane.nonfinite, no_lane.measured) == (4, None)


def test_a_floored_cell_is_written_with_its_floor_and_demoted_when_it_falls_short(
    table,
):
    _record_grid(GELU, F16, F16, _WORST, floor=_FLOOR)
    write_table(table)
    row = _rows(table)[_pair(F16, F16)]
    assert (row.measured, row.near_zero_atol, row.contract.max_ulp) == (2, 2.4e-7, 2)
    MEASURED.clear()
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


def test_a_cell_with_a_list_is_written_as_the_list_and_its_floor_and_nothing_else(
    table,
):
    _record_grid(GELU, F16, F16, 5, domains=_ONE)
    write_table(table)
    row = _rows(table)[_pair(F16, F16)]
    assert row == Row(input=F16, output=F16, domains=_ONE)
    assert row.contract.max_ulp == 5
    # The entries were measured outside the floor, so it rides with the list even when
    # the cell as a whole is past the ceiling.
    MEASURED.clear()
    DOMAINS.clear()
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


# ── Which lanes count ─────────────────────────────────────────────────────────


def test_a_nonfinite_disagreement_is_reported_rather_than_only_masked_out():
    """The sweep ranks a distance rather than calling passed_test, so a hardware
    overflow would otherwise leave the statistics clean and pass."""
    src = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    result = torch.tensor([1.0, float("inf"), 3.0], dtype=torch.bfloat16)
    assert measurable_mask(src, golden, result, F16B, F16B, D_YES).tolist() == [
        True,
        False,
        True,
    ]
    assert _failures(_OP, src, golden, result, F16B, F16B).tolist() == [
        False,
        True,
        False,
    ]
    # The claim is the whole format, not the driver's sampling window: Abs at 1e20.
    far = torch.tensor([1e20], dtype=torch.bfloat16)
    assert _failures(_OP, far, far.clone(), result[1:2], F16B, F16B).any()


def test_a_flushed_subnormal_input_is_not_a_nonfinite_failure():
    tiny = float(torch.finfo(torch.bfloat16).smallest_normal) / 2
    src = torch.tensor([tiny], dtype=torch.bfloat16)
    golden = torch.tensor([1.0], dtype=torch.bfloat16)
    result = torch.tensor([float("inf")], dtype=torch.bfloat16)
    assert not _failures(_OP, src, golden, result, F16B, F16B).any()


def test_a_finite_answer_to_an_infinite_golden_is_a_nonfinite_failure():
    """Only a saturated store is excused past the output range: the largest finite
    bf16, or an infinity of the wrong sign, against +inf is the kernel being wrong."""
    src = torch.tensor([5.0, 5.0, 5.0], dtype=torch.bfloat16)
    golden = torch.full((3,), float("inf"), dtype=torch.bfloat16)
    result = torch.tensor([3.39e38, float("-inf"), float("inf")], dtype=torch.bfloat16)
    assert _failures(_OP, src, golden, result, F16B, F16B).tolist() == [
        True,
        True,
        False,
    ]


def test_an_infinite_golden_on_the_singularity_point_is_not_a_nonfinite_failure():
    """The value at a pole is a limit (rsqrt(-0) answers +inf against -inf; an fp16
    Dest answers log(0) with -130560); one step off it a finite answer fails."""
    src = torch.tensor([-0.0, 0.0, 1e-30], dtype=torch.bfloat16)
    golden = torch.tensor(
        [float("-inf"), float("inf"), float("inf")], dtype=torch.bfloat16
    )
    result = torch.tensor([float("inf"), -130560.0, 3.0e38], dtype=torch.bfloat16)
    assert _failures(MathOperation.Rsqrt, src, golden, result, F16B, F16B).tolist() == [
        False,
        False,
        True,
    ]


def test_the_unpack_format_bounds_which_inputs_count():
    """A Float32 input into a Float16 output at dest_acc=No unpacks into a Float16
    Dest: below 2**-14 flushed, past 65504 saturated, neither the op's. A bfloat16 Dest
    has fp32's exponent range and saturates nothing."""
    src = torch.tensor([1e-6, 1.0, 1e6], dtype=torch.float32)
    golden = torch.tensor([1.0, 1.0, float("inf")], dtype=torch.float16)
    result = torch.tensor([0.0, 1.0, 11.8], dtype=torch.float16)
    assert measurable_mask(src, golden, result, F32, F16, D_NO).tolist() == [
        False,
        True,
        False,
    ]
    assert measurable_mask(src, golden, result, F32, F16, D_YES)[0]
    assert not _failures(_OP, src, golden, result, F32, F16, dest=D_NO).any()
    assert _failures(_OP, src, golden, result, F32, F16, dest=D_YES)[2]
    largest = torch.tensor(
        [float.fromhex("0x1.fe55920000000p+127")], dtype=torch.float32
    )
    assert float(largest[0]) > torch.finfo(torch.bfloat16).max
    same = largest.to(torch.bfloat16)
    assert measurable_mask(largest, same, same, F32, F16B, D_NO).tolist() == [True]


def test_an_fp16_pack_clamp_is_a_saturated_store():
    """The fp16 pack clamps an out-of-range value to +-65504 of its sign; the wrong
    sign is still the kernel's."""
    src = torch.tensor([5.0, 5.0], dtype=torch.bfloat16)
    golden = torch.tensor([float("-inf"), float("-inf")], dtype=torch.float16)
    result = torch.tensor([-65504.0, 65504.0], dtype=torch.float16)
    assert _failures(_OP, src, golden, result, F16B, F16).tolist() == [False, True]


def test_a_golden_past_the_output_range_is_not_a_nonfinite_failure():
    """Saturating there is the store doing what it must; the same golden into an output
    that could hold it fails."""
    src = torch.tensor([5.0], dtype=torch.bfloat16)
    golden = torch.tensor([1e5], dtype=torch.bfloat16)
    result = torch.tensor([float("nan")], dtype=torch.bfloat16)
    assert not _failures(_OP, src, golden, result, F16B, F16).any()
    assert _failures(_OP, src, golden, result, F16B, F16B).any()


def test_the_sweeps_own_zero_padding_is_not_data():
    """The last 511 bf16 lanes are zeros the sweep never chose to feed; by position,
    since one legitimate 0.0 is swept too."""
    swept = swept_value_count(F16B)
    pad_lanes = 2**16 - _BFLOAT16_SWEPT_VALUES
    src = torch.zeros(swept + pad_lanes, dtype=torch.bfloat16)
    pad = padding_lanes(src, F16B)
    assert int(pad.sum()) == pad_lanes
    assert not bool(pad[:swept].any()) and bool(pad[swept:].all())
    golden = torch.full_like(src, 1.0)
    result = torch.full_like(src, float("inf"))
    assert not measurable_mask(src, golden, result, F16B, F16B, D_YES)[swept:].any()
    assert not _failures(_OP, src, golden, result, F16B, F16B)[swept:].any()


def test_the_claim_limit_follows_what_the_stimuli_format_can_reach():
    """fp16 ends at 65504, inside the range the trig kernels reduce, so an fp16 input
    carries no limit; from bf16, 65504 rounds to 65536, past the limit, and is excused.
    """
    op = MathOperation.Sin
    top = torch.tensor([65504.0, 4.0], dtype=torch.float16)
    golden = torch.tensor([-1.0, -1.0], dtype=torch.float16)
    result = torch.tensor([float("inf"), float("inf")], dtype=torch.float16)
    assert _failures(op, top, golden, result, F16, F16).tolist() == [True, True]
    assert _failures(
        op,
        top.to(torch.bfloat16),
        golden.to(torch.bfloat16),
        result.to(torch.bfloat16),
        F16B,
        F16B,
    ).tolist() == [False, True]
    # The ranking mask keeps the claimed lanes and drops the rest.
    src = torch.tensor([3.0, 4.0, 60000.0, 70000.0], dtype=torch.bfloat16)
    zeros = torch.zeros(4, dtype=torch.bfloat16)
    ranked = ranked_mask(op, src, zeros, zeros.clone(), F16B, F16B, D_NO)
    assert ranked.tolist() == [True, True, True, False]


@pytest.mark.parametrize(
    "fmt, dest_acc, keeps_sign",
    [(F16B, D_NO, False), (F32, D_YES, True)],
    ids=["unpack-drops-the-sign", "unpack-to-dest-keeps-it"],
)
def test_the_golden_sees_the_zero_the_kernel_receives(fmt, dest_acc, keeps_sign):
    dtype = torch.float32 if fmt.is_32_bit() else torch.bfloat16
    src = torch.tensor([-0.0, 0.0, -1.5, 2.0**-100], dtype=dtype)
    received = golden_input(src, fmt, dest_acc)
    assert bool(torch.signbit(received[0])) is keeps_sign
    assert not torch.signbit(received[1])
    assert torch.equal(received, src)  # -0.0 == 0.0: the values themselves are kept


def test_the_unary_golden_keeps_a_float32_output_at_fp32_on_a_16bit_input():
    """#58590: 1/3 is inexact in both formats, so the golden must be fp32's 1/3."""
    src = torch.full((1024,), 3.0, dtype=torch.bfloat16)
    golden = get_golden_generator(UnarySFPUGolden)(
        MathOperation.Reciprocal, src, F32, D_YES, F16B, [32, 32]
    )
    assert golden.dtype == torch.float32
    fp32 = float(torch.tensor(1 / 3, dtype=torch.float32))
    assert float(golden[0]) == fp32
    assert fp32 != float(torch.tensor(1 / 3, dtype=torch.bfloat16))


def test_the_unary_golden_does_not_overflow_a_float16_input_into_a_float32_output():
    """cosh(-89.375) is 3.27e38: finite in fp32, past fp16."""
    src = torch.full((1024,), -89.375, dtype=torch.float16)
    golden = get_golden_generator(UnarySFPUGolden)(
        MathOperation.Cosh, src, F32, D_YES, F16, [32, 32]
    )
    assert golden.dtype == torch.float32
    assert math.isclose(float(golden[0]), math.cosh(89.375), rel_tol=1e-6)


def _binary_golden(a: float, b: float, output_format, dest_acc):
    pair = torch.tensor([a] * 1024 + [b] * 1024, dtype=torch.bfloat16)
    return get_golden_generator(BinarySFPUGolden)(
        MathOperation.SfpuElwmul,
        pair,
        0,
        1,
        0,
        32,
        [64, 32],
        F16B,
        dest_acc=dest_acc,
        output_format=output_format,
    ).flatten()


def test_the_binary_golden_keeps_a_float32_output_at_fp32_on_a_16bit_input():
    x = 1.0 + 2.0**-7
    golden = _binary_golden(x, x, F32, D_YES)
    assert golden.dtype == torch.float32
    assert float(golden[0]) == x * x != float(torch.tensor(x * x).to(torch.bfloat16))


def test_the_binary_golden_hands_a_16bit_output_back_in_the_operand_dtype():
    x = 1.0 + 2.0**-7
    for dest_acc in (D_NO, D_YES):
        golden = _binary_golden(x, x, F16B, dest_acc)
        assert golden.dtype == torch.bfloat16
        assert float(golden[0]) == float(torch.tensor(x * x).to(torch.bfloat16))


def test_widening_an_fp16_nan_keeps_its_sign():
    """torch's fp16 -> fp32 cast drops a NaN's sign; the goldens' permutations must not."""
    half = torch.tensor([-math.nan, math.nan, -1.5], dtype=torch.float16)
    wide = widen_to_float32(half)
    assert torch.isnan(wide[:2]).all() and float(wide[2]) == -1.5
    assert torch.signbit(wide).tolist() == [True, False, True]


def test_an_fp16_dest_saturates_an_infinity_of_either_sign():
    values = torch.tensor([-math.inf, math.inf, -2.0])
    assert convert_inf_to_value(values, 130560.0).tolist() == [
        -130560.0,
        130560.0,
        -2.0,
    ]
    assert convert_inf_to_value([-math.inf, 1.0], 130560.0) == [-130560.0, 1.0]


def test_a_known_nonfinite_lane_is_excused_on_its_cell_and_nowhere_else():
    """Celu returns inf for x in 65408..65504 on a 16-bit Float16 Dest (#58607). The
    lane below the interval is still a failure, so is the same lane on a 32-bit Dest,
    and a named lane that agrees is ranked like any other."""
    src = torch.tensor([65376.0, 65408.0, 65504.0], dtype=torch.float16)
    golden = src.clone()
    result = torch.full_like(src, float("inf"))
    celu = MathOperation.Celu
    assert _failures(celu, src, golden, result, F16, F16, YES, D_NO).tolist() == [
        True,
        False,
        False,
    ]
    assert _failures(celu, src, golden, result, F16, F16, YES, D_YES).all()
    assert _failures(_OP, src, golden, result, F16, F16, YES, D_NO).all()
    assert measurable_mask(src, golden, golden.clone(), F16, F16, D_NO).all()


def test_an_entry_no_lane_of_which_disagrees_is_stale():
    """The day the defect is fixed its lanes agree, and the gate has to say so; an
    entry another exclusion already covers is stale too."""
    src = torch.tensor([65408.0, 65504.0], dtype=torch.float16)
    golden = src.clone()
    cell = dict(approx_mode=NO, dest_acc=D_NO)
    celu = MathOperation.Celu
    still_broken = torch.tensor([float("inf"), 65504.0], dtype=torch.float16)
    assert stale_excuses(celu, src, golden, still_broken, F16, F16, **cell) == []
    fixed = golden.clone()
    assert [
        e.issue for e in stale_excuses(celu, src, golden, fixed, F16, F16, **cell)
    ] == [58607]
    other = dict(approx_mode=NO, dest_acc=D_YES)
    assert stale_excuses(celu, src, golden, fixed, F16, F16, **other) == []
    nan_golden = torch.full_like(golden, float("nan"))
    assert stale_excuses(celu, src, nan_golden, still_broken, F16, F16, **cell) == [
        _KNOWN_NONFINITE_LANES[celu][0]
    ]


def test_every_known_lane_entry_names_an_issue_and_a_gateable_cell():
    for op, entries in _KNOWN_NONFINITE_LANES.items():
        for entry in entries:
            where = f"{op.name}: {entry}"
            assert isinstance(entry.issue, int) and entry.issue > 0, where
            assert entry.low <= entry.high and entry.why, where
            assert has_ulp_gate(entry.output), where


@pytest.mark.parametrize(
    "arch",
    [ChipArchitecture.WORMHOLE, ChipArchitecture.BLACKHOLE],
    ids=lambda a: a.value,
)
def test_the_sweep_cells_are_the_ones_testconfig_builds_as_asked(arch, monkeypatch):
    """A cell is swept exactly when the TestConfig built for it runs the Dest it asks
    for; the promoted exponent-B -> Float16 `No` cells are the ones left out."""
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
    promoted = {(i, o) for i, o, _, d in cells if d == D_NO} ^ {
        (i, o) for i, o, _, d in cells if d == D_YES
    }
    assert promoted == {(F16B, F16)}


def test_no_step_budget_names_an_arch_sweep_cells_does_not_filter_for():
    """`sweep_cells` applies only the Dest promotion off Wormhole, not another arch's
    format support; a ULP row naming one would send unsupported cells to the device."""
    from helpers.sfpu_accuracy_budget import _SFPU_ACCURACY_BUDGET
    from helpers.ulp_table import MEASURED_ARCH

    elsewhere = [
        (op.name, key)
        for op, rows in _SFPU_ACCURACY_BUDGET.items()
        for key, contract in rows.items()
        if contract.metric is Metric.ULP and key.arch not in (None, MEASURED_ARCH)
    ]
    assert elsewhere == []


# The Float32 walk's sample count, written down independently of the code that makes
# it: one sample per normal bfloat16 bit pattern.
_FLOAT32_WALK_SAMPLES = 65024

#: The Float16_b sweep's value count, likewise by hand: the 65,024 nonzero normal bit
#: patterns and one zero (+0.0 and -0.0 are one value); no subnormal.
_BFLOAT16_SWEPT_VALUES = 65025

#: The Float16 sweep's value count, likewise: 2 signs x 30 normal binades x 1,024
#: mantissas, and one zero; no subnormal.
_FLOAT16_SWEPT_VALUES = 61441


def test_the_float32_walk_is_pinned_end_to_end():
    """The tile count holds every swept value, and the Float32 stride walks one sample
    per bfloat16 cell, end to end, none of them a bfloat16 value, both halves of a cell
    reached in every binade (a stride of 2**16 + 1 did not)."""
    import test_unary_sfpu_ulp as sweep

    lanes = sweep.SWEEP_TILE_COUNT * TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1]
    assert (
        sweep.SWEEP_DIMENSIONS[0] * sweep.SWEEP_DIMENSIONS[1] == lanes == _SWEEP_TENSOR
    )
    for fmt in SWEEP_FORMATS:
        assert swept_value_count(fmt) <= lanes, fmt.name
    walk = _enumerate_representable(
        F32, -math.inf, math.inf, _SWEEP_TENSOR, stride=_FP32_STRIDE, normal_only=True
    )
    assert walk.numel() == _FLOAT32_WALK_SAMPLES == swept_value_count(F32)
    assert not bool(
        ((walk != 0) & (walk.abs() < torch.finfo(torch.float32).tiny)).any()
    )
    assert float(walk.min()) == -torch.finfo(torch.float32).max
    assert float(walk.max()) > 3.39e38
    bits = walk.view(torch.int32)
    low = bits & 0xFFFF
    assert bool((low != 0).all()), "a sample is a bfloat16 value"
    assert (bits >> 16).unique().numel() == walk.numel(), "two samples share a cell"
    normal = torch.isfinite(walk) & (walk.abs() >= torch.finfo(torch.float32).tiny)
    binade = ((bits >> 23) & 0x1FF)[normal]  # sign and exponent
    lower = (low < 0x8000)[normal].to(torch.float64)
    for b in binade.unique().tolist():
        share = float(lower[binade == b].mean())
        assert 0.3 < share < 0.7, (hex(b), share)


def test_a_16bit_walk_skips_its_offset_before_striding():
    """`offset` counts in-range values, not strided samples, on every format: the
    float32 walk jumps to it before striding, and the 16-bit enumeration must too.
    Applied after the stride, offset 1 at stride 2 picked values 2, 4, 6 for 1, 3, 5."""
    every = _enumerate_representable(F16B, 1.0, 2.0)
    picked = _enumerate_representable(F16B, 1.0, 2.0, 3, offset=1, stride=2)
    assert picked.tolist() == every[1:7:2].tolist()


@pytest.mark.parametrize("stride", [0, -1])
def test_a_sweep_stride_must_be_positive(stride):
    """Zero divides by zero in the float32 walk and a negative stride gives a negative
    sample count, so the spec refuses both where they are written."""
    with refuses("stride must be positive"):
        StimuliSpec.ulp_sweep(low=-1.0, high=1.0, stride=stride)


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
        (ChipArchitecture.WORMHOLE, D_YES),
        (ChipArchitecture.BLACKHOLE, D_YES),
        (ChipArchitecture.QUASAR, D_NO),
    ],
    ids=lambda v: getattr(v, "value", None) or v.name,
)
def test_effective_dest_acc_promotes_an_exponent_b_input_packed_to_float16(
    arch, promoted
):
    assert effective_dest_acc(F16B, F16, D_NO, arch) == promoted
    assert effective_dest_acc(F16, F16, D_NO, arch) == D_NO


def test_effective_dest_acc_asks_for_the_chip_only_for_an_outlier(monkeypatch):
    """Block sizing calls it for every variant; an ordinary combination must not need a
    chip context to say it is not promoted."""
    import helpers.data_format_inference as inference

    def no_chip():
        raise AssertionError("looked up the chip for a combination it cannot promote")

    monkeypatch.setattr(inference, "get_chip_architecture", no_chip)
    for dest_acc in DestAccumulation:
        assert inference.effective_dest_acc(F16, F16, dest_acc) == dest_acc
    assert inference.effective_dest_acc(F16B, F16, D_YES) == D_YES
    with pytest.raises(AssertionError) as caught:  # allow-pytest.raises: host-only test
        inference.effective_dest_acc(F16B, F16, D_NO)
    assert "looked up the chip" in str(caught.value)


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
    """The claim is `_OP_SINGULARITIES`, the point and the undefined side, not the 1e-6
    guard band the random draw keeps off a pole."""
    finite = torch.tensor([1.0], dtype=torch.bfloat16)
    inf = torch.tensor([float("inf")], dtype=torch.bfloat16)
    for value, claimed in [(v, True) for v in inside] + [(v, False) for v in outside]:
        src = torch.tensor([value], dtype=torch.bfloat16)
        assert bool(_failures(op, src, finite, inf, F16B, F16B).any()) is claimed, value


def test_the_claim_excludes_the_poles_of_the_gamma_family():
    """A pole at every non-positive integer, -0.0 included; -2.5 is not one; an op
    without the lattice keeps every lane."""
    src = torch.tensor([-3.0, -2.5, -0.0, 0.0, 0.5, 2.0], dtype=torch.float32)
    assert claimed_lanes(MathOperation.Digamma, src, F32, F32, D_YES).tolist() == [
        False,
        True,
        False,
        False,
        True,
        True,
    ]
    assert claimed_lanes(MathOperation.Abs, src, F32, F32, D_YES).all()


def test_the_claim_judges_a_float32_input_as_the_16bit_dest_truncates_it():
    """-8326436.5 truncates to the bf16 pole -8323072 at dest_acc=No; a 32-bit Dest
    keeps it off the lattice."""
    src = torch.tensor([-8326436.5, -2.5], dtype=torch.float32)
    op = MathOperation.Polygamma
    for out in (F16B, F32):
        assert claimed_lanes(op, src, F32, out, D_NO).tolist() == [
            False,
            True,
        ], out.name
        assert claimed_lanes(op, src, F32, out, D_YES).all(), out.name


def test_the_polygamma_golden_is_the_pole_on_every_nonpositive_integer():
    """+inf for an odd order, NaN for an even one; torch answers a large finite number
    there. Off the lattice, torch."""
    golden = UnarySFPUGolden()
    golden.data_format = F32
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
    answered with its sign in the Dest's top binade, or NaN, is the Dest's overflow;
    so is a flush across 2**-14. Under a Float16 output nothing is excluded, and a bf16
    Dest has fp32's range."""
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
    assert dest_holds(golden, result, F16, F32, D_NO).tolist() == lost
    assert dest_holds(golden, result, F16, F16, D_NO).all()
    assert dest_holds(golden, result, F16B, F32, D_NO).all()
    assert dest_holds(golden, result, F16, F32, D_YES).all()
    src = torch.full((8,), 2.0, dtype=torch.float16)
    assert measurable_mask(src, golden, result, F16, F32, D_NO).tolist() == lost
    assert not _failures(
        MathOperation.Exp, src, golden, result, F16, F32, dest=D_NO
    ).any()


def test_an_answer_the_dest_did_not_saturate_to_stays_judged():
    """A wrong-signed answer against an overflowed golden, a same-signed one more than
    2x short of the edge, an answer 0 where the golden is a normal 1e-4, and a pair both
    nonzero are the op's; an inf against 1.0, or of the wrong sign, is a failure."""
    tiny = 2.0**-14
    golden = torch.tensor(
        [131008.0, 131008.0, 70000.0, 1e-4, tiny, 1.0, 131008.0], dtype=torch.float32
    )
    result = torch.tensor(
        [-1.0, -131008.0, 30000.0, 0.0, 3 * tiny, math.inf, -math.inf],
        dtype=torch.float32,
    )
    assert dest_holds(golden, result, F16, F32, D_NO).all()
    src = torch.full((7,), 2.0, dtype=torch.float16)
    assert measurable_mask(src, golden, result, F16, F32, D_NO).tolist() == (
        [True] * 5 + [False, False]
    )
    assert _failures(
        MathOperation.Exp, src, golden, result, F16, F32, dest=D_NO
    ).tolist() == ([False] * 5 + [True, True])


def test_the_near_zero_floor_is_derived_from_the_band_and_capped():
    """A cell demoted only by near-zero lanes gets the floor that covers them, exactly
    the band's largest absolute error; a cell that fits gets none; one whose band error
    is past the cap keeps the capped floor with the lanes it leaves past the ceiling, so
    the demotion note can name it; a lane outside the band is never rescued."""
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)
    golden[30], golden[31] = 1e-6, 0.5
    result = golden.clone()
    result[1] = 1.5078125  # one bf16 step above 1.5
    result[30] = 0.0  # a 1e-6 answer flushed: ~14,000 steps, 1e-6 absolute
    mask = torch.ones(32, dtype=torch.bool)

    def floor(res):
        distance = ulp_distance(golden, res, flush_subnormals=True)
        return near_zero_floor(golden, res, distance, mask, F16B)

    residual, atol = floor(result)
    assert residual == 1
    assert atol == float(torch.tensor(1e-6, dtype=torch.bfloat16))
    fits = golden.clone()
    fits[1] = 1.5078125
    assert floor(fits) is None
    past_cap = golden.clone()
    past_cap[30] = 0.01
    assert 0.01 > EMIT_MAX_NEAR_ZERO_ATOL
    residual, atol = floor(past_cap)
    assert atol == EMIT_MAX_NEAR_ZERO_ATOL and residual > 6
    outside_band = golden.clone()
    outside_band[31] = 0.0  # |golden| = 0.5 is not under 1% of 1.5
    assert floor(outside_band) is None


# ── A cell's `domains` list ──────────────────────────────────────────────────


def test_the_tensor_side_draws_the_rule_on_the_lanes_and_measures_each_entry(
    monkeypatch,
):
    """`binade_maxima` profiles the received inputs, `domain_figures` measures each
    drawn span over exactly its lanes, and `entry_readings` is what the recorder
    writes. The knobs are lowered so a tiny tensor can split."""
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
    spans = rule.partition(bins, rule.usable_budget_ceiling(F16B))
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
    every = torch.ones(3, dtype=torch.bool)
    assert floor_covered(golden, result, every, 1e-6).tolist() == [False, True, False]
    assert not floor_covered(golden, result, every, None).any()


def test_the_tensor_binade_is_the_scalar_one():
    """A negative power of two is the low edge of the binade above it, on both sides."""
    values = [0.0, 2.0**-149, -(2.0**-149), 2.0**-126, -(2.0**-126), -0.5, -0.75]
    values += [-1.0, -1.5, -2.0, 1.0, 1.5, 2.0, 65504.0, -65504.0, -(2.0**127)]
    x = torch.tensor(values, dtype=torch.float64)
    assert _binade_index(x).tolist() == [rule.binade(v) for v in values]


def _redrawn(bins, key):
    """The list the rule draws on a recorded binade profile alone, as a reader of the
    `--ulp-measure` artifact would: a binade lies wholly inside one entry, so the worst
    of an entry's binades is the entry's figure."""

    def worst_in(low, high):
        found = [
            w
            for b, (_, r, w) in bins.items()
            if r and low <= rule.binade_edges(b)[0] < high
        ]
        return max(found) if found else None

    ceiling = rule.usable_budget_ceiling(key.output_format)
    return rule.entries_of(rule.partition(bins, ceiling), worst_in)


def test_a_worst_lane_on_a_negative_edge_is_figured_on_the_side_its_entry_holds():
    """-1 belongs to the entry that opens at -1; the emitter, on the lanes, and a reader
    of the recorded profile draw the same list."""
    n = 2048  # each run of lanes is a domain of its own at DOMAIN_MIN_LANES
    x = torch.tensor([-1.5] * n + [-1.0] * n + [-0.75] * n, dtype=torch.float64)
    distance = torch.tensor([1] * n + [9] * n + [9] * n)
    every = torch.ones(3 * n, dtype=torch.bool)
    bins = binade_maxima(x, distance, every, every)
    drawn = domain_figures(
        rule.partition(bins, rule.usable_budget_ceiling(F16B)),
        x,
        distance,
        every,
        every,
    )
    assert drawn == (Domain(-_INF, -1, 1), Domain(-1, _INF, 9))
    key = cell_key(F16B, F16B, NO, D_NO)
    recorded = Measurement.from_json(
        Measurement(
            op=MathOperation.Tanh,
            key=key,
            worst=9,
            lanes=3 * n,
            bins=tuple((b, *counts) for b, counts in sorted(bins.items())),
        ).to_json()
    )
    assert _redrawn({b: tuple(counts) for b, *counts in recorded.bins}, key) == drawn


def test_a_cell_with_no_ranked_span_writes_no_list():
    x = torch.tensor(
        [-1.5] * 64 + [1.0 + i / 64 for i in range(64)], dtype=torch.float64
    )
    distance = torch.tensor([30] * 64 + [1] * 64)
    present = torch.ones(128, dtype=torch.bool)
    few = torch.tensor([i < 3 for i in range(128)])
    nothing = rule.partition(
        binade_maxima(x, distance, few, present), rule.usable_budget_ceiling(F16B)
    )
    assert domain_figures(nothing, x, distance, few, few) == ()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the ULP regression check in ``helpers/ulp_budget_diff.py``.

No device and no torch: that module runs on the host against two revisions of a text
file. Pinned here: which changes count as a regression, and that its parse of
the live table agrees with the real loader's -- it reads the file a second way, so the
two are free to drift unless something says they may not.
"""

import ast
import sys
from datetime import date
from pathlib import Path

import pytest
from helpers.ulp_budget_diff import (
    _MAX_ROWS,
    CONTRACT_FIELDS,
    KEY_FIELDS,
    RECORDED_SKIP,
    ULP_GATED_OUTPUTS,
    DuplicateRow,
    _gates,
    _measured_cells,
    _nonfinite_cells,
    _readings,
    _resolve,
    compare,
    junit_completed,
    junit_failures,
    main,
    parse_table,
    recorded_max,
    recorded_nonfinite,
    render_budget_diff,
    render_headroom,
)
from helpers.ulp_provenance import Provenance

#: The tool's short key spelling -> the registry's BudgetKey field. Both of its copies
#: of the key list are held to this one by test_the_key_dimensions_are_the_registrys.
_BUDGET_KEY_ATTR = {
    "in": "input_format",
    "out": "output_format",
    "approx": "approx_mode",
    "dest": "dest_acc",
    "arch": "arch",
}


def _pinned(key):
    """A registry BudgetKey as the tool's ``((short, value), ...)`` key."""
    return tuple(
        (short, getattr(key, attr).name)
        for short, attr in _BUDGET_KEY_ATTR.items()
        if getattr(key, attr) is not None
    )


_HEADER = "Abs:  # a header is prose: no code reads it"
_BASE_ROWS = (
    "{in: Float16_b, out: Float16_b, max_ulp: 1, measured: 1, run: 2026-01-01}",
    "{in: Float16, out: Float16_b, max_ulp: 4, measured: 3, run: 2026-01-01}",
    "{in: Bfp8_b, out: Bfp8_b, metric: tolerance, measured: 393, run: 2026-01-01}",
    "{in: Float32, out: Float32, max_ulp: 2, measured: 1, run: 2026-01-01}",
)
_BF16 = (("in", "Float16_b"), ("out", "Float16_b"))


def _head(*rows):
    return _HEADER + "\n" + "".join(f"  - {r}\n" for r in rows)


_BASE = _head(*_BASE_ROWS)


def _edited(index, row):
    """The base table with one row replaced."""
    rows = list(_BASE_ROWS)
    rows[index] = row
    return _head(*rows)


def _changes(base, head):
    return compare(parse_table(base), parse_table(head))


def _kinds(base, head):
    return {c.cell[1]: c.kind for c in _changes(base, head)}


def _measured(*cells):
    """``(in, out, max)`` triples as the recorder's rows, folded to one worst per cell."""
    return _measured_cells(
        [
            {"op": "Abs", "in": i, "out": o, "approx": None, "dest": None, "max": m}
            for i, o, m in cells
        ]
    )


def _refuses(kind):
    """The suite's ``expect_error`` fixture needs a device; these are host-only tests."""
    return pytest.raises(kind)  # allow-pytest.raises: host-only test


# ── What counts as a regression ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "provenance, remeasured",
    [
        ("measured: 1, run: 2026-01-01", False),
        ("measured: 1, run: 2026-01-01}  # re-measured 2026-02-02, honestly", False),
        ("measured: 8, run: 2026-02-02", True),
        ("measured: 1, run: 2026-02-02", True),
    ],
    ids=["untouched", "only-the-prose", "re-measured", "same-figure-new-run"],
)
def test_a_raised_budget_is_a_regression_marked_by_its_provenance(
    provenance, remeasured
):
    """The raise is always surfaced. Whether the row's measurement fields changed with
    it is what separates a re-measurement from a number fitted to a failure; editing
    the prose beside the row does not count."""
    if "}" in provenance:
        body, _, comment = provenance.partition("}")
        row = f"{{in: Float16_b, out: Float16_b, max_ulp: 9, {body}}}{comment}"
    else:
        row = f"{{in: Float16_b, out: Float16_b, max_ulp: 9, {provenance}}}"
    head = _edited(0, row)
    (change,) = _changes(_BASE, head)
    assert change.kind == "raised" and change.is_regression
    assert change.remeasured is remeasured


def test_losing_the_gate_entirely_is_a_regression():
    """Demotion to tolerance is the loosest change: the cell stops being judged on a
    step budget at all, and a bounds check cannot see it."""
    head = _edited(
        0,
        "{in: Float16_b, out: Float16_b, metric: tolerance, measured: 14337, run: 2026-01-01}",
    )
    assert _kinds(_BASE, head)[_BF16] == "ungated"


def test_deleting_a_gated_row_is_a_regression_and_a_figureless_tolerance_row_is_not():
    """A row that held its cell to nothing -- no budget, no recorded figure, no declared
    bound -- cannot be a loss when it goes."""
    assert _kinds(_BASE, _head(*_BASE_ROWS[1:]))[_BF16] == "removed"
    bare = "{in: Bfp8_b, out: Bfp8_b, metric: tolerance}  # block-quantized"
    assert _kinds(_head(_BASE_ROWS[0], bare), _head(_BASE_ROWS[0])) == {}


def test_tightening_and_newly_gating_are_not_regressions():
    head = _head(
        "{in: Float16_b, out: Float16_b, max_ulp: 0, measured: 0, run: 2026-01-01}",
        _BASE_ROWS[1],
        "{in: Bfp8_b, out: Bfp8_b, max_ulp: 3, measured: 2, run: 2026-01-01}",
        _BASE_ROWS[3],
        "{in: Float16, out: Float32, max_ulp: 7, measured: 6, run: 2026-01-01}",
    )
    changes = _changes(_BASE, head)
    assert not any(c.is_regression for c in changes)
    assert {c.kind for c in changes} == {"tightened", "gated"}


def test_a_row_the_sweep_collapses_is_not_a_loss_while_its_cells_keep_their_budget():
    """The emitter drops a key dimension when both values measure alike, so a re-emit
    deletes `approx`-keyed rows and writes one without. Nothing any query resolves to
    has changed, and a row diff reported 16 of these as regressions."""
    base = _head(
        '{in: Float16_b, out: Float32, approx: "No", dest: "Yes", max_ulp: 36018, measured: 32744, run: 2026-01-01}  # a',
        '{in: Float16_b, out: Float32, approx: "Yes", dest: "Yes", max_ulp: 36018, measured: 32744, run: 2026-01-01}  # b',
    )
    same = _head(
        '{in: Float16_b, out: Float32, dest: "Yes", max_ulp: 36018, measured: 32744, run: 2026-01-01}  # c'
    )
    changes = _changes(base, same)
    assert not any(c.is_regression for c in changes)
    # The one honest difference: a query with `approx` unset matched neither keyed row
    # before and matches the collapsed one now, so that variant is newly gated.
    assert [(c.kind, c.variants) for c in changes] == [("gated", 1)]
    # A raise through the collapsed row is one change covering both variants,
    # re-measured because the deciding row's measurement is new.
    raised = _head(
        '{in: Float16_b, out: Float32, dest: "Yes", max_ulp: 36030, measured: 32755, '
        "run: 2026-02-02}"
    )
    (change,) = [c for c in _changes(base, raised) if c.is_regression]
    assert change.kind == "raised" and change.variants == 2 and change.remeasured


def test_a_more_specific_row_that_loosens_a_cell_is_a_raise_not_an_addition():
    """The hole a row diff cannot see: no existing row changes, yet every Float32 query
    now resolves to 1000 steps instead of 2. Demoting the same cells through a new
    tolerance row is the same hole, and the loosest gate of all."""
    base = _head("{max_ulp: 2}  # exact")
    loosened = _head("{max_ulp: 2}  # exact", "{in: Float32, max_ulp: 1000}  # wide")
    (change,) = _changes(base, loosened)
    assert change.kind == "raised" and change.cell == ("Abs", (("in", "Float32"),))
    demoted = _head("{max_ulp: 2}  # exact", "{in: Float32, metric: tolerance}  # off")
    assert [c.kind for c in _changes(base, demoted)] == ["ungated"]


def test_a_deleted_row_falling_back_to_an_untouched_one_is_not_a_re_measurement():
    """The two rows a cell resolves to always carry different comments when a deletion
    hands it to a broader row, so comparing those read every such loss as re-measured,
    and the override report left it out of its "no fresh measurement" list. The deciding
    row is the broad one, and its comment did not change."""
    base = _head(
        "{in: Float16_b, metric: tolerance}  # max 393 ULP, block-quantized",
        "{in: Float16_b, out: Float16_b, max_ulp: 1}  # max 1 ULP",
    )
    head = _head("{in: Float16_b, metric: tolerance}  # max 393 ULP, block-quantized")
    (change,) = _changes(base, head)
    assert change.kind == "ungated" and not change.remeasured
    report = render_budget_diff([change])
    assert "| **no** |" in report


def test_a_tie_between_equally_specific_rows_is_refused_rather_than_ordered():
    """``{in: Float16_b}`` and ``{out: Float16_b}`` both match a query naming both, and
    the registry refuses that table; keeping whichever row came first made the guard's
    verdict depend on the file's order, and passed a table that cannot load."""
    table = parse_table(
        _head(
            "{in: Float16_b, max_ulp: 1}  # max 1 ULP",
            "{out: Float16_b, max_ulp: 9}  # max 8 ULP",
        )
    )
    with _refuses(ValueError) as caught:
        _resolve(table, "Abs", _BF16)
    assert "equally specific" in str(caught.value)
    with _refuses(ValueError) as caught:
        compare(parse_table(_BASE), table)
    assert "equally specific" in str(caught.value)


# ── Tolerance cells: what the sweep holds them to ─────────────────────────────


@pytest.mark.parametrize(
    "head_row, kind",
    [
        (
            "{in: Bfp8_b, out: Bfp8_b, metric: tolerance, measured: 500, run: 2026-01-02}",
            "baseline_raised",
        ),
        (
            "{in: Bfp8_b, out: Bfp8_b, metric: tolerance, measured: 393, nonfinite: 3, "
            "run: 2026-01-02}  # not measurable: lanes disagreeing with the golden",
            "baseline_raised",
        ),
        (
            "{in: Bfp8_b, out: Bfp8_b, metric: tolerance}  # block-quantized",
            "baseline_dropped",
        ),
        (
            "{in: Bfp8_b, out: Bfp8_b, metric: tolerance, measured: 12, run: 2026-01-02}",
            None,
        ),
    ],
    ids=["max-raised", "nonfinite-appears", "figure-deleted", "max-lowered"],
)
def test_a_tolerance_rows_recorded_measurement_is_a_baseline(head_row, kind):
    """The headroom report fails the sweep on a tolerance cell past the "max N ULP" or
    the non-finite count its row records, so raising either is loosening that check,
    and dropping the figure stops it altogether. Lowering it is a re-measurement."""
    changes = _changes(_BASE, _edited(2, head_row))
    assert [c.kind for c in changes] == ([kind] if kind else [])
    assert all(c.is_regression for c in changes)


def test_a_re_emit_that_splits_a_tolerance_row_drops_no_baseline():
    """The emitter writes one row per ``dest`` when the two measure apart. A query that
    leaves ``dest`` unset then matches nothing, but no measured cell -- every one names
    in, out, approx and dest -- loses its figure, so nothing the headroom report judges
    has loosened. Raising either half still reads as a raise."""
    one = (
        '{in: Float16_b, out: Float32, approx: "Yes", metric: tolerance, measured: 9, '
        "run: 2026-01-01}"
    )

    def split(no, yes):
        return _head(
            '{in: Float16_b, out: Float32, approx: "Yes", dest: "No", metric: '
            f"tolerance, measured: {no}, run: 2026-01-02}}",
            '{in: Float16_b, out: Float32, approx: "Yes", dest: "Yes", metric: '
            f"tolerance, measured: {yes}, run: 2026-01-02}}",
        )

    assert not [c for c in _changes(_head(one), split(9, 8)) if c.is_regression]
    (raised,) = [c for c in _changes(_head(one), split(9, 12)) if c.is_regression]
    assert raised.kind == "baseline_raised"
    assert dict(raised.cell[1])["dest"] == "Yes"


def test_deleting_a_tolerance_row_with_a_baseline_drops_it():
    """Before, a vanished tolerance row was "no loss": it gated nothing. It did hold the
    sweep's figure, and an op block on tolerance everywhere going away took every
    cell's out of the headroom report without a line in this one."""
    kinds = _kinds(_BASE, _head(_BASE_ROWS[0], _BASE_ROWS[1], _BASE_ROWS[3]))
    assert kinds[(("in", "Bfp8_b"), ("out", "Bfp8_b"))] == "baseline_dropped"


@pytest.mark.parametrize(
    "base_row, head_row, kind",
    [
        (
            "{metric: tolerance, atol: 0.13, rtol: 0.05}",
            "{metric: tolerance, atol: 0.2, rtol: 0.05}",
            "tolerance_widened",
        ),
        (
            "{metric: tolerance, atol: 0.13, rtol: 0.05}",
            "{metric: tolerance, atol: 0.13}",
            "tolerance_widened",
        ),
        (
            "{metric: tolerance, atol: 0.13, rtol: 0.05}",
            "{metric: tolerance, atol: 0.1, rtol: 0.05}",
            None,
        ),
    ],
    ids=["atol-raised", "rtol-dropped", "atol-lowered"],
)
def test_a_declared_tolerance_that_loosens_is_a_regression(base_row, head_row, kind):
    """The eltwise drivers still compare a tolerance cell against its row's atol/rtol,
    so loosening one is loosening that compare. A bound that goes away leaves the cell
    on a default this tool cannot see, so it reads as loosened too."""
    changes = [c for c in _changes(_head(base_row), _head(head_row)) if c.cell[1] == ()]
    assert [c.kind for c in changes] == ([kind] if kind else [])


# ── The arch dimension ────────────────────────────────────────────────────────


def test_an_unkeyed_budget_binds_only_on_the_measured_arch():
    """``accuracy_contract`` hands any other arch the cell's tolerance unless the row
    names that arch. A guard that treated every ULP row as gating everywhere reported
    the loss of a Blackhole gate as a *tightening* (the unkeyed 3 under the BH 5), and
    its addition as a raise rather than a new gate."""
    unkeyed = "{in: Float16_b, out: Float16_b, max_ulp: 3}  # max 2 ULP, wormhole"
    bh = "{in: Float16_b, out: Float16_b, arch: BLACKHOLE, max_ulp: 5}  # max 4 ULP, bh"
    on_bh = _BF16 + (("arch", "BLACKHOLE"),)

    (lost,) = _changes(_head(unkeyed, bh), _head(unkeyed))
    assert (lost.kind, lost.cell[1]) == ("ungated", _BF16)
    assert lost.before.key == on_bh
    (added,) = _changes(_head(unkeyed), _head(unkeyed, bh))
    assert (added.kind, added.cell[1]) == ("gated", on_bh)
    # One change each way: on the measured arch the unkeyed row decides both tables.


def test_an_arch_is_read_by_name_or_by_value():
    """The registry accepts ``arch: wormhole`` and ``arch: WORMHOLE`` alike; the
    measurement rows carry the enum name."""
    by_value = parse_table("Abs:\n  - {arch: blackhole, max_ulp: 1}  # c\n")
    by_name = parse_table("Abs:\n  - {arch: BLACKHOLE, max_ulp: 1}  # c\n")
    assert set(by_value) == set(by_name) == {("Abs", (("arch", "BLACKHOLE"),))}


def test_the_measured_arch_is_the_registrys():
    from helpers import ulp_budget_diff
    from helpers.sfpu_accuracy_budget import MEASURED_ARCH

    assert ulp_budget_diff.MEASURED_ARCH == MEASURED_ARCH.name


_FLOOR_CASES = [
    (
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 1.0e-07, measured: 1, run: 2026-01-01}",
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-05, measured: 1, run: 2026-01-01}",
        "floor_widened",
    ),
    (
        "{in: Float16_b, out: Float16_b, max_ulp: 2, measured: 1, run: 2026-01-01}",
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-05, measured: 1, run: 2026-01-01}",
        "floor_widened",
    ),
    # A smaller max_ulp must not mask a wider floor: the floor can more than pay for it.
    (
        "{in: Float16_b, out: Float16_b, max_ulp: 8, near_zero_atol: 1.0e-07, measured: 1, run: 2026-01-01}",
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-02, measured: 1, run: 2026-01-01}",
        "floor_widened",
    ),
    (
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-05, measured: 1, run: 2026-01-01}",
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 1.0e-07, measured: 1, run: 2026-01-01}",
        None,
    ),
]


@pytest.mark.parametrize(
    "base_row, head_row, kind",
    _FLOOR_CASES,
    ids=[
        "widened",
        "introduced",
        "wider-floor-tighter-budget",
        "narrowed",
    ],
)
def test_the_near_zero_floor_is_part_of_the_gate(base_row, head_row, kind):
    """`ulp_elementwise_valid` accepts a lane inside `near_zero_atol` however many steps
    out it is, so widening the floor loosens the gate and narrowing it does not. A guard
    that watched only `max_ulp` would call every one of these tables unchanged."""
    changes = _changes(_head(base_row), _head(head_row))
    assert [c.kind for c in changes] == ([kind] if kind else [])
    assert all(c.is_regression for c in changes)


def test_the_report_shows_the_floor_on_both_sides():
    """Or both columns read "2" and a widened-floor row looks inert."""
    base_row, head_row, _ = _FLOOR_CASES[0]
    text = render_budget_diff(_changes(_head(base_row), _head(head_row)))
    assert "1e-07" in text and "5e-05" in text


def test_an_unchanged_table_reports_nothing():
    assert _changes(_BASE, _BASE) == []
    assert "No budget changed" in render_budget_diff([])


def test_the_report_names_the_cell_and_the_change():
    head = _head(
        "{in: Float16_b, out: Float16_b, max_ulp: 9, measured: 1, run: 2026-01-01}"
    )
    report = render_budget_diff(_changes(_BASE, head))
    assert "loosen a gate" in report
    assert "in: Float16_b, out: Float16_b" in report
    assert "budget raised" in report
    assert "**no**" in report  # not re-measured


# ── Reading the table ─────────────────────────────────────────────────────────


def test_an_anchor_and_its_alias_are_both_real_rows():
    """The coarse-LUT tolerance pair is a YAML anchor and an alias, and a line-oriented
    reader skipped both -- silently, being tolerance rows with no budget to compare. A
    budget hidden behind an alias would have been invisible; the loader tie-back below
    is what caught it."""
    table = parse_table(
        "GeluAppx:\n"
        "  - &shared {max_ulp: 3, measured: 2, run: 2026-01-01}\n"
        "SigmoidAppx:\n"
        "  - *shared  # same cause, same number\n"
    )
    assert {cell[0] for cell in table} == {"GeluAppx", "SigmoidAppx"}
    assert all(row.max_ulp == 3 for row in table.values())
    # The alias is the anchor's row, measurement included: re-measuring one is
    # re-measuring both.
    shared = Provenance(measured=2, run=date(2026, 1, 1))
    assert {row.provenance for row in table.values()} == {shared}


def test_a_merge_key_row_is_read_through():
    """The registry loads the table with PyYAML's SafeLoader, which resolves `<<`."""
    table = parse_table(
        "Base:\n"
        "  - &b {out: Float16_b, max_ulp: 2, measured: 1, run: 2026-01-01}\n"
        "Abs:\n"
        "  - {<<: *b, max_ulp: 5, measured: 4, run: 2026-01-01}\n"
    )
    assert table[("Abs", (("out", "Float16_b"),))].max_ulp == 5


def test_only_a_rows_own_fields_mark_a_re_measurement():
    """A row's measurement is its own fields; a changed header line or comment beside
    it is prose and re-measures nothing, so it cannot launder a raise."""

    def table(header, budget, provenance):
        return parse_table(
            f"Abs:  # {header}\n"
            f"  - {{out: Float32, max_ulp: {budget}, {provenance}}}  # prose\n"
        )

    base = table("sweep A", 1, "measured: 1, run: 2026-01-01")
    for head, remeasured in (
        (table("sweep B", 4, "measured: 1, run: 2026-01-01"), False),
        (table("sweep A", 4, "measured: 3, run: 2026-02-02"), True),
    ):
        (raised,) = [c for c in compare(base, head) if c.is_regression]
        assert raised.kind == "raised" and raised.remeasured is remeasured


def test_an_unquoted_yaml_boolean_names_the_same_cell_as_a_quoted_one():
    """``dest: Yes`` is a YAML boolean; the registry maps it to the enum member
    (test_a_quoted_and_an_unquoted_no_mean_the_same_thing). Keyed as "True" here, a
    raise written that way would be reported against a cell that does not exist."""
    quoted = parse_table(
        'Abs:\n  - {in: Float16_b, out: Float16_b, dest: "Yes", max_ulp: 9}  # c\n'
    )
    bare = parse_table(
        "Abs:\n  - {in: Float16_b, out: Float16_b, dest: Yes, max_ulp: 9}  # c\n"
    )
    assert set(quoted) == set(bare)
    assert dict(next(iter(bare))[1])["dest"] == "Yes"


def test_a_not_measurable_row_still_records_the_measurable_lanes_maximum():
    """The emitter writes both numbers as fields; each reader takes its own, and the
    finite lanes of a demoted cell stay judged."""
    table = parse_table(
        'I1:\n  - {in: Float16_b, out: Float32, dest: "No", metric: tolerance, '
        "measured: 7, nonfinite: 31120, run: 2026-01-01}  # not measurable: lanes "
        "disagreeing with the golden about being finite (x=91: inf -> -1.16e37)\n"
    )
    row = next(iter(table.values()))
    assert recorded_nonfinite(row) == 31120
    assert recorded_max(row) == 7


def test_a_duplicated_cell_is_refused_rather_than_judged():
    """`_load_table` rejects two rows of equal specificity, so the table cannot load.
    Keeping the last row silently produced a verdict -- a *tightening*, even -- for a
    table the registry would not accept at all."""
    with _refuses(DuplicateRow) as caught:
        parse_table(
            _head(
                "{in: Float16_b, out: Float16_b, max_ulp: 9, measured: 1, run: 2026-01-01}",
                "{in: Float16_b, out: Float16_b, max_ulp: 1, measured: 1, run: 2026-01-01}",
            )
        )
    assert caught.value.cell == ("Abs", _BF16)


# ── The headroom half ─────────────────────────────────────────────────────────


def test_several_measurements_of_one_cell_keep_the_worst():
    """A driver enumerates axes the budget key does not, so one cell is recorded more
    than once. Keeping the last would hide the worst of them."""
    cells = _measured(
        ("Float16_b", "Float16_b", 3),
        ("Float16_b", "Float16_b", 11),
        ("Float16_b", "Float16_b", 5),
    )
    assert list(cells.values()) == [11]


@pytest.mark.parametrize(
    "budget, measured, expect",
    [(1, 9, "over budget"), (1, 1, "no headroom"), (8, 0, "could tighten")],
    ids=["over", "tight", "slack"],
)
def test_the_headroom_report_classifies_against_the_declared_budget(
    budget, measured, expect
):
    table = parse_table(
        f"Abs:\n  - {{in: Float16_b, out: Float16_b, max_ulp: {budget}}}\n"
    )
    report, over = render_headroom(
        table, _measured(("Float16_b", "Float16_b", measured))
    )
    assert expect in report
    assert over == (1 if expect == "over budget" else 0)


def test_an_exact_cell_at_its_zero_budget_is_not_a_warning():
    """`0 == 0` is an op exact by construction doing what it claims, enrolled so any
    drift fails. Calling that "no headroom" buried the real ones on a 130-cell run."""
    table = parse_table("Abs:\n  - {in: Float16_b, out: Float16_b, max_ulp: 0}\n")
    report, over = render_headroom(table, _measured(("Float16_b", "Float16_b", 0)))
    assert over == 0
    assert "no headroom" not in report
    assert "headroom to spare" in report


def test_a_tolerance_cell_is_judged_against_its_recorded_measurement():
    """No budget, so the sweep passes it whatever it measures -- this is the only place
    a regression there can show. The row's own ``measured`` is the baseline; a cell
    whose row records none is not judged at all."""
    table = parse_table(
        "Abs:\n"
        "  - {in: Float16_b, out: Float16_b, metric: tolerance, measured: 393, "
        "run: 2026-01-01}\n"
        "  - {in: Float16, out: Float16_b, metric: tolerance}  # max 9 ULP, prose\n"
    )
    report, over = render_headroom(table, _measured(("Float16_b", "Float16_b", 393)))
    assert over == 0 and "regressed" not in report
    report, over = render_headroom(
        table,
        _measured(("Float16_b", "Float16_b", 500), ("Float16", "Float16_b", 10**6)),
    )
    assert over == 1  # the un-baselined cell is not counted
    assert "| 500 | 393 | regressed |" in report
    assert "Regressed on the tolerance metric" in report


def test_a_measurement_resolves_against_the_most_specific_row():
    """Same rule the registry uses, so the report judges a cell against the budget that
    gates it rather than a broader one."""
    table = parse_table(
        "Abs:\n"
        "  - {out: Float16_b, max_ulp: 100}  # broad\n"
        "  - {in: Float16_b, out: Float16_b, max_ulp: 1}  # specific\n"
    )
    _, over = render_headroom(table, _measured(("Float16_b", "Float16_b", 50)))
    assert over == 1  # 50 is inside the broad 100 but past the specific 1


def test_the_headroom_report_lists_every_failure_and_caps_only_the_slack():
    """The failing cells are what the report is for; slack is a to-do list."""
    extra = 5
    n = _MAX_ROWS + extra
    table = "Abs:\n" + "".join(
        f"  - {{in: F{i}, out: Float16_b, max_ulp: 8}}\n" for i in range(2 * n)
    )
    rows = [
        {"op": "Abs", "in": f"F{i}", "out": "Float16_b", "max": 99} for i in range(n)
    ] + [
        {"op": "Abs", "in": f"F{i}", "out": "Float16_b", "max": 0}
        for i in range(n, 2 * n)
    ]
    report, over = render_headroom(parse_table(table), _measured_cells(rows))
    assert over == n
    assert report.count("over budget |") == n
    assert report.count("could tighten |") == _MAX_ROWS
    # Delimited: a bare "5 more" is also in "45 more".
    assert report.count(f"| _… {extra} more_ |") == 1


def test_the_guard_lists_every_loosened_cell_and_caps_only_the_improvements():
    """The loosened cells are the list the guard exists for (#58590 loosened 87)."""
    extra = 5
    n = _MAX_ROWS + extra

    def table(raised, tightened):
        return _head(
            *(f"{{in: F{i}, out: Float16_b, max_ulp: {raised}}}" for i in range(n)),
            *(
                f"{{in: F{i}, out: Float16_b, max_ulp: {tightened}}}"
                for i in range(n, 2 * n)
            ),
        )

    changes = _changes(table(5, 5), table(9, 1))
    report = render_budget_diff(changes)
    assert report.count("budget raised |") == n
    assert report.count("budget tightened |") == _MAX_ROWS
    assert report.count(f"| _… {extra} more_ |") == 1


def test_a_tolerance_cell_with_no_recorded_figure_is_counted_not_passed_silently():
    """The clean summary says no judged tolerance cell regressed; a cell with nothing to
    be judged against has to show up as such, or the summary reads as covering it."""
    table = parse_table(
        "Abs:\n  - {in: Float16, out: Float16_b, metric: tolerance}  # block-quantized\n"
    )
    report, over = render_headroom(table, _measured(("Float16", "Float16_b", 10**6)))
    assert over == 0
    assert "1 tolerance cell(s) measured but not judged" in report


# ── The slim runner, and the tie-back to the real loader ──────────────────────


#: The modules the slim runner loads, by path: the tool, and the one sibling it reads
#: the table through.
_SLIM_MODULES = ("ulp_budget_diff.py", "ulp_provenance.py")
_SLIM_SIBLINGS = {"ulp_provenance"}


@pytest.mark.parametrize("module", _SLIM_MODULES)
def test_the_tool_imports_nothing_but_the_standard_library_and_yaml(module):
    """The PR check runs on a slim runner: no torch, no ttexalens, no LLK venv. Read off
    each module's own AST: the import list is the actual property, and a subprocess
    would prove it for one environment only. The one sibling allowed is
    ``ulp_provenance``, itself held to the same rule."""
    tool = Path(__file__).parent / "helpers" / module
    roots = set()
    for node in ast.walk(ast.parse(tool.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            # A relative import would reach back into `helpers`; only the sibling the
            # runner also has on its path is allowed.
            if node.level:
                assert node.level == 1 and node.module in _SLIM_SIBLINGS, node.module
                continue
            if node.module:
                roots.add(node.module.split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"yaml"} | _SLIM_SIBLINGS
    assert roots <= allowed, f"imports outside stdlib + yaml: {sorted(roots - allowed)}"
    assert "helpers" not in roots and "torch" not in roots


def test_the_tool_runs_by_path_with_its_sibling(tmp_path, monkeypatch):
    """Run as ``python3 helpers/ulp_budget_diff.py``, the package is not importable
    and the sibling is found on the script's own directory. Run in-process the way the
    interpreter runs a script: as ``__main__``, with no package, and the script's
    directory first on ``sys.path``."""
    import runpy

    helpers = Path(__file__).parent / "helpers"
    (tmp_path / "base.yaml").write_text(_BASE, encoding="utf-8")
    (tmp_path / "head.yaml").write_text(_edited(0, _BASE_ROWS[0]), encoding="utf-8")
    monkeypatch.syspath_prepend(str(helpers))
    monkeypatch.delitem(sys.modules, "ulp_provenance", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ulp_budget_diff.py",
            "diff",
            "--base",
            str(tmp_path / "base.yaml"),
            "--head",
            str(tmp_path / "head.yaml"),
        ],
    )
    with _refuses(SystemExit) as done:
        runpy.run_path(str(helpers / "ulp_budget_diff.py"), run_name="__main__")
    assert done.value.code == 0
    # Imported by its bare name, as on the runner.
    assert "ulp_provenance" in sys.modules


def test_the_key_dimensions_are_the_registrys():
    """A new key dimension in the registry that the tool did not know would make every
    row pinning it invisible to the guard, while each copy agreed with itself."""
    from helpers.sfpu_accuracy_budget import _KEY_FIELDS

    assert _BUDGET_KEY_ATTR == _KEY_FIELDS
    assert KEY_FIELDS == tuple(_KEY_FIELDS)


def test_this_parse_agrees_with_the_registry_loader_on_the_live_table():
    """This module reads the table a second way -- text, no torch -- so the check can
    run without the LLK environment and against an arbitrary base revision. Nothing
    stops the two parses drifting except this. Compared as (op, key, budget, floor,
    atol, rtol): every number a regression is defined over."""
    from helpers.sfpu_accuracy_budget import _SFPU_ACCURACY_BUDGET, _TABLE_PATH, Metric

    mine = {
        (cell[0], cell[1], row.max_ulp, row.near_zero_atol, row.atol, row.rtol)
        for cell, row in parse_table(_TABLE_PATH.read_text(encoding="utf-8")).items()
    }
    theirs = set()
    for op, table in _SFPU_ACCURACY_BUDGET.items():
        for key, contract in table.items():
            ulp = contract.metric == Metric.ULP
            theirs.add(
                (
                    op.name,
                    _pinned(key),
                    contract.max_ulp if ulp else None,
                    contract.near_zero_atol,
                    None if ulp else contract.atol,
                    None if ulp else contract.rtol,
                )
            )
    assert mine == theirs, (
        f"only this parser sees: {sorted(mine - theirs, key=str)[:5]}\n"
        f"only the loader sees: {sorted(theirs - mine, key=str)[:5]}"
    )


def test_this_resolution_agrees_with_the_registry_on_every_swept_cell():
    """Agreeing on the rows is not agreeing on what a query resolves to, and a
    regression is defined over the latter. Every cell the sweep drives, for every op in
    the registry, must land on the same row through ``_resolve`` as through the
    registry's ``_winner``. The query is built the way the headroom report builds it,
    from a measurement row, so a dimension the recorder does not write (``arch``, once)
    shows up as a disagreement here instead of a cell headroom silently skips."""
    from helpers.sfpu_accuracy_budget import (
        _SFPU_ACCURACY_BUDGET,
        _TABLE_PATH,
        MEASURED_ARCH,
        BudgetKey,
        Metric,
        _winner,
    )
    from helpers.ulp import has_ulp_gate
    from helpers.ulp_sweep import sweep_cells

    table = parse_table(_TABLE_PATH.read_text(encoding="utf-8"))
    # The Wormhole grid, whatever CHIP_ARCH this host has: the one headroom judges.
    cells = sweep_cells(MEASURED_ARCH)
    compared = 0
    for op, rows in _SFPU_ACCURACY_BUDGET.items():
        for in_fmt, out_fmt, approx, dest in cells:
            query = BudgetKey(
                approx_mode=approx,
                input_format=in_fmt,
                output_format=out_fmt,
                dest_acc=dest,
                arch=MEASURED_ARCH,
            )
            found = _winner(rows, query, op.name)
            theirs = None if found is None else _pinned(found[0])
            # The fields `_record_ulp_measurement` writes, as it writes them.
            (asked,) = _measured_cells(
                [
                    {
                        "op": op.name,
                        "in": in_fmt.name,
                        "out": out_fmt.name,
                        "approx": approx.name,
                        "dest": dest.name,
                        "arch": MEASURED_ARCH.name,
                        "max": 0,
                    }
                ]
            )
            mine = _resolve(table, *asked)
            assert (mine.key if mine else None) == theirs, asked
            # And whether that row's step budget binds, which the output decides too.
            binds = (
                found is not None
                and found[1].metric is Metric.ULP
                and has_ulp_gate(out_fmt)
            )
            assert _gates(mine, asked[1]) == binds, asked
            compared += 1
    assert cells and _SFPU_ACCURACY_BUDGET, "nothing to compare"
    assert compared == len(cells) * len(_SFPU_ACCURACY_BUDGET) > 0


def test_the_headroom_report_fails_an_overflow_the_row_does_not_account_for():
    """A "not measurable" row records how many lanes went non-finite; the sweep skips
    such a tolerance cell but records the count, and more lanes than the row names --
    or any on a row that names none -- is a regression the sweep must fail on."""
    table = parse_table(
        "Exp:\n"
        '  - {in: Float32, out: Bfp8_b, dest: "Yes", metric: tolerance, measured: 0, '
        "nonfinite: 2, run: 2026-01-01}  # not measurable: x=3e38: 3e38 -> inf\n"
        '  - {in: Float16, out: Float16, dest: "No", metric: tolerance, measured: 511, '
        "run: 2026-01-01}\n"
        "Exp2:\n"
        '  - {in: Float32, out: Bfp8_b, dest: "Yes", metric: tolerance, nonfinite: 2, '
        "run: 2026-01-01}  # a parked row without a figure for the rest\n"
    )

    def judged(cells=None):
        measurements = [
            {"op": "Exp", "in": i, "out": o, "dest": d, "max": 0, "nonfinite": n}
            for (i, o, d), n in (cells or {}).items()
        ]
        _, regressions = render_headroom(
            table, _measured_cells(measurements), _nonfinite_cells(measurements)
        )
        return regressions

    at_row = ("Float32", "Bfp8_b", "Yes")
    plain = ("Float16", "Float16", "No")
    assert judged() == 1  # nothing measured is a sweep that did not run, not a pass
    assert judged({at_row: 2}) == 0  # what the row accounts for
    assert judged({at_row: 3}) == 1  # one more lane than it names
    assert judged({plain: 1}) == 1  # any on a row that names none
    # The count stands on its own, with or without a figure for the rest of the cell.
    (exp2,) = [row for (op, _), row in table.items() if op == "Exp2"]
    assert (recorded_nonfinite(exp2), recorded_max(exp2)) == (2, None)
    # A measurement written before the count existed judges as before.
    _, regressions = render_headroom(
        table,
        _measured_cells(
            [{"op": "Exp", "in": "Float16", "out": "Float16", "dest": "No", "max": 3}]
        ),
    )
    assert regressions == 0


# ── The command line, which is what the guard reports ─────────────────────────


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


_RAISED = _edited(
    0, "{in: Float16_b, out: Float16_b, max_ulp: 9, measured: 1, run: 2026-01-01}"
)


@pytest.mark.parametrize(
    "head, flags, status, says",
    [
        (_BASE, [], 0, "No budget changed"),
        (_RAISED, [], 1, "loosen a gate"),
        (_RAISED, ["--allow-raises"], 0, "carry no fresh measurement"),
        (
            _head(
                "{in: Float16_b, max_ulp: 1}  # a", "{out: Float16_b, max_ulp: 9}  # b"
            ),
            ["--allow-raises"],
            2,
            "cannot load",
        ),
    ],
    ids=["unchanged", "raised", "raised-advisory", "ambiguous-even-advisory"],
)
def test_diff_exit_status(tmp_path, capsys, head, flags, status, says):
    """Under --allow-raises the exit status fails only a table the registry refuses:
    the flag admits a loosened gate, not a table no gate can be read from."""
    out = tmp_path / "report.md"
    argv = [
        "diff",
        "--base",
        _write(tmp_path, "base.yaml", _BASE),
        "--head",
        _write(tmp_path, "head.yaml", head),
        "--out",
        str(out),
        *flags,
    ]
    assert main(argv) == status
    assert says in out.read_text(encoding="utf-8")
    assert says in capsys.readouterr().out


@pytest.mark.parametrize("worst, status", [(1, 0), (9, 1)], ids=["clean", "over"])
def test_headroom_exit_status(tmp_path, worst, status):
    import json

    measured = _write(
        tmp_path,
        "m.jsonl",
        json.dumps({"op": "Abs", "in": "Float16_b", "out": "Float16_b", "max": worst})
        + "\n",
    )
    argv = ["headroom", "--table", _write(tmp_path, "t.yaml", _BASE)]
    assert main([*argv, "--measured", measured]) == status


# ── The sweep's failed tests, for the headroom report ─────────────────────────

_JUNIT = """<testsuites><testsuite name="pytest">
<testcase classname="test_unary_sfpu_ulp"
  name="test_unary_sfpu_ulp_sweep[Abs-in:Float16_b-out:Float16_b-approx:No-dest_acc:Yes]">
  <failure message="AssertionError: Abs Float16_b-&gt;Float16_b approx=No dest_acc=Yes: failed a 1-step budget over 65279 swept lanes; Raw maximum 9 ULP&#10;assert False">trace</failure>
</testcase>
<testcase classname="test_unary_sfpu_ulp"
  name="test_unary_sfpu_ulp_sweep[Exp-in:Float16-out:Float16-approx:No-dest_acc:No]">
  <failure message="AssertionError: Exp Float16-&gt;Float16 approx=No dest_acc=No: 3 lane(s) disagreeing with the golden about being finite (golden -&gt; result: x=11.1: 65504 -&gt; inf)">trace</failure>
</testcase>
<testcase classname="test_unary_sfpu_ulp"
  name="test_unary_sfpu_ulp_sweep[Neg-in:Float16_b-out:Float16_b-approx:No-dest_acc:No]"/>
</testsuite></testsuites>"""


def test_a_failed_sweep_test_is_named_by_its_parameters_and_its_figure():
    """The summary has to say which combination failed and with what value. The
    test id is the combination; the assertion's first line carries the value."""
    first, second = junit_failures(_JUNIT)
    assert first.test == "[Abs-in:Float16_b-out:Float16_b-approx:No-dest_acc:Yes]"
    assert first.cell == ("Abs", "Float16_b", "Float16_b", "No", "Yes")
    assert "Raw maximum 9 ULP" in first.message and "assert False" not in first.message
    assert second.cell == ("Exp", "Float16", "Float16", "No", "No")


def test_a_failure_the_measurements_already_list_is_not_repeated():
    """An over-budget gated cell fails its test and is in the measurements too: listed
    once, with measured and budget. A failure no measurement describes -- here an
    overflow on a gated cell -- gets its own section and fails the comparison."""
    table = parse_table(
        "Abs:\n  - {in: Float16_b, out: Float16_b, max_ulp: 1}  # max 1 ULP\n"
    )
    rows = [
        {
            "op": "Abs",
            "in": "Float16_b",
            "out": "Float16_b",
            "approx": "No",
            "dest": "Yes",
            "arch": "WORMHOLE",
            "max": 9,
        }
    ]
    report, regressions = render_headroom(
        table, _measured_cells(rows), _nonfinite_cells(rows), junit_failures(_JUNIT)
    )
    assert "| 9 | 1 | over budget |" in report
    assert "[Abs-in:Float16_b" not in report  # already in the over-budget row
    assert "[Exp-in:Float16-out:Float16-approx:No-dest_acc:No]" in report
    assert "65504 -> inf" in report
    assert regressions == 2


def test_headroom_reads_the_junit_report_and_survives_no_measurements(tmp_path):
    """A sweep that died before its first cell writes no measurements file; the report
    still names the failures, and the comparison fails."""
    argv = [
        "headroom",
        "--table",
        _write(tmp_path, "t.yaml", _BASE),
        "--measured",
        str(tmp_path / "missing.jsonl"),
        "--junit",
        _write(tmp_path, "r.xml", _JUNIT),
        "--out",
        str(tmp_path / "r.md"),
    ]
    assert main(argv) == 1
    assert "Raw maximum 9 ULP" in (tmp_path / "r.md").read_text(encoding="utf-8")


# ── Fail closed: a cell that ran must be on the record ────────────────────────


def _row(op, in_fmt, out_fmt, approx, dest, worst):
    return {
        "op": op,
        "in": in_fmt,
        "out": out_fmt,
        "approx": approx,
        "dest": dest,
        "arch": "WORMHOLE",
        "max": worst,
    }


def test_a_sweep_test_that_finished_without_a_measurement_fails_the_comparison():
    """The recorder swallows a failed write, so a full disk leaves the JSONL short while
    pytest stays green. Every test that did not fail wrote a row; a missing one is a
    cell nothing judged, and the comparison has to say so and fail."""
    table = parse_table(
        "Abs:\n  - {in: Float16_b, out: Float16_b, max_ulp: 1}  # max 1 ULP\n"
        "Neg:\n  - {in: Float16_b, out: Float16_b, max_ulp: 0}  # max 0 ULP\n"
    )
    assert junit_completed(_JUNIT) == [("Neg", "Float16_b", "Float16_b", "No", "No")]
    abs_only = [_row("Abs", "Float16_b", "Float16_b", "No", "Yes", 1)]
    report, regressions = render_headroom(
        table, _measured_cells(abs_only), completed=junit_completed(_JUNIT)
    )
    assert "Ran without a measurement — 1" in report
    assert "`Neg, Float16_b, Float16_b, No, No`" in report
    assert regressions == 1

    both = abs_only + [_row("Neg", "Float16_b", "Float16_b", "No", "No", 0)]
    report, regressions = render_headroom(
        table, _measured_cells(both), completed=junit_completed(_JUNIT)
    )
    assert "without a measurement" not in report
    assert regressions == 0


def test_a_skipped_tolerance_cell_counts_as_measured_through_its_overflow_row():
    """A tolerance cell nothing can measure records only its non-finite lane count; that
    row is the cell's measurement, so the cell is not reported as unrecorded."""
    table = parse_table(
        "Neg:\n  - {in: Float16_b, out: Float16_b, metric: tolerance, nonfinite: 3, "
        "run: 2026-01-01}\n"
    )
    row = dict(_row("Neg", "Float16_b", "Float16_b", "No", "No", 0), nonfinite=3)
    report, regressions = render_headroom(
        table,
        _measured_cells([row]),
        _nonfinite_cells([row]),
        completed=[("Neg", "Float16_b", "Float16_b", "No", "No")],
    )
    assert "without a measurement" not in report
    assert regressions == 0


def test_an_empty_comparison_is_not_a_pass(tmp_path):
    """Nothing measured and nothing failed is a sweep that did not run."""
    argv = [
        "headroom",
        "--table",
        _write(tmp_path, "t.yaml", _BASE),
        "--measured",
        _write(tmp_path, "m.jsonl", ""),
    ]
    assert main(argv) == 1


def test_a_named_junit_report_that_is_missing_fails_the_comparison(tmp_path, capsys):
    """The cmd always names the report; its absence means pytest never finished, so
    which cells ran is unknown and the measurements alone cannot pass the run."""
    import json

    measured = _write(
        tmp_path,
        "m.jsonl",
        json.dumps({"op": "Abs", "in": "Float16_b", "out": "Float16_b", "max": 1})
        + "\n",
    )
    argv = [
        "headroom",
        "--table",
        _write(tmp_path, "t.yaml", _BASE),
        "--measured",
        measured,
        "--junit",
        str(tmp_path / "never_written.xml"),
    ]
    assert main(argv) == 1
    assert "No JUnit report" in capsys.readouterr().out


# ── What the registry refuses, and what it binds ───────────────────────────────


def test_the_spelled_out_sets_are_the_registrys():
    """The tool cannot import the registry, so its copies are held equal here."""
    from helpers.format_config import DataFormat
    from helpers.sfpu_accuracy_budget import _CONTRACT_FIELDS
    from helpers.ulp import has_ulp_gate

    assert CONTRACT_FIELDS == frozenset(_CONTRACT_FIELDS)
    assert ULP_GATED_OUTPUTS == {f.name for f in DataFormat if has_ulp_gate(f)}


@pytest.mark.parametrize(
    "row",
    [
        "{in: Float16_b, out: Float16_b, max_ulp: true}",
        "{in: Float16_b, out: Float16_b, max_ulp: -1}",
        "{in: Float16_b, out: Float16_b, max_ulp: 1.5}",
        "{in: Float16_b, out: Float16_b, metric: tolerance, atol: .nan}",
        "{in: Float16_b, out: Float16_b, max_ulp: 1, colour: red}",
        "{in: Float16_b, out: Float16_b, max_ulp: 1, atol: 0.1}",
        "{in: Float16_b, out: Float16_b, metric: tolerance, near_zero_atol: 0.5}",
        "{in: Float16_b, out: Float16_b, metric: tolerance, max_ulp: 1}",
        "{in: Float16_b, out: Float16_b, metric: ulps, max_ulp: 1}",
        "{in: Float16_b, out: Float16_b, approx: Maybe, max_ulp: 1}",
        "{in: Float16_b, in: Float16, out: Float16_b, max_ulp: 1}",
    ],
)
def test_a_table_the_registry_refuses_gets_no_verdict(tmp_path, capsys, row):
    head = _head(row)
    argv = [
        "diff",
        "--base",
        _write(tmp_path, "base.yaml", _BASE),
        "--head",
        _write(tmp_path, "head.yaml", head),
        "--allow-raises",
    ]
    assert main(argv) == 2
    assert "cannot load" in capsys.readouterr().out


def test_deleting_a_parent_row_is_seen_on_the_flag_value_no_row_pins():
    """`{approx: Yes}` alone pins one value; the `No` cells fell back to the broad row,
    and deleting it leaves them on nothing."""
    for flag in ("approx", "dest"):
        figure = "measured: 9, run: 2026-01-01"
        base = _head(
            f"{{in: Float16_b, out: Float32, metric: tolerance, {figure}}}",
            f'{{in: Float16_b, out: Float32, {flag}: "Yes", metric: tolerance, {figure}}}',
        )
        head = _head(
            f'{{in: Float16_b, out: Float32, {flag}: "Yes", metric: tolerance, {figure}}}'
        )
        kinds = {c.kind for c in _changes(base, head)}
        assert kinds == {"baseline_dropped"}, flag


def test_deleting_a_row_that_records_only_a_nonfinite_count_drops_a_baseline():
    base = _head(
        "{in: Float32, out: Float16, metric: tolerance, nonfinite: 3, run: 2026-01-01}"
        "  # not measurable: lanes disagreeing with the golden about being finite"
    )
    head = _head("{in: Float16_b, out: Float16_b, max_ulp: 1}  # a")
    assert "baseline_dropped" in {c.kind for c in _changes(base, head)}


@pytest.mark.parametrize(
    "base_row, head_row",
    [
        (
            "{in: Bfp8_b, out: Bfp8_b, metric: tolerance, atol: 0.13}",
            "{in: Bfp8_b, out: Bfp8_b, metric: tolerance, atol: 0.13, rtol: 0.9}",
        ),
        (
            "{in: Bfp8_b, out: Bfp8_b, metric: tolerance}",
            "{in: Bfp8_b, out: Bfp8_b, metric: tolerance, atol: 100}",
        ),
    ],
)
def test_a_bound_that_appears_may_be_looser_than_the_default(base_row, head_row):
    kinds = {c.kind for c in _changes(_head(base_row), _head(head_row))}
    assert kinds == {"tolerance_widened"}


def test_a_new_row_without_provenance_of_its_own_is_not_remeasured():
    base = "Abs:\n  - {max_ulp: 2}  # exact\n"
    head = base + "  - {in: Float32, max_ulp: 1000}  # a comment is not a measurement\n"
    (change,) = _changes(base, head)
    assert change.kind == "raised" and not change.remeasured


def test_a_step_row_on_an_output_with_no_per_element_ulp_does_not_gate():
    """The registry hands such a cell its tolerance; the tool must not judge it."""
    table = parse_table(_head("{in: Float16_b, out: Bfp4_b, max_ulp: 25}"))
    (row,) = table.values()
    assert not _gates(row, (("in", "Float16_b"), ("out", "Bfp4_b")))
    report, over = render_headroom(table, _measured(("Float16_b", "Bfp4_b", 393)))
    assert over == 0 and "over budget" not in report


def test_a_measured_cell_no_row_covers_is_counted_not_dropped():
    table = parse_table(
        _head("{in: Float16_b, out: Float16_b, max_ulp: 1}  # max 1 ULP")
    )
    report, over = render_headroom(table, _measured(("Float16", "Float32", 10**6)))
    assert over == 0
    assert "1 covered by no row at all" in report


def test_a_tolerance_cell_that_ranked_no_lane_regresses_against_its_figure():
    table = parse_table(
        _head(
            "{in: Bfp8_b, out: Bfp8_b, metric: tolerance, measured: 393, run: 2026-01-01}"
        )
    )
    rows = [
        {"op": "Abs", "in": "Bfp8_b", "out": "Bfp8_b", "max": 0, "lanes": 0},
    ]
    worst, nonfinite, empty = _readings(rows)
    report, regressed = render_headroom(table, worst, nonfinite, empty=empty)
    assert regressed == 1 and "no lane measurable" in report


def test_only_a_skip_that_recorded_counts_as_completed():
    junit = (
        '<testsuites><testsuite name="pytest">'
        '<testcase name="t[Abs-in:Float16_b-out:Float16_b-approx:No-dest_acc:Yes]">'
        f'<skipped message="x: {RECORDED_SKIP} -- on the tolerance metric"/></testcase>'
        '<testcase name="t[Abs-in:Float16-out:Float16-approx:No-dest_acc:Yes]">'
        '<skipped message="the table\'s budgets are WORMHOLE measurements"/></testcase>'
        "</testsuite></testsuites>"
    )
    assert junit_completed(junit) == [("Abs", "Float16_b", "Float16_b", "No", "Yes")]


@pytest.mark.parametrize(
    "measured, junit, says",
    [
        ('{"op": "Abs", "in": "Float16_b"\n', None, "is not JSON"),
        ('{"op": "Abs"}\n', None, "malformed"),
        ("", "<testsuites><testcase", "not well-formed"),
    ],
)
def test_an_unreadable_input_is_named_not_blamed_on_the_table(
    tmp_path, capsys, measured, junit, says
):
    argv = [
        "headroom",
        "--table",
        _write(tmp_path, "t.yaml", _BASE),
        "--measured",
        _write(tmp_path, "m.jsonl", measured),
    ]
    if junit is not None:
        argv += ["--junit", _write(tmp_path, "j.xml", junit)]
    assert main(argv) == 2
    out = capsys.readouterr().out
    assert says in out and "cannot load" not in out

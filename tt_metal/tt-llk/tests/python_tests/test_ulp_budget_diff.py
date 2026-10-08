# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the ULP regression check in ``helpers/ulp_budget_diff.py``.

No device and no torch: that module runs on a slim CI runner against two revisions of
a text file. Pinned here: which changes count as a regression, and that its parse of
the live table agrees with the real loader's -- it reads the file a second way, so the
two are free to drift unless something says they may not.
"""

import ast
import sys
from pathlib import Path

import pytest
from helpers.ulp_budget_diff import (
    _MAX_ROWS,
    KEY_FIELDS,
    _measured_cells,
    _nonfinite_cells,
    _resolve,
    compare,
    junit_failures,
    main,
    parse_table,
    recorded_max,
    recorded_nonfinite,
    render_budget_diff,
    render_headroom,
)

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


_HEADER = "Abs:  # measured by: sweep X, wormhole, 2026-01-01"
_BASE_ROWS = (
    "{in: Float16_b, out: Float16_b, max_ulp: 1}  # max 1 ULP",
    "{in: Float16, out: Float16_b, max_ulp: 4}  # max 3 ULP",
    "{in: Bfp8_b, out: Bfp8_b, metric: tolerance}  # max 393 ULP, block-quantized",
    "{in: Float32, out: Float32, max_ulp: 2}  # max 1 ULP",
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


def _refuses(match):
    """The suite's ``expect_error`` fixture needs a device; these are host-only tests."""
    return pytest.raises(ValueError, match=match)  # allow-pytest.raises: host-only test


# ── What counts as a regression ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "comment, remeasured",
    [("max 1 ULP", False), ("max 8 ULP, re-measured 2026-02-02", True)],
    ids=["comment-untouched", "re-measured"],
)
def test_a_raised_budget_is_a_regression_marked_by_its_provenance(comment, remeasured):
    """The raise is always surfaced. Whether the row's comment changed with it is what
    separates a re-measurement from a number fitted to a failure."""
    head = _edited(0, f"{{in: Float16_b, out: Float16_b, max_ulp: 9}}  # {comment}")
    (change,) = _changes(_BASE, head)
    assert change.kind == "raised" and change.is_regression
    assert change.remeasured is remeasured


def test_losing_the_gate_entirely_is_a_regression():
    """Demotion to tolerance is the loosest change: the cell stops being judged on a
    step budget at all, and a bounds check cannot see it."""
    head = _edited(
        0, "{in: Float16_b, out: Float16_b, metric: tolerance}  # max 14337 ULP"
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
        "{in: Float16_b, out: Float16_b, max_ulp: 0}  # max 0 ULP",
        _BASE_ROWS[1],
        "{in: Bfp8_b, out: Bfp8_b, max_ulp: 3}  # max 2 ULP",
        _BASE_ROWS[3],
        "{in: Float16, out: Float32, max_ulp: 7}  # max 6 ULP",
    )
    changes = _changes(_BASE, head)
    assert not any(c.is_regression for c in changes)
    assert {c.kind for c in changes} == {"tightened", "gated"}


def test_a_row_the_sweep_collapses_is_not_a_loss_while_its_cells_keep_their_budget():
    """The emitter drops a key dimension when both values measure alike, so a re-emit
    deletes `approx`-keyed rows and writes one without. Nothing any query resolves to
    has changed, and a row diff reported 16 of these as regressions."""
    base = _head(
        '{in: Float16_b, out: Float32, approx: "No", dest: "Yes", max_ulp: 36018}  # a',
        '{in: Float16_b, out: Float32, approx: "Yes", dest: "Yes", max_ulp: 36018}  # b',
    )
    same = _head('{in: Float16_b, out: Float32, dest: "Yes", max_ulp: 36018}  # c')
    changes = _changes(base, same)
    assert not any(c.is_regression for c in changes)
    # The one honest difference: a query with `approx` unset matched neither keyed row
    # before and matches the collapsed one now, so that variant is newly gated.
    assert [(c.kind, c.variants) for c in changes] == [("gated", 1)]
    # A raise through the collapsed row is one change covering both variants,
    # re-measured because the deciding row's comment is new.
    raised = _head('{in: Float16_b, out: Float32, dest: "Yes", max_ulp: 36030}  # c')
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
    report = render_budget_diff([change], "the-label")
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
    with _refuses("equally specific"):
        _resolve(table, "Abs", _BF16)
    with _refuses("equally specific"):
        compare(parse_table(_BASE), table)


# ── Tolerance cells: what the sweep holds them to ─────────────────────────────


@pytest.mark.parametrize(
    "head_row, kind",
    [
        (
            "{in: Bfp8_b, out: Bfp8_b, metric: tolerance}  # max 500 ULP",
            "baseline_raised",
        ),
        (
            "{in: Bfp8_b, out: Bfp8_b, metric: tolerance}  # not measurable: 3 lane(s) "
            "disagreeing with the golden about being finite (x); max 393 ULP over the "
            "65000 measurable lanes",
            "baseline_raised",
        ),
        (
            "{in: Bfp8_b, out: Bfp8_b, metric: tolerance}  # block-quantized",
            "baseline_dropped",
        ),
        ("{in: Bfp8_b, out: Bfp8_b, metric: tolerance}  # max 12 ULP", None),
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
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 1.0e-07}  # max 1 ULP",
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-05}  # max 1 ULP",
        "floor_widened",
    ),
    (
        "{in: Float16_b, out: Float16_b, max_ulp: 2}  # max 1 ULP",
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-05}  # max 1 ULP",
        "floor_widened",
    ),
    # A smaller max_ulp must not mask a wider floor: the floor can more than pay for it.
    (
        "{in: Float16_b, out: Float16_b, max_ulp: 8, near_zero_atol: 1.0e-07}  # max 1 ULP",
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-02}  # max 1 ULP",
        "floor_widened",
    ),
    (
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-05}  # max 1 ULP",
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 1.0e-07}  # max 1 ULP",
        None,
    ),
    # On a tolerance row nothing consults the floor.
    (
        "{in: Float16_b, out: Float16_b, metric: tolerance}  # max 393 ULP",
        "{in: Float16_b, out: Float16_b, metric: tolerance, near_zero_atol: 0.5}  # max 393 ULP",
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
        "tolerance-row",
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
    text = render_budget_diff(
        _changes(_head(base_row), _head(head_row)), "ulp-budget-raise-approved"
    )
    assert "1e-07" in text and "5e-05" in text


def test_an_unchanged_table_reports_nothing():
    assert _changes(_BASE, _BASE) == []
    assert "No budget changed" in render_budget_diff([], "the-label")


def test_the_report_names_the_cell_the_change_and_the_label():
    head = _head("{in: Float16_b, out: Float16_b, max_ulp: 9}  # max 1 ULP")
    report = render_budget_diff(_changes(_BASE, head), "ulp-budget-raise-approved")
    assert "loosen a gate" in report
    assert "in: Float16_b, out: Float16_b" in report
    assert "budget raised" in report
    assert "ulp-budget-raise-approved" in report
    assert "**no**" in report  # not re-measured


# ── Reading the table ─────────────────────────────────────────────────────────


def test_an_anchor_and_its_alias_are_both_real_rows():
    """The coarse-LUT tolerance pair is a YAML anchor and an alias, and a line-oriented
    reader skipped both -- silently, being tolerance rows with no budget to compare. A
    budget hidden behind an alias would have been invisible; the loader tie-back below
    is what caught it."""
    table = parse_table(
        "GeluAppx:\n"
        "  - &shared {max_ulp: 3}  # max 2 ULP\n"
        "SigmoidAppx:\n"
        "  - *shared  # same cause, same number\n"
    )
    assert {cell[0] for cell in table} == {"GeluAppx", "SigmoidAppx"}
    assert all(row.max_ulp == 3 for row in table.values())
    # Each keeps its own comment, so a re-measurement on one is visible.
    assert table[("SigmoidAppx", ())].provenance == "same cause, same number"


def test_a_merge_key_row_is_read_through():
    """The registry loads the table with PyYAML's SafeLoader, which resolves `<<`."""
    table = parse_table(
        "Base:\n"
        "  - &b {out: Float16_b, max_ulp: 2}  # max 1 ULP\n"
        "Abs:\n"
        "  - {<<: *b, max_ulp: 5}  # max 4 ULP\n"
    )
    assert table[("Abs", (("out", "Float16_b"),))].max_ulp == 5


def test_a_row_without_a_comment_inherits_its_op_header():
    """The table puts the run identity on the op header, once, and many gated rows carry
    no inline comment. Without the fallback their provenance is permanently empty, so a
    raise on one could never register as re-measured. A row's own comment still wins."""

    def table(run, budget):
        return parse_table(
            f"Abs:  # measured by: {run}, wormhole\n"
            f"  - {{out: Float32, max_ulp: {budget}}}\n"
            "  - {out: Float16_b, max_ulp: 1}  # max 0 ULP, its own row\n"
        )

    base = table("sweep A", 1)
    assert base[("Abs", (("out", "Float32"),))].provenance.startswith(
        "measured by: sweep A"
    )
    assert base[("Abs", (("out", "Float16_b"),))].provenance == "max 0 ULP, its own row"
    # Raising the comment-less row shows as re-measured only through a new header.
    (raised,) = [c for c in compare(base, table("sweep B", 4)) if c.is_regression]
    assert raised.kind == "raised" and raised.remeasured
    (stale,) = [c for c in compare(base, table("sweep A", 4)) if c.is_regression]
    assert stale.kind == "raised" and not stale.remeasured


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
    """The emitter writes both numbers into one note; each reader takes its own, and
    the finite lanes of a demoted cell stay judged."""
    table = parse_table(
        'I1:\n  - {in: Float16_b, out: Float32, dest: "No", metric: tolerance}  # not '
        "measurable: 31120 lane(s) disagreeing with the golden about being finite "
        "(golden -> result: x=91: inf -> -1.16e37); max 7 ULP over the 34159 measurable "
        "lanes\n"
    )
    row = next(iter(table.values()))
    assert recorded_nonfinite(row) == 31120
    assert recorded_max(row) == 7


def test_a_duplicated_cell_is_refused_rather_than_judged():
    """`_load_table` rejects two rows of equal specificity, so the table cannot load.
    Keeping the last row silently produced a verdict -- a *tightening*, even -- for a
    table the registry would not accept at all."""
    with _refuses("duplicate row"):
        parse_table(
            _head(
                "{in: Float16_b, out: Float16_b, max_ulp: 9}  # max 1 ULP",
                "{in: Float16_b, out: Float16_b, max_ulp: 1}  # max 1 ULP",
            )
        )


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
    a regression there can show. The row's own "max N ULP" is the baseline; a cell
    whose row records none is not judged at all."""
    table = parse_table(
        "Abs:\n"
        "  - {in: Float16_b, out: Float16_b, metric: tolerance}  # max 393 ULP, budget 433 > ceiling 7\n"
        "  - {in: Float16, out: Float16_b, metric: tolerance}  # block-quantized, so tolerance\n"
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


def test_a_long_report_is_capped_and_says_how_many_it_withheld():
    """A PR comment has a size limit, and 2,000-odd gated cells could blow past it."""
    extra = 5
    n = _MAX_ROWS + extra
    table = "Abs:\n" + "".join(
        f'  - {{in: Float16_b, out: Float16_b, dest: "{i}", max_ulp: 8}}\n'
        for i in range(n)
    )
    rows = [
        {"op": "Abs", "in": "Float16_b", "out": "Float16_b", "dest": str(i), "max": 99}
        for i in range(n)
    ]
    report, over = render_headroom(parse_table(table), _measured_cells(rows))
    assert over == n
    # Delimited: a bare "5 more" is also in "45 more".
    assert f"| _… {extra} more_ |" in report
    assert report.count("over budget |") == _MAX_ROWS


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


def test_the_tool_imports_nothing_but_the_standard_library_and_yaml():
    """The PR check runs on a slim runner: no torch, no ttexalens, no LLK venv. Read off
    the module's own AST: the import list is the actual property, and a subprocess would
    prove it for one environment only."""
    tool = Path(__file__).parent / "helpers" / "ulp_budget_diff.py"
    roots = set()
    for node in ast.walk(ast.parse(tool.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            # A relative import would reach back into `helpers`, the thing this module
            # must not do.
            assert node.level == 0, f"relative import of {node.module!r}"
            if node.module:
                roots.add(node.module.split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"yaml"}
    assert roots <= allowed, f"imports outside stdlib + yaml: {sorted(roots - allowed)}"
    assert "helpers" not in roots and "torch" not in roots


def test_the_workflow_invokes_the_tool_by_path_not_as_a_module():
    """`helpers/__init__.py` imports ttexalens, which the slim runner does not have, so
    `-m helpers.ulp_budget_diff` would fail there however clean this module's own
    imports are."""
    repo = Path(__file__).resolve().parents[4]
    workflow = repo / ".github/workflows/llk-sfpu-accuracy.yaml"
    if not workflow.exists():  # pragma: no cover - the guard ships with the workflow
        pytest.skip("workflow not in this checkout")
    # Comments stripped: the workflow explains in prose why it does *not* use `-m`.
    body = "\n".join(
        line
        for line in workflow.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "helpers/ulp_budget_diff.py diff" in body
    assert "-m helpers.ulp_budget_diff" not in body


def test_the_key_dimensions_are_the_registrys():
    """A new key dimension in the registry that the tool did not know would make every
    row pinning it invisible to the guard, while each copy agreed with itself."""
    from helpers.sfpu_accuracy_budget import _KEY_FIELDS

    assert _BUDGET_KEY_ATTR == _KEY_FIELDS
    assert KEY_FIELDS == tuple(_KEY_FIELDS)


def test_both_checks_are_part_of_the_required_pr_gate():
    """A red check in a workflow of its own blocked nothing: no branch rule requires it,
    and a path-filtered check cannot be required. Both count because PR Gate calls them
    and "PR Gate Status" -- a required check -- fails when either fails. The sweep runs
    only when an SFPU kernel or the SFPI pin changed."""
    import yaml

    repo = Path(__file__).resolve().parents[4]
    gate = repo / ".github/workflows/pr-gate.yaml"
    if not gate.exists():  # pragma: no cover - the checks ship with the workflow
        pytest.skip("workflow not in this checkout")
    jobs = yaml.safe_load(gate.read_text(encoding="utf-8"))["jobs"]
    status = jobs["workflow-status"]
    (check,) = [
        step
        for step in status["steps"]
        if step.get("uses") == "./.github/actions/workflow-status"
    ]
    listed = check["with"]["required-jobs"] + "," + check["with"]["optional-jobs"]
    listed = {job.strip() for job in listed.split(",")}
    for name, mode in (
        ("llk-sfpu-ulp-budget-guard", "guard"),
        ("llk-sfpu-ulp-sweep", "sweep"),
    ):
        job = jobs[name]
        assert job["uses"] == "./.github/workflows/llk-sfpu-accuracy.yaml"
        assert job["with"]["check"] == mode
        assert name in status["needs"] and name in listed
    assert "llk-sfpu-kernels-changed == 'true'" in jobs["llk-sfpu-ulp-sweep"]["if"]
    outputs = jobs["find-changed-files"]["outputs"]
    assert "llk-sfpu-kernels-changed" in outputs


def test_the_sweep_runs_the_sweep_and_the_comparison_and_fails_on_either():
    """It must run the sweep with --ulp-measure *and* the headroom comparison over it,
    with the JUnit report so failures no measurement describes reach the PR comment, and
    exit with either's failure: a cmd that stopped after a red sweep would publish no
    report, and one that ignored headroom would pass every tolerance-cell regression."""
    import yaml

    repo = Path(__file__).resolve().parents[4]
    matrix = repo / "tests/pipeline_reorg/llk_sfpu_accuracy_tests.yaml"
    if not matrix.exists():  # pragma: no cover - the entry ships with the matrix
        pytest.skip("matrix not in this checkout")
    (entry,) = yaml.safe_load(matrix.read_text(encoding="utf-8"))
    cmd = entry["cmd"]
    junit = "ulp_sweep_junit.xml"
    assert "--ulp-measure=ulp_measurements.jsonl" in cmd
    assert "test_unary_sfpu_ulp.py || status=$?" in cmd
    assert "helpers/ulp_budget_diff.py headroom" in cmd
    assert "--measured ulp_measurements.jsonl" in cmd
    assert f"--junitxml={junit}" in cmd and f"--junit {junit}" in cmd
    # The workflow comments this file on the PR.
    assert "--out ulp_headroom_report.md" in cmd
    assert cmd.count("|| status=$?") == 2 and 'exit "$status"' in cmd
    assert "set -e" not in cmd.replace("set -uo", ""), "-e would skip the report"
    assert set(entry["skus"]) == {"wh_n150_civ2"}
    # verify-changed-tests would run the cmd from the repo root, where it cannot work.
    verify = (repo / ".github/workflows/verify-changed-tests.yaml").read_text(
        encoding="utf-8"
    )
    unsupported = yaml.safe_load(verify)["env"]["UNSUPPORTED_YAMLS"].split()
    assert "llk_sfpu_accuracy_tests.yaml" in unsupported


def test_this_parse_agrees_with_the_registry_loader_on_the_live_table():
    """This module reads the table a second way -- text, no torch -- so the CI check can
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
        _winner,
    )
    from helpers.ulp_sweep import sweep_cells

    table = parse_table(_TABLE_PATH.read_text(encoding="utf-8"))
    cells = sweep_cells()
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
            compared += 1
    assert cells and _SFPU_ACCURACY_BUDGET, "nothing to compare"
    assert compared == len(cells) * len(_SFPU_ACCURACY_BUDGET) > 0


def test_the_headroom_report_fails_an_overflow_the_row_does_not_account_for():
    """A "not measurable" row records how many lanes went non-finite; the sweep skips
    such a tolerance cell but records the count, and more lanes than the row names --
    or any on a row that names none -- is a regression the sweep must fail on."""
    table = parse_table(
        "Exp:\n"
        '  - {in: Float32, out: Bfp8_b, dest: "Yes", metric: tolerance}'
        "  # not measurable: 2 lane(s) disagreeing with the golden about being finite "
        "(golden -> result: x=3e38: 3e38 -> inf)\n"
        '  - {in: Float16, out: Float16, dest: "No", metric: tolerance}  # max 511 ULP\n'
        "Exp2:\n"
        '  - {in: Float32, out: Bfp8_b, dest: "Yes", metric: tolerance}'
        "  # not measurable: 2 lane(s) non-finite against a finite golden (x=3e38)\n"
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
    assert judged() == 0
    assert judged({at_row: 2}) == 0  # what the row accounts for
    assert judged({at_row: 3}) == 1  # one more lane than it names
    assert judged({plain: 1}) == 1  # any on a row that names none
    # The older spelling of a "not measurable" row still names its count.
    (exp2,) = [row for (op, _), row in table.items() if op == "Exp2"]
    assert recorded_nonfinite(exp2) == 2
    # A measurement written before the count existed judges as before.
    _, regressions = render_headroom(
        table,
        _measured_cells(
            [{"op": "Exp", "in": "Float16", "out": "Float16", "dest": "No", "max": 3}]
        ),
    )
    assert regressions == 0


# ── The command line, which is what the guard blocks on ───────────────────────


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


_RAISED = _edited(0, "{in: Float16_b, out: Float16_b, max_ulp: 9}  # max 1 ULP")


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
    ids=["unchanged", "raised", "raised-with-label", "ambiguous-even-with-label"],
)
def test_diff_exit_status(tmp_path, capsys, head, flags, status, says):
    """The exit status is the whole of what the guard blocks on. A table the registry
    refuses is refused under the label too: the label admits a loosened gate, not a
    table no gate can be read from."""
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


# ── The sweep's failed tests, for the PR comment ──────────────────────────────

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
    """The PR comment has to say which combination failed and with what value. The
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

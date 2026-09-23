# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the ULP regression check in ``helpers/ulp_budget_diff.py``.

No device and no torch: the point of that module is that it runs on a slim CI runner
against two revisions of a text file. What is pinned here is which changes count as
a regression, and that its parse of the live table agrees with the real loader's --
it reads the file a second way, so the two are free to drift unless something says
they may not.
"""


import pytest
from helpers.ulp_budget_diff import (
    _measured_cells,
    compare,
    parse_table,
    render_budget_diff,
    render_headroom,
)

_BASE = """\
Abs:  # measured by: sweep X, wormhole, 2026-01-01, except where a row says otherwise
  - {in: Float16_b, out: Float16_b, max_ulp: 1}  # max 1 ULP
  - {in: Float16, out: Float16_b, max_ulp: 4}  # max 3 ULP
  - {in: Bfp8_b, out: Bfp8_b, metric: tolerance}  # max 393 ULP, block-quantized
  - {in: Float32, out: Float32, max_ulp: 2}  # max 1 ULP
"""


def _head(*rows):
    return "Abs:  # measured by: sweep X, wormhole, 2026-01-01\n" + "".join(
        f"  - {r}\n" for r in rows
    )


def _refuses(match):
    """The suite's ``expect_error`` fixture needs a device; these are host-only tests."""
    return pytest.raises(ValueError, match=match)  # allow-pytest.raises: host-only test


def _kinds(base, head):
    return {c.cell[1]: c.kind for c in compare(parse_table(base), parse_table(head))}


def test_a_raised_budget_is_a_regression():
    head = _head(
        "{in: Float16_b, out: Float16_b, max_ulp: 9}  # max 1 ULP",
        "{in: Float16, out: Float16_b, max_ulp: 4}  # max 3 ULP",
        "{in: Bfp8_b, out: Bfp8_b, metric: tolerance}  # max 393 ULP, block-quantized",
        "{in: Float32, out: Float32, max_ulp: 2}  # max 1 ULP",
    )
    changes = compare(parse_table(_BASE), parse_table(head))
    assert [c.kind for c in changes] == ["raised"]
    assert changes[0].is_regression
    # The provenance is untouched, so the number was edited rather than measured.
    assert not changes[0].remeasured


def test_a_raise_with_a_fresh_measurement_is_still_reported_but_marked():
    """Both halves matter. The raise is always surfaced -- it is a real loss of
    tightness -- but whether the row was re-measured is what separates a legitimate
    one from a number fitted to a failure."""
    head = _head(
        "{in: Float16_b, out: Float16_b, max_ulp: 9}  # max 8 ULP, re-measured 2026-02-02",
        "{in: Float16, out: Float16_b, max_ulp: 4}  # max 3 ULP",
        "{in: Bfp8_b, out: Bfp8_b, metric: tolerance}  # max 393 ULP, block-quantized",
        "{in: Float32, out: Float32, max_ulp: 2}  # max 1 ULP",
    )
    changes = compare(parse_table(_BASE), parse_table(head))
    assert [c.kind for c in changes] == ["raised"]
    assert changes[0].is_regression and changes[0].remeasured


def test_losing_the_gate_entirely_is_a_regression():
    """Demotion to tolerance is the loosest possible change: the cell stops being
    judged on a step budget at all, and a bounds check cannot see it."""
    head = _head(
        "{in: Float16_b, out: Float16_b, metric: tolerance}  # max 14337 ULP",
        "{in: Float16, out: Float16_b, max_ulp: 4}  # max 3 ULP",
        "{in: Bfp8_b, out: Bfp8_b, metric: tolerance}  # max 393 ULP, block-quantized",
        "{in: Float32, out: Float32, max_ulp: 2}  # max 1 ULP",
    )
    assert _kinds(_BASE, head)[(("in", "Float16_b"), ("out", "Float16_b"))] == "ungated"


def test_deleting_a_gated_row_is_a_regression_and_a_tolerance_row_is_not():
    """A row that gated nothing cannot be a loss when it goes."""
    head = _head(
        "{in: Float16, out: Float16_b, max_ulp: 4}  # max 3 ULP",
        "{in: Float32, out: Float32, max_ulp: 2}  # max 1 ULP",
    )
    kinds = _kinds(_BASE, head)
    assert kinds[(("in", "Float16_b"), ("out", "Float16_b"))] == "removed"
    assert kinds[(("in", "Bfp8_b"), ("out", "Bfp8_b"))] == "added"  # not a loss


def test_tightening_and_newly_gating_are_not_regressions():
    head = _head(
        "{in: Float16_b, out: Float16_b, max_ulp: 0}  # max 0 ULP",
        "{in: Float16, out: Float16_b, max_ulp: 4}  # max 3 ULP",
        "{in: Bfp8_b, out: Bfp8_b, max_ulp: 3}  # max 2 ULP",
        "{in: Float32, out: Float32, max_ulp: 2}  # max 1 ULP",
        "{in: Float16, out: Float32, max_ulp: 7}  # max 6 ULP",
    )
    changes = compare(parse_table(_BASE), parse_table(head))
    assert not any(c.is_regression for c in changes)
    assert {c.kind for c in changes} == {"tightened", "gated", "added"}


def test_an_unchanged_table_reports_nothing():
    assert compare(parse_table(_BASE), parse_table(_BASE)) == []
    assert "No budget changed" in render_budget_diff([], "the-label")


def test_the_report_names_the_cell_the_change_and_the_label():
    head = _head("{in: Float16_b, out: Float16_b, max_ulp: 9}  # max 1 ULP")
    report = render_budget_diff(
        compare(parse_table(_BASE), parse_table(head)), "ulp-budget-raise-approved"
    )
    assert "loosen a gate" in report
    assert "in: Float16_b, out: Float16_b" in report
    assert "budget raised" in report
    assert "ulp-budget-raise-approved" in report
    assert "**no**" in report  # not re-measured


def test_an_anchor_and_its_alias_are_both_real_rows():
    """The coarse-LUT tolerance pair is written as a YAML anchor and an alias, and a
    line-oriented reader skipped both -- silently, because they are tolerance rows
    with no budget to compare. A budget hidden behind an alias would have been
    invisible to this check; the loader tie-back below is what caught it."""
    table = parse_table(
        "GeluAppx:\n"
        "  - &shared {max_ulp: 3}  # max 2 ULP\n"
        "SigmoidAppx:\n"
        "  - *shared  # same cause, same number\n"
    )
    assert {cell[0] for cell in table} == {"GeluAppx", "SigmoidAppx"}
    assert all(row.max_ulp == 3 for row in table.values())
    # And each keeps its own comment, so a re-measurement on one is visible.
    assert table[("SigmoidAppx", ())].provenance == "same cause, same number"


def test_a_row_without_a_comment_inherits_its_op_header():
    """The table puts the run identity on the op header, once, and 42 gated rows carry
    no inline comment at all. Without the fallback their provenance is permanently
    empty, so a raise on one could never register as re-measured and the audit would
    report `no` whatever the author did.

    A row's own comment still wins, so a per-row measurement is not masked by the
    header.
    """
    base = parse_table(
        "Abs:  # measured by: sweep A, wormhole, 2026-01-01\n"
        "  - {out: Float32, max_ulp: 1}\n"
        "  - {out: Float16_b, max_ulp: 1}  # max 0 ULP, its own row\n"
    )
    assert base[("Abs", (("out", "Float32"),))].provenance.startswith(
        "measured by: sweep A"
    )
    assert base[("Abs", (("out", "Float16_b"),))].provenance == "max 0 ULP, its own row"

    # Raising the comment-less row and re-measuring shows up through the header.
    head = parse_table(
        "Abs:  # measured by: sweep B, wormhole, 2026-02-02\n"
        "  - {out: Float32, max_ulp: 4}\n"
        "  - {out: Float16_b, max_ulp: 1}  # max 0 ULP, its own row\n"
    )
    raised = [c for c in compare(base, head) if c.is_regression]
    assert [c.kind for c in raised] == ["raised"]
    assert raised[0].remeasured, "a header-only re-measurement must count"

    # And a raise with the header untouched still reports as not re-measured.
    stale = parse_table(
        "Abs:  # measured by: sweep A, wormhole, 2026-01-01\n"
        "  - {out: Float32, max_ulp: 4}\n"
        "  - {out: Float16_b, max_ulp: 1}  # max 0 ULP, its own row\n"
    )
    raised = [c for c in compare(base, stale) if c.is_regression]
    assert [c.kind for c in raised] == ["raised"]
    assert not raised[0].remeasured


def test_a_merge_key_row_is_read_through():
    """`yaml_table` gained `<<` support, so the table may use it; values come from
    PyYAML here for exactly that reason."""
    table = parse_table(
        "Base:\n"
        "  - &b {out: Float16_b, max_ulp: 2}  # max 1 ULP\n"
        "Abs:\n"
        "  - {<<: *b, max_ulp: 5}  # max 4 ULP\n"
    )
    assert table[("Abs", (("out", "Float16_b"),))].max_ulp == 5


# ─────────────────────────────────────────────────────────────────────────────
# The headroom half
# ─────────────────────────────────────────────────────────────────────────────


def _measure(**kw):
    row = {"op": "Abs", "in": None, "out": None, "approx": None, "dest": None, "max": 0}
    row.update(kw)
    return row


def test_several_measurements_of_one_cell_keep_the_worst():
    """A driver enumerates axes the budget key does not, so one cell is recorded
    more than once. Keeping the last would hide the worst of them."""
    cells = _measured_cells(
        [
            _measure(**{"in": "Float16_b", "out": "Float16_b", "max": 3}),
            _measure(**{"in": "Float16_b", "out": "Float16_b", "max": 11}),
            _measure(**{"in": "Float16_b", "out": "Float16_b", "max": 5}),
        ]
    )
    assert list(cells.values()) == [11]


def test_an_exact_cell_at_its_zero_budget_is_not_a_warning():
    """`0 == 0` is an op exact by construction doing what it claims, and the table
    enrols it precisely so any drift fails. Calling that "no headroom" buried the real
    ones under 52 lines of nothing on a 130-cell run."""
    table = parse_table("Abs:\n  - {in: Float16_b, out: Float16_b, max_ulp: 0}\n")
    report, over = render_headroom(
        table,
        _measured_cells(
            [_measure(**{"in": "Float16_b", "out": "Float16_b", "max": 0})]
        ),
    )
    assert over == 0
    assert "no headroom" not in report
    assert "headroom to spare" in report


def test_a_long_report_is_capped_and_says_how_many_it_withheld():
    """A PR comment has a size limit, and 2,321 gated cells could blow past it."""
    from helpers.ulp_budget_diff import _MAX_ROWS

    table = "Abs:\n" + "".join(
        f'  - {{in: Float16_b, out: Float16_b, dest: "{i}", max_ulp: 8}}\n'
        for i in range(_MAX_ROWS + 5)
    )
    rows = [
        _measure(**{"in": "Float16_b", "out": "Float16_b", "dest": str(i), "max": 99})
        for i in range(_MAX_ROWS + 5)
    ]
    report, over = render_headroom(parse_table(table), _measured_cells(rows))
    assert over == _MAX_ROWS + 5
    assert "5 more" in report
    assert report.count("over budget |") == _MAX_ROWS


@pytest.mark.parametrize(
    "measured, expect",
    [(9, "over budget"), (1, "no headroom"), (0, "could tighten")],
    ids=["over", "tight", "slack"],
)
def test_the_headroom_report_classifies_against_the_declared_budget(measured, expect):
    table = parse_table(
        "Abs:\n  - {in: Float16_b, out: Float16_b, max_ulp: 1}  # max 1 ULP\n"
        if expect != "could tighten"
        else "Abs:\n  - {in: Float16_b, out: Float16_b, max_ulp: 8}  # max 7 ULP\n"
    )
    report, over = render_headroom(
        table,
        _measured_cells(
            [_measure(**{"in": "Float16_b", "out": "Float16_b", "max": measured})]
        ),
    )
    assert expect in report
    assert over == (1 if expect == "over budget" else 0)


def test_a_cell_on_tolerance_is_not_judged_for_headroom():
    """There is no budget to have headroom against, and the measurement is recorded
    on the row for a human rather than checked here."""
    table = parse_table(
        "Abs:\n  - {in: Float16_b, out: Float16_b, metric: tolerance}\n"
    )
    report, over = render_headroom(
        table,
        _measured_cells(
            [_measure(**{"in": "Float16_b", "out": "Float16_b", "max": 99999})]
        ),
    )
    assert over == 0 and "over budget" not in report


def test_a_measurement_resolves_against_the_most_specific_row():
    """Same rule the registry itself uses, so the report judges a cell against the
    budget that actually gates it rather than a broader one."""
    table = parse_table(
        "Abs:\n"
        "  - {out: Float16_b, max_ulp: 100}  # broad\n"
        "  - {in: Float16_b, out: Float16_b, max_ulp: 1}  # specific\n"
    )
    _, over = render_headroom(
        table,
        _measured_cells(
            [_measure(**{"in": "Float16_b", "out": "Float16_b", "max": 50})]
        ),
    )
    assert over == 1  # 50 is inside the broad 100 but past the specific 1


def test_the_tool_imports_nothing_but_the_standard_library_and_yaml():
    """The PR check runs on a slim runner: no torch, no ttexalens, no LLK venv.

    Read off the module's own AST rather than by launching an interpreter. A subprocess
    proves it for one environment and trips the repo's command-injection scanner; the
    import list is the actual property, and checking it is exact.

    The other half is that the workflow must invoke the tool as a *file*, not
    `-m helpers.ulp_budget_diff`: `helpers/__init__.py` imports ttexalens, so a module
    invocation would drag it in whatever this module imports. That is asserted below.
    """
    import ast
    import sys
    from pathlib import Path

    tool = Path(__file__).parent / "helpers" / "ulp_budget_diff.py"
    tree = ast.parse(tool.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            # A relative import would reach back into `helpers`, which is the thing
            # this module must not do.
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
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    workflow = repo / ".github/workflows/llk-sfpu-ulp-budget-guard.yaml"
    if not workflow.exists():  # pragma: no cover - the guard ships with the workflow
        pytest.skip("workflow not in this checkout")
    # Comments stripped: the workflow explains in prose why it does *not* use `-m`,
    # and that sentence is not an invocation.
    body = "\n".join(
        line
        for line in workflow.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "helpers/ulp_budget_diff.py diff" in body
    assert "-m helpers.ulp_budget_diff" not in body


# ─────────────────────────────────────────────────────────────────────────────
# The tie-back
# ─────────────────────────────────────────────────────────────────────────────


def test_this_parse_agrees_with_the_registry_loader_on_the_live_table():
    """This module reads the table a second way -- text, no torch -- so that the CI
    check can run without the LLK environment and against an arbitrary base
    revision. Nothing stops the two parses drifting except this.

    Compared as (op, key, budget), which is exactly what a regression is defined
    over; the loader's extra fields are not this tool's business.
    """
    from helpers.sfpu_accuracy_budget import _SFPU_ACCURACY_BUDGET, _TABLE_PATH, Metric

    mine = {
        (cell[0], cell[1], row.max_ulp)
        for cell, row in parse_table(_TABLE_PATH.read_text(encoding="utf-8")).items()
    }
    theirs = set()
    for op, table in _SFPU_ACCURACY_BUDGET.items():
        for key, contract in table.items():
            pinned = tuple(
                (short, getattr(key, attr).name)
                for short, attr in (
                    ("in", "input_format"),
                    ("out", "output_format"),
                    ("approx", "approx_mode"),
                    ("dest", "dest_acc"),
                    ("arch", "arch"),
                )
                if getattr(key, attr) is not None
            )
            budget = contract.max_ulp if contract.metric == Metric.ULP else None
            theirs.add((op.name, pinned, budget))

    assert mine == theirs, (
        f"only this parser sees: {sorted(mine - theirs)[:5]}\n"
        f"only the loader sees: {sorted(theirs - mine)[:5]}"
    )


def test_a_widened_near_zero_floor_is_a_regression():
    """The floor rescues lanes the budget would fail, so widening it loosens the gate.

    `ulp_elementwise_valid` accepts a lane inside `near_zero_atol` however many steps
    out it is. A guard that watched only `max_ulp` would call this table unchanged.
    """
    base = _head(
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 1.0e-07}  # max 1 ULP"
    )
    head = _head(
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-05}  # max 1 ULP"
    )
    (change,) = compare(parse_table(base), parse_table(head))
    assert change.kind == "floor_widened"
    assert change.is_regression
    # And the report has to show it, or both columns read "2" and the row looks inert.
    text = render_budget_diff([change], "ulp-budget-raise-approved")
    assert "1e-07" in text and "5e-05" in text


def test_introducing_a_floor_where_there_was_none_is_a_regression():
    base = _head("{in: Float16_b, out: Float16_b, max_ulp: 2}  # max 1 ULP")
    head = _head(
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-05}  # max 1 ULP"
    )
    (change,) = compare(parse_table(base), parse_table(head))
    assert change.kind == "floor_widened"


def test_a_tightened_budget_with_a_wider_floor_is_still_a_regression():
    """The two can move opposite ways, and the floor can more than pay for the budget."""
    base = _head(
        "{in: Float16_b, out: Float16_b, max_ulp: 8, near_zero_atol: 1.0e-07}  # max 1 ULP"
    )
    head = _head(
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-02}  # max 1 ULP"
    )
    (change,) = compare(parse_table(base), parse_table(head))
    assert (
        change.kind == "floor_widened"
    ), "a smaller max_ulp must not mask a wider floor"
    assert change.is_regression


def test_a_narrowed_floor_is_not_a_regression():
    base = _head(
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 5.0e-05}  # max 1 ULP"
    )
    head = _head(
        "{in: Float16_b, out: Float16_b, max_ulp: 2, near_zero_atol: 1.0e-07}  # max 1 ULP"
    )
    assert [
        c for c in compare(parse_table(base), parse_table(head)) if c.is_regression
    ] == []


def test_a_floor_on_a_tolerance_row_is_not_a_gate_change():
    """`near_zero_atol` is a ULP-gate floor; on a tolerance row nothing consults it."""
    base = _head("{in: Float16_b, out: Float16_b, metric: tolerance}  # max 393 ULP")
    head = _head(
        "{in: Float16_b, out: Float16_b, metric: tolerance, near_zero_atol: 0.5}  # max 393 ULP"
    )
    assert [
        c for c in compare(parse_table(base), parse_table(head)) if c.is_regression
    ] == []


def test_a_duplicated_cell_is_refused_rather_than_judged():
    """`_load_table` rejects two rows of equal specificity, so the table cannot load.

    Keeping the last row silently produced a verdict -- a *tightening*, if the second
    row happened to be smaller -- for a table the registry would not accept at all.
    """
    duplicated = _head(
        "{in: Float16_b, out: Float16_b, max_ulp: 9}  # max 1 ULP",
        "{in: Float16_b, out: Float16_b, max_ulp: 1}  # max 1 ULP",
    )
    with _refuses("duplicate row"):
        parse_table(duplicated)

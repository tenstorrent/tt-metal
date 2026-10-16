# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for ``helpers/ulp_provenance.py``: the budget table read as rows,
and each row's provenance as fields of its own.

No code reads a row's comment, so nothing here pins one: the provenance is
``measured``, ``measured_all``, ``nonfinite`` and ``run`` / ``sampled``, which PyYAML
loads and the registry loader validates.
"""

from datetime import date

import pytest
from helpers.sfpu_accuracy_budget import _SFPU_ACCURACY_BUDGET, _TABLE_PATH
from helpers.sfpu_domains import _UNARY_OPS_NOT_SWEPT, sfpu_unary_ops
from helpers.ulp_provenance import (
    PROVENANCE_FIELDS,
    BudgetTable,
    Provenance,
    render_row,
)

_DAY = date(2026, 10, 5)


@pytest.mark.parametrize(
    "provenance",
    [
        Provenance(measured=5, run=_DAY),
        Provenance(measured=0, sampled=_DAY),
        Provenance(measured=2, measured_all=14337, run=_DAY),
        Provenance(measured=393, nonfinite=6, run=_DAY),
        Provenance(nonfinite=3, run=_DAY),
        Provenance(run=_DAY),
        Provenance(),
    ],
    ids=lambda p: ", ".join(f"{k}={v}" for k, v in p.as_fields().items()) or "none",
)
def test_a_written_row_reads_back_as_the_provenance_it_was_written_with(provenance):
    """``render_row`` writes a date unquoted, so PyYAML reads it back as a date and not
    a string the loader would refuse."""
    fields = {"in": "Float16", "out": "Float16", "max_ulp": 6, **provenance.as_fields()}
    (row,) = BudgetTable("Gelu:\n" + render_row(fields, "prose")).rows
    assert row.provenance == provenance
    assert row.comment == "prose"
    assert provenance.problems() == []


@pytest.mark.parametrize(
    "fields, problem",
    [
        (dict(measured=1), "needs the `run` or `sampled` date"),
        (dict(measured=1, run=_DAY, sampled=_DAY), "or sampled, not both"),
        (dict(measured=True, run=_DAY), "integer >= 0"),
        (dict(measured=-1, run=_DAY), "integer >= 0"),
        (dict(nonfinite=0, run=_DAY), "integer >= 1"),
        (dict(measured=1, run="2026-10-05"), "must be a date"),
        (dict(sampled=_DAY), "without the `measured` figure"),
        (dict(measured_all=3, run=_DAY), "needs `measured`"),
        (dict(measured=5, measured_all=3, run=_DAY), "cannot be below it"),
    ],
    ids=[
        "no-date",
        "run-and-sampled",
        "bool",
        "negative",
        "nonfinite-zero",
        "string-date",
        "sampled-alone",
        "measured-all-alone",
        "measured-all-below",
    ],
)
def test_a_malformed_provenance_says_what_is_wrong(fields, problem):
    """Each problem by name; the loader prefixes the row it came from."""
    problems = Provenance(**fields).problems()
    assert len(problems) == 1 and problem in problems[0], problems


def test_which_run_measured_a_row_and_whether_it_was_measurable():
    assert Provenance(measured=1, run=_DAY).exhaustive
    assert not Provenance(measured=1, sampled=_DAY).exhaustive
    assert Provenance(measured=7, nonfinite=2, run=_DAY).unmeasurable
    assert Provenance(run=_DAY).unmeasurable  # the sweep ran and ranked no lane
    assert not Provenance(measured=7, run=_DAY).unmeasurable


def test_an_alias_row_a_yaml_boolean_and_a_date_read_as_the_loader_reads_them():
    table = BudgetTable(
        "Anchor:\n"
        "  - &lut {metric: tolerance, atol: 0.13}  # the anchor\n"
        "Gelu:\n"
        "  - *lut  # through the alias\n"
        "  - {in: Float16, out: Float16, dest: Yes, max_ulp: 2, measured: 1, "
        "run: 2026-10-05}\n"
    )
    alias, keyed = table.rows_of("Gelu")
    assert alias.alias and alias.values["atol"] == 0.13
    assert alias.comment == "through the alias"
    assert keyed.pinned["dest"] == "Yes"
    assert keyed.provenance == Provenance(measured=1, run=_DAY)


def test_a_row_is_found_by_exactly_its_key():
    table = BudgetTable("Abs:\n  - {in: Float16_b, out: Float16_b, max_ulp: 0}\n")
    assert table.row("Abs", in_="Float16_b", out="Float16_b").max_ulp == 0
    with pytest.raises(LookupError):  # allow-pytest.raises: host-only test
        table.row("Abs", out="Float16_b")
    with pytest.raises(TypeError):  # allow-pytest.raises: host-only test
        table.row("Abs", input_format="Float16_b")


# ── The live table ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def live():
    return BudgetTable.load(_TABLE_PATH)


def test_the_walk_sees_every_row_the_registry_loads(live):
    """``BudgetTable`` locates rows through PyYAML's events; the registry loads them
    through PyYAML's values. Tie the two together, per op and per row."""
    for op, contracts in _SFPU_ACCURACY_BUDGET.items():
        rows = live.rows_of(op.name)
        assert len(rows) == len(contracts), op.name
        budgets = sorted(
            (c.max_ulp if c.metric.name == "ULP" else -1) for c in contracts.values()
        )
        assert sorted(r.max_ulp if r.max_ulp is not None else -1 for r in rows) == (
            budgets
        ), op.name
    assert set(live.blocks) == {op.name for op in _SFPU_ACCURACY_BUDGET}


def _swept_ops():
    """The unary ops the sweep collects: those keyed in the table, less the excused
    (``test_the_sweep_collects_every_keyed_op_whether_gating_or_emitting``)."""
    return {op.name for op in _SFPU_ACCURACY_BUDGET if op in sfpu_unary_ops()} - {
        op.name for op in _UNARY_OPS_NOT_SWEPT
    }


def test_every_row_the_sweep_drives_names_its_run(live):
    """A row of a swept op that pins ``in`` and ``out`` on this arch is one the
    emitter wrote, so it names the day the sweep measured it. A row there without a
    ``run`` is one nothing re-derives: a hand edit, or an emitter that stopped writing
    it."""
    swept = _swept_ops()
    stray = [
        row.describe()
        for row in live.rows
        if row.op in swept
        and "in" in row.pinned
        and "out" in row.pinned
        and "arch" not in row.pinned
        and row.provenance.run is None
    ]
    assert not stray, "\n".join(stray)


def test_only_the_sweep_claims_a_run(live):
    """``run`` is the exhaustive sweep's, and the audit holds such a row to exactly the
    emitter's budget. A row of an op the sweep never collects was sampled by a
    functional driver, and ``run`` on it would read a sample as exhaustive."""
    swept = _swept_ops()
    claimed = [
        row.describe()
        for row in live.rows
        if row.op not in swept and row.provenance.run is not None
    ]
    assert not claimed, "\n".join(claimed)


def test_no_key_line_carries_a_run_clause(live):
    """Since P12 a row names its own run; a `measured by:` clause on a key line would
    be a second, unread statement of it that could only drift."""
    assert not [op for op, b in live.blocks.items() if "measured by:" in b.header]


def test_the_provenance_fields_are_the_loaders():
    """The loader accepts exactly these as non-contract, non-key fields."""
    from helpers.sfpu_accuracy_budget import _CONTRACT_FIELDS, _KEY_FIELDS

    assert not set(PROVENANCE_FIELDS) & (set(_CONTRACT_FIELDS) | set(_KEY_FIELDS))

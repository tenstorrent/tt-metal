# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for ``helpers/ulp_provenance.py``: the budget table read as rows,
and each row's provenance as fields of its own.

No verdict depends on a row's comment: the provenance is ``measured``,
``measured_all``, ``nonfinite`` and ``run`` / ``sampled``, which PyYAML loads and the
registry loader validates. Where a comment is asserted, it is to check that the prose
is read back as written.
"""

from dataclasses import fields
from datetime import date, datetime

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
        (dict(nonfinite=7), "`nonfinite` needs the `run` or `sampled` date"),
        (dict(measured=1, run=_DAY, sampled=_DAY), "or sampled, not both"),
        (dict(measured=True, run=_DAY), "integer >= 0"),
        (dict(measured=-1, run=_DAY), "integer >= 0"),
        (dict(nonfinite=0, run=_DAY), "integer >= 1"),
        (dict(measured=1, run="2026-10-05"), "must be a date"),
        # An unquoted `2026-10-05 12:00:00` loads as a datetime, which is a date too.
        (dict(measured=1, run=datetime(2026, 10, 5, 12, 0)), "must be a date"),
        (dict(sampled=_DAY), "without the `measured` figure"),
        (dict(measured_all=3, run=_DAY), "needs `measured`"),
        (dict(measured=5, measured_all=3, run=_DAY), "cannot be below it"),
    ],
    ids=[
        "no-date",
        "nonfinite-no-date",
        "run-and-sampled",
        "bool",
        "negative",
        "nonfinite-zero",
        "string-date",
        "datetime",
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
    assert Provenance(measured=1, run=_DAY).swept
    assert not Provenance(measured=1, sampled=_DAY).swept
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


def test_an_op_whose_whole_row_list_is_an_alias_has_the_anchors_rows():
    """`Neg: *low` once read as an op with no rows, so every reader skipped budgets the
    registry enforces. Each is an alias row, located on the key line."""
    table = BudgetTable(
        "Abs: &low\n"
        "  - {out: Float16_b, max_ulp: 2, measured: 2, run: 2026-10-05}\n"
        "  - {out: Float32, metric: tolerance}\n"
        "Neg: *low\n"
    )
    rows = table.rows_of("Neg")
    assert [(r.alias, r.first_line, r.last_line) for r in rows] == [(True, 3, 3)] * 2
    assert [r.values for r in rows] == [r.values for r in table.rows_of("Abs")]


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
    """``run`` is the sweep's, and the audit holds such a row to exactly the emitter's
    budget. A row of an op the sweep never collects was sampled by a functional driver,
    and ``run`` on it would read a sample as swept."""
    swept = _swept_ops()
    claimed = [
        row.describe()
        for row in live.rows
        if row.op not in swept and row.provenance.run is not None
    ]
    assert not claimed, "\n".join(claimed)


def test_no_key_line_carries_a_run_clause(live):
    """A row names its own run; a `measured by:` clause on a key line would be a
    second, unread statement of it that could only drift."""
    assert not [op for op, b in live.blocks.items() if "measured by:" in b.header]


def test_the_provenance_fields_are_the_loaders(tmp_path):
    """The fields are the :class:`Provenance` dataclass's, none of them a contract or key
    field, and the loader accepts each one on a row."""
    from helpers.sfpu_accuracy_budget import _CONTRACT_FIELDS, _KEY_FIELDS, _load_table

    assert PROVENANCE_FIELDS == tuple(f.name for f in fields(Provenance))
    assert not set(PROVENANCE_FIELDS) & (set(_CONTRACT_FIELDS) | set(_KEY_FIELDS))
    path = tmp_path / "budget.yaml"
    path.write_text(
        "Abs:\n"
        "  - {out: Float16_b, max_ulp: 2, near_zero_atol: 1.0e-3, measured: 1,"
        " measured_all: 9, run: 2026-10-05}\n"
        "  - {out: Float16, metric: tolerance, measured: 3, nonfinite: 2,"
        " sampled: 2026-10-05}\n",
        encoding="utf-8",
    )
    (entries,) = _load_table(path).values()
    assert len(entries) == 2


def test_a_block_style_row_reads_its_own_fields_not_the_next_rows():
    table = BudgetTable(
        "Abs:\n"
        "  - in: Float16\n"
        "    out: Float16\n"
        "    max_ulp: 1\n"
        "    measured: 1\n"
        "    run: 2026-01-01\n"
        "  - {in: Float32, out: Float32, max_ulp: 2, measured: 2, run: 2026-01-01}\n"
    )
    first, second = table.rows_of("Abs")
    assert first.last_line == 5 and first.provenance.measured == 1
    assert second.provenance.measured == 2


def test_an_arch_is_keyed_by_its_enum_name_whichever_way_it_is_spelt():
    table = BudgetTable("Abs:\n  - {in: Float16_b, arch: wormhole, max_ulp: 1}\n")
    (row,) = table.rows_of("Abs")
    assert dict(row.key)["arch"] == "WORMHOLE"


@pytest.mark.parametrize(
    "text, match",
    [
        (
            "Abs:\n  - {max_ulp: 0}\nAbs:\n  - {max_ulp: 1}\n",
            "line 3: duplicate entry for 'Abs'",
        ),
        ("Abs:\n  - {out: Float16_b, out: Float32, max_ulp: 1}\n", "for 'out'"),
    ],
)
def test_the_table_refuses_a_repeated_key_as_the_registry_does(text, match):
    """PyYAML keeps the last of two equal keys; the guard reads the table through this
    class, and must not judge a table the registry cannot load."""
    with pytest.raises(ValueError, match=match):  # allow-pytest.raises: host-only test
        BudgetTable(text)


def test_a_merge_key_may_override_what_it_merges():
    (row,) = BudgetTable(
        "Anchor:\n  - &base {metric: tolerance, atol: 0.1}\n"
        "Abs:\n  - {<<: *base, atol: 0.2}\n"
    ).rows_of("Abs")
    assert row.values["atol"] == 0.2


def test_a_borrowed_measurement_is_refused_with_its_line():
    """On a 4,000-row table the author needs to know which `<<` row it was."""
    with pytest.raises(  # allow-pytest.raises: host-only test
        ValueError, match="line 4: `measured`, `run` reaches a row only through"
    ):
        BudgetTable(
            "Abs:\n"
            "  - &b {out: Float16_b, max_ulp: 2, measured: 2, run: 2026-10-05}\n"
            "Neg:\n"
            "  - {<<: *b, out: Float32, max_ulp: 5}\n"
        )


@pytest.mark.parametrize(
    "text",
    [
        "<<: {Abs: [{out: Float16_b, max_ulp: 2, measured: 2, run: 2026-10-05}]}\n",
        "x: &x {measured: 2, run: 2026-10-05}\n"
        "<<: {Abs: [{<<: *x, out: Float16, max_ulp: 2}]}\n",
    ],
    ids=["op-through-a-root-merge", "borrowed-inside-a-root-merge"],
)
def test_a_top_level_merge_key_is_refused(text):
    """Ops arriving through a root `<<` had no key line, so every row walk read them
    as nothing -- a raised budget there compared clean -- and the borrowed-provenance
    check never looked inside the merge value."""
    with pytest.raises(  # allow-pytest.raises: host-only test
        ValueError, match="a top-level `<<` merge key"
    ):
        BudgetTable(text)


def test_an_op_the_walk_cannot_place_is_refused():
    """Fail closed: an op the loader resolves but the walk cannot locate would read as
    no rows on both sides of a diff. A YAML boolean key is one: the walk sees `true`,
    the loader `True`."""
    with pytest.raises(  # allow-pytest.raises: host-only test
        ValueError, match=r"cannot locate op\(s\) \['True'\]"
    ):
        BudgetTable(
            "true:\n  - {out: Float16_b, max_ulp: 2, measured: 2, run: 2026-10-05}\n"
        )

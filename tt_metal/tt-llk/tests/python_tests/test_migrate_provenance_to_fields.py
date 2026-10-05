# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for ``helpers/migrate_provenance_to_fields.py``: the one-time move
of the budget table's provenance from comments into row fields.

The comment grammar it reads (P11's ``Provenance``, kept there as ``Note``) is pinned
here by its round trip, and the migration by the facts it must carry over: for every
row, what P11's readers took from the comment is what P12's take from the fields.
Deleted in P13 with the script.
"""

import itertools
from datetime import date

import pytest
from helpers.migrate_provenance_to_fields import (
    KeyLine,
    Kind,
    Note,
    RunIdentity,
    check,
    migrate,
)
from helpers.sfpu_accuracy_budget import _TABLE_PATH, _load_table
from helpers.ulp_provenance import BudgetTable, Provenance

_SWEEP = RunIdentity(
    sweep="exhaustive Float16_b/Float16/Bfp8_b/Bfp4_b + strided Float32 sweep",
    arch="wormhole",
    date="2026-10-05",
)
_SAMPLE = RunIdentity(arch="wormhole", date="2026-09-21")


def _generated():
    """Every machine-written shape, crossed with a run and with a person's tail where
    the grammar admits one. Figures at the edges: 0, a bf16 ceiling, an fp32 figure."""
    figures = (0, 6, 2**29 + 1)
    floors = (2.4e-07, 1e-03, 5.59e-07)
    cores = []
    for m in figures:
        cores += [
            Note(Kind.EMITTED, measured=m),
            Note(Kind.BLOCK, measured=m),
            Note(Kind.DEMOTED, measured=m, budget_needed=m + 1, ceiling=51),
        ]
        for floor in floors:
            cores += [
                Note(Kind.FLOORED, measured=m, floor=floor, measured_all=14337),
                Note(
                    Kind.DEMOTED,
                    measured=m,
                    budget_needed=m + 1,
                    ceiling=6,
                    floor=floor,
                    residual=100,
                ),
            ]
        cores += [
            Note(
                Kind.UNMEASURABLE,
                nonfinite_lanes=3,
                prose="disagreeing with the golden about being finite "
                "(golden -> result: x=2: 2 -> inf; x=-3e+38: -inf -> -65504)",
                measured=m,
                lanes=65021,
            ),
        ]
    cores += [
        Note(Kind.UNMEASURABLE, prose="no measurable lane"),
        Note(
            Kind.UNMEASURABLE,
            nonfinite_lanes=2,
            prose="non-finite against a finite golden (x=3e38)",
        ),
    ]
    for core, run in itertools.product(cores, (None, _SWEEP)):
        yield core if run is None else core.with_run(run)
    for m, tail in itertools.product(
        figures,
        ("", "; budget kept at the 2 the same measurement carried before"),
    ):
        yield Note(Kind.SAMPLED, measured=m, run=_SAMPLE, prose=tail)
        yield Note(Kind.EMITTED, measured=m, run=_SWEEP, prose=tail)


@pytest.mark.parametrize("provenance", list(_generated()), ids=lambda p: p.render())
def test_parse_reads_back_exactly_what_render_wrote(provenance):
    assert Note.parse(provenance.render()) == provenance


@pytest.mark.parametrize(
    "key_line",
    [KeyLine(), KeyLine("0 ULP, 115 variants / 2.09M lanes, 2026-09-18")]
    + [KeyLine(header, _SWEEP) for header in ("", "tracked: #55356 (no shift)")],
    ids=lambda k: k.render() or "empty",
)
def test_a_key_line_reads_back_exactly_what_it_wrote(key_line):
    assert KeyLine.parse(key_line.render()) == key_line


@pytest.mark.parametrize(
    "comment",
    [
        # P6's demotion wording, which P7's guard regex silently stopped matching.
        "max 5 ULP, budget would be 6 > 5-step ceiling",
        # A sample described by hand that starts like an emitted note.
        "max 384 ULP, 40 variants / 737k lanes",
        "max 65536 ULP, 5 variants / 20480 lanes, 2026-09-18",
        # A floor note missing its parenthesis, and one with a misspelt unit.
        "max 2 ULP outside a 2.40e-07 near-zero floor (14337 steps over every lane",
        "max 2 ULPs",
        "0 ULP, 16 variants",
    ],
)
def test_a_reworded_note_is_hand_written_not_half_understood(comment):
    """A machine kind is only ever what ``render`` would write character for character,
    so a reworded note cannot be read for some of its fields and not others -- which
    is how a guard went vacuous between P6 and P7. It reads as hand-written, with the
    text intact, and the live-table test below fails if a swept row is one."""
    parsed = Note.parse(comment)
    assert parsed.kind is Kind.HAND and parsed.render() == comment


def test_a_hand_note_keeps_the_figure_and_the_day_a_person_wrote():
    note = Note.parse("wh: max 4 ULP, 7 variants, 2026-10-05")
    assert (note.measured, note.recorded_max, note.names_its_run) == (4, 4, True)
    bare = Note.parse("0 ULP, 115 variants / 2.09M lanes")
    assert (bare.measured, bare.recorded_max, bare.names_its_run) == (0, None, False)


def test_which_run_measured_a_row():
    """Its own, when it names one; its key line's otherwise. Only the sweep is
    exhaustive -- a functional driver's sample is not, whatever its figure."""
    swept, sampled = KeyLine(run=_SWEEP), KeyLine("hand header")
    emitted = Note(Kind.EMITTED, measured=1)
    assert emitted.run_is_exhaustive(swept) and not emitted.run_is_exhaustive(sampled)
    assert not Note(Kind.SAMPLED, measured=0, run=_SAMPLE).run_is_exhaustive(swept)
    assert emitted.with_run(_SWEEP).run_is_exhaustive(sampled)
    dated = Note.parse("max 1 ULP, 4 variants / 16384 lanes, 2026-09-18")
    assert not dated.run_is_exhaustive(swept)


# ── The migration ─────────────────────────────────────────────────────────────

#: One of every comment shape the stack wrote, as P11 left the table.
_P11 = """\
Abs:  # 0 ULP, 16 variants / 65536 lanes, 2026-09-18; measured by: exhaustive Float16_b/Float16/Bfp8_b/Bfp4_b + strided Float32 sweep, wormhole, 2026-10-05, except where a row says otherwise
  - {in: Float16, out: Float16, max_ulp: 6}  # max 5 ULP
  - {in: Float16, out: Bfp8_b, metric: tolerance}  # max 393 ULP, block-quantized
  - {in: Float16_b, out: Float16_b, metric: tolerance}  # max 100 ULP, budget 110 > ceiling 6
  - {in: Float16, out: Float32, metric: tolerance}  # max 14337 ULP, budget 15771 > ceiling 419431; 100 ULP outside a 1.00e-03 near-zero floor
  - {in: Bfp8_b, out: Float16, max_ulp: 3, near_zero_atol: 2.40e-07}  # max 2 ULP outside a 2.40e-07 near-zero floor (14337 steps over every lane)
  - {in: Float32, out: Float16_b, metric: tolerance}  # not measurable: 3 lane(s) disagreeing with the golden about being finite (golden -> result: x=2: 2 -> inf); max 7 ULP over the 65021 measurable lanes
  - {in: Float32, out: Float32, max_ulp: 1}  # max 1 ULP, exhaustive Float16_b/Float16/Bfp8_b sweep, wormhole, 2026-09-30
  - {in: Float16_b, out: Float32, arch: BLACKHOLE, max_ulp: 44}  # bh
  - {out: Float16, max_ulp: 0}

Floor:
  - {max_ulp: 0}  # wh: 0 ULP, as Ceil, 2026-09-16
  - {out: Float32, max_ulp: 2}  # max 1 ULP, wormhole, 2026-09-21; budget kept at the 2 the same measurement carried before
  - {in: Float16_b, out: Float16_b, metric: tolerance}  # max 384 ULP, 40 variants / 737k lanes, 2026-09-18
  - {in: Float16, out: Float16_b, metric: tolerance}  # see the Bfp8_b note above, 2026-09-16
"""

_DAY = date(2026, 10, 5)


@pytest.fixture(scope="module")
def migrated():
    return migrate(_P11)


def test_every_fact_the_comments_carried_is_a_field_after(migrated):
    assert check(_P11, migrated) == []


def test_each_shape_becomes_the_fields_it_stated(migrated):
    rows = {row.describe(): row for row in BudgetTable(migrated).rows}

    def provenance(op, **key):
        return BudgetTable(migrated).row(op, **key).provenance

    assert provenance("Abs", in_="Float16", out="Float16") == Provenance(
        measured=5, run=_DAY
    )
    assert provenance("Abs", in_="Bfp8_b", out="Float16") == Provenance(
        measured=2, measured_all=14337, run=_DAY
    )
    assert provenance("Abs", in_="Float32", out="Float16_b") == Provenance(
        measured=7, nonfinite=3, run=_DAY
    )
    # A row a re-emit stamped with the run it had keeps that run, not the key line's.
    assert provenance("Abs", in_="Float32", out="Float32") == Provenance(
        measured=1, run=date(2026, 9, 30)
    )
    # A bare row of a swept op was measured by the run on its key line.
    assert provenance("Abs", out="Float16") == Provenance(measured=0, run=_DAY)
    # A functional driver's sample, and a hand-written figure with its day.
    assert provenance("Floor", out="Float32") == Provenance(
        measured=1, sampled=date(2026, 9, 21)
    )
    assert provenance("Floor") == Provenance(measured=0, sampled=date(2026, 9, 16))
    assert provenance("Floor", in_="Float16_b", out="Float16_b") == Provenance(
        measured=384, sampled=date(2026, 9, 18)
    )
    # Nothing to carry: no figure, or another arch's row the sweep never measured.
    assert provenance("Floor", in_="Float16", out="Float16_b") == Provenance()
    assert rows["Abs {in: Float16_b, out: Float32, arch: BLACKHOLE}"].comment == "bh"


def test_the_comment_keeps_only_what_no_field_holds(migrated):
    table = BudgetTable(migrated)
    assert table.row("Abs", in_="Float16", out="Float16").comment == ""
    assert table.row("Abs", in_="Float16_b", out="Float16_b").comment == ""
    assert "x=2: 2 -> inf" in table.row("Abs", in_="Float32", out="Float16_b").comment
    assert "near-zero floor" in table.row("Abs", in_="Float16", out="Float32").comment
    assert table.row("Floor", out="Float32").comment.startswith("budget kept at")
    # The key line keeps its prose and drops the run, which the rows now name.
    assert table.blocks["Abs"].header == "0 ULP, 16 variants / 65536 lanes, 2026-09-18"


def test_the_migrated_table_loads_with_its_contracts_unchanged(migrated, tmp_path):
    """The P12 loader accepts every migrated row, and only provenance fields were
    added: every key and contract field is what it was."""
    path = tmp_path / "after.yaml"
    path.write_text(migrated, encoding="utf-8")
    loaded = _load_table(path)
    before, after = BudgetTable(_P11).rows, BudgetTable(migrated).rows
    assert sum(len(rows) for rows in loaded.values()) == len(before) == len(after)
    added = set(Provenance.__dataclass_fields__)
    for was, now in zip(before, after):
        assert {k: v for k, v in now.fields if k not in added} == dict(was.fields)


def test_a_migrated_table_is_not_migrated_twice(migrated):
    with pytest.raises(ValueError):  # allow-pytest.raises: host-only test
        migrate(migrated)


def test_a_fact_that_moved_is_reported(migrated):
    """The check is what makes the migration trustworthy, so it must be able to fail:
    a figure edited on the way is named."""
    edited = migrated.replace(
        "max_ulp: 6, measured: 5, run: 2026-10-05",
        "max_ulp: 6, measured: 6, run: 2026-10-05",
    )
    assert edited != migrated
    (problem,) = check(_P11, edited)
    assert "Abs {in: Float16, out: Float16}" in problem


def test_the_live_table_is_migrated():
    """Every row with a figure carries it as a field; the loader refuses a step budget
    without one, and no key line still carries a run clause."""
    live = BudgetTable.load(_TABLE_PATH)
    budgets = [row for row in live.rows if row.max_ulp is not None]
    assert budgets and all(row.provenance.measured is not None for row in budgets)
    assert not any(KeyLine.parse(b.header).run for b in live.blocks.values())

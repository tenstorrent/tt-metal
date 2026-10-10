# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for ``helpers/ulp_provenance.py``: the one place the budget table's
comment grammar is written and read.

The grammar is pinned here once, as a round trip over every shape the emitter and the
functional drivers write, and nowhere else: every other suite compares ``Provenance``
values, so rewording a note is a change to ``render`` and this file, not to a dozen
string assertions across the stack.
"""

import itertools

import pytest
from helpers.sfpu_accuracy_budget import _SFPU_ACCURACY_BUDGET, _TABLE_PATH
from helpers.ulp_provenance import (
    EMITTER_KINDS,
    BudgetTable,
    KeyLine,
    Kind,
    Provenance,
    RunIdentity,
)

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
            Provenance(Kind.EMITTED, measured=m),
            Provenance(Kind.BLOCK, measured=m),
            Provenance(Kind.DEMOTED, measured=m, budget_needed=m + 1, ceiling=51),
        ]
        for floor in floors:
            cores += [
                Provenance(Kind.FLOORED, measured=m, floor=floor, measured_all=14337),
                Provenance(
                    Kind.DEMOTED,
                    measured=m,
                    budget_needed=m + 1,
                    ceiling=6,
                    floor=floor,
                    residual=100,
                ),
            ]
        cores += [
            Provenance(
                Kind.UNMEASURABLE,
                nonfinite_lanes=3,
                prose="disagreeing with the golden about being finite "
                "(golden -> result: x=2: 2 -> inf; x=-3e+38: -inf -> -65504)",
                measured=m,
                lanes=65021,
            ),
        ]
    cores += [
        Provenance(Kind.UNMEASURABLE, prose="no measurable lane"),
        Provenance(
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
        yield Provenance(Kind.SAMPLED, measured=m, run=_SAMPLE, prose=tail)
        yield Provenance(Kind.EMITTED, measured=m, run=_SWEEP, prose=tail)


@pytest.mark.parametrize("provenance", list(_generated()), ids=lambda p: p.render())
def test_parse_reads_back_exactly_what_render_wrote(provenance):
    assert Provenance.parse(provenance.render()) == provenance


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
    parsed = Provenance.parse(comment)
    assert parsed.kind is Kind.HAND and parsed.render() == comment


def test_a_hand_note_keeps_the_figure_and_the_day_a_person_wrote():
    note = Provenance.parse("wh: max 4 ULP, 7 variants, 2026-10-05")
    assert (note.measured, note.recorded_max, note.names_its_run) == (4, 4, True)
    bare = Provenance.parse("0 ULP, 115 variants / 2.09M lanes")
    assert (bare.measured, bare.recorded_max, bare.names_its_run) == (0, None, False)


def test_which_run_measured_a_row():
    """Its own, when it names one; its key line's otherwise. Only the sweep is
    exhaustive -- a functional driver's sample is not, whatever its figure."""
    swept, sampled = KeyLine(run=_SWEEP), KeyLine("hand header")
    emitted = Provenance(Kind.EMITTED, measured=1)
    assert emitted.run_is_exhaustive(swept) and not emitted.run_is_exhaustive(sampled)
    assert not Provenance(Kind.SAMPLED, measured=0, run=_SAMPLE).run_is_exhaustive(
        swept
    )
    assert emitted.with_run(_SWEEP).run_is_exhaustive(sampled)
    dated = Provenance.parse("max 1 ULP, 4 variants / 16384 lanes, 2026-09-18")
    assert not dated.run_is_exhaustive(swept)


# ── The live table ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def live():
    return BudgetTable.load(_TABLE_PATH)


def test_every_comment_in_the_live_table_reads_back_unchanged(live):
    """Parse is lossless on the real table, row and key line alike: nothing a person
    wrote is dropped by reading it."""
    for row in live.rows:
        last = live.lines[row.last_line]
        _, sep, comment = last[row.end_column :].partition("#")
        rendered = row.provenance.render() if row.provenance else ""
        assert rendered == (comment.strip() if sep else ""), row.describe()
    for op, block in live.blocks.items():
        _, sep, comment = live.lines[block.line].partition("#")
        assert block.key_line.render() == (comment.strip() if sep else ""), op


def test_every_swept_row_carries_a_machine_written_note(live):
    """In a block a sweep wrote (its key line names a run), a row pinning ``in`` and
    ``out`` on this arch is one the emitter rendered. A hand-written or missing note
    there is a row nothing re-derives: an edit that reworded or dropped it, or an
    emitter grammar change ``parse`` was not taught."""
    stray = [
        f"{row.describe()}  # {row.provenance.render() if row.provenance else ''}"
        for row in live.rows
        if live.blocks[row.op].key_line.run is not None
        and "in" in row.pinned
        and "out" in row.pinned
        and "arch" not in row.pinned
        and (row.provenance is None or row.provenance.kind not in EMITTER_KINDS)
    ]
    assert not stray, "\n".join(stray)


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


def test_a_row_is_found_by_exactly_its_key(live):
    with pytest.raises(LookupError):  # allow-pytest.raises: host-only test
        live.row("Abs", in_="NotAFormat", out="Float16_b")
    with pytest.raises(TypeError):  # allow-pytest.raises: host-only test
        live.row("Abs", input_format="Float16_b")


def test_an_alias_row_and_a_yaml_boolean_read_as_the_loader_reads_them():
    table = BudgetTable(
        "Anchor:\n"
        "  - &lut {metric: tolerance, atol: 0.13}  # the anchor\n"
        "Gelu:\n"
        "  - *lut  # through the alias\n"
        "  - {in: Float16, out: Float16, dest: Yes, max_ulp: 2}  # max 1 ULP\n"
    )
    alias, keyed = table.rows_of("Gelu")
    assert alias.alias and alias.values["atol"] == 0.13
    assert alias.provenance == Provenance.hand("through the alias")
    assert keyed.pinned["dest"] == "Yes"
    assert keyed.provenance == Provenance(Kind.EMITTED, measured=1)

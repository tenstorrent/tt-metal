# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whether a change to the SFPU accuracy table loosens a gate, and whether the hardware
still fits the gates it declares. ``diff`` resolves every variant the two tables can
tell apart by the registry's own rules: a figure that goes up, a cell that stops being
gated, or a tolerance cell whose recorded measurement or declared bound loosens is a
regression. ``headroom`` judges the ``--ulp-measure`` records against the table: the
tolerance cells the sweep cannot fail, and the cells that now measure better than their
row. Run from ``python_tests`` as ``python -m helpers.ulp_budget_diff <command>``."""

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Iterable

from .llk_params import MathOperation
from .ulp_domains import (
    Domain,
    TableError,
    binade_edges,
    entries_of,
    largest_figure,
    partition,
    usable_budget_ceiling,
)
from .ulp_table import (
    MEASURED_ARCH,
    BudgetKey,
    Measurement,
    Metric,
    Row,
    Table,
    _winner,
    load_table,
    most_specific,
    read_measurements,
    steps_bind,
)

# The op and the key of the row that decides a cell.
Cell = tuple[MathOperation, BudgetKey]

# A collapsed section shows this many rows and says how many it withheld; a list of
# what failed or loosened is never cut.
_MAX_ROWS = 40

_INF = float("inf")


def _resolve(
    table: Table, op: MathOperation, query: BudgetKey, only=lambda row: True
) -> Row | None:
    """The row of *op* the registry would resolve *query* to, among those *only*
    admits."""
    found = most_specific(
        {row.key: row for row in table.get(op, ()) if only(row)}, query
    )
    return None if found is None else found[1]


# ── What a pull request does to the gates ─────────────────────────────────────


class Kind(Enum):
    """How a variant's gate changed, in report order; the value is the report's phrase."""

    RAISED = "budget raised"
    FLOOR_WIDENED = "near-zero floor widened"
    UNGATED = "gating lost (now tolerance)"
    REMOVED = "gate lost (no row covers it)"
    TOLERANCE_WIDENED = "tolerance bound widened"
    BASELINE_RAISED = "recorded measurement raised"
    BASELINE_DROPPED = "recorded measurement dropped"
    TIGHTENED = "budget tightened"
    GATED = "newly gated"
    ADDED = "row added"

    @property
    def loosens(self) -> bool:
        return self in _LOOSENS


_LOOSENS = frozenset(
    {
        Kind.RAISED,
        Kind.FLOOR_WIDENED,
        Kind.UNGATED,
        Kind.REMOVED,
        Kind.TOLERANCE_WIDENED,
        Kind.BASELINE_RAISED,
        Kind.BASELINE_DROPPED,
    }
)
_REPORT_ORDER = list(Kind)


class State(Enum):
    """What one segment of the input is held to: a gated figure, a figure past the
    ceiling the headroom report holds, an entry no lane is ranked in, or nothing."""

    GATED = "gated"
    PAST = "past the ceiling"
    NULL = "unranked"
    NONE = "none"


@dataclass(frozen=True)
class Span:
    low: float
    high: float
    state: State
    figure: int | None = None


def _held_text(state: State, figure: int | None, row: Row | None) -> str:
    """What a span is held to, for the report."""
    if state is State.GATED:
        floor = row.near_zero_atol if row is not None else None
        return f"{figure} (floor {floor:g})" if floor else str(figure)
    if state is State.PAST:
        return f"tolerance (max {figure})"
    return "unranked" if state is State.NULL else "—"


@dataclass(frozen=True)
class Held:
    """What one query variant is held to in one revision of the table: the row it
    resolves to, whether that row's step budget binds, the most specific row declaring
    ``atol``/``rtol``, and whether the variant is a full measured cell (every dimension
    some row pins is set)."""

    row: Row | None
    gated: bool
    declared: Row | None
    full: bool

    @property
    def baseline(self) -> int | None:
        """The measurement a tolerance row records for a full cell; only a row pinning
        both formats is one, as the emitter writes every row."""
        row = self.row
        if row is None or self.gated or not self.full:
            return None
        return (
            row.measured if row.input is not None and row.output is not None else None
        )

    @property
    def nonfinite(self) -> int:
        if self.row is None or self.gated or not self.full:
            return 0
        return self.row.nonfinite or 0

    def describe(self) -> str:
        if self.row is None:
            return "—"
        if self.gated:
            return _held_text(State.GATED, self.row.contract.max_ulp, self.row)
        parts = []
        if self.baseline is not None:
            parts.append(f"max {self.baseline}")
        if self.nonfinite:
            parts.append(f"{self.nonfinite} non-finite")
        if self.declared is not None:
            parts += [
                f"{label} {value:g}"
                for label, value in (
                    ("atol", self.declared.atol),
                    ("rtol", self.declared.rtol),
                )
                if value is not None
            ]
        return "tolerance" + (f" ({', '.join(parts)})" if parts else "")


def _held(table: Table, op: MathOperation, query: BudgetKey, full: bool) -> Held:
    row = _resolve(table, op, query)
    gated = (
        row is not None
        and row.contract.metric is Metric.ULP
        and steps_bind(row.key, query)
    )
    declared = (
        None
        if gated
        else _resolve(table, op, query, only=lambda r: r.contract.declares_tolerance)
    )
    return Held(row, gated, declared, full)


def _variants(
    base: Table, head: Table, op: MathOperation
) -> Iterable[tuple[BudgetKey, bool]]:
    """Every query the registry could be asked about *op*, as far as either table can
    tell them apart (each dimension takes every value some row pins, or is left unset; a
    pinned two-valued flag takes both values; the arch always includes
    :data:`MEASURED_ARCH`), and whether it is a full measured cell: every dimension some
    row pins is set."""
    names = [f.name for f in fields(BudgetKey)]
    keys = [row.key for row in (*base.get(op, ()), *head.get(op, ()))]
    values = {name: {getattr(key, name) for key in keys} - {None} for name in names}
    pinned = [name for name in names if values[name]]
    for flag in ("approx_mode", "dest_acc"):
        if values[flag]:
            values[flag] |= set(type(next(iter(values[flag]))))
    values["arch"].add(MEASURED_ARCH)
    axes = [
        sorted(values[name], key=lambda v: v.name) + ([] if name == "arch" else [None])
        for name in names
    ]
    for combo in itertools.product(*axes):
        query = BudgetKey(**dict(zip(names, combo)))
        yield query, all(getattr(query, name) is not None for name in pinned)


def _widened(was: Row | None, now: Row | None) -> bool:
    """Whether a tolerance compare may have got looser: a bound went up, went away, or
    appeared."""
    if was is None:
        return now is not None
    if now is None:
        return True
    return any(
        (old is None) != (new is None) or (old is not None and new > old)
        for old, new in ((was.atol, now.atol), (was.rtol, now.rtol))
    )


def _classify(was: Held, now: Held) -> Kind | None:
    """How the gate a plain-row variant resolves to has changed, or ``None``."""
    if was.gated and not now.gated:
        return Kind.REMOVED if now.row is None else Kind.UNGATED
    if not was.gated and now.gated:
        return Kind.GATED
    if was.gated and now.gated:
        before, after = was.row.contract.max_ulp, now.row.contract.max_ulp
        if after > before:
            return Kind.RAISED
        if (now.row.near_zero_atol or 0) > (was.row.near_zero_atol or 0):
            return Kind.FLOOR_WIDENED
        return Kind.TIGHTENED if after < before else None
    if _widened(was.declared, now.declared):
        return Kind.TOLERANCE_WIDENED
    if (was.baseline is not None and now.baseline is None) or (
        was.nonfinite and now.row is None
    ):
        return Kind.BASELINE_DROPPED
    if (
        was.baseline is not None
        and now.baseline > was.baseline
        or now.nonfinite > was.nonfinite
    ):
        return Kind.BASELINE_RAISED
    return Kind.ADDED if was.row is None and now.row is not None else None


def _spans(held: Held, query: BudgetKey) -> list[Span]:
    """*held*'s row over ``(-inf, inf)``: its list's entries, or one span for a plain
    row."""
    row = held.row
    if row is None:
        return [Span(-_INF, _INF, State.NONE)]
    if row.domains:
        binds, ceiling = steps_bind(row.key, query), usable_budget_ceiling(row.output)
        spans = []
        for d in row.domains:
            if d.ulp is None:
                spans.append(Span(d.low, d.high, State.NULL))
            elif binds and d.ulp <= ceiling:
                spans.append(Span(d.low, d.high, State.GATED, d.ulp))
            elif held.full:
                spans.append(Span(d.low, d.high, State.PAST, d.ulp))
            else:
                spans.append(Span(d.low, d.high, State.NONE))
        return spans
    if held.gated:
        return [Span(-_INF, _INF, State.GATED, row.contract.max_ulp)]
    figure = held.baseline
    state = State.PAST if figure is not None else State.NONE
    return [Span(-_INF, _INF, state, figure)]


def _span_at(spans: list[Span], x: float) -> Span:
    return next(
        (s for s in spans if s.low <= x < s.high), Span(-_INF, _INF, State.NONE)
    )


def _classify_segment(was: Span, now: Span) -> Kind | None:
    before, after = was.state, now.state
    if before is State.GATED:
        if after is State.GATED:
            if now.figure > was.figure:
                return Kind.RAISED
            return Kind.TIGHTENED if now.figure < was.figure else None
        return Kind.REMOVED if after is State.NONE else Kind.UNGATED
    if before is State.PAST:
        if after is State.GATED:
            return Kind.GATED
        if after is State.PAST:
            return Kind.BASELINE_RAISED if now.figure > was.figure else None
        return Kind.BASELINE_DROPPED
    if after is State.GATED:
        return Kind.GATED
    return Kind.ADDED if after is State.PAST else None


def _inside(low: float, high: float) -> float:
    """A point strictly inside ``[low, high)``, with infinite ends; an offset of 1 from
    a finite edge past 2**53 rounds back onto it, so step by its magnitude."""
    if (low, high) == (-_INF, _INF):
        return 0.0
    if high == _INF:
        return low + max(1.0, abs(low))
    if low == -_INF:
        return high - max(1.0, abs(high))
    return (low + high) / 2


@dataclass(frozen=True)
class Change:
    """One resolved cell whose gate differs between two revisions of the table: *cell*
    names the row that decides it after the change (before it, for a cell nothing
    covers any more), *variants* how many query variants that row decides differently,
    *segment* the ``[low, high)`` input range a list's change is about."""

    cell: Cell
    kind: Kind
    held_before: str
    held_after: str
    variants: int = 1
    segment: tuple[float, float] | None = None


def _segment_changes(
    was: Held, now: Held, query: BudgetKey
) -> list[tuple[Kind, tuple[float, float], str, str]]:
    """``(kind, segment, held_before, held_after)`` per run of the input over which the
    two rows' gates differ, over the union of both lists' edges. On a segment gated on
    both sides a widened floor ranks after a raise and before a tightening."""
    before, after = _spans(was, query), _spans(now, query)
    edges = sorted(
        {e for spans in (before, after) for s in spans for e in (s.low, s.high)}
    )
    floor_widened = (
        was.row is not None
        and now.row is not None
        and (now.row.near_zero_atol or 0) > (was.row.near_zero_atol or 0)
    )
    found: list[list] = []
    for low, high in zip(edges, edges[1:]):
        a, b = _span_at(before, _inside(low, high)), _span_at(after, _inside(low, high))
        kind = _classify_segment(a, b)
        if (
            floor_widened
            and a.state is not State.NONE
            and b.state is not State.NONE
            and kind in (None, Kind.TIGHTENED, Kind.GATED)
        ):
            kind = Kind.FLOOR_WIDENED
        if kind is None:
            continue
        held = (
            kind,
            _held_text(a.state, a.figure, was.row),
            _held_text(b.state, b.figure, now.row),
        )
        if found and found[-1][0] == held and found[-1][2] == low:
            found[-1][2] = high
        else:
            found.append([held, low, high])
    changes = [(kind, (low, high), t0, t1) for (kind, t0, t1), low, high in found]
    if not was.gated and not now.gated and _widened(was.declared, now.declared):
        # Off a whole-cell gate the functional drivers compare on the declared bounds.
        changes.append((Kind.TOLERANCE_WIDENED, None, was.describe(), now.describe()))
    if now.nonfinite > was.nonfinite:
        # Lanes the headroom report excuses are a cell-wide figure, not a segment's.
        changes.append((Kind.BASELINE_RAISED, None, was.describe(), now.describe()))
    return changes


def compare(base: Table, head: Table) -> list[Change]:
    """Every cell whose *resolved* gate differs, strongest change first; where either
    side's row carries a list, segment by segment."""
    grouped: dict[tuple, Change] = {}
    for op in sorted(set(base) | set(head), key=lambda o: o.name):
        for query, full in _variants(base, head, op):
            was, now = _held(base, op, query, full), _held(head, op, query, full)
            deciding = now.row if now.row is not None else was.row
            if deciding is None:
                continue
            if (was.row is not None and was.row.domains) or (
                now.row is not None and now.row.domains
            ):
                found = _segment_changes(was, now, query)
            else:
                kind = _classify(was, now)
                found = (
                    []
                    if kind is None
                    else [(kind, None, was.describe(), now.describe())]
                )
            for kind, segment, before, after in found:
                ident = (op, kind, deciding.key, segment, before, after)
                prior = grouped.get(ident)
                grouped[ident] = Change(
                    (op, deciding.key),
                    kind,
                    before,
                    after,
                    variants=prior.variants + 1 if prior else 1,
                    segment=segment,
                )
    return sorted(
        grouped.values(),
        key=lambda c: (
            _REPORT_ORDER.index(c.kind),
            c.cell[0].name,
            c.cell[1].sort_key,
            c.segment or (),
        ),
    )


def _where(cell: Cell, segment=None, variants: int = 1) -> str:
    op, key = cell
    text = f"{op.name} {key.describe()}"
    if variants > 1:
        text += f" ×{variants}"
    if segment is not None:
        text += f" x in [{segment[0]:g}, {segment[1]:g})"
    return text


def _section(
    heading: str, header: tuple[str, ...], lines: list[str], *, collapsed: bool
) -> list[str]:
    """A report section: a bold heading over a table, or a collapsed one cut at
    :data:`_MAX_ROWS`; nothing for no lines."""
    if not lines:
        return []
    withheld = len(lines) - _MAX_ROWS
    if collapsed and withheld > 0:
        lines = lines[:_MAX_ROWS] + [
            f"| _… {withheld} more_ |" + " |" * (len(header) - 1)
        ]
    table = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header), *lines, ""]
    if collapsed:
        return [f"<details><summary>{heading}</summary>", "", *table, "</details>", ""]
    return [f"**{heading}**", "", *table]


def render_budget_diff(changes: list[Change]) -> str:
    """The diff report: what loosens first, the rest collapsed."""
    if not changes:
        return "### SFPU ULP budgets\n\nNo budget changed.\n"
    header = ("cell", "change", "before", "after")

    def lines(selected):
        return [
            f"| `{_where(c.cell, c.segment, c.variants)}` | {c.kind.value} "
            f"| {c.held_before} | {c.held_after} |"
            for c in selected
        ]

    loosened = [c for c in changes if c.kind.loosens]
    other = [c for c in changes if not c.kind.loosens]
    out = [
        "### SFPU ULP budgets",
        "",
        *_section(
            f"{len(loosened)} cell(s) loosen a gate",
            header,
            lines(loosened),
            collapsed=False,
        ),
        *_section(
            f"{len(other)} other budget change(s)", header, lines(other), collapsed=True
        ),
    ]
    return "\n".join(out) + "\n"


# ── What the hardware measured against what the table declares ────────────────


@dataclass
class CellReading:
    """Every record of one cell folded together: the worst of each figure, whether any
    lane was ranked, its binade profile, and per entry of its list the worst lane and
    the most ranked lanes."""

    worst: int = 0
    nonfinite: int = 0
    ranked: bool = False
    bins: tuple[tuple[int, int, int, int], ...] = ()
    entries: dict[tuple[float, float], tuple[int | None, int]] = field(
        default_factory=dict
    )

    def fold(self, m: Measurement) -> None:
        self.worst = max(self.worst, m.worst)
        self.nonfinite = max(self.nonfinite, m.nonfinite)
        self.ranked = self.ranked or m.lanes > 0
        self.bins = self.bins or m.bins
        for r in m.readings:
            worst, lanes = self.entries.get((r.low, r.high), (None, 0))
            if r.worst is not None:
                worst = r.worst if worst is None else max(worst, r.worst)
            self.entries[(r.low, r.high)] = (worst, max(lanes, r.lanes))


def read_cells(measurements: Iterable[Measurement]) -> dict[Cell, CellReading]:
    cells: dict[Cell, CellReading] = {}
    for m in measurements:
        cells.setdefault((m.op, m.key), CellReading()).fold(m)
    return cells


class Verdict(Enum):
    """What the headroom check found in one cell, or one entry of its list; the value is
    the report's phrase."""

    REGRESSED = "regressed"
    NONFINITE = "non-finite lanes"
    NO_LANE = "no lane measurable"
    TIGHTENS = "re-emit tightens"

    @property
    def fails(self) -> bool:
        return self is not Verdict.TIGHTENS


@dataclass(frozen=True)
class Finding:
    cell: Cell
    verdict: Verdict
    seen: int | None
    reference: int
    entry: Domain | None = None


@dataclass
class Headroom:
    """What the records say against the table; nothing measured is not a pass."""

    measured: int = 0
    findings: list[Finding] = field(default_factory=list)
    unjudged: int = 0
    uncovered: int = 0

    @property
    def regressions(self) -> int:
        return sum(f.verdict.fails for f in self.findings) if self.measured else 1


def judge(table: Table, measurements: Iterable[Measurement]) -> Headroom:
    """Every recorded cell against the row it resolves to: a gated cell over its figure
    or a tolerance cell over its recorded one has regressed, and a better reading is a
    re-emit's to tighten. A list is judged entry by entry, a null entry's ranked strays
    held to the row's largest figure. A cell that ranked no lane against a figure has
    regressed too."""
    cells = read_cells(measurements)
    out = Headroom(measured=len(cells))
    for cell, reading in sorted(
        cells.items(), key=lambda item: (item[0][0].name, item[0][1].sort_key)
    ):
        op, key = cell
        row = _resolve(table, op, key)
        if row is None:
            out.uncovered += 1
            continue
        found = out.findings.append
        if reading.nonfinite > (row.nonfinite or 0):
            found(
                Finding(cell, Verdict.NONFINITE, reading.nonfinite, row.nonfinite or 0)
            )
        if row.domains:
            largest = largest_figure(row.domains)
            if not reading.ranked:
                found(Finding(cell, Verdict.NO_LANE, None, largest))
                continue
            for entry in row.domains:
                got = reading.entries.get((entry.low, entry.high))
                if got is None:
                    if reading.bins or reading.entries:
                        # A sweep record: drawn over another partition (the table
                        # changed after the run), so this entry was never read.
                        found(Finding(cell, Verdict.NOT_READ, None, entry.ulp, entry))
                    else:
                        out.unjudged += 1  # a functional driver's: no readings at all
                    continue
                seen, lanes = got
                figure = largest if entry.ulp is None else entry.ulp
                if entry.ulp is not None and lanes == 0:
                    found(Finding(cell, Verdict.NO_LANE, None, entry.ulp, entry))
                elif seen is not None and seen > figure:
                    found(Finding(cell, Verdict.REGRESSED, seen, figure, entry))
                elif entry.ulp is not None and seen < entry.ulp:
                    found(Finding(cell, Verdict.TIGHTENS, seen, entry.ulp, entry))
            continue
        if row.contract.metric is Metric.ULP and steps_bind(row.key, key):
            budget = row.contract.max_ulp
            if reading.worst != budget:
                verdict = (
                    Verdict.REGRESSED if reading.worst > budget else Verdict.TIGHTENS
                )
                found(Finding(cell, verdict, reading.worst, budget))
            continue
        was = row.measured if row.input is not None and row.output is not None else None
        if was is None:
            out.unjudged += 1
        elif not reading.ranked:
            found(Finding(cell, Verdict.NO_LANE, None, was))
        elif reading.worst > was:
            found(Finding(cell, Verdict.REGRESSED, reading.worst, was))
    return out


def _finding_line(f: Finding) -> str:
    segment = None if f.entry is None else (f.entry.low, f.entry.high)
    seen = 0 if f.seen is None else f.seen
    return (
        f"| `{_where(f.cell, segment)}` | {seen} | {f.reference} | {f.verdict.value} |"
    )


def render_headroom(headroom: Headroom) -> str:
    out = ["### SFPU ULP sweep vs the declared budgets", ""]
    if not headroom.measured:
        out.append(
            "No measurements recorded: nothing was compared, so this is not a pass."
        )
        return "\n".join(out) + "\n"
    out += [f"{headroom.measured} cell(s) measured.", ""]
    if headroom.unjudged or headroom.uncovered:
        out += [
            f"{headroom.unjudged + headroom.uncovered} reading(s) not judged: "
            f"{headroom.unjudged} with no recorded figure, {headroom.uncovered} covered "
            "by no row at all.",
            "",
        ]
    header = ("cell", "measured", "recorded", "")
    failed = [_finding_line(f) for f in headroom.findings if f.verdict.fails]
    better = [_finding_line(f) for f in headroom.findings if not f.verdict.fails]
    out += _section(
        f"Past the row's figure — {len(failed)}", header, failed, collapsed=False
    )
    out += _section(
        f"Measured better than the row's figure (a re-emit tightens these) — "
        f"{len(better)}",
        header,
        better,
        collapsed=True,
    )
    if not headroom.findings:
        out += [
            "Every gated cell measures exactly its figure, and no judged tolerance cell "
            "is past its recorded measurement.",
            "",
        ]
    return "\n".join(out) + "\n"


def draw(reading: CellReading, query: BudgetKey) -> tuple[Domain, ...]:
    """The list the emitter's rule draws on *reading*'s binade profile: a binade lies
    wholly inside one entry, so the worst of an entry's binades is the figure the
    emitter measures over its lanes."""
    profile = {
        b: (present, ranked, worst) for b, present, ranked, worst in reading.bins
    }

    def worst_in(low: float, high: float) -> int | None:
        found = [
            w
            for b, (_, r, w) in profile.items()
            if r and low <= binade_edges(b)[0] < high
        ]
        return max(found) if found else None

    ceiling = usable_budget_ceiling(query.output_format)
    return entries_of(partition(profile, ceiling), worst_in)


# ── Command line ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    diff = sub.add_parser("diff", help="what a change to the table does to the gates")
    diff.add_argument("--base", type=Path, required=True)
    diff.add_argument("--head", type=Path, required=True)
    diff.add_argument("--allow-raises", action="store_true", help="report, do not fail")
    headroom = sub.add_parser(
        "headroom", help="--ulp-measure records against the table"
    )
    headroom.add_argument("--table", type=Path, required=True)
    headroom.add_argument("--measured", type=Path, required=True)
    for command in (diff, headroom):
        command.add_argument("--out", type=Path, help="also write the report here")
    args = parser.parse_args(argv)
    try:
        if args.command == "diff":
            changes = compare(load_table(args.base), load_table(args.head))
            report = render_budget_diff(changes)
            loosened = any(c.kind.loosens for c in changes)
            status = 1 if loosened and not args.allow_raises else 0
        else:
            headroom = judge(load_table(args.table), read_measurements(args.measured))
            report = render_headroom(headroom)
            status = 1 if headroom.regressions else 0
    except (TableError, json.JSONDecodeError, OSError) as refused:
        report, status = f"cannot load: {refused}\n", 2
    print(report, end="")
    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
    return status


if __name__ == "__main__":
    sys.exit(main())

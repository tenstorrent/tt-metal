# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whether a change to the SFPU accuracy table loosens a gate, and whether the
hardware still fits the gates it declares.

* ``diff``: what a pull request does to the budgets. Every variant the two tables can
  tell apart is resolved in both, by the registry's own rules, and a figure that goes
  up, a cell that stops being gated, or a tolerance cell whose recorded measurement or
  declared ``atol``/``rtol`` loosens is a *regression*.
* ``headroom``: the ``--ulp-measure`` records against the table. The sweep fails a
  gated cell over its figure itself; this judges the tolerance cells the sweep cannot,
  and lists the cells that now measure better than their row.
* ``domains``: the list the emitter's rule would draw on each recorded profile.

Run from ``python_tests`` as ``python -m helpers.ulp_budget_diff <command>``.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .llk_params import MathOperation
from .sfpu_accuracy_budget import MEASURED_ARCH, _winner, steps_bind
from .ulp import ULP_FORMATS
from .ulp_domains import (
    DOMAIN_MIN_BIN_LANES,
    Domain,
    binade_edges,
    entries_of,
    partition,
    usable_budget_ceiling,
)
from .ulp_table import (
    BudgetKey,
    Measurement,
    Metric,
    Row,
    Table,
    load_table,
    read_measurements,
)

#: A cell identity: the op and the key of the row that decides it.
Cell = Tuple[MathOperation, BudgetKey]

#: An informational section shows this many rows and says how many it withheld. A list
#: of what failed or loosened is never cut.
_MAX_ROWS = 40

_INF = float("inf")


def _name(op: MathOperation, key: BudgetKey) -> str:
    return f"{op.name} {key.describe()}"


def _ceiling(query: BudgetKey) -> float:
    fmt = query.output_format
    return usable_budget_ceiling(fmt) if fmt in ULP_FORMATS else _INF


def _resolve(
    table: Table, op: MathOperation, query: BudgetKey, only=lambda row: True
) -> Optional[Row]:
    """The row of *op* the registry would resolve *query* to, among those *only*
    admits."""
    found = _winner({row.key: row for row in table.get(op, ()) if only(row)}, query)
    return None if found is None else found[1]


def _recorded(row: Row) -> Optional[int]:
    """The measurement a row records as a baseline: only a row pinning both formats is
    one, as the emitter writes every row."""
    if row.input is None or row.output is None:
        return None
    return row.measured


# ─────────────────────────────────────────────────────────────────────────────
# What a pull request does to the gates
# ─────────────────────────────────────────────────────────────────────────────


class Kind(Enum):
    """How a variant's gate changed, in report order."""

    RAISED = auto()
    FLOOR_WIDENED = auto()
    UNGATED = auto()
    REMOVED = auto()
    TOLERANCE_WIDENED = auto()
    BASELINE_RAISED = auto()
    BASELINE_DROPPED = auto()
    TIGHTENED = auto()
    GATED = auto()
    ADDED = auto()

    @property
    def loosens(self) -> bool:
        return self.value <= Kind.BASELINE_DROPPED.value


_KIND_TEXT = {
    Kind.RAISED: "budget raised",
    Kind.FLOOR_WIDENED: "near-zero floor widened",
    Kind.UNGATED: "gating lost (now tolerance)",
    Kind.REMOVED: "gate lost (no row covers it)",
    Kind.TOLERANCE_WIDENED: "tolerance bound widened",
    Kind.BASELINE_RAISED: "recorded measurement raised",
    Kind.BASELINE_DROPPED: "recorded measurement dropped",
    Kind.TIGHTENED: "budget tightened",
    Kind.GATED: "newly gated",
    Kind.ADDED: "row added",
}

#: The key dimensions every measured cell names (``arch`` aside, which every variant
#: names): the shape of a cell the headroom report can judge.
_MEASURED_KEY = ("approx_mode", "input_format", "output_format", "dest_acc")


@dataclass(frozen=True)
class Held:
    """What one query variant is held to in one revision of the table: the row it
    resolves to, whether that row's step budget binds, the most specific row declaring
    ``atol``/``rtol``, and whether the variant is a measured cell's shape."""

    row: Optional[Row]
    gated: bool
    declared: Optional[Row]
    measured_shape: bool

    @property
    def _tolerance_row(self) -> Optional[Row]:
        if not self.measured_shape or self.row is None or self.gated:
            return None
        return self.row

    @property
    def baseline(self) -> Optional[int]:
        row = self._tolerance_row
        return None if row is None else _recorded(row)

    @property
    def nonfinite(self) -> int:
        row = self._tolerance_row
        return 0 if row is None else row.nonfinite or 0

    def describe(self) -> str:
        if self.row is None:
            return "—"
        if self.gated:
            text = str(self.row.contract.max_ulp)
            if self.row.near_zero_atol:
                text += f" (floor {self.row.near_zero_atol:g})"
            return text
        parts = []
        if self.baseline is not None:
            parts.append(f"max {self.baseline}")
        if self.nonfinite:
            parts.append(f"{self.nonfinite} non-finite")
        if self.declared is not None:
            for label, value in (
                ("atol", self.declared.atol),
                ("rtol", self.declared.rtol),
            ):
                if value is not None:
                    parts.append(f"{label} {value:g}")
        return "tolerance" + (f" ({', '.join(parts)})" if parts else "")


def _held(table: Table, op: MathOperation, query: BudgetKey, open_dims) -> Held:
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
    shape = all(
        getattr(query, name) is not None or name in open_dims for name in _MEASURED_KEY
    )
    return Held(row, gated, declared, shape)


def _variants(base: Table, head: Table, op: MathOperation) -> Iterable[BudgetKey]:
    """Every query the registry could be asked about *op*, as far as either table can
    tell them apart: each dimension takes every value some row pins, or is left unset.
    The arch always includes :data:`MEASURED_ARCH`; a pinned two-valued flag takes both
    of its values, since pinning one splits the other off a broader row."""
    names = [f.name for f in fields(BudgetKey)]
    keys = [row.key for row in (*base.get(op, ()), *head.get(op, ()))]
    values = {name: {getattr(key, name) for key in keys} - {None} for name in names}
    for flag in ("approx_mode", "dest_acc"):
        if values[flag]:
            values[flag] |= set(type(next(iter(values[flag]))))
    values["arch"].add(MEASURED_ARCH)
    axes = [
        sorted(values[name], key=lambda v: v.name) + ([] if name == "arch" else [None])
        for name in names
    ]
    for combo in itertools.product(*axes):
        yield BudgetKey(**dict(zip(names, combo)))


def _widened(was: Optional[Row], now: Optional[Row]) -> bool:
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


def _classify(was: Held, now: Held) -> Optional[Kind]:
    """How the gate one variant resolves to has changed, or ``None``."""
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


class State(Enum):
    """What one segment of the input is held to: a gated figure, a figure past the
    ceiling the headroom report holds, an entry no lane is ranked in, or nothing."""

    GATED = auto()
    PAST = auto()
    NULL = auto()
    NONE = auto()


@dataclass(frozen=True)
class Span:
    low: float
    high: float
    state: State
    figure: Optional[int] = None


def _spans(held: Held, query: BudgetKey) -> List[Span]:
    """*held*'s row over ``(-inf, inf)``: its list's entries, or one span for a plain
    row. A recorded figure is a baseline only for a variant of a measured cell's shape,
    as on the plain path."""
    row = held.row
    if row is None:
        return [Span(-_INF, _INF, State.NONE)]
    if row.domains:
        binds, ceiling = steps_bind(row.key, query), _ceiling(query)
        spans = []
        for d in row.domains:
            if d.ulp is None:
                spans.append(Span(d.low, d.high, State.NULL))
            elif binds and d.ulp <= ceiling:
                spans.append(Span(d.low, d.high, State.GATED, d.ulp))
            elif held.measured_shape:
                spans.append(Span(d.low, d.high, State.PAST, d.ulp))
            else:
                spans.append(Span(d.low, d.high, State.NONE))
        return spans
    if held.gated:
        return [Span(-_INF, _INF, State.GATED, row.contract.max_ulp)]
    figure = _recorded(row) if held.measured_shape else None
    state = State.PAST if figure is not None else State.NONE
    return [Span(-_INF, _INF, state, figure)]


def _span_at(spans: List[Span], x: float) -> Span:
    return next(
        (s for s in spans if s.low <= x < s.high), Span(-_INF, _INF, State.NONE)
    )


def _span_text(span: Span, row: Optional[Row]) -> str:
    if span.state is State.GATED:
        if row is not None and row.near_zero_atol:
            return f"{span.figure} (floor {row.near_zero_atol:g})"
        return str(span.figure)
    if span.state is State.PAST:
        return f"tolerance (max {span.figure})"
    return "unranked" if span.state is State.NULL else "—"


def _classify_segment(was: Span, now: Span) -> Optional[Kind]:
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
    """One resolved cell whose gate differs between two revisions of the table. *cell*
    names the row that decides it after the change (before it, for a cell nothing
    covers any more); *variants* is how many query variants that row decides
    differently; *segment* is the ``[low, high)`` input range a list's change is about.
    """

    cell: Cell
    kind: Kind
    before: Optional[Row]
    after: Optional[Row]
    variants: int = 1
    held_before: str = "—"
    held_after: str = "—"
    segment: Optional[Tuple[float, float]] = None

    @property
    def is_regression(self) -> bool:
        return self.kind.loosens


def _segment_changes(
    was: Held, now: Held, query: BudgetKey
) -> List[Tuple[Kind, Tuple[float, float], str, str]]:
    """``(kind, segment, held_before, held_after)`` for each run of the input over which
    the two rows' gates differ, over the union of both lists' edges. On a segment gated
    on both sides a widened floor ranks after a raise and before a tightening, as in
    :func:`_classify`."""
    before, after = _spans(was, query), _spans(now, query)
    edges = sorted(
        {e for spans in (before, after) for s in spans for e in (s.low, s.high)}
    )
    floor_widened = (
        was.row is not None
        and now.row is not None
        and (now.row.near_zero_atol or 0) > (was.row.near_zero_atol or 0)
    )
    found: List[list] = []
    for low, high in zip(edges, edges[1:]):
        a, b = _span_at(before, _inside(low, high)), _span_at(after, _inside(low, high))
        kind = _classify_segment(a, b)
        if (
            floor_widened
            and a.state is b.state is State.GATED
            and kind is not Kind.RAISED
        ):
            kind = Kind.FLOOR_WIDENED
        if kind is None:
            continue
        held = (kind, _span_text(a, was.row), _span_text(b, now.row))
        if found and found[-1][0] == held and found[-1][2] == low:
            found[-1][2] = high
        else:
            found.append([held, low, high])
    return [(kind, (low, high), t0, t1) for (kind, t0, t1), low, high in found]


def compare(base: Table, head: Table) -> List[Change]:
    """Every cell whose *resolved* gate differs, strongest change first. Where either
    side's row carries a list, the cell is judged segment by segment."""
    grouped: Dict[Tuple, Change] = {}
    for op in sorted(set(base) | set(head), key=lambda o: o.name):
        pinned = {
            name
            for row in (*base.get(op, ()), *head.get(op, ()))
            for name in (f.name for f in fields(BudgetKey))
            if getattr(row.key, name) is not None
        }
        open_dims = {f.name for f in fields(BudgetKey)} - pinned
        for query in _variants(base, head, op):
            was = _held(base, op, query, open_dims)
            now = _held(head, op, query, open_dims)
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
            for kind, segment, before_text, after_text in found:
                ident = (op, kind, deciding.key, segment, before_text, after_text)
                prior = grouped.get(ident)
                grouped[ident] = Change(
                    (op, deciding.key),
                    kind,
                    was.row,
                    now.row,
                    variants=prior.variants + 1 if prior else 1,
                    held_before=before_text,
                    held_after=after_text,
                    segment=segment,
                )
    return sorted(
        grouped.values(),
        key=lambda c: (
            c.kind.value,
            c.cell[0].name,
            c.cell[1].describe(),
            c.segment or (),
        ),
    )


def _describe(change: Change) -> str:
    text = _name(*change.cell)
    if change.variants > 1:
        text += f" ×{change.variants}"
    if change.segment is not None:
        low, high = change.segment
        text += f" x in [{low:g}, {high:g})"
    return text


def _md_table(
    header: Sequence[str], lines: List[str], *, capped: bool = False
) -> List[str]:
    withheld = len(lines) - _MAX_ROWS
    if capped and withheld > 0:
        lines = lines[:_MAX_ROWS] + [
            f"| _… {withheld} more_ |" + " |" * (len(header) - 1)
        ]
    return ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header), *lines]


def render_budget_diff(changes: List[Change]) -> str:
    """The diff report: regressions first."""
    if not changes:
        return "### SFPU ULP budgets\n\nNo budget changed.\n"
    line = {
        c: f"| `{_describe(c)}` | {_KIND_TEXT[c.kind]} | {c.held_before} | {c.held_after} |"
        for c in changes
    }
    regressions = [c for c in changes if c.is_regression]
    improvements = [c for c in changes if not c.is_regression]
    header = ("cell", "change", "before", "after")
    out = ["### SFPU ULP budgets", ""]
    if regressions:
        out += [
            f"**{len(regressions)} cell(s) loosen a gate.** A gate is the row's "
            "`measured` figure, so raising it is recording a worse measurement, in a "
            "change of its own.",
            "",
            *_md_table(header, [line[c] for c in regressions]),
            "",
        ]
    if improvements:
        out += [
            f"<details><summary>{len(improvements)} other budget change(s)</summary>",
            "",
            *_md_table(header, [line[c] for c in improvements], capped=True),
            "",
            "</details>",
            "",
        ]
    return "\n".join(out) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# What the hardware measured against what the table declares
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CellReading:
    """Every record of one cell folded together: the worst of each figure, whether any
    lane was ranked, its per-binade profile, and per entry of its list the worst lane
    and the most ranked lanes."""

    worst: int = 0
    nonfinite: int = 0
    ranked: bool = False
    bins: Tuple[Tuple[int, int, int, int], ...] = ()
    entries: Dict[Tuple[float, float], Tuple[Optional[int], int]] = field(
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

    def most_ranked_in_a_binade(self, low: float, high: float) -> Optional[int]:
        """The most ranked lanes any one binade of ``[low, high)`` holds, or ``None``
        without a profile: what the rule asks of a binade before it is ranked."""
        if not self.bins:
            return None
        return max(
            (r for b, _, r, _ in self.bins if low <= binade_edges(b)[0] < high),
            default=0,
        )


def read_cells(measurements: Iterable[Measurement]) -> Dict[Cell, CellReading]:
    cells: Dict[Cell, CellReading] = {}
    for m in measurements:
        cells.setdefault((m.op, m.key), CellReading()).fold(m)
    return cells


class Verdict(Enum):
    """What the headroom check found in one cell, or one entry of its list."""

    OVER_BUDGET = auto()
    REGRESSED = auto()
    NONFINITE = auto()
    NO_LANE = auto()
    TIGHTENS = auto()
    NOW_ANSWERS = auto()

    @property
    def fails(self) -> bool:
        return self not in (Verdict.TIGHTENS, Verdict.NOW_ANSWERS)


_VERDICT_TEXT = {
    Verdict.OVER_BUDGET: "over budget",
    Verdict.REGRESSED: "regressed",
    Verdict.NONFINITE: "non-finite lanes",
    Verdict.NO_LANE: "no lane measurable",
    Verdict.TIGHTENS: "re-emit tightens",
    Verdict.NOW_ANSWERS: "now answers",
}


@dataclass(frozen=True)
class Finding:
    cell: Cell
    verdict: Verdict
    seen: Optional[int]
    reference: Optional[int]
    entry: Optional[Domain] = None


@dataclass
class Headroom:
    """What the records say against the table. Nothing measured is not a pass."""

    measured: int = 0
    findings: List[Finding] = field(default_factory=list)
    unjudged: int = 0
    uncovered: int = 0

    @property
    def regressions(self) -> int:
        failed = sum(f.verdict.fails for f in self.findings)
        return failed if self.measured else 1

    def of(self, verdict: Verdict) -> List[Finding]:
        return [f for f in self.findings if f.verdict is verdict]


def _past(seen: int, figure: int, ceiling: float) -> Verdict:
    # The sweep fails a figure within the ceiling itself; past it, only this does.
    return Verdict.OVER_BUDGET if figure <= ceiling else Verdict.REGRESSED


def judge(table: Table, measurements: Iterable[Measurement]) -> Headroom:
    """Every recorded cell against the row it resolves to.

    A gated cell over its figure is over budget, and a better reading is a re-emit's to
    tighten. A tolerance cell is held to the figure and the non-finite count its row
    records. A cell with a list is judged entry by entry, its null entries' few ranked
    lanes held to the row's largest figure. A cell that ranked no lane against a
    recorded figure has regressed."""
    cells = read_cells(measurements)
    out = Headroom(measured=len(cells))
    for cell, reading in sorted(
        cells.items(), key=lambda item: (item[0][0].name, item[0][1].describe())
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
            largest = max(d.ulp for d in row.domains if d.ulp is not None)
            if not reading.ranked:
                found(Finding(cell, Verdict.NO_LANE, None, largest))
                continue
            ceiling = _ceiling(key)
            for entry in row.domains:
                got = reading.entries.get((entry.low, entry.high))
                if got is None:
                    out.unjudged += 1
                    continue
                seen, lanes = got
                if entry.ulp is None:
                    per_binade = reading.most_ranked_in_a_binade(entry.low, entry.high)
                    if (
                        lanes if per_binade is None else per_binade
                    ) >= DOMAIN_MIN_BIN_LANES:
                        found(Finding(cell, Verdict.NOW_ANSWERS, seen, None, entry))
                    if seen is not None and seen > largest:
                        found(
                            Finding(
                                cell,
                                _past(seen, largest, ceiling),
                                seen,
                                largest,
                                entry,
                            )
                        )
                elif lanes == 0:
                    found(Finding(cell, Verdict.NO_LANE, None, entry.ulp, entry))
                elif seen > entry.ulp:
                    found(
                        Finding(
                            cell,
                            _past(seen, entry.ulp, ceiling),
                            seen,
                            entry.ulp,
                            entry,
                        )
                    )
                elif seen < entry.ulp:
                    found(Finding(cell, Verdict.TIGHTENS, seen, entry.ulp, entry))
            continue
        if row.contract.metric is Metric.ULP and steps_bind(row.key, key):
            budget = row.contract.max_ulp
            if reading.worst != budget:
                verdict = (
                    Verdict.OVER_BUDGET if reading.worst > budget else Verdict.TIGHTENS
                )
                found(Finding(cell, verdict, reading.worst, budget))
            continue
        was = _recorded(row)
        if was is None:
            out.unjudged += 1
        elif not reading.ranked:
            found(Finding(cell, Verdict.NO_LANE, None, was))
        elif reading.worst > was:
            found(Finding(cell, Verdict.REGRESSED, reading.worst, was))
    return out


_SECTIONS = (
    ("Over budget", (Verdict.OVER_BUDGET,), "budget", False),
    (
        "Regressed against the row's recorded measurement",
        (Verdict.REGRESSED, Verdict.NONFINITE, Verdict.NO_LANE),
        "recorded",
        False,
    ),
    (
        "Measured better than the row's figure (a re-emit tightens these)",
        (Verdict.TIGHTENS,),
        "figure",
        True,
    ),
    (
        "Entries held unranked where the op now answers",
        (Verdict.NOW_ANSWERS,),
        "table",
        True,
    ),
)


def _finding_line(f: Finding) -> str:
    where = _name(*f.cell)
    if f.entry is not None:
        where += f" x in [{f.entry.low:g}, {f.entry.high:g})"
    seen = 0 if f.seen is None else f.seen
    reference = "unranked" if f.reference is None else f.reference
    return f"| `{where}` | {seen} | {reference} | {_VERDICT_TEXT[f.verdict]} |"


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
    for title, verdicts, reference, collapsed in _SECTIONS:
        lines = [_finding_line(f) for f in headroom.findings if f.verdict in verdicts]
        if not lines:
            continue
        heading = f"{title} — {len(lines)}"
        out += [
            f"<details><summary>{heading}</summary>" if collapsed else f"**{heading}**",
            "",
            *_md_table(("cell", "measured", reference, ""), lines, capped=collapsed),
            "",
        ]
        if collapsed:
            out += ["</details>", ""]
    if not headroom.findings:
        out += [
            "Every gated cell measures exactly its figure, and no judged tolerance cell "
            "is past its recorded measurement.",
            "",
        ]
    return "\n".join(out) + "\n"


def render_domains(
    measurements: List[Measurement],
    table: Optional[Table] = None,
    op: Optional[MathOperation] = None,
) -> str:
    """Per recorded cell with a profile, the list the emitter's rule draws on it beside
    the list the table holds: what a re-emit would write, with no device."""
    out = ["### SFPU ULP domains from the sweep's binade profiles", ""]
    for (cell_op, key), reading in read_cells(measurements).items():
        if (op is not None and cell_op is not op) or not reading.bins:
            continue
        out.append(f"- `{_name(cell_op, key)}`: {_domains_text(draw(reading, key))}")
        held = _resolve(table, cell_op, key) if table is not None else None
        if held is not None and held.domains:
            out.append(f"  - table: {_domains_text(held.domains)}")
    return "\n".join(out) + "\n"


def draw(reading: CellReading, query: BudgetKey) -> Tuple[Domain, ...]:
    """The list the emitter's rule draws on *reading*'s binade profile. A binade lies
    wholly inside one entry (``ulp_domains.binade``), so the worst of an entry's binades
    is the figure the emitter measures over its lanes."""
    profile = {
        b: (present, ranked, worst) for b, present, ranked, worst in reading.bins
    }

    def worst_in(low: float, high: float) -> Optional[int]:
        found = [
            w
            for b, (_, r, w) in profile.items()
            if r and low <= binade_edges(b)[0] < high
        ]
        return max(found) if found else None

    return entries_of(partition(profile, _ceiling(query)), worst_in)


def _domains_text(domains: Tuple[Domain, ...]) -> str:
    if not domains:
        return "no list (no entry has a figure)"
    return ", ".join(f"[{d.low:g}, {d.high:g}): {d.ulp}" for d in domains)


# ─────────────────────────────────────────────────────────────────────────────
# Command line
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
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
    domains = sub.add_parser("domains", help="the lists the rule draws on the profiles")
    domains.add_argument("--measured", type=Path, required=True)
    domains.add_argument("--table", type=Path)
    domains.add_argument("--op", type=lambda name: MathOperation[name])
    for command in (diff, headroom, domains):
        command.add_argument("--out", type=Path, help="also write the report here")
    args = parser.parse_args(argv)
    try:
        if args.command == "diff":
            changes = compare(load_table(args.base), load_table(args.head))
            report = render_budget_diff(changes)
            loosened = any(c.is_regression for c in changes)
            status = 1 if loosened and not args.allow_raises else 0
        elif args.command == "headroom":
            headroom = judge(load_table(args.table), read_measurements(args.measured))
            report = render_headroom(headroom)
            status = 1 if headroom.regressions else 0
        else:
            table = load_table(args.table) if args.table else None
            report = render_domains(read_measurements(args.measured), table, args.op)
            status = 0
    except ValueError as refused:  # a TableError, or a record that does not parse
        report, status = f"cannot load: {refused}\n", 2
    print(report, end="")
    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
    return status


if __name__ == "__main__":
    sys.exit(main())

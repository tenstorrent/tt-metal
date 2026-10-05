# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whether a change to the SFPU accuracy table loosens a gate, and whether the
hardware still fits the gates it declares.

* ``diff``: what a pull request does to the budgets. A budget that goes up, a row that
  stops being gated, or a tolerance cell whose recorded measurement or declared
  ``atol``/``rtol`` loosens, is a *regression* whatever the reason; the table's rule is
  that it may only happen alongside a fresh measurement. No hardware needed.
* ``headroom``: what the hardware measured, from the ``--ulp-measure`` rows, against
  the budgets. The sweep already fails a cell it cannot meet; this says which budgets
  are about to become regressions, and judges the tolerance cells the sweep cannot.

Standalone on purpose -- ``yaml``, the standard library and the sibling
``ulp_provenance`` (itself the same), so the PR check runs on a slim runner and can
parse a *base* revision of the table. ``test_ulp_budget_diff.py`` ties this parse back
to the real loader so the two cannot drift.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from xml.etree import ElementTree

if __package__:
    from .ulp_provenance import BudgetTable, KeyLine, Kind, Provenance
else:  # run by path on the slim runner, where `helpers/__init__.py` cannot import
    from ulp_provenance import BudgetTable, KeyLine, Kind, Provenance

#: The key dimensions of a row, in the order a cell is named in a report. The registry's
#: own map is ``sfpu_accuracy_budget._KEY_FIELDS``; a host test holds the two equal.
KEY_FIELDS = ("in", "out", "approx", "dest", "arch")

#: ``sfpu_accuracy_budget.MEASURED_ARCH``, by enum name. A ULP row that does not name an
#: arch binds only here; anywhere else the registry hands the cell its tolerance. Spelled
#: out because this module may not import the registry; a host test ties them together.
MEASURED_ARCH = "WORMHOLE"

#: A cell identity: the op plus whichever key dimensions the row pins.
Cell = Tuple[str, Tuple[Tuple[str, str], ...]]

#: Each report section shows this many rows and says how many it withheld: a PR
#: comment has a size limit, and a report that lists every cell is one nobody reads.
_MAX_ROWS = 40

#: Below this fraction of its budget a cell carries slack the sweep cannot justify. Not
#: a failure -- tightening is a deliberate change with its own measurement -- but it is
#: the list someone should work through.
_SLACK_FRACTION = 0.5

#: The smallest budget the slack list considers. A budget of 1 measured at 0 is the
#: emitter's floor for a strided Float32 cell, where a sample cannot prove the 0 holds
#: over the lanes it skipped (``ulp_sweep._verdict``); the only tighter budget is the 0
#: the emitter deliberately refused to write.
_SLACK_MIN_BUDGET = 2


def _cell_name(op: str, key: Tuple[Tuple[str, str], ...]) -> str:
    if not key:
        return f"{op} (default)"
    return f"{op} {{" + ", ".join(f"{k}: {v}" for k, v in key) + "}"


@dataclass(frozen=True)
class Row:
    """One table row, reduced to what a regression is defined over."""

    op: str
    key: Tuple[Tuple[str, str], ...]
    max_ulp: Optional[int]
    #: The row's own provenance, or its op's key line when it has no comment: the table
    #: states the run once there, and updating either registers as a re-measurement.
    provenance: Union[Provenance, KeyLine]
    #: The near-zero floor is part of the gate: `ulp_elementwise_valid` accepts a lane
    #: inside it however many steps out it is, so widening it loosens the gate.
    near_zero_atol: Optional[float] = None
    #: A tolerance row's declared bounds. The functional drivers still compare a
    #: tolerance cell with them, so widening one loosens that cell.
    atol: Optional[float] = None
    rtol: Optional[float] = None

    @property
    def gated(self) -> bool:
        """Whether the row carries a step budget. Whether that budget *binds* depends on
        the arch it is asked about too: see :func:`_gates`."""
        return self.max_ulp is not None

    @property
    def declares_tolerance(self) -> bool:
        """Whether the row has numbers a tolerance compare uses, as in the registry."""
        return self.atol is not None or self.rtol is not None

    @property
    def floor(self) -> float:
        """The floor as a number, so an absent one compares as no rescue at all."""
        return self.near_zero_atol or 0.0

    def describe(self) -> str:
        return _cell_name(self.op, self.key)


@dataclass(frozen=True)
class Change:
    """One resolved cell whose gate differs between two revisions of the table.

    *cell* names the row that decides the cell after the change (before it, for a cell
    nothing covers any more); *variants* is how many query variants that row decides
    differently than the base did, since a collapsed row governs several. *held_before*
    and *held_after* say what the cell was held to on each side, as the report shows it.

    *remeasured* is whether the deciding row's provenance comment differs from the same
    row's in the base table -- the table's rule for raising a budget. Read off that one
    row, not off the two rows the cell resolves to: when a deleted row hands its cells to
    a broader one, those two comments differ although no comment was touched.
    """

    cell: Cell
    kind: str  # one of _KIND_ORDER
    before: Optional[Row]
    after: Optional[Row]
    variants: int = 1
    held_before: str = "—"
    held_after: str = "—"
    remeasured: bool = False

    #: The kinds that weaken a gate. Everything else is neutral or an improvement.
    REGRESSIONS = frozenset(
        {
            "raised",
            "floor_widened",
            "ungated",
            "removed",
            "tolerance_widened",
            "baseline_raised",
            "baseline_dropped",
        }
    )

    @property
    def is_regression(self) -> bool:
        return self.kind in self.REGRESSIONS


# ─────────────────────────────────────────────────────────────────────────────
# Reading the table
# ─────────────────────────────────────────────────────────────────────────────


class DuplicateRow(ValueError):
    """Two rows of one op pin the same cell: ``_load_table`` refuses such a table, so no
    verdict about it is meaningful. *cell* is the cell they both pin."""

    def __init__(self, cell: Cell):
        self.cell = cell
        super().__init__(
            f"{cell[0]}: duplicate row for {_cell_name(*cell)}. The registry refuses "
            "two rows of equal specificity, so this table cannot load; no budget "
            "verdict is meaningful for it."
        )


def _key(row) -> Tuple[Tuple[str, str], ...]:
    """A row's key as the registry reads it. An arch is accepted by enum name or value
    (``WORMHOLE`` or ``wormhole``) and named by the enum name, as a measurement is."""
    return tuple((k, v.upper() if k == "arch" else v) for k, v in row.key)


def _number(value) -> Optional[float]:
    return value if isinstance(value, (int, float)) and value is not True else None


def parse_table(text: str) -> Dict[Cell, Row]:
    """Every row of the table, keyed by the cell it governs, read through
    :class:`BudgetTable`: values as PyYAML loads them, so anchors and ``<<`` merge keys
    mean what they mean, and a YAML boolean names the cell the registry would."""
    table = BudgetTable(text)
    rows: Dict[Cell, Row] = {}
    for row in table.rows:
        cell: Cell = (row.op, _key(row))
        if cell in rows:
            # `_load_table` refuses two rows of equal specificity, so this table
            # cannot load; keeping the last row silently produced a verdict for it.
            raise DuplicateRow(cell)
        values = row.values
        rows[cell] = Row(
            op=row.op,
            key=cell[1],
            max_ulp=row.max_ulp,
            provenance=row.provenance or table.blocks[row.op].key_line,
            near_zero_atol=row.near_zero_atol,
            atol=_number(values.get("atol")),
            rtol=_number(values.get("rtol")),
        )
    return rows


def _resolve(
    table: Dict[Cell, Row],
    op: str,
    key: Tuple[Tuple[str, str], ...],
    only: Optional[Callable[[Row], bool]] = None,
) -> Optional[Row]:
    """The most specific row of *op* covering *key*, by the registry's own rule, among
    the rows *only* admits. Two equally specific matches are refused, as ``_winner``
    refuses them: the registry will not load such a table, so no verdict on it means
    anything, and keeping the first would make the verdict depend on the file's order.
    """
    asked = dict(key)
    matched = [
        row
        for (row_op, row_key), row in table.items()
        if row_op == op
        and all(asked.get(k) == v for k, v in row_key)
        and (only is None or only(row))
    ]
    if not matched:
        return None
    best = max(len(row.key) for row in matched)
    winners = [row for row in matched if len(row.key) == best]
    if len(winners) > 1:
        raise ValueError(
            f"{op} has {len(winners)} equally specific rows matching "
            f"{_cell_name(op, key)}: {', '.join(r.describe() for r in winners)}. The "
            "registry refuses this table, so no budget verdict is meaningful for it."
        )
    return winners[0]


def _gates(row: Optional[Row], key: Tuple[Tuple[str, str], ...]) -> bool:
    """Whether *row* holds the cell *key* names to a step budget: it carries one, and
    the cell is on the arch the budget was measured on -- :data:`MEASURED_ARCH` for a
    row that names none (an unset arch is that one), or the arch the row names."""
    if row is None or not row.gated:
        return False
    return dict(key).get("arch", MEASURED_ARCH) == MEASURED_ARCH or "arch" in dict(
        row.key
    )


# ─────────────────────────────────────────────────────────────────────────────
# What a pull request does to the gates
# ─────────────────────────────────────────────────────────────────────────────


def _variants(base: Dict[Cell, Row], head: Dict[Cell, Row], op: str):
    """Every query the registry could be asked about *op*, as far as either table can
    tell them apart: each key dimension takes every value some row pins, or is left
    unset -- an unset query dimension matches only a wildcard, as in the registry.

    The arch is never unset (``accuracy_contract`` requires it) and always includes
    :data:`MEASURED_ARCH`, where an unkeyed budget binds. Any other arch no row names
    binds no budget in either table, so it cannot change and is not asked."""
    values: Dict[str, set] = {k: set() for k in KEY_FIELDS}
    for table in (base, head):
        for row_op, key in table:
            if row_op == op:
                for k, v in key:
                    values[k].add(v)
    axes = [
        (
            sorted(values[k] | {MEASURED_ARCH})
            if k == "arch"
            else sorted(values[k]) + [None]
        )
        for k in KEY_FIELDS
    ]
    for combo in itertools.product(*axes):
        yield tuple((k, v) for k, v in zip(KEY_FIELDS, combo) if v is not None)


@dataclass(frozen=True)
class _Held:
    """What one query variant is held to in one revision of the table."""

    row: Optional[Row]  # the row the variant resolves to
    gated: bool  # whether that row's step budget binds here
    #: The most specific row declaring ``atol``/``rtol``: what a tolerance compare uses.
    declared: Optional[Row]
    #: Whether the variant resolves as a measured cell does: every dimension of
    #: ``_MEASURED_KEY`` named, or pinned by no row of the op (so naming it would
    #: resolve the same). Only such a cell is ever held to a recorded figure. A query
    #: that leaves a pinned dimension unset is a registry query, and a re-emit that
    #: splits a row by that dimension leaves it matching nothing without any measured
    #: cell losing its baseline.
    measured_shape: bool = True

    @property
    def _tolerance_row(self) -> Optional[Row]:
        """The row, if it is a tolerance row: the only kind whose comment is a baseline.
        A step budget that does not bind on this arch is not one, and its "max N ULP"
        describes the arch it binds on."""
        if not self.measured_shape or self.row is None or self.row.gated:
            return None
        return self.row

    @property
    def baseline(self) -> Optional[int]:
        """The measurement the headroom report holds a tolerance cell to."""
        row = self._tolerance_row
        return None if row is None else recorded_max(row)

    @property
    def nonfinite(self) -> int:
        row = self._tolerance_row
        return 0 if row is None else recorded_nonfinite(row)

    def describe(self) -> str:
        if self.row is None:
            return "—"
        if self.gated:
            text = str(self.row.max_ulp)
            if self.row.near_zero_atol:  # or a widened floor reads alike on both sides
                text += f" (floor {self.row.near_zero_atol:g})"
            return text
        parts = []
        if self.baseline is not None:
            parts.append(f"max {self.baseline}")
        if self.nonfinite:
            parts.append(f"{self.nonfinite} non-finite")
        if self.declared is not None:
            for name in ("atol", "rtol"):
                value = getattr(self.declared, name)
                if value is not None:
                    parts.append(f"{name} {value:g}")
        return "tolerance" + (f" ({', '.join(parts)})" if parts else "")


#: The key dimensions every ``--ulp-measure`` row names (``arch`` aside, which the
#: variants always name): the shape of a cell the headroom report can judge.
_MEASURED_KEY = ("in", "out", "approx", "dest")


def _held(table: Dict[Cell, Row], op: str, key, open_dims=frozenset()) -> _Held:
    """What *key* is held to in *table*. *open_dims* are the dimensions no row of *op*
    pins in either revision, which a query may leave unset and still be a measured
    cell's."""
    row = _resolve(table, op, key)
    gated = _gates(row, key)
    declared = (
        None if gated else _resolve(table, op, key, only=lambda r: r.declares_tolerance)
    )
    pinned = dict(key)
    shape = all(k in pinned or k in open_dims for k in _MEASURED_KEY)
    return _Held(row, gated, declared, shape)


def _widened(was: Optional[Row], now: Optional[Row]) -> bool:
    """Whether a tolerance compare got looser: a bound went up, or a declared bound went
    away and left the cell on a default this tool cannot see."""
    if was is None:
        return False
    if now is None:
        return True
    return any(
        old is not None and (new is None or new > old)
        for old, new in ((was.atol, now.atol), (was.rtol, now.rtol))
    )


def _classify(was: _Held, now: _Held) -> Optional[str]:
    """How the gate one variant resolves to has changed, or ``None`` if it has not."""
    if was.gated and not now.gated:
        # Only a loss if it was gating; a tolerance cell that loses its row gates
        # exactly as it did.
        return "removed" if now.row is None else "ungated"
    if not was.gated and now.gated:
        return "gated"
    if was.gated and now.gated:
        if now.row.max_ulp > was.row.max_ulp:
            return "raised"
        if now.row.floor > was.row.floor:
            # Before `tightened`: a smaller `max_ulp` with a wider floor rescues more
            # lanes than it fails, and the budget alone would call that an improvement.
            return "floor_widened"
        if now.row.max_ulp < was.row.max_ulp:
            return "tightened"
        return None
    # Neither side gates, but the cell is still held to something: the eltwise drivers
    # compare it with the declared atol/rtol, and PR Gate's headroom report with the
    # measurement and the non-finite count its row records. Loosening either needs the
    # same re-measurement a raised budget does.
    if _widened(was.declared, now.declared):
        return "tolerance_widened"
    if was.baseline is not None and now.baseline is None:
        return "baseline_dropped"
    if (
        was.baseline is not None
        and now.baseline > was.baseline
        or now.nonfinite > was.nonfinite
    ):
        return "baseline_raised"
    # A new tolerance row is worth a line; a vanished one that held nothing is not.
    return "added" if was.row is None and now.row is not None else None


_KIND_ORDER = (
    "raised",
    "floor_widened",
    "ungated",
    "removed",
    "tolerance_widened",
    "baseline_raised",
    "baseline_dropped",
    "tightened",
    "gated",
    "added",
)


def compare(base: Dict[Cell, Row], head: Dict[Cell, Row]) -> List[Change]:
    """Every cell whose *resolved* gate differs, strongest change first.

    Resolved, not row by row, because the table's shape is not the gate: deleting a
    row the sweep collapsed is no loss while a broader row gives the same budget, and a
    new `{in: Float32, max_ulp: 1000}` under a `max_ulp: 2` default loosens every
    Float32 cell without any row being raised. Variants the same pair of rows decides
    the same way collapse into one ``Change``.
    """
    grouped: Dict[Tuple, Change] = {}
    for op in sorted({c[0] for c in base} | {c[0] for c in head}):
        pinned_anywhere = {
            k for table in (base, head) for (o, key) in table if o == op for k, _ in key
        }
        open_dims = frozenset(KEY_FIELDS) - pinned_anywhere
        for key in _variants(base, head, op):
            was = _held(base, op, key, open_dims)
            now = _held(head, op, key, open_dims)
            kind = _classify(was, now)
            if kind is None:
                continue
            deciding = now.row if now.row is not None else was.row
            # Grouped by the deciding row and what the cell is held to on each side: a
            # collapsed row replacing N keyed rows of one budget is one line marked xN.
            ident = (op, kind, deciding.key, was.describe(), now.describe())
            found = grouped.get(ident)
            grouped[ident] = Change(
                (op, deciding.key),
                kind,
                was.row,
                now.row,
                variants=found.variants + 1 if found else 1,
                held_before=was.describe(),
                held_after=now.describe(),
                remeasured=_remeasured(base, now.row),
            )
    return sorted(grouped.values(), key=lambda c: (_KIND_ORDER.index(c.kind), c.cell))


def _remeasured(base: Dict[Cell, Row], row: Optional[Row]) -> bool:
    """Whether *row*, the head row deciding a cell, carries a provenance comment the base
    table's same row did not -- a new row's comment is new by definition."""
    if row is None:
        return False
    was = base.get((row.op, row.key))
    return was is None or was.provenance != row.provenance


_KIND_TEXT = {
    "raised": "budget raised",
    "floor_widened": "near-zero floor widened",
    "ungated": "gating lost (now tolerance)",
    "removed": "gate lost (no row covers it)",
    "tolerance_widened": "declared atol/rtol loosened",
    "baseline_raised": "recorded measurement raised",
    "baseline_dropped": "recorded measurement dropped",
    "tightened": "budget tightened",
    "gated": "newly gated",
    "added": "new row",
}


def _describe(c: Change) -> str:
    row = c.after or c.before
    return row.describe() + (f" ×{c.variants}" if c.variants > 1 else "")


def _md_table(header: Sequence[str], lines: List[str]) -> List[str]:
    """A markdown table of at most ``_MAX_ROWS`` rows, then one row saying how many were
    withheld. The column count comes from *header*, so the filler row cannot drift."""
    withheld = len(lines) - _MAX_ROWS
    if withheld > 0:
        lines = lines[:_MAX_ROWS] + [
            f"| _… {withheld} more_ |" + " |" * (len(header) - 1)
        ]
    return [
        "| " + " | ".join(header) + " |",
        "|" + " --- |" * len(header),
        *lines,
    ]


def _change_row(c: Change, *, with_provenance: bool) -> str:
    cells = f"| `{_describe(c)}` | {_KIND_TEXT[c.kind]} | {c.held_before} | "
    cells += f"{c.held_after} |"
    if with_provenance:
        cells += f" {'yes' if c.remeasured else '**no**'} |"
    return cells


def render_budget_diff(changes: List[Change], label_hint: str) -> str:
    """The PR comment. Regressions first, and what to do about them."""
    if not changes:
        return "### SFPU ULP budgets\n\nNo budget changed.\n"
    regressions = [c for c in changes if c.is_regression]
    improvements = [c for c in changes if not c.is_regression]
    out = ["### SFPU ULP budgets", ""]
    if regressions:
        out += [
            f"**{len(regressions)} cell(s) loosen a gate.** A budget may only be raised "
            "by re-measuring and updating that row's provenance comment in the same "
            "change — that is the table's own rule, and it is what makes every number "
            "in it traceable.",
            "",
            *_md_table(
                ("cell", "change", "before", "after", "re-measured"),
                [_change_row(c, with_provenance=True) for c in regressions],
            ),
            "",
            "If these are genuine re-measurements, say so in the PR body and add the "
            f"`{label_hint}` label. If they are not, the budget is being fitted to a "
            "failure and the kernel or the golden is what moved.",
            "",
        ]
    if improvements:
        out += [
            f"<details><summary>{len(improvements)} other budget change(s)</summary>",
            "",
            *_md_table(
                ("cell", "change", "before", "after"),
                [_change_row(c, with_provenance=False) for c in improvements],
            ),
            "",
            "</details>",
            "",
        ]
    return "\n".join(out) + "\n"


def _allowed_note(regressions: List[Change], label_hint: str) -> str:
    """Appended when the override label admits the loosened gates; the ones admitted
    without a fresh measurement are still named."""
    note = (
        f"\n_Allowed: the `{label_hint}` label is set on this pull request, admitting "
        f"{len(regressions)} loosened gate(s)._\n"
    )
    unmeasured = [c for c in regressions if not c.remeasured]
    if unmeasured:
        note += (
            f"\n**{len(unmeasured)} of them carry no fresh measurement** -- the row's "
            "provenance comment is unchanged, so the budget was edited rather than "
            "re-measured:\n\n"
        )
        note += "".join(
            f"- `{(c.after or c.before).describe()}`\n" for c in unmeasured[:_MAX_ROWS]
        )
        if len(unmeasured) > _MAX_ROWS:
            note += f"- _… {len(unmeasured) - _MAX_ROWS} more_\n"
    return note


# ─────────────────────────────────────────────────────────────────────────────
# What the hardware measured against what the table declares
# ─────────────────────────────────────────────────────────────────────────────


def _measured_cell(row: dict) -> Cell:
    """The cell one ``--ulp-measure`` row describes, keyed as a table row is."""
    return (
        row["op"],
        tuple((k, str(row[k])) for k in KEY_FIELDS if row.get(k) is not None),
    )


def _measured_cells(rows: Iterable[dict]) -> Dict[Cell, int]:
    """The worst lane each variant reached, from ``--ulp-measure`` rows. A driver
    enumerates axes the budget key does not, so several rows land on one cell and the
    worst wins, as in ``ulp_sweep.record``."""
    worst: Dict[Cell, int] = {}
    for row in rows:
        cell = _measured_cell(row)
        worst[cell] = max(worst.get(cell, 0), int(row["max"]))
    return worst


def _nonfinite_cells(rows: Iterable[dict]) -> Dict[Cell, int]:
    """Per cell, the most lanes any measurement saw where the hardware and the golden
    disagree about being finite: an inf/NaN against a finite golden, a finite answer to
    an infinite one, or infinities of opposite sign (0 for rows written before the sweep
    recorded the count)."""
    worst: Dict[Cell, int] = {}
    for row in rows:
        cell = _measured_cell(row)
        worst[cell] = max(worst.get(cell, 0), int(row.get("nonfinite", 0)))
    return worst


def recorded_nonfinite(row: Row) -> int:
    """How many non-finite lanes the row already accounts for: the count on a "not
    measurable" row, 0 on any other."""
    p = row.provenance
    if isinstance(p, Provenance) and p.kind is Kind.UNMEASURABLE:
        return p.nonfinite_lanes or 0
    return 0


def recorded_max(row: Row) -> Optional[int]:
    """The measurement a row records, or ``None`` if there is none to judge against.

    Only a row that pins both ``in`` and ``out`` is a baseline: the emitter writes every
    row that way, and its figure is the whole-format worst lane the sweep re-measures.
    A hand-written broader row lists a sampled driver's figure, or one per format
    ("max 868220929 ULP Float32 / 13249 Float16_b / ..."), and judging a cell against
    either would report a regression the kernel never had."""
    pinned = dict(row.key)
    if "in" not in pinned or "out" not in pinned:
        return None
    p = row.provenance
    if isinstance(p, KeyLine):
        p = p.header_provenance
    return None if p is None else p.recorded_max


def _headroom_line(cell: Cell, worst: int, reference: int, verdict: str) -> str:
    return f"| `{_cell_name(*cell)}` | {worst} | {reference} | {verdict} |"


#: The sweep's test id (test_unary_sfpu_ulp.py's parametrize ids), read back into the
#: cell it ran: ``test_unary_sfpu_ulp_sweep[Abs-in:Float16_b-out:Float16_b-approx:No-
#: dest_acc:Yes]``.
_SWEEP_ID = re.compile(r"\[(\w+)-in:(\w+)-out:(\w+)-approx:(\w+)-dest_acc:(\w+)\]$")

#: A failure message longer than this is cut: the first line of an assertion already
#: names the cell and the figure, and a PR comment has a size limit.
_MESSAGE_CHARS = 300


@dataclass(frozen=True)
class Failure:
    """One test the sweep failed, from its JUnit report."""

    test: str  # the parametrize id: the combination of parameters that failed
    message: str  # the first line of the failure, which carries the measured figure
    #: ``(op, in, out, approx, dest)`` when the id is the sweep's, else ``None``.
    cell: Optional[Tuple[str, str, str, str, str]] = None


def junit_failures(text: str) -> List[Failure]:
    """Every failed or errored test case in a JUnit report. The measurements say which
    cells went over a budget; this says which tests failed for any other reason -- a
    gated cell with no measurable lane, a stale known-lane excuse, a crash -- which
    write no measurement at all."""
    failures = []
    for case in ElementTree.fromstring(text).iter("testcase"):
        bad = case.find("failure")
        if bad is None:
            bad = case.find("error")
        if bad is None:
            continue
        name = case.get("name", "")
        lines = (bad.get("message") or bad.text or "").strip().splitlines()
        found = _SWEEP_ID.search(name)
        failures.append(
            Failure(
                test=name[name.find("[") :] if "[" in name else name,
                message=(lines[0] if lines else "")[:_MESSAGE_CHARS],
                cell=found.groups() if found else None,
            )
        )
    return failures


def junit_completed(text: str) -> List[Tuple[str, str, str, str, str]]:
    """The cell of every sweep test case in a JUnit report that neither failed nor
    errored. Each wrote a measurement -- a gated cell from ``passed_test``, a tolerance
    cell before its skip -- so one with none means the recorder lost it, and nothing
    judged that cell."""
    cells = []
    for case in ElementTree.fromstring(text).iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            continue
        found = _SWEEP_ID.search(case.get("name", ""))
        if found:
            cells.append(found.groups())
    return cells


def _same_cell(cell: Cell) -> Tuple[str, ...]:
    """A measured cell as a failure's ``(op, in, out, approx, dest)``."""
    key = dict(cell[1])
    return (cell[0], *(key.get(k) for k in ("in", "out", "approx", "dest")))


def render_headroom(
    table: Dict[Cell, Row],
    measured: Dict[Cell, int],
    nonfinite: Optional[Dict[Cell, int]] = None,
    failures: Iterable[Failure] = (),
    completed: Iterable[Tuple[str, str, str, str, str]] = (),
) -> Tuple[str, int]:
    """The report, and the regression count the workflow fails on.

    A gated cell is judged against its budget; the sweep already fails one it cannot
    meet, so the value here is which cells have no headroom left or carry slack. A
    tolerance cell has no budget and the sweep passes it whatever it measures, so it is
    judged against the measurement its own row records, and only this report sees it.
    The same holds for a cell the sweep could not measure: an overflow is recorded as a
    lane count, and more such lanes than its row accounts for is a regression too.

    *failures* are the sweep's failed tests (:func:`junit_failures`). One whose cell is
    already listed as over budget or regressed is not repeated; the rest -- failures no
    measurement describes -- get a section of their own and count as regressions.

    *completed* are the sweep's tests that did not fail (:func:`junit_completed`). Each
    should have written a measurement; one that did not was judged by nothing, since
    the recorder swallows a failed write, so it is listed and counts as a regression.
    An empty comparison -- nothing measured, nothing failed -- fails too: it is a sweep
    that did not run, not one that passed.
    """
    over: List[str] = []
    regressed: List[str] = []
    tight: List[str] = []
    slack: List[str] = []
    unjudged = 0
    listed = set()  # the cells a section above already names, as failures key them
    for cell, count in sorted((nonfinite or {}).items()):
        row = _resolve(table, *cell)
        if row is None or count <= recorded_nonfinite(row):
            continue
        regressed.append(
            _headroom_line(cell, count, recorded_nonfinite(row), "non-finite lanes")
        )
        listed.add(_same_cell(cell))
    for cell, worst in sorted(measured.items()):
        row = _resolve(table, *cell)
        if row is None:
            continue
        if not _gates(row, cell[1]):
            was = recorded_max(row)
            if was is None:
                unjudged += 1
            elif worst > was:
                regressed.append(_headroom_line(cell, worst, was, "regressed"))
                listed.add(_same_cell(cell))
        elif worst > row.max_ulp:
            over.append(_headroom_line(cell, worst, row.max_ulp, "over budget"))
            listed.add(_same_cell(cell))
        elif worst == row.max_ulp and row.max_ulp > 0:
            # `0 == 0` is an exact-by-construction op doing what it claims, enrolled so
            # that any drift fails; reporting it as "no headroom" buried the real ones.
            tight.append(_headroom_line(cell, worst, row.max_ulp, "no headroom"))
        elif row.max_ulp >= _SLACK_MIN_BUDGET and worst < _SLACK_FRACTION * row.max_ulp:
            slack.append(_headroom_line(cell, worst, row.max_ulp, "could tighten"))

    failed = [
        f"| `{f.test}` | {f.message.replace('|', '&#124;')} |"
        for f in failures
        if f.cell is None or f.cell not in listed
    ]

    recorded = {_same_cell(cell) for cell in measured}
    recorded |= {_same_cell(cell) for cell in (nonfinite or {})}
    unrecorded = sorted(set(completed) - recorded)

    out = ["### SFPU ULP sweep vs the declared budgets", ""]
    if not measured and not failed and not unrecorded:
        out += [
            "No measurements recorded and no test failed: nothing was compared, so "
            "this is not a pass.",
            "",
        ]
        return "\n".join(out), 1
    out += [f"{len(measured)} cell(s) measured.", ""]
    if unjudged:
        # Not a failure, but not a pass either: without a recorded figure there is
        # nothing to hold the cell to, and a clean summary must not read as covering it.
        out += [
            f"{unjudged} tolerance cell(s) measured but not judged: their row records no "
            "`max N ULP` to hold them to, or does not pin both `in` and `out`.",
            "",
        ]
    sections = (
        ("Over budget", over, "budget", False),
        (
            "Regressed on the tolerance metric (past the row's recorded measurement)",
            regressed,
            "last measured",
            False,
        ),
        ("No headroom left", tight, "budget", True),
        (
            f"Carrying slack (measured under {_SLACK_FRACTION:.0%} of budget)",
            slack,
            "budget",
            True,
        ),
    )
    for title, lines, reference, collapsed in sections:
        if not lines:
            continue
        heading = f"{title} — {len(lines)}"
        out += [
            f"<details><summary>{heading}</summary>" if collapsed else f"**{heading}**",
            "",
            *_md_table(("cell", "measured", reference, ""), lines),
            "",
        ]
        if collapsed:
            out += ["</details>", ""]
    if failed:
        out += [
            f"**Failed in the sweep without a measurement above — {len(failed)}**",
            "",
            *_md_table(("test (op, in, out, approx, dest_acc)", "failure"), failed),
            "",
        ]
    if unrecorded:
        lines = [f"| `{', '.join(cell)}` |" for cell in unrecorded]
        out += [
            f"**Ran without a measurement — {len(unrecorded)}**",
            "",
            "These tests finished without failing but wrote no measurement, so nothing "
            "judged them (an unwritable or full disk loses rows without failing a "
            "test).",
            "",
            *_md_table(("cell (op, in, out, approx, dest_acc)",), lines),
            "",
        ]
    if not (over or regressed or tight or slack or failed or unrecorded):
        out += [
            "Every gated cell is inside its budget with headroom to spare, and no "
            "judged tolerance cell is past its recorded measurement.",
            "",
        ]
    return "\n".join(out), len(over) + len(regressed) + len(failed) + len(unrecorded)


# ─────────────────────────────────────────────────────────────────────────────
# Command line
# ─────────────────────────────────────────────────────────────────────────────


def _diff(args) -> Tuple[str, int]:
    changes = compare(
        parse_table(args.base.read_text(encoding="utf-8")),
        parse_table(args.head.read_text(encoding="utf-8")),
    )
    report = render_budget_diff(changes, args.label_hint)
    regressions = [c for c in changes if c.is_regression]
    if regressions and args.allow_raises:
        report += _allowed_note(regressions, args.label_hint)
    return report, 0 if not regressions or args.allow_raises else 1


def _headroom(args) -> Tuple[str, int]:
    table = parse_table(args.table.read_text(encoding="utf-8"))
    # A sweep that died before its first cell writes no file; the report then says it
    # measured nothing, and the JUnit failures say why.
    text = args.measured.read_text(encoding="utf-8") if args.measured.exists() else ""
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    failures, completed = [], []
    if args.junit is not None:
        # Named but absent is a sweep that never finished writing it: without it there
        # is no telling which cells ran, so nothing below could be called a pass.
        if not args.junit.exists():
            return (
                "### SFPU ULP sweep vs the declared budgets\n\n"
                f"No JUnit report at `{args.junit}`: the sweep did not finish, so "
                "which cells it judged is unknown.\n",
                1,
            )
        junit = args.junit.read_text(encoding="utf-8")
        failures, completed = junit_failures(junit), junit_completed(junit)
    report, regressions = render_headroom(
        table, _measured_cells(rows), _nonfinite_cells(rows), failures, completed
    )
    return report, 1 if regressions else 0


def main(argv: Optional[List[str]] = None) -> int:
    # The whole first paragraph: it is hard-wrapped, so its first line stops mid-sentence.
    summary = " ".join(__doc__.split("\n\n")[0].split())
    parser = argparse.ArgumentParser(description=summary)
    sub = parser.add_subparsers(dest="mode", required=True)

    d = sub.add_parser("diff", help="compare two revisions of the budget table")
    d.add_argument("--base", type=Path, required=True)
    d.add_argument("--head", type=Path, required=True)
    d.add_argument(
        "--allow-raises",
        action="store_true",
        help="report loosened gates but exit 0 (the override label is set)",
    )
    d.add_argument("--label-hint", default="ulp-budget-raise-approved")
    d.add_argument("--out", type=Path, help="write the report here as well as stdout")

    h = sub.add_parser("headroom", help="compare a --ulp-measure run against the table")
    h.add_argument("--table", type=Path, required=True)
    h.add_argument("--measured", type=Path, required=True)
    h.add_argument(
        "--junit",
        type=Path,
        help="the sweep's JUnit report; its failed tests no measurement describes, and "
        "its passed or skipped tests that wrote no measurement, are listed and fail the "
        "comparison (a named file that is missing fails it too)",
    )
    h.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    try:
        report, status = (_diff if args.mode == "diff" else _headroom)(args)
    except ValueError as refused:
        # A table the registry would refuse to load. No verdict on it means anything,
        # and the override label cannot admit one: the fix is the table.
        report, status = (
            f"### SFPU ULP budgets\n\nThe table cannot load: {refused}\n",
            2,
        )
    sys.stdout.write(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

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

Standalone on purpose -- ``yaml`` and the standard library, so the PR check runs on a
slim runner and can parse a *base* revision of the table. ``test_ulp_budget_diff.py``
ties this parse back to the real loader so the two cannot drift.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

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

#: The worst lane a row's comment says the last sweep measured. Written by the emitter
#: on every row it produces ("max 393 ULP, budget 433 > ceiling 25"); a row without one
#: has no baseline and is not judged.
_RECORDED_MAX = re.compile(r"\bmax (\d+) ULP")

#: The lanes a "not measurable" row says disagreed with the golden about being finite.
#: Both spellings: rows emitted before the check also judged a finite answer to an
#: infinite golden read "non-finite against a finite golden".
_RECORDED_NONFINITE = re.compile(
    r"not measurable: (\d+) lane\(s\) (?:non-finite|disagreeing with the golden)"
)


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
    provenance: str
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


def _comment(line: str) -> str:
    """The provenance comment of one line, or ``""``."""
    _, sep, comment = line.partition("#")
    return comment.strip() if sep else ""


def _provenance_by_op(text: str) -> Dict[str, Tuple[str, List[str]]]:
    """Each op's header comment and its row comments, in file order. PyYAML drops
    comments, and they are what says whether a raised budget was re-measured."""
    by_op: Dict[str, Tuple[str, List[str]]] = {}
    op: Optional[str] = None
    for line in text.splitlines():
        head = re.match(r"^([A-Za-z_]\w*):", line)
        if head:
            op = head.group(1)
            by_op.setdefault(op, (_comment(line), []))
        elif op is not None and line.strip().startswith("- "):
            by_op[op][1].append(_comment(line))
    return by_op


def _key_value(field: str, value) -> str:
    """A key field as the registry reads it. YAML 1.1 reads a bare ``Yes``/``No`` as a
    boolean and the registry maps it back to the enum member; keyed as ``"True"`` the
    row would describe a cell that does not exist, and the two readers would disagree
    about which cell a raise landed on. An arch is accepted by enum name or value
    (``WORMHOLE`` or ``wormhole``) and named by the enum name, as a measurement is."""
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    if field == "arch":
        return str(value).upper()
    return str(value)


def _number(value) -> Optional[float]:
    return value if isinstance(value, (int, float)) and value is not True else None


def parse_table(text: str) -> Dict[Cell, Row]:
    """Every row of the table, keyed by the cell it governs: values through PyYAML, so
    anchors and ``<<`` merge keys mean what they mean, and comments through a positional
    scan zipped with the loaded rows (sound while every row is one inline mapping)."""
    loaded = yaml.safe_load(text) or {}
    comments = _provenance_by_op(text)
    rows: Dict[Cell, Row] = {}
    for op, entries in loaded.items():
        if not isinstance(entries, list):
            continue
        header, found = comments.get(op, ("", []))
        # A row split over several lines would slide every comment after it by one;
        # rather than mislabel provenance, drop it for that op and keep the budgets.
        if len(found) != len(entries):
            found = [""] * len(entries)
        for fields, comment in zip(entries, found):
            if not isinstance(fields, dict):
                continue
            key = tuple(
                (k, _key_value(k, fields[k]))
                for k in KEY_FIELDS
                if fields.get(k) is not None
            )
            if (op, key) in rows:
                # `_load_table` refuses two rows of equal specificity, so this table
                # cannot load; keeping the last row silently produced a verdict for it.
                raise ValueError(
                    f"{op}: duplicate row for {_cell_name(op, key)}. The registry "
                    "refuses two rows of equal specificity, so this table cannot load; "
                    "no budget verdict is meaningful for it."
                )
            max_ulp = fields.get("max_ulp")
            if fields.get("metric") == "tolerance":
                max_ulp = None
            rows[(op, key)] = Row(
                op=op,
                key=key,
                max_ulp=max_ulp if isinstance(max_ulp, int) else None,
                # A row's own comment wins; one without inherits the op header's run
                # identity, so updating either registers as a re-measurement.
                provenance=comment or header,
                near_zero_atol=_number(fields.get("near_zero_atol")),
                atol=_number(fields.get("atol")),
                rtol=_number(fields.get("rtol")),
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

    @property
    def _tolerance_row(self) -> Optional[Row]:
        """The row, if it is a tolerance row: the only kind whose comment is a baseline.
        A step budget that does not bind on this arch is not one, and its "max N ULP"
        describes the arch it binds on."""
        return None if self.row is None or self.row.gated else self.row

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


def _held(table: Dict[Cell, Row], op: str, key) -> _Held:
    row = _resolve(table, op, key)
    gated = _gates(row, key)
    declared = (
        None if gated else _resolve(table, op, key, only=lambda r: r.declares_tolerance)
    )
    return _Held(row, gated, declared)


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
    # compare it with the declared atol/rtol, and the nightly's headroom report with the
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
        for key in _variants(base, head, op):
            was, now = _held(base, op, key), _held(head, op, key)
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
    found = _RECORDED_NONFINITE.search(row.provenance)
    return int(found.group(1)) if found else 0


def recorded_max(row: Row) -> Optional[int]:
    """The measurement a row records, or ``None`` if there is none to judge against.

    Only a row that pins both ``in`` and ``out`` is a baseline: the emitter writes every
    row that way, and its figure is the whole-format worst lane the nightly re-measures.
    A hand-written broader row lists a sampled driver's figure, or one per format
    ("max 868220929 ULP Float32 / 13249 Float16_b / ..."), and judging a cell against
    either would report a regression the kernel never had."""
    pinned = dict(row.key)
    if "in" not in pinned or "out" not in pinned:
        return None
    found = _RECORDED_MAX.search(row.provenance)
    return int(found.group(1)) if found else None


def _headroom_line(cell: Cell, worst: int, reference: int, verdict: str) -> str:
    return f"| `{_cell_name(*cell)}` | {worst} | {reference} | {verdict} |"


def render_headroom(
    table: Dict[Cell, Row],
    measured: Dict[Cell, int],
    nonfinite: Optional[Dict[Cell, int]] = None,
) -> Tuple[str, int]:
    """The report, and the regression count the workflow fails on.

    A gated cell is judged against its budget; the sweep already fails one it cannot
    meet, so the value here is which cells have no headroom left or carry slack. A
    tolerance cell has no budget and the sweep passes it whatever it measures, so it is
    judged against the measurement its own row records, and only this report sees it.
    The same holds for a cell the sweep could not measure: an overflow is recorded as a
    lane count, and more such lanes than its row accounts for is a regression too.
    """
    over: List[str] = []
    regressed: List[str] = []
    tight: List[str] = []
    slack: List[str] = []
    unjudged = 0
    for cell, count in sorted((nonfinite or {}).items()):
        row = _resolve(table, *cell)
        if row is None or count <= recorded_nonfinite(row):
            continue
        regressed.append(
            _headroom_line(cell, count, recorded_nonfinite(row), "non-finite lanes")
        )
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
        elif worst > row.max_ulp:
            over.append(_headroom_line(cell, worst, row.max_ulp, "over budget"))
        elif worst == row.max_ulp and row.max_ulp > 0:
            # `0 == 0` is an exact-by-construction op doing what it claims, enrolled so
            # that any drift fails; reporting it as "no headroom" buried the real ones.
            tight.append(_headroom_line(cell, worst, row.max_ulp, "no headroom"))
        elif row.max_ulp >= _SLACK_MIN_BUDGET and worst < _SLACK_FRACTION * row.max_ulp:
            slack.append(_headroom_line(cell, worst, row.max_ulp, "could tighten"))

    out = ["### SFPU ULP sweep vs the declared budgets", ""]
    if not measured:
        return "\n".join(out + ["No measurements recorded.", ""]), 0
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
        ("No headroom left", tight, "budget", False),
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
    if not (over or regressed or tight or slack):
        out += [
            "Every gated cell is inside its budget with headroom to spare, and no "
            "judged tolerance cell is past its recorded measurement.",
            "",
        ]
    return "\n".join(out), len(over) + len(regressed)


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
    rows = [
        json.loads(line)
        for line in args.measured.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report, regressions = render_headroom(
        table, _measured_cells(rows), _nonfinite_cells(rows)
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

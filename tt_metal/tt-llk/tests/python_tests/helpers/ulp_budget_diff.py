# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whether a change to the SFPU accuracy table loosens a gate, and whether the
hardware still fits the gates it declares.

* ``diff``: what a pull request does to the budgets. A budget that goes up, or a row
  that stops being gated, is a *regression* whatever the reason; the table's rule is
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
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

#: The key dimensions of a row, in the order a cell is named in a report.
KEY_FIELDS = ("in", "out", "approx", "dest", "arch")

#: A cell identity: the op plus whichever key dimensions the row pins.
Cell = Tuple[str, Tuple[Tuple[str, str], ...]]

#: Each report section shows this many rows and says how many it withheld: a PR
#: comment has a size limit, and a report that lists every cell is one nobody reads.
_MAX_ROWS = 40

#: Below this fraction of its budget a cell carries slack the sweep cannot justify. Not
#: a failure -- tightening is a deliberate change with its own measurement -- but it is
#: the list someone should work through.
_SLACK_FRACTION = 0.5

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

    @property
    def gated(self) -> bool:
        return self.max_ulp is not None

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
    differently than the base did, since a collapsed row governs several.
    """

    cell: Cell
    kind: str  # raised | floor_widened | ungated | removed | tightened | gated | added
    before: Optional[Row]
    after: Optional[Row]
    variants: int = 1

    #: The kinds that weaken a gate. Everything else is neutral or an improvement.
    REGRESSIONS = frozenset({"raised", "ungated", "removed", "floor_widened"})

    @property
    def is_regression(self) -> bool:
        return self.kind in self.REGRESSIONS

    @property
    def remeasured(self) -> bool:
        """Whether the row's provenance comment changed in the same diff -- the table's
        rule for raising a budget. A raise with the comment untouched is a number edited
        to make a failure go away."""
        if self.before is None or self.after is None:
            return False
        return self.before.provenance != self.after.provenance


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


def _key_value(value) -> str:
    """A key field as the registry reads it. YAML 1.1 reads a bare ``Yes``/``No`` as a
    boolean and the registry maps it back to the enum member; keyed as ``"True"`` the
    row would describe a cell that does not exist, and the two readers would disagree
    about which cell a raise landed on."""
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return str(value)


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
                (k, _key_value(fields[k]))
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
            floor = fields.get("near_zero_atol")
            rows[(op, key)] = Row(
                op=op,
                key=key,
                max_ulp=max_ulp if isinstance(max_ulp, int) else None,
                # A row's own comment wins; one without inherits the op header's run
                # identity, so updating either registers as a re-measurement.
                provenance=comment or header,
                near_zero_atol=floor if isinstance(floor, (int, float)) else None,
            )
    return rows


def _resolve(
    table: Dict[Cell, Row], op: str, key: Tuple[Tuple[str, str], ...]
) -> Optional[Row]:
    """The most specific row of *op* covering *key*, by the registry's own rule."""
    asked = dict(key)
    best: Optional[Row] = None
    for (row_op, row_key), row in table.items():
        if row_op != op or any(asked.get(k) != v for k, v in row_key):
            continue
        if best is None or len(row_key) > len(best.key):
            best = row
    return best


# ─────────────────────────────────────────────────────────────────────────────
# What a pull request does to the gates
# ─────────────────────────────────────────────────────────────────────────────


def _variants(base: Dict[Cell, Row], head: Dict[Cell, Row], op: str):
    """Every query the registry could be asked about *op*, as far as either table can
    tell them apart: each key dimension takes every value some row pins, or is left
    unset -- an unset query dimension matches only a wildcard, as in the registry."""
    values: Dict[str, set] = {k: set() for k in KEY_FIELDS}
    for table in (base, head):
        for row_op, key in table:
            if row_op == op:
                for k, v in key:
                    values[k].add(v)
    axes = [sorted(values[k]) + [None] for k in KEY_FIELDS]
    for combo in itertools.product(*axes):
        yield tuple((k, v) for k, v in zip(KEY_FIELDS, combo) if v is not None)


def _classify(was: Optional[Row], now: Optional[Row]) -> Optional[str]:
    """How the gate one variant resolves to has changed, or ``None`` if it has not."""
    gated_before = was is not None and was.gated
    gated_after = now is not None and now.gated
    if gated_before and not gated_after:
        # Only a loss if it was gating; a tolerance cell that loses its row gates
        # exactly as it did.
        return "removed" if now is None else "ungated"
    if not gated_before and gated_after:
        return "gated"
    if gated_before and gated_after:
        if now.max_ulp > was.max_ulp:
            return "raised"
        if now.floor > was.floor:
            # Before `tightened`: a smaller `max_ulp` with a wider floor rescues more
            # lanes than it fails, and the budget alone would call that an improvement.
            return "floor_widened"
        if now.max_ulp < was.max_ulp:
            return "tightened"
        return None
    # Neither side gates. A new tolerance row is worth a line; a vanished one is not.
    return "added" if was is None and now is not None else None


_KIND_ORDER = (
    "raised",
    "floor_widened",
    "ungated",
    "removed",
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
            was, now = _resolve(base, op, key), _resolve(head, op, key)
            kind = _classify(was, now)
            if kind is None:
                continue
            deciding = now if now is not None else was
            # Grouped by the deciding row and the budgets on each side: a collapsed row
            # replacing N keyed rows of one budget is one line marked xN.
            ident = (
                op,
                kind,
                deciding.key,
                (was.max_ulp, was.floor) if was else None,
                (now.max_ulp, now.floor) if now else None,
            )
            found = grouped.get(ident)
            variants = found.variants + 1 if found else 1
            grouped[ident] = Change((op, deciding.key), kind, was, now, variants)
    return sorted(grouped.values(), key=lambda c: (_KIND_ORDER.index(c.kind), c.cell))


_KIND_TEXT = {
    "raised": "budget raised",
    "floor_widened": "near-zero floor widened",
    "ungated": "gating lost (now tolerance)",
    "removed": "gate lost (no row covers it)",
    "tightened": "budget tightened",
    "gated": "newly gated",
    "added": "new row",
}


def _budget(row: Optional[Row]) -> str:
    if row is None:
        return "—"
    text = str(row.max_ulp) if row.gated else "tolerance"
    if row.near_zero_atol:  # or a widened floor shows the same number on both sides
        text += f" (floor {row.near_zero_atol:g})"
    return text


def _describe(c: Change) -> str:
    row = c.after or c.before
    return row.describe() + (f" ×{c.variants}" if c.variants > 1 else "")


def _capped(lines: List[str], columns: int) -> List[str]:
    """At most ``_MAX_ROWS`` table rows, then one row saying how many were withheld."""
    withheld = len(lines) - _MAX_ROWS
    if withheld <= 0:
        return lines
    return lines[:_MAX_ROWS] + [f"| _… {withheld} more_ |" + " |" * (columns - 1)]


def _change_row(c: Change, with_provenance: bool) -> str:
    cells = f"| `{_describe(c)}` | {_KIND_TEXT[c.kind]} | {_budget(c.before)} | "
    cells += f"{_budget(c.after)} |"
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
            "| cell | change | before | after | re-measured |",
            "| --- | --- | --- | --- | --- |",
            *_capped([_change_row(c, True) for c in regressions], 5),
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
            "| cell | change | before | after |",
            "| --- | --- | --- | --- |",
            *_capped([_change_row(c, False) for c in improvements], 4),
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


def _measured_cells(rows: Iterable[dict]) -> Dict[Cell, int]:
    """The worst lane each variant reached, from ``--ulp-measure`` rows. A driver
    enumerates axes the budget key does not, so several rows land on one cell and the
    worst wins, as in ``ulp_sweep.record``."""
    worst: Dict[Cell, int] = {}
    for row in rows:
        key = tuple((k, str(row[k])) for k in KEY_FIELDS if row.get(k) is not None)
        cell: Cell = (row["op"], key)
        worst[cell] = max(worst.get(cell, 0), int(row["max"]))
    return worst


def _nonfinite_cells(rows: Iterable[dict]) -> Dict[Cell, int]:
    """Per cell, the most lanes any measurement saw go non-finite against a finite
    golden (0 for rows written before the sweep recorded it)."""
    worst: Dict[Cell, int] = {}
    for row in rows:
        key = tuple((k, str(row[k])) for k in KEY_FIELDS if row.get(k) is not None)
        cell: Cell = (row["op"], key)
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
        if not row.gated:
            was = recorded_max(row)
            if was is not None and worst > was:
                regressed.append(_headroom_line(cell, worst, was, "regressed"))
        elif worst > row.max_ulp:
            over.append(_headroom_line(cell, worst, row.max_ulp, "over budget"))
        elif worst == row.max_ulp and row.max_ulp > 0:
            # `0 == 0` is an exact-by-construction op doing what it claims, enrolled so
            # that any drift fails; reporting it as "no headroom" buried the real ones.
            tight.append(_headroom_line(cell, worst, row.max_ulp, "no headroom"))
        elif row.max_ulp > 1 and worst < _SLACK_FRACTION * row.max_ulp:
            slack.append(_headroom_line(cell, worst, row.max_ulp, "could tighten"))

    out = ["### SFPU ULP sweep vs the declared budgets", ""]
    if not measured:
        return "\n".join(out + ["No measurements recorded.", ""]), 0
    out += [f"{len(measured)} cell(s) measured.", ""]
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
            f"| cell | measured | {reference} | |",
            "| --- | --- | --- | --- |",
            *_capped(lines, 4),
            "",
        ]
        if collapsed:
            out += ["</details>", ""]
    if not (over or regressed or tight or slack):
        out += [
            "Every gated cell is inside its budget with headroom to spare, and no "
            "tolerance cell is past its recorded measurement.",
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    report, status = (_diff if args.mode == "diff" else _headroom)(args)
    sys.stdout.write(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whether a change to the SFPU accuracy table loosens a gate, and whether the
hardware still fits the gates it declares.

Two questions, one vocabulary, because both are about the same rows:

* **a diff of the table** -- what a pull request is doing to the budgets. A budget
  that goes up, or a row that stops being gated at all, is a *regression* whatever
  the reason, and the table's own rule is that it may only happen alongside a fresh
  measurement. Needs no hardware, so it runs on every PR that touches the file.
* **a run of the sweep against the table** -- what the hardware actually measures,
  read from the ``--ulp-measure`` rows. The sweep already fails a cell it cannot
  meet; this turns the rest of the run into a headroom report, which is what says a
  budget is about to become a regression before it does.

Deliberately standalone: ``yaml`` and the standard library, no torch, no
``helpers.ulp``. The PR check runs on a slim runner with no LLK environment, and it
has to be able to parse a *base* revision of the table as well as the head one.
``test_ulp_budget_diff.py`` ties this parse back to the real loader so the two
cannot drift.
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

#: What a row can say about the gate itself.
_METRIC = "metric"
_MAX_ULP = "max_ulp"
_NEAR_ZERO_ATOL = "near_zero_atol"

#: A cell identity: the op plus whichever key dimensions the row pins.
Cell = Tuple[str, Tuple[Tuple[str, str], ...]]


@dataclass(frozen=True)
class Row:
    """One table row, reduced to what a regression is defined over."""

    op: str
    key: Tuple[Tuple[str, str], ...]
    max_ulp: Optional[int]
    provenance: str
    #: The absolute-error floor under the budget, where the reference crosses zero.
    #: Part of the contract, not decoration: `ulp_elementwise_valid` accepts a lane
    #: inside it however many steps out it is, so widening the floor loosens the gate
    #: exactly as raising `max_ulp` does.
    near_zero_atol: Optional[float] = None

    @property
    def gated(self) -> bool:
        """Whether this row enforces a step budget at all."""
        return self.max_ulp is not None

    @property
    def floor(self) -> float:
        """The floor as a number, so an absent one compares as no rescue at all."""
        return self.near_zero_atol or 0.0

    def describe(self) -> str:
        if not self.key:
            return f"{self.op} (default)"
        return f"{self.op} {{" + ", ".join(f"{k}: {v}" for k, v in self.key) + "}"


@dataclass(frozen=True)
class Change:
    """One resolved cell whose gate differs between two revisions of the table.

    *cell* names the row that decides the cell after the change (or before it, for a
    cell nothing covers any more), and *variants* is how many query variants that row
    decides differently than the base did -- the sweep's collapsed rows govern several.
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
        """Whether the row's provenance comment changed in the same diff.

        The table's rule, verbatim: "A budget may only be *raised* by re-measuring and
        updating that comment in the same change." A raise whose comment is untouched
        is a number edited to make a failure go away.
        """
        if self.before is None or self.after is None:
            return False
        return self.before.provenance != self.after.provenance


def _strip_comment(line: str) -> Tuple[str, str]:
    """A row line split into its YAML body and its provenance comment."""
    body, sep, comment = line.partition("#")
    return body, comment.strip() if sep else ""


def _provenance_by_op(text: str) -> Dict[str, Tuple[str, List[str]]]:
    """Each op's header comment and its row comments, in file order.

    The comments are the half PyYAML throws away, and they are what says whether a
    raised budget was re-measured. Scanned positionally and zipped with the loaded
    rows below, which is sound only while every row is a single inline mapping --
    checked, not assumed.

    The header is returned too because the table puts the run identity there, once per
    op, and 42 gated rows carry no inline comment at all. Their provenance would
    otherwise be permanently empty, so a raise on one could never register as
    re-measured and the audit would report `no` whatever the author did.
    """
    by_op: Dict[str, Tuple[str, List[str]]] = {}
    op: Optional[str] = None
    for line in text.splitlines():
        head = re.match(r"^([A-Za-z_]\w*):", line)
        if head:
            op = head.group(1)
            by_op.setdefault(op, (_strip_comment(line)[1], []))
            continue
        if op is not None and line.strip().startswith("- "):
            by_op[op][1].append(_strip_comment(line)[1])
    return by_op


def parse_table(text: str) -> Dict[Cell, Row]:
    """Every row of the table, keyed by the cell it governs.

    Values through PyYAML, so anchors, aliases and ``<<`` merge keys mean what they
    mean -- the coarse-LUT tolerance pair is written as an anchor and an alias, and a
    line-oriented reader skipped both. Comments through a positional scan, because
    PyYAML discards them and the provenance is half of what this compares.
    """
    loaded = yaml.safe_load(text) or {}
    comments = _provenance_by_op(text)
    rows: Dict[Cell, Row] = {}
    for op, entries in loaded.items():
        if not isinstance(entries, list):
            continue
        header, found = comments.get(op, ("", []))
        # A row split over several lines would slide every comment after it by one.
        # Rather than mislabel provenance, drop it for that op and keep the budgets.
        aligned = found if len(found) == len(entries) else [""] * len(entries)
        # The op header is the default; a row's own comment overrides it. So a row with
        # no comment inherits the run identity that governs it, and updating either one
        # registers as a re-measurement.
        aligned = [row_comment or header for row_comment in aligned]
        for fields, provenance in zip(entries, aligned):
            if not isinstance(fields, dict):
                continue
            key = tuple(
                (k, str(fields[k]))
                for k in KEY_FIELDS
                if k in fields and fields[k] is not None
            )
            max_ulp = fields.get(_MAX_ULP)
            if fields.get(_METRIC) == "tolerance":
                max_ulp = None
            floor = fields.get(_NEAR_ZERO_ATOL)
            if (op, key) in rows:
                # `_load_table` refuses two rows of equal specificity, so a table with
                # one cannot load at all. Keeping the last silently produced a verdict
                # -- "tightened", even -- for a table the registry rejects.
                raise ValueError(
                    f"{op}: duplicate row for {Row(op, key, None, '').describe()}. "
                    "The registry refuses two rows of equal specificity, so this "
                    "table cannot load; no budget verdict is meaningful for it."
                )
            rows[(op, key)] = Row(
                op=op,
                key=key,
                max_ulp=max_ulp if isinstance(max_ulp, int) else None,
                provenance=provenance,
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
        if row_op != op:
            continue
        if any(asked.get(k) != v for k, v in row_key):
            continue
        if best is None or len(row_key) > len(best.key):
            best = row
    return best


def _variants(base: Dict[Cell, Row], head: Dict[Cell, Row], op: str):
    """Every query the registry could be asked about *op*, as far as either table can
    tell them apart: each key dimension takes every value some row pins, or is left
    unset -- an unset query dimension matches only a wildcard, as in the registry."""
    values: Dict[str, set] = {k: set() for k in KEY_FIELDS}
    for table in (base, head):
        for (row_op, key), _ in table.items():
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
            # Checked before `tightened`, because the two can move opposite ways: a
            # smaller `max_ulp` with a wider floor rescues more lanes than it fails,
            # and reporting only the budget would call that an improvement.
            return "floor_widened"
        if now.max_ulp < was.max_ulp:
            return "tightened"
        return None
    # Neither side gates. A new tolerance row is worth a line; a vanished one is not.
    return "added" if was is None and now is not None else None


def compare(base: Dict[Cell, Row], head: Dict[Cell, Row]) -> List[Change]:
    """Every cell whose *resolved* gate differs, strongest change first.

    Resolved, not row by row: the registry answers a query with its most specific
    matching row, so the table's shape is not the gate. A row the sweep collapses over
    `approx` still governs both approx values, and deleting it is no loss while a
    broader row gives the same budget -- a row diff reported 16 "removed" regressions
    on exactly that. And the reverse hole was worse: a new `{in: Float32, max_ulp:
    1000}` under a `max_ulp: 2` default loosens every Float32 cell without any row
    being "raised", so a row diff called it an improvement.

    Variants that the same pair of rows decides the same way collapse into one
    ``Change``, so the report stays one line per row rather than one per query.
    """
    grouped: Dict[Tuple, Change] = {}
    for op in sorted({c[0] for c in base} | {c[0] for c in head}):
        for key in _variants(base, head, op):
            was, now = _resolve(base, op, key), _resolve(head, op, key)
            kind = _classify(was, now)
            if kind is None:
                continue
            deciding = now if now is not None else was
            # Grouped by the deciding row and the budgets on each side, not by the
            # base row: a collapsed row replacing N keyed rows of one budget is one
            # line marked xN, not N lines saying the same thing.
            ident = (
                op,
                kind,
                deciding.key,
                (was.max_ulp, was.floor) if was else None,
                (now.max_ulp, now.floor) if now else None,
            )
            found = grouped.get(ident)
            if found is None:
                grouped[ident] = Change((op, deciding.key), kind, was, now)
            else:
                grouped[ident] = Change(
                    found.cell, kind, was, now, variants=found.variants + 1
                )
    order = {
        "raised": 0,
        "floor_widened": 1,
        "ungated": 2,
        "removed": 3,
        "tightened": 4,
        "gated": 5,
        "added": 6,
    }
    return sorted(grouped.values(), key=lambda c: (order[c.kind], c.cell))


def _describe(c: Change) -> str:
    row = c.after or c.before
    return row.describe() + (f" ×{c.variants}" if c.variants > 1 else "")


def _budget(row: Optional[Row]) -> str:
    if row is None:
        return "—"
    text = str(row.max_ulp) if row.gated else "tolerance"
    # Without the floor a widened-floor row shows the same number on both sides.
    if row.near_zero_atol:
        text += f" (floor {row.near_zero_atol:g})"
    return text


_KIND_TEXT = {
    "raised": "budget raised",
    "floor_widened": "near-zero floor widened",
    "ungated": "gating lost (now tolerance)",
    "removed": "gate lost (no row covers it)",
    "tightened": "budget tightened",
    "gated": "newly gated",
    "added": "new row",
}


def render_budget_diff(changes: List[Change], label_hint: str) -> str:
    """The PR comment. Regressions first, and what to do about them."""
    regressions = [c for c in changes if c.is_regression]
    if not changes:
        return "### SFPU ULP budgets\n\nNo budget changed.\n"

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
        ]
        for c in regressions[:_MAX_ROWS]:
            mark = "yes" if c.remeasured else "**no**"
            out.append(
                f"| `{_describe(c)}` | {_KIND_TEXT[c.kind]} | {_budget(c.before)} | "
                f"{_budget(c.after)} | {mark} |"
            )
        if len(regressions) > _MAX_ROWS:
            out.append(f"| _… {len(regressions) - _MAX_ROWS} more_ | | | | |")
        out += [
            "",
            "If these are genuine re-measurements, say so in the PR body and add the "
            f"`{label_hint}` label. If they are not, the budget is being fitted to a "
            "failure and the kernel or the golden is what moved.",
            "",
        ]

    improvements = [c for c in changes if not c.is_regression]
    if improvements:
        out += [
            f"<details><summary>{len(improvements)} other budget change(s)</summary>",
            "",
            "| cell | change | before | after |",
            "| --- | --- | --- | --- |",
        ]
        for c in improvements[:_MAX_ROWS]:
            out.append(
                f"| `{_describe(c)}` | {_KIND_TEXT[c.kind]} | {_budget(c.before)} | "
                f"{_budget(c.after)} |"
            )
        if len(improvements) > _MAX_ROWS:
            out.append(f"| _… {len(improvements) - _MAX_ROWS} more_ | | | |")
        out += ["", "</details>", ""]
    return "\n".join(out) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# The other question: what the hardware measured against what the table declares
# ─────────────────────────────────────────────────────────────────────────────

#: Below this fraction of its budget a cell is carrying slack the sweep cannot
#: justify. Not a failure -- tightening is a deliberate change with its own
#: measurement -- but it is the list someone should work through.
_SLACK_FRACTION = 0.5

#: A report that lists every cell is a report nobody reads, and a PR comment has a
#: size limit. Each section shows this many and says how many it withheld.
_MAX_ROWS = 40


def _measured_cells(rows: Iterable[dict]) -> Dict[Cell, int]:
    """The worst lane each variant reached, from ``--ulp-measure`` rows.

    Several rows land on one cell -- a driver enumerates axes the budget key does
    not -- so the worst of them wins, the same rule ``ulp_sweep.record`` applies.
    """
    worst: Dict[Cell, int] = {}
    for row in rows:
        key = tuple((k, str(row[k])) for k in KEY_FIELDS if row.get(k) is not None)
        cell: Cell = (row["op"], key)
        worst[cell] = max(worst.get(cell, 0), int(row["max"]))
    return worst


def render_headroom(
    table: Dict[Cell, Row], measured: Dict[Cell, int]
) -> Tuple[str, int]:
    """What the run says about the gates. Returns the report and the over-budget count.

    The sweep already fails a cell it cannot meet, so an over-budget line here is a
    second voice on a test that is already red. The value is the rest: which cells
    have no headroom left, and which are carrying slack.
    """
    over: List[str] = []
    tight: List[str] = []
    slack: List[str] = []
    for cell, worst in sorted(measured.items()):
        row = _resolve(table, cell[0], cell[1])
        if row is None or not row.gated:
            continue
        named = f"{cell[0]} {{" + ", ".join(f"{k}: {v}" for k, v in cell[1]) + "}"
        if worst > row.max_ulp:
            over.append(f"| `{named}` | {worst} | {row.max_ulp} | over budget |")
        elif worst == row.max_ulp and row.max_ulp > 0:
            # `0 == 0` is excluded deliberately: a 0-step budget measuring 0 is an op
            # that is exact by construction doing exactly what it claims, and the table
            # enrols it precisely so that any drift fails. Reporting those as "no
            # headroom" buried the real ones under 52 lines of nothing on a 130-cell
            # run.
            tight.append(f"| `{named}` | {worst} | {row.max_ulp} | no headroom |")
        elif row.max_ulp > 1 and worst < _SLACK_FRACTION * row.max_ulp:
            slack.append(f"| `{named}` | {worst} | {row.max_ulp} | could tighten |")

    out = ["### SFPU ULP sweep vs the declared budgets", ""]
    if not measured:
        return "\n".join(out + ["No measurements recorded.", ""]), 0
    out.append(f"{len(measured)} cell(s) measured.")
    out.append("")
    for title, lines in (
        ("Over budget", over),
        ("No headroom left", tight),
        (f"Carrying slack (measured under {_SLACK_FRACTION:.0%} of budget)", slack),
    ):
        if not lines:
            continue
        collapse = title.startswith("Carrying")
        if collapse:
            out.append(f"<details><summary>{title} — {len(lines)}</summary>")
            out.append("")
        else:
            out.append(f"**{title} — {len(lines)}**")
            out.append("")
        shown, withheld = lines[:_MAX_ROWS], max(0, len(lines) - _MAX_ROWS)
        out += ["| cell | measured | budget | |", "| --- | --- | --- | --- |"] + shown
        if withheld:
            out.append(f"| _… {withheld} more_ | | | |")
        out.append("")
        if collapse:
            out += ["</details>", ""]
    if not (over or tight or slack):
        out.append("Every gated cell is inside its budget with headroom to spare.")
        out.append("")
    return "\n".join(out), len(over)


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

    if args.mode == "diff":
        changes = compare(
            parse_table(args.base.read_text(encoding="utf-8")),
            parse_table(args.head.read_text(encoding="utf-8")),
        )
        report = render_budget_diff(changes, args.label_hint)
        regressions = [c for c in changes if c.is_regression]
        status = 0 if (not regressions or args.allow_raises) else 1
        if regressions and args.allow_raises:
            # The label is the single override, deliberately: one visible act by a
            # second person, which removing re-runs the check. What it must not do is
            # hide *which* rows it admitted without a fresh measurement, so those are
            # named here and the workflow raises a warning annotation for them.
            unmeasured = [c for c in regressions if not c.remeasured]
            report += (
                f"\n_Allowed: the `{args.label_hint}` label is set on this pull "
                f"request, admitting {len(regressions)} loosened gate(s)._\n"
            )
            if unmeasured:
                report += (
                    f"\n**{len(unmeasured)} of them carry no fresh measurement** -- "
                    "the row's provenance comment is unchanged, so the budget was "
                    "edited rather than re-measured:\n\n"
                    + "".join(
                        f"- `{(c.after or c.before).describe()}`\n"
                        for c in unmeasured[:_MAX_ROWS]
                    )
                )
                if len(unmeasured) > _MAX_ROWS:
                    report += f"- _… {len(unmeasured) - _MAX_ROWS} more_\n"
    else:
        table = parse_table(args.table.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in args.measured.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report, over = render_headroom(table, _measured_cells(rows))
        status = 1 if over else 0

    sys.stdout.write(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One-time migration: the budget table's provenance comments become row fields.

Until P12 a row's measurement lived in the comment beside it (``# max 5 ULP``,
``# max 1 ULP, wormhole, 2026-09-21``) and the run behind it on the op's key line
(``# measured by: <sweep>, wormhole, <date>, except where a row says otherwise``).
This reads that grammar -- P11's ``Provenance.parse``, kept here as :class:`Note` and
nowhere else -- and writes each fact as a field of the row it belongs to::

    - {in: Float16, out: Float16, max_ulp: 6}  # max 5 ULP
    + {in: Float16, out: Float16, max_ulp: 6, measured: 5, run: 2026-10-05}

Each fact is taken exactly as P11's readers took it, and :func:`facts` is the check
that they did: the audit's figure and exhaustive/sampled call, the headroom report's
recorded maximum and non-finite lane count, before and after, for every row. The
comment keeps only what no field holds -- a person's prose, a not-measurable row's
named lanes, the floor a demotion tried.

Run once, from ``python_tests``::

    python3 helpers/migrate_provenance_to_fields.py helpers/sfpu_accuracy_budget.yaml

Deleted in P13, with the grammar it is the last reader of.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__:
    from .ulp_provenance import BudgetTable, Provenance, Row
else:  # run by path, like the guard
    from ulp_provenance import BudgetTable, Provenance, Row

#: The architectures a run identity can name, as ``ChipArchitecture`` values.
ARCHES = ("wormhole", "blackhole", "quasar")

#: The clause a key line carries after the run identity.
_EXCEPT = ", except where a row says otherwise"
_MEASURED_BY = "measured by: "
_NOT_MEASURABLE = "not measurable: "


class Kind(Enum):
    """What wrote a row's comment, and so which fields it carries."""

    EMITTED = "emitted"  # max N ULP
    FLOORED = (
        "floored"  # max N ULP outside a A near-zero floor (M steps over every lane)
    )
    DEMOTED = "demoted"  # max N ULP, budget B > ceiling C
    BLOCK = "block"  # max N ULP, block-quantized
    UNMEASURABLE = "unmeasurable"  # not measurable: ...
    SAMPLED = "sampled"  # max N ULP, <arch>, <date>: a functional driver's sample
    HAND = "hand"  # free prose; only its figure and date are read


#: The kinds the emitter writes. A row of one of these that names no run of its own was
#: measured by the run on its op's key line.
EMITTER_KINDS = frozenset(
    {Kind.EMITTED, Kind.FLOORED, Kind.DEMOTED, Kind.BLOCK, Kind.UNMEASURABLE}
)


def _is_date(text: str) -> bool:
    """``YYYY-MM-DD``."""
    return (
        len(text) == 10
        and text[4] == "-"
        and text[7] == "-"
        and (text[:4] + text[5:7] + text[8:]).isdigit()
    )


def _is_count(text: str) -> bool:
    return text.isdigit() and text.isascii()


def _split_parts(text: str) -> List[Tuple[str, str]]:
    """*text* cut at every ``", "`` and ``"; "``, as ``(separator, part)`` pairs; the
    first separator is ``""``. Joining the pairs back gives *text*."""
    parts: List[Tuple[str, str]] = []
    sep, start, i = "", 0, 0
    while i < len(text) - 1:
        if text[i] in ",;" and text[i + 1] == " ":
            parts.append((sep, text[start:i]))
            sep, start, i = text[i : i + 2], i + 2, i + 2
            continue
        i += 1
    parts.append((sep, text[start:]))
    return parts


@dataclass(frozen=True)
class RunIdentity:
    """Which run measured a row: the sweep's description (``""`` for a functional
    driver's sample), the architecture and the day."""

    arch: str
    date: str
    sweep: str = ""

    def render(self) -> str:
        return ", ".join(part for part in (self.sweep, self.arch, self.date) if part)

    @property
    def exhaustive(self) -> bool:
        """Whether the run saw every value of its inputs: the exhaustive sweep, as
        ``finish_emit`` names it, rather than a sample."""
        return self.sweep.startswith("exhaustive ")

    @classmethod
    def parse(cls, text: str) -> Optional["RunIdentity"]:
        """*text* as a run identity, or ``None`` if it is not one exactly."""
        parts = text.split(", ")
        if len(parts) == 2:
            sweep, (arch, date) = "", parts
        elif len(parts) == 3:
            sweep, arch, date = parts
        else:
            return None
        if (
            arch not in ARCHES
            or not _is_date(date)
            or ", " in sweep
            or not (sweep or len(parts) == 2)
        ):
            return None
        return cls(arch=arch, date=date, sweep=sweep)


def _take_run(
    parts: List[Tuple[str, str]],
) -> Tuple[Optional[RunIdentity], List[Tuple[str, str]]]:
    """A run identity at the head of *parts* (each joined by ``", "``), and the parts
    after it."""
    for width in (3, 2):
        if len(parts) < width or any(sep != ", " for sep, _ in parts[:width]):
            continue
        run = RunIdentity.parse(", ".join(part for _, part in parts[:width]))
        if run is not None:
            return run, parts[width:]
    return None, parts


def _join(parts: List[Tuple[str, str]]) -> str:
    return "".join(sep + part for sep, part in parts)


def _figure_before_ulp(words: List[str], need_max: bool) -> Optional[int]:
    """The first ``N ULP`` (``max N ULP`` with *need_max*) among *words*."""
    for i in range(1, len(words)):
        if words[i].rstrip(",;):") != "ULP" or not _is_count(words[i - 1]):
            continue
        if need_max and (i < 2 or words[i - 2] != "max"):
            continue
        return int(words[i - 1])
    return None


def _floor_text(atol: float) -> str:
    return f"{atol:.2e}"


@dataclass(frozen=True)
class Note:
    """Everything a row's comment says that code reads.

    *measured* is the figure the budget was derived from: the worst lane, or for a
    floored row the worst lane outside the floor. A hand-written comment keeps its
    first ``N ULP`` figure here, and *recorded_max* its first ``max N ULP`` -- the two
    readers of free prose disagreed on bare ``0 ULP, 115 variants`` notes, and both
    readings are kept until those rows carry the figure as a field.
    """

    kind: Kind
    measured: Optional[int] = None
    #: A floored row's maximum over every lane.
    measured_all: Optional[int] = None
    #: A demoted row: the budget the measurement would have needed, and the ceiling.
    budget_needed: Optional[int] = None
    ceiling: Optional[int] = None
    #: The near-zero floor a floored row carries, or a demoted row tried.
    floor: Optional[float] = None
    #: A demoted row that tried a floor: the worst lane outside it.
    residual: Optional[int] = None
    #: A not-measurable row: lanes disagreeing with the golden about being finite, and
    #: the measurable lanes its *measured* was taken over.
    nonfinite_lanes: Optional[int] = None
    lanes: Optional[int] = None
    #: The run the row names itself; ``None`` means its op's key line's.
    run: Optional[RunIdentity] = None
    #: Text no field holds: a hand-written comment whole, a not-measurable row's
    #: reason, or what a person appended after a machine-written note (with its
    #: separator).
    prose: str = ""
    #: A hand-written comment's first ``max N ULP``, if any.
    hand_max: Optional[int] = None
    #: Whether a hand-written comment names a date anywhere, and the run identity it
    #: spells out, if it spells one out whole.
    hand_dated: bool = False
    hand_run: Optional[RunIdentity] = None

    # ── Reading ──────────────────────────────────────────────────────────────

    @property
    def recorded_max(self) -> Optional[int]:
        """The worst lane the comment records as ``max N ULP``, or ``None``."""
        if self.kind is Kind.HAND:
            return self.hand_max
        return self.measured

    @property
    def names_its_run(self) -> bool:
        """Whether the comment says which run measured it, so the key line's does not
        apply. A hand-written comment does by naming a day."""
        if self.kind is Kind.HAND:
            return self.hand_dated
        if self.kind is Kind.UNMEASURABLE and self.run is None:
            # A reason settled by hand may name the day it was settled.
            return any(_is_date(w.strip(",;():")) for w in self.prose.split())
        return self.run is not None

    def run_is_exhaustive(self, key_line: "KeyLine") -> bool:
        """Whether the run behind this row saw every value: its own, when it names one,
        and its op's key line's otherwise."""
        if self.kind is Kind.HAND:
            if self.hand_dated:
                return self.hand_run is not None and self.hand_run.exhaustive
            return key_line.exhaustive
        if self.run is not None:
            return self.run.exhaustive
        return key_line.exhaustive

    # ── The grammar ──────────────────────────────────────────────────────────

    def render(self) -> str:
        """The comment, without the ``# ``. The only place the grammar is written."""
        if self.kind is Kind.HAND:
            return self.prose
        if self.kind is Kind.UNMEASURABLE:
            core = _NOT_MEASURABLE
            if self.nonfinite_lanes is not None:
                core += f"{self.nonfinite_lanes} lane(s) "
            core += self.prose
            if self.lanes is not None:
                core += (
                    f"; max {self.measured} ULP over the {self.lanes} measurable lanes"
                )
            return core + (f", {self.run.render()}" if self.run else "")
        core = f"max {self.measured} ULP"
        if self.kind is Kind.FLOORED:
            core += (
                f" outside a {_floor_text(self.floor)} near-zero floor "
                f"({self.measured_all} steps over every lane)"
            )
        elif self.kind is Kind.DEMOTED:
            core += f", budget {self.budget_needed} > ceiling {self.ceiling}"
            if self.floor is not None:
                core += (
                    f"; {self.residual} ULP outside a {_floor_text(self.floor)} "
                    "near-zero floor"
                )
        elif self.kind is Kind.BLOCK:
            core += ", block-quantized"
        if self.run is not None:
            core += f", {self.run.render()}"
        return core + self.prose

    @classmethod
    def parse(cls, comment: str) -> "Note":
        """*comment* (without the ``# ``) as data. The only place the grammar is read.

        A comment is machine-written only if :meth:`render` gives it back exactly;
        anything else is :attr:`Kind.HAND`, carrying its text and the figure and date
        a person wrote into it."""
        comment = comment.strip()
        for parsed in (cls._parse_unmeasurable(comment), cls._parse_measured(comment)):
            if parsed is not None and parsed.render() == comment:
                return parsed
        return cls.hand(comment)

    @classmethod
    def hand(cls, text: str) -> "Note":
        words = text.split()
        parts = _split_parts(text)
        hand_run = None
        for start in range(len(parts)):
            hand_run, _ = _take_run([(", ", p) for _, p in parts[start:]])
            if hand_run is not None:
                break
        return cls(
            kind=Kind.HAND,
            prose=text,
            measured=_figure_before_ulp(words, need_max=False),
            hand_max=_figure_before_ulp(words, need_max=True),
            hand_dated=any(_is_date(w.strip(",;():")) for w in words),
            hand_run=hand_run,
        )

    @classmethod
    def _parse_unmeasurable(cls, comment: str) -> Optional["Note"]:
        if not comment.startswith(_NOT_MEASURABLE):
            return None
        reason = comment[len(_NOT_MEASURABLE) :]
        # A run identity a re-emit stamped on is the last two or three parts.
        run, parts = None, reason.split(", ")
        for width in (3, 2):
            if len(parts) > width:
                run = RunIdentity.parse(", ".join(parts[-width:]))
                if run is not None:
                    reason = ", ".join(parts[:-width])
                    break
        nonfinite = None
        count, sep, rest = reason.partition(" lane(s) ")
        if sep and _is_count(count):
            nonfinite, reason = int(count), rest
        measured = lanes = None
        body, sep, tail = reason.rpartition("; max ")
        if sep:
            words = tail.split(" ")
            if (
                len(words) == 7
                and _is_count(words[0])
                and words[1:4] == ["ULP", "over", "the"]
                and _is_count(words[4])
                and words[5:] == ["measurable", "lanes"]
            ):
                measured, lanes, reason = int(words[0]), int(words[4]), body
        if lanes is None:
            # A reason settled by hand names its figure wherever it likes.
            measured = _figure_before_ulp(reason.split(), need_max=True)
        return cls(
            kind=Kind.UNMEASURABLE,
            measured=measured,
            lanes=lanes,
            nonfinite_lanes=nonfinite,
            prose=reason,
            run=run,
        )

    @classmethod
    def _parse_measured(cls, comment: str) -> Optional["Note"]:
        parts = _split_parts(comment)
        words = parts[0][1].split(" ")
        if len(words) < 3 or words[0] != "max" or not _is_count(words[1]):
            return None
        if words[2] != "ULP":
            return None
        measured = int(words[1])
        fields: Dict[str, Any] = {"measured": measured}
        rest = parts[1:]
        if len(words) == 3:
            kind = Kind.EMITTED
            if rest and rest[0] == (", ", "block-quantized"):
                kind, rest = Kind.BLOCK, rest[1:]
            elif rest and rest[0][0] == ", " and rest[0][1].startswith("budget "):
                demotion = rest[0][1].split(" ")
                if len(demotion) != 5 or demotion[2:4] != [">", "ceiling"]:
                    return None
                if not (_is_count(demotion[1]) and _is_count(demotion[4])):
                    return None
                kind = Kind.DEMOTED
                fields.update(budget_needed=int(demotion[1]), ceiling=int(demotion[4]))
                rest = rest[1:]
                if rest and rest[0][0] == "; ":
                    tried = rest[0][1].split(" ")
                    if (
                        len(tried) == 7
                        and _is_count(tried[0])
                        and tried[1:4] == ["ULP", "outside", "a"]
                        and tried[5:] == ["near-zero", "floor"]
                    ):
                        floor = _float(tried[4])
                        if floor is None:
                            return None
                        fields.update(residual=int(tried[0]), floor=floor)
                        rest = rest[1:]
        elif words[3:5] == ["outside", "a"]:
            # max R ULP outside a A near-zero floor (M steps over every lane)
            if (
                len(words) != 13
                or words[6:8] != ["near-zero", "floor"]
                or not words[8].startswith("(")
                or words[9:] != ["steps", "over", "every", "lane)"]
            ):
                return None
            floor, every = _float(words[5]), words[8][1:]
            if floor is None or not _is_count(every):
                return None
            kind = Kind.FLOORED
            fields.update(floor=floor, measured_all=int(every))
        else:
            return None
        run, rest = _take_run(rest)
        if rest and run is None:
            # Text after the note is a person's, and is only admitted after a run
            # identity: `max 65536 ULP, 5 variants / 20480 lanes, 2026-09-18` is a
            # sample described by hand, not an emitted figure.
            return None
        if kind is Kind.EMITTED and run is not None and not run.sweep:
            kind = Kind.SAMPLED
        return cls(kind=kind, run=run, prose=_join(rest), **fields)

    # ── Building ─────────────────────────────────────────────────────────────

    def with_run(self, run: RunIdentity) -> "Note":
        """This provenance credited to *run*."""
        return replace(self, run=run)


def _float(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class KeyLine:
    """An op's key line: the header prose a person wrote, and the run that measured
    the op's rows unless a row names its own."""

    header: str = ""
    run: Optional[RunIdentity] = None

    def render(self) -> str:
        if self.run is None:
            return self.header
        clause = f"{_MEASURED_BY}{self.run.render()}{_EXCEPT}"
        return f"{self.header}; {clause}" if self.header else clause

    @classmethod
    def parse(cls, comment: str) -> "KeyLine":
        comment = comment.strip()
        if comment.endswith(_EXCEPT):
            header, sep, run_text = comment[: -len(_EXCEPT)].rpartition(_MEASURED_BY)
            run = RunIdentity.parse(run_text) if sep else None
            if run is not None:
                parsed = cls(header=header.rstrip().rstrip(";").rstrip(), run=run)
                if parsed.render() == comment:
                    return parsed
        return cls(header=comment)

    @property
    def header_note(self) -> Note:
        """The header read as a hand-written note: what a row without a comment of its
        own, or one naming no figure, falls back to."""
        return Note.hand(self.header)

    @property
    def exhaustive(self) -> bool:
        return self.run is not None and self.run.exhaustive

    def with_run(self, run: RunIdentity) -> "KeyLine":
        return replace(self, run=run)


# ─────────────────────────────────────────────────────────────────────────────
# The migration
# ─────────────────────────────────────────────────────────────────────────────


def _first_date(text: str) -> Optional[date]:
    for word in text.split():
        word = word.strip(",;():")
        if _is_date(word):
            return date.fromisoformat(word)
    return None


@dataclass(frozen=True)
class Facts:
    """What P11's readers took from one row: the audit's figure and exhaustive call
    (step-budget rows), the headroom report's baseline (tolerance rows pinning ``in``
    and ``out``) and non-finite lane count, and whether the not-measurable audit
    counted the row."""

    measured: Optional[int]
    exhaustive: Optional[bool]
    recorded_max: Optional[int]
    nonfinite: int
    not_measurable: bool


def legacy_facts(row: Row, key_line: KeyLine) -> Facts:
    """The facts as P11 read them from the comments."""
    note = Note.parse(row.comment) if row.comment else None
    gated = row.max_ulp is not None
    pinned = "in" in row.pinned and "out" in row.pinned
    measured = exhaustive = None
    if gated:
        has_own = note is not None and note.measured is not None
        measured = note.measured if has_own else key_line.header_note.measured
        exhaustive = note.run_is_exhaustive(key_line) if note else key_line.exhaustive
    recorded = None
    if pinned and not gated:
        recorded = (note or key_line.header_note).recorded_max
    unmeasurable = note is not None and note.kind is Kind.UNMEASURABLE
    return Facts(
        measured=measured,
        exhaustive=exhaustive,
        recorded_max=recorded,
        nonfinite=(note.nonfinite_lanes or 0) if unmeasurable else 0,
        not_measurable=unmeasurable,
    )


def field_facts(row: Row) -> Facts:
    """The same facts as P12's readers take them, from the row's fields alone."""
    p = row.provenance
    gated = row.max_ulp is not None
    pinned = "in" in row.pinned and "out" in row.pinned
    return Facts(
        measured=p.measured if gated else None,
        exhaustive=p.exhaustive if gated else None,
        recorded_max=p.measured if pinned and not gated else None,
        nonfinite=p.nonfinite or 0,
        not_measurable=p.unmeasurable,
    )


def _tail(prose: str) -> str:
    """A person's text after a machine-written note, without its separator."""
    for sep in ("; ", ", "):
        if prose.startswith(sep):
            return prose[len(sep) :]
    return prose


def migrated(row: Row, key_line: KeyLine) -> Tuple[Provenance, str]:
    """*row*'s provenance fields, and the comment it keeps."""
    note = Note.parse(row.comment) if row.comment else None
    if note is not None and note.kind is not Kind.HAND:
        run = note.run or key_line.run
        if run is None:
            raise ValueError(f"{row.describe()}: a machine-written note with no run")
        if run.sweep and not run.exhaustive:
            raise ValueError(f"{row.describe()}: run {run.render()!r} is not the sweep")
        when = date.fromisoformat(run.date)
        provenance = Provenance(
            measured=note.measured,
            measured_all=note.measured_all if note.kind is Kind.FLOORED else None,
            nonfinite=note.nonfinite_lanes if note.kind is Kind.UNMEASURABLE else None,
            run=when if run.sweep else None,
            sampled=None if run.sweep else when,
        )
        if note.kind is Kind.UNMEASURABLE:
            return provenance, f"not measurable: lanes {note.prose}"
        comment = _tail(note.prose)
        if note.kind is Kind.DEMOTED and note.floor is not None:
            tried = (
                f"{note.residual} ULP outside a {_floor_text(note.floor)} "
                "near-zero floor"
            )
            comment = f"{tried}; {comment}" if comment else tried
        return provenance, comment
    # Hand-written, or no comment: the figure and the day a person wrote, read the way
    # the audit (step-budget rows) and the headroom report (tolerance rows) read them.
    gated = row.max_ulp is not None
    header = key_line.header_note
    if gated:
        has_own = note is not None and note.measured is not None
        source = note if has_own else header
        measured = source.measured
        exhaustive = note.run_is_exhaustive(key_line) if note else key_line.exhaustive
    else:
        source = note or header
        measured = source.recorded_max
        exhaustive = False
    if measured is None:
        return Provenance(), row.comment
    if exhaustive:
        own = note.hand_run if note is not None and note.hand_dated else None
        when = date.fromisoformat((own or key_line.run).date)
        return Provenance(measured=measured, run=when), row.comment
    when = _first_date(source.prose)
    if when is None:
        raise ValueError(f"{row.describe()}: a sampled figure with no date")
    return Provenance(measured=measured, sampled=when), row.comment


def migrate(text: str) -> str:
    """*text* with every row's provenance moved from its comment into its fields, and
    every key line's run clause dropped (its rows now name their run)."""
    table = BudgetTable(text)
    done = [r.describe() for r in table.rows if r.provenance.as_fields()]
    if done:
        raise ValueError(
            f"{len(done)} row(s) already carry provenance fields (first: {done[0]}); "
            "the table is migrated"
        )
    lines = list(table.lines)
    for block in table.blocks.values():
        key_line = KeyLine.parse(block.header)
        lines[block.line] = f"{block.op}:" + (
            f"  # {key_line.header}\n" if key_line.header else "\n"
        )
        for row in block.rows:
            provenance, comment = migrated(row, key_line)
            added = provenance.as_fields()
            if not added and comment == row.comment:
                continue
            if row.alias or row.first_line != row.last_line:
                raise ValueError(f"{row.describe()}: not a one-line inline row")
            body = lines[row.last_line][: row.end_column].rstrip()
            if not body.endswith("}"):
                raise ValueError(f"{row.describe()}: not a flow mapping")
            if added:
                extra = ", ".join(f"{k}: {v}" for k, v in added.items())
                body = f"{body[:-1]}, {extra}}}"
            lines[row.last_line] = body + (f"  # {comment}" if comment else "") + "\n"
    return "".join(lines)


def check(before: str, after: str) -> List[str]:
    """Every row whose facts or contract differ between the two, as messages; ``[]``
    if none."""
    old, new = BudgetTable(before), BudgetTable(after)
    problems = []
    if list(old.blocks) != list(new.blocks):
        problems.append("the ops differ")
    for op, block in old.blocks.items():
        key_line = KeyLine.parse(block.header)
        rows = new.rows_of(op)
        if len(rows) != len(block.rows):
            problems.append(f"{op}: {len(block.rows)} rows became {len(rows)}")
            continue
        for was, now in zip(block.rows, rows):
            kept = {
                k: v for k, v in now.fields if k not in Provenance.__dataclass_fields__
            }
            if dict(was.fields) != kept:
                problems.append(f"{was.describe()}: a contract or key field moved")
            if legacy_facts(was, key_line) != field_facts(now):
                problems.append(
                    f"{was.describe()}: {legacy_facts(was, key_line)} became "
                    f"{field_facts(now)}"
                )
    return problems


def main(argv: List[str]) -> int:
    path = Path(argv[0])
    before = path.read_text(encoding="utf-8")
    after = migrate(before)
    problems = check(before, after)
    if problems:
        sys.stderr.write("\n".join(problems) + "\n")
        return 1
    path.write_text(after, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

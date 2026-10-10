# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The SFPU accuracy table's provenance, as data.

Each row of ``sfpu_accuracy_budget.yaml`` records the measurement its budget came from
in the comment beside it, and each op's key line records the run that measured its
rows. PyYAML drops comments, so before this module four readers -- the emitter's
``_stamp_kept``, the live-table audits, the headroom report and the PR budget guard --
each scanned the text with a regex of its own, and each re-implemented a walk over the
file's lines. A wording change in the emitter turned one of those guards vacuous
without failing anything.

Here the comment grammar is written in one place, :meth:`Provenance.render`, and read
in one place, :meth:`Provenance.parse`. ``parse`` accepts a comment as machine-written
only when rendering what it read gives the comment back character for character, so a
reworded note can never be half-understood: it reads as :attr:`Kind.HAND`, and the
audits that expect a machine-written kind say so. :class:`BudgetTable` is the one walk
over the file, locating rows through PyYAML's own event stream rather than by the shape
of a line.

Standalone on purpose -- ``yaml`` and the standard library -- because the PR budget
guard (``ulp_budget_diff.py``) runs on a slim runner and imports it by path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

#: The key dimensions of a row, in the order a cell is named.
KEY_FIELDS = ("in", "out", "approx", "dest", "arch")

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
class Provenance:
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
    def parse(cls, comment: str) -> "Provenance":
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
    def hand(cls, text: str) -> "Provenance":
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
    def _parse_unmeasurable(cls, comment: str) -> Optional["Provenance"]:
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
    def _parse_measured(cls, comment: str) -> Optional["Provenance"]:
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

    def with_run(self, run: RunIdentity) -> "Provenance":
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
    def header_provenance(self) -> Provenance:
        """The header read as a hand-written note: what a row without a comment of its
        own, or one naming no figure, falls back to."""
        return Provenance.hand(self.header)

    @property
    def exhaustive(self) -> bool:
        return self.run is not None and self.run.exhaustive

    def with_run(self, run: RunIdentity) -> "KeyLine":
        return replace(self, run=run)


# ─────────────────────────────────────────────────────────────────────────────
# The table
# ─────────────────────────────────────────────────────────────────────────────


def key_value(value: Any) -> str:
    """A key field as the registry reads it. YAML 1.1 reads a bare ``Yes``/``No`` as a
    boolean, and the registry maps it back to the enum member."""
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return str(value)


@dataclass(frozen=True)
class Row:
    """One row of the table: its fields as PyYAML loads them, and its comment as data.

    *first_line*/*last_line* are 0-based indices into the file, for rewriting it and
    for messages."""

    op: str
    fields: Tuple[Tuple[str, Any], ...]
    provenance: Optional[Provenance]
    first_line: int
    last_line: int
    #: Where the row's node ends on *last_line*; its comment, if any, follows.
    end_column: int
    #: A ``- *anchor`` row: its fields are not inline, so it pins nothing a reader of
    #: the line could see.
    alias: bool = False

    @property
    def values(self) -> Dict[str, Any]:
        return dict(self.fields)

    @property
    def key(self) -> Tuple[Tuple[str, str], ...]:
        """The key dimensions the row pins, as ``(field, value)`` in KEY_FIELDS order."""
        values = self.values
        return tuple(
            (k, key_value(values[k])) for k in KEY_FIELDS if values.get(k) is not None
        )

    @property
    def pinned(self) -> Dict[str, str]:
        return dict(self.key)

    @property
    def metric(self) -> str:
        return str(self.values.get("metric", "ulp"))

    @property
    def max_ulp(self) -> Optional[int]:
        """The step budget, or ``None`` on the tolerance metric."""
        value = self.values.get("max_ulp")
        if self.metric == "tolerance" or not isinstance(value, int):
            return None
        return value

    @property
    def near_zero_atol(self) -> Optional[float]:
        value = self.values.get("near_zero_atol")
        return value if isinstance(value, (int, float)) else None

    def describe(self) -> str:
        pinned = ", ".join(f"{k}: {v}" for k, v in self.key)
        return f"{self.op} {{{pinned}}}" if pinned else f"{self.op} (default)"


@dataclass(frozen=True)
class OpBlock:
    """Where an op sits in the file: its key line and its rows."""

    op: str
    key_line: KeyLine
    line: int
    rows: Tuple[Row, ...]


class BudgetTable:
    """The table as rows and key lines, located through PyYAML's event stream: values
    mean what the loader says they mean (anchors, ``<<`` merges, YAML booleans), and a
    comment is the text after ``#`` past the end of the node it follows. Tolerant of a
    top-level entry that is not a list of rows -- the guard parses arbitrary base
    revisions; ``_load_table`` is what refuses them."""

    def __init__(self, text: str, path: Optional[Path] = None):
        self.text = text
        self.path = path
        self.lines = text.splitlines(keepends=True)
        self.blocks: Dict[str, OpBlock] = {}
        self._parse()

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BudgetTable":
        path = Path(path)
        return cls(path.read_text(encoding="utf-8"), path)

    # ── Queries ──────────────────────────────────────────────────────────────

    @property
    def rows(self) -> List[Row]:
        return [row for block in self.blocks.values() for row in block.rows]

    @property
    def key_lines(self) -> Dict[str, KeyLine]:
        return {op: block.key_line for op, block in self.blocks.items()}

    def rows_of(self, op: str) -> List[Row]:
        block = self.blocks.get(op)
        return list(block.rows) if block else []

    def row(self, op: str, **key: str) -> Row:
        """The one row of *op* pinning exactly *key*. Key fields are spelt as in the
        table, with ``in_`` for ``in``: ``row("Gelu", in_="Float16", out="Float16")``.
        """
        wanted = tuple(
            (k, key[name])
            for k, name in zip(KEY_FIELDS, ("in_", "out", "approx", "dest", "arch"))
            if name in key
        )
        unknown = set(key) - {"in_", "out", "approx", "dest", "arch"}
        if unknown:
            raise TypeError(f"not key fields: {sorted(unknown)}")
        found = [r for r in self.rows_of(op) if r.key == wanted]
        if len(found) != 1:
            raise LookupError(f"{op}: {len(found)} rows pin {dict(wanted)}, not one")
        return found[0]

    def effective_run(self, row: Row) -> Optional[RunIdentity]:
        """The run that measured *row*: its own, or its op's key line's."""
        if row.provenance is not None and row.provenance.run is not None:
            return row.provenance.run
        return self.blocks[row.op].key_line.run

    # ── Parsing ──────────────────────────────────────────────────────────────

    def _comment_after(self, mark) -> str:
        """The comment on *mark*'s line past *mark*, without the ``#``; ``""`` if none."""
        line = self.lines[mark.line] if mark.line < len(self.lines) else ""
        _, sep, comment = line[mark.column :].partition("#")
        return comment.strip() if sep else ""

    def _parse(self) -> None:
        loaded = yaml.safe_load(self.text) or {}
        if not isinstance(loaded, dict):
            return
        events = list(yaml.parse(self.text, Loader=yaml.SafeLoader))
        i = 0
        # Down to the top-level mapping.
        while i < len(events) and not isinstance(events[i], yaml.MappingStartEvent):
            i += 1
        i += 1
        while i < len(events) and not isinstance(events[i], yaml.MappingEndEvent):
            key_event = events[i]
            i += 1
            if not isinstance(key_event, yaml.ScalarEvent):
                i = _skip_node(events, i - 1)
                i = _skip_node(events, i)
                continue
            op = key_event.value
            positions: List[Tuple[Any, Any, bool]] = []
            if isinstance(events[i], yaml.SequenceStartEvent):
                i += 1
                while not isinstance(events[i], yaml.SequenceEndEvent):
                    start = events[i]
                    end_index = _skip_node(events, i)
                    end = events[end_index - 1]
                    positions.append(
                        (
                            start.start_mark,
                            end.end_mark,
                            isinstance(start, yaml.AliasEvent),
                        )
                    )
                    i = end_index
                i += 1
            else:
                i = _skip_node(events, i)
            entries = loaded.get(op)
            rows: List[Row] = []
            if isinstance(entries, list) and len(entries) == len(positions):
                for fields, (start, end, alias) in zip(entries, positions):
                    if not isinstance(fields, dict):
                        continue
                    comment = self._comment_after(end)
                    rows.append(
                        Row(
                            op=op,
                            fields=tuple(fields.items()),
                            provenance=Provenance.parse(comment) if comment else None,
                            first_line=start.line,
                            last_line=end.line,
                            end_column=end.column,
                            alias=alias,
                        )
                    )
            self.blocks[op] = OpBlock(
                op=op,
                key_line=KeyLine.parse(self._comment_after(key_event.end_mark)),
                line=key_event.start_mark.line,
                rows=tuple(rows),
            )


def _skip_node(events: list, i: int) -> int:
    """The index just past the node starting at *events[i]*."""
    depth = 0
    while True:
        event = events[i]
        i += 1
        if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
            depth += 1
        elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
            depth -= 1
        if depth == 0:
            return i


def render_row(row_fields: Dict[str, Any], provenance: Provenance) -> str:
    """One inline row as the emitter writes it, with its newline. Key fields first in
    KEY_FIELDS order, ``approx``/``dest`` quoted so YAML keeps them strings."""
    body = ", ".join(
        f'{k}: "{v}"' if k in ("approx", "dest") else f"{k}: {v}"
        for k, v in row_fields.items()
    )
    return f"  - {{{body}}}  # {provenance.render()}\n"

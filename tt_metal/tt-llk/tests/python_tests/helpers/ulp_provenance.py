# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The SFPU accuracy table as rows, and each row's provenance as fields.

Every row of ``sfpu_accuracy_budget.yaml`` that carries a budget says which measurement
it came from in fields of its own -- ``measured``, ``measured_all``, ``nonfinite``, and
``run`` or ``sampled`` -- which PyYAML loads and the registry loader validates like any
other field. The comment beside a row is prose for people; no code reads it.

Until P12 those facts lived in the comments, and four readers each parsed them with a
regex of their own; a wording change in the emitter turned one of those guards vacuous
without failing anything. :class:`Provenance` is the typed view of the fields, and
:class:`BudgetTable` is the one walk over the file, locating rows through PyYAML's own
event stream rather than by the shape of a line.

Standalone on purpose -- ``yaml`` and the standard library -- because the PR budget
guard (``ulp_budget_diff.py``) runs on a slim runner and imports it by path.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

#: The key dimensions of a row, in the order a cell is named.
KEY_FIELDS = ("in", "out", "approx", "dest", "arch")

#: The provenance fields of a row, in the order the emitter writes them.
PROVENANCE_FIELDS = ("measured", "measured_all", "nonfinite", "run", "sampled")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Provenance:
    """The measurement behind a row, as the row's own fields.

    * ``measured`` -- the worst lane the budget was derived from: over every ranked
      lane, or for a row with a ``near_zero_atol`` floor, over the lanes outside it. On a
      row the sweep could not measure, the worst of the lanes it could.
    * ``measured_all`` -- a floored row's worst lane over every lane, floor included.
    * ``nonfinite`` -- lanes disagreeing with the golden about being finite, which hold
      the row's cell on tolerance; the headroom report fails a run in which it grows.
    * ``run`` -- the day the exhaustive sweep measured the row, on ``MEASURED_ARCH`` (or
      the row's own ``arch``).
    * ``sampled`` -- the day a functional driver's sample measured it instead. A finite
      sample cannot assert what the sweep can: an exhaustive row carries exactly the
      emitter's budget, a sampled one may sit above its figure.
    """

    measured: Optional[int] = None
    measured_all: Optional[int] = None
    nonfinite: Optional[int] = None
    run: Optional[date] = None
    sampled: Optional[date] = None

    @classmethod
    def of(cls, values: Dict[str, Any]) -> "Provenance":
        """The provenance fields of a loaded row, unchecked: a reader of an arbitrary
        revision takes what is there, and :meth:`problems` is the loader's check."""
        return cls(**{k: values[k] for k in PROVENANCE_FIELDS if k in values})

    @property
    def exhaustive(self) -> bool:
        """Whether the run behind the row saw every value of its input."""
        return self.run is not None

    @property
    def unmeasurable(self) -> bool:
        """Whether the sweep could not measure the row's cell: lanes disagreed with the
        golden about being finite, or it ran and no lane could be ranked."""
        return self.nonfinite is not None or (
            self.run is not None and self.measured is None
        )

    @property
    def when(self) -> Optional[date]:
        return self.run or self.sampled

    def as_fields(self) -> Dict[str, Any]:
        """The fields to write, in table order, without the unset ones."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

    def problems(self) -> List[str]:
        """What is wrong with these fields on their own, as messages; ``[]`` if
        nothing. Combinations with the contract are the loader's to check."""
        found = []
        for name, least in (("measured", 0), ("measured_all", 0), ("nonfinite", 1)):
            value = getattr(self, name)
            if value is not None and not (_is_int(value) and value >= least):
                found.append(f"`{name}` must be an integer >= {least}, got {value!r}")
        for name in ("run", "sampled"):
            value = getattr(self, name)
            # A datetime is a date too; a row names a day.
            if value is not None and (type(value) is not date):
                found.append(
                    f"`{name}` must be a date (YYYY-MM-DD, unquoted), got {value!r}"
                )
        if found:
            return found
        if self.run is not None and self.sampled is not None:
            found.append("a row is measured by the sweep (`run`) or sampled, not both")
        if self.measured is not None and self.when is None:
            found.append("`measured` needs the `run` or `sampled` date it came from")
        if self.sampled is not None and self.measured is None:
            found.append("`sampled` without the `measured` figure it records")
        if self.measured_all is not None and (
            self.measured is None or self.measured_all < self.measured
        ):
            found.append(
                "`measured_all` is the worst lane over every lane, so it needs "
                "`measured` and cannot be below it"
            )
        return found


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
    """One row of the table: its fields as PyYAML loads them, and where it sits.

    *first_line*/*last_line* are 0-based indices into the file, for rewriting it and
    for messages; the row's node ends at *end_column* of *last_line*."""

    op: str
    fields: Tuple[Tuple[str, Any], ...]
    first_line: int
    last_line: int
    end_column: int
    #: The prose after the row's ``#``, for people. No code reads it.
    comment: str = ""
    #: A ``- *anchor`` row: its fields are not inline, so it pins nothing a reader of
    #: the line could see.
    alias: bool = False

    @property
    def values(self) -> Dict[str, Any]:
        return dict(self.fields)

    @property
    def provenance(self) -> Provenance:
        return Provenance.of(self.values)

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
    """Where an op sits in the file: its key line, the prose after it, and its rows."""

    op: str
    line: int
    rows: Tuple[Row, ...]
    header: str = ""


class BudgetTable:
    """The table as rows, located through PyYAML's event stream: values mean what the
    loader says they mean (anchors, ``<<`` merges, YAML booleans and dates), and a
    row's comment is the text after ``#`` past the end of its node. Tolerant of a
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

    def rows_of(self, op: str) -> List[Row]:
        block = self.blocks.get(op)
        return list(block.rows) if block else []

    def row(self, op: str, **key: str) -> Row:
        """The one row of *op* pinning exactly *key*. Key fields are spelt as in the
        table, with ``in_`` for ``in``: ``row("Gelu", in_="Float16", out="Float16")``.
        """
        names = ("in_", "out", "approx", "dest", "arch")
        unknown = set(key) - set(names)
        if unknown:
            raise TypeError(f"not key fields: {sorted(unknown)}")
        wanted = tuple(
            (k, key[name]) for k, name in zip(KEY_FIELDS, names) if name in key
        )
        found = [r for r in self.rows_of(op) if r.key == wanted]
        if len(found) != 1:
            raise LookupError(f"{op}: {len(found)} rows pin {dict(wanted)}, not one")
        return found[0]

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
                i = _skip_node(events, _skip_node(events, i - 1))
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
                for values, (start, end, alias) in zip(entries, positions):
                    if not isinstance(values, dict):
                        continue
                    rows.append(
                        Row(
                            op=op,
                            fields=tuple(values.items()),
                            first_line=start.line,
                            last_line=end.line,
                            end_column=end.column,
                            comment=self._comment_after(end),
                            alias=alias,
                        )
                    )
            self.blocks[op] = OpBlock(
                op=op,
                line=key_event.start_mark.line,
                rows=tuple(rows),
                header=self._comment_after(key_event.end_mark),
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


def render_row(row_fields: Dict[str, Any], comment: str = "") -> str:
    """One inline row as the emitter writes it, with its newline. ``approx``/``dest``
    quoted so YAML keeps them strings; a date unquoted so YAML reads it as one."""
    body = ", ".join(
        f'{k}: "{v}"' if k in ("approx", "dest") else f"{k}: {v}"
        for k, v in row_fields.items()
    )
    return f"  - {{{body}}}" + (f"  # {comment}" if comment else "") + "\n"

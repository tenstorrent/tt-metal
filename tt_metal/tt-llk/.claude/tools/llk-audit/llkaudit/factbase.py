# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
FactBase — loads the C++ extractor's per-file JSON output, dedups, and offers
query helpers the checkers reason over.

Each header is re-parsed via many translation units (it is #included widely), so
the same physical fact appears in multiple per-file objects; we dedup by
identity. Ordering within a function is by file offset (the extractor emits a
byte offset for every fact), which is deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable


@dataclass
class Function:
    name: str
    file: str
    begin_off: int
    end_off: int
    line: int

    def contains(self, file: str, off: int) -> bool:
        return file == self.file and self.begin_off <= off <= self.end_off


class FactBase:
    def __init__(self, arch: str, facts: list[dict], parse_errors: int = 0):
        self.arch = arch
        self.facts = facts
        self.parse_errors = parse_errors
        #: {NAME_ADDR32: int} resolved from cfg_defines.h; set by the CLI when a
        #: metal root is available (used by cfg-word-overlap). Empty otherwise.
        self.addr32: dict = {}
        self._funcs = [
            Function(
                f.get("name", ""),
                f["file"],
                f["off"],
                f.get("end_off", f["off"]),
                f["line"],
            )
            for f in facts
            if f["family"] == "function"
        ]
        # index functions by file for fast enclosing() lookup
        self._funcs_by_file: dict[str, list[Function]] = {}
        for fn in self._funcs:
            self._funcs_by_file.setdefault(fn.file, []).append(fn)

    # -- construction ---------------------------------------------------------
    @staticmethod
    def _dedup_key(f: dict) -> tuple:
        return (
            f.get("family"),
            f.get("file"),
            f.get("line"),
            f.get("off"),
            f.get("name", ""),
            f.get("index_text", ""),
            f.get("producer", ""),
            f.get("text", ""),
        )

    @classmethod
    def from_objects(cls, arch: str, objects: Iterable[dict]) -> "FactBase":
        """Merge many per-file extractor objects into one deduped FactBase."""
        seen: dict[tuple, dict] = {}
        parse_errors = 0
        for o in objects:
            parse_errors += o.get("parse_errors", 0)
            for f in o.get("facts", []):
                seen.setdefault(cls._dedup_key(f), f)
        return cls(arch, list(seen.values()), parse_errors)

    @classmethod
    def from_concatenated_json(cls, arch: str, text: str) -> "FactBase":
        """Parse a stream of pretty/among-line-concatenated JSON objects."""
        objs, buf, depth, in_str, esc = [], [], 0, False, False
        for ch in text:
            buf.append(ch)
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    objs.append(json.loads("".join(buf)))
                    buf = []
        return cls.from_objects(arch, objs)

    # -- queries --------------------------------------------------------------
    def family(self, fam: str) -> list[dict]:
        return [f for f in self.facts if f["family"] == fam]

    @property
    def functions(self) -> list[Function]:
        return self._funcs

    def enclosing(self, file: str, off: int) -> Function | None:
        best = None
        for fn in self._funcs_by_file.get(file, []):
            if fn.contains(file, off) and (
                best is None or fn.begin_off > best.begin_off
            ):
                best = fn
        return best

    def facts_in(
        self, fn: Function, families: tuple[str, ...] | None = None
    ) -> list[dict]:
        out = [
            f
            for f in self.facts
            if f["family"] != "function"
            and f["file"] == fn.file
            and fn.begin_off <= f["off"] <= fn.end_off
            and (families is None or f["family"] in families)
        ]
        out.sort(key=lambda f: f["off"])
        return out

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load optimizer-block metadata from catalog.yaml and instantiate blocks.

The catalog is the single source of truth for what the user sees in the
report sidebar and in `perf blocks list`. The implementations live in
separate modules so adding a block is: (1) write the module, (2) add
its YAML entry.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


CATALOG_PATH = Path(__file__).parent / "catalog.yaml"


@dataclass
class CatalogEntry:
    name: str
    level: int
    module: str
    class_name: str
    requires: List[str]
    when: str
    what: str
    safety: str
    expected_gain: str
    examples: str

    @property
    def description(self) -> str:
        return self.what.strip().splitlines()[0] if self.what else ""


_CATALOG_CACHE: Optional[Dict[str, CatalogEntry]] = None


def load_catalog() -> Dict[str, CatalogEntry]:
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE
    if not CATALOG_PATH.exists():
        _CATALOG_CACHE = {}
        return _CATALOG_CACHE
    raw = yaml.safe_load(CATALOG_PATH.read_text()) or {}
    entries: Dict[str, CatalogEntry] = {}
    for b in raw.get("blocks", []):
        entries[b["name"]] = CatalogEntry(
            name=b["name"],
            level=int(b.get("level", 0)),
            module=b.get("module", ""),
            class_name=b.get("class", ""),
            requires=list(b.get("requires", []) or []),
            when=str(b.get("when", "")),
            what=str(b.get("what", "")),
            safety=str(b.get("safety", "")),
            expected_gain=str(b.get("expected_gain", "")),
            examples=str(b.get("examples", "")),
        )
    _CATALOG_CACHE = entries
    return _CATALOG_CACHE


def list_blocks() -> List[CatalogEntry]:
    return sorted(load_catalog().values(), key=lambda e: (e.level, e.name))


def get_block(name: str):
    cat = load_catalog().get(name)
    if cat is None:
        raise KeyError(f"unknown optimizer block: {name}")
    mod = importlib.import_module(cat.module)
    cls = getattr(mod, cat.class_name)
    return cls()


def catalog_for_sidebar() -> List[Tuple[str, int, str]]:
    """Return (name, level, description) tuples for the report sidebar."""
    return [(e.name, e.level, e.description) for e in list_blocks()]


def render_blocks_list_text() -> str:
    out: List[str] = []
    out.append(f"{'NAME':<28s}  {'L':>1s}  REQUIRES")
    out.append("-" * 78)
    for e in list_blocks():
        req = ", ".join(e.requires) if e.requires else "-"
        out.append(f"{e.name:<28s}  L{e.level}  {req}")
    out.append("")
    out.append("Use `tt_hw_planner perf blocks show <name>` for details.")
    return "\n".join(out)


def render_block_show_text(name: str) -> str:
    cat = load_catalog().get(name)
    if cat is None:
        return f"unknown optimizer block: {name}"
    out: List[str] = []
    out.append(f"=== {cat.name}  (L{cat.level}) ===")
    out.append("")
    out.append("WHEN to apply:")
    for line in cat.when.strip().splitlines():
        out.append(f"  {line}")
    out.append("")
    out.append("WHAT it does:")
    for line in cat.what.strip().splitlines():
        out.append(f"  {line}")
    out.append("")
    out.append("SAFETY:")
    for line in cat.safety.strip().splitlines():
        out.append(f"  {line}")
    out.append("")
    out.append(f"Expected gain: {cat.expected_gain}")
    out.append(f"Examples:      {cat.examples}")
    if cat.requires:
        out.append(f"Requires:      {', '.join(cat.requires)} (must be GREEN first)")
    return "\n".join(out)

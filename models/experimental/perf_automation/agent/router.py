"""M2 · Routing core + cache_playbook (PLAN section 7.2, tags per section 4.1).

Routing = string-equality matching between a bucket's tags and the tags each
playbook section declares in its `<!-- route -->` block. Pure, deterministic,
no LLM. The agent never sees this index.

  build_index(dir)            -> [{id, title, file, lever_type, <8 dims>}, ...]
  route(index, query)         -> matching entries, in document order
  read_section(id, dir)       -> section text from `{#id}` heading to next `## `
  coverage_lint(index, keys)  -> the possible keys matching zero sections
  cache_playbook(dir, path)   -> build/refresh `.cache/playbook_index.json`
                                 (content-hash keyed; rebuilt on change)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .atomic import atomic_write

# The eight routing dimensions (PLAN section 4.1). `lever_type` is metadata, not routed.
DIMENSIONS = (
    "op_class",
    "bound",
    "rank",
    "fidelity",
    "grid",
    "dispatch",
    "memory",
    "regime",
)
WILDCARD = "*"

# Closed routing vocabulary (PLAN section 4.1). Queries must use these exact
# values; a parser/vocab drift must fail loudly at the routing boundary rather
# than silently flooding or starving SELECT.
VOCABULARY: dict[str, frozenset[str]] = {
    "op_class": frozenset(
        {
            "matmul",
            "attention",
            "reduction",
            "eltwise",
            "datamove",
            "embedding",
            "conv_pool",
            "ccl",
            "host_fallback",
            "other",
        }
    ),
    "bound": frozenset({"dram", "flop", "both", "slow", "host"}),
    "rank": frozenset({"time", "count"}),
    "fidelity": frozenset({"lofi", "hifi2", "hifi3", "hifi4", "na"}),
    "grid": frozenset({"full", "partial", "tiny"}),
    "dispatch": frozenset({"ok", "gappy"}),
    "memory": frozenset({"dram_interleaved", "l1_interleaved", "sharded"}),
    "regime": frozenset({"prefill", "decode", "na"}),
}

_ROOT = Path(__file__).resolve().parent.parent
GUIDELINES_DIR = _ROOT / "GUIDELINES"
CACHE_PATH = _ROOT / ".cache" / "playbook_index.json"

_H2_RE = re.compile(r"^##\s", re.M)
_ANCHOR_RE = re.compile(r"\{#([a-z0-9-]+)\}")
_ROUTE_RE = re.compile(r"<!--\s*route\s*(.*?)-->", re.S)


def _sections(text: str):
    """Yield (heading_line, block) for each level-2 (`## `) section.

    block spans from the heading line to just before the next `## ` heading.
    """
    starts = [m.start() for m in _H2_RE.finditer(text)]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[start:end]
        heading_line = block.splitlines()[0] if block else ""
        yield heading_line, block


def _anchor_of(heading_line: str) -> str | None:
    m = _ANCHOR_RE.search(heading_line)
    return m.group(1) if m else None


def _title_of(heading_line: str) -> str:
    title = heading_line.lstrip("#").strip()
    return _ANCHOR_RE.sub("", title).strip()


def _parse_route_block(body: str) -> dict[str, list[str]]:
    """Parse `key: v1,v2` lines inside a `<!-- route ... -->` block."""
    dims: dict[str, list[str]] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, raw = line.partition(":")
        values = [v.strip() for v in raw.split(",") if v.strip()]
        if values:
            dims[key.strip()] = values
    return dims


def build_index(playbook_dir: str | os.PathLike[str] = GUIDELINES_DIR) -> list[dict[str, Any]]:
    """Harvest `{#id}` anchors + `<!-- route -->` blocks into the index.

    Only sections that have BOTH an anchor and a route block are indexed
    (process docs without route blocks are skipped). Document order is
    preserved: files sorted by name, sections in file order.
    """
    index: list[dict[str, Any]] = []
    for path in sorted(Path(playbook_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for heading_line, block in _sections(text):
            anchor = _anchor_of(heading_line)
            if not anchor:
                continue
            route_match = _ROUTE_RE.search(block)
            if not route_match:
                continue
            dims = _parse_route_block(route_match.group(1))
            entry: dict[str, Any] = {
                "id": anchor,
                "title": _title_of(heading_line),
                "file": path.name,
                "lever_type": (dims.get("lever_type") or ["single-shot"])[0],
            }
            for dim in DIMENSIONS:
                entry[dim] = dims.get(dim, [WILDCARD])
            index.append(entry)
    return index


def _dim_match(section_values: list[str], query_value: Any) -> bool:
    """A dimension matches if either side is a wildcard, else value-in-list."""
    if WILDCARD in section_values:
        return True
    if query_value is None or query_value == WILDCARD:
        return True
    return query_value in section_values


def _validate_query(query: dict[str, Any]) -> None:
    """Reject unknown dimensions and out-of-vocabulary values (PLAN section 4.1).

    Fail loudly at the routing boundary so a parser/vocab drift surfaces here
    instead of silently flooding or starving SELECT downstream.
    """
    for key, value in query.items():
        if key not in VOCABULARY:
            raise ValueError(f"unknown routing dimension {key!r}; valid dimensions: " f"{sorted(VOCABULARY)}")
        if value is None or value == WILDCARD:
            continue
        if value not in VOCABULARY[key]:
            raise ValueError(
                f"invalid value {value!r} for dimension {key!r}; valid values: " f"{sorted(VOCABULARY[key])}"
            )


def route(index: list[dict[str, Any]], query: dict[str, Any]) -> list[dict[str, Any]]:
    """Return index entries whose declared tags match the bucket `query`.

    Tag-equality across all 8 dimensions with wildcard on either side. Fidelity
    exhaustion falls out naturally: a `fidelity: hifi4,hifi3,hifi2` section does
    not match a `fidelity=lofi` bucket. Results keep document order.

    Raises ValueError on an unknown dimension key or out-of-vocabulary value.
    """
    _validate_query(query)
    return [entry for entry in index if all(_dim_match(entry[dim], query.get(dim)) for dim in DIMENSIONS)]


def read_section(anchor: str, playbook_dir: str | os.PathLike[str] = GUIDELINES_DIR) -> str:
    """Return the section text from the `{#anchor}` heading to the next `## `."""
    for path in sorted(Path(playbook_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for heading_line, block in _sections(text):
            if _anchor_of(heading_line) == anchor:
                return block
    raise KeyError(f"no section with anchor {{#{anchor}}}")


def coverage_lint(index: list[dict[str, Any]], possible_keys: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the possible tag-tuples that match ZERO sections (playbook gaps)."""
    return [key for key in possible_keys if not route(index, key)]


def _playbook_hash(playbook_dir: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    for path in sorted(Path(playbook_dir).glob("*.md")):
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def cache_playbook(
    playbook_dir: str | os.PathLike[str] = GUIDELINES_DIR,
    cache_path: str | os.PathLike[str] = CACHE_PATH,
) -> list[dict[str, Any]]:
    """Return the index, (re)building `.cache/playbook_index.json` on change.

    Content-hash keyed: if the cache exists and its hash matches the current
    playbook content, return it without rewriting; otherwise rebuild + persist.
    """
    cache_path = Path(cache_path)
    current_hash = _playbook_hash(playbook_dir)
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached = {}
        if cached.get("hash") == current_hash:
            return cached["index"]
    index = build_index(playbook_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(cache_path, json.dumps({"hash": current_hash, "index": index}, indent=2))
    return index

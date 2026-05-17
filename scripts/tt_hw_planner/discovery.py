# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Automatic model discovery: derive "is this model supported?", "which file
runs it?", and "what does it need?" from the tt-metal source tree, without
hand-maintained registries.

Three signals are consulted, in order:

  1. `models/model_targets.yaml` — the centralized CI/perf manifest. If the
     HF id appears in any target's `aliases:` list, the model is officially
     targeted (status: `active` or `TODO`).

  2. `git grep` for the HF id literal across `models/`. The set of matching
     files tells us *where* the model lives and which path drives it:
       - tt_transformers/ matches → standard simple_text_demo path
       - models/demos/<x>/ test/demo file → external demo at that path
       - only doc/yaml matches → tracked but not yet wired in

  3. Path-based heuristics over the matched files: `wormhole/` → tested on
     Wormhole, `blackhole/` → tested on Blackhole, `t3000/` → T3K-validated,
     `tg/` or `galaxy/` → multi-mesh-only, etc.

Nothing in this module is hand-curated. Adding a new HF model to tt-metal
(in source or in `model_targets.yaml`) automatically makes the planner
recognize it.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# How "interesting" each kind of reference is. Higher = more likely the demo
# entry point.
_FILE_KIND_PRIORITY = {
    "test_demo": 100,
    "demo_py": 80,
    "test_other": 60,
    "model_config": 40,
    "yaml": 20,
    "doc": 5,
    "other": 1,
}


@dataclass
class FileMatch:
    path: Path  # repo-relative
    kind: str  # see _FILE_KIND_PRIORITY
    priority: int


@dataclass
class TargetEntry:
    """Subset of a `model_targets.yaml` entry. Captures only the fields the
    planner needs."""

    key: str  # e.g. "llama3.2-1b"
    aliases: List[str]
    skus: Dict[str, str]  # sku -> "active" | "TODO" (worst-case status)
    owner_id: Optional[str] = None
    team: Optional[str] = None


@dataclass
class ModelDiscovery:
    hf_id: str
    references: List[FileMatch] = field(default_factory=list)
    target_entry: Optional[TargetEntry] = None
    primary_demo: Optional[Path] = None  # repo-relative
    is_supported: bool = False
    in_tt_transformers: bool = False
    in_external_demo: bool = False
    # Empty set = unconstrained; otherwise a strict arch whitelist.
    arch_compatibility: frozenset = field(default_factory=frozenset)
    notes: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.is_supported:
            return "SUPPORTED"
        if self.target_entry is not None:
            return "TARGETED"
        return "UNKNOWN"

    def runnable_on_arch(self, arch: str) -> bool:
        return not self.arch_compatibility or arch in self.arch_compatibility


# ---------------------------------------------------------------------------
# model_targets.yaml loader
# ---------------------------------------------------------------------------


def _model_targets_path(repo_root: Path) -> Path:
    return repo_root / "models" / "model_targets.yaml"


_TARGETS_CACHE: Optional[Dict[str, TargetEntry]] = None


def _load_targets(repo_root: Path) -> Dict[str, TargetEntry]:
    """Parse model_targets.yaml and return a flat alias -> TargetEntry map."""
    global _TARGETS_CACHE
    if _TARGETS_CACHE is not None:
        return _TARGETS_CACHE

    path = _model_targets_path(repo_root)
    if not path.is_file() or yaml is None:
        _TARGETS_CACHE = {}
        return _TARGETS_CACHE

    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, Exception):
        _TARGETS_CACHE = {}
        return _TARGETS_CACHE

    by_alias: Dict[str, TargetEntry] = {}
    for key, body in (data or {}).get("targets", {}).items():
        if not isinstance(body, dict):
            continue
        aliases = list(body.get("aliases") or [])
        # Roll the worst status across all entries on a sku
        sku_status: Dict[str, str] = {}
        owner = None
        team = None
        for sku, sku_body in (body.get("skus") or {}).items():
            statuses: List[str] = []
            for entry in (sku_body or {}).get("entries", []) or []:
                statuses.append(str(entry.get("status") or "unknown"))
                owner = owner or entry.get("owner_id")
                team = team or entry.get("team")
            if not statuses:
                continue
            sku_status[sku] = "active" if "active" in statuses else statuses[0]
        entry = TargetEntry(
            key=key,
            aliases=aliases,
            skus=sku_status,
            owner_id=owner,
            team=team,
        )
        # The key itself is searchable too
        for alias in [key] + aliases:
            by_alias[alias] = entry
    _TARGETS_CACHE = by_alias
    return _TARGETS_CACHE


def _match_target(hf_id: str, repo_root: Path) -> Optional[TargetEntry]:
    """Return the target entry matching an HF id (full path or tail)."""
    targets = _load_targets(repo_root)
    if hf_id in targets:
        return targets[hf_id]
    tail = hf_id.split("/")[-1]
    if tail in targets:
        return targets[tail]
    # Case-insensitive contains: HF "Llama-3.1-8B" should match alias
    # "meta-llama/Llama-3.1-8B-Instruct" (the YAML often lists the Instruct
    # variant). Look for an alias that contains the tail or vice versa.
    for alias, entry in targets.items():
        if tail and (tail in alias or alias in tail):
            return entry
    return None


# ---------------------------------------------------------------------------
# Source-tree reference grep
# ---------------------------------------------------------------------------


def _classify_path(path: Path) -> str:
    s = str(path)
    name = path.name.lower()
    if "test_" in name and "demo" in name:
        return "test_demo"
    if name == "demo.py" or (name.startswith("demo_") and name.endswith(".py")):
        return "demo_py"
    if name.startswith("test_") and name.endswith(".py"):
        return "test_other"
    if name == "model_config.py":
        return "model_config"
    if name.endswith(".yaml") or name.endswith(".yml"):
        return "yaml"
    if name.endswith(".md") or name.endswith(".rst") or name.endswith(".txt"):
        return "doc"
    return "other"


def _git_grep(needle: str, repo_root: Path, scope: str = "models/") -> List[Path]:
    """Return repo-relative paths matching `needle` literally. Falls back to
    a Python walker if git isn't available."""
    gitexe = shutil.which("git")
    if gitexe and (repo_root / ".git").exists():
        try:
            res = subprocess.run(
                [gitexe, "grep", "-l", "-F", "--", needle, "--", scope],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if res.returncode in (0, 1):  # 1 == no matches
                return [Path(line.strip()) for line in res.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.TimeoutExpired):
            pass
    # Fallback: walk the tree
    out: List[Path] = []
    root = repo_root / scope
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".py", ".yaml", ".yml", ".md", ".rst", ".txt"}:
            continue
        try:
            if needle in p.read_text(errors="ignore"):
                out.append(p.relative_to(repo_root))
        except OSError:
            continue
    return out


def _gather_references(hf_id: str, repo_root: Path) -> List[FileMatch]:
    """Find all files that mention the HF id (full path or just the tail).
    Tail-only matches are valuable because many tt_transformers tests use
    `Llama-3.1-8B` rather than the full org-prefixed id."""
    seen: Dict[Path, FileMatch] = {}
    needles = [hf_id]
    tail = hf_id.split("/")[-1]
    if tail and tail != hf_id:
        needles.append(tail)
    for needle in needles:
        for rel in _git_grep(needle, repo_root):
            if rel in seen:
                continue
            kind = _classify_path(rel)
            seen[rel] = FileMatch(
                path=rel,
                kind=kind,
                priority=_FILE_KIND_PRIORITY.get(kind, 0),
            )
    return sorted(seen.values(), key=lambda m: (-m.priority, str(m.path)))


# ---------------------------------------------------------------------------
# Primary demo picker
# ---------------------------------------------------------------------------


def _pick_primary_demo(refs: List[FileMatch]) -> Optional[FileMatch]:
    """Choose the most-likely pytest entry point from the references.

    Preference order:
      1. A test_*demo*.py file that references the model literally (real
         entry points: `pytest <this file>`).
      2. A demo.py file (less common as a direct entry; some demos ship
         their own `__main__`).
      3. As a last resort, a model_config.py reference inside tt_transformers
         — that signals the model is wired into the standard demo even if no
         test file pins the HF id literally."""
    for m in refs:
        if m.kind == "test_demo":
            return m
    for m in refs:
        if m.kind == "demo_py":
            return m
    for m in refs:
        if m.kind == "model_config" and "tt_transformers" in str(m.path):
            return m
    return None


_SKU_ARCH_PREFIXES = {
    "wh_": "Wormhole",
    "bh_": "Blackhole",
    "n150": "Wormhole",
    "n300": "Wormhole",
    "t3k": "Wormhole",
    "tg": "Wormhole",
    "p150": "Blackhole",
    "qb": "Blackhole",
    "bhglx": "Blackhole",
}


def _arch_compatibility_from_skus(skus: Dict[str, str]) -> frozenset:
    archs = set()
    for sku in skus:
        s = sku.lower()
        for prefix, arch in _SKU_ARCH_PREFIXES.items():
            if s.startswith(prefix):
                archs.add(arch)
                break
    return frozenset(archs)


def _arch_compatibility_from_path(path: Path) -> frozenset:
    parts = [p.lower() for p in path.parts]
    if "tt_transformers" in parts:
        return frozenset({"Wormhole", "Blackhole"})
    if "wormhole" in parts or "t3000" in parts or "galaxy" in parts:
        return frozenset({"Wormhole"})
    if "blackhole" in parts:
        return frozenset({"Blackhole"})
    # `tg/` left unconstrained — could be wormhole-tg or blackhole-tg.
    return frozenset()


def _path_notes(path: Path, arches: frozenset) -> List[str]:
    parts = [p.lower() for p in path.parts]
    notes: List[str] = []
    if "tt_transformers" in parts:
        notes.append("Driven by `tt_transformers/simple_text_demo.py` (arch-portable).")
        return notes
    notes.append(f"External demo path: `{path.as_posix()}`.")
    if arches == frozenset({"Wormhole"}):
        if "t3000" in parts:
            notes.append("Path includes `t3000/` — REQUIRES a Wormhole 8-chip (T3K) host.")
        elif "galaxy" in parts:
            notes.append("Path includes `galaxy/` — REQUIRES a Wormhole multi-mesh (Galaxy) host.")
        else:
            notes.append("Path includes `wormhole/` — REQUIRES Wormhole hardware (will not compile on Blackhole).")
    elif arches == frozenset({"Blackhole"}):
        notes.append("Path includes `blackhole/` — REQUIRES Blackhole hardware (will not compile on Wormhole).")
    elif not arches:
        notes.append("No explicit arch-named directory — assumed arch-portable.")
    return notes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def discover_model(hf_id: str, repo_root: Path = REPO_ROOT) -> ModelDiscovery:
    target = _match_target(hf_id, repo_root)
    references = _gather_references(hf_id, repo_root)
    primary = _pick_primary_demo(references)

    discovery = ModelDiscovery(
        hf_id=hf_id,
        references=references,
        target_entry=target,
    )
    if primary is not None:
        discovery.primary_demo = primary.path
        s = str(primary.path)
        discovery.is_supported = True
        discovery.in_tt_transformers = "tt_transformers" in s
        discovery.in_external_demo = "models/demos/" in s
        discovery.arch_compatibility = _arch_compatibility_from_path(primary.path)
        discovery.notes = _path_notes(primary.path, discovery.arch_compatibility)
    # Fall back to SKU prefixes only when the path didn't pin an arch — the
    # path is authoritative because it points at code that actually exists.
    if not discovery.arch_compatibility and target is not None:
        sku_archs = _arch_compatibility_from_skus(target.skus)
        if sku_archs and len(sku_archs) < 2:
            discovery.arch_compatibility = sku_archs
            arch_name = next(iter(sku_archs))
            sku_list = ", ".join(target.skus.keys()) or "(none)"
            discovery.notes.append(
                f"model_targets.yaml lists only {arch_name} SKUs ({sku_list}); " f"treating as {arch_name}-only."
            )
    return discovery


# ---------------------------------------------------------------------------
# Lightweight pretty-printer used by `compat` (full text rendering lives in
# report.py to keep this module dependency-free).
# ---------------------------------------------------------------------------


def format_inline(discovery: ModelDiscovery, *, indent: str = "  ") -> List[str]:
    """Return a list of lines summarizing the discovery. Caller appends them
    to its own renderer."""
    out: List[str] = []
    out.append(f"{indent}Repo discovery: {discovery.status}")
    if discovery.target_entry is not None:
        t = discovery.target_entry
        sku_str = ", ".join(f"{k}={v}" for k, v in t.skus.items()) or "(none)"
        out.append(f"{indent}  Targeted in model_targets.yaml as `{t.key}`")
        out.append(f"{indent}  SKUs: {sku_str}")
    if discovery.primary_demo is not None:
        kind = (
            "tt_transformers"
            if discovery.in_tt_transformers
            else ("external demo" if discovery.in_external_demo else "other")
        )
        out.append(f"{indent}  Primary demo ({kind}): {discovery.primary_demo.as_posix()}")
        if discovery.arch_compatibility:
            arch_str = ", ".join(sorted(discovery.arch_compatibility))
            out.append(f"{indent}  Runs on:    {arch_str}")
        else:
            out.append(f"{indent}  Runs on:    (unconstrained — no explicit arch directory)")
    for n in discovery.notes:
        for line in _wrap(n, indent + "  "):
            out.append(line)
    if not discovery.references and discovery.target_entry is None:
        out.append(f"{indent}  No references found in models/ — not yet ported.")
    return out


def _wrap(text: str, indent: str, width: int = 88) -> List[str]:
    words = text.split()
    if not words:
        return [indent.rstrip()]
    out, line = [], indent
    for w in words:
        if len(line) + len(w) + 1 > width and line.strip():
            out.append(line.rstrip())
            line = indent + w
        else:
            line += (" " + w) if line.strip() else w
    if line.strip():
        out.append(line.rstrip())
    return out

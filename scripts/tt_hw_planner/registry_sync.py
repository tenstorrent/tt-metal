"""Registry drift detection for tt_hw_planner (fixes-plan Point 2a).

The Layer-2 mapping registries point at concrete tt-metal paths:

  * ``family_backends._BACKENDS`` -> ``demo_path`` / ``smoke_test_entry``
  * ``compatibility.BUILDING_BLOCKS`` -> ``tt_path`` / ``registry_tt_path``

When the repo moves or renames those files the registries silently go stale,
which is the root cause of wrong-sibling-template and unknown-arch findings.
:func:`check_registry_drift` verifies every registered path still exists in the
checkout and (reverse) flags reusable ``tt_transformers/tt`` modules that no
registry entry references, so ``sync-registry --check`` fails loudly instead of
the planner mis-pointing at a path that no longer exists.

This is the "validate now" half of Point 2a; the AST-declared auto-generation
of the registry ("generate later") builds on the same path model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class DriftIssue:
    """A single registry/tree mismatch.

    kind == 'missing_path' : a registered path no longer exists (hard drift).
    kind == 'unmapped'     : a reusable tt module exists but nothing maps it (soft).
    """

    kind: str
    where: str
    path: str
    detail: str = ""


def _extract_path(raw: Optional[str]) -> Optional[str]:
    """Leading repo-relative path token of a registry path field, or None.

    Returns None for non-filesystem sentinels the registries also store in
    these fields — bare markers like ``"HF"`` and op references like
    ``"ttnn.argmax"`` (no path separator) are not checkable paths. Handles
    annotations (``"a/b.py (wraps c/d.py)"``) and trailing-slash directories.
    """
    if not raw:
        return None
    tok = raw.strip().split(" ")[0].strip().rstrip("/")
    if not tok or "/" not in tok:
        return None
    return tok


def _registered_paths() -> List[tuple]:
    """Every (where, path) the deterministic registries point at."""
    from .compatibility import BUILDING_BLOCKS
    from .family_backends import all_backends

    out: List[tuple] = []
    for b in all_backends():
        for fld in ("demo_path", "smoke_test_entry"):
            p = _extract_path(getattr(b, fld, None))
            if p:
                out.append((f"family_backends[{b.name}].{fld}", p))
    for bb in BUILDING_BLOCKS:
        for fld in ("tt_path", "registry_tt_path"):
            p = _extract_path(getattr(bb, fld, None))
            if p:
                out.append((f"building_blocks[{bb.name}].{fld}", p))
    return out


def check_registry_drift(repo_root, include_unmapped: bool = True) -> List[DriftIssue]:
    """Return every registry/tree mismatch under ``repo_root``.

    Hard drift (``missing_path``) is a registered path that no longer exists.
    Soft drift (``unmapped``) is a ``models/tt_transformers/tt/*.py`` reusable
    module that no registry entry references — a hint that a new building block
    may need mapping. Never raises; a missing tree just yields no unmapped hints.
    """
    root = Path(repo_root)
    issues: List[DriftIssue] = []
    referenced: set = set()

    for where, rel in _registered_paths():
        referenced.add(rel)
        if not (root / rel).exists():
            issues.append(DriftIssue("missing_path", where, rel, "registered path does not exist in the checkout"))

    if include_unmapped:
        ttt = root / "models" / "tt_transformers" / "tt"
        if ttt.is_dir():
            for f in sorted(ttt.glob("*.py")):
                if f.name.startswith("_") or f.name == "__init__.py":
                    continue
                rel = str(f.relative_to(root))
                if not any(rel == r or rel.startswith(r + "/") or r.startswith(rel) for r in referenced):
                    issues.append(
                        DriftIssue(
                            "unmapped",
                            "tt_transformers/tt",
                            rel,
                            "reusable module not referenced by any registry entry",
                        )
                    )

    return issues


def format_drift(issues: List[DriftIssue]) -> str:
    """Human-readable drift report (grouped: hard drift first, then hints)."""
    if not issues:
        return "registry OK — every mapped path exists; no unmapped reusable modules."
    missing = [i for i in issues if i.kind == "missing_path"]
    unmapped = [i for i in issues if i.kind == "unmapped"]
    lines: List[str] = []
    if missing:
        lines.append(f"STALE registry paths ({len(missing)}) — these no longer exist in the checkout:")
        for i in missing:
            lines.append(f"  [MISSING] {i.where}\n            -> {i.path}")
    if unmapped:
        lines.append(f"Unmapped reusable modules ({len(unmapped)}) — present in the tree, no registry entry:")
        for i in unmapped:
            lines.append(f"  [unmapped] {i.path}")
    return "\n".join(lines)


def has_hard_drift(issues: List[DriftIssue]) -> bool:
    """True if any registered path is missing (the loud-failure condition)."""
    return any(i.kind == "missing_path" for i in issues)

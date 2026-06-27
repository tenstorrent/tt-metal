"""G2: source-file resolver.

Given the first-diverging :class:`.diverge.ModulePair`, find the two
source files that implement that pair (HF side + TT side). The result
is fed to the LLM as the SOLE pair of files to inspect, replacing the
hardcoded ``Suspect.files`` lists.

This is purely Python introspection. It works for any model whose
classes are importable; no architecture-specific knowledge is needed.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from .diverge import ModulePair


@dataclass
class SuspectSourceFiles:
    """Output of :func:`resolve_suspect_files`.

    ``hf_file`` and ``tt_file`` are absolute paths to the source files
    that implement the first-diverging pair. Either can be ``None`` if
    resolution failed (e.g. the class is dynamically generated)."""

    hf_class: str
    tt_class: str
    hf_file: Optional[Path]
    tt_file: Optional[Path]
    note: str = "ok"


def _resolve_class(class_name: str, model_id: str = "") -> Optional[type]:
    """Best-effort import-by-class-name. Tries a few sensible module
    paths in order; returns the first hit.

    No category-specific lookup: we search known top-level package
    namespaces and resolve via :func:`importlib.import_module`."""

    try:
        import transformers

        cls = getattr(transformers, class_name, None)
        if isinstance(cls, type):
            return cls
    except Exception:
        pass

    try:
        from transformers import models as _hf_models

        for sub in dir(_hf_models):
            if sub.startswith("_"):
                continue
            try:
                m = importlib.import_module(f"transformers.models.{sub}")
                cls = getattr(m, class_name, None)
                if isinstance(cls, type):
                    return cls
            except Exception:
                continue
    except Exception:
        pass

    try:
        import diffusers

        cls = getattr(diffusers, class_name, None)
        if isinstance(cls, type):
            return cls
    except Exception:
        pass
    return None


def _resolve_tt_file_for_class(
    tt_class_name: str,
    *,
    workspace_root: Path,
) -> Optional[Path]:
    """Find the TT-side .py file defining a class with the given name.

    Strategy: grep the workspace for ``class <name>``. This is robust
    across the entire tt-metal source tree without hardcoding paths
    like ``models/tt_transformers/tt/...``."""
    if not tt_class_name:
        return None
    import re

    pat = re.compile(rf"^\s*class\s+{re.escape(tt_class_name)}\b")

    search_roots = [
        workspace_root / "models",
        workspace_root / "scripts",
    ]

    def _excluded(p: Path) -> bool:
        s = str(p)
        return "/experimental/" in s or "/tt_transformers_v2/" in s

    for root in search_roots:
        if not root.is_dir():
            continue
        for production_only in (True, False):
            for py in root.rglob("*.py"):
                if production_only and _excluded(py):
                    continue
                try:
                    text = py.read_text(encoding="utf-8", errors="ignore")
                    if tt_class_name not in text:
                        continue
                    for line in text.splitlines():
                        if pat.match(line):
                            return py
                except Exception:
                    continue
    return None


def resolve_suspect_files(
    pair: ModulePair,
    *,
    workspace_root: Path,
    model_id: str = "",
) -> SuspectSourceFiles:
    """For a divergence pair, return the HF and TT source files."""
    hf_file: Optional[Path] = None
    tt_file: Optional[Path] = None
    notes: List[str] = []

    hf_cls = _resolve_class(pair.hf_class, model_id=model_id)
    if hf_cls is not None:
        try:
            src = inspect.getsourcefile(hf_cls) or inspect.getfile(hf_cls)
            if src:
                hf_file = Path(src).resolve()
        except Exception as exc:
            notes.append(f"hf-getsourcefile-failed: {type(exc).__name__}")
    else:
        notes.append(f"hf-class-not-imported: {pair.hf_class}")

    tt_file = _resolve_tt_file_for_class(pair.tt_class, workspace_root=workspace_root)
    if tt_file is None:
        if pair.tt_class == pair.hf_class and hf_file is not None:
            tt_file = hf_file
            notes.append(f"tt-file=hf-file (tt_class=={pair.hf_class})")
        else:
            notes.append(f"tt-class-not-found: {pair.tt_class}")

    note = "ok" if not notes else "; ".join(notes)
    return SuspectSourceFiles(
        hf_class=pair.hf_class,
        tt_class=pair.tt_class,
        hf_file=hf_file,
        tt_file=tt_file,
        note=note,
    )


def read_file_excerpt(path: Optional[Path], *, max_chars: int = 8000) -> str:
    """Read a file's contents up to ``max_chars`` characters. Returns
    "(unreadable: ...)" on error so the LLM still gets useful context."""
    if path is None:
        return "(no path)"
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return f"(unreadable: {type(exc).__name__}: {exc})"
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n... (truncated, full size {len(text)} bytes)"
    return text


__all__ = ["SuspectSourceFiles", "read_file_excerpt", "resolve_suspect_files"]

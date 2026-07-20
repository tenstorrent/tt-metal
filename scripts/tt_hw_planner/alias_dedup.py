"""E: credit duplicate-named components that are the SAME module.

The scaffolder can list one physical module under several component names —
e.g. ``layer``, ``encoder_stack`` and ``seamless_m4_tv2_encoder_layer`` are all
``text_encoder.layers.0`` in seamless-m4t-v2. When one graduates, the others
keep showing "not done" even though the work is on the device.

This credits them — but ONLY when proven identical by **resolved
submodule_path** (a path identifies exactly one module, so same resolved path =
same module). It NEVER merges by fuzzy class-name match: that once mis-resolved
``layer`` to a ``LayerNorm`` and would false-merge. So resolution here uses only
reliable signals (the recorded submodule_path and the test's explicit
``_CANDIDATE_SUBMODULE_PATHS``), never the class-name fallback.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


def _normalize_path(p: str) -> str:
    return re.sub(r"\[(\d+)\]", r".\1", p or "").strip(".")


def _strict_resolve_path(
    model: Any,
    recorded_path: Optional[str],
    candidate_paths: List[str],
    resolve_attr: Callable[[Any, str], Any],
) -> Optional[str]:
    """Return the canonical (normalized) path this component resolves to using
    ONLY reliable signals — the recorded submodule_path then the explicit
    candidate paths. No fuzzy class-name fallback. None if nothing resolves."""
    tries: List[str] = []
    if recorded_path:
        tries.append(recorded_path)
    tries.extend(candidate_paths or [])
    for path in tries:
        if not path:
            continue
        try:
            resolve_attr(model, path)
            return _normalize_path(path)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
    return None


def group_and_credit(resolved: Dict[str, str], graduated: Set[str]) -> Dict[str, str]:
    """Pure, model-free core: given ``{comp: canonical_path}`` and the graduated
    set, return ``{pending_comp: twin}`` for every pending component that shares
    its canonical path with a graduated component (its twin)."""
    by_path: Dict[str, List[str]] = {}
    for comp, path in resolved.items():
        if path:
            by_path.setdefault(path, []).append(comp)
    credits: Dict[str, str] = {}
    for _path, comps in by_path.items():
        grad_twins = [c for c in comps if c in graduated]
        if not grad_twins:
            continue
        for c in comps:
            if c not in graduated:
                credits[c] = grad_twins[0]
    return credits


def reconcile_alias_duplicates(
    *,
    model_id: str,
    demo_dir: Path,
    graduated_set: Set[str],
) -> List[str]:
    """Load the best-coverage model variant, resolve each NEW component's
    canonical path (reliable signals only), and persist an alias-credit for
    every pending component proven to be the same module as a graduated one.

    Returns the list of newly-credited component names. Best-effort: any error
    is swallowed (returns what it managed), so it can never break the loop.
    """
    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        return []
    try:
        status = json.loads(status_path.read_text())
    except Exception:
        return []
    comps = [c for c in (status.get("components") or []) if c.get("status") == "NEW" and c.get("name")]
    names = [c["name"] for c in comps]
    if len(names) < 2:
        return []
    recorded = {c["name"]: c.get("submodule_path") for c in comps}

    try:
        from .agentic.probe import iter_hf_model_variants
        from .capture_inputs import _read_candidates_from_test, _resolve_attr, _safe_id
        from .overlay_manager import load_alias_credits, persist_alias_credit
    except Exception:
        return []

    candidates: Dict[str, List[str]] = {}
    for n in names:
        paths: List[str] = []
        for src in (
            demo_dir / "tests" / "pcc" / f"test_{_safe_id(n)}.py",
            demo_dir / "_stubs" / f"{_safe_id(n)}.py",
        ):
            for p in _read_candidates_from_test(src):
                if p and p not in paths:
                    paths.append(p)
        candidates[n] = paths

    def _norm_set(n: str) -> Set[str]:
        s: Set[str] = set()
        if recorded.get(n):
            s.add(_normalize_path(recorded[n]))
        for p in candidates.get(n, []):
            s.add(_normalize_path(p))
        return s

    graduated = set(graduated_set)
    graduated_paths: Set[str] = set()
    for n in names:
        if n in graduated:
            graduated_paths |= _norm_set(n)
    if not graduated_paths or not any(n not in graduated and (_norm_set(n) & graduated_paths) for n in names):
        return []

    def _coverage(model: Any) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for n in names:
            p = _strict_resolve_path(model, recorded.get(n), candidates.get(n, []), _resolve_attr)
            if p:
                out[n] = p
        return out

    best_resolved: Dict[str, str] = {}
    try:
        for cand_model, _loader in iter_hf_model_variants(model_id, torch_dtype="float32"):
            res = _coverage(cand_model)
            del cand_model
            if len(res) > len(best_resolved):
                best_resolved = res
            if len(best_resolved) >= len(names):
                break
    except Exception:
        return []

    if not best_resolved:
        return []

    credits = group_and_credit(best_resolved, set(graduated_set))
    already = set(load_alias_credits(model_id).keys())
    newly: List[str] = []
    for comp, twin in credits.items():
        if comp in already:
            continue
        persist_alias_credit(model_id, comp, canonical_path=best_resolved.get(comp, ""), twin=twin)
        newly.append(comp)
    return sorted(newly)

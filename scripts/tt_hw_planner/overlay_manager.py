from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


_THIS_DIR = Path(__file__).resolve().parent
_OVERLAYS_DIR = _THIS_DIR / "overlays"
_INDEX_NAME = "index.json"

_SHARED_PREFIXES = (
    "models/tt_transformers/",
    "models/common/",
    "models/perf/",
)

_DEMO_CATEGORY_DIRS = frozenset(
    {
        "multimodal",
        "vision",
        "speech_recognition",
        "audio",
        "diffusion",
    }
)


_REPO_OVERRIDE: Optional[Path] = None


@contextmanager
def using_repo(repo_dir: Path) -> Iterator[None]:
    global _REPO_OVERRIDE
    prev = _REPO_OVERRIDE
    _REPO_OVERRIDE = Path(repo_dir).resolve()
    try:
        yield
    finally:
        _REPO_OVERRIDE = prev


def _repo_root() -> Path:
    if _REPO_OVERRIDE is not None:
        return _REPO_OVERRIDE
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(out.stdout.strip())


def classify_path(rel_path: str) -> str:
    norm = rel_path.replace("\\", "/")
    if norm.startswith("scripts/tt_hw_planner/"):
        return "tool"
    if any(norm.startswith(p) for p in _SHARED_PREFIXES):
        return "shared"
    if norm.startswith("models/demos/"):
        parts = norm.split("/")
        for i in range(3, len(parts)):
            if parts[i] in ("tt", "tests", "demo"):
                between = parts[2:i]
                if len(between) <= 1:
                    return "shared"
                if len(between) == 2 and between[0] in _DEMO_CATEGORY_DIRS:
                    return "shared"
                return "model_local"
        return "other"
    return "other"


def _slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def _model_dir(model_id: str) -> Path:
    return _OVERLAYS_DIR / _slug(model_id)


def _index_path(model_id: str) -> Path:
    return _model_dir(model_id) / _INDEX_NAME


def _load_index(model_id: str) -> Dict[str, dict]:
    p = _index_path(model_id)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_index(model_id: str, idx: Dict[str, dict]) -> None:
    p = _index_path(model_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(idx, indent=2, sort_keys=True))


def _skipped_components_path(model_id: str) -> Path:
    return _model_dir(model_id) / "skipped_components.json"


def load_persistent_skips(model_id: str) -> Dict[str, dict]:
    """Return the persistent {comp_name: {reason, captured_ts}} dict for `model_id`.

    Empty dict if no skip-list file exists yet. Components on this list are
    known harness-untestable from a prior run -- the auto-iterate loop reads
    this at session start and adds these to `permanently_skipped` so no LLM
    budget is spent re-attempting them.
    """
    p = _skipped_components_path(model_id)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def persist_skip(model_id: str, comp_name: str, reason: str = "", *, category: str = "COLD") -> None:
    """Add `comp_name` to the persistent skip-list for `model_id`.

    Behavior on duplicate (entry already exists):
      - captured_ts: preserved from the FIRST persist (audit-trail; we
        want to know when the component was originally flagged)
      - reason: updated to the new value (the latest reason is most
        informative)
      - category: KERNEL_MISSING is "sticky" — once a component has a
        verified-missing TTNN op, later runs CAN'T downgrade it (the
        gap doesn't go away just because a different failure mode
        manifests on a re-run). Any other transition is allowed; the
        newer verdict wins.

    Without this update logic, the seed-phase persist (default COLD)
    would block a later _skip_component_to_fallback from labeling the
    same component as KERNEL_MISSING — silently mis-categorizing it.

    Valid categories (from `failure_classifier.SKIP_CATEGORY_FOR_CLASS`):
      - "COLD"               : not invoked in workload (CPU correct, perf-irrelevant)
      - "KERNEL_MISSING"     : TTNN op verified missing (TTNN dev work)
      - "CONSTRAINT_MISMATCH": TTNN op present, failure is dtype/layout/shape
      - "TOOL_BUG"           : scaffolder produced bad inputs (tool fix)
      - "HF_ERROR"           : HF reference itself errored (not TTNN)
      - "ITERATION_BUDGET"   : hit attempt cap; retry next run with bigger budget
      - "AGENT_STUCK"        : NO_OP / empty agent; decomposition didn't help
    """
    new_category = category.upper()
    skips = load_persistent_skips(model_id)
    # Specificity ladder (lowest -> highest). A more-specific existing
    # category cannot be downgraded by a less-specific new one. Tied
    # specificity allows newer-wins (e.g. ITERATION_BUDGET ->
    # AGENT_STUCK across runs both at "specific" level).
    _SPECIFICITY = {
        "": 0,  # missing/legacy
        "COLD": 1,  # generic fallback
        "TOOL_BUG": 2,  # specific classifier verdict
        "HF_ERROR": 2,
        "CONSTRAINT_MISMATCH": 2,
        "ITERATION_BUDGET": 2,
        "AGENT_STUCK": 2,
        "KERNEL_MISSING": 3,  # sticky: TTNN gap verified
    }
    # Retryable categories get a retry counter. After N retries on the
    # same retryable verdict, auto-escalate to COLD so the loop stops
    # burning budget. Tracks via the `retry_count` field on the entry.
    _RETRYABLE = {"ITERATION_BUDGET", "AGENT_STUCK"}
    MAX_RETRIES_BEFORE_ESCALATION = 3

    if comp_name in skips:
        existing = skips[comp_name]
        existing_category = (existing.get("category") or "").upper()
        # Bump retry counter ONLY when the same retryable verdict
        # re-appears. Once retries cross the threshold, the IF below
        # auto-escalates the new_category to COLD and flags
        # `escalated_now` so the specificity guard knows to allow the
        # otherwise-forbidden 2->1 demotion.
        escalated_now = False
        retry_bumped = False
        if existing_category == new_category and new_category in _RETRYABLE:
            retry_count = int(existing.get("retry_count") or 0) + 1
            existing["retry_count"] = retry_count
            retry_bumped = True
            if retry_count >= MAX_RETRIES_BEFORE_ESCALATION:
                new_category = "COLD"
                escalated_now = True
                reason = (
                    f"{reason} — auto-escalated to COLD after {retry_count} "
                    f"retries on {existing_category} (max reached)"
                )
        # Specificity guard: never downgrade a more-specific category
        # to a less-specific one EXCEPT in the just-escalated case.
        category_changed = False
        old_spec = _SPECIFICITY.get(existing_category, 0)
        new_spec = _SPECIFICITY.get(new_category, 0)
        if new_spec < old_spec and not escalated_now:
            pass  # keep existing (more specific)
        elif existing_category != new_category:
            existing["category"] = new_category
            category_changed = True
            if escalated_now:
                existing["retry_count"] = 0  # reset after demotion
        # Update reason if a richer one is provided
        reason_changed = False
        if reason and reason != existing.get("reason"):
            existing["reason"] = reason
            reason_changed = True
        if not (category_changed or reason_changed or retry_bumped):
            return
        skips[comp_name] = existing
    else:
        skips[comp_name] = {
            "reason": reason or "harness-incompatible (auto-detected)",
            "category": new_category,
            "captured_ts": time.time(),
            "retry_count": 0,
        }
    p = _skipped_components_path(model_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(skips, indent=2, sort_keys=True))


def clear_persistent_skips(model_id: str, *, category: Optional[str] = None) -> int:
    """Drop skip-list entries for `model_id`. Returns the count removed.

    Args:
      category: If provided (e.g. "ITERATION_BUDGET", "TOOL_BUG"),
                clear only entries matching this category. Useful for
                granular reset — e.g. clear ITERATION_BUDGET after
                bumping the iter cap without nuking verified
                KERNEL_MISSING entries. Case-insensitive. If None
                (default), clear ALL entries (legacy behavior).

    Used when:
      - the test harness was fixed → ``--category TOOL_BUG``
      - the iter cap was bumped → ``--category ITERATION_BUDGET``
      - TTNN shipped the missing op → ``--category KERNEL_MISSING``
      - full reset → no ``--category``
    """
    skips = load_persistent_skips(model_id)
    if not skips:
        return 0
    if category is None:
        p = _skipped_components_path(model_id)
        if p.is_file():
            p.unlink()
        return len(skips)
    target_cat = category.strip().upper()
    kept = {name: entry for name, entry in skips.items() if (entry.get("category") or "").upper() != target_cat}
    removed = len(skips) - len(kept)
    if removed == 0:
        return 0
    p = _skipped_components_path(model_id)
    if not kept:
        if p.is_file():
            p.unlink()
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(kept, indent=2, sort_keys=True))
    return removed


def _no_emit_tests_path(model_id: str) -> Path:
    return _model_dir(model_id) / "no_emit_tests.json"


def load_no_emit_tests(model_id: str) -> Dict[str, dict]:
    """Return the persistent ``{comp_name: {reason, captured_ts}}`` dict for
    components whose standalone PCC test should NOT be emitted on future
    runs.

    Populated by Phase 2 when it drops ModuleList-style components (they
    have no testable forward as a unit; the parent's PCC test covers them).
    Read by ``_emit_pcc_template`` so the next scaffold run doesn't
    re-create the files we just dropped.

    Empty dict if no file exists yet -- never raises so a malformed file
    can't crash the scaffold path.
    """
    p = _no_emit_tests_path(model_id)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def persist_no_emit_test(model_id: str, comp_name: str, reason: str = "") -> None:
    """Add ``comp_name`` to the persistent no-emit-tests list for ``model_id``.

    Idempotent: re-adding an existing component is a no-op (the captured_ts
    of the original entry is preserved). Called from Phase 2's ModuleList
    drop path so the structural decision ("this component is a container
    with no testable forward") survives scaffold regeneration.
    """
    listing = load_no_emit_tests(model_id)
    if comp_name in listing:
        return
    listing[comp_name] = {
        "reason": reason or "structurally untestable as standalone",
        "captured_ts": time.time(),
    }
    p = _no_emit_tests_path(model_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(listing, indent=2, sort_keys=True))


def remove_no_emit_test(model_id: str, comp_name: str) -> bool:
    """Drop a single entry from the no-emit-tests list. Returns True if
    the entry was present and removed. Used when a human decides to
    reverse a ModuleList drop (e.g., they wrote a real forward method
    and want to standalone-test it again)."""
    listing = load_no_emit_tests(model_id)
    if comp_name not in listing:
        return False
    del listing[comp_name]
    p = _no_emit_tests_path(model_id)
    if listing:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(listing, indent=2, sort_keys=True))
    elif p.is_file():
        p.unlink()
    return True


def is_no_emit_test(model_id: str, comp_name: str) -> bool:
    """Fast lookup: is this component on the no-emit list for `model_id`?

    Used as the gate in ``_emit_pcc_template`` so the scaffold can skip
    emission without loading + parsing the full list each time."""
    return comp_name in load_no_emit_tests(model_id)


def _hot_cold_path(model_id: str) -> Path:
    return _model_dir(model_id) / "hot_cold.json"


def load_hot_cold(model_id: str) -> Dict[str, str]:
    """Return ``{comp_name: "HOT" | "COLD" | "UNRESOLVED"}`` for the model.

    Backwards-compat shim: if the on-disk file uses the enriched
    evidence schema (per-component dict with ``kind`` + measurements),
    extract just the ``kind`` for callers that only care about the
    label. For callers needing the full evidence record use
    :func:`load_hot_cold_evidence`.

    Populated by the ``classify-hot-cold`` or ``profile-cold`` CLI
    commands. Read by the auto-iterate loop so COLD components (never
    invoked / no perf value) don't get targeted by the picker.

    Empty dict if no file exists yet -- never raises so a malformed
    file can't crash the auto-iterate path. In that case the loop
    treats every NEW component as HOT (conservative)."""
    raw = _load_hot_cold_raw(model_id)
    out: Dict[str, str] = {}
    for name, val in raw.items():
        if isinstance(val, str):
            out[name] = val
        elif isinstance(val, dict):
            out[name] = str(val.get("kind", "UNRESOLVED")).upper()
        else:
            out[name] = "UNRESOLVED"
    return out


def load_hot_cold_evidence(model_id: str) -> Dict[str, dict]:
    """Return the full evidence record per component (kind + frequency +
    cpu_latency_ms + cpu_latency_pct + ops_count + io_bytes +
    compute_density + evidence reasons).

    Populated by the ``profile-cold`` CLI command. Returns an empty dict
    if no probe has been run. Legacy entries (just the string kind) are
    normalized into a minimal dict with ``{"kind": <str>}`` so callers
    have a single shape to handle."""
    raw = _load_hot_cold_raw(model_id)
    out: Dict[str, dict] = {}
    for name, val in raw.items():
        if isinstance(val, dict):
            out[name] = val
        elif isinstance(val, str):
            out[name] = {"kind": val.upper()}
        else:
            out[name] = {"kind": "UNRESOLVED"}
    return out


def _load_hot_cold_raw(model_id: str) -> dict:
    p = _hot_cold_path(model_id)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def persist_hot_cold(model_id: str, classification: Dict[str, Any]) -> None:
    """Write the classification map. Accepts either the simple
    ``{comp: kind}`` shape (legacy) or the enriched ``{comp: evidence_dict}``
    shape (current). Overwrites any prior file -- the classifier is
    the source of truth, not an accumulator."""
    p = _hot_cold_path(model_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(classification, indent=2, sort_keys=True))


def is_cold_component(model_id: str, comp_name: str) -> bool:
    """Fast lookup used by the auto-iterate loop. Returns True only if
    we have a positive COLD classification for this component (UNRESOLVED
    is treated as if HOT -- conservative -- so we never silently skip a
    component we lack signal on)."""
    return load_hot_cold(model_id).get(comp_name) == "COLD"


def _missing_kernels_path(model_id: str) -> Path:
    return _model_dir(model_id) / "missing_kernels.json"


def load_missing_kernels(model_id: str) -> Dict[str, dict]:
    """Return the persistent ``{comp_name: {missing_op, detected_ts}}`` dict
    for components whose iteration was halted because TTNN lacks the
    operation they need.

    The tool's job is NOT to write missing TTNN kernels -- that's a
    separate engineering workstream. The tool's job is to FLAG the
    missing operation explicitly so:

      1. The end-to-end demo can still emit (KERNEL_MISSING is allowed)
      2. The user gets a clear list of "ttnn dev work needed: op X for
         component Y" rather than a vague "stuck" status
      3. Future runs skip the component until a TTNN release adds the op

    Empty dict if no file exists yet."""
    p = _missing_kernels_path(model_id)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def persist_missing_kernel(model_id: str, comp_name: str, *, missing_op: str = "") -> None:
    """Add ``comp_name`` to the persistent missing-kernels list.

    Idempotent: re-adding preserves the original detected_ts. ``missing_op``
    can be the agent's reported error string, or a short summary like
    'ttnn.permute on sparse_coo'."""
    listing = load_missing_kernels(model_id)
    if comp_name in listing:
        return
    listing[comp_name] = {
        "missing_op": missing_op or "(unspecified — agent failure trace matched kernel-missing pattern)",
        "detected_ts": time.time(),
    }
    p = _missing_kernels_path(model_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(listing, indent=2, sort_keys=True))


def remove_missing_kernel(model_id: str, comp_name: str) -> bool:
    """Drop a single entry (e.g., after a TTNN release adds the op).
    Returns True if entry was present and removed."""
    listing = load_missing_kernels(model_id)
    if comp_name not in listing:
        return False
    del listing[comp_name]
    p = _missing_kernels_path(model_id)
    if listing:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(listing, indent=2, sort_keys=True))
    elif p.is_file():
        p.unlink()
    return True


def is_missing_kernel(model_id: str, comp_name: str) -> bool:
    """Fast lookup for the auto-iterate loop to short-circuit components
    we've already flagged as needing a TTNN kernel addition."""
    return comp_name in load_missing_kernels(model_id)


def remove_persistent_skip(model_id: str, comp_name: str) -> bool:
    """Drop a SINGLE entry from the persistent skip-list. Returns True if
    the entry was present and removed, False if it wasn't there.

    Used by the Phase-2 ``tackle-skipped`` orchestrator when an individual
    component has been resolved (e.g., ModuleList components dropped from
    scaffold, or missing-arg components unblocked by an LLM-drafted driver).
    The all-or-nothing ``clear_persistent_skips`` is too coarse for this case:
    other components on the list may still be genuinely untestable.
    """
    skips = load_persistent_skips(model_id)
    if comp_name not in skips:
        return False
    del skips[comp_name]
    p = _skipped_components_path(model_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    if skips:
        p.write_text(json.dumps(skips, indent=2, sort_keys=True))
    elif p.is_file():
        p.unlink()
    return True


def _patch_filename(rel_path: str) -> str:
    safe = rel_path.replace("/", "__")
    return f"{safe}.patch"


@dataclass
class OverlayRecord:
    model_id: str
    rel_path: str
    patch_path: Path
    captured_ts: float
    sha256: str


def _git_diff_file(rel_path: str, *, against: str = "HEAD") -> str:
    out = subprocess.run(
        ["git", "diff", against, "--", rel_path],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout


def _git_apply(patch_text: str, *, reverse: bool = False, check_only: bool = False) -> Tuple[int, str]:
    args = ["git", "apply"]
    if reverse:
        args.append("--reverse")
    if check_only:
        args.append("--check")
    proc = subprocess.run(
        args,
        cwd=_repo_root(),
        input=patch_text,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, (proc.stderr or proc.stdout)


def _git_checkout_file(rel_path: str) -> Tuple[int, str]:
    proc = subprocess.run(
        ["git", "checkout", "HEAD", "--", rel_path],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stderr


def capture(model_id: str, rel_path: str) -> Optional[OverlayRecord]:
    diff = _git_diff_file(rel_path)
    if not diff.strip():
        return None
    cls = classify_path(rel_path)
    if cls not in ("shared",):
        return None

    patch_name = _patch_filename(rel_path)
    md = _model_dir(model_id)
    md.mkdir(parents=True, exist_ok=True)
    patch_file = md / patch_name
    patch_file.write_text(diff)

    sha = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    idx = _load_index(model_id)
    idx[rel_path] = {
        "patch_file": patch_name,
        "captured_ts": time.time(),
        "sha256": sha,
        "line_count": diff.count("\n"),
        "classification": cls,
    }
    _save_index(model_id, idx)

    rc, err = _git_checkout_file(rel_path)
    if rc != 0:
        print(
            f"[overlay] WARNING captured {rel_path} for {model_id} "
            f"but `git checkout` failed (rc={rc}): {err.strip()}",
            file=sys.stderr,
        )

    return OverlayRecord(
        model_id=model_id,
        rel_path=rel_path,
        patch_path=patch_file,
        captured_ts=time.time(),
        sha256=sha,
    )


def list_overlays(model_id: Optional[str] = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not _OVERLAYS_DIR.is_dir():
        return rows
    model_dirs = [_model_dir(model_id)] if model_id else sorted([p for p in _OVERLAYS_DIR.iterdir() if p.is_dir()])
    for md in model_dirs:
        if not md.is_dir():
            continue
        mid = md.name
        idx_path = md / _INDEX_NAME
        if not idx_path.is_file():
            continue
        try:
            idx = json.loads(idx_path.read_text())
        except Exception:
            continue
        for rel, meta in sorted(idx.items()):
            rows.append(
                {
                    "model_dir_slug": mid,
                    "rel_path": rel,
                    "patch_file": str(md / meta.get("patch_file", _patch_filename(rel))),
                    "captured_ts": meta.get("captured_ts"),
                    "line_count": meta.get("line_count"),
                    "classification": meta.get("classification"),
                }
            )
    return rows


def apply_for(model_id: str) -> Tuple[int, List[str]]:
    idx = _load_index(model_id)
    applied: List[str] = []
    if not idx:
        return 0, applied
    md = _model_dir(model_id)
    for rel, meta in idx.items():
        patch_text = (md / meta["patch_file"]).read_text()
        rc_check, _ = _git_apply(patch_text, check_only=True)
        if rc_check != 0:
            continue
        rc, err = _git_apply(patch_text)
        if rc == 0:
            applied.append(rel)
        else:
            print(
                f"[overlay] apply failed for {rel} (model={model_id}): {err.strip()}",
                file=sys.stderr,
            )
    return len(applied), applied


def revert_for(model_id: str) -> Tuple[int, List[str]]:
    idx = _load_index(model_id)
    reverted: List[str] = []
    if not idx:
        return 0, reverted
    md = _model_dir(model_id)
    for rel, meta in idx.items():
        patch_text = (md / meta["patch_file"]).read_text()
        rc, _ = _git_apply(patch_text, reverse=True, check_only=True)
        if rc != 0:
            continue
        rc, err = _git_apply(patch_text, reverse=True)
        if rc == 0:
            reverted.append(rel)
        else:
            print(
                f"[overlay] revert failed for {rel} (model={model_id}): {err.strip()}",
                file=sys.stderr,
            )
    return len(reverted), reverted


def drop(model_id: str, rel_path: str) -> bool:
    idx = _load_index(model_id)
    if rel_path not in idx:
        return False
    md = _model_dir(model_id)
    patch_file = md / idx[rel_path]["patch_file"]
    if patch_file.is_file():
        patch_file.unlink()
    del idx[rel_path]
    if idx:
        _save_index(model_id, idx)
    else:
        _index_path(model_id).unlink(missing_ok=True)
        try:
            md.rmdir()
        except OSError:
            pass
    return True


def promote(model_id: str, rel_path: str) -> Tuple[bool, str]:
    idx = _load_index(model_id)
    if rel_path not in idx:
        return False, f"no overlay for {rel_path} under {model_id}"
    md = _model_dir(model_id)
    patch_text = (md / idx[rel_path]["patch_file"]).read_text()
    rc_check, err_check = _git_apply(patch_text, check_only=True)
    if rc_check != 0:
        return False, f"patch would not apply cleanly: {err_check.strip()}"
    rc, err = _git_apply(patch_text)
    if rc != 0:
        return False, f"git apply failed: {err.strip()}"
    drop(model_id, rel_path)
    return True, f"applied {rel_path} to working tree; overlay removed"


def extract_from_working_tree(
    model_id: str,
    rel_path: str,
    *,
    hunks_matching: Optional[str] = None,
    intended_for_production: bool = False,
) -> Tuple[bool, str]:
    diff = _git_diff_file(rel_path)
    if not diff.strip():
        return False, f"no changes to {rel_path} in working tree"

    if hunks_matching:
        diff = _filter_hunks(diff, pattern=hunks_matching)
        if not diff.strip():
            return False, f"no hunks matched /{hunks_matching}/ in {rel_path}"

    rec = _store_extracted(
        model_id,
        rel_path,
        diff,
        intended_for_production=intended_for_production,
    )
    rc_rev, err = _git_apply(diff, reverse=True)
    if rc_rev != 0:
        return False, (
            f"stored overlay at {rec.patch_path.relative_to(_repo_root())} "
            f"but reverse-apply on working tree FAILED ({err.strip()}). "
            f"working tree unchanged; you may need to revert manually."
        )
    return True, (
        f"extracted to {rec.patch_path.relative_to(_repo_root())}; " f"shared file {rel_path} reverted in working tree"
    )


def _store_extracted(
    model_id: str,
    rel_path: str,
    diff: str,
    *,
    intended_for_production: bool = False,
) -> OverlayRecord:
    md = _model_dir(model_id)
    md.mkdir(parents=True, exist_ok=True)
    patch_file = md / _patch_filename(rel_path)
    patch_file.write_text(diff)
    sha = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    idx = _load_index(model_id)
    idx[rel_path] = {
        "patch_file": patch_file.name,
        "captured_ts": time.time(),
        "sha256": sha,
        "line_count": diff.count("\n"),
        "classification": classify_path(rel_path),
        "source": "extracted_from_working_tree",
        "intended_for_production": intended_for_production,
    }
    _save_index(model_id, idx)
    return OverlayRecord(
        model_id=model_id,
        rel_path=rel_path,
        patch_path=patch_file,
        captured_ts=time.time(),
        sha256=sha,
    )


def store_patch(
    model_id: str,
    rel_path: str,
    patch_text: str,
    *,
    source: str = "captured",
) -> Optional[OverlayRecord]:
    if not patch_text.strip():
        return None
    md = _model_dir(model_id)
    md.mkdir(parents=True, exist_ok=True)
    patch_file = md / _patch_filename(rel_path)
    patch_file.write_text(patch_text)
    sha = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
    idx = _load_index(model_id)
    idx[rel_path] = {
        "patch_file": patch_file.name,
        "captured_ts": time.time(),
        "sha256": sha,
        "line_count": patch_text.count("\n"),
        "classification": classify_path(rel_path),
        "source": source,
    }
    _save_index(model_id, idx)
    return OverlayRecord(
        model_id=model_id,
        rel_path=rel_path,
        patch_path=patch_file,
        captured_ts=time.time(),
        sha256=sha,
    )


def _filter_hunks(diff: str, *, pattern: str) -> str:
    rx = re.compile(pattern, re.IGNORECASE)
    lines = diff.splitlines(keepends=True)
    out: List[str] = []
    header: List[str] = []
    in_hunk = False
    hunk_buf: List[str] = []

    def flush_hunk():
        if hunk_buf and any(rx.search(ln) for ln in hunk_buf):
            out.extend(hunk_buf)

    for ln in lines:
        if ln.startswith("diff --git ") or ln.startswith("index ") or ln.startswith("--- ") or ln.startswith("+++ "):
            if in_hunk:
                flush_hunk()
                hunk_buf = []
                in_hunk = False
            header.append(ln)
            continue
        if ln.startswith("@@"):
            if in_hunk:
                flush_hunk()
                hunk_buf = []
            if header:
                out.extend(header)
                header = []
            in_hunk = True
            hunk_buf = [ln]
            continue
        if in_hunk:
            hunk_buf.append(ln)
    if in_hunk:
        flush_hunk()
    return "".join(out)

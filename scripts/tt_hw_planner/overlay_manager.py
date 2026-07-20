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
_INDEX_NAME = "index.json"
_OVERLAYS_HOME_ENV = "TT_HW_PLANNER_OVERLAYS_HOME"
_OVERLAYS_DIR: Optional[Path] = None
_overlays_dir_cache: Optional[Path] = None


def _main_repo_overlays_dir() -> Path:
    if _OVERLAYS_DIR is not None:
        return Path(_OVERLAYS_DIR)
    global _overlays_dir_cache
    if _overlays_dir_cache is not None:
        return _overlays_dir_cache
    env = os.environ.get(_OVERLAYS_HOME_ENV)
    if env:
        p = Path(env)
        resolved = p if p.name == "overlays" else p / "overlays"
        _overlays_dir_cache = resolved
        return resolved
    resolved: Optional[Path] = None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(_THIS_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        common = Path(out.stdout.strip())
        if not common.is_absolute():
            common = (_THIS_DIR / common).resolve()
        candidate = common.parent / "scripts" / "tt_hw_planner"
        if candidate.is_dir():
            resolved = candidate / "overlays"
    except Exception:
        resolved = None
    if resolved is None:
        resolved = _THIS_DIR / "overlays"
    os.environ[_OVERLAYS_HOME_ENV] = str(resolved)
    _overlays_dir_cache = resolved
    return resolved


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
    return _main_repo_overlays_dir() / _slug(model_id)


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


def persist_skip(model_id: str, comp_name: str, reason: str = "", *, category: str = "KERNEL_MISSING") -> None:
    """Add `comp_name` to the persistent skip-list — ONLY for verified
    KERNEL_MISSING gaps.

    Design rule: the torch reference runs every submodule on the same
    device (GPU). The only legitimate reason to keep a component on CPU
    is that a TTNN kernel it needs does not exist yet. Every other
    failure mode (TOOL_BUG, ITERATION_BUDGET, AGENT_STUCK,
    CONSTRAINT_MISMATCH, HF_ERROR) is tooling debt — the component
    should be retried on the next run, not permanently sidelined.

    Behavior:
      - category == "KERNEL_MISSING" → persisted; sticky (newer reason
        overwrites, but the entry isn't downgraded).
      - any other category → not persisted. Logs a one-line note for
        the caller's benefit; the component stays in the active queue
        and will be re-attempted next session.
    """
    new_category = (category or "").upper()
    if new_category != "KERNEL_MISSING":
        # Non-kernel-missing failures are no longer recorded as
        # permanent skips. The bringup loop's session-local state
        # decides when to stop retrying THIS run; nothing here.
        return
    skips = load_persistent_skips(model_id)
    if comp_name in skips:
        existing = skips[comp_name]
        if reason and reason != existing.get("reason"):
            existing["reason"] = reason
        existing["category"] = "KERNEL_MISSING"
        skips[comp_name] = existing
    else:
        skips[comp_name] = {
            "reason": reason or "TTNN kernel verified missing",
            "category": "KERNEL_MISSING",
            "captured_ts": time.time(),
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


def _locked_modules_path(model_id: str) -> Path:
    return _model_dir(model_id) / "locked_modules.json"


def load_locked_modules(model_id: str) -> Dict[str, dict]:
    """Return the persistent ``{comp_name: {reason, locked_ts}}`` dict for
    recomposed parents that must NEVER be decomposed again.

    A parent is locked once its children have all graduated and it has been
    restored as a whole-module target (recompose). From then on the only
    path forward is re-iterating the whole module — re-decomposition is
    forbidden. Durable in the overlay so the lock survives worktree
    cleanup and is honored by up / auto-up / promote alike.

    Empty dict if no file exists yet — never raises.
    """
    p = _locked_modules_path(model_id)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def persist_locked_module(model_id: str, comp_name: str, reason: str = "") -> None:
    """Add ``comp_name`` to the persistent locked-modules list for ``model_id``.

    Idempotent: re-locking an existing module preserves the original
    locked_ts."""
    listing = load_locked_modules(model_id)
    if comp_name in listing:
        return
    listing[comp_name] = {
        "reason": reason or "recomposed from graduated children; re-decomposition forbidden",
        "locked_ts": time.time(),
    }
    p = _locked_modules_path(model_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(listing, indent=2, sort_keys=True))


def remove_locked_module(model_id: str, comp_name: str) -> bool:
    """Drop a single entry from the locked-modules list. Returns True if
    the entry was present and removed."""
    listing = load_locked_modules(model_id)
    if comp_name not in listing:
        return False
    del listing[comp_name]
    p = _locked_modules_path(model_id)
    if listing:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(listing, indent=2, sort_keys=True))
    elif p.is_file():
        p.unlink()
    return True


def is_locked_module(model_id: str, comp_name: str) -> bool:
    """Fast lookup: is this module locked against re-decomposition?"""
    return comp_name in load_locked_modules(model_id)


def _alias_credits_path(model_id: str) -> Path:
    return _model_dir(model_id) / "alias_credits.json"


def load_alias_credits(model_id: str) -> Dict[str, dict]:
    """Return the persistent ``{comp_name: {canonical_path, twin, credited_ts}}``
    dict for components credited on-device because they are the SAME module as a
    graduated component (proven by identical resolved submodule_path).

    Empty dict if no file exists yet — never raises.
    """
    p = _alias_credits_path(model_id)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def persist_alias_credit(model_id: str, comp_name: str, canonical_path: str = "", twin: str = "") -> None:
    """Credit ``comp_name`` as on-device because it resolves to the same module
    (``canonical_path``) as a graduated component ``twin``. Idempotent."""
    listing = load_alias_credits(model_id)
    if comp_name in listing:
        return
    listing[comp_name] = {
        "canonical_path": canonical_path,
        "twin": twin,
        "credited_ts": time.time(),
    }
    p = _alias_credits_path(model_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(listing, indent=2, sort_keys=True))


def remove_alias_credit(model_id: str, comp_name: str) -> bool:
    """Drop a single alias-credit entry. Returns True if present and removed."""
    listing = load_alias_credits(model_id)
    if comp_name not in listing:
        return False
    del listing[comp_name]
    p = _alias_credits_path(model_id)
    if listing:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(listing, indent=2, sort_keys=True))
    elif p.is_file():
        p.unlink()
    return True


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


def _git_apply(
    patch_text: str, *, reverse: bool = False, check_only: bool = False, include: "Optional[List[str]]" = None
) -> Tuple[int, str]:
    args = ["git", "apply"]
    if reverse:
        args.append("--reverse")
    if check_only:
        args.append("--check")
    for pat in include or []:
        args += ["--include", pat]
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
    _overlays_dir = _main_repo_overlays_dir()
    if not _overlays_dir.is_dir():
        return rows
    model_dirs = [_model_dir(model_id)] if model_id else sorted([p for p in _overlays_dir.iterdir() if p.is_dir()])
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


_GRADUATION_SNAPSHOT_SUFFIXES = (".py.last_good_native", ".py.last_good_sharded")


def _prune_stale_graduation_snapshots(applied_rel: List[str], model_id: str) -> None:
    """Delete any restored `.py.last_good_native` / `.py.last_good_sharded` whose sibling stub is
    a torch wrapper. Without this, the bring-up gate reads the stale snapshot as "graduated" and
    the loop exits early with components still on CPU fallback."""
    if not applied_rel:
        return
    try:
        from scripts.tt_hw_planner.bringup_loop import _stub_body_is_native
    except Exception:
        return
    try:
        repo_root = _repo_root()
    except Exception:
        return
    pruned: List[str] = []
    for rel in applied_rel:
        matched_suffix = next((s for s in _GRADUATION_SNAPSHOT_SUFFIXES if rel.endswith(s)), None)
        if matched_suffix is None:
            continue
        snap_path = repo_root / rel
        stub_path = snap_path.with_suffix("")
        if not snap_path.is_file() or not stub_path.is_file():
            continue
        try:
            if _stub_body_is_native(stub_path):
                continue
        except Exception:
            continue
        try:
            snap_path.unlink()
            pruned.append(rel)
        except OSError as exc:
            print(
                f"[overlay] apply_for({model_id}): could not prune stale snapshot {rel}: {exc}",
                file=sys.stderr,
            )
    if pruned:
        print(
            f"[overlay] apply_for({model_id}): pruned {len(pruned)} stale graduation "
            f"snapshot(s) whose sibling stub is a torch wrapper: {', '.join(pruned)}",
            file=sys.stderr,
        )


_GRADUATION_STATE_INCLUDES = (
    "*bringup_cc_state.json",
    "*bringup_status.json",
    "*last_good_native",
    "*last_good_sharded",
)


def _salvage_graduation_state(patch_text: str) -> List[str]:
    """Apply ONLY the graduation-state files carried by ``patch_text``
    (.bringup_cc_state.json / bringup_status.json / .py.last_good_native /
    .py.last_good_sharded), ignoring everything else in the patch.

    A captured demo's graduation state is bundled in the directory-level patch,
    which is skipped whole when its target directory has any drift ("already
    exists") — so an overlay-materialized demo ends up with stubs + tests but no
    graduation markers, and emit-e2e / optimize then see zero graduated modules.
    These state files are new-file additions that apply cleanly even when the
    rest of the patch conflicts, so salvaging them makes the restored demo
    optimize-ready and emit-e2e-ready. Best-effort; returns the repo-relative
    files written (empty if nothing salvageable)."""
    include = list(_GRADUATION_STATE_INCLUDES)
    rc_check, _ = _git_apply(patch_text, check_only=True, include=include)
    if rc_check != 0:
        return []
    rc, _ = _git_apply(patch_text, include=include)
    if rc != 0:
        return []
    import fnmatch as _fn

    written: List[str] = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            p = line[len("+++ b/") :].strip()
            if any(_fn.fnmatch(p, pat) for pat in _GRADUATION_STATE_INCLUDES):
                written.append(p)
    return written


def apply_for(model_id: str) -> Tuple[int, List[str]]:
    """Apply every overlay patch registered under ``model_id``.

    Returns ``(count_applied, applied_rel_paths)``. Skips a patch when
    ``git apply --check`` says it can't apply (target file drifted)
    AND prints a one-line WARN so the operator can see WHY the count
    is low — the previous silent ``continue`` hid cases where an
    entire scope had no applying patches (e.g. _shared overlays
    captured against an old model_config.py that no longer matches).

    Also prints a WARN when ``index.json`` doesn't exist or is empty,
    since callers (cli.py:_cmd_up_isolated, instrumentation
    _apply_overlays_for_active_model) treat ``n == 0`` as "no
    overlays registered" — silently true when the index file was
    deleted from disk, which has been a recurring source of
    "_shared overlays not applied" bugs.
    """
    md = _model_dir(model_id)
    idx_path = _index_path(model_id)
    idx = _load_index(model_id)
    applied: List[str] = []
    if not idx:
        # Distinguish "scope intentionally empty" from "index file
        # missing." The latter is usually a deletion / missing-checkout
        # bug the operator wants to know about.
        if not idx_path.is_file():
            print(
                f"[overlay] apply_for({model_id}): no index.json at "
                f"{idx_path} — scope appears uninitialized OR the index "
                f"was deleted from disk. No patches will apply.",
                file=sys.stderr,
            )
        return 0, applied
    skipped_rel: List[str] = []
    for rel, meta in idx.items():
        patch_text = (md / meta["patch_file"]).read_text()
        rc_check, check_err = _git_apply(patch_text, check_only=True)
        if rc_check != 0:
            skipped_rel.append(rel)
            print(
                f"[overlay] apply_for({model_id}): skipped {rel} — "
                f"git apply --check returned rc={rc_check}. The target "
                f"file has drifted since the patch was captured. "
                f"Tail: {check_err.strip().splitlines()[-1] if check_err.strip() else '(empty)'}",
                file=sys.stderr,
            )
            _salvaged = _salvage_graduation_state(patch_text)
            if _salvaged:
                applied.extend(_salvaged)
                print(
                    f"[overlay] apply_for({model_id}): salvaged {len(_salvaged)} graduation-state "
                    f"file(s) from skipped {rel} — restores the graduated set emit-e2e/optimize read",
                    file=sys.stderr,
                )
            continue
        rc, err = _git_apply(patch_text)
        if rc == 0:
            applied.append(rel)
        else:
            print(
                f"[overlay] apply failed for {rel} (model={model_id}): {err.strip()}",
                file=sys.stderr,
            )
    if not applied and skipped_rel:
        print(
            f"[overlay] apply_for({model_id}): 0/{len(idx)} patches "
            f"applied — every patch was skipped due to drift. The scope "
            f"may need re-capture against current HEAD.",
            file=sys.stderr,
        )
    _prune_stale_graduation_snapshots(applied, model_id)
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


def drop_scope(model_id: str) -> Tuple[int, List[str]]:
    """Drop EVERY overlay registered under ``model_id``.

    Iterates the model's overlay index, deletes each patch file, then
    removes the index + scope directory. Returns ``(count_dropped,
    dropped_rel_paths)``. Returns ``(0, [])`` when no overlays exist
    for the scope (no-op, no error).

    Used by the CLI's ``overlay-drop <model_id>`` (rel_path omitted)
    form to give operators a single-call wipe instead of looping
    :func:`drop` per file. Especially useful when overlays were
    captured under a broken-gate regime and the operator wants a
    clean slate.
    """
    idx = _load_index(model_id)
    if not idx:
        return 0, []
    dropped: List[str] = []
    md = _model_dir(model_id)
    for rel_path, entry in list(idx.items()):
        patch_file = md / entry.get("patch_file", "")
        try:
            if patch_file.is_file():
                patch_file.unlink()
        except OSError:
            pass
        dropped.append(rel_path)
    _index_path(model_id).unlink(missing_ok=True)
    try:
        md.rmdir()
    except OSError:
        pass
    return len(dropped), dropped


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

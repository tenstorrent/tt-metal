"""Brain primitive: at end of a successful auto-iterate run, sync the
brain's graduated work product from the worktree back to the main repo.

THE PROBLEM this solves
=======================
Auto-iterate runs in an isolated git worktree under ``/tmp/`` so the
LLM and brain can edit files freely without contaminating the main
repo. At end of a run, the existing overlay-capture mechanism stores
patches and applies them on the NEXT run via overlay-apply.

But overlay-apply has been observed (2026-05-30, SAM2) to leave the
main tree with SCAFFOLD-STAGE stubs rather than the brain's final
iterated stubs. Result: the main tree fails pytest on components that
PASS in the worktree.

This primitive bypasses the overlay system for the graduated stubs:
the brain explicitly writes its work product to the main tree at the
moment it KNOWS the work is good (component graduated + final pytest
passed). Simple, direct, robust against overlay-system gaps.

WHEN to call
============
At end of ``_run_auto_iterate_loop``, after final pytest is observed
to pass, ONLY for components in ``graduated_this_run``. Each component
is synced atomically (single ``shutil.copy2``). Non-fatal: any error
logs a warning and continues, since the worktree is preserved on
failure and the user can recover manually.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass
class SyncResult:
    """Outcome of :func:`sync_graduated_to_main_tree`.

    Attributes:
        synced: components whose stub (and optional snapshot) was
            copied to the main tree.
        skipped: components that couldn't be synced (missing source,
            same-file copy, etc.) — paired with a reason in `notes`.
        notes: per-component trace lines, e.g. ``"video_layer_norm: synced"``
            or ``"video_layer_norm: skipped (source missing)"``. Surfaced
            to the user.
        main_tree_path: the resolved source_repo path (None if we
            couldn't determine it — typically means we're not running
            in a worktree).
    """

    synced: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    main_tree_path: Optional[Path] = None


def _read_source_repo_from_session(worktree_root: Path) -> Optional[Path]:
    """Read the brain's session.json to find the main tree path.

    Returns None if the session file is missing or malformed — the
    caller treats that as "we're not in a worktree" and skips the
    sync entirely (the worktree state IS the main tree).
    """
    sess = worktree_root / ".tt_hw_planner_session.json"
    if not sess.is_file():
        return None
    try:
        data = json.loads(sess.read_text())
    except Exception:
        return None
    raw = data.get("source_repo")
    if not isinstance(raw, str) or not raw:
        return None
    p = Path(raw)
    if not p.is_dir():
        return None
    return p


def sync_graduated_to_main_tree(
    *,
    worktree_root: Path,
    demo_subpath: str,
    graduated_components: Sequence[str],
    safe_id_fn,
) -> SyncResult:
    """Copy each graduated component's stub from the worktree's demo
    dir to the main tree's demo dir.

    Parameters:
        worktree_root: the cwd of the auto-iterate run. Normally a
            worktree under ``/tmp/``; can be the main tree itself if
            ``--isolation none`` was used (in which case this is a
            no-op).
        demo_subpath: the demo dir relative to the repo root,
            e.g. ``models/demos/vision/segmentation/sam2_hiera_tiny``.
        graduated_components: brain's ``graduated_this_run`` list.
            Only these get synced — never sync stubs we don't trust.
        safe_id_fn: ``bringup_loop._safe_id`` (passed in to avoid a
            heavyweight import from this brain module).

    Returns a :class:`SyncResult` summarizing what happened. Always
    non-fatal — any per-component error is logged in ``notes`` and
    the next component is attempted.
    """
    result = SyncResult()

    main_tree = _read_source_repo_from_session(worktree_root)
    result.main_tree_path = main_tree

    if main_tree is None:
        result.notes.append(
            "no .tt_hw_planner_session.json with source_repo — not in a worktree, "
            "or session metadata missing; skipping sync (worktree IS the main tree)"
        )
        return result

    if worktree_root.resolve() == main_tree.resolve():
        result.notes.append(f"worktree_root ({worktree_root}) == main_tree ({main_tree}); " "no sync needed")
        return result

    if not graduated_components:
        result.notes.append("no graduated components this run; nothing to sync")
        return result

    src_demo = worktree_root / demo_subpath
    dst_demo = main_tree / demo_subpath
    if not src_demo.is_dir():
        result.notes.append(f"worktree demo dir missing: {src_demo}")
        return result
    if not dst_demo.is_dir():
        result.notes.append(f"main-tree demo dir missing: {dst_demo}")
        return result

    src_stubs = src_demo / "_stubs"
    dst_stubs = dst_demo / "_stubs"
    if not src_stubs.is_dir():
        result.notes.append(f"worktree _stubs dir missing: {src_stubs}")
        return result
    dst_stubs.mkdir(parents=True, exist_ok=True)

    for comp in sorted(set(graduated_components)):
        safe = safe_id_fn(comp)
        src_stub = src_stubs / f"{safe}.py"
        dst_stub = dst_stubs / f"{safe}.py"

        if not src_stub.is_file():
            result.skipped.append(comp)
            result.notes.append(f"{comp}: skipped (worktree stub missing: {src_stub.name})")
            continue

        try:
            shutil.copy2(src_stub, dst_stub)
        except Exception as exc:
            result.skipped.append(comp)
            result.notes.append(f"{comp}: copy failed ({type(exc).__name__}: {exc})")
            continue

        # Best-effort: also copy the graduation snapshot. Without this,
        # the next `up` run won't recognize the component as
        # already-graduated and will re-iterate it.
        snap_name = f"{safe}.py.last_good_native"
        src_snap = src_stubs / snap_name
        if src_snap.is_file():
            try:
                shutil.copy2(src_snap, dst_stubs / snap_name)
            except Exception as exc:
                result.notes.append(f"{comp}: snapshot copy failed (stub did copy) " f"({type(exc).__name__}: {exc})")

        result.synced.append(comp)
        result.notes.append(f"{comp}: synced")

    return result


@dataclass
class DemoSyncResult:
    """Richer return shape for :func:`sync_demo_to_main_tree`.

    Distinguishes "no-op because not in a worktree" (a legitimate
    case — user ran with --isolation none) from "tried to sync but
    failed" (a real warning the caller must surface). Without this
    distinction the caller reports SUCCESS even when the brain's
    recovery work product silently failed to land in main tree.
    (B-FIX #12, 2026-05-31.)

    Attributes:
        status: one of ``"synced"``, ``"noop_not_in_worktree"``,
            ``"noop_worktree_is_main_tree"``, ``"sync_failed"``,
            ``"source_missing"``.
        synced_path: destination on success, None otherwise.
        reason: human-readable explanation surfaced in banners.
    """

    status: str
    synced_path: Optional[Path] = None
    reason: str = ""


def sync_demo_to_main_tree(*, worktree_demo_path: Path) -> DemoSyncResult:
    """Sync a modified demo.py from worktree to main tree.

    Called by the brain's demo-recovery path: when the brain modifies
    the demo (e.g. disables a broken wired component), the modified
    file needs to land in main tree so the user can actually run it
    without manual re-applying the brain's fix.

    Reads ``.tt_hw_planner_session.json`` (in the worktree root, walking
    upward) to find source_repo. Returns a :class:`DemoSyncResult`
    with explicit ``status`` so the caller can distinguish "no-op"
    from "tried-and-failed" — the latter must be surfaced to the user
    even though both used to return None.
    """
    if not worktree_demo_path.is_file():
        return DemoSyncResult(
            status="source_missing",
            reason=f"worktree demo path does not exist: {worktree_demo_path}",
        )

    # Walk upward from the demo to find the worktree root (where
    # .tt_hw_planner_session.json lives).
    worktree_root: Optional[Path] = None
    for ancestor in worktree_demo_path.resolve().parents:
        if (ancestor / ".tt_hw_planner_session.json").is_file():
            worktree_root = ancestor
            break
    if worktree_root is None:
        return DemoSyncResult(
            status="noop_not_in_worktree",
            reason="no .tt_hw_planner_session.json found walking up from demo path; not in an isolated worktree",
        )

    main_tree = _read_source_repo_from_session(worktree_root)
    if main_tree is None:
        return DemoSyncResult(
            status="noop_not_in_worktree",
            reason="session.json missing or has no source_repo",
        )
    if worktree_root.resolve() == main_tree.resolve():
        return DemoSyncResult(
            status="noop_worktree_is_main_tree",
            reason=f"worktree_root ({worktree_root}) == main_tree; no sync needed",
        )

    rel = worktree_demo_path.resolve().relative_to(worktree_root.resolve())
    dst = main_tree / rel
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(worktree_demo_path, dst)
    except Exception as exc:
        return DemoSyncResult(
            status="sync_failed",
            reason=f"copy failed: {type(exc).__name__}: {exc}",
        )
    return DemoSyncResult(
        status="synced",
        synced_path=dst,
        reason=f"copied {rel} → {dst}",
    )


__all__ = [
    "DemoSyncResult",
    "SyncResult",
    "sync_demo_to_main_tree",
    "sync_graduated_to_main_tree",
]

"""Phase 2 orchestrator: walk a model's persistent skip-list and route each
entry to the right unblock strategy.

Components end up on the skip-list when the test harness can't even invoke
them (ModuleList with no forward, wrong synthesized arg shapes, etc.). Once
listed, they stay listed forever -- the standard ``up`` loop excludes them
from candidate selection so they cost no LLM budget.

This command is the retroactive cleanup pass. It's designed to run AFTER
the standard auto-iterate loop has graduated all currently-tested
components, so Phase 2 work doesn't compete with Phase 1.

Per-category routing:

  MODULELIST       -- structurally untestable as standalone (no forward
                      method). Backup + drop the stub + test files; remove
                      from skip-list. The parent component's PCC test
                      already covers this functionality.

  MISSING_ARG /    -- test scaffold synthesized wrong inputs. Retry capture
  SHAPE_MISMATCH      with ``TT_PLANNER_AUTO_ONBOARD_DRIVER=1`` so the
                      generic capture chain can lean on an LLM-drafted
                      driver. If capture now succeeds, regenerate the test
                      scaffold from real inputs and remove from skip-list.

  DRIVER_FAILURE   -- explicit auto-onboard driver invocation per entry.

  UNKNOWN          -- no automatic action. Reported in the dry-run output
                      so a human can decide.

All operations are reversible: file removals back up to
``_phase2_dropped/<comp>.{py,test_py}.bak`` and the skip-list edit
is keyed per-entry so unrelated entries are untouched on partial
failure.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


_REASON_PATTERNS: List[Tuple[str, List[str]]] = [
    ("MODULELIST", [r"ModuleList no forward", r"Module \[ModuleList\]"]),
    ("MISSING_ARG", [r"missing.*required.*positional", r"missing positional arg"]),
    (
        "SHAPE_MISMATCH",
        [r"permute.*dim mismatch", r"weight shape mismatch", r"groups=\d+", r"shape '.*' is invalid"],
    ),
    ("DRIVER_FAILURE", [r"capture driver", r"no driver matched"]),
]


def _classify_skip_reason(reason: str) -> str:
    """Map a free-text skip reason to one of the routing categories.

    Returns 'UNKNOWN' if no pattern matches -- never raises so a single
    malformed entry can't crash the orchestrator.
    """
    if not reason:
        return "UNKNOWN"
    for category, patterns in _REASON_PATTERNS:
        for pat in patterns:
            if re.search(pat, reason, re.IGNORECASE):
                return category
    return "UNKNOWN"


def _safe_id(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_") or "unknown"


def _backup_and_remove(file_path: Path, backup_dir: Path) -> bool:
    """Move ``file_path`` to ``backup_dir/<filename>.bak``. Returns True if a
    file was moved, False if the source didn't exist. Backup directory is
    created if needed."""
    if not file_path.is_file():
        return False
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / (file_path.name + ".bak")
    shutil.move(str(file_path), str(dest))
    return True


def _drop_component_files(demo_dir: Path, comp: str, *, dry_run: bool) -> List[str]:
    """Back up and remove the per-component stub + PCC test files. Returns
    a list of human-readable lines describing what was/would be moved.
    Backups land under ``_phase2_dropped/<safe>.{py,test_py}.bak``.
    """
    safe = _safe_id(comp)
    stub_path = demo_dir / "_stubs" / f"{safe}.py"
    test_path = demo_dir / "tests" / "pcc" / f"test_{safe}.py"
    backup_dir = demo_dir / "_phase2_dropped"
    msgs: List[str] = []
    for src in (stub_path, test_path):
        if not src.is_file():
            continue
        if dry_run:
            msgs.append(f"would back up {src} -> {backup_dir / (src.name + '.bak')}")
        else:
            if _backup_and_remove(src, backup_dir):
                msgs.append(f"backed up {src.name} -> _phase2_dropped/")
    return msgs


_CAPTURE_ARTIFACT_FILES: Tuple[str, ...] = ("args.pt", "kwargs.pt", "output.pt")


def _verify_capture_artifacts(demo_dir: Path, comp: str) -> Tuple[bool, str]:
    """Check that the capture-inputs step actually produced artifacts for
    ``comp``. Returns (ok, message). Used by ``_retry_capture`` to detect
    the case where ``cmd_capture_inputs`` returns rc=0 but the specific
    component wasn't captured (the driver's forward path didn't reach it).

    Generic across all models: checks the same canonical paths the
    test scaffold's ``_maybe_load_captured`` reads from.
    """
    safe = _safe_id(comp)
    comp_dir = demo_dir / "_captured" / safe
    if not comp_dir.is_dir():
        return False, f"no _captured/{safe}/ directory created"
    missing = [f for f in _CAPTURE_ARTIFACT_FILES if not (comp_dir / f).is_file()]
    if missing:
        return False, f"_captured/{safe}/ exists but missing {missing}"
    return True, f"_captured/{safe}/ has all artifacts ({', '.join(_CAPTURE_ARTIFACT_FILES)})"


def _retry_capture(
    model_id: str,
    comp: str,
    *,
    dry_run: bool,
    demo_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    """Attempt to re-capture inputs for ``comp`` with the auto-onboard
    driver path enabled. Returns ``(succeeded, message)``.

    'Succeeded' means BOTH ``cmd_capture_inputs`` returned rc=0 AND the
    capture artifacts (``args.pt`` / ``kwargs.pt`` / ``output.pt``) now
    exist for this specific component. rc=0 alone is necessary but not
    sufficient -- the auto-onboard driver may run successfully and yet
    never invoke the target component's forward (because that component
    is cold in the driver's exercised code path). Verifying artifacts
    after the call catches that false-positive.

    Imports are deferred so this module stays cheap to import in tests
    that monkey-patch the capture path.
    """
    if dry_run:
        return False, f"would invoke capture-inputs --component {comp} with TT_PLANNER_AUTO_ONBOARD_DRIVER=1"

    if demo_dir is None:
        from ..bringup_loop import find_demo_dir

        demo_dir = find_demo_dir(model_id)
        if demo_dir is None:
            return False, "demo dir not resolvable (cannot verify capture artifacts)"

    from ..cli import cmd_capture_inputs

    prev = os.environ.get("TT_PLANNER_AUTO_ONBOARD_DRIVER")
    os.environ["TT_PLANNER_AUTO_ONBOARD_DRIVER"] = "1"
    args = argparse.Namespace(
        model_id=model_id,
        component=comp,
        no_upgrade_tests=False,
        image_size=None,
    )
    try:
        rc = cmd_capture_inputs(args)
    except Exception as exc:
        return False, f"capture-inputs raised: {type(exc).__name__}: {exc}"
    finally:
        if prev is None:
            os.environ.pop("TT_PLANNER_AUTO_ONBOARD_DRIVER", None)
        else:
            os.environ["TT_PLANNER_AUTO_ONBOARD_DRIVER"] = prev

    if rc != 0:
        return False, f"capture-inputs rc={rc} (non-zero)"

    artifacts_ok, artifacts_msg = _verify_capture_artifacts(demo_dir, comp)
    if not artifacts_ok:
        return False, f"capture-inputs rc=0 but {artifacts_msg} (false-positive)"

    return True, f"capture-inputs rc=0 and {artifacts_msg}"


def cmd_tackle_skipped(args: argparse.Namespace) -> int:
    """Walk the persistent skip-list for ``args.model_id`` and route each
    entry to its unblock strategy. With ``--dry-run`` no files are moved
    and no LLM calls are made -- the command only prints what it would do.
    """
    from ..overlay_manager import load_persistent_skips, remove_persistent_skip
    from ..bringup_loop import find_demo_dir

    model_id = args.model_id
    skips = load_persistent_skips(model_id)
    if not skips:
        print(f"no persistent skips on file for {model_id} — nothing to tackle")
        return 0

    demo_dir = find_demo_dir(model_id)
    if demo_dir is None:
        print(f"demo dir not found for {model_id} — has scaffold been run?", file=sys.stderr)
        return 2

    by_category: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for comp, info in skips.items():
        category = _classify_skip_reason(str(info.get("reason", "")))
        by_category[category].append((comp, str(info.get("reason", ""))))

    dry_run = bool(getattr(args, "dry_run", False))
    only_modulelist = bool(getattr(args, "only_modulelist", False))
    only_capture = bool(getattr(args, "only_capture", False))

    print(f"Phase 2 tackle-skipped for {model_id}")
    print(f"  demo_dir   : {demo_dir}")
    print(f"  dry-run    : {dry_run}")
    print(f"  total skips: {len(skips)}")
    for cat, entries in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(entries):2}x  {cat}")
        for comp, reason in entries:
            print(f"         {comp}: {reason}")
    print()

    dropped: List[str] = []
    recaptured_ok: List[str] = []
    recaptured_failed: List[str] = []
    unknown: List[str] = []

    if not only_capture:
        from ..overlay_manager import persist_no_emit_test

        for comp, reason in by_category.get("MODULELIST", []):
            print(f"[MODULELIST] {comp}")
            msgs = _drop_component_files(demo_dir, comp, dry_run=dry_run)
            for m in msgs:
                print(f"    {m}")
            if not dry_run:
                if remove_persistent_skip(model_id, comp):
                    print(f"    removed from persistent skip-list")
                persist_no_emit_test(model_id, comp, reason=f"ModuleList drop ({reason})")
                print(f"    added to persistent no-emit-tests list (scaffold will not re-emit)")
                dropped.append(comp)

    if not only_modulelist:
        capture_categories = ("MISSING_ARG", "SHAPE_MISMATCH", "DRIVER_FAILURE")
        for cat in capture_categories:
            for comp, _reason in by_category.get(cat, []):
                print(f"[{cat}] {comp}")
                ok, msg = _retry_capture(model_id, comp, dry_run=dry_run, demo_dir=demo_dir)
                print(f"    {msg}")
                if ok and not dry_run:
                    if remove_persistent_skip(model_id, comp):
                        print(f"    removed from persistent skip-list")
                    recaptured_ok.append(comp)
                else:
                    if not dry_run:
                        recaptured_failed.append(comp)

    for comp, reason in by_category.get("UNKNOWN", []):
        print(f"[UNKNOWN] {comp}: {reason}")
        print(f"    no automatic action -- inspect manually")
        unknown.append(comp)

    print()
    print("Phase 2 summary:")
    print(f"  ModuleList dropped              : {len(dropped)}")
    print(f"  re-captured (now testable)      : {len(recaptured_ok)}")
    print(f"  re-capture failed (still stuck) : {len(recaptured_failed)}")
    print(f"  unknown reason (no action)      : {len(unknown)}")
    if dry_run:
        print()
        print("DRY RUN: no files modified, no LLM invocations made.")
        print("Re-run without --dry-run to apply.")

    if recaptured_failed or unknown:
        return 1
    return 0


def run_phase2_stage(
    model_id: str,
    demo_dir: Path,
    *,
    only_modulelist: bool = False,
    only_capture: bool = False,
    sep: str = "=" * 78,
) -> Tuple[int, List[str], List[str]]:
    """Phase 2 stage callable from inside an active worktree (used by `up --phase2`).

    Unlike ``cmd_tackle_skipped`` which is the CLI command, this function:
      - Operates on a *known* live demo_dir (passed in, not resolved)
      - Returns structured outputs so the caller can re-run PCC on
        newly-unblocked components and persist via overlay-capture
      - Skips all dry-run handling -- the caller already knows it's
        committing to changes

    Returns ``(rc, dropped, recaptured_ok)`` where:
      - rc is 0 if every entry was either dropped or successfully
        unblocked, 1 if any remained stuck
      - dropped is the list of ModuleList components whose files moved
        to ``_phase2_dropped/``
      - recaptured_ok is the list of components whose capture now
        succeeded (the caller can re-run PCC on these to see if they
        graduate)
    """
    from ..overlay_manager import load_persistent_skips, remove_persistent_skip

    print(sep)
    print(f"  Phase 2 stage for {model_id}")
    print(sep)

    skips = load_persistent_skips(model_id)
    if not skips:
        print("  no persistent skips on file — nothing to do")
        return 0, [], []

    by_category: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for comp, info in skips.items():
        category = _classify_skip_reason(str(info.get("reason", "")))
        by_category[category].append((comp, str(info.get("reason", ""))))

    print(f"  total skips: {len(skips)}")
    for cat, entries in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
        print(f"    {len(entries):2}x  {cat}")

    dropped: List[str] = []
    recaptured_ok: List[str] = []
    recaptured_failed: List[str] = []

    if not only_capture:
        from ..overlay_manager import persist_no_emit_test

        for comp, reason in by_category.get("MODULELIST", []):
            print(f"  [MODULELIST] {comp}")
            msgs = _drop_component_files(demo_dir, comp, dry_run=False)
            for m in msgs:
                print(f"      {m}")
            if remove_persistent_skip(model_id, comp):
                print(f"      removed from persistent skip-list")
            persist_no_emit_test(model_id, comp, reason=f"ModuleList drop ({reason})")
            print(f"      added to persistent no-emit-tests list (scaffold will not re-emit)")
            dropped.append(comp)

    if not only_modulelist:
        capture_categories = ("MISSING_ARG", "SHAPE_MISMATCH", "DRIVER_FAILURE")
        for cat in capture_categories:
            for comp, _reason in by_category.get(cat, []):
                print(f"  [{cat}] {comp}")
                ok, msg = _retry_capture(model_id, comp, dry_run=False, demo_dir=demo_dir)
                print(f"      {msg}")
                if ok:
                    if remove_persistent_skip(model_id, comp):
                        print(f"      removed from persistent skip-list")
                    recaptured_ok.append(comp)
                else:
                    recaptured_failed.append(comp)

    print(f"  Phase 2 stage summary:")
    print(f"    ModuleList dropped         : {len(dropped)}")
    print(f"    capture-retried (success)  : {len(recaptured_ok)}")
    print(f"    capture-retried (failed)   : {len(recaptured_failed)}")
    print(f"    UNKNOWN entries left alone : {len(by_category.get('UNKNOWN', []))}")
    print(sep)

    rc = 1 if (recaptured_failed or by_category.get("UNKNOWN")) else 0
    return rc, dropped, recaptured_ok

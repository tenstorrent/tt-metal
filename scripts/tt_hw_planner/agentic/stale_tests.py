"""Brain primitive: detect when a failing pytest is testing a "phantom"
component — one whose actual work has been moved elsewhere (typically
via decomposition into children).

Without this primitive, a parent that was decomposed leaves its OLD
standalone PCC test in place. The final pytest sweep discovers it via
filesystem glob, runs it, and reports a phantom failure — even though
the actual work IS on device via the children.

The brain's decision flow:
  1. A test fails. The loop asks: is this a phantom?
  2. Brain checks no_emit_tests.json (the authoritative "parent was
     decomposed" record) for this component.
  3. If the component is registered as no_emit with a decomposition
     reason, the verdict is STALE_DECOMPOSED_PARENT → archive the
     test file, re-run pytest without it.

The mechanical safety net in decomposition_consumer.py archives the
test at decomposition WRITE time. This brain primitive catches:
  * Decompositions performed by external tooling
  * Old decompositions from before the consumer learned cleanup
  * Manual decomposition workflows where the consumer didn't run
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class StaleVerdict:
    """Brain's per-component stale-test verdict.

    Attributes:
        is_stale: whether this failure is a phantom from a stale test.
        action: what the caller should do — currently "archive_test"
            or "" (no action when not stale).
        reason: human-readable trace surfaced in banners + telemetry.
    """

    is_stale: bool
    action: str
    reason: str


def detect_stale_decomposed_test(
    *,
    component: str,
    no_emit_tests: dict,
) -> StaleVerdict:
    """Brain's decision: is a pytest failure on ``component`` a phantom
    from a stale test file left over after decomposition?

    Caller supplies the loaded no_emit_tests dict (from
    overlay_manager.load_no_emit_tests) rather than the brain
    re-reading it — keeps the brain pure / testable in isolation.

    A component qualifies as STALE_DECOMPOSED_PARENT when its entry in
    no_emit_tests has a decomposition-related reason. ModuleList-drop
    entries (`reason="ModuleList drop ..."`) are NOT classified stale
    because those tests are smoke-pass — they shouldn't be failing in
    the first place, and if they do it's a real bug.
    """
    if not component:
        return StaleVerdict(is_stale=False, action="", reason="empty-component")

    entry = no_emit_tests.get(component) if isinstance(no_emit_tests, dict) else None
    if not entry or not isinstance(entry, dict):
        return StaleVerdict(
            is_stale=False,
            action="",
            reason=f"`{component}` not in no_emit_tests — not a decomposed parent",
        )

    reason = str(entry.get("reason") or "").lower()
    # Decomposition-related reasons. The only producer of these
    # entries today is decomposition_consumer.py which writes
    # "decomposition consumer split parent into children at <date>"
    # (G2-FIX: prior versions matched two additional strings with no
    # producers anywhere in the repo — removed to avoid the misleading
    # impression of broader coverage).
    decomposition_markers = ("decomposition consumer split",)
    if any(marker in reason for marker in decomposition_markers):
        original_reason = str(entry.get("reason") or "").strip()
        return StaleVerdict(
            is_stale=True,
            action="archive_test",
            reason=(
                f"`{component}` is a decomposed parent (no_emit reason: "
                f'"{original_reason}") — its standalone test is '
                f"phantom; the children carry the real work"
            ),
        )

    return StaleVerdict(
        is_stale=False,
        action="",
        reason=(
            f"`{component}` is in no_emit_tests but reason is not "
            f"decomposition-related ({reason!r}) — failure is real"
        ),
    )


def archive_stale_test(*, demo_dir: Path, component: str, safe_id: str) -> Optional[Path]:
    """Move a component's stale standalone PCC test to a ``.stale_after_decomposition``
    sibling so the next pytest sweep skips it. Returns the archived path,
    or None if there was no test file to archive.

    Idempotent: if the file is already archived, returns None silently.
    S3-FIX: also returns None when an archive already exists at the
    destination — without this, ``rename()`` silently clobbers the prior
    archive on POSIX systems.
    """
    test_file = demo_dir / "tests" / "pcc" / f"test_{safe_id}.py"
    if not test_file.is_file():
        return None
    stale_path = test_file.with_suffix(".py.stale_after_decomposition")
    if stale_path.exists():
        # Already archived previously. Leave the existing archive alone
        # and DON'T clobber it. Leave the new live test in place so
        # the user can manually compare/decide.
        return None
    test_file.rename(stale_path)
    return stale_path


def restore_stale_test(*, demo_dir: Path, component: str, safe_id: str) -> Optional[Path]:
    """Restore a parent's ``.stale_after_decomposition`` test back to its
    live ``test_<safe_id>.py`` path so the whole module is tested again.

    The inverse of :func:`archive_stale_test`, used by the recompose path
    once a decomposed parent's children have all graduated. Returns the
    restored live path, or None if there was no archived test (or a live
    test already exists).
    """
    tests_dir = demo_dir / "tests" / "pcc"
    live = tests_dir / f"test_{safe_id}.py"
    stale_path = live.with_suffix(".py.stale_after_decomposition")
    if not stale_path.is_file():
        return None
    if live.exists():
        return None
    stale_path.rename(live)
    return live


def restore_orphaned_stale_tests(*, model_id: str, demo_dir: Path) -> List[str]:
    """Self-heal: restore any ``*.stale_after_decomposition`` test whose live
    counterpart is missing AND whose component is NOT in ``no_emit_tests``.

    Under normal decomposition/recompose flow, a stale archive and a no_emit
    entry are created together and cleared together — but partial-failure
    decomposition runs, overlay resets, or a scaffold rerun on top of a
    partially-restored state can leave a component in the corrupted "stale
    file exists, not suppressed, not live" state. In that state the gate
    has no test to run on the parent, never selects it, and its attempt
    count stays 0 forever.

    Only restores when the parent is NOT currently in no_emit — a
    deliberately-decomposed parent's stale test is left in place.
    """
    from ..overlay_manager import load_no_emit_tests

    pcc_dir = demo_dir / "tests" / "pcc"
    if not pcc_dir.is_dir():
        return []
    no_emit = set(load_no_emit_tests(model_id).keys())

    restored: List[str] = []
    for stale in pcc_dir.glob("test_*.py.stale_after_decomposition"):
        live = stale.with_suffix("")
        if live.exists():
            continue
        comp_safe = live.stem[len("test_") :] if live.stem.startswith("test_") else live.stem
        if comp_safe in no_emit:
            continue
        try:
            stale.rename(live)
        except OSError:
            continue
        restored.append(comp_safe)
    return restored


__all__ = [
    "StaleVerdict",
    "archive_stale_test",
    "detect_stale_decomposed_test",
    "restore_orphaned_stale_tests",
    "restore_stale_test",
]

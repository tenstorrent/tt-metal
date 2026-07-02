"""Tests for the brain's stale-decomposed-test detector.

The brain (agentic.stale_tests) decides whether a pytest failure is a
phantom from a stale parent test file left over after decomposition.
This was identified 2026-05-30 as the cause of SAM2's reported
`OUTCOME: FAIL rc=1` despite all real work being on device — the
parent `video_memory_encoder` was decomposed into children that all
pass, but its old test file kept failing."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.agentic.stale_tests import (
    StaleVerdict,
    archive_stale_test,
    detect_stale_decomposed_test,
)


def _decomp_entry(when: str = "2026-05-30") -> dict:
    return {
        "captured_ts": 1780171249.0,
        "reason": f"decomposition consumer split parent into children at {when}",
    }


def _modulelist_drop_entry() -> dict:
    return {
        "captured_ts": 1780010424.0,
        "reason": "ModuleList drop (harness: ModuleList no forward (v13))",
    }


# ---------------------------------------------------------------------------
# Verdict shape
# ---------------------------------------------------------------------------


def test_verdict_shape() -> None:
    v = StaleVerdict(is_stale=True, action="archive_test", reason="test")
    assert v.is_stale is True
    assert v.action == "archive_test"
    assert v.reason == "test"


# ---------------------------------------------------------------------------
# Positive: decomposed parent classified as stale
# ---------------------------------------------------------------------------


def test_decomposed_parent_classified_stale() -> None:
    no_emit = {"video_memory_encoder": _decomp_entry()}
    v = detect_stale_decomposed_test(
        component="video_memory_encoder",
        no_emit_tests=no_emit,
    )
    assert v.is_stale is True
    assert v.action == "archive_test"
    assert "decomposed parent" in v.reason
    assert "video_memory_encoder" in v.reason


# ---------------------------------------------------------------------------
# Negative: ModuleList drops are NOT classified stale (different cause)
# ---------------------------------------------------------------------------


def test_modulelist_drop_not_classified_stale() -> None:
    """ModuleList drops are smoke-pass tests. If one fails, it's a real
    bug — not a phantom. Brain must NOT archive these."""
    no_emit = {"multi_scale_block": _modulelist_drop_entry()}
    v = detect_stale_decomposed_test(
        component="multi_scale_block",
        no_emit_tests=no_emit,
    )
    assert v.is_stale is False
    assert v.action == ""
    assert "not decomposition-related" in v.reason or "real" in v.reason


# ---------------------------------------------------------------------------
# Negative: component not in no_emit_tests
# ---------------------------------------------------------------------------


def test_non_no_emit_component_not_stale() -> None:
    """A normal failed component (not a decomposed parent) must not be
    classified stale — its failure is real and the loop should iterate."""
    no_emit = {"video_memory_encoder": _decomp_entry()}
    v = detect_stale_decomposed_test(
        component="vision_neck",  # not in no_emit
        no_emit_tests=no_emit,
    )
    assert v.is_stale is False
    assert "not a decomposed parent" in v.reason


# ---------------------------------------------------------------------------
# Empty input / edge cases
# ---------------------------------------------------------------------------


def test_empty_component_name_handled() -> None:
    v = detect_stale_decomposed_test(component="", no_emit_tests={"x": _decomp_entry()})
    assert v.is_stale is False


def test_empty_no_emit_dict_handled() -> None:
    v = detect_stale_decomposed_test(component="something", no_emit_tests={})
    assert v.is_stale is False


def test_none_no_emit_dict_handled() -> None:
    """Non-dict input must not crash."""
    v = detect_stale_decomposed_test(component="something", no_emit_tests=None)
    assert v.is_stale is False


# ---------------------------------------------------------------------------
# archive_stale_test behavior (isolated)
# ---------------------------------------------------------------------------


def test_archive_stale_test_moves_file(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    tests_dir = demo_dir / "tests" / "pcc"
    tests_dir.mkdir(parents=True)
    test_file = tests_dir / "test_foo_parent.py"
    test_file.write_text("# stale\n")

    archived = archive_stale_test(demo_dir=demo_dir, component="foo_parent", safe_id="foo_parent")

    assert archived is not None
    assert archived.name == "test_foo_parent.py.stale_after_decomposition"
    assert archived.is_file()
    assert not test_file.exists()


def test_archive_stale_test_idempotent_when_no_file(tmp_path: Path) -> None:
    """Returns None silently when there's nothing to archive."""
    demo_dir = tmp_path / "demo"
    (demo_dir / "tests" / "pcc").mkdir(parents=True)
    archived = archive_stale_test(demo_dir=demo_dir, component="nothing_here", safe_id="nothing_here")
    assert archived is None


def test_restore_stale_test_round_trips(tmp_path: Path) -> None:
    """restore_stale_test is the inverse of archive_stale_test — it brings
    the archived parent test back to its live path for recompose."""
    from scripts.tt_hw_planner.agentic.stale_tests import restore_stale_test

    demo_dir = tmp_path / "demo"
    tests_dir = demo_dir / "tests" / "pcc"
    tests_dir.mkdir(parents=True)
    test_file = tests_dir / "test_foo_parent.py"
    test_file.write_text("# parent whole-module test\n")

    archived = archive_stale_test(demo_dir=demo_dir, component="foo_parent", safe_id="foo_parent")
    assert archived is not None and not test_file.exists()

    restored = restore_stale_test(demo_dir=demo_dir, component="foo_parent", safe_id="foo_parent")
    assert restored is not None
    assert restored == test_file
    assert test_file.is_file()
    assert not archived.exists()


def test_restore_stale_test_none_when_nothing_archived(tmp_path: Path) -> None:
    from scripts.tt_hw_planner.agentic.stale_tests import restore_stale_test

    demo_dir = tmp_path / "demo"
    (demo_dir / "tests" / "pcc").mkdir(parents=True)
    assert restore_stale_test(demo_dir=demo_dir, component="nope", safe_id="nope") is None


def test_restore_stale_test_does_not_clobber_live(tmp_path: Path) -> None:
    """If a live test already exists, restore is a no-op (returns None)
    rather than overwriting it."""
    from scripts.tt_hw_planner.agentic.stale_tests import restore_stale_test

    demo_dir = tmp_path / "demo"
    tests_dir = demo_dir / "tests" / "pcc"
    tests_dir.mkdir(parents=True)
    live = tests_dir / "test_foo_parent.py"
    live.write_text("# live\n")
    (tests_dir / "test_foo_parent.py.stale_after_decomposition").write_text("# stale\n")

    assert restore_stale_test(demo_dir=demo_dir, component="foo_parent", safe_id="foo_parent") is None
    assert live.read_text() == "# live\n"


# ---------------------------------------------------------------------------
# Structural: auto_iterate wires the brain primitive into final-pytest path
# ---------------------------------------------------------------------------


def test_auto_iterate_consults_stale_brain_after_final_pytest() -> None:
    """Pin: at final pytest, the loop must consult the brain for each
    failed component and archive phantoms. Without this wiring, the
    brain primitive exists but isn't used."""
    src = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "_cli_helpers" / "auto_iterate.py").read_text()
    assert "from ..agentic.stale_tests import" in src, "auto_iterate must import the stale-test brain primitive"
    assert "detect_stale_decomposed_test" in src
    assert "archive_stale_test" in src
    assert "PHANTOM-CLEANUP (brain G8)" in src, "auto_iterate must emit a visible banner when archiving phantoms"


def test_phantom_handler_called_from_both_exit_paths() -> None:
    """Pin: the phantom-failure handler must be invoked from BOTH the
    early-exit path (every component already at cap → CPU fallback →
    final pytest → check phantoms BEFORE concluding) AND the fall-
    through path (loop ended normally → final pytest → check phantoms).

    Without this, runs that exit early bypass the brain entirely —
    exactly the SAM2 2026-05-30 regression where phantoms surfaced
    in the early-exit final pytest but the brain's stale-test
    detector at the bottom of the function was unreachable."""
    src = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "_cli_helpers" / "auto_iterate.py").read_text()
    # The factored helper must exist
    assert "_brain_handle_phantom_failures(" in src, (
        "phantom handling must be in a factored helper so it can be "
        "called from multiple exit paths without duplication"
    )
    # And must be CALLED at least twice (early-exit + fall-through)
    call_count = src.count("_brain_handle_phantom_failures(")
    # 1 def + at least 2 call sites = at least 3 occurrences
    assert call_count >= 3, (
        f"phantom handler must be called from BOTH early-exit and "
        f"fall-through paths (found only {call_count - 1} call sites)"
    )


def test_decomposition_consumer_safety_net_still_present() -> None:
    """Pin: the mechanical safety net in decomposition_consumer remains.
    Brain detection covers stale tests from OLD decompositions; the
    mechanical archiver in the consumer covers fresh decompositions
    at write time."""
    src = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "decomposition_consumer.py").read_text()
    assert "stale_after_decomposition" in src, "decomposition_consumer must archive parent's stale test at write time"
    assert "tests" in src and "pcc" in src, "decomposition_consumer must locate the parent's test file"


def test_sam2_video_memory_encoder_would_be_detected() -> None:
    """Reproduce the exact SAM2 case that motivated this primitive:
    video_memory_encoder was decomposed; its test file failed; brain
    must classify it stale."""
    no_emit_for_sam2 = {
        "video_memory_encoder": {
            "captured_ts": 1780171249.503409,
            "reason": "decomposition consumer split parent into children at 2026-05-30",
        }
    }
    v = detect_stale_decomposed_test(
        component="video_memory_encoder",
        no_emit_tests=no_emit_for_sam2,
    )
    assert v.is_stale is True
    assert v.action == "archive_test"

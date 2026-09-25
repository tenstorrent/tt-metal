"""Tests for the session-local-artifact exclusion in
``_capture_worktree_deltas_as_overlay``.

Background — Phi-3.5 attention bring-up case (2026-06-03):

  The auto-iterate loop uses several session-scoped working-state
  files inside a single ``up`` run:

    * ``.py.best_native``      — highest-PCC in-session snapshot, used
                                 for rollback if a later iter regresses.
    * ``.py.preiter_native``   — pre-iter floor for restore-on-cap-out.
    * ``.py.last_good_native`` — most recent graduation snapshot.
    * ``.py.auto_stabilize.bak``, ``.py.bak`` — local backups.

  The overlay capture pass was including these via
  ``git status --porcelain``. Reapplied next ``up`` run, a FAILED iter's
  ``.best_native`` became the next run's rollback floor — and the
  regression-detection logic kept restoring it. Phi-3.5 attention's
  iter_005 ``use_hf_rope=True`` wrapper TT_FATAL'd, was captured as
  ``.best_native``, and poisoned every subsequent ``up`` run until the
  overlay was manually deleted.

These tests pin the exclusion rule so future refactors can't silently
re-introduce session-local leakage.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _cli():
    return importlib.import_module("scripts.tt_hw_planner.cli")


# ─── pure helper: _is_session_local_artifact ────────────────────────


def test_is_session_local_artifact_matches_best_native():
    cli = _cli()
    assert cli._is_session_local_artifact("models/x/_stubs/attention.py.best_native") is True


def test_is_session_local_artifact_matches_preiter_native():
    cli = _cli()
    assert cli._is_session_local_artifact("models/x/_stubs/attention.py.preiter_native") is True


def test_is_session_local_artifact_matches_last_good_native():
    cli = _cli()
    assert cli._is_session_local_artifact("models/x/_stubs/mlp.py.last_good_native") is True


def test_is_session_local_artifact_matches_auto_stabilize_bak():
    cli = _cli()
    assert cli._is_session_local_artifact("models/x/_stubs/attention.py.auto_stabilize.bak") is True


def test_is_session_local_artifact_matches_bak():
    cli = _cli()
    assert cli._is_session_local_artifact("models/x/_stubs/attention.py.bak") is True


def test_is_session_local_artifact_does_not_match_normal_py():
    cli = _cli()
    assert cli._is_session_local_artifact("models/x/_stubs/attention.py") is False


def test_is_session_local_artifact_does_not_match_test_file():
    cli = _cli()
    assert cli._is_session_local_artifact("models/x/tests/pcc/test_attention.py") is False


def test_is_session_local_artifact_does_not_match_arbitrary_artifacts():
    """Don't accidentally exclude files we DO want to capture.
    e.g. _captured/manifest.json, kernel_findings.json, bringup_status.json,
    BRING_UP_PLAN.md — all real overlay content."""
    cli = _cli()
    for path in (
        "models/x/_captured/attention/manifest.json",
        "models/x/kernel_findings.json",
        "models/x/bringup_status.json",
        "models/x/BRING_UP_PLAN.md",
        "models/x/_attempts/attention/iter_005.json",
    ):
        assert cli._is_session_local_artifact(path) is False, f"{path!r} must NOT be flagged session-local"


# ─── integration: _capture_worktree_deltas_as_overlay end-to-end ────


def _setup_git_worktree(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(root), check=True)
    (root / "README.md").write_text("test")
    subprocess.run(["git", "add", "README.md"], cwd=str(root), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=str(root), check=True)


def _make_demo(root: Path, model_id: str) -> Path:
    demo = root / "models" / "demos" / "test_model"
    (demo / "_stubs").mkdir(parents=True)
    (demo / "tests" / "pcc").mkdir(parents=True)
    import json

    (demo / "bringup_status.json").write_text(
        json.dumps(
            {
                "new_model_id": model_id,
                "components": [],
            }
        )
    )
    return demo


def test_capture_skips_best_native_files(tmp_path, monkeypatch):
    """Regression for the Phi-3.5 overlay-poisoning bug. A
    ``.py.best_native`` left in the worktree at capture time must
    NOT make it into the overlay."""
    cli = _cli()
    om = importlib.import_module("scripts.tt_hw_planner.overlay_manager")
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    _setup_git_worktree(tmp_path)
    demo = _make_demo(tmp_path, "test/model-best-native")

    # Two files in the worktree: a normal stub (should capture) and
    # a .best_native (should be skipped).
    (demo / "_stubs" / "attention.py").write_text("class TtAttention: pass\n")
    (demo / "_stubs" / "attention.py.best_native").write_text(
        "# poisoned snapshot from a failed iter — must NOT be captured\n"
        "class TtAttention:\n    use_hf_rope = True  # the bad choice\n"
    )

    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _m, repo_root=None: demo,
    )

    captured, ok = cli._capture_worktree_deltas_as_overlay(tmp_path, "test/model-best-native")
    assert ok is True

    # Inspect the overlay store to confirm what got captured.
    from scripts.tt_hw_planner.overlay_manager import list_overlays

    patches = list_overlays("test/model-best-native")
    rel_paths = {p.get("rel_path") for p in patches if isinstance(p, dict)}

    assert any(
        p.endswith("_stubs/attention.py") for p in rel_paths if p
    ), f"normal attention.py must be captured; got {rel_paths}"
    assert not any(
        p.endswith(".py.best_native") for p in rel_paths if p
    ), f".py.best_native must NOT be captured; got {rel_paths}"


def test_capture_skips_all_session_local_suffixes(tmp_path, monkeypatch):
    """Same regression, broader scope — all 5 suffixes from
    _SESSION_LOCAL_OVERLAY_SUFFIXES must be skipped."""
    cli = _cli()
    om = importlib.import_module("scripts.tt_hw_planner.overlay_manager")
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    _setup_git_worktree(tmp_path)
    demo = _make_demo(tmp_path, "test/model-all-suffixes")
    stubs = demo / "_stubs"
    for suffix in (
        ".py",
        ".py.best_native",
        ".py.preiter_native",
        ".py.last_good_native",
        ".py.auto_stabilize.bak",
        ".py.bak",
    ):
        (stubs / f"attention{suffix}").write_text(f"# {suffix}\n")

    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _m, repo_root=None: demo,
    )

    cli._capture_worktree_deltas_as_overlay(tmp_path, "test/model-all-suffixes")

    from scripts.tt_hw_planner.overlay_manager import list_overlays

    patches = list_overlays("test/model-all-suffixes")
    rel_paths = {p.get("rel_path") for p in patches if isinstance(p, dict)}

    # The plain .py must be captured.
    assert any(
        p.endswith("attention.py") and not p.endswith(".py.best_native") and not p.endswith(".py.bak")
        for p in rel_paths
        if p
    ), f"plain attention.py must be captured; got {rel_paths}"

    # NONE of the session-local suffixes may be captured.
    for forbidden in (".best_native", ".preiter_native", ".last_good_native", ".auto_stabilize.bak", ".bak"):
        assert not any(
            p.endswith(forbidden) for p in rel_paths if p
        ), f"file ending in {forbidden!r} leaked into overlay; got {rel_paths}"

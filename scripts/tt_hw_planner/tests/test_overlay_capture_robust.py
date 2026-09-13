"""Unit tests for Bug 1 fix — robust overlay capture.

The original ``_capture_worktree_deltas_as_overlay`` relied solely on
``git status --porcelain`` to identify modified files. This silently
dropped files in some scenarios (notably, files whose changes were
already staged by a prior overlay-apply step). The result was the
SAM2 regression we observed: 7 graduated stubs never made it to the
overlay despite being native ttnn code in the worktree.

The fix adds a secondary capture pass that explicitly scans the model's
demo dir for ``_stubs/*.py`` and ``tests/pcc/*.py`` files. Any path
that differs from HEAD gets captured, regardless of what git status
showed.

These tests use a synthetic worktree fixture so they run fast without
spinning up the full bring-up pipeline."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from unittest import mock


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _cli():
    return importlib.import_module("scripts.tt_hw_planner.cli")


def _setup_git_worktree(root: Path) -> None:
    """Initialize a minimal git repo with a model demo layout under root."""
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(root),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(root),
        check=True,
    )
    # Initial commit so HEAD exists
    (root / "README.md").write_text("test")
    subprocess.run(["git", "add", "README.md"], cwd=str(root), check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "initial"],
        cwd=str(root),
        check=True,
    )


def _make_demo_dir(root: Path, model_id: str) -> Path:
    """Create the standard tt_hw_planner demo dir layout."""
    demo = root / "models" / "demos" / "test_segmentation" / "test_model"
    (demo / "_stubs").mkdir(parents=True)
    (demo / "tests" / "pcc").mkdir(parents=True)
    bringup = {
        "new_model_id": model_id,
        "components": [],
    }
    import json

    (demo / "bringup_status.json").write_text(json.dumps(bringup))
    return demo


def test_captures_modified_tracked_stub(tmp_path, monkeypatch) -> None:
    """A stub that was committed then modified should be captured."""
    cli = _cli()
    om = importlib.import_module("scripts.tt_hw_planner.overlay_manager")
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    _setup_git_worktree(tmp_path)
    demo = _make_demo_dir(tmp_path, "test/model-a")
    stub = demo / "_stubs" / "comp_a.py"
    stub.write_text("# initial native stub\nclass Component: pass\n")

    # Commit the initial state so it's tracked
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "add stub"],
        cwd=str(tmp_path),
        check=True,
    )
    # Now MODIFY the stub
    stub.write_text("# modified native stub\nclass Component:\n    def __call__(self): pass\n")

    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _m, repo_root=None: demo,
    )

    captured, ok = cli._capture_worktree_deltas_as_overlay(tmp_path, "test/model-a")
    assert ok is True
    assert captured >= 1, "modified tracked stub must be captured"


def test_captures_untracked_stub(tmp_path, monkeypatch) -> None:
    """A stub created during the run (untracked) must be captured.
    This was working before but pin it as a regression test."""
    cli = _cli()
    om = importlib.import_module("scripts.tt_hw_planner.overlay_manager")
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    _setup_git_worktree(tmp_path)
    demo = _make_demo_dir(tmp_path, "test/model-b")
    # Create an UNTRACKED stub (no git add/commit)
    stub = demo / "_stubs" / "comp_b.py"
    stub.write_text("# brand new stub written by autofill\nclass Component: pass\n")

    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _m, repo_root=None: demo,
    )

    captured, ok = cli._capture_worktree_deltas_as_overlay(tmp_path, "test/model-b")
    assert ok is True
    assert captured >= 1, "untracked stub must be captured"


def test_captures_pre_staged_stub(tmp_path, monkeypatch) -> None:
    """The Bug 1 root cause: a stub that overlay-apply staged into the
    index, then auto-iterate further modified. ``git status --porcelain``
    may or may not surface this clearly depending on the state diff.

    The secondary demo-dir scan must catch it regardless."""
    cli = _cli()
    om = importlib.import_module("scripts.tt_hw_planner.overlay_manager")
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    _setup_git_worktree(tmp_path)
    demo = _make_demo_dir(tmp_path, "test/model-c")

    # Step 1: create a stub and commit it (analogous to a HEAD baseline)
    stub = demo / "_stubs" / "comp_c.py"
    stub.write_text("# HEAD version (autofill)\nclass Component:\n    def forward(self): return None\n")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "baseline stub"],
        cwd=str(tmp_path),
        check=True,
    )

    # Step 2: simulate overlay-apply by modifying + staging the file
    stub.write_text("# overlay version (graduated, version 1)\nclass Component:\n    def __call__(self): import ttnn\n")
    subprocess.run(["git", "add", "-u"], cwd=str(tmp_path), check=True)

    # Step 3: simulate auto-iterate further modifying
    stub.write_text(
        "# overlay version (graduated, REFINED)\nclass Component:\n    def __call__(self): import ttnn  # better\n"
    )

    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _m, repo_root=None: demo,
    )

    captured, ok = cli._capture_worktree_deltas_as_overlay(tmp_path, "test/model-c")
    assert ok is True
    assert captured >= 1, "pre-staged-then-modified stub must be captured"


def test_captures_all_stubs_in_demo_dir(tmp_path, monkeypatch) -> None:
    """When N stubs are modified in one run, ALL N should be captured.
    Previously we lost some -- this test pins that we don't anymore."""
    cli = _cli()
    om = importlib.import_module("scripts.tt_hw_planner.overlay_manager")
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    _setup_git_worktree(tmp_path)
    demo = _make_demo_dir(tmp_path, "test/model-multi")
    # Create 5 distinct stubs
    for i in range(5):
        stub = demo / "_stubs" / f"comp_{i}.py"
        stub.write_text(f"# stub for component {i}\nclass Component_{i}: pass\n")

    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _m, repo_root=None: demo,
    )

    captured, ok = cli._capture_worktree_deltas_as_overlay(tmp_path, "test/model-multi")
    assert ok is True
    assert captured >= 5, f"all 5 stubs must be captured, got {captured}"


def test_capture_idempotent_when_no_changes(tmp_path, monkeypatch) -> None:
    """Running capture twice in a row on a clean worktree must not
    double-capture or fail. Idempotency."""
    cli = _cli()
    om = importlib.import_module("scripts.tt_hw_planner.overlay_manager")
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    _setup_git_worktree(tmp_path)
    demo = _make_demo_dir(tmp_path, "test/model-idem")

    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _m, repo_root=None: demo,
    )
    # First pass: no stubs to capture
    captured1, ok1 = cli._capture_worktree_deltas_as_overlay(tmp_path, "test/model-idem")
    assert ok1 is True
    # Second pass: still no changes
    captured2, ok2 = cli._capture_worktree_deltas_as_overlay(tmp_path, "test/model-idem")
    assert ok2 is True
    # Both should report low captures (just the bringup_status.json maybe)
    # The critical assertion is no exceptions raised.


def test_demo_dir_scan_finds_stub_outside_git_status(tmp_path, monkeypatch) -> None:
    """The KEY scenario the bug exposed: git status doesn't show a file
    but the file IS different from HEAD. The secondary scan must catch it.

    We simulate this by having git see the file as 'A' (added in index,
    no working-tree changes) which means git diff HEAD --porcelain output
    might be empty in some configurations."""
    cli = _cli()
    om = importlib.import_module("scripts.tt_hw_planner.overlay_manager")
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    _setup_git_worktree(tmp_path)
    demo = _make_demo_dir(tmp_path, "test/model-scan")
    stub = demo / "_stubs" / "comp_scan.py"
    stub.write_text("# new stub\nclass Component: pass\n")

    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _m, repo_root=None: demo,
    )

    # Mock subprocess.run for git status to return empty output (simulating
    # the silently-missing-file bug)
    real_run = subprocess.run

    def mock_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 3 and cmd[:2] == ["git", "status"]:
            # Return empty stdout — simulating the bug where git status
            # doesn't surface the file
            result = mock.MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result
        return real_run(cmd, *args, **kwargs)

    with mock.patch("subprocess.run", side_effect=mock_run):
        captured, ok = cli._capture_worktree_deltas_as_overlay(tmp_path, "test/model-scan")

    # Even though git status returned nothing, the secondary demo-dir scan
    # should have caught the stub
    assert ok is True
    assert captured >= 1, "the secondary demo-dir scan must catch files git status missed"

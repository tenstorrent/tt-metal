"""Unit tests for Layer 6.1 — capture-artifact verification.

The earlier Phase 2 validation reported "capture succeeded" for 5
SAM2 video_* components, but the _captured/<comp>/ directories were
never actually created for them. Root cause: _retry_capture returned
True when cmd_capture_inputs rc=0, but rc=0 doesn't mean the target
component was captured -- it just means the command didn't crash.

The auto-onboard driver can run successfully and yet never invoke
the target component's forward (because that component is in a
code path the driver doesn't exercise). The fix verifies that
args.pt / kwargs.pt / output.pt exist for the specific component
AFTER the capture call returns.

These tests pin the verification semantics so future changes don't
regress to "rc=0 is enough"."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from unittest import mock


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ts():
    return importlib.import_module("scripts.tt_hw_planner.commands.tackle_skipped")


def test_verify_returns_false_when_directory_missing(tmp_path) -> None:
    ts = _ts()
    ok, msg = ts._verify_capture_artifacts(tmp_path / "demo", "video_layer_norm")
    assert ok is False
    assert "no _captured" in msg


def test_verify_returns_false_when_only_some_files_present(tmp_path) -> None:
    """Capture is atomic from the test scaffold's perspective -- it
    needs all three of args/kwargs/output to load. Missing any one
    means the test will fall back to synthesis (which is what we
    were trying to avoid)."""
    ts = _ts()
    demo = tmp_path / "demo"
    comp_dir = demo / "_captured" / "video_layer_norm"
    comp_dir.mkdir(parents=True)
    # Only args.pt, missing kwargs.pt and output.pt
    (comp_dir / "args.pt").write_text("fake")
    ok, msg = ts._verify_capture_artifacts(demo, "video_layer_norm")
    assert ok is False
    assert "missing" in msg
    assert "kwargs.pt" in msg
    assert "output.pt" in msg


def test_verify_returns_true_when_all_three_files_present(tmp_path) -> None:
    ts = _ts()
    demo = tmp_path / "demo"
    comp_dir = demo / "_captured" / "video_layer_norm"
    comp_dir.mkdir(parents=True)
    for f in ("args.pt", "kwargs.pt", "output.pt"):
        (comp_dir / f).write_text("fake")
    ok, msg = ts._verify_capture_artifacts(demo, "video_layer_norm")
    assert ok is True
    assert "has all artifacts" in msg


def test_verify_normalizes_component_name(tmp_path) -> None:
    """Capture saves under _safe_id(name) — verification must use the
    same normalization or it'd false-negative on names with dots/dashes."""
    ts = _ts()
    demo = tmp_path / "demo"
    # Component named "video.layer-norm" gets stored as "video_layer_norm"
    comp_dir = demo / "_captured" / "video_layer_norm"
    comp_dir.mkdir(parents=True)
    for f in ("args.pt", "kwargs.pt", "output.pt"):
        (comp_dir / f).write_text("fake")
    # Verification with the un-normalized name should still find the dir
    ok, _msg = ts._verify_capture_artifacts(demo, "video.layer-norm")
    assert ok is True


def test_retry_capture_dry_run_skips_invocation(tmp_path) -> None:
    """In dry-run mode, no actual capture happens — message must
    indicate that. Returns False because no real action was taken."""
    ts = _ts()
    ok, msg = ts._retry_capture("facebook/test", "comp_a", dry_run=True, demo_dir=tmp_path)
    assert ok is False
    assert "would invoke" in msg


def test_retry_capture_rc_zero_with_artifacts_is_success(tmp_path) -> None:
    """The happy path: cmd_capture_inputs returns 0 AND the artifacts
    are present -> success."""
    ts = _ts()
    demo = tmp_path / "demo"
    comp_dir = demo / "_captured" / "video_layer_norm"
    comp_dir.mkdir(parents=True)
    for f in ("args.pt", "kwargs.pt", "output.pt"):
        (comp_dir / f).write_text("fake")

    with mock.patch("scripts.tt_hw_planner.cli.cmd_capture_inputs", return_value=0):
        ok, msg = ts._retry_capture("facebook/test", "video_layer_norm", dry_run=False, demo_dir=demo)
    assert ok is True
    assert "rc=0" in msg
    assert "has all artifacts" in msg


def test_retry_capture_rc_zero_without_artifacts_is_failure(tmp_path) -> None:
    """The false-positive bug we're fixing: cmd_capture_inputs returns 0
    but no artifacts for the target component were produced -> must NOT
    claim success. This is the case where the auto-onboard driver ran
    but never invoked the target component's forward."""
    ts = _ts()
    demo = tmp_path / "demo"
    demo.mkdir(parents=True)
    # No _captured directory at all

    with mock.patch("scripts.tt_hw_planner.cli.cmd_capture_inputs", return_value=0):
        ok, msg = ts._retry_capture("facebook/test", "video_layer_norm", dry_run=False, demo_dir=demo)
    assert ok is False, "rc=0 alone is not sufficient -- artifacts must exist"
    assert "false-positive" in msg


def test_retry_capture_rc_nonzero_is_failure(tmp_path) -> None:
    """If cmd_capture_inputs returns non-zero, we don't even check
    artifacts -- the call itself failed."""
    ts = _ts()
    with mock.patch("scripts.tt_hw_planner.cli.cmd_capture_inputs", return_value=2):
        ok, msg = ts._retry_capture("facebook/test", "video_layer_norm", dry_run=False, demo_dir=tmp_path)
    assert ok is False
    assert "rc=2" in msg


def test_retry_capture_raises_returns_failure(tmp_path) -> None:
    """If cmd_capture_inputs raises (e.g., HF download failure), we
    catch and return a clear failure message rather than propagating."""
    ts = _ts()
    with mock.patch(
        "scripts.tt_hw_planner.cli.cmd_capture_inputs",
        side_effect=RuntimeError("simulated network failure"),
    ):
        ok, msg = ts._retry_capture("facebook/test", "video_layer_norm", dry_run=False, demo_dir=tmp_path)
    assert ok is False
    assert "raised" in msg
    assert "RuntimeError" in msg


def test_retry_capture_resolves_demo_dir_when_not_passed(tmp_path, monkeypatch) -> None:
    """When the caller doesn't pass demo_dir explicitly, _retry_capture
    must resolve it via find_demo_dir. Used by cmd_tackle_skipped which
    operates without an explicit worktree handle."""
    ts = _ts()
    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _m: tmp_path / "auto_resolved_demo",
    )
    # Set up captured artifacts under the auto-resolved path
    comp_dir = tmp_path / "auto_resolved_demo" / "_captured" / "video_layer_norm"
    comp_dir.mkdir(parents=True)
    for f in ("args.pt", "kwargs.pt", "output.pt"):
        (comp_dir / f).write_text("fake")
    with mock.patch("scripts.tt_hw_planner.cli.cmd_capture_inputs", return_value=0):
        ok, _msg = ts._retry_capture("facebook/test", "video_layer_norm", dry_run=False)
    assert ok is True


def test_retry_capture_returns_failure_when_demo_dir_unresolvable(monkeypatch, tmp_path) -> None:
    """If find_demo_dir returns None (no scaffold yet), _retry_capture
    must fail clearly rather than trying to verify against a missing
    directory."""
    ts = _ts()
    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _m: None,
    )
    ok, msg = ts._retry_capture("facebook/test", "video_layer_norm", dry_run=False)
    assert ok is False
    assert "demo dir not resolvable" in msg


def test_env_var_restored_after_call(tmp_path) -> None:
    """The TT_PLANNER_AUTO_ONBOARD_DRIVER env var must be restored to
    its prior value after the capture call, even if the call fails.
    Otherwise concurrent / sequential operations on different models
    could leak the env into unrelated runs."""
    ts = _ts()
    import os

    prior = os.environ.get("TT_PLANNER_AUTO_ONBOARD_DRIVER")
    if prior is not None:
        del os.environ["TT_PLANNER_AUTO_ONBOARD_DRIVER"]
    try:
        with mock.patch("scripts.tt_hw_planner.cli.cmd_capture_inputs", return_value=1):
            ts._retry_capture("facebook/test", "x", dry_run=False, demo_dir=tmp_path)
        assert "TT_PLANNER_AUTO_ONBOARD_DRIVER" not in os.environ
    finally:
        if prior is not None:
            os.environ["TT_PLANNER_AUTO_ONBOARD_DRIVER"] = prior

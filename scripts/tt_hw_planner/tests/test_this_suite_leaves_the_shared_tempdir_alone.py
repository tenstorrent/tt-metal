# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""This suite must not write into the system temp directory a live run reads.

Tests that drive the real gate make it call `tempfile.mkdtemp(prefix="e2e_gate_")`, which put
fixture text -- "RuntimeError: boom", "NOC0 is hung on PCIe device ID 9", "[e2e] denoise step 37/50
done" -- into the same temp namespace a LIVE bring-up uses. A running agent inspects those
directories to diagnose its own gate, read this suite's fixtures twice on 2026-09-28, and reported
its gate SIGKILLed mid-run with hours of device time lost that had not been lost.

conftest redirects `tempfile.tempdir` per test. These tests pin that, because the failure mode is
silent: the suite passes either way and the damage lands in another process's diagnosis.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def test_tempfile_is_redirected_away_from_the_system_temp(_tempdir_is_not_shared):
    """mkdtemp must land under the per-test directory, not /tmp."""
    made = Path(tempfile.mkdtemp(prefix="e2e_gate_"))
    try:
        assert made.is_relative_to(_tempdir_is_not_shared), f"{made} escaped the per-test tempdir"
    finally:
        made.rmdir()


def test_gettempdir_agrees_with_it(_tempdir_is_not_shared):
    """A test that globs the temp dir for its own leaks must see the redirected one, or it would be
    inspecting the shared directory instead."""
    assert Path(tempfile.gettempdir()) == _tempdir_is_not_shared


def test_the_env_var_agrees_too(_tempdir_is_not_shared, monkeypatch):
    """Tool code spawns subprocesses; they read TMPDIR, not this process's tempfile global."""
    import os

    assert Path(os.environ["TMPDIR"]) == _tempdir_is_not_shared


def test_the_real_gate_writes_inside_it(_tempdir_is_not_shared, tmp_path, monkeypatch):
    """End to end: the production path that caused the litter now lands in the private dir.

    Drives `_run_deterministic_gates` exactly as the other gate tests do, and asserts the directory
    the gate created is under the per-test tempdir."""
    from models.experimental.perf_automation.agent import probes as _PR
    from scripts.tt_hw_planner.commands import emit_e2e as E

    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tests" / "e2e").mkdir(parents=True)
    (demo / "tests" / "e2e" / "test_e2e_m.py").write_text("def test_e2e():\n    pass\n")
    seen = {}

    def _exec(cmd, cwd, env, timeout_s, log_path, **k):
        seen["path"] = Path(log_path)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("1 passed")
        return 0

    monkeypatch.setenv("E2E_REQUIRE_ON_DEVICE", "0")
    monkeypatch.delenv(E.E2E_GATE_LOG_ENV, raising=False)
    monkeypatch.setattr(_PR, "_execute", _exec)
    E._run_deterministic_gates(demo, 0.99, 60)
    assert seen["path"].is_relative_to(_tempdir_is_not_shared), f"{seen['path']} escaped to the shared tempdir"

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F238: a detached device job outlives the call that started it, reports its rc, and is never killed by a wait."""

import subprocess
import sys
import time

from models.demos.common.bringup import devjob


def test_a_job_outlives_its_starter_and_reports_its_rc(tmp_path, monkeypatch):
    monkeypatch.setenv("BRINGUP_DEVJOB_DIR", str(tmp_path))
    # started from a child process that exits at once, as an agent's tool call does at its time limit
    starter = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.common.bringup.devjob",
            "start",
            "j1",
            "--",
            "bash",
            "-c",
            "sleep 1; exit 3",
        ],
        env={**__import__("os").environ, "BRINGUP_DEVJOB_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert starter.returncode == 0, starter.stderr
    assert devjob.wait("j1", timeout=20, poll=0.1) == 3


def test_a_wait_that_times_out_leaves_the_job_running(tmp_path, monkeypatch):
    monkeypatch.setenv("BRINGUP_DEVJOB_DIR", str(tmp_path))
    devjob.start("j2", ["bash", "-c", "sleep 2; echo done"])
    assert devjob.wait("j2", timeout=0.2, poll=0.05) == devjob.RUNNING
    assert devjob.wait("j2", timeout=20, poll=0.1) == 0
    assert "done" in (tmp_path / "j2" / "log").read_text()


def test_a_running_job_cannot_be_started_twice(tmp_path, monkeypatch):
    monkeypatch.setenv("BRINGUP_DEVJOB_DIR", str(tmp_path))
    devjob.start("j3", ["bash", "-c", "sleep 2"])
    try:
        devjob.start("j3", ["true"])
    except SystemExit as e:
        assert "already running" in str(e)
    else:
        raise AssertionError("a second start of a running job was accepted")
    time.sleep(0)
    assert devjob.wait("j3", timeout=20, poll=0.1) == 0


def test_the_command_check_looks_inside_a_devjob():
    """F238: `devjob start NAME -- CMD` is judged by CMD: the safe runner passes, a direct pytest does not."""
    from pathlib import Path

    from models.demos.common.bringup.orchestrator import command_violations

    repo = Path(__file__).resolve().parents[5]
    ok = "python -m models.demos.common.bringup.devjob start g -- scripts/run_safe_pytest.sh --run-all tests/x.py"
    bad = "python -m models.demos.common.bringup.devjob start g -- pytest tests/x.py"
    assert command_violations([ok], repo) == []
    assert any("direct pytest" in v for v in command_violations([bad], repo))


def test_a_plain_gate_commits_a_pass():
    """F239: `gate <task>` with no flag commits a PASS (FIX.19's stayed uncommitted); --no-commit opts out."""
    import argparse

    from models.demos.common.bringup.__main__ import gate_commits

    assert gate_commits(argparse.Namespace(no_commit=False, commit=False, continue_=False)) is True
    assert gate_commits(argparse.Namespace(no_commit=True, commit=False, continue_=False)) is False

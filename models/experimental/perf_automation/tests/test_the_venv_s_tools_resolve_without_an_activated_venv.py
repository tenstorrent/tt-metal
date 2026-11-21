"""The agent's console-script dependency (tt-perf-report) is found beside the running interpreter, not
only on PATH. A launcher that never activated the venv -- the dashboard's optimize button, cron, a bare
`bash -c` -- used to make the tool's own preflight refuse to start (`FileNotFoundError:
'tt-perf-report'`, 19 tests) while the same command worked from a shell with the venv active.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest


def _fake_venv(tmp_path: Path, monkeypatch, with_tool: bool = True) -> Path:
    """A python beside (optionally) a tt-perf-report console script, as a venv lays them out."""
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    py = bin_dir / "python"
    py.write_text("#!/bin/sh\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    if with_tool:
        tool = bin_dir / "tt-perf-report"
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(sys, "executable", str(py))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))  # the venv the interpreter belongs to
    monkeypatch.setenv("PATH", "/nonexistent")  # the dashboard launch: nothing of the venv on PATH
    return bin_dir


def test_the_tool_is_found_beside_the_interpreter_with_a_bare_path(tmp_path, monkeypatch):
    from agent.pkgtools import venv_tool

    bin_dir = _fake_venv(tmp_path, monkeypatch)
    assert venv_tool("tt-perf-report") == str(bin_dir / "tt-perf-report")


def test_a_missing_tool_is_still_missing(tmp_path, monkeypatch):
    from agent.pkgtools import venv_tool

    _fake_venv(tmp_path, monkeypatch, with_tool=False)
    assert venv_tool("tt-perf-report") is None


def test_check_dependencies_passes_without_the_venv_on_path(tmp_path, monkeypatch):
    from agent.before_loop import check_dependencies

    _fake_venv(tmp_path, monkeypatch)
    assert check_dependencies() == []


def test_refine_runs_the_resolved_executable(tmp_path, monkeypatch):
    """The subprocess gets the absolute path, so it does not depend on PATH at exec time either."""
    import subprocess

    from agent import tracy_tool

    bin_dir = _fake_venv(tmp_path, monkeypatch)
    seen = {}

    def _run(cmd, **kw):
        seen["cmd"] = list(cmd)
        Path(cmd[3]).write_text("")  # the --csv output refine returns
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr("agent.probes.adaptive_op_timeout", lambda *a, **k: 60)
    raw = tmp_path / "raw.csv"
    raw.write_text("")
    tracy_tool.refine(raw, tmp_path / "report.csv")
    assert seen["cmd"][0] == str(bin_dir / "tt-perf-report")


def test_the_real_venv_resolves_its_own_tool():
    """On the box the suite runs on, the interpreter's own bin holds the script (the agent deps are
    installed into the venv); skip rather than fail where they are not."""
    from agent.pkgtools import venv_tool

    beside = Path(sys.executable).absolute().parent / "tt-perf-report"
    if not beside.exists():
        pytest.skip("agent deps not installed beside this interpreter")
    assert venv_tool("tt-perf-report") == str(beside)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""pkgtools is the pip-vs-uv switch every install/remediation path routes through.
A pip-based command on a uv-only venv (no `pip` module) is the exact clean-build break
this guards against, so both environment shapes are exercised here with `pip` presence faked."""
import sys

from agent import pkgtools


def test_pip_env_uses_python_m_pip(monkeypatch):
    monkeypatch.setattr(pkgtools, "have_pip", lambda: True)
    cmd = pkgtools.pip_cmd(["install", "-U", "claude-agent-sdk"])
    assert cmd == [sys.executable, "-m", "pip", "install", "-U", "claude-agent-sdk"]
    assert pkgtools.installer_hint() == "pip install"


def test_uv_only_env_uses_uv_pip_with_python_target(monkeypatch):
    monkeypatch.setattr(pkgtools, "have_pip", lambda: False)
    cmd = pkgtools.pip_cmd(["install", "-U", "claude-agent-sdk"])
    assert cmd == ["uv", "pip", "install", "--python", sys.executable, "-U", "claude-agent-sdk"]
    assert "pip" != cmd[0]
    assert pkgtools.installer_hint() == "uv pip install"


def test_uv_only_preserves_subcommand_and_arg_order(monkeypatch):
    monkeypatch.setattr(pkgtools, "have_pip", lambda: False)
    cmd = pkgtools.pip_cmd(["uninstall", "-y", "tt-lang", "--quiet"])
    assert cmd == ["uv", "pip", "uninstall", "--python", sys.executable, "-y", "tt-lang", "--quiet"]


def test_run_pip_invokes_built_command(monkeypatch):
    seen = {}

    def fake_run(argv, capture_output, text, timeout, check):
        seen["argv"] = argv
        seen["check"] = check
        seen["timeout"] = timeout

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(pkgtools, "have_pip", lambda: True)
    monkeypatch.setattr(pkgtools.subprocess, "run", fake_run)
    pkgtools.run_pip(["install", "tt-lang==1.0.1", "--no-deps"], timeout_s=42, check=True)
    assert seen["argv"][:3] == [sys.executable, "-m", "pip"]
    assert seen["check"] is True and seen["timeout"] == 42


def test_run_pip_routes_through_uv_when_no_pip(monkeypatch):
    seen = {}
    monkeypatch.setattr(pkgtools, "have_pip", lambda: False)
    monkeypatch.setattr(
        pkgtools.subprocess,
        "run",
        lambda argv, **kw: seen.setdefault("argv", argv) or type("R", (), {"returncode": 0})(),
    )
    pkgtools.run_pip(["install", "tt-lang==1.0.1", "--no-deps"])
    assert seen["argv"][0] == "uv" and seen["argv"][1] == "pip"
    assert "--python" in seen["argv"]

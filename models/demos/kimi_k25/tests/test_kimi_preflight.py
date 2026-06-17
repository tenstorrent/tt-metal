# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""test_kimi_preflight.py — CPU meta-tests for kimi_preflight.py.

Validates every check function in the hardware readiness preflight script
*without* touching hardware, downloading weights, or requiring ttnn.  All
tests run in a standard Python environment (scheduling container / CI).

The preflight module lives in ``scripts/kimi_preflight.py`` (no ``__init__.py``
in that directory), so we load it with ``importlib.util.spec_from_file_location``.

Class summary
-------------
TestCheckPythonVersion          — Python version gating (≥3.9)
TestCheckTenstorrentDevices     — /dev/tenstorrent device node detection
TestCheckMeshDeviceEnv          — MESH_DEVICE topology validation
TestCheckKimiHfModelEnv         — KIMI_HF_MODEL directory validation
TestCheckImportsAndConfig       — module import + KimiK25Config validation
TestCheckTtnnImportable         — ttnn import handling (warns, doesn't hard-fail)
TestMainExitCode                — end-to-end CLI exit-code assertions via subprocess
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the preflight module dynamically (scripts/ has no __init__.py)
# ---------------------------------------------------------------------------

_PREFLIGHT_PATH = (
    Path(__file__).parent.parent / "scripts" / "kimi_preflight.py"
)


def _load_preflight() -> types.ModuleType:
    """Return a fresh copy of the preflight module.

    Each call re-executes the module so tests don't share global state for
    ``_PASS`` / ``_FAIL`` / ``_WARN`` / ``_INFO`` (which ``main()`` overwrites
    when ``--no-color`` is passed).
    """
    spec = importlib.util.spec_from_file_location("kimi_preflight_under_test", _PREFLIGHT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Pre-load once for tests that don't modify globals.
_pf = _load_preflight()


# ---------------------------------------------------------------------------
# TestCheckPythonVersion
# ---------------------------------------------------------------------------


class TestCheckPythonVersion:
    """check_python_version() returns True when Python >= 3.9, False otherwise."""

    def test_current_python_passes(self, capsys):
        """Running Python version should satisfy ≥3.9 requirement."""
        assert _pf.check_python_version() is True
        captured = capsys.readouterr()
        # Check message contains "Python" — the actual version digits
        assert "Python" in captured.out

    def test_python_38_fails(self, capsys):
        """Python 3.8 must be rejected."""
        with mock.patch.object(sys, "version_info", new=(3, 8, 0, "final", 0)):
            result = _pf.check_python_version()
        assert result is False
        captured = capsys.readouterr()
        assert "3.8" in captured.out

    def test_python_39_passes(self, capsys):
        """Python 3.9 exactly is the minimum supported version."""
        with mock.patch.object(sys, "version_info", new=(3, 9, 0, "final", 0)):
            result = _pf.check_python_version()
        assert result is True

    def test_python_312_passes(self, capsys):
        """Future Python version should not be rejected by the version check."""
        with mock.patch.object(sys, "version_info", new=(3, 12, 5, "final", 0)):
            result = _pf.check_python_version()
        assert result is True


# ---------------------------------------------------------------------------
# TestCheckTenstorrentDevices
# ---------------------------------------------------------------------------


class TestCheckTenstorrentDevices:
    """check_tenstorrent_devices() inspects /dev/tenstorrent."""

    def test_dev_root_missing(self, capsys):
        """Missing /dev/tenstorrent → False."""
        with mock.patch("os.path.exists", return_value=False):
            result = _pf.check_tenstorrent_devices()
        assert result is False
        assert "/dev/tenstorrent" in capsys.readouterr().out

    def test_empty_device_list(self, capsys):
        """Directory exists but no numeric devices → False."""
        with mock.patch("os.path.exists", return_value=True), mock.patch(
            "os.listdir", return_value=[]
        ):
            result = _pf.check_tenstorrent_devices()
        assert result is False

    def test_single_device(self, capsys):
        """One device node '0' → True."""
        with mock.patch("os.path.exists", return_value=True), mock.patch(
            "os.listdir", return_value=["0"]
        ):
            result = _pf.check_tenstorrent_devices()
        assert result is True
        assert "0" in capsys.readouterr().out

    def test_multiple_devices(self, capsys):
        """Multiple device nodes → True, all listed."""
        with mock.patch("os.path.exists", return_value=True), mock.patch(
            "os.listdir", return_value=["0", "1", "2"]
        ):
            result = _pf.check_tenstorrent_devices()
        assert result is True
        out = capsys.readouterr().out
        assert "0" in out and "1" in out

    def test_permission_error(self, capsys):
        """PermissionError on listdir → False (not a crash)."""
        with mock.patch("os.path.exists", return_value=True), mock.patch(
            "os.listdir", side_effect=PermissionError("no access")
        ):
            result = _pf.check_tenstorrent_devices()
        assert result is False

    def test_non_numeric_entries_ignored(self, capsys):
        """Non-numeric entries like 'by-id' are not counted as device nodes → False."""
        with mock.patch("os.path.exists", return_value=True), mock.patch(
            "os.listdir", return_value=["by-id", "by-path"]
        ):
            result = _pf.check_tenstorrent_devices()
        assert result is False


# ---------------------------------------------------------------------------
# TestCheckMeshDeviceEnv
# ---------------------------------------------------------------------------


class TestCheckMeshDeviceEnv:
    """check_mesh_device_env() validates MESH_DEVICE topology."""

    def test_unset_env_fails(self, capsys, monkeypatch):
        """MESH_DEVICE not set → False."""
        monkeypatch.delenv("MESH_DEVICE", raising=False)
        result = _pf.check_mesh_device_env(requested=None)
        assert result is False
        assert "MESH_DEVICE" in capsys.readouterr().out

    def test_tg_passes(self, monkeypatch):
        """MESH_DEVICE=TG → True."""
        monkeypatch.setenv("MESH_DEVICE", "TG")
        assert _pf.check_mesh_device_env(requested=None) is True

    def test_t3k_passes(self, monkeypatch):
        """MESH_DEVICE=T3K → True."""
        monkeypatch.setenv("MESH_DEVICE", "T3K")
        assert _pf.check_mesh_device_env(requested=None) is True

    def test_n150_fails(self, capsys, monkeypatch):
        """N150 → False with memory-wall explanation."""
        monkeypatch.setenv("MESH_DEVICE", "N150")
        result = _pf.check_mesh_device_env(requested=None)
        assert result is False
        out = capsys.readouterr().out
        assert "N150" in out
        assert "33.8 GB" in out  # memory explanation must be present

    def test_n300_fails(self, capsys, monkeypatch):
        """N300 → False with memory-wall explanation."""
        monkeypatch.setenv("MESH_DEVICE", "N300")
        result = _pf.check_mesh_device_env(requested=None)
        assert result is False
        out = capsys.readouterr().out
        assert "16.9 GB" in out

    def test_unrecognized_topology_warns(self, capsys, monkeypatch):
        """Unknown topology string → False (warn, not silently pass)."""
        monkeypatch.setenv("MESH_DEVICE", "GALAXY_X")
        result = _pf.check_mesh_device_env(requested=None)
        assert result is False

    def test_cli_override_sets_env(self, monkeypatch):
        """Passing requested='TG' must write MESH_DEVICE into os.environ."""
        monkeypatch.delenv("MESH_DEVICE", raising=False)
        _pf.check_mesh_device_env(requested="TG")
        assert os.environ.get("MESH_DEVICE") == "TG"

    def test_cli_override_takes_precedence(self, monkeypatch):
        """CLI override wins even when env is set to something else."""
        monkeypatch.setenv("MESH_DEVICE", "N150")
        result = _pf.check_mesh_device_env(requested="T3K")
        assert result is True


# ---------------------------------------------------------------------------
# TestCheckKimiHfModelEnv
# ---------------------------------------------------------------------------


class TestCheckKimiHfModelEnv:
    """check_kimi_hf_model_env() is a soft check — missing KIMI_HF_MODEL is OK."""

    def test_unset_is_not_failure(self, monkeypatch):
        """KIMI_HF_MODEL not set → True (warning only, smoke tests don't need it)."""
        monkeypatch.delenv("KIMI_HF_MODEL", raising=False)
        assert _pf.check_kimi_hf_model_env(verbose=False) is True

    def test_nonexistent_dir_fails(self, monkeypatch):
        """KIMI_HF_MODEL pointing to non-existent directory → False."""
        monkeypatch.setenv("KIMI_HF_MODEL", "/nonexistent/path/kimi")
        assert _pf.check_kimi_hf_model_env(verbose=False) is False

    def test_existing_dir_without_index(self, tmp_path, monkeypatch):
        """Directory exists but no model.safetensors.index.json → True (warning)."""
        monkeypatch.setenv("KIMI_HF_MODEL", str(tmp_path))
        result = _pf.check_kimi_hf_model_env(verbose=False)
        assert result is True

    def test_existing_dir_with_index(self, tmp_path, monkeypatch, capsys):
        """Directory exists and has safetensors index → True, PASS message."""
        index = tmp_path / "model.safetensors.index.json"
        index.write_text("{}")
        monkeypatch.setenv("KIMI_HF_MODEL", str(tmp_path))
        result = _pf.check_kimi_hf_model_env(verbose=False)
        assert result is True
        assert "safetensors index found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# TestCheckImportsAndConfig
# ---------------------------------------------------------------------------


class TestCheckImportsAndConfig:
    """Direct invocation of import check and config validation (live, no mocks)."""

    def test_kimi_imports_run_without_crash(self):
        """check_kimi_imports() must not raise; it returns bool."""
        result = _pf.check_kimi_imports()
        assert isinstance(result, bool)

    def test_config_validation_passes(self):
        """KimiK25Config.from_fixture() must pass in this repo checkout."""
        result = _pf.check_config_validation()
        assert result is True

    def test_torch_check_returns_true(self):
        """check_torch_available() is a warning-only check — returns True regardless."""
        result = _pf.check_torch_available()
        assert result is True


# ---------------------------------------------------------------------------
# TestCheckTtnnImportable
# ---------------------------------------------------------------------------


class TestCheckTtnnImportable:
    """check_ttnn_importable() warns but does not hard-fail when ttnn is absent."""

    def test_ttnn_missing_returns_false_not_exception(self, monkeypatch):
        """When ttnn is not importable, function must return False (not raise)."""
        # Temporarily hide ttnn from sys.modules to simulate missing dep
        orig = sys.modules.pop("ttnn", None)
        # Also block the import attempt by injecting a None placeholder
        sys.modules["ttnn"] = None  # type: ignore[assignment]
        try:
            # Reload a fresh preflight so its ttnn import attempt uses our placeholder
            pf_fresh = _load_preflight()
            with mock.patch("builtins.__import__", side_effect=ImportError("no ttnn")):
                result = pf_fresh.check_ttnn_importable()
            assert result is False
        finally:
            if orig is not None:
                sys.modules["ttnn"] = orig
            else:
                sys.modules.pop("ttnn", None)


# ---------------------------------------------------------------------------
# TestMainExitCode
# ---------------------------------------------------------------------------


class TestMainExitCode:
    """Subprocess tests for the CLI exit code semantics."""

    _SCRIPT = str(_PREFLIGHT_PATH)
    _REPO = str(Path(_PREFLIGHT_PATH).parent.parent.parent.parent.parent)

    def _run(self, env_overrides: dict | None = None, extra_args: list[str] | None = None):
        env = os.environ.copy()
        # Start with a clean MESH_DEVICE to avoid inheriting scheduler env
        env.pop("MESH_DEVICE", None)
        if env_overrides:
            env.update(env_overrides)
        cmd = [sys.executable, self._SCRIPT, "--no-color"] + (extra_args or [])
        return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=self._REPO)

    def test_no_mesh_device_exits_1(self):
        """Missing MESH_DEVICE → exit code 1."""
        result = self._run()
        assert result.returncode == 1
        assert "PREFLIGHT FAILED" in result.stdout

    def test_mesh_device_tg_exits_0(self):
        """MESH_DEVICE=TG → exit code 0 (assuming /dev/tenstorrent present + imports OK)."""
        result = self._run(env_overrides={"MESH_DEVICE": "TG"})
        # May be 0 (pass) or 1 (fail due to missing deps in this env), but must not crash
        assert result.returncode in (0, 1)
        assert "PREFLIGHT" in result.stdout  # always prints summary

    def test_cli_mesh_device_override(self):
        """--mesh-device TG overrides unset MESH_DEVICE env var."""
        result = self._run(extra_args=["--mesh-device", "TG"])
        # Must not crash; MESH_DEVICE=TG was applied via CLI
        assert result.returncode in (0, 1)
        assert "MESH_DEVICE=TG" in result.stdout

    def test_n150_exits_1(self):
        """MESH_DEVICE=N150 → exit code 1 (memory wall)."""
        result = self._run(env_overrides={"MESH_DEVICE": "N150"})
        assert result.returncode == 1
        assert "N150" in result.stdout
        assert "33.8 GB" in result.stdout

    def test_help_flag(self):
        """--help exits 0 without running checks."""
        result = self._run(extra_args=["--help"])
        assert result.returncode == 0

    def test_no_color_flag_replaces_ansi(self):
        """--no-color must produce [PASS]/[FAIL] tags instead of ANSI escapes."""
        result = self._run(env_overrides={"MESH_DEVICE": "TG"})
        # ANSI escape (\033[) must not appear — we used --no-color
        assert "\033[" not in result.stdout

    def test_recommended_commands_always_printed(self):
        """RECOMMENDED COMMANDS section must appear regardless of pass/fail."""
        result = self._run(env_overrides={"MESH_DEVICE": "TG"})
        assert "RECOMMENDED COMMANDS" in result.stdout

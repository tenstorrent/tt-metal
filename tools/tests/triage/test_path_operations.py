# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for path-related operations across triage modules.

Covers path resolution, file lookup, caching, and OS-level helpers in:
  - parse_inspector_logs.py  (log directory resolution, YAML reading, kernel loading)
  - elfs_cache.py            (ELF path validation, cache lifecycle)
  - system_info.py           (OS version extraction from /etc/os-release)
  - dump_risc_debug_signals.py (git commit hash helper)
  - triage.py                (run_script path logic)

All tests are mock-based and require no Tenstorrent hardware.
Written to validate path handling during the pathlib migration.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# sys.path setup — mirrors the pattern in test_lw_assert_handling.py
# ---------------------------------------------------------------------------
metal_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
triage_home = os.path.join(metal_home, "tools", "triage")
if triage_home not in sys.path:
    sys.path.insert(0, triage_home)

# ---------------------------------------------------------------------------
# Pre-mock heavy third-party and ttexalens dependencies so that triage modules
# can be imported without a real tt-exalens / capnp / rich installation.
# This block MUST run before any triage module is imported at module level.
# ---------------------------------------------------------------------------
_MOCK_MODULES = {
    # ttexalens family
    "ttexalens": MagicMock(),
    "ttexalens.tt_exalens_init": MagicMock(),
    "ttexalens.tt_exalens_lib": MagicMock(),
    "ttexalens.context": MagicMock(),
    "ttexalens.device": MagicMock(),
    "ttexalens.coordinate": MagicMock(),
    "ttexalens.elf": MagicMock(),
    "ttexalens.hardware": MagicMock(),
    "ttexalens.hardware.risc_debug": MagicMock(),
    "ttexalens.umd_device": MagicMock(),
    "ttexalens.util": MagicMock(),
    # capnp
    "capnp": MagicMock(),
    # rich (triage.py imports rich.console, rich.progress, rich.table)
    "rich": MagicMock(),
    "rich.console": MagicMock(),
    "rich.progress": MagicMock(),
    "rich.table": MagicMock(),
    # triage peer modules used by elfs_cache / dump_risc_debug_signals
    "triage_session": MagicMock(),
    "triage_hw_utils": MagicMock(),
    "run_checks": MagicMock(),
    "dispatcher_data": MagicMock(),
    "inspector_capnp": MagicMock(),
    # ryml (optional fast YAML parser used in read_yaml)
    "ryml": MagicMock(),
}

# Install mock modules — only if they are not already importable.
for mod_name, mock_obj in _MOCK_MODULES.items():
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mock_obj

# Now safe to import triage modules.
from triage import run_script, TTTriageError  # noqa: E402
from parse_inspector_logs import get_log_directory, read_yaml, get_kernels, get_data  # noqa: E402
from system_info import get_os_version  # noqa: E402


# ===================================================================
# parse_inspector_logs.py — log directory, YAML reading, kernels
# ===================================================================


class TestGetLogDirectory:
    """Tests for parse_inspector_logs.get_log_directory."""

    def test_get_log_directory_explicit_path(self):
        """When a path string is passed, it is returned as a Path."""
        result = get_log_directory("/some/explicit/path")
        assert isinstance(result, Path)
        assert result == Path("/some/explicit/path")

    def test_get_log_directory_env_var(self, monkeypatch):
        """TT_METAL_LOGS_PATH set -> returns $VAR/generated/inspector as Path."""
        monkeypatch.setenv("TT_METAL_LOGS_PATH", "/custom/logs")
        result = get_log_directory(None)
        assert isinstance(result, Path)
        assert result == Path("/custom/logs") / "generated" / "inspector"

    def test_get_log_directory_fallback(self, monkeypatch):
        """No env var -> returns path under tempfile.gettempdir() as Path."""
        monkeypatch.delenv("TT_METAL_LOGS_PATH", raising=False)
        result = get_log_directory(None)
        assert isinstance(result, Path)
        assert result == Path(tempfile.gettempdir()) / "tt-metal" / "inspector"


class TestReadYaml:
    """Tests for parse_inspector_logs.read_yaml."""

    def test_read_yaml_missing_file(self, tmp_path):
        """Non-existent file path -> returns []."""
        with patch("parse_inspector_logs.utils.WARN"):
            result = read_yaml(str(tmp_path / "nonexistent.yaml"))
        assert result == []

    def test_read_yaml_valid(self, tmp_path):
        """Create a temp YAML file, verify it is parsed correctly."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("- kernel:\n" "    name: foo\n" "    value: 42\n")
        # read_yaml first tries ryml (mocked — will raise), then falls back to pyyaml.
        # Force the fallback by making the mock ryml raise on parse_in_arena.
        with patch.dict("sys.modules", {"ryml": MagicMock(**{"parse_in_arena.side_effect": Exception("no ryml")})}):
            result = read_yaml(str(yaml_file))
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["kernel"]["name"] == "foo"
        assert result[0]["kernel"]["value"] == 42


class TestGetKernels:
    """Tests for parse_inspector_logs.get_kernels."""

    def test_get_kernels_path(self, tmp_path):
        """Creates a temp dir with kernels.yaml, verifies get_kernels() reads it."""
        yaml_content = (
            "- kernel:\n"
            "    watcher_kernel_id: 1\n"
            "    name: my_kernel\n"
            "    path: /some/path\n"
            "    source: dataflow\n"
            "    program_id: 10\n"
        )
        kernels_file = tmp_path / "kernels.yaml"
        kernels_file.write_text(yaml_content)

        # Ensure ryml fallback
        with patch.dict("sys.modules", {"ryml": MagicMock(**{"parse_in_arena.side_effect": Exception})}):
            kernels = get_kernels(str(tmp_path))
        assert 1 in kernels
        assert kernels[1].name == "my_kernel"
        assert kernels[1].path == "/some/path"
        assert kernels[1].programId == 10


class TestGetData:
    """Tests for parse_inspector_logs.get_data when directory is missing."""

    def test_get_data_missing_directory(self, tmp_path):
        """Raises ValueError when the log directory does not exist."""
        missing = str(tmp_path / "does_not_exist")
        with pytest.raises(ValueError, match="does not exist"):
            get_data(missing)


# ===================================================================
# elfs_cache.py — ELF path validation, cache lifecycle
# ===================================================================


class TestElfsCache:
    """Tests for the ElfsCache class from elfs_cache.py.

    We import the class directly and instantiate with a mock context,
    bypassing the triage_singleton entry point.
    """

    @pytest.fixture(autouse=True)
    def _setup_cache(self):
        """Set up a fresh ElfsCache with a mocked context."""
        # elfs_cache imports from triage and ttexalens — both already mocked above.
        # Force-reimport to pick up mocks cleanly each time.
        if "elfs_cache" in sys.modules:
            del sys.modules["elfs_cache"]
        from elfs_cache import ElfsCache

        self.mock_context = MagicMock()
        self.cache = ElfsCache(self.mock_context)

    def test_elfs_cache_missing_elf(self):
        """Path that doesn't exist raises TTTriageError."""
        with pytest.raises(TTTriageError, match="does not exist"):
            self.cache["/nonexistent/path/to/elf"]

    def test_elfs_cache_clear(self):
        """clear_cache() empties the cache."""
        self.cache._cache["fake.elf"] = MagicMock()
        assert len(self.cache._cache) == 1
        self.cache.clear_cache()
        assert len(self.cache._cache) == 0

    def test_elfs_cache_get_cached_paths(self):
        """After adding entries, get_cached_paths() returns them."""
        self.cache._cache["/a/b.elf"] = MagicMock()
        self.cache._cache["/c/d.elf"] = MagicMock()
        paths = self.cache.get_cached_paths()
        assert set(paths) == {"/a/b.elf", "/c/d.elf"}

    def test_elfs_cache_has_elf(self):
        """has_elf() returns False before caching."""
        assert self.cache.has_elf("/not/cached.elf") is False


# ===================================================================
# system_info.py — OS version extraction
# ===================================================================


class TestGetOsVersion:
    """Tests for system_info.get_os_version."""

    def test_get_os_version_reads_file(self):
        """Mock file content, verify VERSION_ID is extracted."""
        file_content = 'NAME="Ubuntu"\nVERSION_ID="22.04"\nID=ubuntu\n'
        with patch("pathlib.Path.open", mock_open(read_data=file_content)):
            result = get_os_version()
        assert result == "22.04"

    def test_get_os_version_missing_file(self):
        """Mocked missing file -> returns None."""
        with patch("pathlib.Path.open", side_effect=FileNotFoundError):
            result = get_os_version()
        assert result is None


# ===================================================================
# dump_risc_debug_signals.py — git commit hash helper
# ===================================================================


class TestGetGitCommitHash:
    """Tests for dump_risc_debug_signals._get_git_commit_hash."""

    @pytest.fixture(autouse=True)
    def _ensure_importable(self):
        """Force-reimport to get a clean reference to the private function."""
        if "dump_risc_debug_signals" in sys.modules:
            del sys.modules["dump_risc_debug_signals"]
        from dump_risc_debug_signals import _get_git_commit_hash

        self._get_git_commit_hash = _get_git_commit_hash

    def test_get_git_commit_hash_success(self):
        """Mock subprocess.run returning a SHA -> function returns it."""
        fake_sha = "abc123def456789012345678901234567890abcd"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_sha + "\n"
        with patch("dump_risc_debug_signals.subprocess.run", return_value=mock_result):
            assert self._get_git_commit_hash() == fake_sha

    def test_get_git_commit_hash_failure(self):
        """Mock subprocess failure -> returns 'unknown'."""
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stdout = ""
        with patch("dump_risc_debug_signals.subprocess.run", return_value=mock_result):
            assert self._get_git_commit_hash() == "unknown"


# ===================================================================
# triage.py — run_script path resolution
# ===================================================================


class TestRunScript:
    """Tests for triage.run_script path resolution logic."""

    def test_run_script_appends_py_suffix(self):
        """Path without .py gets .py appended before checking existence.

        run_script adds '.py' to bare names then resolves relative to triage_home.
        Since 'nonexistent_script_name_xyz.py' won't exist, FileNotFoundError
        is raised and the message contains the .py suffix.
        """
        with pytest.raises(FileNotFoundError, match=r"\.py"):
            run_script(script_path="nonexistent_script_name_xyz")

    def test_run_script_nonexistent_raises(self):
        """Raises FileNotFoundError for a non-existent script path."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            run_script(script_path="/absolutely/fake/path/no_such_script.py")

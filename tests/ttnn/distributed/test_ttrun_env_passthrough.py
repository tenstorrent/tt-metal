#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit and integration tests for tt-run environment management.

Covers env propagation, blocklisting, path safety validation, rank-scoped path
scoping, MPI export classification, and end-to-end rank environment assembly.

Unit tests (marked @pytest.mark.unit) require no MPI runtime or hardware.
Multihost integration tests (@pytest.mark.multihost) require a real MPI environment
with tt-run available and a dual-T3K rank binding configuration.

Running from /tmp with PYTHONPATH unset
---------------------------------------
The test runner invokes this file as::

    cd /tmp && env -u PYTHONPATH pytest ... tests/ttnn/distributed/test_ttrun_env_passthrough.py

Reason: when pytest is launched from the repository root, ``PYTHONPATH`` typically
includes ``TT_METAL_HOME``, which puts the top-level ``ttnn/`` CMake directory on
the import path.  That directory contains a stub/build-time ``ttnn/`` folder that
shadows the real installed Python package, breaking ``import ttnn.distributed.ttrun``.
Running from ``/tmp`` with ``PYTHONPATH`` cleared ensures Python resolves ``ttnn``
from the installed package (or the editable install) rather than the CMake tree.
The module-load helper ``_import_ttrun_module()`` additionally falls back to loading
``ttrun.py`` by source path when the package import fails.
"""

import importlib.util
import json
import os
import shutil
import uuid
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import pytest
from click.testing import CliRunner

_TTRUN_MODULE_NAME = "ttnn.distributed.ttrun"
_TTRUN_REL_PARTS = ("ttnn", "ttnn", "distributed", "ttrun.py")
# Symbols that must exist on a usable ttrun module (guards against half-broken imports).
_TTRUN_REQUIRED_ATTRS = ("ENV_BLOCKLIST", "RankBinding", "get_rank_environment", "validate_path_safety")


def _unique_resolved_dirs(paths: Iterable[Path | str | None]) -> list[Path]:
    """Return existing, unique directory paths (resolved, no duplicates)."""
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        if not p:
            continue
        try:
            d = Path(p).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if not d.is_dir():
            continue
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


def _candidate_ttrun_source_paths() -> list[Path]:
    """Ordered list of plausible ``ttrun.py`` locations in a tt-metal checkout."""
    this_file = Path(__file__).resolve()
    raw: list[Path | str | None] = [
        os.environ.get("TT_METAL_HOME"),
        os.environ.get("GITHUB_WORKSPACE"),
        os.environ.get("CI_PROJECT_DIR"),
        os.environ.get("WORKSPACE"),  # some Jenkins / custom runners
        os.environ.get("TT_METAL_SRC_ROOT"),  # optional explicit checkout root
        # Default layout: tests/ttnn/distributed/<this file> → repo root
        this_file.parents[3],
    ]
    # Walk upward: supports extra nesting, symlinks, or non-standard test paths.
    for depth, parent in enumerate(this_file.parents):
        if depth > 20:
            break
        raw.append(parent)

    roots = _unique_resolved_dirs(raw)
    candidates: list[Path] = []
    seen_paths: set[Path] = set()
    for root in roots:
        cand = root.joinpath(*_TTRUN_REL_PARTS).resolve()
        if cand.is_file() and cand not in seen_paths:
            seen_paths.add(cand)
            candidates.append(cand)
    return candidates


def _ttrun_module_looks_valid(mod: object) -> bool:
    return all(hasattr(mod, name) for name in _TTRUN_REQUIRED_ATTRS)


def _load_ttrun_from_source(path: Path, *, first_exc: BaseException | None) -> object:
    """Load ``ttrun`` from *path* and register as ``ttnn.distributed.ttrun``."""
    name = _TTRUN_MODULE_NAME
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}") from first_exc

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise

    if not _ttrun_module_looks_valid(mod):
        sys.modules.pop(name, None)
        raise ImportError(
            f"Loaded {path} as {_TTRUN_MODULE_NAME} but it is missing expected API " f"({_TTRUN_REQUIRED_ATTRS})"
        ) from first_exc
    return mod


def _import_ttrun_module():
    """Load ``ttnn.distributed.ttrun``, with a source-tree fallback.

    When pytest runs from the repository root with ``PYTHONPATH`` including
    ``TT_METAL_HOME``, the top-level ``ttnn/`` CMake directory can shadow the
    real Python package and break ``import ttnn.distributed.ttrun`` (see
    ``setup.py`` editable-install notes).  Loading ``ttrun.py`` by path from
    the checkout avoids that failure while still registering the module as
    ``ttnn.distributed.ttrun`` for tests that monkeypatch ``ttrun``.

    A half-registered broken submodule in ``sys.modules`` (rare, but possible
    after a failed import) is replaced when we load from source.
    """
    first_exc: BaseException | None = None
    try:
        import ttnn.distributed.ttrun as mod  # deferred: try site-packages / editable resolution first

        if _ttrun_module_looks_valid(mod):
            return mod
        first_exc = ImportError(
            f"import {_TTRUN_MODULE_NAME} succeeded but module is missing expected API "
            f"({_TTRUN_REQUIRED_ATTRS}); will try source fallback"
        )
        sys.modules.pop(_TTRUN_MODULE_NAME, None)
    except ImportError as exc:
        first_exc = exc

    existing = sys.modules.get(_TTRUN_MODULE_NAME)
    if existing is not None and not _ttrun_module_looks_valid(existing):
        sys.modules.pop(_TTRUN_MODULE_NAME, None)

    tried_files: list[str] = []
    last_exc: BaseException | None = first_exc
    for path in _candidate_ttrun_source_paths():
        tried_files.append(str(path))
        try:
            return _load_ttrun_from_source(path, first_exc=last_exc)
        except ImportError as load_exc:
            last_exc = load_exc
            continue
        except Exception as load_exc:
            last_exc = load_exc
            continue

    detail = "\n  ".join(tried_files) if tried_files else "(no candidate paths)"
    raise ImportError(
        f"Could not import {_TTRUN_MODULE_NAME} ({last_exc!r}).\n"
        f"Tried source fallbacks ({len(tried_files)} path(s)):\n  {detail}"
    ) from last_exc


ttrun = _import_ttrun_module()

ENV_BLOCKLIST = ttrun.ENV_BLOCKLIST
FORCE_NAME_ONLY_MPI_EXPORT_VARS = ttrun.FORCE_NAME_ONLY_MPI_EXPORT_VARS
RankBinding = ttrun.RankBinding
TTRunConfig = ttrun.TTRunConfig
apply_rank_scoped_paths = ttrun.apply_rank_scoped_paths
build_rank_environment_args = ttrun.build_rank_environment_args
classify_mpi_env_exports = ttrun.classify_mpi_env_exports
get_launcher_environment = ttrun.get_launcher_environment
get_rank_environment = ttrun.get_rank_environment
has_auto_passthrough_prefix = ttrun.has_auto_passthrough_prefix
has_blocked_prefix = ttrun.has_blocked_prefix
is_auto_passthrough_env_var = ttrun.is_auto_passthrough_env_var
is_blocklisted_env_var = ttrun.is_blocklisted_env_var
main = ttrun.main
should_use_name_only_mpi_export = ttrun.should_use_name_only_mpi_export
strip_blocklisted_env_vars = ttrun.strip_blocklisted_env_vars
validate_path_safety = ttrun.validate_path_safety


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_config(tmp_path: Path, binding: RankBinding, global_env: dict | None = None) -> TTRunConfig:
    mesh_graph_desc = tmp_path / "mesh_graph_desc.textproto"
    mesh_graph_desc.write_text("mesh_graph_desc: test\n")
    rank_bindings = []
    for r in range(binding.rank + 1):
        if r == binding.rank:
            rank_bindings.append(binding)
        else:
            rank_bindings.append(
                RankBinding(
                    rank=r,
                    mesh_id=binding.mesh_id,
                    mesh_host_rank=binding.mesh_host_rank,
                    env_overrides={},
                )
            )
    return TTRunConfig(
        rank_bindings=rank_bindings,
        global_env=global_env or {},
        mesh_graph_desc_path=mesh_graph_desc,
    )


# ===========================================================================
# Unit tests — low-level utility functions
# ===========================================================================


# ---------------------------------------------------------------------------
# validate_path_safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidatePathSafety:
    def test_empty_path_is_allowed(self):
        validate_path_safety("", "TEST_VAR")

    def test_absolute_safe_path_passes(self):
        validate_path_safety("/tmp/tt-metal-cache", "TT_METAL_CACHE")

    def test_relative_path_raises(self):
        with pytest.raises(ValueError, match="must be an absolute path"):
            validate_path_safety("relative/path", "TT_METAL_CACHE")

    def test_parent_traversal_raises(self):
        with pytest.raises(ValueError, match="parent directory traversal"):
            validate_path_safety("/tmp/../etc/passwd", "TT_METAL_CACHE")

    def test_sensitive_etc_raises(self):
        with pytest.raises(ValueError, match="sensitive system directory"):
            validate_path_safety("/etc/tt-metal", "TT_METAL_CACHE")

    def test_sensitive_proc_raises(self):
        with pytest.raises(ValueError, match="sensitive system directory"):
            validate_path_safety("/proc/self/cwd", "TT_METAL_LOGS_PATH")

    def test_sensitive_dev_raises(self):
        with pytest.raises(ValueError, match="sensitive system directory"):
            validate_path_safety("/dev/shm/cache", "TT_METAL_CACHE")

    def test_sensitive_boot_raises(self):
        with pytest.raises(ValueError, match="sensitive system directory"):
            validate_path_safety("/boot/something", "TT_METAL_CACHE")

    def test_home_path_passes(self):
        validate_path_safety("/home/user/.cache/tt-metal", "TT_METAL_CACHE")

    def test_tmp_path_passes(self):
        validate_path_safety("/tmp/tt-jit-build", "TT_METAL_JIT_SCRATCH")


# ---------------------------------------------------------------------------
# apply_rank_scoped_paths (direct unit tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyRankScopedPaths:
    def test_scopes_logs_and_scratch(self):
        hostname = socket.gethostname()
        env = {
            "TT_METAL_LOGS_PATH": "/nfs/logs",
            "TT_METAL_JIT_SCRATCH": "/tmp/tt-jit-build",
        }
        apply_rank_scoped_paths(env, rank=3)
        assert env["TT_METAL_LOGS_PATH"] == f"/nfs/logs/{hostname}_rank_3"
        assert env["TT_METAL_JIT_SCRATCH"] == f"/tmp/tt-jit-build/{hostname}_rank_3"

    def test_skips_explicit_keys(self):
        env = {
            "TT_METAL_LOGS_PATH": "/custom/logs",
            "TT_METAL_JIT_SCRATCH": "/tmp/tt-jit-build",
        }
        apply_rank_scoped_paths(env, rank=0, explicit_keys={"TT_METAL_LOGS_PATH"})
        assert env["TT_METAL_LOGS_PATH"] == "/custom/logs"
        hostname = socket.gethostname()
        assert env["TT_METAL_JIT_SCRATCH"] == f"/tmp/tt-jit-build/{hostname}_rank_0"

    def test_does_not_double_scope(self):
        hostname = socket.gethostname()
        already_scoped = f"/nfs/logs/{hostname}_rank_5"
        env = {"TT_METAL_LOGS_PATH": already_scoped}
        apply_rank_scoped_paths(env, rank=5)
        assert env["TT_METAL_LOGS_PATH"] == already_scoped

    def test_missing_keys_are_ignored(self):
        env = {"UNRELATED_VAR": "value"}
        apply_rank_scoped_paths(env, rank=0)
        assert env == {"UNRELATED_VAR": "value"}

    def test_empty_value_is_skipped(self):
        env = {"TT_METAL_LOGS_PATH": ""}
        apply_rank_scoped_paths(env, rank=0)
        assert env["TT_METAL_LOGS_PATH"] == ""


# ---------------------------------------------------------------------------
# classify_mpi_env_exports
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassifyMpiEnvExports:
    def test_simple_values_go_to_direct_exports(self):
        rank_env = {"TT_METAL_HOME": "/opt/tt-metal", "TT_MESH_ID": "0"}
        launcher_env = {}
        direct, name_only, missing = classify_mpi_env_exports(rank_env, launcher_env)
        assert direct == rank_env
        assert name_only == []
        assert missing == []

    def test_special_chars_with_matching_launcher_go_to_name_only(self):
        value_with_spaces = "echo 'hello world'"
        rank_env = {"MY_VAR": value_with_spaces}
        launcher_env = {"MY_VAR": value_with_spaces}
        direct, name_only, missing = classify_mpi_env_exports(rank_env, launcher_env)
        assert "MY_VAR" not in direct
        assert "MY_VAR" in name_only
        assert missing == []

    def test_special_chars_without_launcher_match_stay_direct(self):
        value_with_spaces = "echo 'hello world'"
        rank_env = {"MY_VAR": value_with_spaces}
        launcher_env = {}
        direct, name_only, missing = classify_mpi_env_exports(rank_env, launcher_env)
        assert "MY_VAR" in direct
        assert name_only == []

    def test_force_name_only_vars(self):
        for forced_key in FORCE_NAME_ONLY_MPI_EXPORT_VARS:
            rank_env = {forced_key: "some_value"}
            launcher_env = {forced_key: "some_value"}
            direct, name_only, missing = classify_mpi_env_exports(rank_env, launcher_env)
            assert forced_key in name_only
            assert forced_key not in direct

    def test_force_name_only_missing_from_launcher_reports_missing(self):
        for forced_key in FORCE_NAME_ONLY_MPI_EXPORT_VARS:
            rank_env = {forced_key: "some_value"}
            launcher_env = {}
            direct, name_only, missing = classify_mpi_env_exports(rank_env, launcher_env)
            assert forced_key in direct
            assert forced_key not in name_only


# ---------------------------------------------------------------------------
# strip_blocklisted_env_vars
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStripBlocklistedEnvVars:
    def test_removes_exact_blocklist_keys(self):
        env = {"TT_VISIBLE_DEVICES": "0,1", "SAFE_VAR": "value"}
        assert "TT_VISIBLE_DEVICES" in ENV_BLOCKLIST
        result = strip_blocklisted_env_vars(env)
        assert "TT_VISIBLE_DEVICES" not in result
        assert result["SAFE_VAR"] == "value"

    def test_removes_prefix_blocklisted_keys(self):
        env = {"GITHUB_TOKEN": "secret", "RUNNER_OS": "Linux", "HOME": "/home/user"}
        result = strip_blocklisted_env_vars(env)
        assert "GITHUB_TOKEN" not in result
        assert "RUNNER_OS" not in result
        assert result["HOME"] == "/home/user"

    def test_removes_launcher_only_blocklist_keys(self):
        env = {"CI": "true", "HOSTNAME": "node-1", "PATH": "/usr/bin"}
        result = strip_blocklisted_env_vars(env)
        assert "CI" not in result
        assert "HOSTNAME" not in result
        assert result["PATH"] == "/usr/bin"

    def test_preserves_non_blocklisted_keys(self):
        env = {"HOME": "/home/user", "PATH": "/usr/bin", "CUSTOM_VAR": "value"}
        result = strip_blocklisted_env_vars(env)
        assert result == env


# ---------------------------------------------------------------------------
# should_use_name_only_mpi_export
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShouldUseNameOnlyMpiExport:
    def test_plain_value_returns_false(self):
        assert not should_use_name_only_mpi_export("MY_VAR", "simple_value", {})

    def test_value_with_spaces_and_matching_launcher_returns_true(self):
        assert should_use_name_only_mpi_export("MY_VAR", "has spaces", {"MY_VAR": "has spaces"})

    def test_value_with_spaces_and_different_launcher_returns_false(self):
        assert not should_use_name_only_mpi_export("MY_VAR", "has spaces", {"MY_VAR": "different"})

    def test_value_with_spaces_and_missing_launcher_returns_false(self):
        assert not should_use_name_only_mpi_export("MY_VAR", "has spaces", {})

    def test_value_with_pipe_and_matching_launcher_returns_true(self):
        assert should_use_name_only_mpi_export("CMD", "a | b", {"CMD": "a | b"})

    def test_forced_key_with_launcher_present_returns_true(self):
        for forced_key in FORCE_NAME_ONLY_MPI_EXPORT_VARS:
            assert should_use_name_only_mpi_export(forced_key, "any_value", {forced_key: "any_value"})

    def test_forced_key_without_launcher_returns_false(self):
        for forced_key in FORCE_NAME_ONLY_MPI_EXPORT_VARS:
            assert not should_use_name_only_mpi_export(forced_key, "any_value", {})


# ---------------------------------------------------------------------------
# Prefix / blocklist helper functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnvPrefixHelpers:
    def test_has_auto_passthrough_prefix_tt(self):
        assert has_auto_passthrough_prefix("TT_METAL_HOME")
        assert has_auto_passthrough_prefix("TT_VISIBLE_DEVICES")
        assert has_auto_passthrough_prefix("TTNN_CONFIG_OVERRIDES")

    def test_has_auto_passthrough_prefix_other_prefixes(self):
        assert has_auto_passthrough_prefix("ARCH_NAME")
        assert has_auto_passthrough_prefix("WH_ARCH_YAML")
        assert has_auto_passthrough_prefix("DEEPSEEK_V3_HF_MODEL")
        assert has_auto_passthrough_prefix("MESH_DEVICE")

    def test_has_auto_passthrough_prefix_non_matching(self):
        assert not has_auto_passthrough_prefix("HOME")
        assert not has_auto_passthrough_prefix("PATH")
        assert not has_auto_passthrough_prefix("SLURM_JOB_ID")

    def test_has_blocked_prefix(self):
        assert has_blocked_prefix("GITHUB_TOKEN")
        assert has_blocked_prefix("RUNNER_OS")
        assert not has_blocked_prefix("TT_METAL_HOME")
        assert not has_blocked_prefix("HOME")

    def test_is_blocklisted_env_var_exact_match(self):
        assert is_blocklisted_env_var("TT_VISIBLE_DEVICES")
        assert is_blocklisted_env_var("TT_MESH_ID")

    def test_is_blocklisted_env_var_prefix_match(self):
        assert is_blocklisted_env_var("GITHUB_ACTIONS")
        assert is_blocklisted_env_var("RUNNER_TEMP")

    def test_is_blocklisted_env_var_launcher_only(self):
        assert is_blocklisted_env_var("CI", include_launcher_only=True)
        assert not is_blocklisted_env_var("CI", include_launcher_only=False)

    def test_is_auto_passthrough_excludes_blocklisted(self):
        assert not is_auto_passthrough_env_var("TT_VISIBLE_DEVICES")
        assert not is_auto_passthrough_env_var("TT_MESH_ID")
        assert is_auto_passthrough_env_var("TT_METAL_HOME")
        assert is_auto_passthrough_env_var("TT_METAL_CACHE")


# ===========================================================================
# Integration tests — end-to-end rank environment assembly
# ===========================================================================


@pytest.mark.unit
def test_tt_visible_devices_host_value_is_not_passed_without_rank_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "31")
    monkeypatch.setenv("TTNN_CONFIG_OVERRIDES", '{"foo": "bar"}')

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert "TT_VISIBLE_DEVICES" not in env
    assert env["TTNN_CONFIG_OVERRIDES"] == '{"foo": "bar"}'


@pytest.mark.unit
def test_tt_visible_devices_rank_override_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "31")

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0, env_overrides={"TT_VISIBLE_DEVICES": "0,1"})
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["TT_VISIBLE_DEVICES"] == "0,1"


@pytest.mark.unit
def test_launcher_environment_strips_blocklisted_variables(monkeypatch):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "31")
    monkeypatch.setenv("TT_MESH_ID", "7")
    monkeypatch.setenv("TTNN_CONFIG_OVERRIDES", '{"foo": "bar"}')
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("RUNNER_OS", "Linux")
    monkeypatch.setenv("ACTIONS_ORCHESTRATION_ID", "123")
    monkeypatch.setenv("CCACHE_TEMPDIR", "/tmp/ccache")
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("DEBIAN_FRONTEND", "noninteractive")
    monkeypatch.setenv("HOSTNAME", "runner-host")

    launcher_env = get_launcher_environment()

    assert "TT_VISIBLE_DEVICES" not in launcher_env
    assert "TT_MESH_ID" not in launcher_env
    assert "GITHUB_SHA" not in launcher_env
    assert "RUNNER_OS" not in launcher_env
    assert "ACTIONS_ORCHESTRATION_ID" not in launcher_env
    assert "CCACHE_TEMPDIR" not in launcher_env
    assert "CI" not in launcher_env
    assert "DEBIAN_FRONTEND" not in launcher_env
    assert "HOSTNAME" not in launcher_env
    assert launcher_env["TTNN_CONFIG_OVERRIDES"] == '{"foo": "bar"}'


@pytest.mark.unit
def test_rank_environment_does_not_auto_passthrough_non_prefixed_vars(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("RUNNER_OS", "Linux")
    monkeypatch.setenv("ARCH_NAME", "wormhole_b0")

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["ARCH_NAME"] == "wormhole_b0"
    assert "GITHUB_SHA" not in env
    assert "RUNNER_OS" not in env


@pytest.mark.unit
def test_mpi_export_uses_name_only_for_complex_dispatch_timeout_command(monkeypatch, tmp_path):
    command = "/opt/venv/bin/python3 /work/tools/tt-triage.py --disable-progress 1>&2"
    monkeypatch.setenv("TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE", command)

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0, env_overrides={"TT_VISIBLE_DEVICES": "0,1"})
    config = _build_config(tmp_path, binding)
    launcher_env = get_launcher_environment()

    env_args = build_rank_environment_args(binding, config, launcher_env)
    exported_tokens = [env_args[i + 1] for i, value in enumerate(env_args) if value == "-x"]

    assert "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE" in exported_tokens
    assert not any(token.startswith("TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE=") for token in exported_tokens)
    assert "TT_VISIBLE_DEVICES=0,1" in exported_tokens


@pytest.mark.unit
def test_rank_environment_provides_default_cache_when_unset(monkeypatch, tmp_path):
    """When TT_METAL_CACHE is not set, ttrun provides a shared default path.
    Cache remains shared across ranks; only logs and JIT scratch are rank-scoped."""
    monkeypatch.delenv("TT_METAL_CACHE", raising=False)
    monkeypatch.setenv("HOME", "/home/testuser")

    binding = RankBinding(rank=1, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)
    assert "TT_METAL_CACHE" in env
    assert env["TT_METAL_CACHE"] == "/home/testuser/.cache"


@pytest.mark.unit
def test_rank_environment_provides_default_cache_when_unset_no_home(monkeypatch, tmp_path):
    """When both TT_METAL_CACHE and HOME are unset, fall back to /tmp/tt-metal-cache."""
    monkeypatch.delenv("TT_METAL_CACHE", raising=False)
    monkeypatch.delenv("HOME", raising=False)

    binding = RankBinding(rank=2, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)
    assert env["TT_METAL_CACHE"] == "/tmp/tt-metal-cache"


@pytest.mark.unit
def test_rank_environment_keeps_cache_shared_but_scopes_logs_path_per_rank(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_METAL_CACHE", "/nfs/shared/tt-metal-cache")
    monkeypatch.setenv("TT_METAL_LOGS_PATH", "/nfs/shared/tt-metal-logs")

    binding = RankBinding(rank=3, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)
    rank_suffix = f"{socket.gethostname()}_rank_3"

    assert env["TT_METAL_CACHE"] == "/nfs/shared/tt-metal-cache"
    assert env["TT_METAL_LOGS_PATH"] == f"/nfs/shared/tt-metal-logs/{rank_suffix}"


@pytest.mark.unit
def test_rank_environment_does_not_double_append_rank_scoped_suffix(monkeypatch, tmp_path):
    rank_suffix = f"{socket.gethostname()}_rank_2"
    monkeypatch.setenv("TT_METAL_CACHE", f"/nfs/shared/tt-metal-cache/{rank_suffix}")
    monkeypatch.setenv("TT_METAL_LOGS_PATH", f"/nfs/shared/tt-metal-logs/{rank_suffix}")

    binding = RankBinding(rank=2, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["TT_METAL_CACHE"] == f"/nfs/shared/tt-metal-cache/{rank_suffix}"
    assert env["TT_METAL_LOGS_PATH"] == f"/nfs/shared/tt-metal-logs/{rank_suffix}"


@pytest.mark.unit
def test_rank_environment_preserves_explicit_global_env_rank_scoped_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_METAL_CACHE", "/nfs/shared/tt-metal-cache")
    monkeypatch.setenv("TT_METAL_LOGS_PATH", "/nfs/shared/tt-metal-logs")

    binding = RankBinding(rank=4, mesh_id=0, mesh_host_rank=0)
    config = _build_config(
        tmp_path,
        binding,
        global_env={"TT_METAL_CACHE": "/custom/global/cache", "TT_METAL_LOGS_PATH": "/custom/global/logs"},
    )

    env = get_rank_environment(binding, config)

    assert env["TT_METAL_CACHE"] == "/custom/global/cache"
    assert env["TT_METAL_LOGS_PATH"] == "/custom/global/logs"


@pytest.mark.unit
def test_rank_environment_preserves_explicit_rank_override_rank_scoped_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_METAL_CACHE", "/nfs/shared/tt-metal-cache")
    monkeypatch.setenv("TT_METAL_LOGS_PATH", "/nfs/shared/tt-metal-logs")

    binding = RankBinding(
        rank=5,
        mesh_id=0,
        mesh_host_rank=0,
        env_overrides={"TT_METAL_CACHE": "/custom/rank/cache", "TT_METAL_LOGS_PATH": "/custom/rank/logs"},
    )
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["TT_METAL_CACHE"] == "/custom/rank/cache"
    assert env["TT_METAL_LOGS_PATH"] == "/custom/rank/logs"


# --- TT_METAL_JIT_SCRATCH tests ---


@pytest.mark.unit
def test_rank_environment_provides_default_jit_scratch(monkeypatch, tmp_path):
    """When TT_METAL_JIT_SCRATCH is not set, ttrun provides /tmp/tt-jit-build
    and rank-scopes it so each rank compiles to a local scratch directory."""
    monkeypatch.delenv("TT_METAL_JIT_SCRATCH", raising=False)

    binding = RankBinding(rank=1, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)
    rank_suffix = f"{socket.gethostname()}_rank_1"

    assert "TT_METAL_JIT_SCRATCH" in env
    assert env["TT_METAL_JIT_SCRATCH"] == f"/tmp/tt-jit-build/{rank_suffix}"


@pytest.mark.unit
def test_rank_environment_scopes_explicit_jit_scratch(monkeypatch, tmp_path):
    """When TT_METAL_JIT_SCRATCH is set in the parent, ttrun still rank-scopes it."""
    monkeypatch.setenv("TT_METAL_JIT_SCRATCH", "/fast-local/jit")

    binding = RankBinding(rank=3, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)
    rank_suffix = f"{socket.gethostname()}_rank_3"

    assert env["TT_METAL_JIT_SCRATCH"] == f"/fast-local/jit/{rank_suffix}"


@pytest.mark.unit
def test_rank_environment_preserves_explicit_global_env_jit_scratch(monkeypatch, tmp_path):
    """When TT_METAL_JIT_SCRATCH is set via global_env in the config YAML, it
    is not overridden by the default and not rank-scoped (explicit key)."""
    monkeypatch.delenv("TT_METAL_JIT_SCRATCH", raising=False)

    binding = RankBinding(rank=2, mesh_id=0, mesh_host_rank=0)
    config = _build_config(
        tmp_path,
        binding,
        global_env={"TT_METAL_JIT_SCRATCH": "/custom/jit"},
    )

    env = get_rank_environment(binding, config)

    assert env["TT_METAL_JIT_SCRATCH"] == "/custom/jit"


@pytest.mark.unit
def test_rank_environment_preserves_explicit_rank_override_jit_scratch(monkeypatch, tmp_path):
    """When TT_METAL_JIT_SCRATCH is set via rank env_overrides, it is preserved."""
    monkeypatch.delenv("TT_METAL_JIT_SCRATCH", raising=False)

    binding = RankBinding(
        rank=4,
        mesh_id=0,
        mesh_host_rank=0,
        env_overrides={"TT_METAL_JIT_SCRATCH": "/rank-local/jit"},
    )
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["TT_METAL_JIT_SCRATCH"] == "/rank-local/jit"


@pytest.mark.unit
def test_cli_wraps_popen_failures_in_click_exception(monkeypatch, tmp_path):
    rank_binding = tmp_path / "rank_binding.yaml"
    rank_binding.write_text("rank_bindings: []\n")

    monkeypatch.setattr(
        ttrun,
        "parse_binding_config",
        lambda *_args, **_kwargs: SimpleNamespace(rank_bindings=[], mesh_graph_desc_path=Path("/tmp/mesh")),
    )
    monkeypatch.setattr(ttrun, "get_launcher_environment", lambda: {})
    monkeypatch.setattr(ttrun, "build_mpi_command", lambda *_args, **_kwargs: ["mpirun", "fake_program"])

    def raise_oserror(*_args, **_kwargs):
        raise OSError("launcher unavailable")

    monkeypatch.setattr(ttrun.subprocess, "Popen", raise_oserror)

    result = CliRunner().invoke(
        main, ["--rank-binding", str(rank_binding), "--skip-executable-check", "--bare", "fake_program"]
    )

    assert result.exit_code != 0
    assert "Error launching mpirun: launcher unavailable" in result.output


# --- Additional passthrough prefix tests ---


@pytest.mark.unit
def test_arch_prefix_passthrough(monkeypatch, tmp_path):
    """Test that ARCH_ prefixed variables are auto-propagated."""
    monkeypatch.setenv("ARCH_NAME", "wormhole_b0")

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["ARCH_NAME"] == "wormhole_b0"


@pytest.mark.unit
def test_wh_prefix_passthrough(monkeypatch, tmp_path):
    """Test that WH_ prefixed variables (Wormhole-specific) are auto-propagated."""
    monkeypatch.setenv("WH_ARCH_YAML", "/path/to/wh_arch.yaml")

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["WH_ARCH_YAML"] == "/path/to/wh_arch.yaml"


@pytest.mark.unit
def test_deepseek_prefix_passthrough(monkeypatch, tmp_path):
    """Test that DEEPSEEK_ prefixed variables are auto-propagated."""
    monkeypatch.setenv("DEEPSEEK_V3_HF_MODEL", "deepseek-ai/deepseek-v3")

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["DEEPSEEK_V3_HF_MODEL"] == "deepseek-ai/deepseek-v3"


@pytest.mark.unit
def test_mesh_prefix_passthrough(monkeypatch, tmp_path):
    """Test that MESH_ prefixed variables are auto-propagated."""
    monkeypatch.setenv("MESH_DEVICE", "T3000")

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["MESH_DEVICE"] == "T3000"


@pytest.mark.unit
def test_loguru_prefix_passthrough(monkeypatch, tmp_path):
    """Test that LOGURU_ prefixed variables are auto-propagated."""
    monkeypatch.setenv("LOGURU_LEVEL", "DEBUG")

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["LOGURU_LEVEL"] == "DEBUG"


@pytest.mark.unit
def test_launcher_only_blocklist_keys_stripped(monkeypatch):
    """Test that launcher-only blocklist keys are stripped from launcher environment."""
    monkeypatch.setenv("ACTIONS_ORCHESTRATION_ID", "12345")
    monkeypatch.setenv("CCACHE_TEMPDIR", "/tmp/ccache_temp")
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("DEBIAN_FRONTEND", "noninteractive")
    monkeypatch.setenv("HOSTNAME", "build-runner")

    launcher_env = get_launcher_environment()

    assert "ACTIONS_ORCHESTRATION_ID" not in launcher_env
    assert "CCACHE_TEMPDIR" not in launcher_env
    assert "CI" not in launcher_env
    assert "DEBIAN_FRONTEND" not in launcher_env
    assert "HOSTNAME" not in launcher_env


@pytest.mark.unit
def test_pythonhome_not_set_by_default(monkeypatch, tmp_path):
    """Test that PYTHONHOME is not set unless explicitly provided."""
    monkeypatch.delenv("PYTHONHOME", raising=False)

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert "PYTHONHOME" not in env


@pytest.mark.unit
def test_pythonhome_preserved_when_explicitly_set(monkeypatch, tmp_path):
    """Test that PYTHONHOME is preserved when explicitly set in parent environment."""
    monkeypatch.setenv("PYTHONHOME", "/custom/python/home")

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["PYTHONHOME"] == "/custom/python/home"


@pytest.mark.unit
def test_multi_rank_binding_scopes_paths_correctly(monkeypatch, tmp_path):
    """Cache is shared; logs are rank-scoped."""
    monkeypatch.setenv("TT_METAL_CACHE", "/shared/cache")
    monkeypatch.setenv("TT_METAL_LOGS_PATH", "/shared/logs")

    hostname = socket.gethostname()

    binding0 = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config0 = _build_config(tmp_path, binding0)
    env0 = get_rank_environment(binding0, config0)

    binding1 = RankBinding(rank=1, mesh_id=0, mesh_host_rank=1)
    config1 = _build_config(tmp_path, binding1)
    env1 = get_rank_environment(binding1, config1)

    assert env0["TT_METAL_CACHE"] == "/shared/cache"
    assert env1["TT_METAL_CACHE"] == "/shared/cache"
    assert env0["TT_METAL_LOGS_PATH"] == f"/shared/logs/{hostname}_rank_0"
    assert env1["TT_METAL_LOGS_PATH"] == f"/shared/logs/{hostname}_rank_1"


# ===========================================================================
# Multihost integration tests — real MPI env propagation via tt-run
# ===========================================================================
#
# These tests invoke tt-run (which wraps mpirun) to launch a small probe
# script on each MPI rank.  The probe dumps its environment as JSON to stdout,
# and the test parses the output to verify env propagation behaviour.
#
# Requirements:
#   - mpirun (or mpirun-ulfm) on PATH
#   - tt-run on PATH
#   - /etc/mpirun/hostfile present (dual-T3K CI topology)
#   - TT_METAL_HOME set to the repo root
#
# Run only tests in this section:
#   pytest tests/ttnn/distributed/test_ttrun_env_passthrough.py -m multihost -v
# ===========================================================================

# Sentinel printed by the probe script so we can extract rank env JSON
# from the (possibly noisy) mpirun tagged output.
_ENV_PROBE_SENTINEL = "TT_ENV_PROBE"

# Path to the dual-T3K mesh graph descriptor (relative to TT_METAL_HOME)
_DUAL_T3K_MESH_GRAPH_DESC = "tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.textproto"

# Default timeout for tt-run subprocess calls (seconds)
_TTRUN_TIMEOUT = 120


def _tt_run_available() -> bool:
    """Return True if tt-run is on PATH."""
    return shutil.which("tt-run") is not None


def _mpi_hostfile_exists() -> bool:
    """Return True if the standard multihost hostfile is present."""
    return Path("/etc/mpirun/hostfile").is_file()


def _write_probe_script(path: Path) -> None:
    """Write a small Python script that dumps its environment as JSON.

    Output format per rank (one line):
        TT_ENV_PROBE:<rank>:<json-encoded env dict>

    The sentinel prefix lets us reliably parse rank env data from mpirun's
    tagged output which may include other log lines.
    """
    path.write_text(
        textwrap.dedent(
            """\
            import json, os, sys
            rank = os.environ.get("OMPI_COMM_WORLD_RANK", "?")
            payload = json.dumps(dict(os.environ), separators=(",", ":"))
            # Flush explicitly — mpirun buffers stdout across ranks
            sys.stdout.write(f"TT_ENV_PROBE:{rank}:{payload}\\n")
            sys.stdout.flush()
        """
        )
    )


def _write_rank_binding_yaml(
    path: Path,
    *,
    num_ranks: int = 2,
    global_env: dict | None = None,
    rank_env_overrides: dict[int, dict[str, str]] | None = None,
) -> None:
    """Write a rank binding YAML suitable for dual-T3K topology.

    Each rank gets mesh_id = rank index (matching the production dual_t3k config).
    """
    lines = ["rank_bindings:"]
    for r in range(num_ranks):
        lines.append(f"  - rank: {r}")
        lines.append(f"    mesh_id: {r}")
        overrides = (rank_env_overrides or {}).get(r)
        if overrides:
            lines.append("    env_overrides:")
            for k, v in overrides.items():
                lines.append(f'      {k}: "{v}"')
    if global_env:
        lines.append("global_env:")
        for k, v in global_env.items():
            lines.append(f'  {k}: "{v}"')
    lines.append(f'mesh_graph_desc_path: "{_DUAL_T3K_MESH_GRAPH_DESC}"')
    path.write_text("\n".join(lines) + "\n")


def _run_tt_run(
    rank_binding_path: Path,
    probe_script_path: Path,
    extra_env: dict[str, str] | None = None,
    timeout: int = _TTRUN_TIMEOUT,
) -> dict[int, dict[str, str]]:
    """Invoke tt-run with the probe script and return per-rank env dicts.

    Returns:
        {rank_id: {env_key: env_value, ...}, ...}

    Raises:
        subprocess.TimeoutExpired: if tt-run does not exit within *timeout* seconds.
        RuntimeError: if we cannot parse rank env output for any expected rank.
    """
    mpi_args = "--hostfile /etc/mpirun/hostfile"
    cmd = [
        "tt-run",
        "--rank-binding",
        str(rank_binding_path),
        "--mpi-args",
        mpi_args,
        "python3",
        str(probe_script_path),
    ]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )

    # Parse rank environments from stdout
    rank_envs: dict[int, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        # mpirun --tag-output prefixes lines like: [1,0]<stdout>:TT_ENV_PROBE:0:{...}
        # Strip the tag prefix if present
        if _ENV_PROBE_SENTINEL not in line:
            continue
        idx = line.index(_ENV_PROBE_SENTINEL)
        payload = line[idx:]  # "TT_ENV_PROBE:<rank>:<json>"
        parts = payload.split(":", 2)
        if len(parts) < 3:
            continue
        rank_id = int(parts[1])
        rank_envs[rank_id] = json.loads(parts[2])

    if not rank_envs:
        raise RuntimeError(
            f"No rank env output parsed from tt-run.\n"
            f"  stdout: {result.stdout[:2000]}\n"
            f"  stderr: {result.stderr[:2000]}\n"
            f"  returncode: {result.returncode}"
        )

    return rank_envs


# Skip the entire multihost class if prerequisites are missing
_skip_reason = None
if not _tt_run_available():
    _skip_reason = "tt-run not found on PATH"
elif not _mpi_hostfile_exists():
    _skip_reason = "/etc/mpirun/hostfile not present (not a multihost CI environment)"
elif not os.environ.get("TT_METAL_HOME"):
    _skip_reason = "TT_METAL_HOME not set"


@pytest.mark.multihost
@pytest.mark.skipif(_skip_reason is not None, reason=_skip_reason or "")
class TestMultihostEnvPassthrough:
    """Integration tests that invoke tt-run with real MPI to verify env propagation."""

    @pytest.fixture(autouse=True)
    def _setup(self, request):
        """Create probe + binding files on a filesystem visible to all MPI hosts.

        Pytest's ``tmp_path`` lives under ``/tmp/pytest-of-*`` on the launcher only.
        Remote ranks run ``python3 <probe>`` with the same path and would get ENOENT.
        Use ``$TT_METAL_HOME/generated/...`` (shared checkout in CI) instead.
        """
        metal_home = os.environ["TT_METAL_HOME"]
        # Node id may contain "::", brackets, etc. — keep directory names portable.
        slug = "".join(c if c.isalnum() or c in "-_." else "_" for c in request.node.name)[:200]
        work = (
            Path(metal_home).resolve()
            / "generated"
            / "multihost_ttrun_env_probe"
            / f"{slug}_{os.getpid()}_{uuid.uuid4().hex}"
        )
        work.mkdir(parents=True, exist_ok=False)
        self.probe_script = work / "env_probe.py"
        _write_probe_script(self.probe_script)
        self.tmp_path = work
        yield
        shutil.rmtree(work, ignore_errors=True)

    def test_tt_vars_propagated_to_all_ranks(self):
        """Set a TT_ prefixed var and verify both ranks see it rank-scoped."""
        binding_path = self.tmp_path / "binding.yaml"
        _write_rank_binding_yaml(binding_path)

        rank_envs = _run_tt_run(
            binding_path,
            self.probe_script,
            extra_env={"TT_METAL_LOGS_PATH": "/tmp/test-multihost-logs"},
        )

        assert len(rank_envs) >= 2, f"Expected at least 2 ranks, got {sorted(rank_envs.keys())}"

        for rank_id, env in rank_envs.items():
            assert "TT_METAL_LOGS_PATH" in env, f"Rank {rank_id} missing TT_METAL_LOGS_PATH"
            logs_path = env["TT_METAL_LOGS_PATH"]
            assert (
                "/tmp/test-multihost-logs/" in logs_path
            ), f"Rank {rank_id}: expected rank-scoped path under /tmp/test-multihost-logs/, got {logs_path}"
            assert f"rank_{rank_id}" in logs_path, f"Rank {rank_id}: expected rank_{rank_id} suffix in {logs_path}"

    def test_tt_visible_devices_per_rank_override(self):
        """Set different TT_VISIBLE_DEVICES per rank via env_overrides."""
        binding_path = self.tmp_path / "binding.yaml"
        _write_rank_binding_yaml(
            binding_path,
            rank_env_overrides={
                0: {"TT_VISIBLE_DEVICES": "0,1,2,3"},
                1: {"TT_VISIBLE_DEVICES": "4,5,6,7"},
            },
        )

        rank_envs = _run_tt_run(binding_path, self.probe_script)

        assert (
            rank_envs[0]["TT_VISIBLE_DEVICES"] == "0,1,2,3"
        ), f"Rank 0: expected '0,1,2,3', got {rank_envs[0].get('TT_VISIBLE_DEVICES')}"
        assert (
            rank_envs[1]["TT_VISIBLE_DEVICES"] == "4,5,6,7"
        ), f"Rank 1: expected '4,5,6,7', got {rank_envs[1].get('TT_VISIBLE_DEVICES')}"

    def test_blocklisted_vars_not_propagated(self):
        """Set GITHUB_TOKEN and RUNNER_OS; verify they are NOT on remote ranks."""
        binding_path = self.tmp_path / "binding.yaml"
        _write_rank_binding_yaml(binding_path)

        rank_envs = _run_tt_run(
            binding_path,
            self.probe_script,
            extra_env={
                "GITHUB_TOKEN": "ghp_fake_secret_12345",
                "RUNNER_OS": "Linux",
                "ACTIONS_ORCHESTRATION_ID": "12345",
            },
        )

        for rank_id, env in rank_envs.items():
            assert "GITHUB_TOKEN" not in env, f"Rank {rank_id} leaked GITHUB_TOKEN"
            assert "RUNNER_OS" not in env, f"Rank {rank_id} leaked RUNNER_OS"
            assert "ACTIONS_ORCHESTRATION_ID" not in env, f"Rank {rank_id} leaked ACTIONS_ORCHESTRATION_ID"

    def test_global_env_propagated(self):
        """Set global_env in rank binding YAML; verify all ranks see it."""
        binding_path = self.tmp_path / "binding.yaml"
        _write_rank_binding_yaml(
            binding_path,
            global_env={"TT_CUSTOM_GLOBAL_VAR": "hello_multihost"},
        )

        rank_envs = _run_tt_run(binding_path, self.probe_script)

        for rank_id, env in rank_envs.items():
            assert env.get("TT_CUSTOM_GLOBAL_VAR") == "hello_multihost", (
                f"Rank {rank_id}: expected TT_CUSTOM_GLOBAL_VAR='hello_multihost', "
                f"got {env.get('TT_CUSTOM_GLOBAL_VAR')!r}"
            )

    def test_rank_scoped_paths(self):
        """Verify TT_METAL_LOGS_PATH is uniquely rank-scoped with hostname+rank suffix."""
        binding_path = self.tmp_path / "binding.yaml"
        _write_rank_binding_yaml(binding_path)

        rank_envs = _run_tt_run(
            binding_path,
            self.probe_script,
            extra_env={"TT_METAL_LOGS_PATH": "/tmp/test-scoped-logs"},
        )

        seen_paths: set[str] = set()
        for rank_id, env in rank_envs.items():
            logs_path = env.get("TT_METAL_LOGS_PATH", "")
            assert logs_path not in seen_paths, f"Rank {rank_id}: duplicate logs path {logs_path}"
            seen_paths.add(logs_path)
            assert f"rank_{rank_id}" in logs_path, f"Rank {rank_id}: expected rank_{rank_id} in {logs_path}"
            assert logs_path.startswith(
                "/tmp/test-scoped-logs/"
            ), f"Rank {rank_id}: expected path under /tmp/test-scoped-logs/, got {logs_path}"

    def test_mesh_id_set_per_rank(self):
        """Verify TT_MESH_ID matches the rank binding config (rank N -> mesh_id N)."""
        binding_path = self.tmp_path / "binding.yaml"
        _write_rank_binding_yaml(binding_path)

        rank_envs = _run_tt_run(binding_path, self.probe_script)

        for rank_id, env in rank_envs.items():
            assert env.get("TT_MESH_ID") == str(
                rank_id
            ), f"Rank {rank_id}: expected TT_MESH_ID='{rank_id}', got {env.get('TT_MESH_ID')!r}"

    def test_tt_metal_home_propagated(self):
        """Verify TT_METAL_HOME is set on all ranks and matches the launcher value."""
        binding_path = self.tmp_path / "binding.yaml"
        _write_rank_binding_yaml(binding_path)

        rank_envs = _run_tt_run(binding_path, self.probe_script)

        expected_home = os.environ.get("TT_METAL_HOME", "")
        for rank_id, env in rank_envs.items():
            assert "TT_METAL_HOME" in env, f"Rank {rank_id} missing TT_METAL_HOME"
            assert env["TT_METAL_HOME"] == expected_home, (
                f"Rank {rank_id}: expected TT_METAL_HOME='{expected_home}', " f"got {env['TT_METAL_HOME']!r}"
            )

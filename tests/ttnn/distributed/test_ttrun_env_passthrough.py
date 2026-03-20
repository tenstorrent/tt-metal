#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit and integration tests for tt-run environment management.

Covers env propagation, blocklisting, path safety validation, rank-scoped path
scoping, MPI export classification, and end-to-end rank environment assembly.
No MPI runtime or hardware is required.
"""

import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import ttnn.distributed.ttrun as ttrun
from ttnn.distributed.ttrun import (
    ENV_BLOCKLIST,
    ENV_BLOCKLIST_PREFIXES,
    ENV_LAUNCHER_ONLY_BLOCKLIST,
    FORCE_NAME_ONLY_MPI_EXPORT_VARS,
    RANK_SCOPED_PATH_ENV_VARS,
    RankBinding,
    TTRunConfig,
    apply_rank_scoped_paths,
    build_rank_environment_args,
    classify_mpi_env_exports,
    get_launcher_environment,
    get_rank_environment,
    has_auto_passthrough_prefix,
    has_blocked_prefix,
    is_auto_passthrough_env_var,
    is_blocklisted_env_var,
    main,
    should_use_name_only_mpi_export,
    strip_blocklisted_env_vars,
    validate_path_safety,
)


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


def test_tt_visible_devices_host_value_is_not_passed_without_rank_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "31")
    monkeypatch.setenv("TTNN_CONFIG_OVERRIDES", '{"foo": "bar"}')

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert "TT_VISIBLE_DEVICES" not in env
    assert env["TTNN_CONFIG_OVERRIDES"] == '{"foo": "bar"}'


def test_tt_visible_devices_rank_override_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "31")

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0, env_overrides={"TT_VISIBLE_DEVICES": "0,1"})
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["TT_VISIBLE_DEVICES"] == "0,1"


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


def test_rank_environment_provides_default_cache_when_unset_no_home(monkeypatch, tmp_path):
    """When both TT_METAL_CACHE and HOME are unset, fall back to /tmp/tt-metal-cache."""
    monkeypatch.delenv("TT_METAL_CACHE", raising=False)
    monkeypatch.delenv("HOME", raising=False)

    binding = RankBinding(rank=2, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)
    assert env["TT_METAL_CACHE"] == "/tmp/tt-metal-cache"


def test_rank_environment_keeps_cache_shared_but_scopes_logs_path_per_rank(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_METAL_CACHE", "/nfs/shared/tt-metal-cache")
    monkeypatch.setenv("TT_METAL_LOGS_PATH", "/nfs/shared/tt-metal-logs")

    binding = RankBinding(rank=3, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)
    rank_suffix = f"{socket.gethostname()}_rank_3"

    assert env["TT_METAL_CACHE"] == "/nfs/shared/tt-metal-cache"
    assert env["TT_METAL_LOGS_PATH"] == f"/nfs/shared/tt-metal-logs/{rank_suffix}"


def test_rank_environment_does_not_double_append_rank_scoped_suffix(monkeypatch, tmp_path):
    rank_suffix = f"{socket.gethostname()}_rank_2"
    monkeypatch.setenv("TT_METAL_CACHE", f"/nfs/shared/tt-metal-cache/{rank_suffix}")
    monkeypatch.setenv("TT_METAL_LOGS_PATH", f"/nfs/shared/tt-metal-logs/{rank_suffix}")

    binding = RankBinding(rank=2, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["TT_METAL_CACHE"] == f"/nfs/shared/tt-metal-cache/{rank_suffix}"
    assert env["TT_METAL_LOGS_PATH"] == f"/nfs/shared/tt-metal-logs/{rank_suffix}"


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


def test_rank_environment_scopes_explicit_jit_scratch(monkeypatch, tmp_path):
    """When TT_METAL_JIT_SCRATCH is set in the parent, ttrun still rank-scopes it."""
    monkeypatch.setenv("TT_METAL_JIT_SCRATCH", "/fast-local/jit")

    binding = RankBinding(rank=3, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)
    rank_suffix = f"{socket.gethostname()}_rank_3"

    assert env["TT_METAL_JIT_SCRATCH"] == f"/fast-local/jit/{rank_suffix}"


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


def test_arch_prefix_passthrough(monkeypatch, tmp_path):
    """Test that ARCH_ prefixed variables are auto-propagated."""
    monkeypatch.setenv("ARCH_NAME", "wormhole_b0")

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["ARCH_NAME"] == "wormhole_b0"


def test_wh_prefix_passthrough(monkeypatch, tmp_path):
    """Test that WH_ prefixed variables (Wormhole-specific) are auto-propagated."""
    monkeypatch.setenv("WH_ARCH_YAML", "/path/to/wh_arch.yaml")

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["WH_ARCH_YAML"] == "/path/to/wh_arch.yaml"


def test_deepseek_prefix_passthrough(monkeypatch, tmp_path):
    """Test that DEEPSEEK_ prefixed variables are auto-propagated."""
    monkeypatch.setenv("DEEPSEEK_V3_HF_MODEL", "deepseek-ai/deepseek-v3")

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["DEEPSEEK_V3_HF_MODEL"] == "deepseek-ai/deepseek-v3"


def test_mesh_prefix_passthrough(monkeypatch, tmp_path):
    """Test that MESH_ prefixed variables are auto-propagated."""
    monkeypatch.setenv("MESH_DEVICE", "T3000")

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["MESH_DEVICE"] == "T3000"


def test_loguru_prefix_passthrough(monkeypatch, tmp_path):
    """Test that LOGURU_ prefixed variables are auto-propagated."""
    monkeypatch.setenv("LOGURU_LEVEL", "DEBUG")

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["LOGURU_LEVEL"] == "DEBUG"


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


def test_pythonhome_not_set_by_default(monkeypatch, tmp_path):
    """Test that PYTHONHOME is not set unless explicitly provided."""
    monkeypatch.delenv("PYTHONHOME", raising=False)

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert "PYTHONHOME" not in env


def test_pythonhome_preserved_when_explicitly_set(monkeypatch, tmp_path):
    """Test that PYTHONHOME is preserved when explicitly set in parent environment."""
    monkeypatch.setenv("PYTHONHOME", "/custom/python/home")

    binding = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)

    assert env["PYTHONHOME"] == "/custom/python/home"


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

#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import socket
from types import SimpleNamespace

from click.testing import CliRunner
from ttnn.distributed.ttrun import (
    RankBinding,
    TTRunConfig,
    build_rank_environment_args,
    get_launcher_environment,
    get_rank_environment,
    main,
)
import ttnn.distributed.ttrun as ttrun


def _build_config(tmp_path: Path, binding: RankBinding, global_env: dict | None = None) -> TTRunConfig:
    mesh_graph_desc = tmp_path / "mesh_graph_desc.textproto"
    mesh_graph_desc.write_text("mesh_graph_desc: test\n")
    # TTRunConfig requires contiguous ranks from 0; build bindings 0..binding.rank
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
    """When both TT_METAL_CACHE and HOME are unset, fall back to /tmp."""
    monkeypatch.delenv("TT_METAL_CACHE", raising=False)
    monkeypatch.delenv("HOME", raising=False)

    binding = RankBinding(rank=2, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)
    assert env["TT_METAL_CACHE"] == "/tmp"


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
    # Set all launcher-only blocklist variables
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

    # Build environment for rank 0
    binding0 = RankBinding(rank=0, mesh_id=0, mesh_host_rank=0)
    config0 = _build_config(tmp_path, binding0)
    env0 = get_rank_environment(binding0, config0)

    # Build environment for rank 1
    binding1 = RankBinding(rank=1, mesh_id=0, mesh_host_rank=1)
    config1 = _build_config(tmp_path, binding1)
    env1 = get_rank_environment(binding1, config1)

    assert env0["TT_METAL_CACHE"] == "/shared/cache"
    assert env1["TT_METAL_CACHE"] == "/shared/cache"
    assert env0["TT_METAL_LOGS_PATH"] == f"/shared/logs/{hostname}_rank_0"
    assert env1["TT_METAL_LOGS_PATH"] == f"/shared/logs/{hostname}_rank_1"

#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import socket

from ttnn.distributed.ttrun import (
    RankBinding,
    TTRunConfig,
    build_rank_environment_args,
    get_launcher_environment,
    get_rank_environment,
)


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
    """When TT_METAL_CACHE is not in the parent environment, ttrun must provide a
    default so that apply_rank_scoped_paths can isolate it per rank.  Without this,
    all ranks fall back to the shared C++ default and race on NFS."""
    monkeypatch.delenv("TT_METAL_CACHE", raising=False)
    monkeypatch.setenv("HOME", "/home/testuser")

    binding = RankBinding(rank=1, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)
    rank_suffix = f"{socket.gethostname()}_rank_1"

    assert "TT_METAL_CACHE" in env
    assert env["TT_METAL_CACHE"] == f"/home/testuser/.cache/{rank_suffix}"


def test_rank_environment_provides_default_cache_when_unset_no_home(monkeypatch, tmp_path):
    """When both TT_METAL_CACHE and HOME are unset, fall back to /tmp."""
    monkeypatch.delenv("TT_METAL_CACHE", raising=False)
    monkeypatch.delenv("HOME", raising=False)

    binding = RankBinding(rank=2, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)
    rank_suffix = f"{socket.gethostname()}_rank_2"

    assert env["TT_METAL_CACHE"] == f"/tmp/{rank_suffix}"


def test_rank_environment_scopes_cache_and_logs_path_per_rank(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_METAL_CACHE", "/nfs/shared/tt-metal-cache")
    monkeypatch.setenv("TT_METAL_LOGS_PATH", "/nfs/shared/tt-metal-logs")

    binding = RankBinding(rank=3, mesh_id=0, mesh_host_rank=0)
    config = _build_config(tmp_path, binding)

    env = get_rank_environment(binding, config)
    rank_suffix = f"{socket.gethostname()}_rank_3"

    assert env["TT_METAL_CACHE"] == f"/nfs/shared/tt-metal-cache/{rank_suffix}"
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

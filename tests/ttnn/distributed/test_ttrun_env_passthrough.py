#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

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
    return TTRunConfig(
        rank_bindings=[binding],
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

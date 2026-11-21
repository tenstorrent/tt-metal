# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks that the DeepSeek V3 B1 CLI flags reach the pipeline config."""

import sys
import types

from models.demos.deepseek_v3_b1.demo import cli as ds_cli


def _run_main(monkeypatch, argv):
    captured = {}

    def fake_run_demo(**kwargs):
        captured.update(kwargs)

    ttnn_stub = types.ModuleType("ttnn")
    ttnn_stub.init_distributed_context = lambda: None
    monkeypatch.setitem(sys.modules, "ttnn", ttnn_stub)
    monkeypatch.setattr(ds_cli, "run_demo", fake_run_demo)
    monkeypatch.setattr(ds_cli, "configure_runtime_env", lambda **kwargs: None)

    assert ds_cli.main(argv) == 0
    return captured


BASE_ARGV = ["--weights", "synthetic", "--prompt", "hi", "--max-new-tokens", "4"]


def test_speculative_decode_enabled_by_default(monkeypatch):
    captured = _run_main(monkeypatch, BASE_ARGV)
    assert captured["num_mtp_levels"] == 1


def test_disable_speculative_decode_selects_base_decode(monkeypatch):
    captured = _run_main(monkeypatch, BASE_ARGV + ["--no-enable-speculative-decode"])
    assert captured["num_mtp_levels"] == 0, (
        "--no-enable-speculative-decode must select base decode, got num_mtp_levels=" f"{captured['num_mtp_levels']}"
    )


def test_num_mtp_levels_still_honoured_when_enabled(monkeypatch):
    captured = _run_main(monkeypatch, BASE_ARGV + ["--num-mtp-levels", "3"])
    assert captured["num_mtp_levels"] == 3


def test_model_pipeline_rejects_speculative_flag_that_is_ignored(monkeypatch):
    captured = _run_main(monkeypatch, BASE_ARGV + ["--num-mtp-levels", "2", "--no-enable-speculative-decode"])
    assert captured["num_mtp_levels"] == 0


def test_persistent_mode_flags_reach_pipeline(monkeypatch):
    captured = _run_main(monkeypatch, BASE_ARGV + ["--no-persistent-mode", "--no-fp32"])
    assert captured["lm_head_persistent_mode"] is False
    assert captured["lm_head_fp32_dest_acc_en"] is False

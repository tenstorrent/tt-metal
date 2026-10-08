# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL-4B e2e on WH/BH/Quasar: per-stage PCC against a truncated HF reference."""
import os
from pathlib import Path

import pytest
import torch
import ttnn

from models.experimental.ops.quasar.qwen3_vl.tests.e2e import pcc as P
from models.experimental.ops.quasar.qwen3_vl.tests.e2e import snapshot as S
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.config import HF_MODEL_ID
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.host_reference import load_hf_model, run_reference
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.op_overrides import OverrideSession
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.presets import PRESETS, build_inputs
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.progress import ProgressLog
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.recorder import StageRecorder
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.tt_runner import run_tt


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_qwen3_vl_e2e(mesh_device, qwen_run_config, monkeypatch, request):
    cfg = qwen_run_config
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    grid = mesh_device.compute_with_storage_grid_size()
    if cfg.expect_grid is not None:
        assert (grid.x, grid.y) == cfg.expect_grid, f"device grid {grid.x}x{grid.y} != expected {cfg.expect_grid}"
    # Quasar has no bfp8: without the Quasar config the text weights default to bfloat8_b (and the cache key says native).
    assert cfg.quasar_config or mesh_device.arch() != ttnn.device.Arch.QUASAR, "pass --qwen-quasar-config on Quasar"

    monkeypatch.setenv("HF_MODEL", os.environ.get("HF_MODEL", HF_MODEL_ID))
    cache_root = Path(os.environ.get("TT_CACHE_PATH", Path.home() / ".cache" / "tt_qwen3_vl_quasar"))
    monkeypatch.setenv("TT_CACHE_PATH", str(cache_root / cfg.cache_key((grid.x, grid.y))))

    from transformers import AutoProcessor

    preset = PRESETS[cfg.size]
    inputs = build_inputs(preset, AutoProcessor.from_pretrained(HF_MODEL_ID))
    hf_model = load_hf_model(cfg.vision_layers, cfg.text_layers, cfg.deepstack_at)
    goldens = run_reference(hf_model, inputs, cfg.decode_steps)

    notes = []
    progress = ProgressLog(cfg.run_dir / "progress.log")
    if not progress.hooks_active():
        notes.append("progress.log unavailable: ttnn fast runtime mode is on (set TTNN_CONFIG_OVERRIDES).")
    recorder = StageRecorder(progress)
    session = OverrideSession(mesh_device, cfg.host_ops, cfg.disable_wa, cfg.allow_uncertified)
    session.install(monkeypatch)
    resume_from = request.config.getoption("--qwen-resume-prefill")
    resume = None
    if resume_from:
        resume = S.load(resume_from, S.meta_for(cfg, (grid.x, grid.y), goldens.teacher_tokens))
        notes.append(f"Vision and prefill skipped: decode resumed from the prefill snapshot in {resume_from}.")
    with progress.installed():
        run_tt(
            cfg,
            preset,
            inputs,
            hf_model,
            goldens,
            mesh_device,
            recorder,
            monkeypatch,
            snapshot_out=None if resume else cfg.run_dir / S.SNAPSHOT_NAME,
            resume=resume,
            clear_program_cache_before_decode=request.config.getoption("--qwen-clear-program-cache-before-decode"),
        )
    host_ops, hits = session.host_ops_active, dict(session.hits)

    if request.config.getoption("--qwen-dump-stages"):
        torch.save({"golden": goldens.tensors, "tt": recorder.tensors}, cfg.run_dir / "stages.pt")
    results = P.compare(goldens.tensors, recorder.tensors, P.thresholds_for(cfg.size), recorder.seconds)
    v = P.verdict(results, host_ops, hits, notes)
    (cfg.run_dir / "pcc.md").write_text(v.markdown)
    (cfg.run_dir / "verdict.txt").write_text(v.status + "\n")
    print(v.markdown)
    assert v.status == "PASS", f"{v.status}: first failing stage {v.first_failure}; see {cfg.run_dir}/pcc.md"

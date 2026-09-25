# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline contract tests (not the correctness gate, which is tests/e2e/test_e2e_image_edit.py):
per-stage trace capture at full depth, and the depth knob."""
from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.qwen_image_edit.tt import pipeline as P
from models.demos.qwen_image_edit.tt.inputs import EditConfig

MESH_PARAMS = pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 24576,
            "trace_region_size": P.DEVICE_PARAMS["trace_region_size"],
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
MESH = pytest.mark.parametrize("mesh_device", [P.MESH_SHAPE], indirect=True)
BATCH = 4  # the e2e gate's batch


@pytest.fixture(scope="module")
def hf_pipe():
    return P.load_hf_reference(torch.float32)


@pytest.mark.timeout(2 * 3600)
@MESH_PARAMS
@MESH
def test_trace_capture(mesh_device, hf_pipe):
    """Per stage: capture one step, replay it, compare against eager, release (full depth, gate batch)."""
    pipe = P.build_pipeline(mesh_device, model=hf_pipe, cfg=EditConfig(batch=BATCH))
    ok = pipe.trace_capture_selftest()
    print(f"[trace] report {pipe.trace_report}", flush=True)
    assert ok, pipe.trace_report


@pytest.mark.timeout(3600)
@MESH_PARAMS
@MESH
def test_depth_knob(mesh_device, hf_pipe):
    """layers caps EVERY repeated stack (vision blocks, LM layers, transformer blocks); per-stack
    overrides win; the rest of each model stays built so the capped build still runs end to end."""
    pipe = P.build_pipeline(mesh_device, model=hf_pipe, layers=2, denoise_layers=3)
    te = hf_pipe.text_encoder
    assert len(pipe.stacks["vision_encode"]) == 2 < len(te.model.visual.blocks)
    assert pipe.text_encoder.text_model.num_layers == 2 < len(te.model.language_model.layers)
    assert len(pipe.stacks["denoise"]) == 3 < len(hf_pipe.transformer.transformer_blocks)
    cfg = EditConfig(batch=BATCH, num_inference_steps=2)
    p = pipe.prepare(pipe.encode(cfg))
    out = pipe.run_image_edit(p)
    assert tuple(out.shape) == (BATCH, 3, 256, 256)
    counts = pipe.tracker.snapshot()
    assert counts["qwen_image_transformer_block"] == 3 * 2 * 2  # blocks x (cond + uncond) x steps
    assert counts["v_l_vision_block"] == 2

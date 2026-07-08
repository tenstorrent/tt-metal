# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Perf test for LongCat-Image Call 2 (image-edit) — trace-replay per-DENOISE-STEP device latency.

Same harness as tests/e2e/test_text_to_image_perf.py, but the decode contract uses the EDIT
geometry: the fixed image latents are concatenated onto the noise latents (so the DiT sequence is
~2x the text->image step) and the noise-latent output is sliced back. `_build_for_perf` sets
`ttp._perf_edit = True`, so `decode_prefill` builds the edit-geometry inputs; `measure_adapter`
captures ONE host-op-free edit step as a device trace, replays it, and prints
`TRACE_PER_TOKEN_MS=<ms>` — comparable to the text->image number to show the edit step's extra cost.
LONGCAT_PERF_CQ=2 drives the trace+2CQ replay. Correctness is covered by test_image_edit_e2e.py.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.vision.generative.longcat_image.tt import pipeline as P
from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter
from models.experimental.perf_automation.agent.trace_replay import measure_adapter

HF_MODEL_ID = "meituan-longcat/LongCat-Image"

_PERF_CQ = int(os.environ.get("LONGCAT_PERF_CQ", "1"))
_OPEN_KWARGS = {
    "l1_small_size": 24576,
    "trace_region_size": int(os.environ.get("TT_PERF_TRACE_REGION", "209715200")),  # 200 MB
}
if _PERF_CQ >= 2:
    _OPEN_KWARGS["num_command_queues"] = 2


def _build_for_perf(dev):
    """device -> built TT pipeline flagged for the EDIT decode geometry."""
    from diffusers import LongCatImagePipeline

    pipe = LongCatImagePipeline.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    ttp = P.LongCatImagePipelineTT(dev, pipe)
    ttp._perf_edit = True  # decode_prefill -> edit geometry (image latents concatenated)
    return ttp


def test_image_edit_perf():
    device = ttnn.open_device(device_id=0, **_OPEN_KWARGS)
    try:
        adapter = PipelineDecodeAdapter(_build_for_perf, prompt_ids=None, batch=1)
        mode = "2cq" if _PERF_CQ >= 2 else "1cq"
        per_step_ms = measure_adapter(adapter, device, mode=mode)  # prints TRACE_PER_TOKEN_MS=<ms>
        print(f"[perf] edit denoise per-step device latency: {per_step_ms:.4f} ms", flush=True)
        assert per_step_ms > 0.0
    finally:
        ttnn.close_device(device)

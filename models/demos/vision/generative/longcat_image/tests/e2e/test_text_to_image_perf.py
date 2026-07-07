# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Perf test for LongCat-Image — trace-replay per-DENOISE-STEP device latency.

The DiT denoise step is the dominant, repeated unit of this diffusion model (it runs
`num_inference_steps` times per image), so it is what `optimize` should profile and tune.
This uses the model-agnostic perf harness: `PipelineDecodeAdapter` wraps the pipeline's
decode contract (`decode_prefill`/`decode_step` on `LongCatImagePipelineTT`, which delegate to
the already-traced denoise stage), and `measure_adapter` captures ONE host-op-free denoise step
as a device trace, replays it, and prints `TRACE_PER_TOKEN_MS=<ms>` — the metric the optimize
harness parses. Perf only: correctness is covered by tests/e2e/test_text_to_image_e2e.py.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.vision.generative.longcat_image.tt import pipeline as P
from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter
from models.experimental.perf_automation.agent.trace_replay import measure_adapter

HF_MODEL_ID = "meituan-longcat/LongCat-Image"

# Trace replay needs a trace region; l1_small matches the demo/e2e device open.
_OPEN_KWARGS = {
    "l1_small_size": 24576,
    "trace_region_size": int(os.environ.get("TT_PERF_TRACE_REGION", "209715200")),  # 200 MB
}


def _build_for_perf(dev):
    """device -> built TT pipeline, EXACTLY as the demo/e2e build it (so the trace captures the real program)."""
    from diffusers import LongCatImagePipeline

    pipe = LongCatImagePipeline.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return P.LongCatImagePipelineTT(dev, pipe)


def test_text_to_image_perf():
    device = ttnn.open_device(device_id=0, **_OPEN_KWARGS)
    try:
        adapter = PipelineDecodeAdapter(_build_for_perf, prompt_ids=None, batch=1)
        # single-CQ trace replay: the device is opened with one command queue here, so force 1cq
        # (auto would attempt the 2CQ overlap path and hit a caught-but-noisy cq_id-1 fallback).
        per_step_ms = measure_adapter(adapter, device, mode="1cq")  # prints TRACE_PER_TOKEN_MS=<ms>
        print(f"[perf] denoise per-step device latency: {per_step_ms:.4f} ms", flush=True)
        assert per_step_ms > 0.0
    finally:
        ttnn.close_device(device)

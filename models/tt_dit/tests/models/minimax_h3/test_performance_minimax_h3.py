# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""t2va pipeline wall-clock at the published working points, via `BenchmarkProfiler`.

Six aspect ratios (21:9 .. 9:16) x three durations (5 / 10 / 15 s), 50 steps. The canvas comes from
`resolve_canvas_size` and the frame count from `align_num_frames`. Stage durations log on the host
rank only, and only for the warm generation. Quality gates live in `test_pipeline_minimax_h3.py`.
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames, resolve_canvas_size
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from .common import GALAXY_MESHES
from .common_av import CALIBRATED_FOX_PROMPT, is_host, log_pipeline_perf, run_warm_generation, weights_dir

NUM_INFERENCE_STEPS = 50
SEED = 0

# The published working points: short edge 768 from 16:9 through 9:16, and ~1 MPix for anything
# wider. `resolve_canvas_size` already implements exactly that (768 short edge, area capped at
# 768*1344, both axes snapped to 32), so the canvas is derived rather than tabulated here --
# 21:9 lands on 672x1536, which is the documented example.
ASPECT_RATIOS = [(21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16)]
DURATIONS_S = [5, 10, 15]

PROMPT = CALIBRATED_FOX_PROMPT

SWEEP = [
    pytest.param(ratio, seconds, id=f"{ratio[0]}x{ratio[1]}_{seconds}s")
    for seconds in DURATIONS_S
    for ratio in ASPECT_RATIOS
]


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("aspect_ratio", "duration_s"), SWEEP)
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
def test_t2va_performance(mesh_device, reset_seeds, aspect_ratio, duration_s):
    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    prompt = PROMPT

    HEIGHT, WIDTH = resolve_canvas_size(*aspect_ratio)
    NUM_FRAMES = align_num_frames(round(duration_s * MINIMAX_H3_FPS))
    if is_host():
        logger.info(f"working point: {aspect_ratio[0]}:{aspect_ratio[1]} -> {WIDTH}x{HEIGHT}, {NUM_FRAMES} frames")

    if is_host() and not os.environ.get("TT_DIT_CACHE_DIR"):
        logger.warning(
            "TT_DIT_CACHE_DIR is unset, so every weight load reads safetensors. Prepares are excluded "
            "from the total either way, but the run will take far longer than the reported compute."
        )

    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device, weights_dir=weights, precomputed_adaln=False, dit_fsdp=False
    )

    benchmark_profiler = BenchmarkProfiler()
    output = run_warm_generation(
        pipeline,
        prompt,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
        profiler=benchmark_profiler,
    )

    expected_frames = align_num_frames(NUM_FRAMES)
    assert output.num_frames == expected_frames, f"generated {output.num_frames} frames, expected {expected_frames}"
    assert torch.isfinite(output.video).all() and torch.isfinite(output.audio).all()

    num_forwards = NUM_INFERENCE_STEPS - 1
    log_pipeline_perf(
        benchmark_profiler,
        label="t2va",
        pipeline=pipeline,
        num_forwards=num_forwards,
        width=WIDTH,
        height=HEIGHT,
        num_frames=expected_frames,
        fps=MINIMAX_H3_FPS,
        aspect_ratio=aspect_ratio,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )

    if ttnn.using_distributed_env():
        ttnn.distributed_context_barrier()
    pipeline.release_traces()

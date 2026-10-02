# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Base MiniMax-H3 transformer, no adapter, run at a 4-forward schedule to isolate the DiT's own
per-forward compute on a single galaxy (output quality is meaningless at 4 steps -- this is a timing
probe for the on-device adaln path without any LoRA)."""

from __future__ import annotations

import os

import pytest
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkProfiler

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from .common import GALAXY_MESHES
from .common_av import (
    CALIBRATED_FOX_PROMPT,
    check_audio_sanity,
    log_pipeline_perf,
    run_warm_generation,
    weights_dir,
)

HEIGHT, WIDTH = 768, 1344
VIDEO_SHIFT, AUDIO_SHIFT = 6.0, 3.0
SEED = 0
DURATION_S = int(os.environ.get("MINIMAX_H3_BASE_DURATION_S", "5"))
NUM_FORWARDS = 4
NUM_INFERENCE_STEPS = NUM_FORWARDS + 1


@pytest.mark.timeout(5400)
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
def test_base_nolora_timing(mesh_device, reset_seeds):
    num_frames = align_num_frames(round(DURATION_S * MINIMAX_H3_FPS))

    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights_dir("transformer", "text_encoder", "vae", "audio_vae"),
        dit_fsdp=False,
        vae_output_type="yuv420",
        video_shift=VIDEO_SHIFT,
        audio_shift=AUDIO_SHIFT,
    )
    assert pipeline._lora_handle is None, "no adapter should be bound in the base-model probe"

    benchmark_profiler = BenchmarkProfiler()
    output = run_warm_generation(
        pipeline,
        CALIBRATED_FOX_PROMPT,
        num_frames=num_frames,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
        profiler=benchmark_profiler,
    )

    log_pipeline_perf(
        benchmark_profiler,
        label="base-nolora",
        pipeline=pipeline,
        num_forwards=NUM_FORWARDS,
        width=WIDTH,
        height=HEIGHT,
        num_frames=num_frames,
        fps=MINIMAX_H3_FPS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        extra_lines=("no adapter", f"video shift {VIDEO_SHIFT}, audio shift {AUDIO_SHIFT}"),
    )
    check_audio_sanity(output.audio, sampling_rate=output.sampling_rate, expected_seconds=num_frames / MINIMAX_H3_FPS)
    logger.info(f"base-nolora {DURATION_S}s done: {NUM_FORWARDS} forwards")

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end generation under FastVideo's FastH3 4-step student with VSA on.

Env: MINIMAX_H3_FAST_LORA_PATH (required; the vsa-datafree adapter, the only one carrying VSA gates),
MINIMAX_H3_FAST_KEYFRAME (fl2va), MINIMAX_H3_FAST_TASK (fl2va|t2va). Quality is recorded, not gated.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

from ....models.transformers.minimax_h3.vsa_prepass import MiniMaxH3VSAConfig
from ....pipelines.events import profiler_event_callback
from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS
from ....pipelines.minimax_h3.pipeline_minimax_h3 import AUDIO_SHIFT, VIDEO_SHIFT
from ....pipelines.minimax_h3.pipeline_minimax_h3_fast import (
    FASTH3_BASE_LADDER,
    FASTH3_NUM_GRID_POINTS,
    MiniMaxH3FastPipeline,
)
from ....pipelines.minimax_h3.policy import align_num_frames
from ....pipelines.minimax_h3.scheduler import shift_sigmas
from ....utils.video import Audio, export_video_audio_yuv
from .common import GALAXY_MESHES
from .common_av import CALIBRATED_FOX_PROMPT, artifact_dir, check_audio_sanity, is_host, log_pipeline_perf, weights_dir

HEIGHT, WIDTH = 768, 1344
DURATIONS_S = (5, 10, 15)
SEED = 0
PROMPT = CALIBRATED_FOX_PROMPT
SPARSITY = 0.9
TASK = os.environ.get("MINIMAX_H3_FAST_TASK", "fl2va")

# 362 LoRA pairs, 82 `.diff`/`.diff_b` and one VSA gate per transformer block.
FASTH3_LORA_TARGETS = 362
FASTH3_DENSE_EDITS = 82
FASTH3_GATES = 50


# pytest.ini caps a test at 300 s; a cold pass pays weight conversion and JIT well past that.
@pytest.mark.timeout(10800)
@pytest.mark.parametrize("duration_s", DURATIONS_S, ids=[f"{d}s" for d in DURATIONS_S])
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
def test_fast_end_to_end(mesh_device, reset_seeds, duration_s):
    if TASK not in ("fl2va", "t2va"):
        raise ValueError(f"MINIMAX_H3_FAST_TASK must be fl2va or t2va, got {TASK!r}")
    lora_path = os.environ.get("MINIMAX_H3_FAST_LORA_PATH")
    if not lora_path:
        pytest.skip("set MINIMAX_H3_FAST_LORA_PATH to the FastH3 vsa-datafree adapter")
    keyframe_path = os.environ.get("MINIMAX_H3_FAST_KEYFRAME")
    if TASK == "fl2va" and not keyframe_path:
        pytest.skip("set MINIMAX_H3_FAST_KEYFRAME to the conditioning image for the fl2va task")

    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    keyframe = Image.open(keyframe_path).convert("RGB") if TASK == "fl2va" else None
    num_frames = align_num_frames(round(duration_s * MINIMAX_H3_FPS))
    request = dict(image=keyframe, num_frames=num_frames, height=HEIGHT, width=WIDTH, seed=SEED)

    pipeline = MiniMaxH3FastPipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights,
        dit_fsdp=False,
        vae_output_type="yuv420",
        lora_path=lora_path,
        video_shift=VIDEO_SHIFT,
        audio_shift=AUDIO_SHIFT,
        vsa_config=MiniMaxH3VSAConfig(sparsity=SPARSITY),
    )

    # A cache hit merged nothing this run; its entry is keyed on this adapter's digest and strength.
    if pipeline.merged_targets is None:
        logger.info("FastH3 adapter: weights from cache")
    else:
        assert (pipeline.merged_targets, pipeline.merged_dense, pipeline.merged_sets) == (
            FASTH3_LORA_TARGETS,
            FASTH3_DENSE_EDITS,
            FASTH3_GATES,
        )

    video_scheduler, audio_scheduler = pipeline._build_schedulers(FASTH3_NUM_GRID_POINTS)
    ladder = torch.tensor(FASTH3_BASE_LADDER, dtype=torch.float32)
    assert torch.equal(video_scheduler.sigmas, shift_sigmas(ladder, VIDEO_SHIFT))
    assert torch.equal(audio_scheduler.sigmas, shift_sigmas(ladder, AUDIO_SHIFT))

    # Construction warms only the 5 s rung; warm this duration so compiles stay out of the measurement.
    # `run_warm_generation` is not used: its compile pass runs 3 grid points, which FastH3 refuses.
    with pipeline.quiet():
        pipeline(PROMPT, **request)

    profiler = BenchmarkProfiler()
    with profiler("run", iteration=0):
        output = pipeline(PROMPT, on_event=profiler_event_callback(profiler, 0), **request)
        ttnn.synchronize_device(pipeline.mesh_device)
    missed = {name: count for name, count in pipeline.last_program_cache_misses.items() if count}
    assert not missed, f"the measured call compiled programs after warmup, so this number is not warm: {missed}"

    log_pipeline_perf(
        profiler,
        label=f"{TASK}-fasth3-vsa",
        pipeline=pipeline,
        num_forwards=FASTH3_NUM_GRID_POINTS - 1,
        width=WIDTH,
        height=HEIGHT,
        num_frames=num_frames,
        fps=MINIMAX_H3_FPS,
        num_inference_steps=FASTH3_NUM_GRID_POINTS,
        extra_lines=(f"adapter {Path(lora_path).name}", f"VSA sparsity {SPARSITY}"),
    )

    check_audio_sanity(
        output.audio,
        sampling_rate=output.sampling_rate,
        expected_seconds=num_frames / MINIMAX_H3_FPS,
    )
    assert output.video_format == "yuv420", f"asked for yuv420 but the pipeline returned {output.video_format}"

    if is_host():
        artifacts = artifact_dir("h3_fast_artifacts")
        mp4 = artifacts / f"{TASK}_fasth3_vsa_{WIDTH}x{HEIGHT}_{duration_s}s.mp4"
        export_video_audio_yuv(
            output.video,
            str(mp4),
            fps=output.fps,
            audio=Audio(waveform=output.audio[0], sampling_rate=output.sampling_rate),
        )
        logger.info(f"wrote {mp4}")

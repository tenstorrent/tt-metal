# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end generation under a lightx2v MiniMax-H3-Turbo adapter, at the adapter's own working point.

Two properties of these adapters are silent when wrong, and both are read from the file rather than
assumed:

**Scale lives in the file's metadata, not in its tensors.** The diffusers publish carries no
per-target ``.alpha``, only ``__metadata__["alpha"]``; the real scale is ``alpha / rank`` (0.0625 for
every published Turbo adapter, corroborated by the publisher's own ComfyUI conversion, which states
``training_scale: 0.0625``). Loading at 1 applies every delta 16x too strong -- and because nothing
about that is structurally invalid, it produces a video rather than an error.
``h3_adapter_loader`` derives it, so this test only records what the loader resolved.

**The 768p variants were distilled against video shift 6**, not the checkpoint's 12 (audio stays 3);
the 544p variants keep 12/3. A wrong shift is a valid schedule over the wrong sigma grid: it
completes, and it costs quality rather than correctness. The canvas and its shifts therefore travel
together in ``WORKING_POINTS`` rather than being independently settable.

Quality is recorded, not gated: the bars elsewhere are calibrated against the 49-forward base model
and a 4- or 8-forward student has no reason to reproduce them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from loguru import logger
from PIL import Image

from models.perf.benchmarking_utils import BenchmarkProfiler

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....utils.video import Audio, export_video_audio_yuv
from .common import GALAXY_MESHES
from .common_av import (
    CALIBRATED_FOX_PROMPT,
    artifact_dir,
    check_audio_sanity,
    is_host,
    log_pipeline_perf,
    run_warm_generation,
    weights_dir,
)

# Each adapter names one canvas and one pair of shifts, and the two travel together.
WORKING_POINTS = {
    "768p": {"size": (768, 1344), "video_shift": 6.0, "audio_shift": 3.0},
    "544p": {"size": (544, 960), "video_shift": 12.0, "audio_shift": 3.0},
}
WORKING_POINT = os.environ.get("MINIMAX_H3_TURBO_POINT", "768p")
HEIGHT, WIDTH = WORKING_POINTS[WORKING_POINT]["size"]
VIDEO_SHIFT = WORKING_POINTS[WORKING_POINT]["video_shift"]
AUDIO_SHIFT = WORKING_POINTS[WORKING_POINT]["audio_shift"]

DURATIONS_S = (5, 10, 15)
SEED = 0
PROMPT = CALIBRATED_FOX_PROMPT

# NFE from the model card, plus the terminal sigma: `num_inference_steps` counts grid points.
NUM_FORWARDS = int(os.environ.get("MINIMAX_H3_TURBO_NFE", 4))
NUM_INFERENCE_STEPS = NUM_FORWARDS + 1

# One adapter file serves both; the keyframe is the only difference between the two tasks.
TASK = os.environ.get("MINIMAX_H3_TURBO_TASK", "fl2va")


@pytest.mark.parametrize("duration_s", DURATIONS_S, ids=[f"{d}s" for d in DURATIONS_S])
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
def test_turbo_end_to_end(mesh_device, reset_seeds, duration_s):
    if TASK not in ("fl2va", "t2va"):
        raise ValueError(f"MINIMAX_H3_TURBO_TASK must be fl2va or t2va, got {TASK!r}")
    lora_path = os.environ.get("MINIMAX_H3_TURBO_LORA_PATH")
    if not lora_path:
        pytest.skip("set MINIMAX_H3_TURBO_LORA_PATH to a lightx2v Minimax-h3-Turbo adapter")
    keyframe_path = os.environ.get("MINIMAX_H3_TURBO_KEYFRAME")
    if TASK == "fl2va" and not keyframe_path:
        pytest.skip("set MINIMAX_H3_TURBO_KEYFRAME to the conditioning image for the fl2va task")

    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    keyframe = Image.open(keyframe_path).convert("RGB") if TASK == "fl2va" else None
    num_frames = align_num_frames(round(duration_s * MINIMAX_H3_FPS))

    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights,
        dit_fsdp=False,
        vae_output_type="yuv420",
        lora_path=lora_path,
        video_shift=VIDEO_SHIFT,
        audio_shift=AUDIO_SHIFT,
    )

    benchmark_profiler = BenchmarkProfiler()
    output = run_warm_generation(
        pipeline,
        PROMPT,
        image=keyframe,
        num_frames=num_frames,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
        profiler=benchmark_profiler,
    )

    # The adapter reached the device: `load_h3_adapter_into` raises on a target it cannot place, so
    # what is left to confirm is that it bound anything at all and bound the whole of it.
    handle = pipeline._lora_handle
    assert handle is not None and len(handle) > 0, "the transformer was built without an adapter bound"
    logger.info(f"adapter {handle.name}: {len(handle)} bound targets")

    log_pipeline_perf(
        benchmark_profiler,
        label=f"{TASK}-turbo",
        pipeline=pipeline,
        num_forwards=NUM_FORWARDS,
        width=WIDTH,
        height=HEIGHT,
        num_frames=num_frames,
        fps=MINIMAX_H3_FPS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        extra_lines=(f"adapter {Path(lora_path).name}", f"video shift {VIDEO_SHIFT}, audio shift {AUDIO_SHIFT}"),
    )

    check_audio_sanity(
        output.audio,
        sampling_rate=output.sampling_rate,
        expected_seconds=num_frames / MINIMAX_H3_FPS,
    )
    assert output.video_format == "yuv420", f"asked for yuv420 but the pipeline returned {output.video_format}"

    if is_host():
        artifacts = artifact_dir("h3_turbo_artifacts")
        mp4 = artifacts / f"{TASK}_turbo_{WIDTH}x{HEIGHT}_{duration_s}s_{NUM_FORWARDS}fwd.mp4"
        export_video_audio_yuv(
            output.video,
            str(mp4),
            fps=output.fps,
            audio=Audio(waveform=output.audio[0], sampling_rate=output.sampling_rate),
        )
        logger.info(f"wrote {mp4}")

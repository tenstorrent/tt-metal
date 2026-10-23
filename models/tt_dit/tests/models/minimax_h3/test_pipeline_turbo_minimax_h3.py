# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end generation under a lightx2v MiniMax-H3-Turbo (or HyperFlow) adapter at its own working point.

Env: MINIMAX_H3_TURBO_LORA_PATH (required), MINIMAX_H3_TURBO_KEYFRAME (fl2va), MINIMAX_H3_TURBO_TASK
(fl2va|t2va), MINIMAX_H3_TURBO_POINT (768p|544p|hyperflow), MINIMAX_H3_TURBO_NFE (4|8).

Scale is ``__metadata__["alpha"] / rank`` (0.0625 for Turbo), resolved by the loader; 1.0 would be 16x too
strong yet still produce video. 768p was distilled at video shift 6, 544p and HyperFlow at 12 (audio 3).
Quality is recorded, not gated.
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

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS
from ....pipelines.minimax_h3.pipeline_minimax_h3_turbo import TURBO_NUM_FORWARDS, MiniMaxH3TurboPipeline
from ....pipelines.minimax_h3.policy import align_num_frames
from ....utils import tensor as tt_tensor
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

# A canvas and its shifts travel together: a wrong shift still completes, just on the wrong sigma grid.
WORKING_POINTS = {
    "768p": {"size": (768, 1344), "video_shift": 6.0, "audio_shift": 3.0},
    "544p": {"size": (544, 960), "video_shift": 12.0, "audio_shift": 3.0},
    "hyperflow": {"size": (768, 1344), "video_shift": 12.0, "audio_shift": 3.0},
}
WORKING_POINT = os.environ.get("MINIMAX_H3_TURBO_POINT", "768p")
HEIGHT, WIDTH = WORKING_POINTS[WORKING_POINT]["size"]
VIDEO_SHIFT = WORKING_POINTS[WORKING_POINT]["video_shift"]
AUDIO_SHIFT = WORKING_POINTS[WORKING_POINT]["audio_shift"]

DURATIONS_S = (5, 10, 15)
SEED = 0
PROMPT = CALIBRATED_FOX_PROMPT

# `num_inference_steps` counts grid points (NFE + terminal sigma). Only 4 and 8 were distilled.
NUM_FORWARDS = int(os.environ.get("MINIMAX_H3_TURBO_NFE", 4))
if NUM_FORWARDS not in TURBO_NUM_FORWARDS:
    raise ValueError(f"MINIMAX_H3_TURBO_NFE must be one of {TURBO_NUM_FORWARDS}, got {NUM_FORWARDS}")
NUM_INFERENCE_STEPS = NUM_FORWARDS + 1

# One adapter file serves both; the keyframe is the only difference between the two tasks.
TASK = os.environ.get("MINIMAX_H3_TURBO_TASK", "fl2va")

# 50 transformer blocks and 2 refiner blocks, each with fused to_qkv, to_out, ff1 and ff2.
DEVICE_BOUND_TARGETS = 52 * 4


def _assert_two_time_temb_matches_host(pipeline, mesh_device) -> None:
    """Device two-time blend vs float32 torch: the adapter's float32 embedder deltas must not round away."""
    transformer = pipeline._transformer
    two_time = transformer.two_time
    assert two_time is not None, "a HyperFlow adapter is bound but the transformer has no endpoint embedder"
    assert two_time.gate == pipeline.hyperflow.gate

    t = torch.tensor([0.05, 0.5, 0.93], dtype=torch.float32)
    r = torch.tensor([0.16, 0.7, 1.0], dtype=torch.float32)

    def on_device(levels):
        return tt_tensor.from_torch(levels.reshape(1, 1, -1, 1), device=mesh_device, dtype=ttnn.float32)

    def on_host(value):
        return ttnn.to_torch(ttnn.get_device_tensors(value)[0]).float().reshape(t.numel(), -1)

    freq_t = transformer.time_proj(on_device(t))
    freq_r = transformer.time_proj(on_device(r))
    emb_t = transformer.time_embedder(freq_t)
    emb_r = two_time.embedder(freq_r)
    blended = on_host(ttnn.add(emb_t, ttnn.multiply(ttnn.subtract(emb_r, emb_t), two_time.gate)))

    def reference(state, freq):
        hidden = torch.nn.functional.silu(freq @ state["linear_1.weight"].T + state["linear_1.bias"])
        return hidden @ state["linear_2.weight"].T + state["linear_2.bias"]

    states = pipeline._time_embedder_states
    ref_t = reference(states["time_embedder"], on_host(freq_t))
    ref_r = reference(states["endpoint_time_embedder"], on_host(freq_r))
    expected = ref_t + two_time.gate * (ref_r - ref_t)

    pcc = torch.corrcoef(torch.stack([blended.flatten(), expected.flatten()]))[0, 1].item()
    max_err = (blended - expected).abs().max().item()
    endpoint_shift = (ref_r - ref_t).abs().max().item()
    logger.info(
        f"two-time temb vs float32 host: pcc {pcc:.7f}, max |err| {max_err:.3e}, max |emb_r - emb_t| {endpoint_shift:.3e}"
    )
    assert pcc > 0.9999, f"two-time temb diverges from the float32 host reference: pcc {pcc}"


# pytest.ini caps a test at 300 s; a cold pass pays weight conversion and JIT well past that.
@pytest.mark.timeout(10800)
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

    pipeline = MiniMaxH3TurboPipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights,
        dit_fsdp=False,
        vae_output_type="yuv420",
        lora_path=lora_path,
        video_shift=VIDEO_SHIFT,
        audio_shift=AUDIO_SHIFT,
    )

    # Construction warms only the 5 s rung; warm this duration so compiles stay out of the measurement.
    with pipeline.quiet():
        pipeline(
            PROMPT,
            image=keyframe,
            num_frames=num_frames,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            seed=SEED,
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

    # `load_h3_adapter_into` raises on unplaceable targets; confirm something (and for HyperFlow, all) bound.
    handle = pipeline.adapter
    assert handle is not None and len(handle) > 0, "the transformer was built without an adapter bound"
    logger.info(f"adapter {handle.name}: {len(handle)} bound targets")
    if pipeline.hyperflow is not None:
        assert pipeline.hyperflow.num_forwards == NUM_FORWARDS
        assert (
            len(handle) == DEVICE_BOUND_TARGETS
        ), f"bound {len(handle)} device targets, expected {DEVICE_BOUND_TARGETS}"
        _assert_two_time_temb_matches_host(pipeline, mesh_device)

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

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
from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, resolve_canvas_size
from ....pipelines.minimax_h3.pipeline_minimax_h3 import AUDIO_SHIFT, VIDEO_SHIFT
from ....pipelines.minimax_h3.pipeline_minimax_h3_fast import (
    FASTH3_BASE_LADDER,
    FASTH3_NUM_GRID_POINTS,
    MiniMaxH3FastPipeline,
)
from ....pipelines.minimax_h3.pipeline_minimax_h3_pdd import MiniMaxH3PDDPipeline
from ....pipelines.minimax_h3.policy import align_num_frames, get_num_frames
from ....pipelines.minimax_h3.scheduler import shift_sigmas
from ....utils.video import Audio, export_video_audio_yuv
from .common import GALAXY_MESHES
from .common_av import (
    CALIBRATED_FOX_PROMPT,
    artifact_dir,
    assert_generation_ok,
    check_audio_sanity,
    frames_for_export,
    is_host,
    log_pipeline_perf,
    pretest_user_repl,
    run_user_generations,
    run_user_ref_generations,
    weights_dir,
    write_artifacts,
)
from .test_performance_minimax_h3 import (
    _REF2VA_L1_SMALL,
    REF2VA_ASPECT_RATIO,
    REF2VA_MESHES,
    REF2VA_REFERENCE_SETS,
    read_user_input,
    ref2va_references,
)
from .test_pipeline_pdd_minimax_h3 import _export_dir

HEIGHT, WIDTH = 768, 1344
# Other durations run through the REPL (ENABLE_USER_INPUT).
DURATIONS_S = (15,)
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
    pretest_user_repl()
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

    run_user_generations(
        pipeline,
        default_aspect_ratio=(16, 9),
        default_duration_s=duration_s,
        num_inference_steps=FASTH3_NUM_GRID_POINTS,
        seed=SEED,
        default_prompt=PROMPT,
    )


@pytest.mark.timeout(10800)
@pytest.mark.parametrize("ref_set", REF2VA_REFERENCE_SETS, ids=REF2VA_REFERENCE_SETS)
@pytest.mark.parametrize("duration_s", DURATIONS_S, ids=[f"{d}s" for d in DURATIONS_S])
@pytest.mark.parametrize(("mesh_device", "device_params"), REF2VA_MESHES, indirect=["mesh_device", "device_params"])
def test_fast_omni_end_to_end(mesh_device, reset_seeds, duration_s, ref_set):
    """FastH3-Omni (the OmniRef PDD export) ref2va under the export's trained VSA, then the ref2va REPL."""
    pretest_user_repl()
    export = _export_dir()
    aspect_ratio = REF2VA_ASPECT_RATIO
    height, width = resolve_canvas_size(*aspect_ratio)
    num_frames = get_num_frames(duration_s)
    references = ref2va_references(ref_set)

    pipeline = MiniMaxH3PDDPipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights_dir("text_encoder", "vae", "audio_vae"),
        task="ref2va",
        dit_fsdp=True,
        vae_output_type="yuv420",
        pdd_export_dir=export,
        vsa_config=MiniMaxH3PDDPipeline.export_vsa_config(export),
    )
    steps = pipeline.pdd.num_forwards + 1
    request = dict(references=references, num_frames=num_frames, height=height, width=width, seed=SEED)

    with pipeline.quiet():
        pipeline(PROMPT, num_inference_steps=steps, **request)

    profiler = BenchmarkProfiler()
    with profiler("run", iteration=0):
        output = pipeline(PROMPT, num_inference_steps=steps, on_event=profiler_event_callback(profiler, 0), **request)
        ttnn.synchronize_device(pipeline.mesh_device)

    expected_frames = align_num_frames(num_frames)
    assert_generation_ok(output, expected_frames)
    log_pipeline_perf(
        profiler,
        label=f"ref2va-fasth3-omni {ref_set}",
        pipeline=pipeline,
        num_forwards=pipeline.pdd.num_forwards,
        width=width,
        height=height,
        num_frames=expected_frames,
        fps=MINIMAX_H3_FPS,
        aspect_ratio=aspect_ratio,
        num_inference_steps=steps,
        extra_lines=(f"export {export.name}", f"L1 Small Size: {_REF2VA_L1_SMALL}"),
    )

    if ttnn.using_distributed_env():
        ttnn.distributed_context_barrier()
    if is_host():
        stem = f"ref2va_fasth3_omni_{ref_set}_{width}x{height}_{duration_s}s"
        write_artifacts(
            frames_for_export(output),
            output.audio.cpu().numpy(),
            output.sampling_rate,
            artifact_dir("h3_fast_artifacts"),
            stem=stem,
        )

    run_user_ref_generations(
        pipeline,
        lambda: read_user_input(default_num_steps=steps),
        seed=SEED,
        label="ref2va_fasth3_omni",
        artifact_name="h3_fast_artifacts",
    )

    pipeline.release_traces()

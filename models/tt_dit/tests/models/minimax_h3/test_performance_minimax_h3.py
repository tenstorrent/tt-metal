# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""t2va pipeline wall-clock at the published working points, via `BenchmarkProfiler`.

Six aspect ratios (21:9 .. 9:16) x three durations (5 / 10 / 15 s), 50 steps. The canvas comes from
`resolve_canvas_size` and the frame count from `align_num_frames`. Stage durations log on the host
rank only, and only for the warm generation. Rank 0 writes the muxed mp4; quality gates live in
`test_pipeline_minimax_h3.py`. `ENABLE_USER_INPUT=1` opens a prompt / aspect / duration
loop after the measured generation.
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames, resolve_canvas_size
from ....pipelines.minimax_h3.packing_ref2va import (
    MiniMaxH3Reference,
    decode_reference_audio,
    reference_from_video_file,
)
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from .common import GALAXY_MESHES, create_fractal_image
from .common_av import (
    CALIBRATED_FOX_PROMPT,
    artifact_dir,
    is_host,
    log_pipeline_perf,
    pretest_user_repl,
    read_user_reference_spec,
    run_user_generations,
    run_user_ref_generations,
    run_warm_generation,
    to_uint8_frames,
    weights_dir,
    write_artifacts,
)

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
    pretest_user_repl()
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

    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=weights, dit_fsdp=False)

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
    if is_host():
        artifacts = artifact_dir("h3_t2va_artifacts")
        stem = f"t2va_{aspect_ratio[0]}x{aspect_ratio[1]}_{WIDTH}x{HEIGHT}_{duration_s}s"
        frames = to_uint8_frames(output)
        write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts, stem=stem)

    run_user_generations(
        pipeline,
        default_aspect_ratio=aspect_ratio,
        default_duration_s=duration_s,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    pipeline.release_traces()


# --------------------------------------------------------------------- ref2va perf

# The single working point the ref2va perf run measures. Unlike t2va it does not sweep: ref2va
# padded lengths run 1.2-3.0x t2va's, so one point per process keeps the memory envelope in check.
REF2VA_ASPECT_RATIO = (16, 9)
REF2VA_DURATION_S = 5

# The reference set the ref2va run conditions on -- EDIT these to point at your own media. The image
# defaults to a synthetic fractal so the test runs with nothing on disk; the video and audio are
# placeholders that only join the set once a file exists at their path. ref2va needs at least one
# reference, so keep the image (or supply a video/audio) enabled.
REF2VA_USE_IMAGE = True
REF2VA_VIDEO_FILE = "/data/DC-deploy/vision-models/h3_t2va_artifacts/t2va_16x9_1344x768_5s.mp4"
REF2VA_AUDIO_FILE = "/data/DC-deploy/vision-models/h3_t2va_artifacts/t2va_16x9_1344x768_5s.wav"

# ref2va reaches the video VAE's taps=3 encoder, which clashes with the default L1 pool above it, so
# it runs a smaller L1_SMALL than the other gates and reports it in the measurement line.
_REF2VA_L1_SMALL = 16384
REF2VA_MESHES = [
    pytest.param(shape, {**params, "l1_small_size": _REF2VA_L1_SMALL}, id=param.id)
    for param in GALAXY_MESHES
    for shape, params in [param.values]
]


def ref2va_references() -> list[MiniMaxH3Reference]:
    """The image / video / audio references the ref2va perf run packs. A placeholder set to edit."""
    references: list[MiniMaxH3Reference] = []
    if REF2VA_USE_IMAGE:
        references.append(MiniMaxH3Reference(image=create_fractal_image(512, 512)))
    if False:  # os.path.isfile(REF2VA_VIDEO_FILE):
        references.append(reference_from_video_file(REF2VA_VIDEO_FILE))
    if False:  # os.path.isfile(REF2VA_AUDIO_FILE):
        waveform, sample_rate = decode_reference_audio(REF2VA_AUDIO_FILE)
        references.append(MiniMaxH3Reference(audio=waveform, sample_rate=sample_rate))
    if not references:
        pytest.skip("no ref2va references: enable REF2VA_USE_IMAGE or place a video/audio file")
    return references


def read_user_input() -> tuple[str, list[MiniMaxH3Reference], tuple[int, int], float, int] | None:
    """A `(prompt, references, aspect_ratio, duration_s, num_steps)` request from interactively-entered text and media paths.

    None when ENABLE_USER_INPUT is unset or the entry is aborted, which stops the REPL loop.
    References are packed images-first, then audio, then videos.
    """
    spec = read_user_reference_spec(REF2VA_ASPECT_RATIO, REF2VA_DURATION_S, NUM_INFERENCE_STEPS)
    if not spec:
        return None
    references: list[MiniMaxH3Reference] = []
    for path in spec.get("image", []):
        references.append(MiniMaxH3Reference(image=Image.open(path).convert("RGB")))
    for path in spec.get("audio", []):
        waveform, sample_rate = decode_reference_audio(path)
        references.append(MiniMaxH3Reference(audio=waveform, sample_rate=sample_rate))
    for path in spec.get("video", []):
        references.append(reference_from_video_file(path))
    return (
        spec["prompt"],
        references,
        (int(spec["aspect"][0]), int(spec["aspect"][1])),
        float(spec["duration_s"]),
        int(spec["num_steps"]),
    )


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(("mesh_device", "device_params"), REF2VA_MESHES, indirect=["mesh_device", "device_params"])
def test_ref2va_performance(mesh_device, reset_seeds):
    pretest_user_repl()
    weights = weights_dir("transformer_ref", "text_encoder", "vae", "audio_vae")
    prompt = PROMPT
    aspect_ratio = REF2VA_ASPECT_RATIO
    duration_s = REF2VA_DURATION_S

    HEIGHT, WIDTH = resolve_canvas_size(*aspect_ratio)
    NUM_FRAMES = align_num_frames(round(duration_s * MINIMAX_H3_FPS))
    references = ref2va_references()
    if is_host():
        kinds = ", ".join(reference.kind for reference in references)
        logger.info(f"working point: {aspect_ratio[0]}:{aspect_ratio[1]} -> {WIDTH}x{HEIGHT}, {NUM_FRAMES} frames")
        logger.info(f"references ({len(references)}): {kinds}")

    if is_host() and not os.environ.get("TT_DIT_CACHE_DIR"):
        logger.warning(
            "TT_DIT_CACHE_DIR is unset, so every weight load reads safetensors. Prepares are excluded "
            "from the total either way, but the run will take far longer than the reported compute."
        )

    # `transformer_ref` (~62 GB) is SP-replicated without FSDP, which fills each 32 GB Blackhole chip
    # and OOMs the forward-pass activations. Shard the DiT over the 32-way SP axis to free per-chip DRAM.
    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights,
        task="ref2va",
        dit_fsdp=True,
        trace_denoise=True,
        bucket_denoise=True,
    )

    benchmark_profiler = BenchmarkProfiler()
    output = run_warm_generation(
        pipeline,
        prompt,
        references=references,
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
        label="ref2va",
        pipeline=pipeline,
        num_forwards=num_forwards,
        width=WIDTH,
        height=HEIGHT,
        num_frames=expected_frames,
        fps=MINIMAX_H3_FPS,
        aspect_ratio=aspect_ratio,
        num_inference_steps=NUM_INFERENCE_STEPS,
        extra_lines=(f"L1 Small Size: {_REF2VA_L1_SMALL}",),
    )
    if ttnn.using_distributed_env():
        ttnn.distributed_context_barrier()
    if is_host():
        artifacts = artifact_dir("h3_ref2va_perf_artifacts")
        stem = f"ref2va_{aspect_ratio[0]}x{aspect_ratio[1]}_{WIDTH}x{HEIGHT}_{duration_s}s"
        frames = to_uint8_frames(output)
        write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts, stem=stem)

    run_user_ref_generations(
        pipeline,
        read_user_input,
        seed=SEED,
    )

    pipeline.release_traces()

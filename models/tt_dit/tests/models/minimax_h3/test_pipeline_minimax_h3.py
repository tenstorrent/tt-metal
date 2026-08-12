# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end `t2va` perf + quality gate at the production working point (768x1344, 124f, 50 steps)."""

from __future__ import annotations

import os

import pytest
from loguru import logger

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....utils.test import ring_params_req_exact_devices
from ..wan2_2.common import check_output_sanity
from .common_av import (
    CALIBRATED_FOX_PROMPT,
    artifact_dir,
    check_audio_sanity,
    check_av_sync,
    check_spatial_seams,
    check_written_file,
    clip_gate_enabled,
    gate_clip,
    gate_vbench,
    log_spectral_flatness,
    log_timing_table,
    run_warm_generation,
    to_uint8_frames,
    vbench_gate_enabled,
    weights_dir,
    write_artifacts,
)

HEIGHT, WIDTH = 768, 1344
NUM_FRAMES = 124
NUM_INFERENCE_STEPS = 50
SEED = 0

PROMPT = CALIBRATED_FOX_PROMPT  # tier-6 thresholds are calibrated against this exact prompt

ARTIFACT_ENV = "MINIMAX_H3_ARTIFACT_DIR"

EXPECTED_TOTAL_S = 400.0  # loose did-something-collapse bar, not a perf target

# Ring collectives require FABRIC_1D_RING; a LINE fabric fails as `fabric.cpp:174 forwarding_direction.has_value()`.
MESH_4X8 = [
    pytest.param(
        (4, 8),
        {**ring_params_req_exact_devices, "l1_small_size": 65536},
        id="4x8",
    )
]

# calibrated 2026-08-04, fox prompt, seed 0 (single sample; margins are generous)
VBENCH_THRESHOLDS = {
    "subject_consistency": 0.95,
    "background_consistency": 0.95,
    "motion_smoothness": 0.97,
    "dynamic_degree": 1.0,
    "imaging_quality": 0.64,
}
CLIP_THRESHOLD = 33.0  # measured mean 37.05 (2026-08-04, fox prompt, seed 0)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_t2va_end_to_end(mesh_device, reset_seeds):
    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    artifacts = artifact_dir(ARTIFACT_ENV, "h3_t2va_artifacts")

    vbench_enabled = vbench_gate_enabled()
    run_clip = clip_gate_enabled()

    # MINIMAX_H3_PROMPT overrides the gated prompt; the tier-6 bars are prompt-calibrated, so they turn off.
    prompt = os.environ.get("MINIMAX_H3_PROMPT") or PROMPT
    if prompt is not PROMPT:
        logger.info("MINIMAX_H3_PROMPT override in use; disabling the prompt-calibrated tier-6 gates")
        vbench_enabled = run_clip = False
    # A missing dependency must report SKIPPED, never silently pass as green.
    if run_clip:
        pytest.importorskip("open_clip", reason="RUN_CLIP=1 but open_clip is not installed (set RUN_CLIP=0)")

    if not os.environ.get("TT_DIT_CACHE_DIR"):
        logger.warning(
            "TT_DIT_CACHE_DIR is unset, so every weight load reads safetensors. Prepares are excluded "
            "from the total either way, but the run will take far longer than the reported compute."
        )

    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=weights)

    output = run_warm_generation(
        pipeline,
        prompt,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    expected_frames = align_num_frames(NUM_FRAMES)
    logger.info(
        f"generated {output.num_frames} frames ({output.video_seconds:.3f} s) and "
        f"{output.audio_seconds:.3f} s of audio at {output.sampling_rate} Hz"
    )

    num_forwards = NUM_INFERENCE_STEPS - 1
    video_seconds = expected_frames / MINIMAX_H3_FPS
    log_timing_table(
        pipeline,
        "t2va",
        num_forwards=num_forwards,
        video_seconds=video_seconds,
        expected_total_s=EXPECTED_TOTAL_S,
        extra=(
            f" | {WIDTH}x{HEIGHT}, {expected_frames} frames @ {MINIMAX_H3_FPS} fps "
            f"({video_seconds:.2f} s), {num_forwards} forwards, padded_len {pipeline.last_padded_len}"
        ),
    )

    frames = to_uint8_frames(output)

    check_output_sanity(frames, num_frames=expected_frames, height=HEIGHT, width=WIDTH)
    check_audio_sanity(
        output.audio,
        sampling_rate=output.sampling_rate,
        expected_seconds=expected_frames / MINIMAX_H3_FPS,
    )
    sync = check_av_sync(
        frames,
        output.audio,
        sampling_rate=output.sampling_rate,
        fps=MINIMAX_H3_FPS,
    )
    log_spectral_flatness(output.audio, sampling_rate=output.sampling_rate)

    # Boundaries from the VAE's own tile grid; re-deriving them with a wrong overlap gates non-boundaries.
    ratio = pipeline.vae_config.spatial_compression_ratio
    (y_starts, _, _), (x_starts, _, _) = pipeline.vae.decode_tile_grid(HEIGHT // ratio, WIDTH // ratio)
    check_spatial_seams(frames, vertical_boundaries=x_starts[1:], horizontal_boundaries=y_starts[1:])

    paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts)
    check_written_file(paths, expected_frames)

    gate_clip(frames, prompt, CLIP_THRESHOLD, "t2va", enabled=run_clip)
    gate_vbench(paths, prompt, VBENCH_THRESHOLDS, "t2va", enabled=vbench_enabled)

    logger.info(f"artifacts in {artifacts}: {sorted(p.name for p in artifacts.iterdir())}")
    logger.info(
        "REMINDER: read the artifact rubric against these frames -- the seam and flicker scores above "
        "are statistics, and only looking at the output catches what they average away"
    )
    assert sync["video_seconds"] > 0

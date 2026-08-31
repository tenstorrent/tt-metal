# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end `t2va` quality gate, swept over the published working points.

Six aspect ratios (21:9 .. 9:16) x three durations (5 / 10 / 15 s), 50 steps. The canvas comes from
`resolve_canvas_size` -- short edge 768 from 16:9 through 9:16, ~1 MPix for wider -- and the frame
count from `align_num_frames`, so neither is tabulated here. Each case writes its own artifact stem,
so a sweep does not overwrite itself.

Pipeline wall-clock lives in `test_performance_minimax_h3.py`. Sanity, seam, CLIP, and reminder logs
are silent unless `H3_LOG_QUALITY=1`.
"""

from __future__ import annotations

import os

import pytest
from loguru import logger

import ttnn

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames, resolve_canvas_size
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ..wan2_2.common import check_output_sanity
from .common import GALAXY_MESHES
from .common_av import (
    CALIBRATED_FOX_PROMPT,
    artifact_dir,
    check_audio_sanity,
    check_av_sync,
    check_spatial_seams,
    check_written_file,
    gate_clip,
    gate_vbench,
    is_host,
    log_quality,
    log_quality_warning,
    log_spectral_flatness,
    quality_logs_enabled,
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


# calibrated 2026-08-04, fox prompt, seed 0 (single sample; margins are generous)
VBENCH_THRESHOLDS = {
    "subject_consistency": 0.95,
    "background_consistency": 0.95,
    "motion_smoothness": 0.97,
    "dynamic_degree": 1.0,
    "imaging_quality": 0.64,
}
CLIP_THRESHOLD = 33.0  # measured mean 37.05 (2026-08-04, fox prompt, seed 0)

# tier-6 thresholds are calibrated against this exact prompt; swapping it invalidates both bars.
PROMPT = CALIBRATED_FOX_PROMPT

SWEEP = [
    pytest.param(ratio, seconds, id=f"{ratio[0]}x{ratio[1]}_{seconds}s")
    for seconds in DURATIONS_S
    for ratio in ASPECT_RATIOS
]


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("aspect_ratio", "duration_s"), SWEEP)
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
def test_t2va_end_to_end(mesh_device, reset_seeds, aspect_ratio, duration_s):
    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    artifacts = artifact_dir("h3_t2va_artifacts")
    prompt = PROMPT

    HEIGHT, WIDTH = resolve_canvas_size(*aspect_ratio)
    NUM_FRAMES = align_num_frames(round(duration_s * MINIMAX_H3_FPS))
    # One artifact per working point, so a sweep does not overwrite itself.
    stem = f"t2va_{aspect_ratio[0]}x{aspect_ratio[1]}_{WIDTH}x{HEIGHT}_{duration_s}s"
    if is_host():
        logger.info(
            f"working point: {aspect_ratio[0]}:{aspect_ratio[1]} -> {WIDTH}x{HEIGHT}, {NUM_FRAMES} frames, {stem}"
        )

    # A missing dependency must report SKIPPED before the long generation, never silently pass as green.
    pytest.importorskip("open_clip", reason="the CLIP gate needs open_clip, which is not installed")

    if is_host() and not os.environ.get("TT_DIT_CACHE_DIR"):
        logger.warning(
            "TT_DIT_CACHE_DIR is unset, so every weight load reads safetensors and the run will take far longer."
        )

    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device, weights_dir=weights, precomputed_adaln=False, dit_fsdp=False
    )

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
    log_quality(
        f"generated {output.num_frames} frames ({output.video_seconds:.3f} s) and "
        f"{output.audio_seconds:.3f} s of audio at {output.sampling_rate} Hz"
    )

    frames = to_uint8_frames(output)

    check_output_sanity(
        frames,
        num_frames=expected_frames,
        height=HEIGHT,
        width=WIDTH,
        log=quality_logs_enabled() and is_host(),
    )
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

    # Rank 0 alone writes and scores, and its outcome is held rather than raised: every other rank
    # goes straight to the barrier below, so a gate failure, a VBench `pytest.skip` or an artifact I/O
    # error raised here would leave them blocked and skip the trace release. `pytest.skip.Exception`
    # is named alongside `Exception` because `Skipped` does not derive from it.
    outcome: BaseException | None = None
    is_distributed = ttnn.using_distributed_env()
    if is_host():
        try:
            paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts, stem=stem)
            check_written_file(paths, expected_frames, height=HEIGHT, width=WIDTH)

            # CLIP_THRESHOLD was measured at 16:9 / 5 s. Applying it across the sweep is an extrapolation,
            # but a generous one: the calibrated point measures ~37 against a bar of 33, and the score is a
            # prompt-alignment number rather than a resolution-dependent one. A prompt change would
            # invalidate it outright -- recalibrate before swapping PROMPT.
            gate_clip(frames, prompt, CLIP_THRESHOLD, stem)
            # RUN_VBENCH=0 drops the VBench gate. It needs its own interpreter (~/vbench_env, pinned to
            # numpy<2 / transformers 4.33), and without one `run_vbench` skips -- which marks the whole
            # test SKIPPED *after* the full generation, hiding the A/V results behind a non-result.
            # Off means "not measured": everything above still gates.
            if os.environ.get("RUN_VBENCH", "1") not in ("0", "false", "False"):
                gate_vbench(paths, prompt, VBENCH_THRESHOLDS, stem)
            else:
                log_quality_warning("RUN_VBENCH=0, so the VBench gate did not run; generative quality is UNMEASURED")

            log_quality(f"artifacts in {artifacts}: {sorted(p.name for p in artifacts.iterdir())}")
        except (Exception, pytest.skip.Exception) as exc:
            outcome = exc
    if is_distributed:
        ttnn.distributed_context_barrier()
    log_quality(
        "REMINDER: read the artifact rubric against these frames -- the seam and flicker scores above "
        "are statistics, and only looking at the output catches what they average away"
    )
    # Every rank, and before the fixture closes the mesh: a captured trace holds device buffers
    # for the request. Only the 4x32 preset traces, so this is a no-op at 4x8.
    pipeline.release_traces()
    # Now that every rank has resynchronised and released its trace, rank 0 reports. mpirun surfaces
    # a single failing rank as a failing job, so the run fails rather than hanging.
    if outcome is not None:
        raise outcome
    assert sync["video_seconds"] > 0

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""End-to-end t2va with a distillation adapter, at the adapter's own step count.

This is the first thing that exercises the two halves of adapter loading together: the device half
bound onto the transformer, and the host half folded into the AdaLN precompute. Roughly 40% of a
FastH3 adapter lives in that fold, it has no device module to bind to, and a failure there produces
a perfectly valid table built from unadapted weights -- so an end-to-end run that merely completes
proves nothing. The assertions below check that the fold ran and that both halves are accounted for.

**Quality here is recorded, not gated.** ``test_pipeline_minimax_h3.py``'s CLIP and VBench bars are
calibrated against the 49-forward base model at this exact prompt; a 4-forward distilled student has
no reason to reproduce them and failing it against them would say nothing. The numbers are logged
next to the base model's for comparison, and the only assertions are structural.

Requires ``MINIMAX_H3_LORA_PATH`` pointing at a FastH3 adapter; skips without one rather than
quietly running the base model and reporting adapter timings.
"""

import os

import pytest
from loguru import logger

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames, resolve_canvas_size
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ..wan2_2.common import check_output_sanity
from .common import GALAXY_MESHES
from .common_av import (
    CALIBRATED_FOX_PROMPT,
    artifact_dir,
    check_audio_sanity,
    check_av_sync,
    check_written_file,
    clip_prompt_alignment,
    log_timing_table,
    run_vbench,
    run_warm_generation,
    to_uint8_frames,
    weights_dir,
    write_artifacts,
)

# The manifest's sampling contract: five sigma GRID POINTS, so four transformer forwards. Both
# schedulers keep their base shifts (video 12, audio 3) -- FastVideo's own pipeline asserts on that
# -- so this goes through the native `set_timesteps`, not an explicit sigma list. The manifest's
# `base_timesteps` are DMD2 training provenance and are not an inference schedule.
NUM_INFERENCE_STEPS = 5
EXPECTED_FORWARDS = NUM_INFERENCE_STEPS - 1

SEED = 0
ASPECT_RATIO = (16, 9)
PROMPT = CALIBRATED_FOX_PROMPT

# Matches `test_pipeline_minimax_h3.py`'s sweep so the two are directly comparable at each point.
DURATIONS_S = [5, 15]

# Measured on this mesh and build at 49 forwards, seed 0, same prompt -- so the log carries its own
# A/B rather than pointing at numbers from another day. Nothing here is a threshold.
BASE_MODEL_REFERENCE = {
    5: "49 forwards: 66.4 s compute (denoise 54.6 s), CLIP mean 37.31",
    15: "49 forwards: see test_pipeline_minimax_h3.py at the same point",
}

# Structural only. Four forwards should be far inside this; it exists to fail a run that silently
# fell back to the 49-forward schedule rather than to measure anything.
MAX_S_PER_VIDEO_SECOND = 40.0


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("duration_s", DURATIONS_S, ids=[f"{d}s" for d in DURATIONS_S])
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
def test_t2va_lora_end_to_end(mesh_device, reset_seeds, duration_s):
    lora_path = os.environ.get("MINIMAX_H3_LORA_PATH")
    if not lora_path:
        pytest.skip("set MINIMAX_H3_LORA_PATH to a FastH3 adapter safetensors file")
    strength = float(os.environ.get("FASTH3_LORA_STRENGTH", 1.0))

    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    artifacts = artifact_dir("h3_lora_artifacts")
    pytest.importorskip("open_clip", reason="the CLIP measurement needs open_clip, which is not installed")

    height, width = resolve_canvas_size(*ASPECT_RATIO)
    num_frames = align_num_frames(round(duration_s * MINIMAX_H3_FPS))
    stem = f"t2va_lora_{width}x{height}_{duration_s}s_{EXPECTED_FORWARDS}fwd"
    logger.info(f"adapter {lora_path} at strength {strength}")
    logger.info(f"working point: {width}x{height}, {num_frames} frames, {EXPECTED_FORWARDS} forwards")

    if not os.environ.get("TT_DIT_CACHE_DIR"):
        logger.warning("TT_DIT_CACHE_DIR is unset; every weight load reads safetensors and the run will drag")

    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights,
        lora_path=lora_path,
        lora_strength=strength,
    )

    output = run_warm_generation(
        pipeline,
        PROMPT,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    # Both halves of the adapter, checked rather than assumed. The device half reports itself; the
    # host half is only observable through the fold having been built and consumed.
    report = pipeline._lora_report
    assert report is not None, "the transformer was built without an adapter bound"
    logger.info(f"device half: {report.summary()}")
    assert report.bound, "no low-rank adapter was bound to the transformer"
    assert not report.rejected, f"unapplied payloads: {report.rejected}"
    assert report.host, (
        "no adapter entries were deferred to the host AdaLN fold, but this pipeline builds with "
        "precomputed_adaln -- 50 low-rank pairs and six dense deltas should have landed there"
    )
    logger.info(f"host half: {len(report.host)} entries folded into the AdaLN table")

    log_timing_table(
        pipeline,
        stem,
        num_forwards=EXPECTED_FORWARDS,
        video_seconds=output.video_seconds,
        extra=f"base model for comparison: {BASE_MODEL_REFERENCE.get(duration_s, 'unmeasured')}",
    )

    expected_frames = align_num_frames(num_frames)
    frames = to_uint8_frames(output)
    check_output_sanity(frames, num_frames=expected_frames, height=height, width=width)
    check_audio_sanity(output.audio, sampling_rate=output.sampling_rate, expected_seconds=output.video_seconds)
    check_av_sync(frames, output.audio, sampling_rate=output.sampling_rate, fps=MINIMAX_H3_FPS)

    paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts, stem=stem)
    check_written_file(paths, expected_frames, height=height, width=width)

    alignment = clip_prompt_alignment(frames, PROMPT)
    logger.info(
        f"{stem} CLIP prompt alignment: mean={alignment['mean']:.2f} min={alignment['min']:.2f} "
        f"max={alignment['max']:.2f}  (RECORDED not gated; base at this point: "
        f"{BASE_MODEL_REFERENCE.get(duration_s, 'unmeasured')})"
    )

    if "mp4" in paths and os.environ.get("RUN_VBENCH", "1") not in ("0", "false", "False"):
        dimensions = ("subject_consistency", "background_consistency", "motion_smoothness", "imaging_quality")
        scores = run_vbench(paths["mp4"], PROMPT, dimensions)
        for dimension in dimensions:
            value = scores.get(dimension)
            logger.info(f"{stem} VBench {dimension} = {value if value is None else f'{value:.4f}'}  (RECORDED)")
    else:
        logger.warning("VBench did not run; generative quality is measured by CLIP alone")

    seconds_per_video_second = sum(seconds for _, seconds in pipeline.last_timings) / output.video_seconds
    assert seconds_per_video_second < MAX_S_PER_VIDEO_SECOND, (
        f"{seconds_per_video_second:.1f} s of compute per video second is base-model territory; "
        f"the {EXPECTED_FORWARDS}-forward schedule likely did not take effect"
    )
    logger.info(f"artifacts in {artifacts}: {sorted(p.name for p in artifacts.iterdir())}")

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end t2va with a two-time adapter, on the grid the adapter publishes.

``test_pipeline_lora_minimax_h3.py`` is the same shape for a plain distillation adapter, and the
structural checks it makes -- both halves of the adapter accounted for -- hold here too. This file
adds the three things interval conditioning brings, each of which completes a run and produces video
when it is wrong:

* **The grid came from the file.** The adapter, not the caller, decides the forward count. A run that
  silently fell back to 49 forwards is a fast pass with a slow number, so the resolved count is
  asserted and an explicit mismatching count is asserted to be refused.
* **The endpoint reached the table, completely.** Checked by
  :func:`common_av.assert_hyperflow_applied`, shared with the yuv420 gate.

**Quality here is recorded, not gated**, for the reason
``test_pipeline_lora_minimax_h3.py`` gives: the CLIP and VBench bars in
``test_pipeline_minimax_h3.py`` are calibrated against the 49-forward base model, and an 8-forward
student has no reason to reproduce them.

Requires ``MINIMAX_H3_HYPERFLOW_LORA_PATH``; skips without one rather than quietly measuring the
base model. Kept separate from ``MINIMAX_H3_LORA_PATH`` so a sweep over plain adapters cannot pick
this file up, and so a two-time adapter cannot be handed to the plain test.
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
    assert_hyperflow_applied,
    check_audio_sanity,
    check_av_sync,
    check_written_file,
    clip_prompt_alignment,
    log_timing_table,
    run_warm_generation,
    to_uint8_frames,
    weights_dir,
    write_artifacts,
)

LORA_PATH_ENV = "MINIMAX_H3_HYPERFLOW_LORA_PATH"

SEED = 0
ASPECT_RATIO = (16, 9)
PROMPT = CALIBRATED_FOX_PROMPT
DURATION_S = 5

# Measured on this mesh and build at 49 forwards, seed 0, same prompt -- so the log carries its own
# A/B rather than pointing at numbers from another day. Nothing here is a threshold.
BASE_MODEL_REFERENCE = {5: "49 forwards: 66.4 s compute (denoise 54.6 s), CLIP mean 37.31"}

# Structural only. Eight forwards should be far inside this; it exists to fail a run that silently
# fell back to the 49-forward schedule rather than to measure anything.
MAX_S_PER_VIDEO_SECOND = 40.0


@pytest.mark.timeout(5400)
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
def test_t2va_hyperflow_end_to_end(mesh_device, reset_seeds, expect_error):
    lora_path = os.environ.get(LORA_PATH_ENV)
    if not lora_path:
        pytest.skip(f"set {LORA_PATH_ENV} to a two-time adapter safetensors file")
    strength = float(os.environ.get("FASTH3_LORA_STRENGTH", "1.0"))

    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    artifacts = artifact_dir("h3_hyperflow_artifacts")
    pytest.importorskip("open_clip", reason="the CLIP measurement needs open_clip, which is not installed")

    height, width = resolve_canvas_size(*ASPECT_RATIO)
    num_frames = align_num_frames(round(DURATION_S * MINIMAX_H3_FPS))

    if not os.environ.get("TT_DIT_CACHE_DIR"):
        logger.warning("TT_DIT_CACHE_DIR is unset; every weight load reads safetensors and the run will drag")

    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights,
        lora_path=lora_path,
        lora_strength=strength,
    )

    # 1. The contract, before anything is built: this is what decides the schedule, so a run that
    #    got here with no contract is measuring a plain adapter at Diffusers' default.
    contract = pipeline.hyperflow
    assert contract is not None, (
        f"{lora_path} publishes no sampling contract, so this pipeline would run it at 50 sigma "
        f"points; point {LORA_PATH_ENV} at a two-time adapter or use test_pipeline_lora instead"
    )
    num_forwards = contract.num_forwards
    stem = f"t2va_hyperflow_{width}x{height}_{DURATION_S}s_{num_forwards}fwd"
    logger.info(f"adapter {lora_path} at strength {strength}: {contract.identity()}")
    logger.info(f"working point: {width}x{height}, {num_frames} frames, {num_forwards} forwards")

    # 2. An explicit step count that is not the adapter's is refused, not silently honoured.
    with expect_error(ValueError, "cannot be honoured"):
        pipeline(PROMPT, num_frames=num_frames, height=height, width=width, num_inference_steps=50, seed=SEED)

    output = run_warm_generation(
        pipeline,
        PROMPT,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=SEED,
    )

    # 3. Both halves of the adapter and the schedule that actually ran.
    assert_hyperflow_applied(pipeline, num_forwards=num_forwards)

    log_timing_table(
        pipeline,
        stem,
        num_forwards=num_forwards,
        video_seconds=output.video_seconds,
        extra=f"base model for comparison: {BASE_MODEL_REFERENCE.get(DURATION_S, 'unmeasured')}",
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
        f"{BASE_MODEL_REFERENCE.get(DURATION_S, 'unmeasured')})"
    )

    seconds_per_video_second = sum(seconds for _, seconds in pipeline.last_timings) / output.video_seconds
    assert seconds_per_video_second < MAX_S_PER_VIDEO_SECOND, (
        f"{seconds_per_video_second:.1f} s of compute per video second is base-model territory; "
        f"the {num_forwards}-forward schedule likely did not take effect"
    )
    logger.info(f"artifacts in {artifacts}: {sorted(p.name for p in artifacts.iterdir())}")

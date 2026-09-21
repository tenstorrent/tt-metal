# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Serving-configuration timing for a two-time adapter: ``test_pipeline_hyperflow_minimax_h3.py``'s
working point with ``vae_output_type="yuv420"``.

Timing only, no pixel gates -- the same split ``test_zz_yuv_timing.py`` makes for FastH3, and for the
same reason: ``to_uint8_frames`` and everything downstream of it (CLIP, the sanity checks, the
artifact writer) read the ``rgb_float`` layout, so a yuv420 run cannot go through them. The gated
test therefore runs the float VAE and reports a total the serving path never pays, and this one
reports the total it does.

What is *not* dropped is every structural check on the adapter: those live in
:func:`common_av.assert_hyperflow_applied` and hold whatever the pixel layout is. A yuv timing number
with no proof the 8-step interval-conditioned schedule was the thing being timed is worthless.

Requires ``MINIMAX_H3_HYPERFLOW_LORA_PATH``. Dense attention, matching the gated test, so the two
totals differ only by the VAE readback.
"""

import os

import pytest
from loguru import logger

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames, resolve_canvas_size
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....utils.video import Audio, export_video_audio_yuv
from .common import GALAXY_MESHES
from .common_av import (
    CALIBRATED_FOX_PROMPT,
    artifact_dir,
    assert_hyperflow_applied,
    log_timing_table,
    run_warm_generation,
    weights_dir,
)

LORA_PATH_ENV = "MINIMAX_H3_HYPERFLOW_LORA_PATH"

SEED = 0
ASPECT_RATIO = (16, 9)
DURATIONS_S = [5]

# The float-VAE total for the same working point, measured on this mesh and build at 8 forwards, seed
# 0, same prompt -- so the log carries the A/B this file exists to make. Nothing here is a threshold.
FLOAT_VAE_REFERENCE = {5: "float VAE, 8 forwards: 15.2 s compute (denoise 9.2 s, VAE decode 4.4 s)"}


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("duration_s", DURATIONS_S, ids=[f"{d}s" for d in DURATIONS_S])
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES[:1], indirect=["mesh_device", "device_params"])
def test_t2va_hyperflow_yuv_timing(mesh_device, reset_seeds, duration_s):
    lora_path = os.environ.get(LORA_PATH_ENV)
    if not lora_path:
        pytest.skip(f"set {LORA_PATH_ENV} to a two-time adapter safetensors file")

    # MINIMAX_H3_VAE_PHASES synchronizes between the decode's phases to separate them, which also
    # serializes them: the stage total it reports is inflated and only the shares are readable.
    stitch = os.environ.get("MINIMAX_H3_VAE_STITCH", "gather")
    profile_phases = bool(int(os.environ.get("MINIMAX_H3_VAE_PHASES", "0")))

    height, width = resolve_canvas_size(*ASPECT_RATIO)
    num_frames = align_num_frames(round(duration_s * MINIMAX_H3_FPS))

    if not os.environ.get("TT_DIT_CACHE_DIR"):
        logger.warning("TT_DIT_CACHE_DIR is unset; every weight load reads safetensors and the run will drag")

    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights_dir("transformer", "text_encoder", "vae", "audio_vae"),
        lora_path=lora_path,
        lora_strength=float(os.environ.get("FASTH3_LORA_STRENGTH", "1.0")),
        vae_output_type="yuv420",
        vae_stitch_exchange=stitch,
        vae_profile=profile_phases,
    )

    contract = pipeline.hyperflow
    assert contract is not None, (
        f"{lora_path} publishes no sampling contract, so this pipeline would run it at 50 sigma "
        f"points; point {LORA_PATH_ENV} at a two-time adapter"
    )
    num_forwards = contract.num_forwards
    logger.info(f"adapter {lora_path}: {contract.identity()}")
    logger.info(f"working point: {width}x{height}, {num_frames} frames, {num_forwards} forwards, yuv420 readback")

    output = run_warm_generation(
        pipeline,
        CALIBRATED_FOX_PROMPT,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=SEED,
    )
    assert output.video_format == "yuv420", f"asked for yuv420 but the pipeline returned {output.video_format}"

    assert_hyperflow_applied(pipeline, num_forwards=num_forwards)

    stem = f"t2va_hyperflow_yuv420_{stitch}_{width}x{height}_{duration_s}s_{num_forwards}fwd"
    log_timing_table(
        pipeline,
        stem,
        num_forwards=num_forwards,
        video_seconds=output.video_seconds,
        extra=(
            f", stitch_exchange={stitch}"
            + (", PHASE-SERIALIZED" if profile_phases else "")
            + f" | {FLOAT_VAE_REFERENCE.get(duration_s, 'float VAE unmeasured at this duration')}"
        ),
    )

    # Written after the timing table so the encode never lands inside a measured stage.
    mp4 = artifact_dir("h3_hyperflow_artifacts") / f"{stem}.mp4"
    export_video_audio_yuv(
        output.video,
        str(mp4),
        fps=output.fps,
        audio=Audio(waveform=output.audio[0], sampling_rate=output.sampling_rate),
    )
    logger.info(f"wrote {mp4}")

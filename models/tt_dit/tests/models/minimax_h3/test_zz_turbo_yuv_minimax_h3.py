# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Serving-configuration timing for a lightx2v MiniMax-H3-Turbo adapter: the turbo working point with
``vae_output_type="yuv420"``.

Timing only, no pixel gates, for the reason ``test_zz_hyperflow_yuv_minimax_h3.py`` gives: everything
downstream of ``to_uint8_frames`` reads the ``rgb_float`` layout, so a yuv420 run cannot go through
it. This reports the total the serving path actually pays.

Two properties of these adapters are silent when wrong, and neither is discoverable from the tensors:

* **Scale lives in the file's metadata.** The publish carries no per-target ``.alpha``, only
  ``__metadata__["alpha"]``; the real scale is ``alpha / rank`` -- 0.0625 for every published Turbo
  adapter. The generic loader reads per-target alphas, finds none, and applies 1.0, which is 16x too
  strong and still produces a video. ``MINIMAX_H3_TURBO_STRENGTH`` therefore defaults to 0.0625 here
  rather than to the loader's 1.0.
* **The 768p variants were distilled against video shift 6**, not the checkpoint's 12 (audio stays 3
  either way); 544p keeps 12/3. The canvas and its shifts travel together in ``WORKING_POINTS`` and
  reach the pipeline through ``MINIMAX_H3_VIDEO_SHIFT`` / ``MINIMAX_H3_AUDIO_SHIFT``, which this file
  sets before the pipeline reads them at import.

Requires ``MINIMAX_H3_TURBO_LORA_PATH``; skips without one rather than quietly timing the base model.
"""

import os

import pytest
from loguru import logger

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames
from ....utils.video import Audio, export_video_audio_yuv
from .common import GALAXY_MESHES
from .common_av import (
    CALIBRATED_FOX_PROMPT,
    artifact_dir,
    check_audio_sanity,
    log_timing_table,
    run_warm_generation,
    weights_dir,
)

LORA_PATH_ENV = "MINIMAX_H3_TURBO_LORA_PATH"

SEED = 0
DURATIONS_S = (5, 10, 15)

# Each adapter names one canvas and one pair of shifts, and the two travel together.
WORKING_POINTS = {
    "768p": {"size": (768, 1344), "video_shift": "6.0", "audio_shift": "3.0"},
    "544p": {"size": (544, 960), "video_shift": "12.0", "audio_shift": "3.0"},
}
WORKING_POINT = os.environ.get("MINIMAX_H3_TURBO_POINT", "768p")
HEIGHT, WIDTH = WORKING_POINTS[WORKING_POINT]["size"]

# The pipeline reads the shifts into module constants at import, so they have to be in the
# environment before it is imported.
os.environ.setdefault("MINIMAX_H3_VIDEO_SHIFT", WORKING_POINTS[WORKING_POINT]["video_shift"])
os.environ.setdefault("MINIMAX_H3_AUDIO_SHIFT", WORKING_POINTS[WORKING_POINT]["audio_shift"])

from ....pipelines.minimax_h3.pipeline_minimax_h3 import (  # noqa: E402  (must follow the shift setdefault)
    AUDIO_SHIFT,
    VIDEO_SHIFT,
    MiniMaxH3Pipeline,
)

# NFE from the model card; `num_inference_steps` counts sigma grid points, so it is one more.
NUM_FORWARDS = int(os.environ.get("MINIMAX_H3_TURBO_NFE", "4"))
NUM_INFERENCE_STEPS = NUM_FORWARDS + 1


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("duration_s", DURATIONS_S, ids=[f"{d}s" for d in DURATIONS_S])
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
def test_t2va_turbo_yuv_timing(mesh_device, reset_seeds, duration_s):
    lora_path = os.environ.get(LORA_PATH_ENV)
    if not lora_path:
        pytest.skip(f"set {LORA_PATH_ENV} to a lightx2v Minimax-h3-Turbo adapter safetensors file")

    # The setdefault above only lands if nothing imported the pipeline first; a wrong shift is a
    # valid schedule over the wrong sigma grid, so it has to fail here rather than cost quality.
    assert (VIDEO_SHIFT, AUDIO_SHIFT) == (
        float(WORKING_POINTS[WORKING_POINT]["video_shift"]),
        float(WORKING_POINTS[WORKING_POINT]["audio_shift"]),
    ), (
        f"{WORKING_POINT} needs shifts "
        f"{WORKING_POINTS[WORKING_POINT]['video_shift']}/{WORKING_POINTS[WORKING_POINT]['audio_shift']} "
        f"but the pipeline resolved {VIDEO_SHIFT}/{AUDIO_SHIFT}; set MINIMAX_H3_VIDEO_SHIFT and "
        "MINIMAX_H3_AUDIO_SHIFT in the launch environment"
    )
    strength = float(os.environ.get("MINIMAX_H3_TURBO_STRENGTH", "0.0625"))
    stitch = os.environ.get("MINIMAX_H3_VAE_STITCH", "gather")

    num_frames = align_num_frames(round(duration_s * MINIMAX_H3_FPS))

    if not os.environ.get("TT_DIT_CACHE_DIR"):
        logger.warning("TT_DIT_CACHE_DIR is unset; every weight load reads safetensors and the run will drag")

    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights_dir("transformer", "text_encoder", "vae", "audio_vae"),
        lora_path=lora_path,
        lora_strength=strength,
        vae_output_type="yuv420",
        vae_stitch_exchange=stitch,
    )

    # A turbo adapter publishes no sampling contract, so the step count is the caller's to supply --
    # the inverse of the HyperFlow tests, where a caller-supplied count is refused. Assert the
    # absence so a two-time adapter handed to this file is caught rather than silently re-scheduled.
    assert pipeline.hyperflow is None, (
        f"{lora_path} publishes a sampling contract, so it is a two-time adapter; "
        "use test_zz_hyperflow_yuv_minimax_h3.py for those"
    )
    logger.info(
        f"adapter {os.path.basename(lora_path)} at strength {strength}: {WIDTH}x{HEIGHT}, "
        f"{num_frames} frames, {NUM_FORWARDS} forwards, video shift {VIDEO_SHIFT} / audio {AUDIO_SHIFT}, "
        "yuv420 readback"
    )

    output = run_warm_generation(
        pipeline,
        CALIBRATED_FOX_PROMPT,
        num_frames=num_frames,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )
    assert output.video_format == "yuv420", f"asked for yuv420 but the pipeline returned {output.video_format}"

    stem = f"t2va_turbo_yuv420_{stitch}_{WIDTH}x{HEIGHT}_{duration_s}s_{NUM_FORWARDS}fwd"
    log_timing_table(
        pipeline,
        stem,
        num_forwards=NUM_FORWARDS,
        video_seconds=num_frames / MINIMAX_H3_FPS,
        extra=f" | turbo {WORKING_POINT}, strength {strength}, shift {VIDEO_SHIFT}/{AUDIO_SHIFT}",
    )

    check_audio_sanity(output.audio, sampling_rate=output.sampling_rate, expected_seconds=num_frames / MINIMAX_H3_FPS)

    artifacts = artifact_dir("h3_turbo_artifacts")
    mp4 = artifacts / f"{stem}.mp4"
    export_video_audio_yuv(
        output.video,
        str(mp4),
        fps=output.fps,
        audio=Audio(waveform=output.audio[0], sampling_rate=output.sampling_rate),
    )
    logger.info(f"wrote {mp4}")

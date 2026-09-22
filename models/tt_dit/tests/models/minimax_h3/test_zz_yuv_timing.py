# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Serving-configuration timing for FastH3 + VSA: the same working point as
``test_pipeline_lora_minimax_h3.py`` but with ``vae_output_type="yuv420"``.

Timing only, no pixel gates. ``to_uint8_frames`` and everything downstream of it read the
``rgb_float`` layout, so a yuv420 run cannot go through them -- which is why the gated test runs the
float VAE and reports a total the serving path never pays.
"""

import os

import pytest
from loguru import logger

from ....models.transformers.minimax_h3.vsa_stages_minimax_h3 import MiniMaxH3VSAConfig
from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames, resolve_canvas_size
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....utils.video import Audio, export_video_audio_yuv
from .common import GALAXY_MESHES
from .common_av import CALIBRATED_FOX_PROMPT, artifact_dir, log_timing_table, run_warm_generation, weights_dir

NUM_INFERENCE_STEPS = 5
EXPECTED_FORWARDS = NUM_INFERENCE_STEPS - 1
# MINIMAX_H3_SEED overrides it, so a sweep can move off seed 0 -- the audio a seed produces is part of the
# generation, not the decoder, so comparing decoder configurations does not require keeping it.
SEED = int(os.environ.get("MINIMAX_H3_SEED", "0"))
# MINIMAX_H3_PROMPT swaps the prompt. The calibrated one is what the timing numbers were taken on, so a
# different prompt is for listening to or looking at a clip, not for comparing against those numbers.
PROMPT = os.environ.get("MINIMAX_H3_PROMPT") or CALIBRATED_FOX_PROMPT
ASPECT_RATIO = (16, 9)
DURATIONS_S = [5, 10, 15]
VSA_SPARSITY = 0.9


def _write_frame_crcs(video, height: int, path: str) -> None:
    """One line per frame: crc32 of the planar frame, then of the four row bands of its Y plane (the
    strip stitch's mesh rows), so two runs compare at the raw level and a difference has a location."""
    import zlib

    import numpy as np

    frames = np.asarray(video)
    band = height // 4
    with open(path, "w") as handle:
        for index, frame in enumerate(frames):
            luma = frame[:height]
            bands = " ".join(
                f"{zlib.crc32(np.ascontiguousarray(luma[r : r + band]).tobytes()):08x}" for r in range(0, height, band)
            )
            handle.write(f"{index} {zlib.crc32(np.ascontiguousarray(frame).tobytes()):08x} {bands}\n")


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("duration_s", DURATIONS_S, ids=[f"{d}s" for d in DURATIONS_S])
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES[:1], indirect=["mesh_device", "device_params"])
def test_t2va_lora_yuv_timing(mesh_device, reset_seeds, duration_s):
    lora_path = os.environ.get("MINIMAX_H3_LORA_PATH")
    if not lora_path:
        pytest.skip("set MINIMAX_H3_LORA_PATH to a FastH3 adapter safetensors file")

    # MINIMAX_H3_VAE_PHASES synchronizes between the decode's phases to separate them, which also
    # serializes them: the stage total it reports is inflated and only the shares are readable.
    stitch = os.environ.get("MINIMAX_H3_VAE_STITCH")  # unset: the pipeline's default
    profile_phases = bool(int(os.environ.get("MINIMAX_H3_VAE_PHASES", "0")))

    height, width = resolve_canvas_size(*ASPECT_RATIO)
    num_frames = align_num_frames(round(duration_s * MINIMAX_H3_FPS))

    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights_dir("transformer", "text_encoder", "vae", "audio_vae"),
        lora_path=lora_path,
        lora_strength=float(os.environ.get("FASTH3_LORA_STRENGTH", 1.0)),
        vsa_config=MiniMaxH3VSAConfig(sparsity=VSA_SPARSITY),
        vae_output_type="yuv420",
        vae_profile=profile_phases,
        **({"vae_stitch_exchange": stitch} if stitch else {}),
    )
    stitch = pipeline.vae_stitch_exchange

    output = run_warm_generation(
        pipeline,
        PROMPT,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )
    assert output.video_format == "yuv420", f"asked for yuv420 but the pipeline returned {output.video_format}"
    if os.environ.get("MINIMAX_H3_FRAME_CRC"):
        _write_frame_crcs(output.video, height, os.environ["MINIMAX_H3_FRAME_CRC"])
    if os.environ.get("MINIMAX_H3_FRAME_DUMP"):
        import numpy as np

        np.save(os.environ["MINIMAX_H3_FRAME_DUMP"], np.asarray(output.video))

    report = pipeline._lora_report
    assert report is not None and report.bound, "the transformer was built without an adapter bound"
    assert len(report.replaced) == pipeline.transformer_config["num_layers"], (
        f"{len(report.replaced)} gates assigned for {pipeline.transformer_config['num_layers']} blocks; "
        "VSA is running partly ungated"
    )
    logger.info(f"VSA: {len(report.replaced)} gates assigned and active")

    stem = f"t2va_lora_vsa_yuv420_{stitch}_{width}x{height}_{duration_s}s_{EXPECTED_FORWARDS}fwd"
    log_timing_table(
        pipeline,
        stem,
        num_forwards=EXPECTED_FORWARDS,
        video_seconds=output.video_seconds,
        extra=f", stitch_exchange={stitch}" + (", PHASE-SERIALIZED" if profile_phases else ""),
    )

    # Written after the timing table so the encode never lands inside a measured stage.
    mp4 = artifact_dir("h3_lora_artifacts") / f"{stem}.mp4"
    export_video_audio_yuv(
        output.video,
        str(mp4),
        fps=output.fps,
        audio=Audio(waveform=output.audio[0], sampling_rate=output.sampling_rate),
    )
    logger.info(f"wrote {mp4}")

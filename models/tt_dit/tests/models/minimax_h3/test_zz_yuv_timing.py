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

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

from ....models.transformers.minimax_h3.vsa_stages_minimax_h3 import MiniMaxH3VSAConfig
from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, resolve_canvas_size
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....pipelines.minimax_h3.policy import align_num_frames
from ....utils.test import is_global_rank_zero
from ....utils.video import Audio, export_video_audio_yuv
from .common import GALAXY_MESHES, MESH_4X8_RING
from .common_av import CALIBRATED_FOX_PROMPT, artifact_dir, log_timing_table, weights_dir

NUM_INFERENCE_STEPS = 4
EXPECTED_FORWARDS = 4  # NUM_INFERENCE_STEPS - 1
# MINIMAX_H3_SEED overrides it, so a sweep can move off seed 0 -- the audio a seed produces is part of the
# generation, not the decoder, so comparing decoder configurations does not require keeping it.
SEED = int(os.environ.get("MINIMAX_H3_SEED", "0"))
# MINIMAX_H3_PROMPT swaps the prompt. The calibrated one is what the timing numbers were taken on, so a
# different prompt is for listening to or looking at a clip, not for comparing against those numbers.
PROMPT = os.environ.get("MINIMAX_H3_PROMPT") or CALIBRATED_FOX_PROMPT
ASPECT_RATIO = (16, 9)
DURATIONS_S = [5, 10, 15]
VSA_SPARSITY = 0.9

# 4x8's parameters carry no trace region -- only the quad's, for `trace_denoise` -- so the audio vocoder has nowhere
# to capture into. Reserving costs address space rather than working DRAM; this size fits the vocoder graph.
_MESH_4X8_TRACE = pytest.param(
    MESH_4X8_RING.values[0],
    {**MESH_4X8_RING.values[1], "trace_region_size": 1_200_000_000},
    id=MESH_4X8_RING.id,
)
SERVING_MESHES = [_MESH_4X8_TRACE, *GALAXY_MESHES[1:]]


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


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("duration_s", DURATIONS_S, ids=[f"{d}s" for d in DURATIONS_S])
@pytest.mark.parametrize(("mesh_device", "device_params"), SERVING_MESHES, indirect=["mesh_device", "device_params"])
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
        audio_trace=False,
        **({"vae_stitch_exchange": stitch} if stitch else {}),
    )
    stitch = pipeline.vae_stitch_exchange

    gen_kwargs = dict(
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )

    pipeline.warmup(prompt=PROMPT, **gen_kwargs)
    warm_padded_len = pipeline.last_padded_len
    if pipeline.trace_denoise:
        pipeline(PROMPT, seed=SEED, **gen_kwargs)

    ttnn.synchronize_device(mesh_device)
    if ttnn.using_distributed_env():
        ttnn.distributed_context_barrier()

    profiler = BenchmarkProfiler()
    with profiler("run", iteration=0):
        output = pipeline(PROMPT, seed=SEED, **gen_kwargs)
        ttnn.synchronize_device(mesh_device)

    assert pipeline.last_padded_len == warm_padded_len, (
        f"warmup ran at padded_len {warm_padded_len} but the measured call ran at "
        f"{pipeline.last_padded_len}; this number is not warm"
    )
    assert output.video_format == "yuv420", f"asked for yuv420 but the pipeline returned {output.video_format}"
    if os.environ.get("MINIMAX_H3_FRAME_CRC"):
        _write_frame_crcs(output.video, height, os.environ["MINIMAX_H3_FRAME_CRC"])
    if os.environ.get("MINIMAX_H3_FRAME_DUMP"):
        import numpy as np

        np.save(os.environ["MINIMAX_H3_FRAME_DUMP"], np.asarray(output.video))

    report = pipeline._lora_report
    assert report is not None and report.bound, "the transformer was built without an adapter bound"
    if pipeline.vsa_config is not None:
        assert len(report.replaced) == pipeline.transformer_config["num_layers"], (
            f"{len(report.replaced)} gates assigned for {pipeline.transformer_config['num_layers']} blocks; "
            "VSA is running partly ungated"
        )

    stem = f"t2va_lora_vsa_yuv420_{stitch}_{width}x{height}_{duration_s}s_{EXPECTED_FORWARDS}fwd"
    if is_global_rank_zero():
        logger.info(f"VSA: {len(report.replaced)} gates assigned and active")
        log_timing_table(
            pipeline,
            stem,
            num_forwards=EXPECTED_FORWARDS,
            video_seconds=output.video_seconds,
            extra=(
                f", stitch_exchange={stitch}"
                + (", PHASE-SERIALIZED" if profile_phases else "")
                + f", profiler.run={profiler.get_duration('run', 0):.2f}s"
            ),
        )
        mp4 = artifact_dir("h3_lora_artifacts") / f"{stem}.mp4"
        export_video_audio_yuv(
            output.video,
            str(mp4),
            fps=output.fps,
            audio=Audio(waveform=output.audio[0], sampling_rate=output.sampling_rate),
        )
        logger.info(f"wrote {mp4}")
    if ttnn.using_distributed_env():
        ttnn.distributed_context_barrier()
    pipeline.release_traces()

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
from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames, resolve_canvas_size
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....utils.test import is_global_rank_zero
from ....utils.video import Audio, export_video_audio_yuv
from .common import GALAXY_MESHES
from .common_av import CALIBRATED_FOX_PROMPT, artifact_dir, log_timing_table, weights_dir

NUM_INFERENCE_STEPS = 5
EXPECTED_FORWARDS = NUM_INFERENCE_STEPS - 1
SEED = 0
ASPECT_RATIO = (16, 9)
DURATIONS_S = [5, 10, 15]
VSA_SPARSITY = 0.9


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("duration_s", DURATIONS_S, ids=[f"{d}s" for d in DURATIONS_S])
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
def test_t2va_lora_yuv_timing(mesh_device, reset_seeds, duration_s):
    lora_path = os.environ.get("MINIMAX_H3_LORA_PATH")
    if not lora_path:
        pytest.skip("set MINIMAX_H3_LORA_PATH to a FastH3 adapter safetensors file")

    # MINIMAX_H3_VAE_PHASES synchronizes between the decode's phases to separate them, which also
    # serializes them: the stage total it reports is inflated and only the shares are readable.
    stitch = os.environ.get("MINIMAX_H3_VAE_STITCH", "gather")
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
        vae_stitch_exchange=stitch,
        vae_profile=profile_phases,
    )

    gen_kwargs = dict(
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )

    pipeline.warmup(prompt=CALIBRATED_FOX_PROMPT, **gen_kwargs)
    warm_padded_len = pipeline.last_padded_len
    if pipeline.trace_denoise:
        pipeline(CALIBRATED_FOX_PROMPT, seed=SEED, **gen_kwargs)

    ttnn.synchronize_device(mesh_device)
    if ttnn.using_distributed_env():
        ttnn.distributed_context_barrier()

    profiler = BenchmarkProfiler()
    with profiler("run", iteration=0):
        output = pipeline(CALIBRATED_FOX_PROMPT, seed=SEED, **gen_kwargs)
        ttnn.synchronize_device(mesh_device)

    assert pipeline.last_padded_len == warm_padded_len, (
        f"warmup ran at padded_len {warm_padded_len} but the measured call ran at "
        f"{pipeline.last_padded_len}; this number is not warm"
    )
    assert output.video_format == "yuv420", f"asked for yuv420 but the pipeline returned {output.video_format}"

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

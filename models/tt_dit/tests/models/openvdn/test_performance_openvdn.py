# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""OpenVDN t2va pipeline times after warm-up on the Blackhole 4x8 (TP 4 x SP 8), via `BenchmarkProfiler`, as
`test_performance_minimax_h3.py` does for H3: one quiet generation at the request's shape (3 steps) compiles every
program, then the measured generation reports per-stage times. Its output is kept and held to the end-to-end test's
gates (frame and audio sanity, mean CLIP >= 30), so every perf number comes with its quality check.

The pipeline is coresident (`pipeline_openvdn.py`), so no stage is reloaded per request.

Env: OPENVDN_WEIGHTS_DIR, OPENVDN_CHECKPOINT_DIR, OPENVDN_ARTIFACTS, OPENVDN_DURATION_S (see test_pipeline_openvdn.py),
OPENVDN_CORESIDENT=0 (reload each stage per request, the pre-stage-4 behaviour), OPENVDN_TRACE=1 (traced denoise)

Run: pytest models/tt_dit/tests/models/openvdn/test_performance_openvdn.py -s --tb=short
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from PIL import Image

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, resolve_canvas_size
from ....pipelines.minimax_h3.policy import get_num_frames
from ....pipelines.openvdn.pipeline_openvdn import OPENVDN_NUM_FORWARDS, OpenVDNPipeline
from ..minimax_h3.common import MESH_4X8_RING
from ..minimax_h3.common_av import (
    check_audio_sanity,
    clip_prompt_alignment,
    is_host,
    log_pipeline_perf,
    profiler_event_callback,
    warm_quietly,
    to_uint8_frames,
    write_artifacts,
)
from ..wan2_2.common import check_output_sanity
from .test_pipeline_openvdn import (
    ASPECT_RATIO,
    CLIP_FLOOR,
    DURATION_S,
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    _required_dir,
)

STAGES = ("encoder", "denoising", "vae", "audio", "run")
#: OPENVDN_TRACE=1: the denoise loop captured once and replayed (`trace_denoise`, persistent CCL buffers).
TRACE = os.environ.get("OPENVDN_TRACE") == "1"
MESH = (
    pytest.param((4, 8), {**MESH_4X8_RING.values[1], "trace_region_size": 1_175_000_000}, id="4x8_traced")
    if TRACE
    else MESH_4X8_RING
)


@pytest.mark.timeout(14400)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH], indirect=["mesh_device", "device_params"])
def test_openvdn_t2va_performance(mesh_device, reset_seeds):
    weights_dir = _required_dir("OPENVDN_WEIGHTS_DIR")
    checkpoint_dir = _required_dir("OPENVDN_CHECKPOINT_DIR")
    pytest.importorskip("open_clip", reason="the CLIP gate needs open_clip, which is not installed")
    artifacts = Path(os.environ.get("OPENVDN_ARTIFACTS", Path.home() / "openvdn_artifacts"))
    artifacts.mkdir(parents=True, exist_ok=True)

    height, width = resolve_canvas_size(*ASPECT_RATIO)
    num_frames = get_num_frames(DURATION_S)
    stem = f"openvdn_perf_t2va_{width}x{height}_{num_frames}f_{OPENVDN_NUM_FORWARDS}nfe_seed{SEED}"

    # The pipeline is coresident by default; OPENVDN_CORESIDENT=0 reloads each stage per request, as before stage 4.
    extra = dict(coresident=False) if os.environ.get("OPENVDN_CORESIDENT") == "0" else {}
    if TRACE:
        extra.update(trace_denoise=True, use_persistent_ccl_buffers=True)
    pipeline = OpenVDNPipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights_dir,
        checkpoint_dir=checkpoint_dir,
        vae_output_type="float",
        **extra,
    )
    request = dict(num_frames=num_frames, height=height, width=width, seed=SEED)
    warm_quietly(pipeline, PROMPT, warmup_steps=3, **request)
    # As `run_warm_generation`, but the cache misses are logged, not asserted: the stages reloaded per request
    # (not coresident) may compile upload programs, and the times are wanted either way.
    profiler = BenchmarkProfiler()
    with profiler("run", iteration=0):
        output = pipeline(
            PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            on_event=profiler_event_callback(profiler, 0),
            **request,
        )
        ttnn.synchronize_device(mesh_device)
    misses = {name: count for name, count in pipeline.last_program_cache_misses.items() if count}
    logger.info(f"measured call program cache misses: {misses or 'none'}")
    log_pipeline_perf(
        profiler,
        label=f"openvdn t2va (warm, {'traced' if TRACE else 'eager'})",
        pipeline=pipeline,
        num_forwards=OPENVDN_NUM_FORWARDS,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=MINIMAX_H3_FPS,
        aspect_ratio=ASPECT_RATIO,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )
    times = {s: profiler.get_duration(s, 0) for s in STAGES if profiler.contains_step(s, 0)}

    frames = to_uint8_frames(output)
    check_output_sanity(frames, num_frames=num_frames, height=height, width=width)
    check_audio_sanity(output.audio, sampling_rate=output.sampling_rate, expected_seconds=num_frames / MINIMAX_H3_FPS)
    if not is_host():
        return
    paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts, stem=stem)
    for i in np.linspace(0, frames.shape[0] - 1, 8).astype(int):
        Image.fromarray(frames[i]).save(artifacts / f"{stem}_frame{i:03d}.png")
    alignment = clip_prompt_alignment(frames, PROMPT)
    summary = {
        "traced": TRACE,
        "times_s": times,
        "program_cache_misses": misses,
        "denoise_per_step_s": times.get("denoising", float("nan")) / OPENVDN_NUM_FORWARDS,
        "clip_mean": alignment["mean"],
        "clip_min": alignment["min"],
        "clip_max": alignment["max"],
        "clip_floor": CLIP_FLOOR,
        "passed": alignment["mean"] >= CLIP_FLOOR,
        "files": {k: str(v) for k, v in paths.items()},
    }
    (artifacts / f"{stem}_perf.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"OpenVDN perf: {json.dumps(summary)}")
    assert alignment["mean"] >= CLIP_FLOOR, f"mean CLIP score {alignment['mean']:.2f} is below the {CLIP_FLOOR} floor"

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end t2va timing, 15 s / 768p, real checkpoint: warm stage table (pipeline.last_timings) after a full
warmup generation. VSA_E2E_MODE=vsa|dense (default vsa), VSA_E2E_STEPS (default 50), VSA_E2E_STREAM_ORDER (default: config). Run through
scripts/run_h3_test.sh with SAFE=1; needs MINIMAX_H3_MODEL_PATH (and TT_DIT_CACHE_DIR for the weight cache)."""

import os

import pytest
from loguru import logger

from ....models.transformers.minimax_h3.vsa_stages_minimax_h3 import MiniMaxH3VSAConfig
from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames, resolve_canvas_size
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from .common import GALAXY_MESHES
from .common_av import run_warm_generation, weights_dir


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES[:1], indirect=["mesh_device", "device_params"])
def test_t2va_15s_768p_e2e_perf(mesh_device, reset_seeds):
    mode = os.environ.get("VSA_E2E_MODE", "vsa")
    steps = int(os.environ.get("VSA_E2E_STEPS", "50"))
    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    height, width = resolve_canvas_size(16, 9)  # 768 x 1344
    num_frames = align_num_frames(round(15.0 * MINIMAX_H3_FPS))

    order = os.environ.get("VSA_E2E_STREAM_ORDER")  # e.g. identity | bstride4.16 (default: the config default)
    vsa_kw = {"stream_order": order} if order else {}
    vsa_config = MiniMaxH3VSAConfig(sparsity=0.9, **vsa_kw) if mode == "vsa" else None
    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=weights, vsa_config=vsa_config)
    output = run_warm_generation(
        pipeline,
        "A red fox trots through fresh snow at dawn, breath steaming in the cold air.",
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=steps,
        seed=0,
    )
    rows = pipeline.last_timings
    total = sum(seconds for _, seconds in rows)
    order = vsa_config.stream_order if vsa_config else "-"
    logger.info(
        f"E2E_PERF mode={mode} steps={steps} frames={output.num_frames} stream_order={order} warm_total={total:.1f}s"
    )
    for label, seconds in rows:
        logger.info(f"E2E_PERF   {label:<20} {seconds:8.1f}s")
    denoise = [s for l, s in rows if "denoise" in l.lower()]
    if denoise:
        logger.info(f"E2E_PERF   denoise per step   {denoise[0] / steps:8.3f}s ({steps} steps)")

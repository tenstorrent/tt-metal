# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""`run_warm_generation` pre-runs each request, so only a first request can expose a warmup gap."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from loguru import logger

from ....pipelines.minimax_h3.packing_ref2va import MiniMaxH3Reference
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....pipelines.minimax_h3.policy import get_num_frames
from .common import MESH_4X8_RING, create_fractal_image
from .common_av import weights_dir

PROMPT = "a red fox trots across a snowy field at dawn, its breath visible in the cold air"
STEPS = 3

REF2VA_MESH = [
    pytest.param(shape, {**params, "l1_small_size": 16384}, id=param.id, marks=param.marks)
    for param in [MESH_4X8_RING]
    for shape, params in [param.values]
]


def _assert_no_misses(pipeline: MiniMaxH3Pipeline, label: str, **request) -> None:
    pipeline(PROMPT, num_inference_steps=STEPS, seed=0, **request)
    misses = pipeline.last_program_cache_misses
    logger.info(f"{label}: padded {pipeline.last_seq_len.padded}, misses {misses}")
    missed = {name: count for name, count in misses.items() if count}
    assert not missed, f"{label}: first request after warmup compiled programs: {missed}"


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_t2va_warmup(mesh_device, reset_seeds):
    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=weights_dir())
    _assert_no_misses(
        pipeline,
        "fl2va 5s 16:9",
        image=create_fractal_image(1024, 768),
        last_image=create_fractal_image(768, 1024),
        num_frames=get_num_frames(5),
    )
    _assert_no_misses(pipeline, "t2va 10s 9:16", num_frames=get_num_frames(10), aspect_ratio=(9, 16))


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(("mesh_device", "device_params"), REF2VA_MESH, indirect=["mesh_device", "device_params"])
def test_ref2va_warmup(mesh_device, reset_seeds):
    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device, weights_dir=weights_dir("transformer_ref"), task="ref2va"
    )
    rate = pipeline.audio_sampling_rate
    video = MiniMaxH3Reference(
        video=np.random.randint(0, 256, (get_num_frames(5), 480, 640, 3), dtype=np.uint8),
        fps=24.0,
        audio=0.1 * torch.randn(2, 5 * rate),
        sample_rate=rate,
    )
    _assert_no_misses(
        pipeline,
        "ref2va image + video with audio",
        references=[MiniMaxH3Reference(image=create_fractal_image(1024, 1024)), video],
        num_frames=get_num_frames(5),
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end t2va generation with VSA on, real checkpoint, 15 s / 768p (R5 completion check).

The bar is completion and sane output statistics, NOT video quality: the base checkpoint carries
no trained gate and no VSA finetuning, so at sparsity 0.9 the clip will look degraded by design --
the FastH3 LoRA that restores quality is a separate ticket (VSA_SCOPE.md non-goals). Reduced step
count for runtime; every per-step shape and program matches the 50-step schedule.
"""

import os

import pytest
import torch
from loguru import logger

from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames, resolve_canvas_size
from ....models.transformers.minimax_h3.vsa_stages_minimax_h3 import MiniMaxH3VSAConfig
from .common import GALAXY_MESHES
from .common_av import run_warm_generation, to_uint8_frames, weights_dir

NUM_INFERENCE_STEPS = 4  # completion check, not quality: same shapes/programs as the full schedule


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES[:1], indirect=["mesh_device", "device_params"])
def test_t2va_vsa_15s_768p(mesh_device, reset_seeds):
    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")

    height, width = resolve_canvas_size(16, 9)  # 768 x 1344
    num_frames = align_num_frames(round(15.0 * MINIMAX_H3_FPS))

    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights,
        vsa_config=MiniMaxH3VSAConfig(sparsity=0.9, k_chunk_blocks=2),
    )

    output = run_warm_generation(
        pipeline,
        "A red fox trots through fresh snow at dawn, breath steaming in the cold air.",
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=0,
    )

    frames = to_uint8_frames(output)
    logger.info(
        f"generated {output.num_frames} frames ({output.video_seconds:.2f}s) + "
        f"{output.audio_seconds:.2f}s audio, padded_len={pipeline.last_padded_len}"
    )
    assert output.num_frames == align_num_frames(num_frames)
    video = torch.from_numpy(frames).float() if not torch.is_tensor(frames) else frames.float()
    assert torch.isfinite(video).all()
    # not a quality gate -- just reject an all-black / saturated / constant output
    assert 2.0 < video.std() and 1.0 < video.mean() < 254.0, (video.mean(), video.std())
    assert torch.isfinite(torch.as_tensor(output.audio)).all()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""S2V pipeline E2E smoke test."""

from __future__ import annotations

import os

import numpy as np
import PIL
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.wan.pipeline_wan_s2v import WanPipelineS2V
from models.tt_dit.utils.video import export_to_video_with_audio

from .....utils.test import line_params, ring_params

# Inputs are expected at the repo root (same pattern as test_pipeline_wan_i2v.py).
# Override with env vars when needed.
_REF_IMAGE_PATH = os.environ.get("S2V_REF_IMAGE", "./prompt_image.png")
_AUDIO_PATH = os.environ.get("S2V_AUDIO", "./prompt_audio.wav")
_PROMPT = os.environ.get("S2V_PROMPT", "a person is talking")
_NEGATIVE_PROMPT = (
    # s2v_14B-specific negative prompt from reference wan_s2v_14B.py:55.
    "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    (
        "mesh_device",
        "mesh_shape",
        "sp_axis",
        "tp_axis",
        "num_links",
        "dynamic_load",
        "device_params",
        "topology",
        "is_fsdp",
        "sdpa_t_fracture_w_only",
    ),
    [
        # BH Loud Box (2x4, 8 chips) — sp_factor=4, tp_factor=2.
        pytest.param(
            (2, 4),
            (2, 4),
            1,
            0,
            2,
            False,
            line_params,
            ttnn.Topology.Linear,
            False,
            False,
            id="bh_2x4sp1tp0",
        ),
        # BH Galaxy (4x8, 32 chips) — sp_factor=8, tp_factor=4. Ring fabric.
        pytest.param(
            (4, 8),
            (4, 8),
            1,
            0,
            2,
            False,
            ring_params,
            ttnn.Topology.Ring,
            False,
            False,
            id="bh_4x8sp1tp0",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "width, height",
    [
        (832, 480),
        (1280, 720),
    ],
    ids=["resolution_480p", "resolution_720p"],
)
def test_pipeline_inference_s2v(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    width,
    height,
    is_fsdp,
    sdpa_t_fracture_w_only,
):
    """End-to-end S2V inference: encoder + diffusion + VAE decode + video export."""
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    ref_image = PIL.Image.open(_REF_IMAGE_PATH)

    # ``num_clips`` matches reference per-clip ``infer_frames=80`` (speech2video.py:404).
    # Unset env → ``None`` so pipeline uses ``num_repeat`` (one clip per 5 s of audio).
    # Pass ``S2V_CLIPS=1`` for a smoke-test budget.
    _clips_env = os.environ.get("S2V_CLIPS")
    num_clips = int(_clips_env) if _clips_env is not None else None
    num_inference_steps = int(os.environ.get("S2V_STEPS", 40))
    guidance_scale = 4.5  # reference wan_s2v_14B.py:59

    pipeline = WanPipelineS2V.create_pipeline(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        sdpa_t_fracture_w_only=sdpa_t_fracture_w_only,
        height=height,
        width=width,
        num_frames=81,
    )

    logger.info(f"Running S2V inference: {height}x{width}, {num_clips} clips, {num_inference_steps} steps")
    with torch.no_grad():
        result = pipeline(
            prompt=_PROMPT,
            image_prompt=ref_image,
            audio_prompt=_AUDIO_PATH,
            negative_prompt=_NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_clips=num_clips,
            num_inference_steps=num_inference_steps,
            seed=0,
            guidance_scale=guidance_scale,
            output_type="uint8",
        )

    frames = result.frames if hasattr(result, "frames") else (result[0] if isinstance(result, tuple) else result)
    if isinstance(frames, np.ndarray):
        logger.info(f"output shape: {frames.shape}, range: [{frames.min():.3f}, {frames.max():.3f}]")
    elif isinstance(frames, torch.Tensor):
        logger.info(
            f"output shape: {tuple(frames.shape)}, range: [{frames.min().item():.3f}, {frames.max().item():.3f}]"
        )

    output_path = f"wan_s2v_{width}x{height}.mp4"
    export_to_video_with_audio(frames[0], output_path, audio_path=_AUDIO_PATH, fps=16)
    logger.info(f"Saved video with audio: {output_path}")

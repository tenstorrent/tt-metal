# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import PIL
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.wan.pipeline_wan_s2v import WanPipelineS2V

from ....utils.test import line_params

# Inputs are expected at the repo root (same pattern as test_pipeline_wan_i2v.py).
# Override with env vars when needed.
_REF_IMAGE_PATH = os.environ.get("S2V_REF_IMAGE", "./prompt_image.png")
_AUDIO_PATH = os.environ.get("S2V_AUDIO", "./prompt_audio.wav")
_NEGATIVE_PROMPT = (
    # s2v_14B-specific negative prompt from wan_s2v_14B.py:55 (overrides
    # the shared_config default used by T2V/I2V).
    "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
_PROMPT = "a person is talking"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp, sdpa_t_fracture_w_only",
    [
        # BH Loud Box (2x4, 8 chips) — sp_factor=4, tp_factor=2.
        [(2, 4), (2, 4), 1, 0, 2, False, line_params, ttnn.Topology.Linear, False, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "width, height",
    [
        (832, 480),
    ],
    ids=[
        "resolution_480p",
    ],
)
def test_pipeline_inference(
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
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    ref_image = PIL.Image.open(_REF_IMAGE_PATH)

    # ``num_frames=81`` is the production target. (81-1)/4+1 = 21 latent frames,
    # the natural ``4k+1`` size for the WAN VAE temporal decoder. Reference's
    # ``infer_frames=80`` would round to 81 in our pipeline as well — see
    # PLAN_WAN_S2V_CLEANUP.md for why 80 is blocked on a ttnn ``binary_ng`` bug.
    num_frames = 81
    num_inference_steps = int(os.environ.get("S2V_STEPS", 40))
    # Reference s2v_14B uses sample_guide_scale=4.5 (wan_s2v_14B.py:59).
    guidance_scale = 4.5
    guidance_scale_2 = 4.5

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
        num_frames=num_frames,
    )

    logger.info(f"Running S2V inference: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")
    with torch.no_grad():
        result = pipeline(
            prompt=_PROMPT,
            image_prompt=ref_image,
            audio_prompt=_AUDIO_PATH,
            negative_prompt=_NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=0,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
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
    WanPipelineS2V.export(frames[0], output_path, audio_path=_AUDIO_PATH, fps=16)
    logger.info(f"Saved video with audio: {output_path}")

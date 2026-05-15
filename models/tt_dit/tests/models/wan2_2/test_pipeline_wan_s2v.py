# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import PIL
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.wan.pipeline_wan_s2v import WanPipelineS2V

from ....utils.test import line_params

# Production reference example pair (wan2_2_ref/README.md §S2V):
# --image examples/i2v_input.JPG, --audio examples/talk.wav. Reference image
# + audio are staged at the repo root for the test.
_REF_IMAGE_PATH = "./ref_image.png"
_AUDIO_PATH = "./prompt_audio.wav"
_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
_PROMPT = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp, sdpa_t_fracture_w_only",
    [
        # BH (linear) on 4x8 — the production S2V target.
        [(4, 8), (4, 8), 1, 0, 2, False, line_params, ttnn.Topology.Linear, False, False],
    ],
    ids=["bh_4x8sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "width, height",
    [
        (832, 480),
        (1280, 720),
    ],
    ids=[
        "resolution_480p",
        "resolution_720p",
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

    num_frames = 81
    num_inference_steps = 40
    guidance_scale = 5.0
    guidance_scale_2 = 5.0

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

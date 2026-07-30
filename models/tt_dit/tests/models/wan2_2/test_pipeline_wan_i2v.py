# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import numpy as np
import PIL
import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipelineConfig
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt, WanPipelineI2V

from ....utils.test import line_params_req_exact_devices, ring_params_req_exact_devices, skip_if_unsupported_num_links


def create_fractal_image(width: int, height: int) -> Image.Image:
    c = np.linspace(-2.0, 1.0, width)[None, :] + 1j * np.linspace(-1.5, 1.5, height)[:, None]
    z = np.zeros_like(c)
    img = np.zeros(c.shape, dtype=np.uint8)
    for i in range(32):
        z = z * z + c
        img[(img == 0) & (np.abs(z) > 2)] = 255 - 8 * i
    return Image.fromarray(np.dstack((img, np.roll(img, width // 10, 1), np.roll(img, height // 10, 0))), "RGB")


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 2), (2, 2), 0, 1, 2, False, line_params_req_exact_devices, ttnn.Topology.Linear, True],
        [(2, 4), (2, 4), 0, 1, 1, True, line_params_req_exact_devices, ttnn.Topology.Linear, True],
        [(2, 4), (2, 4), 1, 0, 2, True, line_params_req_exact_devices, ttnn.Topology.Linear, False],
        [(4, 8), (4, 8), 1, 0, 4, False, ring_params_req_exact_devices, ttnn.Topology.Ring, True],
        [(4, 8), (4, 8), 1, 0, 2, False, line_params_req_exact_devices, ttnn.Topology.Linear, False],
    ],
    ids=[
        "2x2sp0tp1nl2_linear_is_fsdp1",
        "2x4sp0tp1nl1_linear_is_fsdp1",
        "2x4sp1tp0nl2_linear_is_fsdp0",  # BH on 2x4 with dynamic_load to avoid init-time DRAM OOM
        "4x8sp1tp0nl4_ring_is_fsdp1",  # WH (ring) on 4x8
        "4x8sp1tp0nl2_linear_is_fsdp0",  # BH (linear) on 4x8
    ],
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
    no_prompt,
):
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    skip_if_unsupported_num_links(mesh_device, num_links)

    if no_prompt:
        test_image = create_fractal_image(width, height)
    else:
        test_image = PIL.Image.open("./prompt_image.png")
    image_prompt = [ImagePrompt(image=test_image, frame_pos=0)]
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    num_frames = 81
    num_inference_steps = 40
    guidance_scale = 3.5
    guidance_scale_2 = 3.5

    h_factor = tuple(mesh_device.shape)[tp_axis]
    w_factor = tuple(mesh_device.shape)[sp_axis]
    parallel_config = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(w_factor, sp_axis), tp=(h_factor, tp_axis))
    vae_parallel_config = VaeHWParallelConfig.from_tuples(height=(h_factor, tp_axis), width=(w_factor, sp_axis))
    encoder_parallel_config = EncoderParallelConfig.from_tuple((h_factor, tp_axis))

    pipeline = WanPipelineI2V(
        device=mesh_device,
        config=WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            dit_parallel_config=parallel_config,
            vae_parallel_config=vae_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            num_links=num_links,
            dynamic_load=dynamic_load,
            topology=topology,
            is_fsdp=is_fsdp,
            checkpoint_name="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            model_type="i2v",
            height=height,
            width=width,
            num_frames=num_frames,
        ),
    )

    prompt = "The cat in the hat runs up the hill to the house."

    def run(*, prompt, number, seed):
        logger.info(f"Running inference with prompt: '{prompt}'")
        logger.info(f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

        with torch.no_grad():
            frames = pipeline(
                prompts=[prompt],
                negative_prompts=[negative_prompt],
                image_prompt=image_prompt,
                num_inference_steps=num_inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                output_type="uint8",
            )

        logger.info(f"Inference completed successfully")
        logger.info(f"  Output shape: {frames.shape if hasattr(frames, 'shape') else 'Unknown'}")
        logger.info(f"  Output type: {type(frames)}")

        if isinstance(frames, np.ndarray):
            logger.info(f"  Video data range: [{frames.min():.3f}, {frames.max():.3f}]")
        elif isinstance(frames, torch.Tensor):
            logger.info(f"  Video data range: [{frames.min().item():.3f}, {frames.max().item():.3f}]")

        # Remove batch dimension
        frames = frames[0]
        output_filename = f"wan_i2v_{width}x{height}_{number}.mp4"
        try:
            from models.tt_dit.utils.video import export_to_video

            export_to_video(frames, output_filename, fps=16)
            logger.info(f"Saved video to: {output_filename}")
        except ImportError:
            logger.info("Could not export video - imageio_ffmpeg not available")

    if no_prompt:
        run(prompt=prompt, number=0, seed=42)
    else:
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, number=i, seed=i)

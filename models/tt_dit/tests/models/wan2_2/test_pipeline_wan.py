# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import numpy as np
import pytest
import torch
from diffusers.utils import export_to_video
from loguru import logger

import ttnn
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline

from ....utils.test import line_params, ring_params


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 2), (2, 2), 0, 1, 2, False, line_params, ttnn.Topology.Linear, True],
        [(2, 4), (2, 4), 0, 1, 1, True, line_params, ttnn.Topology.Linear, True],
        # WH (ring) on 4x8
        [(4, 8), (4, 8), 1, 0, 4, False, ring_params, ttnn.Topology.Ring, True],
        # BH (linear) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, line_params, ttnn.Topology.Linear, False],
    ],
    ids=[
        "2x2sp0tp1",
        "2x4sp0tp1",
        "wh_4x8sp1tp0",
        "bh_4x8sp1tp0",
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

    num_frames = 81
    num_inference_steps = 40

    pipeline = WanPipeline.create_pipeline(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        checkpoint_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )

    prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

    def run(*, prompt, number, seed):
        logger.info(f"Running inference with prompt: '{prompt}'")
        logger.info(f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                seed=seed,
                guidance_scale=4.0,
                guidance_scale_2=3.0,
            )

        if hasattr(result, "frames"):
            frames = result.frames
        else:
            frames = result[0] if isinstance(result, tuple) else result

        logger.info(f"Inference completed successfully")
        logger.info(f"  Output shape: {frames.shape if hasattr(frames, 'shape') else 'Unknown'}")
        logger.info(f"  Output type: {type(frames)}")

        if isinstance(frames, np.ndarray):
            logger.info(f"  Video data range: [{frames.min():.3f}, {frames.max():.3f}]")
        elif isinstance(frames, torch.Tensor):
            logger.info(f"  Video data range: [{frames.min().item():.3f}, {frames.max().item():.3f}]")

        # Remove batch dimension
        frames = frames[0]
        output_filename = f"wan_t2v_{width}x{height}_{number}.mp4"
        try:
            export_to_video(frames, output_filename, fps=16)
            logger.info(f"Saved video to: {output_filename}")
        except AttributeError as e:
            logger.info(f"AttributeError: {e}")

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

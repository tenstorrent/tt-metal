# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from models.experimental.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V
from models.experimental.tt_dit.parallel.config import DiTParallelConfig, VaeHWParallelConfig, ParallelFactor
from diffusers.utils import export_to_video
import pytest
import ttnn
from ....utils.test import line_params, ring_params
import PIL


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology",
    [
        [(2, 4), (2, 4), 0, 1, 1, True, line_params, ttnn.Topology.Linear],
        # WH (ring) on 4x8
        [(4, 8), (4, 8), 1, 0, 4, False, ring_params, ttnn.Topology.Ring],
        # BH (linear) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, line_params, ttnn.Topology.Linear],
    ],
    ids=[
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
        # (1280, 720),
    ],
    ids=[
        "resolution_480p",
        # "resolution_720p",
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
):
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    vae_parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    # Test parameters
    prompt = "The cat in the hat runs up the hill to the house."
    image = PIL.Image.open("./prompt_image.png").resize((width, height))
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    num_frames = 81
    num_inference_steps = 40
    guidance_scale = 3.5
    guidance_scale_2 = 3.5

    print(f"Running inference with prompt: '{prompt}'")
    print(f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

    pipeline = WanPipelineI2V(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        vae_parallel_config=vae_parallel_config,
        num_links=num_links,
        use_cache=True,
        boundary_ratio=0.900,
        dynamic_load=dynamic_load,
        topology=topology,
    )

    # Run inference
    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
        )

    # Check output
    if hasattr(result, "frames"):
        frames = result.frames
    else:
        frames = result[0] if isinstance(result, tuple) else result

    print(f"✓ Inference completed successfully")
    print(f"  Output shape: {frames.shape if hasattr(frames, 'shape') else 'Unknown'}")
    print(f"  Output type: {type(frames)}")

    # Basic validation
    if isinstance(frames, np.ndarray):
        print(f"  Video data range: [{frames.min():.3f}, {frames.max():.3f}]")
    elif isinstance(frames, torch.Tensor):
        print(f"  Video data range: [{frames.min().item():.3f}, {frames.max().item():.3f}]")

    # Save video using diffusers utility
    # Remove batch dimension
    frames = frames[0]

    try:
        export_to_video(frames, "wan_output_video.mp4", fps=16)
    except AttributeError as e:
        print(f"AttributeError: {e}")
    print("✓ Saved video to: wan_output_video.mp4")

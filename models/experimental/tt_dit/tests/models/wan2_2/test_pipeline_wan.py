# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from models.experimental.tt_dit.pipelines.wan.pipeline_wan import WanPipeline
from models.experimental.tt_dit.parallel.config import DiTParallelConfig, VaeHWParallelConfig, ParallelFactor
from diffusers.utils import export_to_video
import pytest
import ttnn
from ....utils.test import line_params, ring_params


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 2), (2, 2), 0, 1, 2, False, line_params, ttnn.Topology.Linear, False],
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
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
    )
    # Test parameters
    prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

    num_frames = 81
    num_inference_steps = 40

    print(f"Running inference with prompt: '{prompt}'")
    print(f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

    pipeline = WanPipeline(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        vae_parallel_config=vae_parallel_config,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
    )

    # Run inference
    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
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

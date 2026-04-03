#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Trace actual conv3d input shapes from a WAN 2.2 VAE decoder run.

Runs the decoder on a 1x1 mesh with random weights and captures the shape
logging from WanCausalConv3d.forward and WanConv2d.forward.

Usage:
    source python_env/bin/activate && export PYTHONPATH=$(pwd)
    python models/tt_dit/tests/models/wan2_2/trace_conv3d_shapes.py

Output: prints all conv3d input shapes for each production config.
"""

import math
import time

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan as TorchAutoencoderKLWan
from loguru import logger

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanDecoder
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import conv_pad_in_channels
from models.tt_dit.utils.tensor import typed_tensor_2dshard

# Production configs: (name, resolution_HxW, h_factor, w_factor, num_frames)
PRODUCTION_CONFIGS = [
    ("bh_4x32_720p", (720, 1280), 4, 32, 81),
    ("bh_4x8_720p", (720, 1280), 4, 8, 81),
    ("bh_4x32_480p", (480, 832), 4, 32, 81),
    ("bh_4x8_480p", (480, 832), 4, 8, 81),
    ("bh_2x4_480p", (480, 832), 2, 4, 81),
]


def trace_shapes_for_config(mesh_device, config_name, resolution, h_factor, w_factor, num_frames, use_cache):
    """Run decoder and capture conv3d shapes from logger output."""
    H_out, W_out = resolution
    vae_temporal_scale = 4
    vae_spatial_scale = 8

    # Latent dimensions
    latent_T = (num_frames - 1) // vae_temporal_scale + 1
    latent_H = H_out // vae_spatial_scale
    latent_W = W_out // vae_spatial_scale
    z_dim = 16

    logger.info(f"\n{'='*80}")
    logger.info(f"CONFIG: {config_name} | use_cache={use_cache}")
    logger.info(f"  Resolution: {H_out}x{W_out}, Frames: {num_frames}")
    logger.info(f"  h_factor={h_factor}, w_factor={w_factor}")
    logger.info(f"  Latent: T={latent_T}, H={latent_H}, W={latent_W}, C={z_dim}")

    # Per-device dims after sharding (simulated on 1x1 mesh)
    H_padded = math.ceil(latent_H / h_factor) * h_factor
    H_per_device = H_padded // h_factor
    W_per_device = latent_W // w_factor
    logger.info(f"  Per-device latent: H={H_per_device}, W={W_per_device} (H_padded={H_padded})")

    # Setup parallel config (1x1 mesh, but factor tells the model what to expect)
    h_axis = 0
    w_axis = 1
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=1, mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=1, mesh_axis=w_axis),
    )

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=1)

    # Build decoder with random weights
    # Create torch model for random weights
    torch_model = TorchAutoencoderKLWan(
        base_dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        out_channels=3,
        is_residual=False,
    )
    torch_model.eval()

    tt_model = WanDecoder(
        base_dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        out_channels=3,
        is_residual=False,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        dtype=ttnn.DataType.BFLOAT16,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # Create input tensor with simulated per-device shape
    # Shape: (B, T, H_per_device, W_per_device, padded_z_dim)
    B = 1
    torch_input = torch.randn(B, z_dim, latent_T, H_per_device, W_per_device, dtype=torch.float32) * 0.1
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # BCTHW -> BTHWC
    tt_input = conv_pad_in_channels(tt_input)  # pad C to alignment

    logical_h = H_per_device * h_factor  # global logical height for masking

    tt_input_tensor = typed_tensor_2dshard(
        tt_input,
        mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_mapping={h_axis: 2, w_axis: 3},
        dtype=ttnn.bfloat16,
    )

    logger.info(f"  Running decoder... (input shape: {tt_input.shape})")
    start = time.time()
    try:
        tt_output, new_logical_h = tt_model(tt_input_tensor, logical_h, use_cache=use_cache)
        elapsed = time.time() - start
        logger.info(f"  Done in {elapsed:.1f}s")
    except Exception as e:
        logger.error(f"  FAILED: {e}")

    # Deallocate
    del tt_model, torch_model


def main():
    logger.info("Opening 1x1 mesh device...")
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
    )
    mesh_device.enable_program_cache()

    try:
        for config_name, resolution, h_factor, w_factor, num_frames in PRODUCTION_CONFIGS:
            # Uncached run
            trace_shapes_for_config(
                mesh_device, config_name, resolution, h_factor, w_factor, num_frames, use_cache=False
            )
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()

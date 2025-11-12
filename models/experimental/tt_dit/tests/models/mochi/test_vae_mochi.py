# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import os
import ttnn

from ....utils.check import assert_quality
from ....models.vae.vae_mochi import (
    Conv1x1 as TtConv1x1,
    ResBlock as TtResBlock,
    CausalUpsampleBlock as TtCausalUpsampleBlock,
    MochiVAEDecoder as TtDecoder,
)
from ....parallel.manager import CCLManager
from ....parallel.config import MochiVAEParallelConfig, ParallelFactor
from diffusers.models.autoencoders.autoencoder_kl_mochi import MochiResnetBlock3D, MochiUpBlock3D, MochiDecoder3D

from loguru import logger


def get_padded_size(numerator, denominator):
    return ((numerator + denominator - 1) // denominator) * denominator


# Custom pytest mark for shared VAE device configuration
def vae_device_config(func):
    """Decorator to apply standard VAE device configuration to tests"""
    func = pytest.mark.parametrize(
        "mesh_device",
        [
            {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
                os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
            )
        ],
        indirect=True,
    )(func)
    func = pytest.mark.parametrize(
        "device_params",
        [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 20000000}],
        indirect=True,
    )(func)
    return func


class Conv3d1x1(nn.Conv3d):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=(1, 1, 1), bias=bias)


def create_random_conv3d_models(mesh_device, in_channels, out_channels, bias=True):
    """Initialize both reference Conv3d and TT models."""
    # Create reference model
    reference_model = Conv3d1x1(in_channels, out_channels, bias=bias)

    # Create TT model
    tt_model = TtConv1x1(
        mesh_device=mesh_device,
        in_channels=in_channels,
        out_channels=out_channels,
        bias=bias,
        torch_ref=reference_model,
    )

    return reference_model, tt_model


@pytest.mark.parametrize(
    "N, C_in, C_out, T, H, W",
    [
        (1, 12, 768, 28, 60, 106),
    ],
    ids=["large_latent"],
)
@vae_device_config
def test_tt_conv3d_1x1x1(mesh_device, N, C_in, C_out, T, H, W, reset_seeds):
    """Test forward pass of TtConv1x1 against Conv3d with 1x1x1 kernel."""
    reference_model, tt_model = create_random_conv3d_models(mesh_device, C_in, C_out)

    if mesh_device.shape[0] == 1:
        w_parallel_factor = 1
    else:
        w_parallel_factor = 2

    vae_parallel_config = MochiVAEParallelConfig(
        time_parallel=ParallelFactor(factor=mesh_device.shape[1], mesh_axis=1),
        w_parallel=ParallelFactor(factor=w_parallel_factor, mesh_axis=0),
        h_parallel=ParallelFactor(factor=mesh_device.shape[0] // w_parallel_factor, mesh_axis=0),
    )
    assert vae_parallel_config.h_parallel.factor * vae_parallel_config.w_parallel.factor == mesh_device.shape[0]
    assert vae_parallel_config.h_parallel.mesh_axis == vae_parallel_config.w_parallel.mesh_axis

    # Create input tensor
    torch_input = torch.randn(N, C_in, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]

    num_devices_T = mesh_device.shape[vae_parallel_config.time_parallel.mesh_axis]
    if T % num_devices_T:
        padded_T = get_padded_size(T, num_devices_T)
        T_padding = padded_T - T
        tt_input = torch.nn.functional.pad(tt_input, pad=(0, 0, 0, 0, 0, 0, 0, T_padding))
    else:
        padded_T = T
    num_devices_W = vae_parallel_config.w_parallel.factor
    if W % num_devices_W:
        padded_W = get_padded_size(W, num_devices_W)
        W_padding = padded_W - W
        tt_input = torch.nn.functional.pad(tt_input, pad=(0, 0, 0, W_padding))
    else:
        padded_W = W
    num_devices_H = vae_parallel_config.h_parallel.factor
    if H % num_devices_H:
        padded_H = get_padded_size(H, num_devices_H)
        H_padding = padded_H - H
        tt_input = torch.nn.functional.pad(tt_input, pad=(0, 0, 0, 0, 0, H_padding))
    else:
        padded_H = H

    tt_input = torch.reshape(
        tt_input,
        (N, padded_T, num_devices_H, padded_H // num_devices_H, num_devices_W, padded_W // num_devices_W, C_in),
    )
    tt_input = tt_input.permute(0, 1, 2, 4, 3, 5, 6)
    tt_input = torch.reshape(
        tt_input,
        (N, padded_T, num_devices_H * num_devices_W, padded_H // num_devices_H, padded_W // num_devices_W, C_in),
    )

    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, 1]),
    )
    tt_input = ttnn.squeeze(tt_input, 2)

    logger.info("Run TtConv1x1 forward (Conv3d mode)")
    tt_output = tt_model(tt_input)
    logger.info("End TtResBlock forward")
    tt_output = ttnn.unsqueeze(tt_output, 2)

    # Convert TT output to torch tensor
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, 0]),
    )

    tt_output_torch = torch.reshape(
        tt_output_torch,
        (N, padded_T, num_devices_H, num_devices_W, padded_H // num_devices_H, padded_W // num_devices_W, C_out),
    )
    tt_output_torch = tt_output_torch.permute(0, 1, 2, 4, 3, 5, 6)
    tt_output_torch = torch.reshape(tt_output_torch, (N, padded_T, padded_H, padded_W, C_out))

    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]
    tt_output_torch = tt_output_torch[0:N, 0:C_out, 0:T, 0:H, 0:W]

    # Get reference output
    with torch.no_grad():
        ref_output = reference_model(torch_input)

    assert_quality(ref_output, tt_output_torch, pcc=0.999_500)


resblock_args = {
    "affine": True,
    "attn_block": None,
    "causal": True,
    "prune_bottleneck": False,
    "padding_mode": "replicate",
    "bias": True,
}


def create_random_resblock_models(mesh_device, parallel_config, ccl_manager, in_channels, nonlinearity):
    """Initialize both reference and TT models."""
    # Create reference model
    reference_model = MochiResnetBlock3D(in_channels=in_channels, act_fn=nonlinearity)

    # Create TT model
    tt_model = TtResBlock(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        torch_ref=reference_model,
    )

    return reference_model, tt_model


@torch.no_grad()
@pytest.mark.parametrize(
    "N, C, T, H, W",
    [
        # small latent
        (1, 768, 28, 40, 50),
        (1, 512, 84, 80, 100),
        (1, 256, 168, 160, 200),
        (1, 128, 168, 320, 400),
        # large latent
        (1, 768, 28, 60, 106),
        (1, 512, 84, 120, 212),
        (1, 256, 168, 240, 424),
        (1, 128, 168, 480, 848),
    ],
    ids=["s768", "s512", "s256", "s128", "l768", "l512", "l256", "l128"],
)
@pytest.mark.parametrize("num_links", [4, 1], ids=["4links", "1link"])
@vae_device_config
def test_tt_resblock_forward(mesh_device, N, C, T, H, W, reset_seeds, num_links):
    """Test complete forward pass of TtResBlock."""
    block_args = resblock_args.copy()
    block_args["channels"] = C
    block_args["nonlinearity"] = "silu"

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)

    if mesh_device.shape[0] == 1:
        w_parallel_factor = 1
    else:
        w_parallel_factor = 2

    vae_parallel_config = MochiVAEParallelConfig(
        time_parallel=ParallelFactor(factor=mesh_device.shape[1], mesh_axis=1),
        w_parallel=ParallelFactor(factor=w_parallel_factor, mesh_axis=0),
        h_parallel=ParallelFactor(factor=mesh_device.shape[0] // w_parallel_factor, mesh_axis=0),
    )
    assert vae_parallel_config.h_parallel.factor * vae_parallel_config.w_parallel.factor == mesh_device.shape[0]
    assert vae_parallel_config.h_parallel.mesh_axis == vae_parallel_config.w_parallel.mesh_axis

    reference_model, tt_model = create_random_resblock_models(
        mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
        in_channels=block_args["channels"],
        nonlinearity=block_args["nonlinearity"],
    )

    # Create input tensor
    torch_input = torch.randn(N, C, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]

    num_devices_T = mesh_device.shape[vae_parallel_config.time_parallel.mesh_axis]
    if T % num_devices_T:
        padded_T = get_padded_size(T, num_devices_T)
        T_padding = padded_T - T
        tt_input = torch.nn.functional.pad(tt_input, pad=(0, 0, 0, 0, 0, 0, 0, T_padding))
    else:
        padded_T = T
    num_devices_W = vae_parallel_config.w_parallel.factor
    if W % num_devices_W:
        padded_W = get_padded_size(W, num_devices_W)
        W_padding = padded_W - W
        tt_input = torch.nn.functional.pad(tt_input, pad=(0, 0, 0, W_padding))
    else:
        padded_W = W
    num_devices_H = vae_parallel_config.h_parallel.factor
    if H % num_devices_H:
        padded_H = get_padded_size(H, num_devices_H)
        H_padding = padded_H - H
        tt_input = torch.nn.functional.pad(tt_input, pad=(0, 0, 0, 0, 0, H_padding))
    else:
        padded_H = H

    tt_input = torch.reshape(
        tt_input, (N, padded_T, num_devices_H, padded_H // num_devices_H, num_devices_W, padded_W // num_devices_W, C)
    )
    tt_input = tt_input.permute(0, 1, 2, 4, 3, 5, 6)
    tt_input = torch.reshape(
        tt_input, (N, padded_T, num_devices_H * num_devices_W, padded_H // num_devices_H, padded_W // num_devices_W, C)
    )

    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, 1]),
    )
    tt_input = ttnn.squeeze(tt_input, 2)

    logger.info(f"TT input shape: {tt_input.shape}")
    logger.info("Run TtResBlock forward")
    tt_output = tt_model(tt_input)
    logger.info("End TtResBlock forward")
    tt_output = ttnn.unsqueeze(tt_output, 2)

    # Convert TT output to torch tensor
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, 0]),
    )

    tt_output_torch = torch.reshape(
        tt_output_torch,
        (N, padded_T, num_devices_H, num_devices_W, padded_H // num_devices_H, padded_W // num_devices_W, C),
    )
    tt_output_torch = tt_output_torch.permute(0, 1, 2, 4, 3, 5, 6)
    tt_output_torch = torch.reshape(tt_output_torch, (N, padded_T, padded_H, padded_W, C))

    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]
    tt_output_torch = tt_output_torch[0:N, 0:C, 0:T, 0:H, 0:W]

    # Get reference output
    logger.info("Run RefResBlock forward")
    with torch.no_grad():
        ref_output = reference_model(torch_input)[0]
    logger.info("End RefResBlock forward")

    logger.info("assert quality")
    for i in range(T):
        ref_output_slice = ref_output[:, :, i, :, :]
        tt_output_torch_slice = tt_output_torch[:, :, i, :, :]
        assert_quality(ref_output_slice, tt_output_torch_slice, pcc=0.9998)


def create_random_causalupsampleblock_models(
    mesh_device,
    in_channels,
    out_channels,
    num_layers,
    temporal_expansion,
    spatial_expansion,
    temporal_offset,
    parallel_config,
    ccl_manager,
):
    """Initialize both reference and TT models with optional real weights."""
    # Create reference model
    reference_model = MochiUpBlock3D(
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        temporal_expansion=temporal_expansion,
        spatial_expansion=spatial_expansion,
    )

    # Create TT model with same weights
    tt_model = TtCausalUpsampleBlock(
        mesh_device=mesh_device,
        in_channels=in_channels,
        out_channels=out_channels,
        torch_ref=reference_model,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        num_res_blocks=num_layers,
        temporal_expansion=temporal_expansion,
        spatial_expansion=spatial_expansion,
        temporal_offset=temporal_offset,
    )

    return reference_model, tt_model


# Test case configurations for different input sizes
@pytest.mark.parametrize(
    "config",
    [
        # large latent
        # First upsample block (768->512)
        {
            "name": "block1_768-512",
            "in_channels": 768,
            "out_channels": 512,
            "num_res_blocks": 6,
            "temporal_expansion": 3,
            "spatial_expansion": 2,
            "input_shape": [1, 768, 28, 60, 106],
            "expected_output_shape": (1, 512, 84, 120, 212),
        },
        # Second upsample block (512->256)
        {
            "name": "block2_512-256",
            "in_channels": 512,
            "out_channels": 256,
            "num_res_blocks": 4,
            "temporal_expansion": 2,
            "spatial_expansion": 2,
            "input_shape": [1, 512, 84, 120, 212],
            "expected_output_shape": (1, 256, 168, 240, 424),
        },
        # Third upsample block (256->128)
        {
            "name": "block3_256-128",
            "in_channels": 256,
            "out_channels": 128,
            "num_res_blocks": 3,
            "temporal_expansion": 1,
            "spatial_expansion": 2,
            "input_shape": [1, 256, 168, 240, 424],
            "expected_output_shape": (1, 128, 168, 480, 848),
        },
        # small latent
        # First upsample block (768->512)
        {
            "name": "block1_768-512",
            "in_channels": 768,
            "out_channels": 512,
            "num_res_blocks": 6,
            "temporal_expansion": 3,
            "spatial_expansion": 2,
            "input_shape": [1, 768, 28, 40, 50],
            "expected_output_shape": (1, 512, 84, 80, 100),
        },
        # Second upsample block (512->256)
        {
            "name": "block2_512-256",
            "in_channels": 512,
            "out_channels": 256,
            "num_res_blocks": 4,
            "temporal_expansion": 2,
            "spatial_expansion": 2,
            "input_shape": [1, 512, 84, 80, 100],
            "expected_output_shape": (1, 256, 168, 160, 200),
        },
        # Third upsample block (256->128)
        {
            "name": "block3_256-128",
            "in_channels": 256,
            "out_channels": 128,
            "num_res_blocks": 3,
            "temporal_expansion": 1,
            "spatial_expansion": 2,
            "input_shape": [1, 256, 168, 160, 200],
            "expected_output_shape": (1, 128, 168, 320, 400),
        },
    ],
    ids=["l768", "l512", "l256", "s768", "s512", "s256"],
)
@pytest.mark.parametrize("num_links", [4, 1], ids=["4links", "1link"])
@vae_device_config
def test_tt_upsample_forward(mesh_device, config, reset_seeds, num_links):
    """Test TtCausalUpsampleBlock against reference implementation."""
    in_channels = config["in_channels"]
    out_channels = config["out_channels"]
    num_res_blocks = config["num_res_blocks"]
    temporal_expansion = config["temporal_expansion"]
    spatial_expansion = config["spatial_expansion"]
    input_shape = config["input_shape"]
    expected_output_shape = config["expected_output_shape"]
    temporal_offset = 0  # temporal_expansion-1
    N, C, T, H, W = input_shape

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)

    if mesh_device.shape[0] == 1:
        w_parallel_factor = 1
    else:
        w_parallel_factor = 2

    vae_parallel_config = MochiVAEParallelConfig(
        time_parallel=ParallelFactor(factor=mesh_device.shape[1], mesh_axis=1),
        w_parallel=ParallelFactor(factor=w_parallel_factor, mesh_axis=0),
        h_parallel=ParallelFactor(factor=mesh_device.shape[0] // w_parallel_factor, mesh_axis=0),
    )
    assert vae_parallel_config.h_parallel.factor * vae_parallel_config.w_parallel.factor == mesh_device.shape[0]
    assert vae_parallel_config.h_parallel.mesh_axis == vae_parallel_config.w_parallel.mesh_axis

    reference_model, tt_model = create_random_causalupsampleblock_models(
        mesh_device,
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_res_blocks,
        temporal_expansion=temporal_expansion,
        spatial_expansion=spatial_expansion,
        temporal_offset=temporal_offset,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
    )

    # Create input tensor
    torch_input = torch.randn(N, C, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]

    num_devices_T = mesh_device.shape[vae_parallel_config.time_parallel.mesh_axis]
    if T % num_devices_T:
        padded_T = get_padded_size(T, num_devices_T)
        T_padding = padded_T - T
        tt_input = torch.nn.functional.pad(tt_input, pad=(0, 0, 0, 0, 0, 0, 0, T_padding))
    else:
        padded_T = T
    num_devices_W = vae_parallel_config.w_parallel.factor
    if W % num_devices_W:
        padded_W = get_padded_size(W, num_devices_W)
        W_padding = padded_W - W
        tt_input = torch.nn.functional.pad(tt_input, pad=(0, 0, 0, W_padding))
    else:
        padded_W = W
    num_devices_H = vae_parallel_config.h_parallel.factor
    if H % num_devices_H:
        padded_H = get_padded_size(H, num_devices_H)
        H_padding = padded_H - H
        tt_input = torch.nn.functional.pad(tt_input, pad=(0, 0, 0, 0, 0, H_padding))
    else:
        padded_H = H

    tt_input = torch.reshape(
        tt_input, (N, padded_T, num_devices_H, padded_H // num_devices_H, num_devices_W, padded_W // num_devices_W, C)
    )
    tt_input = tt_input.permute(0, 1, 2, 4, 3, 5, 6)
    tt_input = torch.reshape(
        tt_input, (N, padded_T, num_devices_H * num_devices_W, padded_H // num_devices_H, padded_W // num_devices_W, C)
    )

    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, 1]),
    )
    tt_input = ttnn.squeeze(tt_input, 2)

    logger.info(f"Input shape: {torch_input.shape}")
    logger.info("Run TtCausalUpsampleBlock forward")
    tt_output = tt_model(tt_input)
    logger.info("End TtCausalUpsampleBlock forward")
    tt_output = ttnn.unsqueeze(tt_output, 2)

    # Convert TT output to torch tensor
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, 0]),
    )

    expected_T = T * temporal_expansion - temporal_offset
    expected_padded_T = get_padded_size(expected_T, num_devices_T)
    expected_H = (H * spatial_expansion) // num_devices_H
    expected_W = (W * spatial_expansion) // num_devices_W
    tt_output_torch = torch.reshape(
        tt_output_torch,
        (N, expected_padded_T, num_devices_H, num_devices_W, expected_H, expected_W, expected_output_shape[1]),
    )
    tt_output_torch = tt_output_torch.permute(0, 1, 2, 4, 3, 5, 6)
    tt_output_torch = torch.reshape(
        tt_output_torch, (N, expected_padded_T, H * spatial_expansion, W * spatial_expansion, expected_output_shape[1])
    )
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]
    if mesh_device.get_num_devices() > 1:
        tt_output_torch = tt_output_torch[
            0:N,
            :,
            0:expected_T,
            0 : H * spatial_expansion,
            0 : W * spatial_expansion,
        ]

    # Get reference output
    logger.info("Run RefCausalUpsampleBlock forward")
    with torch.no_grad():
        ref_output = reference_model(torch_input)[0]
    logger.info("End RefCausalUpsampleBlock forward")

    logger.info("assert quality")
    for i in range(T * temporal_expansion - temporal_offset):
        ref_output_slice = ref_output[:, :, i, :, :]
        tt_output_torch_slice = tt_output_torch[:, :, i, :, :]
        assert_quality(ref_output_slice, tt_output_torch_slice, pcc=0.9995)


def create_decoder_models(
    mesh_device,
    parallel_config,
    ccl_manager,
    latent_dim,
    out_channels,
    base_channels,
    channel_multipliers,
    temporal_expansions,
    spatial_expansions,
    num_res_blocks,
    nonlinearity,
    output_nonlinearity,
):
    """Initialize both reference and TT decoder models with optional real weights."""
    # Create reference model
    reference_model = MochiDecoder3D(
        in_channels=latent_dim,
        out_channels=out_channels,
        block_out_channels=[base_channels * multiplier for multiplier in channel_multipliers],
        layers_per_block=num_res_blocks,
        temporal_expansions=temporal_expansions,
        spatial_expansions=spatial_expansions,
        act_fn=nonlinearity,
    )

    # Create TT model with same weights
    tt_model = TtDecoder(
        mesh_device=mesh_device,
        torch_ref=reference_model,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        temporal_expansions=temporal_expansions,
        spatial_expansions=spatial_expansions,
        num_res_blocks=num_res_blocks,
        latent_dim=latent_dim,
        nonlinearity=nonlinearity,
        output_nonlinearity=output_nonlinearity,
    )

    return reference_model, tt_model


# Test case configurations for different input sizes
decoder_test_configs = [
    {
        "name": "small_latent",
        "input_shape": [1, 12, 28, 40, 50],
        "out_channels": 3,
        "base_channels": 128,
        "channel_multipliers": [1, 2, 4, 6],
        "temporal_expansions": [1, 2, 3],
        "spatial_expansions": [2, 2, 2],
        "num_res_blocks": [3, 3, 4, 6, 3],
        "latent_dim": 12,
        "has_attention": [False, False, False, False, False],
        "output_norm": False,
        "nonlinearity": "silu",
        "output_nonlinearity": "silu",
        "causal": True,
        # Expected output will be approximately: (1, 3, 168, 320, 400)
    },
    {
        "name": "large_latent",
        "input_shape": [1, 12, 28, 60, 106],
        "out_channels": 3,
        "base_channels": 128,
        "channel_multipliers": [1, 2, 4, 6],
        "temporal_expansions": [1, 2, 3],
        "spatial_expansions": [2, 2, 2],
        "num_res_blocks": [3, 3, 4, 6, 3],
        "latent_dim": 12,
        "has_attention": [False, False, False, False, False],
        "output_norm": False,
        "nonlinearity": "silu",
        "output_nonlinearity": "silu",
        "causal": True,
        # Expected output will be approximately: (1, 3, 168, 480, 848)
    },
]


def load_dit(
    mesh_device: ttnn.MeshDevice,
    ccl_manager: CCLManager,
    use_cache: bool,
    model_name: str = "genmo/mochi-1-preview",
):
    # Load pretrained Mochi Transformer
    # First load the torch version to get the config and state dict
    from diffusers import MochiTransformer3DModel as TorchMochiTransformer3DModel
    from ....models.transformers.transformer_mochi import MochiTransformer3DModel
    from ....parallel.config import DiTParallelConfig
    from ....utils.cache import get_cache_path, load_cache_dict

    torch_transformer = TorchMochiTransformer3DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=torch.float32
    )

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[1], mesh_axis=1),
        sequence_parallel=ParallelFactor(factor=mesh_device.shape[0], mesh_axis=0),
    )

    # Create TT version with the same config
    transformer = MochiTransformer3DModel(
        patch_size=torch_transformer.config.patch_size,
        num_attention_heads=torch_transformer.config.num_attention_heads,
        attention_head_dim=torch_transformer.config.attention_head_dim,
        num_layers=torch_transformer.config.num_layers,
        pooled_projection_dim=torch_transformer.config.pooled_projection_dim,
        in_channels=torch_transformer.config.in_channels,
        text_embed_dim=torch_transformer.config.text_embed_dim,
        time_embed_dim=torch_transformer.config.time_embed_dim,
        activation_fn=torch_transformer.config.activation_fn,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=True,
    )

    # Load state dict into TT transformer
    if use_cache:
        cache_path = get_cache_path(
            model_name="mochi-1-preview",
            subfolder="transformer",
            parallel_config=parallel_config,
            mesh_shape=tuple(mesh_device.shape),
            dtype="bf16",
        )
        assert os.path.exists(
            cache_path
        ), f"Cache path: {cache_path} does not exist. Run test_mochi_transformer_model_caching first with the desired parallel config."
        cache_dict = load_cache_dict(cache_path)
        transformer.from_cached_state_dict(cache_dict)
    else:
        transformer.load_state_dict(torch_transformer.state_dict())

    return transformer


@pytest.mark.parametrize(
    "config",
    decoder_test_configs,
    ids=[cfg["name"] for cfg in decoder_test_configs],
)
@pytest.mark.parametrize("load_dit_weights", [False, True], ids=["no_dit", "load_dit"])
@pytest.mark.parametrize("num_links", [4, 1], ids=["4links", "1link"])
@vae_device_config
def test_tt_decoder_forward(mesh_device, config, reset_seeds, load_dit_weights, num_links):
    input_shape = config["input_shape"]
    N, C, T, H, W = input_shape

    logger.info(
        f"Testing decoder with latent_dim={config['latent_dim']}, "
        f"base_channels={config['base_channels']}, "
        f"channel_multipliers={config['channel_multipliers']}, "
    )

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)

    if load_dit_weights:
        # Load DiT weights to device to account for real world DRAM usage, checking for OOM.
        logger.info("Loading DiT weights")
        tt_model_dit = load_dit(mesh_device, ccl_manager, use_cache=False)

    # Create models
    logger.info("Creating VAE decoder models")

    if mesh_device.shape[0] == 1:
        w_parallel_factor = 1
    else:
        w_parallel_factor = 2

    vae_parallel_config = MochiVAEParallelConfig(
        time_parallel=ParallelFactor(factor=mesh_device.shape[1], mesh_axis=1),
        w_parallel=ParallelFactor(factor=w_parallel_factor, mesh_axis=0),
        h_parallel=ParallelFactor(factor=mesh_device.shape[0] // w_parallel_factor, mesh_axis=0),
    )
    assert vae_parallel_config.h_parallel.factor * vae_parallel_config.w_parallel.factor == mesh_device.shape[0]
    assert vae_parallel_config.h_parallel.mesh_axis == vae_parallel_config.w_parallel.mesh_axis

    reference_model, tt_model = create_decoder_models(
        mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
        latent_dim=config["latent_dim"],
        out_channels=config["out_channels"],
        base_channels=config["base_channels"],
        channel_multipliers=config["channel_multipliers"],
        num_res_blocks=config["num_res_blocks"],
        temporal_expansions=config["temporal_expansions"],
        spatial_expansions=config["spatial_expansions"],
        nonlinearity=config["nonlinearity"],
        output_nonlinearity=config["output_nonlinearity"],
    )

    # Create input tensor (latent representation)
    torch_input = torch.randn(N, C, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]
    num_devices_T = mesh_device.shape[vae_parallel_config.time_parallel.mesh_axis]
    if T % num_devices_T:
        padded_T = get_padded_size(T, num_devices_T)
        T_padding = padded_T - T
        tt_input = torch.nn.functional.pad(tt_input, pad=(0, 0, 0, 0, 0, 0, 0, T_padding))
    else:
        padded_T = T
    num_devices_W = vae_parallel_config.w_parallel.factor
    if W % num_devices_W:
        padded_W = get_padded_size(W, num_devices_W)
        W_padding = padded_W - W
        tt_input = torch.nn.functional.pad(tt_input, pad=(0, 0, 0, W_padding))
    else:
        padded_W = W
    num_devices_H = vae_parallel_config.h_parallel.factor
    if H % num_devices_H:
        padded_H = get_padded_size(H, num_devices_H)
        H_padding = padded_H - H
        tt_input = torch.nn.functional.pad(tt_input, pad=(0, 0, 0, 0, 0, H_padding))
    else:
        padded_H = H

    tt_input = torch.reshape(
        tt_input, (N, padded_T, num_devices_H, padded_H // num_devices_H, num_devices_W, padded_W // num_devices_W, C)
    )
    tt_input = tt_input.permute(0, 1, 2, 4, 3, 5, 6)
    tt_input = torch.reshape(
        tt_input, (N, padded_T, num_devices_H * num_devices_W, padded_H // num_devices_H, padded_W // num_devices_W, C)
    )

    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, 1]),
    )
    tt_input = ttnn.squeeze(tt_input, 2)

    logger.info(f"Input shape: {torch_input.shape}")
    logger.info("Run TtDecoder forward")
    tt_output = tt_model(tt_input)
    logger.info("End TtDecoder forward")
    tt_output = ttnn.unsqueeze(tt_output, 2)

    # Convert TT output to torch tensor
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, 1]),
    )

    logger.info(f"TT Output shape {tt_output_torch.shape}")

    # Get reference output
    logger.info("Run RefDecoder forward")
    with torch.no_grad():
        ref_output = reference_model(torch_input)[0]
    logger.info("End RefDecoder forward")

    # unpad tt output
    expected_T = ref_output.shape[2]
    expected_padded_T = get_padded_size(expected_T, num_devices_T)
    expected_H = ref_output.shape[3] // num_devices_H
    expected_W = ref_output.shape[4] // num_devices_W
    tt_output_torch = torch.reshape(
        tt_output_torch,
        (N, expected_padded_T, num_devices_H, num_devices_W, expected_H, expected_W, ref_output.shape[1]),
    )
    tt_output_torch = tt_output_torch.permute(0, 1, 2, 4, 3, 5, 6)
    tt_output_torch = torch.reshape(
        tt_output_torch,
        (N, expected_padded_T, num_devices_H * expected_H, num_devices_W * expected_W, ref_output.shape[1]),
    )
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]

    logger.info("assert quality")
    for i in range(ref_output.shape[2]):
        ref_output_slice = ref_output[:, :, i, :, :]
        tt_output_torch_slice = tt_output_torch[:, :, i, :, :]
        assert_quality(ref_output_slice, tt_output_torch_slice, pcc=0.995)

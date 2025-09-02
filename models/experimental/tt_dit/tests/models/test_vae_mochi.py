# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import os
import ttnn

from ...utils.check import assert_quality
from ...models.vae.vae_mochi import (
    Conv1x1 as TtConv1x1,
    ResBlock as TtResBlock,
    CausalUpsampleBlock as TtCausalUpsampleBlock,
    Decoder as TtDecoder,
)
from ...parallel.manager import CCLManager
from ...parallel.config import VAEParallelConfig, ParallelFactor
from loguru import logger
from genmo.mochi_preview.vae.models import Decoder as RefDecoder
from genmo.mochi_preview.vae.models import ResBlock as RefResBlock
from genmo.mochi_preview.vae.models import CausalUpsampleBlock as RefCausalUpsampleBlock

from pathlib import Path


def div_up(numerator, denominator):
    return ((numerator + denominator - 1) // denominator) * denominator


# Basic decoder configuration that aligns with typical decoder settings
decoder_base_args = {
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
}

vae_shapes = [
    # more optimal reshaped versions
    [16, 60, 106, 768],
]


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


def get_vae_dir():
    mochi_dir = os.environ.get("MOCHI_DIR")
    if not mochi_dir:
        raise ValueError("MOCHI_DIR environment variable must be set")
    vae_dir = Path(mochi_dir)
    assert vae_dir.exists()
    return vae_dir


def load_decoder_weights():
    """Load VAE decoder weights from safetensors file."""
    vae_dir = get_vae_dir()
    path = vae_dir / "decoder.safetensors"

    try:
        from safetensors.torch import load_file

        logger.info(f"Loading VAE decoder weights from {path}")
        return load_file(path)
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"Failed to load decoder weights: {e}")
        return None


class Conv3d1x1(nn.Conv3d):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=(1, 1, 1), bias=bias)


def create_random_conv3d_models(mesh_device, in_channels, out_channels, bias=True):
    """Initialize both reference Conv3d and TT models."""
    # Create reference model
    # reference_model = Conv3d1x1(in_channels, out_channels, bias=bias)

    # Create reference model
    reference_model = RefDecoder(**decoder_base_args)

    # Try to load real weights if requested
    state_dict = load_decoder_weights()
    if state_dict:
        try:
            # Load weights into reference model
            reference_model.load_state_dict(state_dict, strict=True)
            reference_model = reference_model.output_proj
            logger.info(f"Loaded real weights for reference decoder model")
        except Exception as e:
            logger.warning(f"Failed to load weights for reference decoder: {e}")

    # Create TT model
    tt_model = TtConv1x1(
        mesh_device=mesh_device,
        in_channels=reference_model.in_features,
        out_channels=reference_model.out_features,
        bias=bias,
        torch_ref=reference_model,
    )

    return reference_model, tt_model


@pytest.mark.parametrize(
    "N, C_in, C_out, T, H, W",
    [
        (1, 12, 768, 28, 60, 106),
    ],
    ids=["12->768"],
)
@pytest.mark.parametrize("divide_T", [8, 1], ids=["T8", "T1"])  # Emulate T fracturing
@vae_device_config
def test_tt_conv3d_1x1x1(mesh_device, N, C_in, C_out, T, H, W, reset_seeds, divide_T):
    """Test forward pass of TtConv1x1 against Conv3d with 1x1x1 kernel."""
    T = T // divide_T
    reference_model, tt_model = create_random_conv3d_models(mesh_device, C_in, C_out)

    # Create input tensor
    torch_input = torch.randn(N, C_in, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]
    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info("Run TtConv1x1 forward (Conv3d mode)")
    tt_output = tt_model(tt_input)

    # Convert TT output to torch tensor
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]).permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]

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


def create_random_resblock_models(mesh_device, mesh_axis, parallel_config, ccl_manager, **model_args):
    """Initialize both reference and TT models."""
    # Create reference model
    reference_model = RefResBlock(**model_args)

    # Create TT model
    tt_model = TtResBlock(
        mesh_device=mesh_device,
        mesh_axis=mesh_axis,
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
        (1, 768, 28, 30, 53),  # 28 -> 32
        (1, 512, 82, 60, 106),  # 82 -> 88
        (1, 256, 163, 120, 212),  # 163 -> 168
        (1, 128, 163, 240, 424),  # 163 -> 168
        # medium latent
        (1, 768, 28, 40, 76),  # 28 -> 32
        (1, 512, 82, 80, 152),  # 82 -> 88
        (1, 256, 163, 160, 304),  # 163 -> 168
        (1, 128, 163, 360, 608),  # 163 -> 168
        # large latent
        (1, 768, 28, 60, 106),  # 28 -> 32
        (1, 512, 82, 120, 212),  # 82 -> 88
        (1, 256, 163, 240, 424),  # 163 -> 168
        (1, 128, 163, 480, 848),  # 163 -> 168
    ],
    ids=["s768", "s512", "s256", "s128", "m768", "m512", "m256", "m128", "l768", "l512", "l256", "l128"],
)
@pytest.mark.parametrize("divide_T", [8, 1], ids=["T8", "T1"])  # Emulate T fracturing
@vae_device_config
def test_tt_resblock_forward(mesh_device, N, C, T, H, W, reset_seeds, divide_T):
    """Test complete forward pass of TtResBlock."""
    T = T // divide_T
    block_args = resblock_args.copy()
    block_args["channels"] = C

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=1)
    vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=8, mesh_axis=1))

    reference_model, tt_model = create_random_resblock_models(
        mesh_device,
        mesh_axis=None,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
        **block_args,
    )

    # Create input tensor
    torch_input = torch.randn(N, C, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]
    tt_input = torch.nn.functional.pad(
        tt_input, pad=(0, 0, 0, 0, 0, 0, 0, div_up(T, mesh_device.get_num_devices()) - T)
    )
    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, 1]),
    )
    logger.info(f"TT input shape: {tt_input.shape}")
    logger.info("Run TtResBlock forward")
    tt_output = tt_model(tt_input)
    # Convert TT output to torch tensor
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[0, 1]),
    )
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]
    tt_output_torch = tt_output_torch[:, :, 0:T, :, :]

    # Get reference output
    logger.info("Run RefResBlock forward")
    with torch.no_grad():
        ref_output = reference_model(torch_input)

    logger.info("assert quality")
    assert_quality(ref_output, tt_output_torch, pcc=0.999)


# Base configuration that applies to all test cases
upsample_base_args = {
    "affine": True,
    "causal": True,
    "prune_bottleneck": False,
    "padding_mode": "replicate",
    "bias": True,
    "has_attention": False,
}


def create_random_causalupsampleblock_models(
    mesh_device, in_channels, out_channels, use_real_weights, parallel_config, ccl_manager, **model_args
):
    """Initialize both reference and TT models with optional real weights."""
    # Create reference model
    reference_model = RefCausalUpsampleBlock(in_channels=in_channels, out_channels=out_channels, **model_args)

    # Try to load real weights if requested
    if use_real_weights:
        decoder_weights = load_decoder_weights()
        if decoder_weights:
            # Find the right upsample block based on channels
            block_idx = None
            if in_channels == 768 and out_channels == 512:
                block_idx = 1  # First upsample block
            elif in_channels == 512 and out_channels == 256:
                block_idx = 2  # Second upsample block
            elif in_channels == 256 and out_channels == 128:
                block_idx = 3  # Third upsample block

            if block_idx is not None:
                # Extract weights with the correct prefix
                block_prefix = f"blocks.{block_idx}"
                block_state_dict = {}

                # Find all weights belonging to this block
                for key, value in decoder_weights.items():
                    if key.startswith(block_prefix):
                        # Remove the block prefix to match reference model keys
                        local_key = key[len(block_prefix) + 1 :]  # +1 for the dot
                        block_state_dict[local_key] = value

                if block_state_dict:
                    try:
                        # Load weights that match the reference model
                        reference_model.load_state_dict(block_state_dict, strict=False)
                        logger.info(
                            f"Loaded real weights for upsample block {block_idx} ({in_channels}->{out_channels})"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load weights for block {block_idx}: {e}")
                else:
                    logger.warning(f"No weights found for upsample block {block_idx}")
            else:
                logger.warning(f"No matching upsample block for {in_channels}->{out_channels}")

    # Create TT model with same weights
    tt_model = TtCausalUpsampleBlock(
        mesh_device=mesh_device,
        in_channels=in_channels,
        out_channels=out_channels,
        torch_ref=reference_model,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        **model_args,
    )

    return reference_model, tt_model


@pytest.mark.parametrize(
    "config",
    [
        # large latent
        # First upsample block (768->512), T padded from 28->32
        {
            "name": "block1_768-512",
            "in_channels": 768,
            "out_channels": 512,
            "num_res_blocks": 6,
            "temporal_expansion": 3,
            "spatial_expansion": 2,
            "input_shape": [1, 768, 28, 60, 106],
            # "expected_output_shape": (1, 512, 82, 120, 212),
        },
        # Second upsample block (512->256), T padded from 82->88
        {
            "name": "block2_512-256",
            "in_channels": 512,
            "out_channels": 256,
            "num_res_blocks": 4,
            "temporal_expansion": 2,
            "spatial_expansion": 2,
            "input_shape": [1, 512, 82, 120, 212],
            # "expected_output_shape": (1, 256, 163, 240, 424),
        },
        # Third upsample block (256->128), T padded from 163->168
        {
            "name": "block3_256-128",
            "in_channels": 256,
            "out_channels": 128,
            "num_res_blocks": 3,
            "temporal_expansion": 1,
            "spatial_expansion": 2,
            "input_shape": [1, 256, 163, 240, 424],
            # "expected_output_shape": (1, 128, 163, 480, 848),
        },
        # medium latent
        # First upsample block (768->512), T padded from 28->32
        {
            "name": "block1_768-512",
            "in_channels": 768,
            "out_channels": 512,
            "num_res_blocks": 6,
            "temporal_expansion": 3,
            "spatial_expansion": 2,
            "input_shape": [1, 768, 28, 40, 76],
            # "expected_output_shape": (1, 512, 82, 80, 152),
        },
        # Second upsample block (512->256), T padded from 82->88
        {
            "name": "block2_512-256",
            "in_channels": 512,
            "out_channels": 256,
            "num_res_blocks": 4,
            "temporal_expansion": 2,
            "spatial_expansion": 2,
            "input_shape": [1, 512, 82, 80, 152],
            # "expected_output_shape": (1, 256, 163, 160, 304),
        },
        # Third upsample block (256->128), T padded from 163->168
        {
            "name": "block3_256-128",
            "in_channels": 256,
            "out_channels": 128,
            "num_res_blocks": 3,
            "temporal_expansion": 1,
            "spatial_expansion": 2,
            "input_shape": [1, 256, 163, 160, 304],
            # "expected_output_shape": (1, 128, 163, 320, 608),
        },
        # small latent
        # First upsample block (768->512), T padded from 28->32
        {
            "name": "block1_768-512",
            "in_channels": 768,
            "out_channels": 512,
            "num_res_blocks": 6,
            "temporal_expansion": 3,
            "spatial_expansion": 2,
            "input_shape": [1, 768, 28, 30, 53],
            # "expected_output_shape": (1, 512, 82, 60, 106),
        },
        # Second upsample block (512->256), T padded from 82->88
        {
            "name": "block2_512-256",
            "in_channels": 512,
            "out_channels": 256,
            "num_res_blocks": 4,
            "temporal_expansion": 2,
            "spatial_expansion": 2,
            "input_shape": [1, 512, 82, 60, 106],
            # "expected_output_shape": (1, 256, 163, 120, 212),
        },
        # Third upsample block (256->128), T padded from 163->168
        {
            "name": "block3_256-128",
            "in_channels": 256,
            "out_channels": 128,
            "num_res_blocks": 3,
            "temporal_expansion": 1,
            "spatial_expansion": 2,
            "input_shape": [1, 256, 163, 120, 212],
            # "expected_output_shape": (1, 128, 163, 240, 424),
        },
    ],
    ids=["l768", "l512", "l256", "m768", "m512", "m256", "s768", "s512", "s256"],
)
@pytest.mark.parametrize("divide_T", [8, 1], ids=["T8", "T1"])  # Emulate T fracturing
@pytest.mark.parametrize("use_real_weights", [False, True], ids=["random_weights", "real_weights"])
@vae_device_config
def test_tt_upsample_forward(mesh_device, config, divide_T, reset_seeds, use_real_weights):
    """Test TtCausalUpsampleBlock against reference implementation."""
    in_channels = config["in_channels"]
    out_channels = config["out_channels"]
    num_res_blocks = config["num_res_blocks"]
    temporal_expansion = config["temporal_expansion"]
    spatial_expansion = config["spatial_expansion"]
    input_shape = config["input_shape"]
    N, C, T, H, W = input_shape
    T = T // divide_T
    # expected_output_shape = config["expected_output_shape"]

    # Prepare model args
    block_args = upsample_base_args.copy()
    block_args.update(
        {
            "temporal_expansion": temporal_expansion,
            "spatial_expansion": spatial_expansion,
            "num_res_blocks": num_res_blocks,
        }
    )

    logger.info(
        f"Testing upsample with in_channels={in_channels}, out_channels={out_channels}, "
        f"temporal_expansion={temporal_expansion}, "
        f"spatial_expansion={spatial_expansion}, "
        f"num_res_blocks={num_res_blocks}, "
        f"use_real_weights={use_real_weights}"
    )

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=1)
    vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=8, mesh_axis=1))

    reference_model, tt_model = create_random_causalupsampleblock_models(
        mesh_device,
        in_channels=in_channels,
        out_channels=out_channels,
        use_real_weights=use_real_weights,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
        **block_args,
    )

    # Create input tensor with correct shape from the decoder
    torch_input = torch.randn(N, C, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]
    tt_input = torch.nn.functional.pad(
        tt_input, pad=(0, 0, 0, 0, 0, 0, 0, div_up(T, mesh_device.get_num_devices()) - T)
    )
    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, 1]),
    )

    logger.info(f"Input shape: {torch_input.shape}")
    logger.info("Run TtCausalUpsampleBlock forward")
    tt_output = tt_model(tt_input)

    # Convert TT output to torch tensor
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[0, 1]),
    )
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]
    if mesh_device.get_num_devices() > 1:
        tt_output_torch = tt_output_torch[:, :, 0 : T * temporal_expansion - (temporal_expansion - 1), :, :]

    # Get reference output
    logger.info("Run RefResBlock forward")
    with torch.no_grad():
        ref_output = reference_model(torch_input)

    logger.info("assert quality")
    assert_quality(ref_output, tt_output_torch, pcc=0.989)


def create_decoder_models(mesh_device, use_real_weights, parallel_config, ccl_manager, **model_args):
    """Initialize both reference and TT decoder models with optional real weights."""
    # Create reference model
    reference_model = RefDecoder(**model_args)

    # Try to load real weights if requested
    if use_real_weights:
        state_dict = load_decoder_weights()
        if state_dict:
            try:
                # Load weights into reference model
                reference_model.load_state_dict(state_dict, strict=True)
                logger.info(f"Loaded real weights for reference decoder model")
            except Exception as e:
                logger.warning(f"Failed to load weights for reference decoder: {e}")

    # Create TT model with same weights
    tt_model = TtDecoder(
        mesh_device=mesh_device,
        torch_ref=reference_model,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        **model_args,
    )

    return reference_model, tt_model


# Test case configurations for different input sizes
decoder_test_configs = [
    {
        "name": "small_latent",
        "input_shape": [1, 12, 28, 30, 53],
        # Expected output will be approximately: (1, 3, 163, 240, 424)
    },
    {
        "name": "medium_latent",
        "input_shape": [1, 12, 28, 40, 76],
        # Expected output will be approximately: (1, 3, 163, 480, 848)
    },
    {
        "name": "large_latent",
        "input_shape": [1, 12, 28, 60, 106],
        # Expected output will be approximately: (1, 3, 163, 480, 848)
    },
]


@pytest.mark.parametrize(
    "config",
    decoder_test_configs,
    ids=[cfg["name"] for cfg in decoder_test_configs],
)
@pytest.mark.parametrize("divide_T", [8, 1], ids=["T8", "T1"])  # Emulate T fracturing
@pytest.mark.parametrize("use_real_weights", [False, True], ids=["random_weights", "real_weights"])
@pytest.mark.parametrize("load_dit_weights", [False, True], ids=["no_dit", "load_dit"])
@vae_device_config
def test_tt_decoder_forward(mesh_device, config, divide_T, reset_seeds, use_real_weights, load_dit_weights):
    input_shape = config["input_shape"]
    N, C, T, H, W = input_shape
    T = T // divide_T

    # Initialize model arguments
    model_args = decoder_base_args.copy()

    logger.info(
        f"Testing decoder with latent_dim={model_args['latent_dim']}, "
        f"base_channels={model_args['base_channels']}, "
        f"channel_multipliers={model_args['channel_multipliers']}, "
        f"use_real_weights={use_real_weights}"
    )

    # TODO after the new model creation API is set
    # if load_dit_weights:
    #     # Load DiT weights to device to account for real world DRAM usage, checking for OOM.
    #     logger.info("Loading DiT weights")
    #     reference_model, tt_model_dit, state_dict = create_models(mesh_device, n_layers=48)
    #     del reference_model

    # Create models
    logger.info("Creating VAE decoder models")
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=1)
    vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=8, mesh_axis=1))

    reference_model, tt_model = create_decoder_models(
        mesh_device,
        use_real_weights=use_real_weights,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
        **model_args,
    )

    # Create input tensor (latent representation)
    torch_input = torch.randn(N, C, T, H, W)

    # Convert to TTNN format [N, T, H, W, C]
    tt_input = torch_input.permute(0, 2, 3, 4, 1)
    tt_input = torch.nn.functional.pad(
        tt_input, pad=(0, 0, 0, 0, 0, 0, 0, div_up(T, mesh_device.get_num_devices()) - T)
    )
    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, 1]),
    )

    logger.info(f"Input shape: {torch_input.shape}")
    logger.info("Run TtDecoder forward")
    tt_output = tt_model(tt_input)

    # Convert TT output to torch tensor
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[0, 1]),
    )
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]
    logger.info(f"TT output shape: {tt_output_torch.shape}")

    # Get reference output
    logger.info("Run RefDecoder forward")
    with torch.no_grad():
        ref_output = reference_model(torch_input)
    logger.info(f"Reference output shape: {ref_output.shape}")

    # unpad tt output
    if mesh_device.get_num_devices() > 1:
        tt_output_torch = tt_output_torch[:, :, 0 : ref_output.shape[2], :, :]

    logger.info("assert quality")
    assert_quality(ref_output, tt_output_torch, pcc=0.99)

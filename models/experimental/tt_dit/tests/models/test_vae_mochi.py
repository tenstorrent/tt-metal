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
)
from loguru import logger
from genmo.mochi_preview.vae.models import Decoder as RefDecoder
from genmo.mochi_preview.vae.models import ResBlock as RefResBlock
from genmo.mochi_preview.vae.models import CausalUpsampleBlock as RefCausalUpsampleBlock

from pathlib import Path

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
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
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


def create_random_resblock_models(mesh_device, mesh_axis, **model_args):
    """Initialize both reference and TT models."""
    # Create reference model
    reference_model = RefResBlock(**model_args)

    # Create TT model
    tt_model = TtResBlock(mesh_device=mesh_device, mesh_axis=mesh_axis, torch_ref=reference_model)

    return reference_model, tt_model


@torch.no_grad()
@pytest.mark.parametrize(
    "N, C, T, H, W",
    [
        (1, 768, 28, 60, 106),
        (1, 512, 82, 120, 212),
        (1, 256, 163, 240, 424),
        (1, 128, 163, 480, 848),
    ],
    ids=["768", "512", "256", "128"],
)
@pytest.mark.parametrize("divide_T", [8, 1], ids=["T8", "T1"])  # Emulate T fracturing
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_tt_resblock_forward(mesh_device, N, C, T, H, W, reset_seeds, divide_T):
    """Test complete forward pass of TtResBlock."""
    T = T // divide_T
    block_args = resblock_args.copy()
    block_args["channels"] = C
    reference_model, tt_model = create_random_resblock_models(mesh_device, mesh_axis=None, **block_args)

    # Create input tensor
    torch_input = torch.randn(N, C, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]
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
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]

    # Get reference output
    with torch.no_grad():
        ref_output = reference_model(torch_input)

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

# Test case configurations from decoder
test_configs = [
    # First upsample block (768->512)
    {
        "name": "block1_768-512",
        "in_channels": 768,
        "out_channels": 512,
        "num_res_blocks": 6,
        "temporal_expansion": 3,
        "spatial_expansion": 2,
        "input_shape": (1, 768, 28, 60, 106),
        # "expected_output_shape": (1, 512, 82, 120, 212),
    },
    # Second upsample block (512->256)
    {
        "name": "block2_512-256",
        "in_channels": 512,
        "out_channels": 256,
        "num_res_blocks": 4,
        "temporal_expansion": 2,
        "spatial_expansion": 2,
        "input_shape": (1, 512, 82, 120, 212),
        # "expected_output_shape": (1, 256, 163, 240, 424),
    },
    # Third upsample block (256->128)
    {
        "name": "block3_256-128",
        "in_channels": 256,
        "out_channels": 128,
        "num_res_blocks": 3,
        "temporal_expansion": 1,
        "spatial_expansion": 2,
        "input_shape": (1, 256, 163, 240, 424),
        # "expected_output_shape": (1, 128, 163, 480, 848),
    },
]


def create_random_causalupsampleblock_models(
    mesh_device, in_channels, out_channels, use_real_weights=False, **model_args
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
        **model_args,
    )

    return reference_model, tt_model


@pytest.mark.parametrize(
    "config",
    test_configs,
    ids=[cfg["name"] for cfg in test_configs],
)
@pytest.mark.parametrize("divide_T", [8, 1], ids=["T8", "T1"])  # Emulate T fracturing
@pytest.mark.parametrize("use_real_weights", [False, True], ids=["random_weights", "real_weights"])
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_upsample(mesh_device, config, divide_T, reset_seeds, use_real_weights):
    """Test TtCausalUpsampleBlock against reference implementation."""
    in_channels = config["in_channels"]
    out_channels = config["out_channels"]
    num_res_blocks = config["num_res_blocks"]
    temporal_expansion = config["temporal_expansion"]
    spatial_expansion = config["spatial_expansion"]
    input_shape = config["input_shape"]
    N, C, T, H, W = input_shape
    T = T // divide_T
    input_shape = (N, C, T, H, W)
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

    reference_model, tt_model = create_random_causalupsampleblock_models(
        mesh_device, in_channels=in_channels, out_channels=out_channels, use_real_weights=use_real_weights, **block_args
    )

    # Create input tensor with correct shape from the decoder
    N, C, T, H, W = input_shape
    torch_input = torch.randn(N, C, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]
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
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]

    # Get reference output
    with torch.no_grad():
        ref_output = reference_model(torch_input)

    assert_quality(ref_output, tt_output_torch, pcc=0.989)


# def create_decoder_models(mesh_device, use_real_weights=False, **model_args):
#     """Initialize both reference and TT decoder models with optional real weights."""
#     # Create reference model
#     reference_model = RefDecoder(**model_args)

#     # Try to load real weights if requested
#     if use_real_weights:
#         state_dict = load_decoder_weights()
#         if state_dict:
#             try:
#                 # Load weights into reference model
#                 reference_model.load_state_dict(state_dict, strict=True)
#                 logger.info(f"Loaded real weights for reference decoder model")
#             except Exception as e:
#                 logger.warning(f"Failed to load weights for reference decoder: {e}")

#     # Create TT model with same weights
#     tt_model = TtDecoder(
#         mesh_device=mesh_device,
#         state_dict=reference_model.state_dict(),
#         state_dict_prefix="",
#         **model_args,
#     )

#     return reference_model, tt_model

# # adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/autoencoders/vae.py
# class VaeDecoder(torch.nn.Module):
#     def __init__(
#         self,
#         block_out_channels: list[int] | tuple[int, ...] = (128, 256, 512, 512),
#         in_channels: int = 16,
#         out_channels: int = 3,
#         layers_per_block: int = 2,
#         norm_num_groups: int = 32,
#     ) -> None:
#         super().__init__()

#         self.conv_in = torch.nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, padding=1)

#         self.mid_block = UNetMidBlock2D(
#             in_channels=block_out_channels[-1],
#             attention_head_dim=block_out_channels[-1],
#             resnet_groups=norm_num_groups,
#         )

#         self.up_blocks = torch.nn.ModuleList([])

#         reversed_block_out_channels = list(reversed(block_out_channels))
#         prev_output_channel = reversed_block_out_channels[0]
#         for i, output_channel in enumerate(reversed_block_out_channels):
#             is_final_block = i == len(reversed_block_out_channels) - 1

#             up_block = UpDecoderBlock2D(
#                 num_layers=layers_per_block + 1,
#                 in_channels=prev_output_channel,
#                 out_channels=output_channel,
#                 add_upsample=not is_final_block,
#                 resnet_groups=norm_num_groups,
#             )

#             self.up_blocks.append(up_block)
#             prev_output_channel = output_channel

#         self.conv_norm_out = torch.nn.GroupNorm(
#             num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
#         )
#         self.conv_act = torch.nn.SiLU()
#         self.conv_out = torch.nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv_in(x)

#         upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

#         x = self.mid_block(x)
#         x = x.to(upscale_dtype)

#         for up_block in self.up_blocks:
#             x = up_block(x)

#         x = self.conv_norm_out(x)
#         x = self.conv_act(x)
#         return self.conv_out(x)


# # adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/unets/unet_2d_blocks.py
# class UpDecoderBlock2D(torch.nn.Module):
#     def __init__(
#         self,
#         *,
#         in_channels: int,
#         out_channels: int,
#         num_layers: int,
#         resnet_groups: int,
#         add_upsample: bool,
#     ) -> None:
#         super().__init__()

#         self.resnets = torch.nn.ModuleList(
#             [
#                 ResnetBlock2D(
#                     in_channels=in_channels if i == 0 else out_channels,
#                     out_channels=out_channels,
#                     groups=resnet_groups,
#                 )
#                 for i in range(num_layers)
#             ]
#         )

#         if add_upsample:
#             self.upsamplers = torch.nn.ModuleList([Upsample2D(channels=out_channels, out_channels=out_channels)])
#         else:
#             self.upsamplers = torch.nn.ModuleList([])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for resnet in self.resnets:
#             x = resnet(x)

#         for upsampler in self.upsamplers:
#             x = upsampler(x)

#         return x


# # adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/unets/unet_2d_blocks.py
# class UNetMidBlock2D(torch.nn.Module):
#     def __init__(
#         self,
#         *,
#         in_channels: int,
#         resnet_groups: int,
#         attention_head_dim: int,
#     ) -> None:
#         super().__init__()

#         self.attentions = torch.nn.ModuleList(
#             [
#                 Attention(
#                     query_dim=in_channels,
#                     heads=in_channels // attention_head_dim,
#                     dim_head=attention_head_dim,
#                     norm_num_groups=resnet_groups,
#                 )
#             ]
#         )

#         self.resnets = torch.nn.ModuleList(
#             [
#                 ResnetBlock2D(
#                     in_channels=in_channels,
#                     out_channels=in_channels,
#                     groups=resnet_groups,
#                 )
#                 for _ in range(2)
#             ]
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.resnets[0](x)
#         x = self.attentions[0](x)
#         return self.resnets[1](x)


# # adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/resnet.py
# class ResnetBlock2D(torch.nn.Module):
#     def __init__(
#         self,
#         *,
#         in_channels: int,
#         out_channels: int,
#         groups: int,
#         eps: float = 1e-6,
#     ) -> None:
#         super().__init__()

#         self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps)
#         self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
#         self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

#         self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.nonlinearity = torch.nn.SiLU()

#         if in_channels != out_channels:
#             self.conv_shortcut = torch.nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=1,
#                 stride=1,
#                 padding=0,
#             )
#         else:
#             self.conv_shortcut = None

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         residual = x

#         x = self.norm1(x)
#         x = self.nonlinearity(x)
#         x = self.conv1(x)
#         x = self.norm2(x)
#         x = self.nonlinearity(x)
#         x = self.conv2(x)

#         if self.conv_shortcut is not None:
#             residual = self.conv_shortcut(residual)

#         return residual + x


# # adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/upsampling.py
# class Upsample2D(torch.nn.Module):
#     def __init__(self, *, channels: int, out_channels: int) -> None:
#         super().__init__()

#         self.channels = channels
#         self.conv = torch.nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         assert x.shape[1] == self.channels
#         x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
#         return self.conv(x)


# # adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py
# class Attention(torch.nn.Module):
#     def __init__(self, *, query_dim: int, norm_num_groups: int, heads: int, dim_head: int) -> None:
#         super().__init__()

#         self.heads = query_dim // dim_head

#         self.group_norm = torch.nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=1e-6, affine=True)

#         inner_dim = dim_head * heads

#         self.to_q = torch.nn.Linear(query_dim, inner_dim)
#         self.to_k = torch.nn.Linear(query_dim, inner_dim)
#         self.to_v = torch.nn.Linear(query_dim, inner_dim)

#         self.to_out = torch.nn.ModuleList([torch.nn.Linear(inner_dim, query_dim)])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         assert x.ndim == 4  # noqa: PLR2004

#         residual = x

#         batch_size, features, height, width = x.shape
#         x = x.view(batch_size, features, height * width).transpose(1, 2)

#         x = self.group_norm(x.transpose(1, 2)).transpose(1, 2)

#         q = self.to_q(x)
#         k = self.to_k(x)
#         v = self.to_v(x)

#         inner_dim = k.shape[-1]
#         head_dim = inner_dim // self.heads

#         q = q.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
#         k = k.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
#         v = v.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

#         x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

#         x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)

#         x = self.to_out[0](x)

#         x = x.transpose(-1, -2).reshape(batch_size, features, height, width)

#         return x + residual


# class ResnetBlock2D(torch.nn.Module):
#     def __init__(self, *, in_channels, out_channels, groups, eps=1e-6):
#         super().__init__()

#         self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps)
#         self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
#         self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

#         self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.nonlinearity = torch.nn.SiLU()

#         if in_channels != out_channels:
#             self.conv_shortcut = torch.nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=1,
#                 stride=1,
#                 padding=0,
#             )
#         else:
#             self.conv_shortcut = None

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         residual = x

#         x = self.norm1(x)
#         x = self.nonlinearity(x)
#         x = self.conv1(x)
#         x = self.norm2(x)
#         x = self.nonlinearity(x)
#         x = self.conv2(x)

#         if self.conv_shortcut is not None:
#             residual = self.conv_shortcut(residual)

#         return residual + x


# # Custom pytest mark for shared VAE device configuration
# def vae_device_config(func):
#     """Decorator to apply standard VAE device configuration to tests"""
#     func = pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)(func)
#     func = pytest.mark.parametrize(
#         "device_params",
#         [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 20000000}],
#         indirect=True,
#     )(func)
#     return func


# @vae_device_config
# @pytest.mark.parametrize(
#     (
#         "batch",
#         "height",
#         "width",
#         "in_channels",
#         "out_channels",
#         "groups",
#     ),
#     [(1, 1024, 1024, 256, 128, 32)],
# )
# def test_sd35_vae_resnet_block(
#     *,
#     mesh_device: ttnn.Device,
#     batch: int,
#     height: int,
#     width: int,
#     in_channels: int,
#     out_channels: int,
#     groups: int,
# ) -> None:
#     torch_model = ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, groups=groups)
#     torch_model.eval()

#     # breakpoint()  # Commented out for normal test execution
#     vae_parallel_manager = create_vae_parallel_manager(
#         mesh_device, CCLManager(mesh_device, topology=ttnn.Topology.Linear)
#     )

#     tt_model = vae_sd35.ResnetBlock.from_torch(
#         torch_ref=torch_model, mesh_device=mesh_device, mesh_axis=1, parallel_manager=vae_parallel_manager
#     )

#     torch_input = torch.randn(batch, in_channels, height, width)

#     tt_input_tensor = ttnn.from_torch(
#         torch_input.permute(0, 2, 3, 1),
#         dtype=ttnn.bfloat16,
#         device=mesh_device,
#         mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
#         layout=ttnn.TILE_LAYOUT,
#     )

#     with torch.no_grad():
#         torch_output = torch_model(torch_input)

#     tt_out = tt_model(tt_input_tensor)

#     tt_out = vae_parallel_manager.vae_all_gather(tt_out)

#     tt_final_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
#     assert_quality(torch_output, tt_final_out_torch, pcc=0.999_500)


# @vae_device_config
# @pytest.mark.parametrize(
#     ("batch", "in_channels", "out_channels", "height", "width", "num_layers", "num_groups", "add_upsample"),
#     [
#         (1, 512, 512, 128, 128, 2, 32, False),
#         (1, 512, 512, 128, 128, 2, 32, True),
#     ],
# )
# def test_sd35_vae_up_decoder_block(
#     *,
#     mesh_device: ttnn.Device,
#     batch: int,
#     in_channels: int,
#     out_channels: int,
#     height: int,
#     width: int,
#     num_layers: int,
#     num_groups: int,
#     add_upsample: bool,
# ) -> None:
#     torch_model = UpDecoderBlock2D(
#         in_channels=in_channels,
#         out_channels=out_channels,
#         num_layers=num_layers,
#         resnet_groups=num_groups,
#         add_upsample=add_upsample,
#     )
#     torch_model.eval()

#     vae_parallel_manager = create_vae_parallel_manager(
#         mesh_device, CCLManager(mesh_device, topology=ttnn.Topology.Linear)
#     )

#     tt_model = vae_sd35.UpDecoderBlock2D.from_torch(
#         torch_ref=torch_model, mesh_device=mesh_device, mesh_axis=1, parallel_manager=vae_parallel_manager
#     )

#     torch_input = torch.randn(batch, in_channels, height, width)

#     tt_input_tensor = ttnn.from_torch(
#         torch_input.permute(0, 2, 3, 1),
#         dtype=ttnn.bfloat16,
#         device=mesh_device,
#         mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
#         layout=ttnn.TILE_LAYOUT,
#     )

#     # TODO: Refactor common test components
#     with torch.no_grad():
#         torch_output = torch_model(torch_input)

#     tt_out = tt_model(tt_input_tensor)

#     tt_out = vae_parallel_manager.vae_all_gather(tt_out)

#     tt_final_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
#     assert_quality(torch_output, tt_final_out_torch, pcc=0.999_500)


# @vae_device_config
# @pytest.mark.parametrize(
#     ("batch", "in_channels", "height", "width", "num_groups", "num_heads"),
#     [
#         (1, 512, 128, 128, 32, 4),  # slice 128, output blocks 32. Need to parametize
#         # (1, 512, 128, 128, 32, 4, False),  # slice 128, output blocks 32. Need to parametize
#     ],
# )
# def test_sd35_vae_attention(
#     *,
#     mesh_device: ttnn.Device,
#     batch: int,
#     in_channels: int,
#     height: int,
#     width: int,
#     num_groups: int,
#     num_heads: int,
# ):
#     torch_model = Attention(
#         query_dim=in_channels, heads=num_heads, dim_head=in_channels // num_heads, norm_num_groups=num_groups
#     )
#     torch_model.eval()

#     vae_parallel_manager = create_vae_parallel_manager(
#         mesh_device, CCLManager(mesh_device, topology=ttnn.Topology.Linear)
#     )

#     tt_model = vae_sd35.Attention.from_torch(
#         torch_ref=torch_model, mesh_device=mesh_device, mesh_axis=1, parallel_manager=vae_parallel_manager
#     )

#     torch_input = torch.randn(batch, in_channels, height, width)

#     tt_input_tensor = ttnn.from_torch(
#         torch_input.permute(0, 2, 3, 1),
#         dtype=ttnn.bfloat16,
#         device=mesh_device,
#         mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
#         layout=ttnn.TILE_LAYOUT,
#     )

#     with torch.no_grad():
#         torch_output = torch_model(torch_input)

#     tt_out = tt_model(tt_input_tensor)

#     tt_out = vae_parallel_manager.vae_all_gather(tt_out)

#     tt_final_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
#     assert_quality(torch_output, tt_final_out_torch, pcc=0.999_500)


# @vae_device_config
# @pytest.mark.parametrize(
#     ("batch", "in_channels", "height", "width", "num_groups", "num_heads"),
#     [
#         (1, 512, 128, 128, 32, 4),  # slice 128, output blocks 32. Need to parametize
#     ],
# )
# def test_sd35_vae_unet_mid_block2d(
#     *,
#     mesh_device: ttnn.Device,
#     batch: int,
#     in_channels: int,
#     height: int,
#     width: int,
#     num_groups: int,
#     num_heads: int,
# ):
#     torch_model = UNetMidBlock2D(
#         in_channels=in_channels, resnet_groups=num_groups, attention_head_dim=in_channels // num_heads
#     )
#     torch_model.eval()

#     vae_parallel_manager = create_vae_parallel_manager(
#         mesh_device, CCLManager(mesh_device, topology=ttnn.Topology.Linear)
#     )

#     tt_model = vae_sd35.UnetMidBlock2D.from_torch(
#         torch_ref=torch_model, mesh_device=mesh_device, mesh_axis=1, parallel_manager=vae_parallel_manager
#     )

#     torch_input = torch.randn(batch, in_channels, height, width)

#     tt_input_tensor = ttnn.from_torch(
#         torch_input.permute(0, 2, 3, 1),
#         dtype=ttnn.bfloat16,
#         device=mesh_device,
#         mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
#         layout=ttnn.TILE_LAYOUT,
#     )

#     with torch.no_grad():
#         torch_output = torch_model(torch_input)

#     tt_out = tt_model(tt_input_tensor)

#     tt_out = vae_parallel_manager.vae_all_gather(tt_out)

#     tt_final_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
#     assert_quality(torch_output, tt_final_out_torch, pcc=0.999_000)


# @vae_device_config
# @pytest.mark.parametrize(
#     (
#         "batch",
#         "in_channels",
#         "out_channels",
#         "layers_per_block",
#         "height",
#         "width",
#         "norm_num_groups",
#         "block_out_channels",
#     ),
#     [
#         (1, 16, 3, 2, 128, 128, 32, (128, 256, 512, 512)),
#     ],
# )


# # Basic decoder configuration that aligns with typical decoder settings
# decoder_base_args = {
#     "out_channels": 3,
#     "base_channels": 128,
#     "channel_multipliers": [1, 2, 4, 6],
#     "temporal_expansions": [1, 2, 3],
#     "spatial_expansions": [2, 2, 2],
#     "num_res_blocks": [3, 3, 4, 6, 3],
#     "latent_dim": 12,
#     "has_attention": [False, False, False, False, False],
#     "output_norm": False,
#     "nonlinearity": "silu",
#     "output_nonlinearity": "silu",
#     "causal": True,
# }

# # Test case configurations for different input sizes
# test_configs = [
#     {
#         "name": "small_latent",
#         "input_shape": (1, 12, 28, 30, 53),
#         # Expected output will be approximately: (1, 3, 163, 240, 424)
#     },
#     {
#         "name": "medium_latent",
#         "input_shape": (1, 12, 28, 40, 76),
#         # Expected output will be approximately: (1, 3, 163, 480, 848)
#     },
#     {
#         "name": "large_latent",
#         "input_shape": (1, 12, 28, 60, 106),
#         # Expected output will be approximately: (1, 3, 163, 480, 848)
#     },
# ]

# @pytest.mark.parametrize(
#     "config",
#     test_configs,
#     ids=[cfg["name"] for cfg in test_configs],
# )
# @pytest.mark.parametrize("divide_T", [8, 1], ids=["T8", "T1"])  # Emulate T fracturing
# @pytest.mark.parametrize("use_real_weights", [False, True], ids=["random_weights", "real_weights"])
# @pytest.mark.parametrize("load_dit_weights", [False, True], ids=["no_dit", "load_dit"])
# def test_movchi_vae_vae_decoder(
#     *,
#     mesh_device: ttnn.Device,
#     batch: int,
#     in_channels: int,
#     out_channels: int,
#     layers_per_block: int,
#     height: int,
#     width: int,
#     norm_num_groups: int,
#     block_out_channels: list[int] | tuple[int, ...],
# ):
#     input_shape = config["input_shape"]
#     N, C, T, H, W = input_shape
#     T = T // divide_T
#     input_shape = (N, C, T, H, W)

#     # Initialize model arguments
#     model_args = decoder_base_args.copy()

#     logger.info(
#         f"Testing decoder with latent_dim={model_args['latent_dim']}, "
#         f"base_channels={model_args['base_channels']}, "
#         f"channel_multipliers={model_args['channel_multipliers']}, "
#         f"use_real_weights={use_real_weights}"
#     )

#     # TODO after the new model creation API is set
#     # if load_dit_weights:
#     #     # Load DiT weights to device to account for real world DRAM usage, checking for OOM.
#     #     logger.info("Loading DiT weights")
#     #     reference_model, tt_model_dit, state_dict = create_models(mesh_device, n_layers=48)
#     #     del reference_model

#     # Create models
#     logger.info("Creating VAE decoder models")
#     reference_model, tt_model = create_decoder_models(mesh_device, use_real_weights=use_real_weights, **model_args)

#     # Create input tensor (latent representation)
#     N, C, T, H, W = input_shape
#     torch_input = torch.randn(N, C, T, H, W)

#     # Convert to TTNN format [N, T, H, W, C]
#     tt_input = torch_input.permute(0, 2, 3, 4, 1)
#     tt_input = ttnn.from_torch(
#         tt_input,
#         device=mesh_device,
#         dtype=ttnn.DataType.BFLOAT16,
#         layout=ttnn.ROW_MAJOR_LAYOUT,
#         memory_config=ttnn.DRAM_MEMORY_CONFIG,
#         mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
#     )

#     logger.info(f"Input shape: {torch_input.shape}")
#     logger.info("Run TtDecoder forward")
#     tt_output = tt_model.forward(tt_input)

#     # Convert TT output to torch tensor (from NTHWC to NCTHW)
#     tt_output_torch = to_torch_tensor(tt_output, mesh_device, dim=1)
#     tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]
#     logger.info(f"TT output shape: {tt_output_torch.shape}")

#     # Get reference output
#     with torch.no_grad():
#         ref_output = reference_model(torch_input)
#     logger.info(f"Reference output shape: {ref_output.shape}")

#     # Verify output shapes match
#     assert (
#         ref_output.shape == tt_output_torch.shape
#     ), f"Output shapes do not match: {ref_output.shape} vs {tt_output_torch.shape}"

#     validate_outputs(tt_output_torch, ref_output, "TtDecoder forward")


#     vae_parallel_manager = create_vae_parallel_manager(
#         mesh_device, CCLManager(mesh_device, topology=ttnn.Topology.Linear)
#     )

#     tt_model = vae_sd35.VAEDecoder.from_torch(
#         torch_ref=torch_model, mesh_device=mesh_device, mesh_axis=1, parallel_manager=vae_parallel_manager
#     )

#     torch_input = torch.randn(batch, in_channels, height, width)

#     tt_input_tensor = ttnn.from_torch(
#         torch_input.permute(0, 2, 3, 1),
#         dtype=ttnn.bfloat16,
#         device=mesh_device,
#         mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
#     )

#     with torch.no_grad():
#         torch_output = torch_model(torch_input)

#     tt_out = tt_model(tt_input_tensor)
#     # tt_out = vae_parallel_manager.vae_all_gather(tt_out)

#     tt_final_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
#     assert_quality(torch_output, tt_final_out_torch, pcc=0.99_000)

#     start = time()
#     tt_out = tt_model(tt_input_tensor)
#     ttnn.synchronize_device(mesh_device)
#     logger.info(f"VAE Time taken: {time() - start}")

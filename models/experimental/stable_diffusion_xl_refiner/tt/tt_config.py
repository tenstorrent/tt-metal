# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Tuple
from models.experimental.stable_diffusion_xl_refiner.tt.components.group_normalization_layer import (
    NormConfig,
    make_norm_config,
)
from models.experimental.stable_diffusion_xl_refiner.tt.components.convolution_layer import ConvConfig, make_conv_config

# Issue: https://github.com/tenstorrent/tt-metal/issues/30191


@dataclass
class ResnetBlockConfig:
    norm1: NormConfig = None
    conv1: ConvConfig = None
    norm2: NormConfig = None
    conv2: ConvConfig = None
    conv_shortcut: ConvConfig = None
    use_conv_shortcut: bool = False


# default is 1 out_block, 8x8 grid
def make_resnet_config(
    norm1_sharded: bool = True,
    norm1_out_blocks: int = 1,
    norm1_grid: Optional[Tuple[int, int]] = (8, 8),
    conv1_kernel: Optional[Tuple[int, int]] = (3, 3),
    conv1_stride: Optional[Tuple[int, int]] = (1, 1),
    conv1_padding: Optional[Tuple[int, int]] = (1, 1),
    conv1_split_in: int = 1,
    norm2_sharded: bool = True,
    norm2_out_blocks: int = 1,
    norm2_grid: Optional[Tuple[int, int]] = (8, 8),
    conv2_kernel: Optional[Tuple[int, int]] = (3, 3),
    conv2_stride: Optional[Tuple[int, int]] = (1, 1),
    conv2_padding: Optional[Tuple[int, int]] = (1, 1),
    conv2_split_in: int = 1,
    use_conv_shortcut: bool = False,
    conv_shortcut_kernel: Optional[Tuple[int, int]] = (1, 1),
    conv_shortcut_stride: Optional[Tuple[int, int]] = (1, 1),
    conv_shortcut_padding: Optional[Tuple[int, int]] = (0, 0),
    conv_shortcut_split_in: int = 1,
) -> ResnetBlockConfig:
    return ResnetBlockConfig(
        norm1=make_norm_config(norm1_sharded, norm1_out_blocks, norm1_grid),
        conv1=make_conv_config(conv1_kernel, conv1_stride, conv1_padding, conv1_split_in),
        norm2=make_norm_config(norm2_sharded, norm2_out_blocks, norm2_grid),
        conv2=make_conv_config(conv2_kernel, conv2_stride, conv2_padding, conv2_split_in),
        conv_shortcut=make_conv_config(
            conv_shortcut_kernel, conv_shortcut_stride, conv_shortcut_padding, conv_shortcut_split_in
        ),
        use_conv_shortcut=use_conv_shortcut,
    )


# SDXL Refiner ResNet block configurations
resnet_configs = {
    # Down blocks
    "down_blocks.0.resnets.0": make_resnet_config(
        norm1_out_blocks=3, norm1_grid=(4, 4), norm2_out_blocks=3, norm2_grid=(4, 4)
    ),
    "down_blocks.0.resnets.1": make_resnet_config(
        norm1_out_blocks=3, norm1_grid=(4, 4), norm2_out_blocks=3, norm2_grid=(4, 4)
    ),
    "down_blocks.1.resnets.0": make_resnet_config(norm1_grid=(4, 4), use_conv_shortcut=True),
    "down_blocks.1.resnets.1": make_resnet_config(),
    "down_blocks.2.resnets.0": make_resnet_config(use_conv_shortcut=True),
    "down_blocks.2.resnets.1": make_resnet_config(),
    "down_blocks.3.resnets.0": make_resnet_config(),
    "down_blocks.3.resnets.1": make_resnet_config(),
    # Mid blocks x2
    "mid_block.resnets.0": make_resnet_config(),
    "mid_block.resnets.1": make_resnet_config(),
    # Up blocks
    "up_blocks.0.resnets.0": make_resnet_config(use_conv_shortcut=True),
    "up_blocks.0.resnets.1": make_resnet_config(use_conv_shortcut=True),
    "up_blocks.0.resnets.2": make_resnet_config(use_conv_shortcut=True),
    "up_blocks.1.resnets.0": make_resnet_config(use_conv_shortcut=True),
    "up_blocks.1.resnets.1": make_resnet_config(use_conv_shortcut=True),
    "up_blocks.1.resnets.2": make_resnet_config(use_conv_shortcut=True),
    "up_blocks.2.resnets.0": make_resnet_config(use_conv_shortcut=True),
    "up_blocks.2.resnets.1": make_resnet_config(use_conv_shortcut=True),
    "up_blocks.2.resnets.2": make_resnet_config(norm1_grid=(4, 4), use_conv_shortcut=True),
    "up_blocks.3.resnets.0": make_resnet_config(
        norm1_sharded=False,
        norm1_out_blocks=3,
        norm1_grid=(4, 4),
        conv1_split_in=2,
        norm2_out_blocks=3,
        norm2_grid=(4, 4),
        use_conv_shortcut=True,
    ),
    "up_blocks.3.resnets.1": make_resnet_config(
        norm1_sharded=False, norm1_out_blocks=2, norm2_out_blocks=3, norm2_grid=(4, 4), use_conv_shortcut=True
    ),
    "up_blocks.3.resnets.2": make_resnet_config(
        norm1_sharded=False, norm1_out_blocks=2, norm2_out_blocks=3, norm2_grid=(4, 4), use_conv_shortcut=True
    ),
}


def get_resnet_config(module_path: str) -> ResnetBlockConfig:
    config = resnet_configs.get(module_path)
    return config


########## Upsample block configurations ##########
@dataclass
class UpsampleConfig:
    conv: ConvConfig = None


def make_upsample_config(
    conv_kernel: Optional[Tuple[int, int]] = (3, 3),
    conv_stride: Optional[Tuple[int, int]] = (1, 1),
    conv_padding: Optional[Tuple[int, int]] = (1, 1),
    conv_split_in: int = 1,
) -> UpsampleConfig:
    return UpsampleConfig(conv=make_conv_config(conv_kernel, conv_stride, conv_padding, conv_split_in))


upsample_configs = {
    "up_blocks.0.upsamplers.0": make_upsample_config(),
    "up_blocks.1.upsamplers.0": make_upsample_config(),
    "up_blocks.2.upsamplers.0": make_upsample_config(),
}


def get_upsample_config(module_path: str) -> UpsampleConfig:
    config = upsample_configs.get(module_path)
    return config


########## Downsample block configurations ##########
@dataclass
class DownsampleConfig:
    conv: ConvConfig = None


def make_downsample_config(
    conv_kernel: Optional[Tuple[int, int]] = (3, 3),
    conv_stride: Optional[Tuple[int, int]] = (2, 2),
    conv_padding: Optional[Tuple[int, int]] = (1, 1),
    conv_split_in: int = 1,
) -> DownsampleConfig:
    return DownsampleConfig(conv=make_conv_config(conv_kernel, conv_stride, conv_padding, conv_split_in))


downsample_configs = {
    "down_blocks.0.downsamplers.0": make_downsample_config(),
    "down_blocks.1.downsamplers.0": make_downsample_config(),
    "down_blocks.2.downsamplers.0": make_downsample_config(),
}


def get_downsample_config(module_path: str) -> DownsampleConfig:
    config = downsample_configs.get(module_path)
    return config


########## UpBlock configurations ##########
@dataclass
class UpBlockConfig:
    num_resnets: int = 3
    has_upsample: bool = True
    resnet_configs: dict = None
    upsample_config: UpsampleConfig = None


def make_upblock_config(
    num_resnets: int = 3,
    has_upsample: bool = True,
    block_id: int = None,
) -> UpBlockConfig:
    resnet_config_dict = {}
    for i in range(num_resnets):
        if block_id is not None:
            existing_path = f"up_blocks.{block_id}.resnets.{i}"
            existing_config = resnet_configs.get(existing_path)
            if existing_config is not None:
                resnet_config_dict[i] = existing_config

    upsample_config = make_upsample_config() if has_upsample else None

    return UpBlockConfig(
        num_resnets=num_resnets,
        has_upsample=has_upsample,
        resnet_configs=resnet_config_dict,
        upsample_config=upsample_config,
    )


# UpBlock configurations for SDXL Refiner
upblock_configs = {
    "up_blocks.0": make_upblock_config(
        num_resnets=3,
        has_upsample=True,
        block_id=0,
    ),
    "up_blocks.3": make_upblock_config(
        num_resnets=3,
        has_upsample=False,
        block_id=3,
    ),
}


def get_upblock_config(module_path: str) -> UpBlockConfig:
    config = upblock_configs.get(module_path)
    return config


############ DownBlock configurations ##########
@dataclass
class DownBlockConfig:
    num_resnets: int = 2
    has_downsample: bool = True
    resnet_configs: dict = None
    downsample_config: DownsampleConfig = None


def make_downblock_config(
    num_resnets: int = 2,
    has_downsample: bool = True,
    block_id: int = None,
) -> DownBlockConfig:
    # Create resnet configs for all resnets in this block
    resnet_config_dict = {}
    for i in range(num_resnets):
        if block_id is not None:
            # Look up in the global resnet_configs
            existing_path = f"down_blocks.{block_id}.resnets.{i}"
            existing_config = resnet_configs.get(existing_path)
            if existing_config is not None:
                resnet_config_dict[i] = existing_config

    downsample_config = make_downsample_config() if has_downsample else None

    return DownBlockConfig(
        num_resnets=num_resnets,
        has_downsample=has_downsample,
        resnet_configs=resnet_config_dict,
        downsample_config=downsample_config,
    )


# DownBlock configurations for SDXL Refiner
downblock_configs = {
    "down_blocks.0": make_downblock_config(
        num_resnets=2,
        has_downsample=True,
        block_id=0,
    ),
    "down_blocks.3": make_downblock_config(
        num_resnets=2,
        has_downsample=False,
        block_id=3,
    ),
}


def get_downblock_config(module_path: str) -> DownBlockConfig:
    config = downblock_configs.get(module_path)
    return config


########### TransformerBlock configurations ##########
@dataclass
class TransformerBlockConfig:
    num_attn_heads: int = 8


def make_transformerblock_config(
    num_attn_heads: int = 8,
) -> TransformerBlockConfig:
    return TransformerBlockConfig(
        num_attn_heads=num_attn_heads,
    )


# TransformerBlock configurations for SDXL Refiner
transformerblock_configs = {
    # First CrossAttnDownBlock2D - 2x Transformer2DModel - 768 channels
    # Each Transformer2DModel has 4x TransformerBlocks, each TransformerBlock has 2x Attention layers
    "down_blocks.1.attentions.0.transformer_blocks.0": make_transformerblock_config(num_attn_heads=12),
    "down_blocks.1.attentions.0.transformer_blocks.1": make_transformerblock_config(num_attn_heads=12),
    "down_blocks.1.attentions.0.transformer_blocks.2": make_transformerblock_config(num_attn_heads=12),
    "down_blocks.1.attentions.0.transformer_blocks.3": make_transformerblock_config(num_attn_heads=12),
    "down_blocks.1.attentions.1.transformer_blocks.0": make_transformerblock_config(num_attn_heads=12),
    "down_blocks.1.attentions.1.transformer_blocks.1": make_transformerblock_config(num_attn_heads=12),
    "down_blocks.1.attentions.1.transformer_blocks.2": make_transformerblock_config(num_attn_heads=12),
    "down_blocks.1.attentions.1.transformer_blocks.3": make_transformerblock_config(num_attn_heads=12),
    # Second CrossAttnDownBlock2D - 2x Transformer2DModel - 1536 channels
    # Each Transformer2DModel has 4x TransformerBlocks
    "down_blocks.2.attentions.0.transformer_blocks.0": make_transformerblock_config(num_attn_heads=24),
    "down_blocks.2.attentions.0.transformer_blocks.1": make_transformerblock_config(num_attn_heads=24),
    "down_blocks.2.attentions.0.transformer_blocks.2": make_transformerblock_config(num_attn_heads=24),
    "down_blocks.2.attentions.0.transformer_blocks.3": make_transformerblock_config(num_attn_heads=24),
    "down_blocks.2.attentions.1.transformer_blocks.0": make_transformerblock_config(num_attn_heads=24),
    "down_blocks.2.attentions.1.transformer_blocks.1": make_transformerblock_config(num_attn_heads=24),
    "down_blocks.2.attentions.1.transformer_blocks.2": make_transformerblock_config(num_attn_heads=24),
    "down_blocks.2.attentions.1.transformer_blocks.3": make_transformerblock_config(num_attn_heads=24),
    # First CrossAttnUpBlock2D - 3x Transformer2DModel - 1536 channels
    # Each Transformer2DModel has 4x TransformerBlocks
    "up_blocks.1.attentions.0.transformer_blocks.0": make_transformerblock_config(num_attn_heads=24),
    "up_blocks.1.attentions.0.transformer_blocks.1": make_transformerblock_config(num_attn_heads=24),
    "up_blocks.1.attentions.0.transformer_blocks.2": make_transformerblock_config(num_attn_heads=24),
    "up_blocks.1.attentions.0.transformer_blocks.3": make_transformerblock_config(num_attn_heads=24),
    "up_blocks.1.attentions.1.transformer_blocks.0": make_transformerblock_config(num_attn_heads=24),
    "up_blocks.1.attentions.1.transformer_blocks.1": make_transformerblock_config(num_attn_heads=24),
    "up_blocks.1.attentions.1.transformer_blocks.2": make_transformerblock_config(num_attn_heads=24),
    "up_blocks.1.attentions.1.transformer_blocks.3": make_transformerblock_config(num_attn_heads=24),
    "up_blocks.1.attentions.2.transformer_blocks.0": make_transformerblock_config(num_attn_heads=24),
    "up_blocks.1.attentions.2.transformer_blocks.1": make_transformerblock_config(num_attn_heads=24),
    "up_blocks.1.attentions.2.transformer_blocks.2": make_transformerblock_config(num_attn_heads=24),
    "up_blocks.1.attentions.2.transformer_blocks.3": make_transformerblock_config(num_attn_heads=24),
    # Second CrossAttnUpBlock2D - 3x Transformer2DModel - 768 channels
    # Each Transformer2DModel has 4x TransformerBlocks
    "up_blocks.2.attentions.0.transformer_blocks.0": make_transformerblock_config(num_attn_heads=12),
    "up_blocks.2.attentions.0.transformer_blocks.1": make_transformerblock_config(num_attn_heads=12),
    "up_blocks.2.attentions.0.transformer_blocks.2": make_transformerblock_config(num_attn_heads=12),
    "up_blocks.2.attentions.0.transformer_blocks.3": make_transformerblock_config(num_attn_heads=12),
    "up_blocks.2.attentions.1.transformer_blocks.0": make_transformerblock_config(num_attn_heads=12),
    "up_blocks.2.attentions.1.transformer_blocks.1": make_transformerblock_config(num_attn_heads=12),
    "up_blocks.2.attentions.1.transformer_blocks.2": make_transformerblock_config(num_attn_heads=12),
    "up_blocks.2.attentions.1.transformer_blocks.3": make_transformerblock_config(num_attn_heads=12),
    "up_blocks.2.attentions.2.transformer_blocks.0": make_transformerblock_config(num_attn_heads=12),
    "up_blocks.2.attentions.2.transformer_blocks.1": make_transformerblock_config(num_attn_heads=12),
    "up_blocks.2.attentions.2.transformer_blocks.2": make_transformerblock_config(num_attn_heads=12),
    "up_blocks.2.attentions.2.transformer_blocks.3": make_transformerblock_config(num_attn_heads=12),
    # MidBlock2DCrossAttention - 1x Transformer2DModel - 1536 channels, 16x16 resolution
    # The Transformer2DModel has 4x TransformerBlocks
    "mid_block.attentions.0.transformer_blocks.0": make_transformerblock_config(num_attn_heads=24),
    "mid_block.attentions.0.transformer_blocks.1": make_transformerblock_config(num_attn_heads=24),
    "mid_block.attentions.0.transformer_blocks.2": make_transformerblock_config(num_attn_heads=24),
    "mid_block.attentions.0.transformer_blocks.3": make_transformerblock_config(num_attn_heads=24),
}


def get_transformerblock_config(module_path: str) -> TransformerBlockConfig:
    config = transformerblock_configs.get(module_path)
    return config


########## TransformerModel configurations ##########
@dataclass
class Transformer2DModelConfig:
    num_transformer_blocks: int = 4
    norm: NormConfig = None


def make_transformer2dmodel_config(
    norm_sharded: bool = True,
    norm_out_blocks: int = 1,
    norm_grid: Optional[Tuple[int, int]] = (8, 8),
    eps: float = 1e-6,
) -> Transformer2DModelConfig:
    return Transformer2DModelConfig(
        norm=make_norm_config(norm_sharded, norm_out_blocks, norm_grid, eps=eps),
    )


# Transformer2DModel configurations for SDXL Refiner
def get_transformer2dmodel_config(module_path: str) -> Transformer2DModelConfig:
    config = make_transformer2dmodel_config()
    return config

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for YOLO26 TTNN implementation.

Includes:
- BatchNorm folding into Conv weights
- Sharding utilities
- Common layer wrappers
"""

import math
import torch
import ttnn


def fold_bn_to_conv_weights_bias(conv_weight, bn_weight, bn_bias, bn_running_mean, bn_running_var, eps=1e-5):
    """
    Fold BatchNorm parameters into Conv2d weights and bias.

    This optimization eliminates the BatchNorm layer at inference time by
    absorbing its parameters into the preceding convolution.

    Args:
        conv_weight: Conv2d weight tensor [out_ch, in_ch, kH, kW]
        bn_weight: BatchNorm gamma (scale) [out_ch]
        bn_bias: BatchNorm beta (shift) [out_ch]
        bn_running_mean: BatchNorm running mean [out_ch]
        bn_running_var: BatchNorm running variance [out_ch]
        eps: BatchNorm epsilon for numerical stability

    Returns:
        Tuple of (folded_weight, folded_bias) as ttnn tensors
    """
    # Reshape BN parameters for broadcasting
    bn_weight = bn_weight.view(-1, 1, 1, 1)
    bn_bias = bn_bias.view(-1, 1, 1, 1)
    bn_running_mean = bn_running_mean.view(-1, 1, 1, 1)
    bn_running_var = bn_running_var.view(-1, 1, 1, 1)

    # Fold BN into conv: w_new = w * gamma / sqrt(var + eps)
    std = torch.sqrt(bn_running_var + eps)
    folded_weight = conv_weight * (bn_weight / std)

    # Fold bias: b_new = gamma * (b - mean) / sqrt(var + eps) + beta
    # Since conv typically has no bias before BN, b = 0
    folded_bias = bn_bias - (bn_weight * bn_running_mean / std)
    folded_bias = folded_bias.reshape(1, 1, 1, -1)

    return ttnn.from_torch(folded_weight), ttnn.from_torch(folded_bias)


def determine_num_cores(nhw: int, width: int, max_cores: int = 64) -> int:
    """
    Determine optimal number of cores for height sharding.

    Args:
        nhw: N * H * W (batch * height * width)
        width: Channel width
        max_cores: Maximum available cores

    Returns:
        Optimal number of cores to use
    """
    gcd_nhw_width = math.gcd(nhw, width)
    cores = nhw // gcd_nhw_width
    if cores > max_cores:
        for divisor in range(max_cores, 0, -1):
            if nhw % divisor == 0 and (nhw // divisor) % width == 0:
                cores = divisor
                break
    return cores


def get_core_grid_from_num_cores(num_cores: int, grid_rows: int = 8, grid_cols: int = 8):
    """
    Create CoreRangeSet from number of cores.

    Args:
        num_cores: Total number of cores to use
        grid_rows: Grid row count (default 8 for Wormhole)
        grid_cols: Grid column count (default 8 for Wormhole)

    Returns:
        ttnn.CoreRangeSet for the specified cores
    """
    rows = num_cores // grid_cols
    assert rows <= grid_rows, "Not enough cores for specified core grid"
    ranges = []
    if rows != 0:
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(grid_cols - 1, rows - 1),
            )
        )
    remainder = num_cores % grid_cols
    if remainder != 0:
        assert rows + 1 <= grid_rows, "Not enough cores for specified core grid"
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, rows),
                ttnn.CoreCoord(remainder - 1, rows),
            )
        )
    return ttnn.CoreRangeSet({*ranges})


def get_shard_layout(batch_size: int, height: int, width: int, channels: int):
    """
    Determine optimal shard layout based on tensor dimensions.

    Rules:
    - HEIGHT_SHARDED if N*H*W >> C (spatial dominant)
    - BLOCK_SHARDED if N*H*W ≈ C (balanced)
    - WIDTH_SHARDED if C >> N*H*W (channel dominant)

    Args:
        batch_size: Batch size N
        height: Feature map height H
        width: Feature map width W
        channels: Number of channels C

    Returns:
        ttnn.TensorMemoryLayout enum value
    """
    nhw = batch_size * height * width
    ratio = nhw / channels if channels > 0 else float("inf")

    if ratio > 4:
        return ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    elif ratio < 0.25:
        return ttnn.TensorMemoryLayout.WIDTH_SHARDED
    else:
        return ttnn.TensorMemoryLayout.BLOCK_SHARDED


def sharded_concat(input_tensors, num_cores: int = 56, dim: int = 3):
    """
    Perform sharded concatenation for better performance.

    Args:
        input_tensors: List of tensors to concatenate
        num_cores: Number of cores for sharding
        dim: Dimension to concatenate along

    Returns:
        Concatenated tensor with sharded memory config
    """
    shard_grid = get_core_grid_from_num_cores(num_cores=num_cores)
    in_shard_width = input_tensors[0].shape[-1]
    shard_height = (input_tensors[0].shape[2] + num_cores - 1) // num_cores

    input_sharded_memory_config = ttnn.create_sharded_memory_config_(
        (shard_height, in_shard_width),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    out_shard_width = sum(t.shape[-1] for t in input_tensors)

    # Convert inputs to sharded format
    sharded_inputs = []
    for tensor in input_tensors:
        sharded_inputs.append(ttnn.to_memory_config(tensor, input_sharded_memory_config))

    output_sharded_memory_config = ttnn.create_sharded_memory_config_(
        (shard_height, out_shard_width),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    return ttnn.concat(sharded_inputs, dim, memory_config=output_sharded_memory_config)


def safe_reshape(x: ttnn.Tensor, shape, memory_config=None) -> ttnn.Tensor:
    """
    Safely reshape tensor - converts sharded to interleaved first if needed.

    Args:
        x: Input tensor
        shape: Target shape
        memory_config: Target memory config (default L1)

    Returns:
        Reshaped tensor
    """
    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG

    if x.memory_config().is_sharded():
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(x, shape, memory_config=memory_config)


def to_nhwc(
    x: ttnn.Tensor, batch_size: int, height: int, width: int, channels: int, to_dram: bool = False
) -> ttnn.Tensor:
    """
    Reshape tensor to NHWC format.

    Args:
        x: Input tensor
        batch_size: Batch size N
        height: Height H
        width: Width W
        channels: Channels C
        to_dram: If True, prepare for to_torch() by converting to DRAM + ROW_MAJOR

    Returns:
        Tensor in NHWC format
    """
    x = safe_reshape(x, [batch_size, height, width, channels], memory_config=ttnn.L1_MEMORY_CONFIG)
    if to_dram:
        if x.memory_config().is_sharded():
            x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if x.layout == ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    return x


# Device configuration - critical for CNN models
YOLO26_L1_SMALL_SIZE = 24576  # Same as YUNet/ResNet50, for Wormhole compatibility

# Model configuration constants
YOLO26_VARIANTS = {
    "yolo26n": {
        "depth_multiple": 0.33,
        "width_multiple": 0.25,
        "channels": [64, 128, 256, 512, 1024],
    },
    "yolo26s": {
        "depth_multiple": 0.33,
        "width_multiple": 0.50,
        "channels": [64, 128, 256, 512, 1024],
    },
    "yolo26m": {
        "depth_multiple": 0.67,
        "width_multiple": 0.75,
        "channels": [64, 128, 256, 512, 1024],
    },
    "yolo26l": {
        "depth_multiple": 1.0,
        "width_multiple": 1.0,
        "channels": [64, 128, 256, 512, 1024],
    },
    "yolo26x": {
        "depth_multiple": 1.0,
        "width_multiple": 1.25,
        "channels": [64, 128, 256, 512, 1024],
    },
}

# Default input sizes
DEFAULT_INPUT_SIZE = 640
SUPPORTED_INPUT_SIZES = [320, 416, 512, 640, 1024]

# Detection head constants
NUM_CLASSES = 80  # COCO classes
MAX_DETECTIONS = 300  # End-to-end output limit

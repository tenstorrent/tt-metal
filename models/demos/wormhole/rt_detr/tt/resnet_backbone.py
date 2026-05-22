# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# PResNet-50 backbone forward pass.

import ttnn
import torch
from tt.resnet_blocks import conv_block, residual_block

_STAGE_BLOCKS = [3, 4, 6, 3]

def _stem(x, params, device):
    h, w = 640, 640
    configs = [(3, 2, 1), (3, 1, 1), (3, 1, 1)]
    for i, (k, s, p) in enumerate(configs):
        x, (h, w) = conv_block(
            x, params.stem[i], device,
            kernel_size=(k, k), stride=(s, s), padding=(p, p),
            input_height=h, input_width=w, activation="relu",
        )
    x = ttnn.max_pool2d(
        x, batch_size=x.shape[0], input_h=h, input_w=w, channels=x.shape[-1],
        kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,  
    )
    return x, h // 2, w // 2  # 160x160

def _stage(x, block_params, device, stride_first, h, w):
    for i, bp in enumerate(block_params):
        stride = stride_first if i == 0 else 1
        x, h, w = residual_block(x, bp, device, stride=stride, input_height=h, input_width=w)
    return x, h, w

def presnet50(x, params, device):
    x, h, w = _stem(x, params, device)
    x, h, w = _stage(x, params.stages[0], device, 1, h, w)   
    x, h, w = _stage(x, params.stages[1], device, 2, h, w)   
    s3 = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=x.dtype)
    x, h, w = _stage(x, params.stages[2], device, 2, h, w)   
    s4 = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=x.dtype)
    x, h, w = _stage(x, params.stages[3], device, 2, h, w)   
    s5 = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=x.dtype)
    return s3, s4, s5


def format_bias(bias_tensor, device):
    if bias_tensor is None: return None
    return ttnn.from_torch(bias_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

def width_shard_weights(weight_tensor, device, core_grid=(8, 1)):
    original_shape = weight_tensor.shape
    height = original_shape[0]
    width = original_shape[1] if len(original_shape) == 2 else original_shape[1] * original_shape[2] * original_shape[3]
    if len(original_shape) == 4: weight_tensor = weight_tensor.reshape(height, width)

    total_cores = core_grid[0] * core_grid[1]
    shard_width = width // total_cores
    
    if shard_width % 32 != 0:
        return ttnn.from_torch(weight_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(
            grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid[0]-1, core_grid[1]-1))}),
            shard_shape=[height, shard_width], shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    return ttnn.from_torch(weight_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)

def block_shard_weights(weight_tensor, device, core_grid=(4, 4)):
    original_shape = weight_tensor.shape
    height = original_shape[0]
    width = original_shape[1] if len(original_shape) == 2 else original_shape[1] * original_shape[2] * original_shape[3]
    if len(original_shape) == 4: weight_tensor = weight_tensor.reshape(height, width)

    shard_height, shard_width = height // core_grid[1], width // core_grid[0]

    if shard_height % 32 != 0 or shard_width % 32 != 0:
        return ttnn.from_torch(weight_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(
            grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid[0]-1, core_grid[1]-1))}),
            shard_shape=[shard_height, shard_width], shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    return ttnn.from_torch(weight_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)

# height sharding
def custom_height_shard_weights(weight_tensor, device, target_cores=8, max_grid_size=[8, 8]):
    original_shape = weight_tensor.shape
    height = original_shape[0]
    width = original_shape[1] if len(original_shape) == 2 else original_shape[1] * original_shape[2] * original_shape[3]
    if len(original_shape) == 4: weight_tensor = weight_tensor.reshape(height, width)

    shard_height = height // target_cores
    if shard_height % 32 != 0:
        return ttnn.from_torch(weight_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(
            grid=ttnn.num_cores_to_corerangeset(target_num_cores=target_cores, grid_size=max_grid_size, row_wise=True),
            shard_shape=[shard_height, width], shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    return ttnn.from_torch(weight_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)


def preprocess_stage(stage_params, device, shard_func, **kwargs):
    """Generic traversal to apply the chosen sharding function to a stage."""
    for block_params in stage_params:
        for conv_name in ["conv1", "conv2", "conv3"]:
            if hasattr(block_params, conv_name):
                conv = getattr(block_params, conv_name)
                conv.weight = shard_func(conv.weight, device, **kwargs)
                if hasattr(conv, 'bias'): conv.bias = format_bias(conv.bias, device)
        if hasattr(block_params, "shortcut"):
            block_params.shortcut.weight = shard_func(block_params.shortcut.weight, device, **kwargs)
            if hasattr(block_params.shortcut, 'bias'): block_params.shortcut.bias = format_bias(block_params.shortcut.bias, device)

def initialize_model_v1(params, device):
    # Stage 0 & 1: Width Sharding (e.g., 8x1 grid) - N*H*W >> C
    preprocess_stage(params.stages[0], device, width_shard_weights, core_grid=(8, 1))
    preprocess_stage(params.stages[1], device, width_shard_weights, core_grid=(8, 1))
    
    # Stage 2: Block Sharding (4x4 grid) - N*H*W ~= C
    preprocess_stage(params.stages[2], device, block_shard_weights, core_grid=(4, 4))
    
    # Stage 3: Height Sharding (8 Cores)
    preprocess_stage(params.stages[3], device, custom_height_shard_weights, target_cores=8, max_grid_size=[8, 8])
    
    return params

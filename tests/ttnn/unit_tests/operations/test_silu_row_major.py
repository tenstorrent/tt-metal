# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import math
from loguru import logger

import torch
import torch.nn as nn
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores

TILE_WIDTH = 32


@pytest.mark.parametrize(
    "input_shape",
    [
        # [2, 1280, 4, 4],  # 256x256
        [2, 640, 16, 16],
        [2, 1280, 8, 8],  # 512x512
        [2, 1280, 16, 16],
        [1, 64, 132, 10],
        [2, 128, 56, 56],
        [2, 256, 28, 28],
        [2, 512, 14, 14],
        [1, 64, 320, 320],
        [1, 768, 40, 40],
        [1, 384, 40, 40],
        [1, 1024, 20, 20],
        [1, 512, 20, 20],
        [1, 768, 20, 20],
        [1, 512, 40, 40],
        [1, 256, 80, 80],
        [1, 512, 80, 80],
        [1, 128, 160, 160],
        [1, 64, 640, 640],
        [1, 256, 160, 160],
        [1, 128, 320, 320],
        [2, 64, 320, 320],
        [2, 768, 40, 40],
        [2, 384, 40, 40],
        [2, 1024, 20, 20],
        [2, 512, 20, 20],
        [2, 768, 20, 20],
        [2, 512, 40, 40],
        [2, 256, 80, 80],
        [2, 512, 80, 80],
        [2, 128, 160, 160],
        [2, 64, 640, 640],
        [2, 256, 160, 160],
        [2, 128, 320, 320],
        [4, 64, 320, 320],
        [4, 768, 40, 40],
        [4, 384, 40, 40],
        [4, 1024, 20, 20],
        [4, 512, 20, 20],
        [4, 768, 20, 20],
        [4, 512, 40, 40],
        [4, 256, 80, 80],
        [4, 512, 80, 80],
        [4, 128, 160, 160],
        [4, 64, 640, 640],
        [4, 256, 160, 160],
        [4, 128, 320, 320],
        [8, 64, 320, 320],
        [8, 768, 40, 40],
        [8, 384, 40, 40],
        [8, 1024, 20, 20],
        [8, 512, 20, 20],
        [8, 768, 20, 20],
        [8, 512, 40, 40],
        [8, 256, 80, 80],
        [8, 512, 80, 80],
        [8, 128, 160, 160],
        [8, 64, 640, 640],
        [8, 256, 160, 160],
        [8, 128, 320, 320],
        [16, 64, 320, 320],
        [16, 768, 40, 40],
        [16, 384, 40, 40],
        [16, 1024, 20, 20],
        [16, 512, 20, 20],
        [16, 768, 20, 20],
        [16, 512, 40, 40],
        [16, 256, 80, 80],
        [16, 512, 80, 80],
        [16, 128, 160, 160],
        [16, 64, 640, 640],
        [16, 256, 160, 160],
        [16, 128, 320, 320],
        [32, 64, 320, 320],
        [32, 768, 40, 40],
        [32, 384, 40, 40],
        [32, 1024, 20, 20],
        [32, 512, 20, 20],
        [32, 768, 20, 20],
        [32, 512, 40, 40],
        [32, 256, 80, 80],
        [32, 512, 80, 80],
        [32, 128, 160, 160],
        [32, 64, 640, 640],
        [32, 256, 160, 160],
        [32, 128, 320, 320],
    ],
)
@pytest.mark.parametrize("shard_strategy", [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.BLOCK])
def test_silu_multi_core(device, input_shape, shard_strategy):
    ## input shape is N C H W
    batch_size, num_channels, height, width = input_shape
    torch.manual_seed(0)
    input = torch.rand(input_shape, dtype=torch.bfloat16)

    ## golden reference using torch
    torch_silu = torch.nn.SiLU()
    torch_result = torch_silu(input)

    ## permute to N H W C, which is what the upsample op expects
    tt_input = input.permute(0, 2, 3, 1)

    num_bytes = 2  ## only BFLOAT16 is supported

    ## calculate ncores, corresponding grid_size and in_shard_shape based on the input_shape
    ncores = None
    max_grid_size = (device.compute_with_storage_grid_size().y, device.compute_with_storage_grid_size().x)
    if shard_strategy == ttnn.ShardStrategy.HEIGHT:
        ## nsticks per shard should be divisible by in_w
        max_nshards = min(batch_size * height, max_grid_size[0] * max_grid_size[1])
        nshards = max_nshards
        while nshards > 0:
            if batch_size * height % nshards == 0:
                break
            nshards -= 1
        ncores = nshards
    elif shard_strategy == ttnn.ShardStrategy.BLOCK:
        max_nshards_h = min(batch_size * height, max_grid_size[0])  ## height along NHW
        max_nshards_w = min(num_channels, max_grid_size[1])  ## width along C
        ## find nshards_h along NHW
        nshards_h = max_nshards_h
        while nshards_h > 0:
            if batch_size * height % nshards_h == 0:
                break
            nshards_h -= 1
        ## find nshards_w along C
        nshards_w = max_nshards_w
        while nshards_w > 0:
            ## make sure: 1. nshards_w divides num_channels, and 2. shard_shape[1] is aligned to 32B
            if num_channels % nshards_w == 0 and math.ceil(num_channels * num_bytes / nshards_w) % TILE_WIDTH == 0:
                break
            nshards_w -= 1
        if nshards_w == 0 or nshards_h == 0:
            raise ValueError("nshards_h or nshards_w is 0")
        ncores = (nshards_h, nshards_w)

    shard_grid = get_shard_grid_from_num_cores(ncores, device)
    shard_orientation = ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR

    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        tensor_memory_layout = ttnn.types.TensorMemoryLayout.BLOCK_SHARDED
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        tensor_memory_layout = ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED

    ## input shard
    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        shard_height = math.ceil(batch_size * height * width / ncores[0])
        shard_width = math.ceil(num_channels / ncores[1])
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        shard_height = math.ceil(batch_size * height * width / ncores)
        shard_width = num_channels

    shard_shape = (shard_height, shard_width)
    logger.debug(f"shard_shape={shard_shape}")
    shard_spec = ttnn.experimental.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, False)
    in_sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.types.BufferType.L1, shard_spec)

    logger.debug(f"in_shard_mem_config: {in_sharded_mem_config}")
    logger.debug(f"ncore --> {ncores}")

    ## ttnn uses NHWC, so need to set scale_factor_c = 1
    input_tensor = ttnn.from_torch(tt_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=in_sharded_mem_config)
    output_tensor = ttnn.silu(input_tensor, memory_config=in_sharded_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)

    ## compare the results
    torch_result = torch_result.permute(0, 2, 3, 1)
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_result, output_tensor, 0.999)
    logger.info(pcc_msg)
    assert passing

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import math

from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import reshard_to


class upsample_nearest2d:
    def __init__(self, input_height, input_width, in_channels, scale_factor):
        batch_size = 2
        num_bytes = 2  ## only BFLOAT16 is supported

        self.scale_factor = scale_factor
        TILE_WIDTH = 32

        ## calculate ncores, corresponding grid_size and in_shard_shape based on the input_shape
        ncores = None
        max_grid_size = (8, 8)
        max_nshards_h = min(batch_size * input_height * input_width, max_grid_size[0])  ## height along NHW
        max_nshards_w = min(in_channels, max_grid_size[1])  ## width along C
        ## find nshards_h along NHW
        nshards_h = max_nshards_h
        while nshards_h > 0:
            if batch_size * input_height % nshards_h == 0:
                break
            nshards_h -= 1
        ## find nshards_w along C
        nshards_w = max_nshards_w
        while nshards_w > 0:
            ## make sure: 1. nshards_w divides num_channels, and 2. shard_shape[1] is aligned to 32B
            if in_channels % nshards_w == 0 and math.ceil(in_channels * num_bytes / nshards_w) % TILE_WIDTH == 0:
                break
            nshards_w -= 1
        if nshards_w == 0 or nshards_h == 0:
            raise ValueError("nshards_h or nshards_w is 0")
        ncores = (nshards_h, nshards_w)
        assert ncores == max_grid_size

        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(max_grid_size[1] - 1, max_grid_size[0] - 1),
                )
            }
        )

        shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
        tensor_memory_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        ## input shard
        shard_height = math.ceil(batch_size * input_height * input_width / ncores[0])
        shard_width = math.ceil(in_channels / ncores[1])

        shard_shape = (shard_height, shard_width)
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation, False)
        self.in_sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.BufferType.L1, shard_spec)

        ## output shard
        shard_height = shard_height * scale_factor * scale_factor
        shard_shape = (shard_height, shard_width)
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation, False)
        self.out_sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.BufferType.L1, shard_spec)

    def __call__(self, input):
        if input.memory_config() != self.in_sharded_mem_config:
            input = ttnn.to_memory_config(input, memory_config=self.in_sharded_mem_config)
        up_output = ttnn.upsample(input, self.scale_factor, memory_config=self.out_sharded_mem_config)
        return up_output

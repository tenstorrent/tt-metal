# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import math

import torch
import torch.nn as nn
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "input_shapes",
    [
        [2, 1280, 4, 4],  # 256x256
        [2, 1280, 8, 8],
        [2, 640, 16, 16],
        [2, 1280, 8, 8],  # 512x512
        [2, 1280, 16, 16],
        [2, 1280, 16, 16],
    ],
)
@pytest.mark.parametrize("scale_h", [2])
@pytest.mark.parametrize("scale_w", [2])
def test_upsample_single_core(device, input_shapes, scale_h, scale_w):
    batch_size, height, width, num_channels = input_shapes

    torch.manual_seed(0)
    input = torch.rand(input_shapes, dtype=torch.bfloat16)
    tt_input = input.permute(0, 3, 1, 2)

    scale_factor = (scale_h, scale_w)
    m = nn.Upsample(scale_factor=scale_factor, mode="nearest")
    torch_result = m(tt_input)
    torch_result = torch_result.permute(0, 2, 3, 1)

    ## ttnn uses NHWC, so need to set scale_factor_c = 1
    scale_factor = (scale_h, scale_w, 1)
    input_tensor = ttnn.from_torch(input, device=device)
    output_tensor = ttnn.upsample(input_tensor, scale_factor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_result, output_tensor)

    allclose = torch.allclose(output_tensor, torch_result)
    isclose = torch.all(torch.isclose(output_tensor, torch_result))
    isequal = torch.equal(output_tensor, torch_result)

    assert allclose
    assert isclose
    assert isequal


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "input_shape",
    [
        [2, 1280, 4, 4],  # 256x256
        [2, 1280, 8, 8],
        [2, 640, 16, 16],
        [2, 1280, 8, 8],  # 512x512
        [2, 1280, 16, 16],
        [2, 1280, 16, 16],
    ],
)
@pytest.mark.parametrize("scale_h", [2])
@pytest.mark.parametrize("scale_w", [2])
@pytest.mark.parametrize("shard_strategy", [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.BLOCK])
def test_upsample_multi_core(device, input_shape, scale_h, scale_w, shard_strategy):
    ## input shape is N C H W
    batch_size, num_channels, height, width = input_shape
    torch.manual_seed(0)
    input = torch.rand(input_shape, dtype=torch.bfloat16)

    ## golden reference using torch
    scale_factor = (scale_h, scale_w)
    torch_upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
    torch_result = torch_upsample(input)
    output_shape = torch_result.shape

    ## calculated ttnn result

    ## permute to N H W C
    tt_input = input.permute(0, 2, 3, 1)

    ## calculate ncores, corresponding grid_size and in_shard_shape based on the input_shape
    max_grid_size = (9, 12)  ## (y, x)
    if shard_strategy == ttnn.ShardStrategy.HEIGHT:
        ## nsticks per shard should be divisible by in_w
        max_nshards = min(batch_size * height, max_grid_size[0] * max_grid_size[1])
        nshards = max_nshards
        while nshards > 0:
            if batch_size * height % nshards == 0:
                break
            nshards -= 1

        ncores = nshards
        if ncores % max_grid_size[1] == 0:
            grid_size = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
        else:
            if ncores < max_grid_size[1]:
                grid_size = ttnn.CoreGrid(y=1, x=ncores)
            else:
                grid1_size = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
                grid2_size = ttnn.CoreGrid(y=ncores // max_grid_size[1] + 1, x=ncores % max_grid_size[1])
                grid_size = (grid1_size, grid2_size)

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
            if num_channels % nshards_w == 0:
                break
            nshards_w -= 1

        if nshards_w == 0 or nshards_h == 0:
            raise ValueError("nshards_h or nshards_w is 0")

        ## calculate grid_size and shard_shape
        grid_size = ttnn.CoreGrid(y=nshards_h, x=nshards_w)

    in_sharded_mem_config = ttnn.create_sharded_memory_config(
        [batch_size, height, width, num_channels],
        grid_size,
        shard_strategy,
        use_height_and_width_as_shard_shape=shard_strategy == ttnn.ShardStrategy.HEIGHT,
    )
    out_sharded_mem_config = ttnn.create_sharded_memory_config(
        [batch_size, height * scale_h, width * scale_w, num_channels],
        grid_size,
        shard_strategy,
        use_height_and_width_as_shard_shape=shard_strategy == ttnn.ShardStrategy.HEIGHT,
    )

    ## ttnn uses NHWC, so need to set scale_factor_c = 1
    scale_factor = (scale_h, scale_w, 1)
    input_tensor = ttnn.from_torch(tt_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=in_sharded_mem_config)
    output_tensor = ttnn.upsample(input_tensor, scale_factor, memory_config=out_sharded_mem_config)
    output_tensor = ttnn.to_memory_config(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    ## compare the results
    torch_result = torch_result.permute(0, 2, 3, 1)
    assert_with_pcc(torch_result, output_tensor)

    allclose = torch.allclose(output_tensor, torch_result)
    isclose = torch.all(torch.isclose(output_tensor, torch_result))
    isequal = torch.equal(output_tensor, torch_result)

    assert allclose
    assert isclose
    assert isequal

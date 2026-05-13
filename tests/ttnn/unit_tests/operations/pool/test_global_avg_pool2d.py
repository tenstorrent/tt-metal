# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import math
from models.common.utility_functions import skip_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn


@pytest.mark.parametrize(
    "input_shape",
    (([1, 2048, 7, 7], ([1, 64, 1, 32]))),
    ids=["resnet50_unpadded", "tile_divisible"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
def test_run_average_pool2d(
    input_shape,
    dtype,
    device,
):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(input_shape)
    torch_output_tensor = torch.nn.functional.adaptive_avg_pool2d(torch_input_tensor, (1, 1))

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))  # ttnn operates on channels-last tensors
    input_tensor = ttnn.from_torch(input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.global_avg_pool2d(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 144, 7, 7],  # EfficientNet case: 144 channels (not tile-aligned)
        [12, 144, 56, 56],  # Larger batch with non-aligned channels
        [1, 48, 14, 14],  # 48 channels (padded to 64)
        [1, 80, 14, 14],  # 80 channels (padded to 96)
        [1, 112, 14, 14],  # 112 channels (padded to 128)
    ),
    ids=["efficientnet_144", "efficientnet_144_batch12", "mobilenet_48", "mobilenet_80", "mobilenet_112"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
def test_global_avg_pool2d_non_tile_aligned(
    input_shape,
    dtype,
    device,
):
    """
    Regression test for non-tile-aligned channel dimensions.

    This test ensures that global_avg_pool2d correctly handles tensors where
    the channel dimension is not a multiple of 32 (tile size). Previously,
    the operation would return the padded channel count in the output shape
    instead of the logical (unpadded) channel count.

    Example: Input with 144 channels would incorrectly output 160 channels
    (144 padded to next multiple of 32) instead of preserving 144 channels.
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(input_shape)
    torch_output_tensor = torch.nn.functional.adaptive_avg_pool2d(torch_input_tensor, (1, 1))

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))  # ttnn operates on channels-last tensors
    input_tensor = ttnn.from_torch(input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.global_avg_pool2d(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "h_w, channels, grid_end",
    [
        (49, 1280, (7, 0)),  # 1280 ch (EfficientNet head), 8-way ND shard; h_w=49 is not tile-aligned
        (25, 320, (4, 0)),
        (20, 64, (0, 0)),
    ],
)
def test_global_avg_pool2d_nd_sharded_row_major_non_tile_aligned_h(device, h_w, channels, grid_end):
    torch.manual_seed(0)

    end_x, end_y = grid_end
    grid_size = device.compute_with_storage_grid_size()
    if end_x >= grid_size.x or end_y >= grid_size.y:
        pytest.skip(f"Device grid {grid_size.x}x{grid_size.y} is smaller than required {end_x + 1}x{end_y + 1}")

    num_cores = (end_x + 1) * (end_y + 1)
    assert channels % num_cores == 0, "channels must split evenly across cores"
    assert h_w % 32 != 0, "test targets non-tile-aligned H*W"

    torch_input = torch.randn(1, 1, h_w, channels, dtype=torch.bfloat16)
    torch_output = torch_input.mean(dim=2, keepdim=True)

    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))])
    shard_shape = ttnn.Shape([1, 1, h_w, channels // num_cores])
    memory_config = ttnn.MemoryConfig(ttnn.BufferType.L1, ttnn.NdShardSpec(shard_shape, grid))

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
        device=device,
    )

    output_tensor = ttnn.global_avg_pool2d(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output, output_tensor, 0.99)

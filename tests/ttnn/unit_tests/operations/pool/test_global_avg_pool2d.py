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


# Regression test for https://github.com/tenstorrent/tt-metal/issues/43843.
# The reduce-op validation added in PR #43253 rejects the input-derived
# output_mem_config carried into ttnn::prim::reduce, even though
# build_reduce_output_tensor_spec overwrites shard_shape[-2] to 1 and rounds
# up to tile alignment for an H-reduce. This mirrors the failing call at
# models/experimental/efficientnetb0/tt/efficientnetb0.py:483, where the input
# is [1, 1, 49, 1280] ROW_MAJOR with an ND shard spec whose shard_shape[-2] = 49
# (= 7*7, the spatial-flattened H of the EfficientNet-B0 feature map).
#
# The TT_FATAL is host-side (validate_on_program_cache_miss) and topology-
# independent, so each of these parameter sets reproduces on any single-chip
# WH (N150) as well as on N300. The single_core_min variant is included so
# the repro survives the most aggressive harvesting configs (only core (0,0)
# is required) and so it never depends on enough cores existing in row 0.
@pytest.mark.parametrize(
    "h_w, channels, grid_end",
    [
        (49, 1280, (7, 0)),  # EfficientNet-B0 final feature map: matches the failing call
        (25, 320, (3, 0)),  # 5x5 spatial, modest 4-core grid
        (49, 64, (0, 0)),  # single-core minimum-hardware variant
    ],
    ids=["efficientnetb0_7x7_1280", "5x5_320_4core", "single_core_min"],
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

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("h", [20])
@pytest.mark.parametrize("w", [4])
def test_concat(device, h, w):
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=0)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.concat([input_tensor_a, input_tensor_b], dim=0)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize(
    "input_shape, shard_shape, output_shard_shape, shard_grid",
    (
        (
            (1, 1, 16, 16),
            (8, 16),
            (8, 32),
            ttnn.experimental.tensor.CoreRangeSet(
                {
                    ttnn.experimental.tensor.CoreRange(
                        ttnn.experimental.tensor.CoreCoord(0, 0), ttnn.experimental.tensor.CoreCoord(0, 1)
                    )
                }
            ),
        ),
        (
            (1, 1, 160, 32),
            (80, 32),
            (80, 64),
            ttnn.experimental.tensor.CoreRangeSet(
                {
                    ttnn.experimental.tensor.CoreRange(
                        ttnn.experimental.tensor.CoreCoord(0, 0), ttnn.experimental.tensor.CoreCoord(0, 1)
                    )
                }
            ),
        ),
    ),
)
def test_sharded_concat(device, input_shape, shard_shape, output_shard_shape, shard_grid):
    input_sharded_memory_config = ttnn.create_sharded_memory_config(
        shard_shape, core_grid=shard_grid, strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
    )
    output_sharded_memory_config = ttnn.create_sharded_memory_config(
        output_shard_shape,
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.concat([torch_input_tensor, torch_input_tensor], dim=3)

    input_tensor_a = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor_a = ttnn.to_memory_config(input_tensor_a, input_sharded_memory_config)
    input_tensor_b = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor_b = ttnn.to_memory_config(input_tensor_b, input_sharded_memory_config)

    output = ttnn.concat([input_tensor_a, input_tensor_b], dim=3, memory_config=output_sharded_memory_config)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)

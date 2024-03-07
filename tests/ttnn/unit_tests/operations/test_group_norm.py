# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


# @pytest.mark.parametrize("h", [32])
# @pytest.mark.parametrize("w", [64])
# @pytest.mark.parametrize("num_groups", [2])
# def test_group_norm(device, h, w, num_groups):
#     torch.manual_seed(0)

#     torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
#     torch_output_tensor = torch.nn.functional.group_norm(torch_input_tensor, num_groups)

#     input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
#     output_tensor = ttnn.group_norm(input_tensor, num_groups=num_groups)
#     output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
#     output_tensor = ttnn.from_device(output_tensor)
#     output_tensor = ttnn.to_torch(output_tensor)

#     assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


# @pytest.mark.parametrize("h", [32])
# @pytest.mark.parametrize("w", [64])
# @pytest.mark.parametrize("num_groups", [2])
# def test_group_norm_with_weight_and_bias(device, h, w, num_groups):
#     torch.manual_seed(0)

#     torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
#     torch_weight = torch.rand((w,), dtype=torch.bfloat16)
#     torch_bias = torch.rand((w,), dtype=torch.bfloat16)
#     torch_output_tensor = torch.nn.functional.group_norm(
#         torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
#     )

#     input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
#     weight = ttnn.from_torch(torch_weight, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
#     bias = ttnn.from_torch(torch_bias, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

#     output_tensor = ttnn.group_norm(input_tensor, num_groups=num_groups, weight=weight, bias=bias)
#     output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
#     output_tensor = ttnn.from_device(output_tensor)
#     output_tensor = ttnn.to_torch(output_tensor)

#     assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("N", [2])
@pytest.mark.parametrize("C", [320])
@pytest.mark.parametrize("H", [32])
@pytest.mark.parametrize("W", [32])
@pytest.mark.parametrize("num_groups", [32])
def test_group_norm_with_height_sharded(device, N, C, H, W, num_groups):
    torch.manual_seed(0)

    grid_size = ttnn.CoreGrid(y=1, x=8)

    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, num_groups)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, num_groups)

    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    sharded_mem_config = ttnn.create_sharded_memory_config(
        input_tensor.shape,
        grid_size,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.COLUMN_MAJOR,
    )
    input_tensor = ttnn.to_memory_config(input_tensor, sharded_mem_config)

    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        weight=gamma_t,
        bias=beta_t,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
    )

    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("N", [1])
@pytest.mark.parametrize("C", [1280])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("W", [16])
@pytest.mark.parametrize("num_groups", [32])
def test_group_norm_with_block_sharded(device, N, C, H, W, num_groups):
    torch.manual_seed(0)

    grid_size = ttnn.CoreGrid(y=8, x=4)

    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, num_groups)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, num_groups)

    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # sharded_mem_config = ttnn.create_sharded_memory_config(
    #     input_tensor.shape,
    #     grid_size,
    #     ttnn.ShardStrategy.BLOCK,
    #     ttnn.ShardOrientation.COLUMN_MAJOR,
    # )
    grid_coord = ttnn.experimental.tensor.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.experimental.tensor.CoreRangeSet(
        {ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), grid_coord)}
    )
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.experimental.tensor.ShardSpec(
        shard_grid, shard_shape, ttnn.experimental.tensor.ShardOrientation.COL_MAJOR, False
    )
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    input_tensor = ttnn.to_memory_config(input_tensor, sharded_mem_config)

    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        weight=gamma_t,
        bias=beta_t,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
    )

    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)

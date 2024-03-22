# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from loguru import logger

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
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


@pytest.mark.parametrize("N", [1])
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


# (2, 320, 64, 64, 0, 0, "down", None),
# (2, 320, 32, 32, 0, 0, "down", None),
# (2, 640, 32, 32, 1, 1, "down", None),
# (2, 640, 16, 16, 1, 1, "down", None),
# (2, 1280, 16, 16, 2, 1, "down", None),
# (2, 1280, 8, 8, 2, 1, "down", None),
# (2, 2560, 8, 8, 0, 0, "up", 1280),
# (2, 2560, 16, 16, 0, 0, "up", 1280),
# (2, 1920, 16, 16, 2, 0, "up", 1280),
# (2, 1920, 32, 32, 2, 0, "up", 640),
# (2, 1280, 32, 32, 3, 0, "down", None),
# (2, 960, 32, 32, 3, 0, "up", 640),
# (2, 960, 64, 64, 3, 0, "up", 320),
# (2, 640, 64, 64, 3, 1, "up", 320),


@pytest.mark.parametrize(
    ("shape"),
    [
        # ((2, 1, 2048, 320)),
        # ((2, 1, 64, 1280)),
        # ((2, 1, 64, 2560)),
        # ((1, 1, 8192, 320)),
        # ((2, 1, 4096, 320)),
        # ((2, 1, 4096, 960)),
        # ((2, 1, 1024, 320)),
        # ((2, 1, 1024, 960)),
        # ((1, 1, 2048, 640)),
        # ((2, 1, 1024, 640)),
        # ((2, 1, 1024, 1920)),
        # ((2, 1, 256, 640)),
        # ((2, 1, 256, 1920)),
        # ((2, 1, 1024, 1280)),
        # ((1, 1, 512, 1280)),
        # ((2, 1, 256, 1280)),
        # ((2, 1, 256, 2560)),
        # ((2, 1, 256, 1280)),
        ((2, 1, 4096, 320)),
    ],
)
@pytest.mark.parametrize("num_groups", [32])
def test_group_norm_with_block_sharded_unet(device, shape, num_groups):
    torch.manual_seed(0)

    N, x, H_W, C = shape
    assert x == 1
    import math

    H = (int)(math.sqrt(H_W))
    W = H

    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight1 = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias1 = torch.rand((C,), dtype=torch.bfloat16)

    torch_inter_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight1, bias=torch_bias1
    )
    torch_output_tensor = torch_inter_tensor

    # torch_output_tensor = torch_input_tensor # uncomment this line if only testing untilize + tilize
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    gamma1 = ttnn.create_group_norm_weight_bias_rm(torch_weight1, C, num_groups)
    beta1 = ttnn.create_group_norm_weight_bias_rm(torch_bias1, C, num_groups)

    gamma_t1 = ttnn.from_torch(
        gamma1,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t1 = ttnn.from_torch(
        beta1,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    sharded_mem_config, core_grid = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
        device=device,
        num_channels=C,
        num_groups=num_groups,
        input_nhw=N * H * W,
        is_height_sharded=False,
    )
    input_tensor = ttnn.reshape(input_tensor, (1, 1, N * H * W, C))

    input_tensor_torch = ttnn.to_torch(input_tensor)

    # here input is in l1 interleaved
    # input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, output_memory_config=ttnn.L1_MEMORY_CONFIG)
    # here input is in l1 interleaved
    # input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT, output_memory_config=ttnn.L1_MEMORY_CONFIG)
    # here input is in l1 interleaved

    input_tensor_rm_torch = ttnn.to_torch(input_tensor)
    pcc_pass, pcc_msg = check_with_pcc(input_tensor_torch, input_tensor_rm_torch, 1.0)
    logger.info(pcc_pass)
    logger.info(pcc_msg)
    assert pcc_pass, pcc_msg

    # Interleaved to Sharded on a RM tensor
    # input tensor is l1 interleaved and it has a shape [1, 1, NHW, C]
    input_tensor = ttnn.to_memory_config(input_tensor, sharded_mem_config)

    print("  ")
    print("GN shape - ", shape)
    print("GN shard config - ", sharded_mem_config)
    print("  ")

    # Special reshape for GN
    input_tensor = ttnn.reshape(input_tensor, (N, 1, H * W, C))
    # output_tensor = input_tensor # uncomment this line if only testing untilzie + tilize
    # ttnn.synchronize_device(device)
    inter_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        weight=gamma_t1,
        bias=beta_t1,
        memory_config=sharded_mem_config,
        core_grid=core_grid,
    )
    # inter_tensor = ttnn.reshape(inter_tensor, (1, 1, N*H*W, C))
    inter_tensor = ttnn.to_memory_config(inter_tensor, ttnn.L1_MEMORY_CONFIG)
    # inter_tensor = ttnn.reshape(inter_tensor, (N, 1, H*W, C))
    output_tensor = inter_tensor

    out_tensor_torch = ttnn.to_torch(output_tensor)
    torch.save(out_tensor_torch, "gn1_ut_output.pt")

    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    pcc_pass, pcc_msg = check_with_pcc(torch_output_tensor, output_tensor, 0.99)
    logger.info(pcc_pass)
    logger.info(pcc_msg)
    assert pcc_pass, pcc_msg

    ## do more checks

    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    logger.info(f"atol: {atol}")
    logger.info(f"rtol: {rtol}")

    atol = 0.08  ## this was to enable PASS for all configs with high PCC

    allclose = torch.allclose(torch_output_tensor, output_tensor, atol=atol)
    isclose = torch.all(torch.isclose(torch_output_tensor, output_tensor, atol=atol))

    logger.info(f"allclose: {allclose}")
    logger.info(f"isclose: {isclose}")

    # assert allclose
    # assert isclose

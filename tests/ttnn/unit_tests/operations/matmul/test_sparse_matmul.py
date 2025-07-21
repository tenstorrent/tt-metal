# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

from loguru import logger
import pytest
import torch
import math
import ttnn

from models.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc


def find_max_subblock(out_block_h, out_block_w):
    max_product = 0
    best_h = 1
    best_w = 1

    for h in range(1, out_block_h + 1):
        if out_block_h % h == 0:  # h is a divisor of out_block_h
            for w in range(1, out_block_w + 1):
                if out_block_w % w == 0 and h * w <= 8:  # w is a divisor and product condition met
                    if h * w > max_product:
                        max_product = h * w
                        best_h = h
                        best_w = w
    if out_block_w > best_w:
        best_h = 1
    return best_h, best_w, max_product


# @pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.parametrize("m", [256])
@pytest.mark.parametrize("k", [256])
@pytest.mark.parametrize("n", [256])
@pytest.mark.parametrize("tile_h", [32])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("in0_sharded", [False])
@pytest.mark.parametrize("in1_sharded", [False])
@pytest.mark.parametrize("out_sharded", [False])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b])
def test_sparse_matmul_reuse_config_sharded_fd_column(
    device, m, k, n, tile_h, tile_w, in0_sharded, in1_sharded, out_sharded, in1_dtype
):
    torch.manual_seed(0)

    compute_grid_size = device.compute_with_storage_grid_size()
    b = compute_grid_size.x
    h = compute_grid_size.y

    grid_size = (b, h)

    in0 = torch.randn((2 * b, h, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((2 * b, h, k, n), dtype=torch.bfloat16)

    if in0_sharded:
        in0_memory_config = ttnn.create_sharded_memory_config(
            (b, h, m, k),
            core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    else:
        in0_memory_config = ttnn.DRAM_MEMORY_CONFIG
    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((tile_h, 32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
    )

    if in1_sharded:
        in1_memory_config = ttnn.create_sharded_memory_config(
            (b, h, k, n),
            core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    else:
        in1_memory_config = ttnn.DRAM_MEMORY_CONFIG
    in1_t = ttnn.from_torch(
        in1,
        tile=ttnn.Tile((32, tile_w)),
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
    )

    sparsity = torch.ones((2 * b, h), dtype=torch.bfloat16)
    sparsity[0, 0] = 0
    sparsity_t = ttnn.from_torch(
        sparsity,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    out_block_h = m // tile_h
    out_block_w = n // tile_w
    out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=k // 32,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
    )
    if out_sharded:
        out_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG
    # override the tile width for later ops
    if out_sharded and tile_h <= 16:
        output_tile = ttnn.Tile([tile_h, 32])
    else:
        output_tile = ttnn.Tile([tile_h, tile_w])
    output_tile = ttnn.Tile([tile_h, tile_w])
    output_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity_t,
        1,
        program_config=program_config,
        memory_config=out_mem_config,
        output_tile=output_tile,
    )
    output_tensor = ttnn.to_torch(output_t)
    pt_out = torch.matmul(in0, in1)
    if in1_dtype == ttnn.bfloat8_b:
        expected_pcc = 0.999
    elif in1_dtype == ttnn.bfloat4_b:
        expected_pcc = 0.993
    else:
        expected_pcc = 1.0

    pcc_passed, pcc_message = comp_pcc(pt_out, output_tensor, expected_pcc)
    logger.info(pcc_message)
    assert_with_pcc(pt_out, output_tensor, expected_pcc)

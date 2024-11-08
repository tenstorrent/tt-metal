# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_wormhole_b0
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero, roundup32
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
import random
import math

random.seed(10)


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


from models.utility_functions import is_wormhole_b0, is_grayskull, is_wormhole_b0, is_blackhole


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "out_sharded, M, K, N, activation, in0_dtype, in1_dtype, fidelity, packer_l1_acc, fp32_acc_mode",
    [
        # 32, 2304, 3840
        (
            True,
            32,
            2304,
            3840,
            None,
            ttnn.bfloat16,
            ttnn.bfloat4_b,
            ttnn.MathFidelity.LoFi,
            True,
            True,
        ),
        # 256, 8192, 8192
        (
            True,
            256,
            1024,
            8192,
            None,
            ttnn.bfloat16,
            ttnn.bfloat4_b,
            ttnn.MathFidelity.HiFi4,
            True,
            True,
        ),
    ],
)
@pytest.mark.parametrize(
    "grid",
    [
        # (3, 1),
        # (8, 1),
        # (8, 2),
        (8, 3),
        (8, 4),
        # (8, 8)
    ],
)
@pytest.mark.parametrize(
    "use_arbitrary_cores",
    [True, False],
)
def test_multi_core_matmul_1d_wh(
    device,
    in0_dtype,
    in1_dtype,
    fidelity,
    out_sharded,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    M,
    K,
    N,
    activation,
    grid,
    use_arbitrary_cores,
    function_level_defaults,
):
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]
    num_cores = grid[0] * grid[1]

    in0_block_h = M // ttnn.TILE_SIZE
    in0_block_w = K // num_cores // ttnn.TILE_SIZE
    out_block_h = M // ttnn.TILE_SIZE
    out_block_w = N // num_cores // ttnn.TILE_SIZE

    num_blocks_y = (M // ttnn.TILE_SIZE - 1) // out_block_h + 1
    num_blocks_x = (N // ttnn.TILE_SIZE - 1) // out_block_w + 1
    num_blocks_total = num_blocks_y * num_blocks_x

    if num_blocks_total != num_cores:
        pytest.skip(f"num_blocks_total {num_blocks_total} != num_cores {num_cores}")

    out_subblock_h = 1
    out_subblock_w = 4
    while out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1

    logger.debug("in0 block h w " + str(in0_block_h) + " " + str(in0_block_w))
    logger.debug("in1 block h w " + str(in0_block_w) + " " + str(out_block_w))
    logger.debug("out block h w " + str(out_block_h) + " " + str(out_block_w))
    logger.debug("out subblock h w " + str(out_subblock_h) + " " + str(out_subblock_w))

    if use_arbitrary_cores:
        # x, y
        CORE_RANGE = [(x, y) for y in range(grid[1]) for x in range(grid[0])]
        random.shuffle(CORE_RANGE)

        core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in CORE_RANGE
            ]
        )
    else:
        core_range_set = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid[0] - 1, grid[1] - 1),
                ),
            }
        )

    in0_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_set,
            [M, K // num_cores],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    in1_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_set,
            [K, N // num_cores],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    output_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_set,
            [M, N // num_cores],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    in0 = torch.randn(in0_shape)
    in1 = torch.randn(in1_shape)
    bias = torch.randn(bias_shape)

    in0_t = ttnn.from_torch(
        in0,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=in0_dtype,
        memory_config=in0_sharded_mem_config,
    )
    in1_t = ttnn.from_torch(
        in1,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=in1_dtype,
        memory_config=in1_sharded_mem_config,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=fp32_acc_mode,
        packer_l1_acc=packer_l1_acc,
    )

    output_t = ttnn.matmul(
        in0_t,
        in1_t,
        program_config=program_config,
        memory_config=output_sharded_mem_config,
        compute_kernel_config=compute_kernel_config,
    )
    tt_out = ttnn.to_torch(output_t)

    pt_out = in0 @ in1
    if has_bias:
        pt_out += bias

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing

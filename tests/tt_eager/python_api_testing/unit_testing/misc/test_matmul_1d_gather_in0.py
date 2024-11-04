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
    "out_sharded, M, K, N, activation, dtype, fidelity, packer_l1_acc, fp32_acc_mode",
    [
        # 32, 2304, 3840
        (
            True,
            32,
            2304,
            3840,
            None,
            ttnn.bfloat16,
            ttnn.MathFidelity.HiFi2,
            True,
            True,
        ),
    ],
)
@pytest.mark.parametrize(
    "grid_size",
    [
        # (3, 1),
        # (8, 1),
        # (8, 2),
        (8, 3),
        # (8, 4),
        # (8, 8)
    ],
)
def test_multi_core_matmul_1d_wh(
    device,
    dtype,
    fidelity,
    out_sharded,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    M,
    K,
    N,
    activation,
    grid_size,
    function_level_defaults,
):
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]
    num_cores = grid_size[0] * grid_size[1]

    in0_block_h = M // 32
    in0_block_w = K // num_cores // 32
    out_block_h = M // 32
    out_block_w = N // num_cores // 32

    out_subblock_h = 1
    out_subblock_w = 8
    while out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1

    logger.debug("in0 block h w " + str(in0_block_h) + " " + str(in0_block_w))
    logger.debug("in1 block h w " + str(in0_block_w) + " " + str(out_block_w))
    logger.debug("out block h w " + str(out_block_h) + " " + str(out_block_w))
    logger.debug("out subblock h w " + str(out_subblock_h) + " " + str(out_subblock_w))

    in0_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=in0_shape,
        core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    in1_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=in1_shape,
        core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    in0 = torch.randn(in0_shape)
    in1 = torch.randn(in1_shape)
    bias = torch.randn(bias_shape)

    # FIXME: Because .set_globally_allocated address is broken, need to add padding to input tensor
    padded_K = K * (num_cores - 1)
    input_tensor_padding = torch.randn([1, 1, M, padded_K]).float()
    in0_padded_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input_tensor_padding.shape,
        core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    _ = ttnn.from_torch(
        input_tensor_padding,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=in0_padded_sharded_mem_config,
    )

    in0_t = ttnn.from_torch(
        in0,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=in0_sharded_mem_config,
    )
    in1_t = ttnn.from_torch(
        in1,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=in1_sharded_mem_config,
    )

    output_mem_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
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
        memory_config=output_mem_config,
        dtype=dtype,
        compute_kernel_config=compute_kernel_config,
    )
    tt_out = ttnn.to_torch(output_t)

    pt_out = in0 @ in1
    if has_bias:
        pt_out += bias

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing

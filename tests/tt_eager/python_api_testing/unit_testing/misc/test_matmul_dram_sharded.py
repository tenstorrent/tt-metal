# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import tt_lib as ttl
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_wormhole_b0
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero, roundup32
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)


def find_max_subblock(out_block_h, out_block_w):
    max_product = 0
    best_h = 1
    best_w = 1

    for h in range(1, out_block_h + 1):
        if out_block_h % h == 0:
            for w in range(1, out_block_w + 1):
                if out_block_w % w == 0 and h * w <= 8:
                    if h * w > max_product:
                        max_product = h * w
                        best_h = h
                        best_w = w
    if out_block_w > best_w:
        best_h = 1
    return best_h, best_w, max_product


def pad_to_dram_banks(num, lcm=32 * 12):
    remainder = num % lcm
    if remainder == 0:
        return num
    padding_needed = lcm - remainder
    padded_number = num + padding_needed
    return padded_number


@pytest.mark.parametrize(
    "fidelity",
    [
        ttl.tensor.MathFidelity.HiFi2,
        ttl.tensor.MathFidelity.LoFi,
    ],
    ids=["HiFi2", "LoFi"],
)
@pytest.mark.parametrize(
    "has_bias",
    [
        False,
        True,
    ],
    ids=["no_bias", "bias"],
)
@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    [
        (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ],
)
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, grid_size",
    # "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, grid_size, in0_dtype, in1_dtype, out_dtype",
    [
        (False, True, True, 32, 8192, 1280, None, (8, 1)),
        (False, True, True, 32, 8192, 4096, None, (8, 2)),
        (False, True, True, 32, 8192, 1024, None, (8, 1)),
        (False, True, True, 32, 32768, 1024, None, (8, 2)),
        # (False, True, True, 32, 4096, 6144, None, (8, 2), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
        # (False, True, True, 32, 4096, 14336, None, (8, 2), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT4_B, ttl.tensor.DataType.BFLOAT8_B),
        # (False, True, True, 32, 14336, 4096, None, (8, 2), ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT8_B),
        # (False, True, True, 32, 4096, 14336, None, (8, 2), ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT4_B, ttl.tensor.DataType.BFLOAT8_B),
    ],
)
def test_matmul_in1_dram_sharded(
    device,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    M,
    K,
    N,
    fidelity,
    has_bias,
    activation,
    grid_size,
    in0_dtype,
    in1_dtype,
    out_dtype,
    function_level_defaults,
):
    if is_grayskull() and (N == 4096 or K == 32768):
        pytest.skip("Skipping too large tensor test on Grayskull")

    if is_grayskull():
        N_padded = N
        num_banks = 8
    else:
        N_padded = pad_to_dram_banks(N)
        num_banks = 12

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    in1_shape_padded = [1, 1, K, N_padded]
    in1_shard_shape = [K, N_padded // num_banks]
    bias_shape = [1, 1, 1, N]
    bias_shape_padded = [1, 1, 32, N_padded]
    bias_shard_shape = [32, N_padded // num_banks]
    num_cores = grid_size[0] * grid_size[1]

    in0_block_h = M // 32
    in0_block_w = K // num_cores // 32
    out_block_h = M // 32
    out_block_w = N // num_cores // 32

    out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

    logger.debug("N_padded " + str(N_padded))
    logger.debug("in0 block h w " + str(in0_block_h * 32) + " " + str(in0_block_w * 32))
    logger.debug("in1 block h w " + str(in0_block_w * 32) + " " + str(out_block_w * 32))
    logger.debug("out block h w " + str(out_block_h * 32) + " " + str(out_block_w * 32))
    logger.debug("out subblock h w " + str(out_subblock_h * 32) + " " + str(out_subblock_w * 32))

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in1_shard_grid = ttl.tensor.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttl.tensor.ShardSpec(in1_shard_grid, in1_shard_shape, ttl.tensor.ShardOrientation.ROW_MAJOR, False)
    in1_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, ttl.tensor.BufferType.DRAM, in1_shard_spec
    )

    logger.debug("in1_shard_shape " + str(in1_shard_shape))
    logger.debug("in1_shard_grid " + str(in1_shard_grid))

    in0 = torch.randn(in0_shape).bfloat16().float()
    # step = K // num_cores
    # in0 = torch.ones(in0_shape).bfloat16().float()
    # for i in range(num_cores):  # since 32768 / 16 = 2048
    #     in0[:, :, :, i * step : (i + 1) * step] = i + 1
    in1 = torch.randn(in1_shape).bfloat16().float()
    # in1 = torch.ones(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()
    # bias = torch.ones(bias_shape).bfloat16().float() * 10

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=in0_dtype)
    in1_t = ttl.tensor.Tensor(in1.flatten().tolist(), in1_shape, in1_dtype, ttl.tensor.Layout.ROW_MAJOR)
    in1_t = in1_t.pad(in1_shape_padded, (0, 0, 0, 0), 0).to(ttl.tensor.Layout.TILE)
    in1_t = in1_t.to(device, in1_mem_config)

    if has_bias:
        bias_shard_grid = ttl.tensor.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
        bias_shard_grid = ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), bias_shard_grid)})
        bias_shard_spec = ttl.tensor.ShardSpec(
            bias_shard_grid, bias_shard_shape, ttl.tensor.ShardOrientation.ROW_MAJOR, False
        )
        bias_mem_config = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, ttl.tensor.BufferType.DRAM, bias_shard_spec
        )

        bias_t = ttl.tensor.Tensor(
            bias.flatten().tolist(), bias_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR
        )
        bias_t = bias_t.pad(bias_shape_padded, (0, 0, 0, 0), 0).to(ttl.tensor.Layout.TILE)
        bias_t = bias_t.to(device, bias_mem_config)

    in0_t = ttl.tensor.interleaved_to_sharded(
        in0_t,
        grid_size,
        [M, int(in0_block_w * 32)],
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w // 4,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        skip_compute=False,
        skip_in0_mcast=False,
        skip_write_back=False,
    )

    if is_grayskull():
        compute_kernel_config = ttl.tensor.GrayskullComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=True,
        )
    else:
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    if has_bias:
        output_t = ttl.operations.primary.matmul_dram_sharded(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            output_mem_config=sharded_mem_config,
            output_dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        output_t = ttl.operations.primary.matmul_dram_sharded(
            in0_t,
            in1_t,
            program_config=program_config,
            output_mem_config=sharded_mem_config,
            output_dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )
    output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)

    pt_out = in0 @ in1
    if has_bias:
        pt_out += bias

    tt_out = tt2torch_tensor(output_t)

    print(pt_out)
    print(tt_out)

    pt_out_unpad = pt_out[:, :, :, 0:N]
    tt_out_unpad = tt_out[:, :, :, 0:N]
    passing, output = comp_pcc(pt_out_unpad, tt_out_unpad)
    logger.info(output)
    assert passing

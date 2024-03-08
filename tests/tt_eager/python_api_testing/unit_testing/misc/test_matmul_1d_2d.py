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


@skip_for_wormhole_b0("WH ND hang, see issue #4392")
@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, dtype, fidelity, packer_l1_acc, fp32_acc_mode",
    [
        # 256 256 256
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            True,
            False,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            True,
            False,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
            True,
            False,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            4096,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
            True,
            False,
        ),
        # 512 512 512 x 8 subblock 4 2
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            True,
            True,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            True,
            True,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
            True,
            True,
        ),
        (
            False,
            True,
            True,
            1792,
            2048,
            2048,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
            True,
            True,
        ),
    ],
)
def test_multi_core_matmul_2d(
    device,
    dtype,
    fidelity,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    M,
    K,
    N,
    activation,
    function_level_defaults,
):
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]
    grid_size = (8, 7)

    in0_block_h = M // grid_size[1] // 32
    in0_block_w = K // grid_size[0] // 32
    out_block_h = M // grid_size[1] // 32
    out_block_w = N // grid_size[0] // 32

    if fp32_acc_mode == True:
        out_subblock_w = 4
        out_subblock_h = 1
    else:
        out_subblock_w = 8
        out_subblock_h = 1

    logger.debug("in0 block h w " + str(in0_block_h * 32) + " " + str(in0_block_w * 32))
    logger.debug("in1 block h w " + str(in0_block_w * 32) + " " + str(out_block_w * 32))
    logger.debug("out block h w " + str(out_block_h * 32) + " " + str(out_block_w * 32))
    logger.debug("out subblock h w " + str(out_subblock_h * 32) + " " + str(out_subblock_w * 32))

    interleaved_mem_config_L1 = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    interleaved_mem_config_DRAM = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=dtype)

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config_L1

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[1], K // grid_size[0]],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=False,
        fused_activation=activation,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=fp32_acc_mode,
        packer_l1_acc=packer_l1_acc,
    )

    output_t = ttl.operations.primary.matmul(
        in0_t,
        in1_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        compute_kernel_config=compute_kernel_config,
    )

    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config_L1)

    pt_out = in0 @ in1

    if has_bias:
        pt_out = pt_out + bias

    if activation != None:
        pt_out = torch.nn.functional.gelu(pt_out)
    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@skip_for_wormhole_b0("WH ND hang, see issue #4392")
@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, dtype, fidelity, packer_l1_acc, fp32_acc_mode",
    [
        # 512, 8192, 8192
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            False,
        ),
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            True,
            False,
        ),
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            True,
            False,
        ),
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
            True,
            False,
        ),
        (
            False,
            True,
            True,
            512,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
            True,
            False,
        ),
        # 256, 8192, 8192
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
            False,
            True,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.LoFi,
            True,
            True,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT8_B,
            ttl.tensor.MathFidelity.HiFi2,
            True,
            True,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.LoFi,
            True,
            True,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.MathFidelity.HiFi2,
            True,
            True,
        ),
    ],
)
def test_multi_core_matmul_1d(
    device,
    dtype,
    fidelity,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    M,
    K,
    N,
    activation,
    function_level_defaults,
):
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]
    grid_size = (8, 4)
    num_cores = grid_size[0] * grid_size[1]

    in0_block_h = M // 32
    in0_block_w = K // num_cores // 32
    out_block_h = M // 32
    out_block_w = N // num_cores // 32

    if fp32_acc_mode == True:
        out_subblock_w = 4
        out_subblock_h = 1
    else:
        out_subblock_w = 8
        out_subblock_h = 1

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

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)

    output_mem_config = sharded_mem_config

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M, int(out_block_w * 32)],
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=fp32_acc_mode,
        packer_l1_acc=packer_l1_acc,
    )

    output_t = ttl.operations.primary.matmul_1d(
        in0_t,
        in1_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
        output_dtype=dtype,
        compute_kernel_config=compute_kernel_config,
    )
    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing

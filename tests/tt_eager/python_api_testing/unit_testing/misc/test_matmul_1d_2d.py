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


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.skipif(is_grayskull(), reason="no llama2 test on GS")
@pytest.mark.parametrize(
    "packer_l1_acc",
    [
        True,
    ],
    ids=["pack_l1"],
)
@pytest.mark.parametrize(
    "fp32_acc_mode",
    [
        False,
    ],
    ids=["no_fp32"],
)
@pytest.mark.parametrize(
    "fidelity",
    [
        ttnn.MathFidelity.LoFi,
    ],
    ids=["LoFi"],
)
@pytest.mark.parametrize(
    "has_bias",
    [
        False,
    ],
    ids=["no_bias"],
)
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, grid_size",
    [
        (False, True, True, 32, 8192, 1280, None, (8, 1)),
        (False, True, True, 32, 8192, 4096, None, (8, 4)),
        (False, True, True, 32, 8192, 1024, None, (8, 4)),
        (False, True, True, 32, 32768, 1024, None, (8, 4)),
    ],
)
def test_llama2_matmul(
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
    packer_l1_acc,
    fp32_acc_mode,
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

    out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

    logger.debug("in0 block h w " + str(in0_block_h * 32) + " " + str(in0_block_w * 32))
    logger.debug("in1 block h w " + str(in0_block_w * 32) + " " + str(out_block_w * 32))
    logger.debug("out block h w " + str(out_block_h * 32) + " " + str(out_block_w * 32))
    logger.debug("out subblock h w " + str(out_subblock_h * 32) + " " + str(out_subblock_w * 32))

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    output_mem_config = sharded_mem_config

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=ttnn.bfloat16)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=ttnn.bfloat8_b)

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M, int(in0_block_w * 32)],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=compute_kernel_config,
    )
    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.LoFi,
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
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
            ttnn.bfloat16,
            ttnn.MathFidelity.LoFi,
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
            ttnn.bfloat16,
            ttnn.MathFidelity.HiFi2,
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.LoFi,
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
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
            ttnn.bfloat16,
            ttnn.MathFidelity.LoFi,
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
            ttnn.bfloat16,
            ttnn.MathFidelity.HiFi2,
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.LoFi,
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
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
            ttnn.bfloat16,
            ttnn.MathFidelity.LoFi,
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
            ttnn.bfloat16,
            ttnn.MathFidelity.HiFi2,
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.LoFi,
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
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
            ttnn.bfloat16,
            ttnn.MathFidelity.LoFi,
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
            ttnn.bfloat16,
            ttnn.MathFidelity.HiFi2,
            True,
            True,
        ),
    ],
)
def test_multi_core_matmul_2d_wh(
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

    interleaved_mem_config_L1 = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    interleaved_mem_config_DRAM = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=dtype)

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config_L1

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[1], K // grid_size[0]],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=False,
        fused_activation=activation,
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
        compute_kernel_config=compute_kernel_config,
    )

    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config_L1)

    pt_out = in0 @ in1

    if has_bias:
        pt_out = pt_out + bias

    if activation != None:
        pt_out = torch.nn.functional.gelu(pt_out)
    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.LoFi,
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
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
            ttnn.bfloat16,
            ttnn.MathFidelity.LoFi,
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
            ttnn.bfloat16,
            ttnn.MathFidelity.HiFi2,
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.LoFi,
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
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
            ttnn.bfloat16,
            ttnn.MathFidelity.LoFi,
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
            ttnn.bfloat16,
            ttnn.MathFidelity.HiFi2,
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.LoFi,
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
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
            ttnn.bfloat16,
            ttnn.MathFidelity.LoFi,
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
            ttnn.bfloat16,
            ttnn.MathFidelity.HiFi2,
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.LoFi,
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
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
            True,
            True,
        ),
        # (
        #     False,
        #     True,
        #     True,
        #     256,
        #     8192,
        #     8192,
        #     None,
        #     ttnn.bfloat16,
        #     ttnn.MathFidelity.LoFi,
        #     True,
        #     True,
        # ),
        # (
        #     False,
        #     True,
        #     True,
        #     256,
        #     8192,
        #     8192,
        #     None,
        #     ttnn.bfloat16,
        #     ttnn.MathFidelity.HiFi2,
        #     True,
        #     True,
        # ),
    ],
)
def test_multi_core_matmul_1d_wh(
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

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)

    output_mem_config = sharded_mem_config

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M, int(out_block_w * 32)],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, dtype, fidelity",
    [
        # 256 256 256
        (
            False,
            True,
            True,
            3072,
            2048,
            4096,
            None,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.LoFi,
        ),
        (
            False,
            True,
            True,
            3072,
            2048,
            4096,
            None,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
        ),
        (
            False,
            True,
            True,
            3072,
            2048,
            4096,
            None,
            ttnn.bfloat16,
            ttnn.MathFidelity.LoFi,
        ),
        (
            False,
            True,
            True,
            3072,
            2048,
            4096,
            None,
            ttnn.bfloat16,
            ttnn.MathFidelity.HiFi2,
        ),
        # 512 512 512 x 8 subblock 4 2
        (
            False,
            True,
            True,
            3072,
            2048,
            2048,
            None,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.LoFi,
        ),
        (
            False,
            True,
            True,
            3072,
            2048,
            2048,
            None,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
        ),
        (
            False,
            True,
            True,
            3072,
            2048,
            2048,
            None,
            ttnn.bfloat16,
            ttnn.MathFidelity.LoFi,
        ),
        (
            False,
            True,
            True,
            3072,
            2048,
            2048,
            None,
            ttnn.bfloat16,
            ttnn.MathFidelity.HiFi2,
        ),
    ],
)
def test_multi_core_matmul_2d_gs(
    device,
    dtype,
    fidelity,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    has_bias,
    M,
    K,
    N,
    activation,
    function_level_defaults,
):
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]
    grid_size = (12, 8)

    in0_block_w = K // grid_size[1] // 32  # 16
    in0_block_h = M // grid_size[0] // 32
    out_block_h = M // grid_size[0] // 32
    out_block_w = N // grid_size[1] // 32

    if out_block_w <= 8:
        out_subblock_w = out_block_w
        out_subblock_h = 8 // out_subblock_w
    else:
        out_subblock_h = 1
        out_subblock_w = 8 // out_subblock_h
        while out_block_w % out_subblock_w != 0:
            out_subblock_w = out_block_w // 2

    logger.debug("in0 block w h " + str(in0_block_w * 32) + " " + str(in0_block_h * 32))
    logger.debug("in1 block w h " + str(out_block_w * 32) + " " + str(in0_block_w * 32))
    logger.debug("out block w h " + str(out_block_w * 32) + " " + str(out_block_h * 32))
    logger.debug("out subblock w h " + str(out_subblock_w * 32) + " " + str(out_subblock_h * 32))

    interleaved_mem_config_L1 = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    interleaved_mem_config_DRAM = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttnn.bfloat8_b)

    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttnn.bfloat8_b)

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config_L1
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config_L1, tt_dtype=ttnn.bfloat8_b)[0]

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[0], K // grid_size[1]],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=True,
        fused_activation=activation,
    )

    compute_kernel_config = ttnn.GrayskullComputeKernelConfig(math_fidelity=fidelity, math_approx_mode=True)

    if has_bias:
        output_t = ttnn.linear(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            memory_config=output_mem_config,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=output_mem_config,
            compute_kernel_config=compute_kernel_config,
        )

    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config_L1)

    pt_out = in0 @ in1

    if has_bias:
        pt_out = pt_out + bias

    if activation != None:
        pt_out = torch.nn.functional.gelu(pt_out)
    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, dtype, fidelity",
    [
        # 256, 8192, 8192
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.LoFi,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttnn.bfloat16,
            ttnn.MathFidelity.LoFi,
        ),
        (
            False,
            True,
            True,
            256,
            8192,
            8192,
            None,
            ttnn.bfloat16,
            ttnn.MathFidelity.HiFi2,
        ),
    ],
)
def test_multi_core_matmul_1d_gs(
    device,
    dtype,
    fidelity,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    has_bias,
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

    out_subblock_w = 8
    out_subblock_h = 1

    logger.debug("in0 block h w " + str(in0_block_h * 32) + " " + str(in0_block_w * 32))
    logger.debug("in1 block h w " + str(in0_block_w * 32) + " " + str(out_block_w * 32))
    logger.debug("out block h w " + str(out_block_h * 32) + " " + str(out_block_w * 32))
    logger.debug("out subblock h w " + str(out_subblock_h * 32) + " " + str(out_subblock_w * 32))

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)

    output_mem_config = sharded_mem_config

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M, int(out_block_w * 32)],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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

    compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
    )

    output_t = ttnn.matmul(
        in0_t,
        in1_t,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=dtype,
        compute_kernel_config=compute_kernel_config,
    )
    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing

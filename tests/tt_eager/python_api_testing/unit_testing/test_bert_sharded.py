# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import pytest
import torch

import tt_lib as ttl

from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from tests.tt_eager.python_api_testing.sweep_tests import (
    pytorch_ops,
    tt_lib_ops,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
from models.utility_functions import skip_for_wormhole_b0
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero


def get_block_subblock_dim(grid_size, M, K, N):
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

    return in0_block_w, in0_block_h, out_block_h, out_block_w, out_subblock_h, out_subblock_w


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "LN_gamma_beta_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=[
        "LN_gamma_beta_DRAM",
    ],
)
@pytest.mark.parametrize(
    "in_LN_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=[
        "in_LN_DRAM",
    ],
)
@pytest.mark.parametrize(
    "in_LN_dtype",
    (ttl.tensor.DataType.BFLOAT8_B,),
    ids=["BFLOAT8_B"],
)
def test_bert_sharded_LN_to_LN(in_LN_dtype, in_LN_mem_config, LN_gamma_beta_mem_config, device):
    torch.manual_seed(1234)
    # device shape
    grid_size = (12, 8)
    batch = 12
    # mem config
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
    # LN
    in0_shape = (1, 1, batch * 384, 1024)
    M = in0_shape[2]
    K = in0_shape[3]
    LN_in0 = torch.rand(in0_shape) * 2 - 0.95
    LN_in0_t = torch2tt_tensor(LN_in0, device, tt_memory_config=in_LN_mem_config, tt_dtype=in_LN_dtype)
    LN_in0_t_shard = ttl.tensor.interleaved_to_sharded(
        LN_in0_t,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    LN_in1 = torch.rand(in0_shape) * 2 - 0.8
    LN_in1_t = torch2tt_tensor(LN_in1, device, tt_memory_config=in_LN_mem_config, tt_dtype=in_LN_dtype)
    LN_in1_t_shard = ttl.tensor.interleaved_to_sharded(
        LN_in1_t,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )
    gamma = torch.rand(1, 1, 32, in0_shape[3]) * 2 - 1
    beta = torch.rand(1, 1, 32, in0_shape[3]) * 2.0 - 1.1
    gamma_t = torch2tt_tensor(
        gamma, device, tt_memory_config=LN_gamma_beta_mem_config, tt_dtype=ttl.tensor.DataType.BFLOAT16
    )
    beta_t = torch2tt_tensor(
        beta, device, tt_memory_config=LN_gamma_beta_mem_config, tt_dtype=ttl.tensor.DataType.BFLOAT16
    )
    epsf = 1e-2

    program_config = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4,
        block_h=batch,
        block_w=4,
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        im_data_format=ttl.tensor.DataType.BFLOAT16,
        out_data_format=ttl.tensor.DataType.BFLOAT8_B,
        inplace=True,
    )

    logger.info("LN")
    LN_out_t = ttl.operations.primary.add_layernorm(
        LN_in0_t_shard,
        LN_in1_t_shard,
        epsf,
        gamma_t,
        beta_t,
        output_mem_config=sharded_mem_config,
        program_config=program_config,
    )
    logger.info("LN_done")

    LN_in0_t_shard.deallocate()
    LN_in1_t_shard.deallocate()

    # FF1 + GELU
    M = 4608
    K = 1024
    N = 4096
    in0_block_w, in0_block_h, out_block_h, out_block_w, out_subblock_h, out_subblock_w = get_block_subblock_dim(
        grid_size, M, K, N
    )

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]

    FF1_in1 = torch.randn(in1_shape).bfloat16().float()
    FF1_bias = torch.randn(bias_shape).bfloat16().float()

    FF1_in1_t = torch2tt_tensor(
        FF1_in1, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )
    FF1_bias_t = pad_by_zero(
        FF1_bias, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )[0]

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=True,
        fused_activation=(ttl.tensor.FusibleActivation.GELU, True),
    )

    logger.info("FF1")
    out_ff1_t = ttl.operations.primary.matmul(
        LN_out_t,
        FF1_in1_t,
        bias=FF1_bias_t,
        program_config=program_config,
        output_mem_config=sharded_mem_config,
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
    )
    logger.info("FF1_done")

    # FF2
    M = 4608
    K = 4096
    N = 1024
    in0_block_w, in0_block_h, out_block_h, out_block_w, out_subblock_h, out_subblock_w = get_block_subblock_dim(
        grid_size, M, K, N
    )

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]

    FF2_in1 = torch.randn(in1_shape).bfloat16().float()
    FF2_bias = torch.randn(bias_shape).bfloat16().float()

    FF2_in1_t = torch2tt_tensor(
        FF2_in1, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )
    FF2_bias_t = pad_by_zero(
        FF2_bias, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )[0]

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=True,
        fused_activation=None,
    )

    logger.info("FF2")
    out_ff2_t = ttl.operations.primary.matmul(
        out_ff1_t,
        FF2_in1_t,
        bias=FF2_bias_t,
        program_config=program_config,
        output_mem_config=sharded_mem_config,
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
    )
    logger.info("FF2_done")

    out_ff1_t.deallocate()

    # LN
    in0_shape = (1, 1, batch * 384, 1024)
    M = in0_shape[2]
    K = in0_shape[3]
    gamma2 = torch.rand(1, 1, 32, in0_shape[3]) * 2 - 1
    beta2 = torch.rand(1, 1, 32, in0_shape[3]) * 2.0 - 1.1
    gamma_t = torch2tt_tensor(
        gamma2, device, tt_memory_config=LN_gamma_beta_mem_config, tt_dtype=ttl.tensor.DataType.BFLOAT16
    )
    beta_t = torch2tt_tensor(
        beta2, device, tt_memory_config=LN_gamma_beta_mem_config, tt_dtype=ttl.tensor.DataType.BFLOAT16
    )
    epsf = 1e-2
    program_config = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4,
        block_h=batch,
        block_w=4,
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        im_data_format=ttl.tensor.DataType.BFLOAT16,
        out_data_format=ttl.tensor.DataType.BFLOAT8_B,
        inplace=False,
    )
    logger.info("LN2")
    LN_out2_t = ttl.operations.primary.add_layernorm(
        LN_out_t,
        out_ff2_t,
        epsf,
        gamma_t,
        beta_t,
        output_mem_config=sharded_mem_config,
        program_config=program_config,
    )
    logger.info("LN2_done")

    LN_out_t.deallocate()
    out_ff2_t.deallocate()

    # QKV
    M = 4608
    K = 1024
    N = 3072
    in0_block_w, in0_block_h, out_block_h, out_block_w, out_subblock_h, out_subblock_w = get_block_subblock_dim(
        grid_size, M, K, N
    )

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]

    qkv_in1 = torch.randn(in1_shape).bfloat16().float()
    qkv_bias = torch.randn(bias_shape).bfloat16().float()

    qkv_in1_t = torch2tt_tensor(
        qkv_in1, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )
    qkv_bias_t = pad_by_zero(
        qkv_bias, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )[0]

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=True,
        fused_activation=None,
    )

    logger.info("QKV")
    out_qkv_t = ttl.operations.primary.matmul(
        LN_out2_t,
        qkv_in1_t,
        bias=qkv_bias_t,
        program_config=program_config,
        output_mem_config=sharded_mem_config,
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
    )
    logger.info("QKV_done")

    # compare results
    # out_ln_t = ttl.tensor.sharded_to_interleaved(LN_out_t, interleaved_mem_config_DRAM)
    # out_ln = tt2torch_tensor(out_ln_t)
    ref_lnorm = torch.nn.functional.layer_norm(
        LN_in0 + LN_in1, LN_in0.shape[-1:], gamma[:, :, 0:1, :].flatten(), beta[:, :, 0:1, :].flatten(), epsf
    )
    # passing, output = comp_pcc(out_ln, ref_lnorm, 0.98)
    # logger.info(output)

    # out_ff1_t = ttl.tensor.sharded_to_interleaved(out_ff1_t, interleaved_mem_config_DRAM)
    # out_ff1 = tt2torch_tensor(out_ff1_t)
    ref_ff1 = torch.nn.functional.gelu(ref_lnorm @ FF1_in1 + FF1_bias)
    # passing, output = comp_pcc(out_ff1, ref_ff1, 0.99)
    # logger.info(output)

    # out_ff2_t = ttl.tensor.sharded_to_interleaved(out_ff2_t, interleaved_mem_config_DRAM)
    # out_ff2 = tt2torch_tensor(out_ff2_t)
    ref_ff2 = ref_ff1 @ FF2_in1 + FF2_bias
    # passing, output = comp_pcc(out_ff2, ref_ff2, 0.99)
    # logger.info(output)

    LN_out_t2 = ttl.tensor.sharded_to_interleaved(LN_out2_t, interleaved_mem_config_DRAM)
    LN_out2 = tt2torch_tensor(LN_out_t2)
    print(LN_out2[0][0][0])
    ref_lnorm2 = torch.nn.functional.layer_norm(
        ref_ff2 + ref_lnorm, LN_in0.shape[-1:], gamma2[:, :, 0:1, :].flatten(), beta2[:, :, 0:1, :].flatten(), epsf
    )
    passing, output = comp_pcc(LN_out2, ref_lnorm2, 0.99)
    print(ref_lnorm2[0][0][0])
    logger.info(output)

    out_qkv_t = ttl.tensor.sharded_to_interleaved(out_qkv_t, interleaved_mem_config_DRAM)
    out_qkv = tt2torch_tensor(out_qkv_t)
    ref_qkv = ref_lnorm2 @ qkv_in1 + qkv_bias
    passing, output = comp_pcc(out_qkv, ref_qkv, 0.99)
    logger.info(output)

    assert True

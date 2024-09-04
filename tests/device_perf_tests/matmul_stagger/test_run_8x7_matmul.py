# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest

from loguru import logger
import torch

from models.utility_functions import run_for_wormhole_b0
import ttnn
from tests.device_perf_tests.matmul_stagger.test_matmul_stagger import MATMUL_VARIANTS
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "program_config_variant, seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, mcast_in0",
    (
        ("MatmulMultiCoreReuseMultiCast1DProgramConfig", 1 * 32, 1 * 32, 56 * 32, 1, 1, 1, 1, 1, True),
        ("MatmulMultiCoreReuseMultiCast1DProgramConfig", 56 * 32, 1 * 32, 1 * 32, 1, 1, 1, 1, 1, False),
        ("MatmulMultiCoreReuseMultiCastProgramConfig", 7 * 32, 1 * 32, 8 * 32, 1, 1, 1, 1, 1, False),
        ("MatmulMultiCoreReuseProgramConfig", 56 * 32, 1 * 32, 56 * 32, 1, 56, 1, 1, 1, False),
    ),
    ids=MATMUL_VARIANTS,
)
def test_run_8x7_matmul(
    device,
    program_config_variant,
    seq_len,
    inner_dim,
    weights_n,
    per_core_M,
    per_core_N,
    in_block_w,
    out_subblock_h,
    out_subblock_w,
    mcast_in0,
):
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    dtype = ttnn.DataType.BFLOAT8_B
    grid_size = (8, 7)

    a_shape = [1, 1, seq_len, inner_dim]
    b_shape = [1, 1, inner_dim, weights_n]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape)

    input0 = ttnn.as_tensor(
        A,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
    )
    input0 = ttnn.to_device(input0, device)

    input1 = ttnn.as_tensor(
        B,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
    )
    input1 = ttnn.to_device(input1, device)

    if program_config_variant == "MatmulMultiCoreReuseMultiCastProgramConfig":
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=None,
        )
    elif program_config_variant == "MatmulMultiCoreReuseMultiCast1DProgramConfig":
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=mcast_in0,
        )
    else:
        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
        )

    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    out = ttnn.matmul(
        input0,
        input1,
        program_config=program_config,
        memory_config=mem_config,
        dtype=dtype,
        compute_kernel_config=compute_config,
    )

    pt_out = A @ B
    out = ttnn.to_torch(out)

    passing, output = comp_pcc(pt_out, out)
    logger.info(output)
    assert passing

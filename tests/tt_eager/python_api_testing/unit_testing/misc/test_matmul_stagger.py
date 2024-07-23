# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from models.utility_functions import is_grayskull
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)


@pytest.mark.skipif(is_grayskull(), reason="Stagger only supported on wormhole")
@pytest.mark.parametrize(
    "program_config_variant, seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, mcast_in0",
    (
        ("MatmulMultiCoreReuseMultiCast1DProgramConfig", 32 * 32, 144 * 32, 56 * 32, 32, 1, 1, 1, 1, True),
        ("MatmulMultiCoreReuseMultiCast1DProgramConfig", 56 * 32, 144 * 32, 32 * 32, 1, 32, 1, 1, 1, False),
        ("MatmulMultiCoreReuseMultiCastProgramConfig", 35 * 32, 144 * 32, 56 * 32, 5, 8, 1, 1, 1, False),
        ("MatmulMultiCoreReuseProgramConfig", 56 * 32, 144 * 32, 56 * 32, 1, 56, 1, 1, 1, False),
    ),
    ids=["matmul1d", "matmul1d_transposed", "matmul2d", "matmul_no_mcast"],
)
@pytest.mark.parametrize("disable_stagger", [True, False])
def test_matmul_stagger(
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
    disable_stagger,
):
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    dtype = ttnn.DataType.BFLOAT8_B
    grid_size = (8, 7)

    a_shape = [1, 1, seq_len, inner_dim]
    b_shape = [1, 1, inner_dim, weights_n]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape)

    a_t = torch2tt_tensor(A, device, ttnn.Layout.TILE, mem_config, dtype)
    b_t = torch2tt_tensor(B, device, ttnn.Layout.TILE, mem_config, dtype)

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
            disable_stagger=disable_stagger,
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
            disable_stagger=disable_stagger,
        )
    else:
        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            disable_stagger=disable_stagger,
        )

    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    out = ttnn.matmul(
        a_t,
        b_t,
        program_config=program_config,
        memory_config=mem_config,
        dtype=dtype,
        compute_kernel_config=compute_config,
    )

    pt_out = A @ B
    out = tt2torch_tensor(out)

    passing, output = comp_pcc(pt_out, out)
    logger.info(output)
    assert passing

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest

import tt_lib as ttl
import ttnn
from models.utility_functions import comp_pcc, tt2torch_tensor, torch2tt_tensor
import torch


# Used to reproduce issue #7066 with matmul 1D (Falcon 7b matmuls)
@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    (
        (32, 8192, 8128, 1, 4, 4, 1, 4, 1000),
        (128, 8192, 8128, 4, 4, 4, 1, 4, 2000),
        (32, 8192, 4096, 1, 2, 4, 1, 2, 1000),
        (128, 8192, 4096, 4, 2, 4, 2, 2, 5000),
    ),
    ids=["lm_seq_len32", "lm_seq_len128", "mlp_4h_seq_len32", "mlp_4h_seq_len128"],
)
def test_reproduce_matmul_1d(
    device,
    seq_len,
    inner_dim,
    weights_n,
    per_core_M,
    per_core_N,
    in_block_w,
    out_subblock_h,
    out_subblock_w,
    loop_count,
):
    torch.manual_seed(1234)

    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    in1_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    in0_dtype = ttl.tensor.DataType.BFLOAT8_B
    in1_dtype = ttl.tensor.DataType.BFLOAT8_B
    out_dtype = ttl.tensor.DataType.BFLOAT8_B

    a_shape = [1, 1, seq_len, inner_dim]
    b_shape = [1, 1, inner_dim, weights_n]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape)

    a_t = torch2tt_tensor(A, device, ttl.tensor.Layout.TILE, in0_mem_config, in0_dtype)
    b_t = torch2tt_tensor(B, device, ttl.tensor.Layout.TILE, in1_mem_config, in1_dtype)

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    compute_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,  # fails with l1 acc turned on as well, just needs more iterations
    )

    # First run for a reference output
    out = ttl.operations.primary.matmul_1d(
        a_t,
        b_t,
        program_config=program_config,
        output_mem_config=out_mem_config,
        output_dtype=out_dtype,
        compute_kernel_config=compute_config,
    )
    ref_out = tt2torch_tensor(out)

    nd_output_count = 0

    # loop_count iterations to test determinism/hang
    for _ in range(loop_count):
        out.deallocate(True)

        out = ttl.operations.primary.matmul_1d(
            a_t,
            b_t,
            program_config=program_config,
            output_mem_config=out_mem_config,
            output_dtype=out_dtype,
            compute_kernel_config=compute_config,
        )

        pt_out = tt2torch_tensor(out)

        _, output_pcc = comp_pcc(ref_out, pt_out, 1)
        nd_output_count += 1 if output_pcc != 1 else 0

        logger.debug(f"Output pcc={output_pcc}")

    print(f"Iterations with nd output: {nd_output_count}")

    assert nd_output_count != 0


# Used to reproduce issue #7066 with matmul 2D (Falcon 40b matmuls)
@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    (
        (2048, 8192, 4096, 8, 16, 8, 1, 4, 100),
        (2048, 32768, 1024, 8, 4, 16, 1, 4, 100),
        (2048, 8192, 1024, 8, 4, 16, 1, 4, 100),
    ),
    ids=[
        "mlp_4h_seq_len2048",
        "mlp_h_seq_len2048",
        "attn_seq_len2048",
    ],
)
def test_reproduce_matmul_2d(
    device,
    seq_len,
    inner_dim,
    weights_n,
    per_core_M,
    per_core_N,
    in_block_w,
    out_subblock_h,
    out_subblock_w,
    loop_count,
):
    torch.manual_seed(1234)

    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    in1_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    in0_dtype = ttl.tensor.DataType.BFLOAT8_B
    in1_dtype = ttl.tensor.DataType.BFLOAT8_B
    out_dtype = ttl.tensor.DataType.BFLOAT8_B

    a_shape = [1, 1, seq_len, inner_dim]
    b_shape = [1, 1, inner_dim, weights_n]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape)

    a_t = torch2tt_tensor(A, device, ttl.tensor.Layout.TILE, in0_mem_config, in0_dtype)
    b_t = torch2tt_tensor(B, device, ttl.tensor.Layout.TILE, in1_mem_config, in1_dtype)

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )

    compute_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # First run for a reference output
    out = ttl.operations.primary.matmul(
        a_t,
        b_t,
        program_config=program_config,
        output_mem_config=out_mem_config,
        output_dtype=out_dtype,
        compute_kernel_config=compute_config,
    )
    ref_out = tt2torch_tensor(out)

    nd_output_count = 0

    # loop_count iterations to test determinism/hang
    for _ in range(loop_count):
        out.deallocate(True)

        out = ttl.operations.primary.matmul(
            a_t,
            b_t,
            program_config=program_config,
            output_mem_config=out_mem_config,
            output_dtype=out_dtype,
            compute_kernel_config=compute_config,
        )

        pt_out = tt2torch_tensor(out)

        _, output_pcc = comp_pcc(ref_out, pt_out, 1)
        nd_output_count += 1 if output_pcc != 1 else 0

        logger.debug(f"Output pcc={output_pcc}")

    print(f"Iterations with nd output: {nd_output_count}")

    assert nd_output_count != 0

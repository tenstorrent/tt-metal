# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import get_devices_for_t3000, skip_for_grayskull, torch2tt_tensor, tt2torch_tensor
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def run_test_rotary_matmul1(
    devices,
    batch,
    seq_len,
):
    # Prepare input
    head_dim = 128
    n_head = 8
    query_torch = torch.rand(seq_len, n_head, batch, head_dim)
    key_torch = torch.rand(seq_len, 1, batch, head_dim)
    rotary_mat = torch.rand(1, 1, head_dim, head_dim)

    # memory configs
    shard_spec_8_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 0),
            ),
        }
    )

    ROT_MAT_Q_MM_OUTPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_8_cores_grid,
            [
                32,
                head_dim,  # head dim
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    L1_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    ROT_MAT_Q_MM_PROGCFG = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 1),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=4,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )

    ROT_MAT_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest fidelity
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Prepare tt input
    query_tt = torch2tt_tensor(query_torch, tt_device=None).to(device=devices[0], mem_config=ROT_MAT_Q_MM_OUTPUT_MEMCFG)
    key_tt = torch2tt_tensor(key_torch, devices[0])
    rotary_mat_tt = torch2tt_tensor(rotary_mat, devices[0])

    # tt operation
    query_tt = ttnn.matmul(
        query_tt,
        rotary_mat_tt,
        program_config=ROT_MAT_Q_MM_PROGCFG,
        memory_config=ROT_MAT_Q_MM_OUTPUT_MEMCFG,
        compute_kernel_config=ROT_MAT_COMPUTE_KERNEL_CONFIG,
        # [seqlen, n_heads, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, n_heads, bsz, head_dim]
    )
    key_tt = ttnn.matmul(
        key_tt,
        rotary_mat_tt,
        memory_config=L1_MEMCFG,
        compute_kernel_config=ROT_MAT_COMPUTE_KERNEL_CONFIG,
        # [seqlen, 1, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, 1, bsz, head_dim]
    )

    # torch operation
    query_torch = query_torch @ rotary_mat
    key_torch = key_torch @ rotary_mat

    # compare
    query_tt_cpu = tt2torch_tensor(query_tt)
    key_tt_cpu = tt2torch_tensor(key_tt)
    out_pass_q, output_pcc_q = comp_pcc(query_tt_cpu, query_torch)
    out_pass_k, output_pcc_k = comp_pcc(key_tt_cpu, key_torch)
    logger.info(f"PCC value: {output_pcc_q}")
    logger.info(f"PCC value: {output_pcc_k}")
    assert out_pass_q and out_pass_k


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len",
    ((32, 1),),
)
def test_rotray_matmul1(
    batch,
    seq_len,
    all_devices,
):
    n_devices = 8
    devices = get_devices_for_t3000(all_devices, num_devices=1)
    torch.manual_seed(0)

    run_test_rotary_matmul1(
        devices,
        batch,
        seq_len,
    )


def run_test_rotary_matmul2(
    devices,
    batch,
    seq_len,
):
    # Prepare input
    head_dim = 128
    n_head = 8
    query_torch = torch.rand(seq_len, batch, max(n_head, 32), head_dim)
    key_torch = torch.rand(seq_len, batch, max(1, 32), head_dim)
    rotary_mat = torch.rand(1, 1, head_dim, head_dim)

    # memory configs
    shard_spec_32_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 3),
            ),
        }
    )

    ROT_MAT_MM_OUTPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_32_cores_grid,
            [
                32,
                head_dim,  # head dim
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    L1_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    ROT_MAT_MM_PROGCFG = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=4,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )

    ROT_MAT_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest fidelity
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Prepare tt input
    query_tt = torch2tt_tensor(query_torch, tt_device=None).to(device=devices[0], mem_config=ROT_MAT_MM_OUTPUT_MEMCFG)
    key_tt = torch2tt_tensor(key_torch, tt_device=None).to(device=devices[0], mem_config=ROT_MAT_MM_OUTPUT_MEMCFG)
    rotary_mat_tt = torch2tt_tensor(rotary_mat, devices[0])

    # tt operation
    query_tt = ttnn.matmul(
        query_tt,
        rotary_mat_tt,
        program_config=ROT_MAT_MM_PROGCFG,
        memory_config=ROT_MAT_MM_OUTPUT_MEMCFG,
        compute_kernel_config=ROT_MAT_COMPUTE_KERNEL_CONFIG,
        # [seqlen, n_heads, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, n_heads, bsz, head_dim]
    )
    key_tt = ttnn.matmul(
        key_tt,
        rotary_mat_tt,
        program_config=ROT_MAT_MM_PROGCFG,
        memory_config=ROT_MAT_MM_OUTPUT_MEMCFG,
        compute_kernel_config=ROT_MAT_COMPUTE_KERNEL_CONFIG,
        # [seqlen, 1, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, 1, bsz, head_dim]
    )

    # torch operation
    query_torch = query_torch @ rotary_mat
    key_torch = key_torch @ rotary_mat

    # compare
    query_tt_cpu = tt2torch_tensor(query_tt)
    key_tt_cpu = tt2torch_tensor(key_tt)
    out_pass_q, output_pcc_q = comp_pcc(query_tt_cpu, query_torch)
    out_pass_k, output_pcc_k = comp_pcc(key_tt_cpu, key_torch)
    logger.info(f"PCC value: {output_pcc_q}")
    logger.info(f"PCC value: {output_pcc_k}")
    assert out_pass_q and out_pass_k


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len",
    ((32, 1),),
)
def test_rotray_matmul2(
    batch,
    seq_len,
    all_devices,
):
    n_devices = 8
    devices = get_devices_for_t3000(all_devices, num_devices=1)
    torch.manual_seed(0)

    run_test_rotary_matmul2(
        devices,
        batch,
        seq_len,
    )


def run_test_rotary_matmul3(
    devices,
    batch,
    seq_len,
):
    # Prepare input
    head_dim = 128
    n_head = 8
    query_torch = torch.rand(seq_len, batch, max(n_head, 32), head_dim)
    key_torch = torch.rand(seq_len, batch, max(1, 32), head_dim)
    rotary_mat = torch.rand(1, batch, head_dim, head_dim)

    # memory configs
    shard_spec_32_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 3),
            ),
        }
    )

    ROT_MAT_MM_OUTPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_32_cores_grid,
            [
                32,
                head_dim,  # head dim
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    L1_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    ROT_MAT_MM_PROGCFG = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        in0_block_w=4,  # 128 // TILE_SIZE (dynamic)
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=4,
    )

    ROT_MAT_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest fidelity
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Prepare tt input
    query_tt = torch2tt_tensor(query_torch, tt_device=None).to(device=devices[0], mem_config=ROT_MAT_MM_OUTPUT_MEMCFG)
    key_tt = torch2tt_tensor(key_torch, tt_device=None).to(device=devices[0], mem_config=ROT_MAT_MM_OUTPUT_MEMCFG)
    rotary_mat_tt = torch2tt_tensor(rotary_mat, devices[0])

    # tt operation
    query_tt = ttnn.matmul(
        query_tt,
        rotary_mat_tt,
        program_config=ROT_MAT_MM_PROGCFG,
        memory_config=ROT_MAT_MM_OUTPUT_MEMCFG,
        compute_kernel_config=ROT_MAT_COMPUTE_KERNEL_CONFIG,
        # [seqlen, n_heads, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, n_heads, bsz, head_dim]
    )
    key_tt = ttnn.matmul(
        key_tt,
        rotary_mat_tt,
        program_config=ROT_MAT_MM_PROGCFG,
        memory_config=ROT_MAT_MM_OUTPUT_MEMCFG,
        compute_kernel_config=ROT_MAT_COMPUTE_KERNEL_CONFIG,
        # [seqlen, 1, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, 1, bsz, head_dim]
    )

    # torch operation
    query_torch = torch.bmm(torch.squeeze(query_torch), torch.squeeze(rotary_mat)).unsqueeze(0)
    key_torch = torch.bmm(torch.squeeze(key_torch), torch.squeeze(rotary_mat)).unsqueeze(0)

    # compare
    query_tt_cpu = tt2torch_tensor(query_tt)
    key_tt_cpu = tt2torch_tensor(key_tt)
    out_pass_q, output_pcc_q = comp_pcc(query_tt_cpu, query_torch)
    out_pass_k, output_pcc_k = comp_pcc(key_tt_cpu, key_torch)
    logger.info(f"PCC value: {output_pcc_q}")
    logger.info(f"PCC value: {output_pcc_k}")
    assert out_pass_q and out_pass_k


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len",
    ((32, 1),),
)
def test_rotray_matmul3(
    batch,
    seq_len,
    all_devices,
):
    n_devices = 8
    devices = get_devices_for_t3000(all_devices, num_devices=1)
    torch.manual_seed(0)

    run_test_rotary_matmul3(
        devices,
        batch,
        seq_len,
    )


def run_test_rotary_matmul4(
    devices,
    batch,
    seq_len,
):
    # Prepare input
    head_dim = 128
    n_head = 8
    query_torch = torch.rand(seq_len, batch, max(n_head, 32), head_dim)
    key_torch = torch.rand(seq_len, batch, max(1, 32), head_dim)
    rotary_mat = torch.rand(1, batch, head_dim, head_dim)

    # memory configs
    shard_spec_32_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 3),
            ),
        }
    )

    ROT_MAT_MM_OUTPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_32_cores_grid,
            [
                32,
                head_dim,  # head dim
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    ROT_MAT_MM_IN1_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_32_cores_grid,
            [
                head_dim,
                head_dim,  # head dim
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    ROT_MAT_MM_PROGCFG = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        in0_block_w=4,  # 128 // TILE_SIZE (dynamic)
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=4,
    )

    ROT_MAT_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest fidelity
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Prepare tt input
    query_tt = torch2tt_tensor(query_torch, tt_device=None).to(device=devices[0], mem_config=ROT_MAT_MM_OUTPUT_MEMCFG)
    key_tt = torch2tt_tensor(key_torch, tt_device=None).to(device=devices[0], mem_config=ROT_MAT_MM_OUTPUT_MEMCFG)
    rotary_mat_tt = torch2tt_tensor(rotary_mat, tt_device=None).to(device=devices[0], mem_config=ROT_MAT_MM_IN1_MEMCFG)

    # tt operation
    query_tt = ttnn.matmul(
        query_tt,
        rotary_mat_tt,
        program_config=ROT_MAT_MM_PROGCFG,
        memory_config=ROT_MAT_MM_OUTPUT_MEMCFG,
        compute_kernel_config=ROT_MAT_COMPUTE_KERNEL_CONFIG,
        # [seqlen, n_heads, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, n_heads, bsz, head_dim]
    )
    key_tt = ttnn.matmul(
        key_tt,
        rotary_mat_tt,
        program_config=ROT_MAT_MM_PROGCFG,
        memory_config=ROT_MAT_MM_OUTPUT_MEMCFG,
        compute_kernel_config=ROT_MAT_COMPUTE_KERNEL_CONFIG,
        # [seqlen, 1, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, 1, bsz, head_dim]
    )

    # torch operation
    query_torch = torch.bmm(torch.squeeze(query_torch), torch.squeeze(rotary_mat)).unsqueeze(0)
    key_torch = torch.bmm(torch.squeeze(key_torch), torch.squeeze(rotary_mat)).unsqueeze(0)

    # compare
    query_tt_cpu = tt2torch_tensor(query_tt)
    key_tt_cpu = tt2torch_tensor(key_tt)
    out_pass_q, output_pcc_q = comp_pcc(query_tt_cpu, query_torch)
    out_pass_k, output_pcc_k = comp_pcc(key_tt_cpu, key_torch)
    logger.info(f"PCC value: {output_pcc_q}")
    logger.info(f"PCC value: {output_pcc_k}")
    assert out_pass_q and out_pass_k


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len",
    ((32, 1),),
)
def test_rotray_matmul4(
    batch,
    seq_len,
    all_devices,
):
    n_devices = 8
    devices = get_devices_for_t3000(all_devices, num_devices=1)
    torch.manual_seed(0)

    run_test_rotary_matmul4(
        devices,
        batch,
        seq_len,
    )

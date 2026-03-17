# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Llama 70B Galaxy Attention Output Projection - MM+RS Fusion Test

From profiler baseline:
- MatmulDeviceOperation: ~475 µs (56 cores, HiFi2)
- ReduceScatterMinimalAsync: ~490 µs
- MM Input: 8192x1024, Weight: 1024x2048, Output: 8192x2048
- Total baseline: ~965 µs
"""

import pytest
import ttnn
import torch


# Attn Out dimensions from Llama baseline profiler
ATTN_OUT_DIMS = {
    "M": 8192,  # sequence length
    "K": 1024,  # input dim per device (after concat heads)
    "N": 2048,  # output dim per device
}


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["8x4"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_attn_out_separate_baseline(mesh_device, topology):
    """
    Attn Out separate MM + RS baseline (target: ~965 µs)
    """
    M = ATTN_OUT_DIMS["M"]
    K = ATTN_OUT_DIMS["K"]
    N = ATTN_OUT_DIMS["N"]

    # Input tensor
    input_tensor = ttnn.from_torch(
        torch.randn([1, 1, M, K], dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Weight tensor (BFLOAT4_B like Llama)
    weight_tensor = ttnn.from_torch(
        torch.randn([1, 1, K, N], dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat4_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # HiFi2 compute config (Attn Out uses HiFi2)
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    # Semaphores
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
    ccl_semaphore = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(3)]

    # MM config (7x8 = 56 cores like baseline)
    minimal_config = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=1,
        subblock_w=8,
        compute_with_storage_grid_size=ttnn.CoreCoord(7, 8),
    )

    cluster_axis = 1
    dim = 3
    num_links = 4
    num_workers_per_link = 4

    print(f"\n=== Attn Out Separate Baseline: M={M}, K={K}, N={N} ===")

    # Run MM
    mm_output = ttnn.experimental.minimal_matmul(
        input_tensor,
        weight_tensor,
        config=minimal_config,
        compute_kernel_config=compute_config,
    )

    # Run RS
    rs_output = ttnn.experimental.reduce_scatter_minimal_async(
        mm_output,
        None,  # persistent_output_buffers
        dim,  # dim
        ccl_semaphore,
        barrier_semaphore=None,
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        cluster_axis=cluster_axis,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=None,
    )

    ttnn.synchronize_device(mesh_device)

    print(f"MM output shape: {mm_output.shape}")
    print(f"RS output shape: {rs_output.shape}")

    # Cleanup
    ttnn.deallocate(input_tensor)
    ttnn.deallocate(weight_tensor)
    ttnn.deallocate(mm_output)
    ttnn.deallocate(rs_output)


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["8x4"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@pytest.mark.parametrize(
    "m_block, k_block, n_block, subblock_h, subblock_w, num_buffers",
    [
        # Start with FF1/FF3 best config adapted for Attn Out
        (5, 8, 32, 1, 16, 16),
        (5, 8, 16, 1, 16, 16),  # N=16 (2048/128 = 16 per core with 4 cores)
        (4, 8, 32, 1, 16, 16),
        (5, 4, 32, 1, 16, 16),  # K=4 (1024/256 = 4)
        (8, 8, 8, 1, 8, 16),  # Original style
    ],
    ids=["5_8_32_1x16_buf16", "5_8_16_1x16_buf16", "4_8_32_1x16_buf16", "5_4_32_1x16_buf16", "8_8_8_1x8_buf16"],
)
def test_attn_out_fused_sweep(
    mesh_device,
    topology,
    m_block,
    k_block,
    n_block,
    subblock_h,
    subblock_w,
    num_buffers,
):
    """
    Attn Out fused MM+RS sweep
    """
    M = ATTN_OUT_DIMS["M"]
    K = ATTN_OUT_DIMS["K"]
    N = ATTN_OUT_DIMS["N"]

    # Best params from FF1/FF3
    mm_grid_x, mm_grid_y = 7, 6
    rs_offset_y = 6
    chunk_width_in_mm_blocks = 1
    num_workers_per_link = 2
    num_links = 4

    # Input tensor
    input_tensor = ttnn.from_torch(
        torch.randn([1, 1, M, K], dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Weight tensor (BFLOAT4_B like Llama)
    weight_tensor = ttnn.from_torch(
        torch.randn([1, 1, K, N], dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat4_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # HiFi2 compute config (Attn Out uses HiFi2)
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    # Semaphores
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
    ccl_semaphore = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(3)]
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    minimal_config = ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=k_block,
        N_block_size=n_block,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=ttnn.CoreCoord(mm_grid_x, mm_grid_y),
    )

    rs_core_grid_offset = ttnn.CoreCoord(0, rs_offset_y)
    cluster_axis = 1
    dim = 3

    print(
        f"\n=== Attn Out Fused: M={m_block}, K={k_block}, N={n_block}, sub={subblock_h}x{subblock_w}, buf={num_buffers} ==="
    )

    # Run fused op
    fused_output = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
        input_tensor,
        weight_tensor,
        dim,
        ccl_semaphore,
        rs_core_grid_offset,
        num_links=num_links,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        rs_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        cluster_axis=cluster_axis,
        config=minimal_config,
        compute_kernel_config=compute_config,
        barrier_semaphore=barrier_semaphore,
        num_workers_per_link=num_workers_per_link,
        chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
        num_buffers_per_channel=num_buffers,
    )

    ttnn.synchronize_device(mesh_device)

    print(f"Completed")

    # Cleanup
    ttnn.deallocate(input_tensor)
    ttnn.deallocate(weight_tensor)
    if isinstance(fused_output, list):
        for t in fused_output:
            ttnn.deallocate(t)
    else:
        ttnn.deallocate(fused_output)

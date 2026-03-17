# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Llama 70B Galaxy QKV Projection - MM+RS Fusion Test

From profiler baseline:
- MatmulDeviceOperation: ~570 µs (56 cores, HiFi2)
- ReduceScatterMinimalAsync: ~266 µs
- MM Input: 8192x2048, Weight: 2048x1280, Output: 8192x1280
- Total baseline: ~836 µs
"""

import pytest
import ttnn
import torch


# QKV dimensions from Llama baseline profiler
# MM: 4x2048x2048 → 4x2048x1280 (batched), then reshape to 8192x1280 for RS
# For fused op, we use the flattened shape: 8192x2048 → 8192x1280
QKV_DIMS = {
    "M": 8192,  # sequence length (4 batches x 2048)
    "K": 2048,  # input dim per device
    "N": 1280,  # QKV output dim per device
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
def test_qkv_separate_baseline(mesh_device, topology):
    """
    QKV separate MM + RS baseline (target: ~836 µs)
    """
    M = QKV_DIMS["M"]
    K = QKV_DIMS["K"]
    N = QKV_DIMS["N"]

    # Input tensor
    input_tensor = ttnn.from_torch(
        torch.randn([1, 1, M, K], dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Weight tensor (BFLOAT8_B - QKV uses HiFi2)
    weight_tensor = ttnn.from_torch(
        torch.randn([1, 1, K, N], dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # HiFi2 compute config (QKV uses HiFi2)
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

    # MM config (7x8 = 56 cores like Llama baseline)
    # Llama uses 56 cores, HiFi2
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

    print(f"\n=== QKV Separate Baseline: M={M}, K={K}, N={N} ===")

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
    "m_block, k_block, n_block, subblock_h, subblock_w, num_buffers, chunk_width, mm_grid_x, mm_grid_y",
    [
        # 7x6 grid (42 cores) - max possible
        # Different chunk widths
        (5, 8, 8, 1, 8, 16, 2, 7, 6),  # chunk=2
        (5, 8, 8, 1, 8, 16, 4, 7, 6),  # chunk=4
        (5, 8, 8, 1, 8, 16, 8, 7, 6),  # chunk=8
        # Different buffers
        (5, 8, 8, 1, 8, 4, 1, 7, 6),  # buf=4
        (5, 8, 8, 1, 8, 32, 1, 7, 6),  # buf=32
        # Different block sizes
        (4, 8, 8, 1, 8, 16, 1, 7, 6),  # M=4
        (8, 8, 8, 1, 8, 16, 1, 7, 6),  # M=8
        (5, 4, 8, 1, 8, 16, 1, 7, 6),  # K=4
        # Baseline config
        (5, 8, 8, 1, 8, 16, 1, 7, 6),  # baseline
    ],
    ids=["chunk2", "chunk4", "chunk8", "buf4", "buf32", "M4", "M8", "K4", "baseline"],
)
def test_qkv_fused_sweep(
    mesh_device,
    topology,
    m_block,
    k_block,
    n_block,
    subblock_h,
    subblock_w,
    num_buffers,
    chunk_width,
    mm_grid_x,
    mm_grid_y,
):
    """
    QKV fused MM+RS sweep
    """
    M = QKV_DIMS["M"]
    K = QKV_DIMS["K"]
    N = QKV_DIMS["N"]

    # RS offset = mm_grid_y (RS cores start after MM cores)
    rs_offset_y = mm_grid_y
    chunk_width_in_mm_blocks = chunk_width
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

    # Weight tensor (BFLOAT8_B - QKV uses HiFi2)
    weight_tensor = ttnn.from_torch(
        torch.randn([1, 1, K, N], dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # HiFi2 compute config (QKV uses HiFi2)
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
        f"\n=== QKV Fused: M={m_block}, K={k_block}, N={n_block}, sub={subblock_h}x{subblock_w}, buf={num_buffers} ==="
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

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
EXACT Llama 70B Galaxy baseline test for FF1/FF3 operations.

From profiler baseline (8k_fused_6x8_3links/8k/ops_perf_results):
- MinimalMatmulDeviceOperation: ~754 µs (63 cores, LoFi, BFLOAT4_B weights)
- ReduceScatterMinimalAsync: ~791 µs (40 cores, num_links=4, num_workers_per_link=4)
- MM Input: 8192x2048, Weight: 2048x3584, Output: 8192x3584
- RS Input: 8192x3584, Output: 8192x896 (3584/4 ring_size)
"""

import pytest
import ttnn
import torch


# EXACT dimensions from Llama baseline profiler
LLAMA_DIMS = {
    "M": 8192,  # sequence length
    "K": 2048,  # input dim per device
    "N": 3584,  # hidden dim per device (FF1/FF3)
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
def test_llama_70b_exact_ff1_baseline(mesh_device, topology):
    """
    Replicate EXACT Llama 70B FF1/FF3 baseline:
    - ttnn.experimental.minimal_matmul with MinimalMatmulConfig(7,9) = 63 cores
    - ttnn.experimental.reduce_scatter_minimal_async with num_links=3
    """

    M, K, N = LLAMA_DIMS["M"], LLAMA_DIMS["K"], LLAMA_DIMS["N"]

    # Create input and weight tensors (replicated to all devices)
    # EXACT from Llama: input is bfloat8_b, weights are bfloat8_b
    input_tensor = ttnn.from_torch(
        torch.randn([1, 1, M, K], dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # EXACT from Llama: weights are BFLOAT4_B
    weight_tensor = ttnn.from_torch(
        torch.randn([1, 1, K, N], dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat4_b,  # EXACT from Llama profiler
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print(f"Input shape: {input_tensor.shape}")
    print(f"Weight shape: {weight_tensor.shape}")

    # EXACT Llama compute config from profiler
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,  # EXACT from Llama: math_approx_mode=0
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
        dst_full_sync_en=True,  # EXACT from Llama: dst_full_sync_en=1
    )

    # EXACT Llama MinimalMatmulConfig for seq_len=8192 (from model_config.py line 867-874)
    # 63 cores = 7x9
    minimal_config = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=1,
        subblock_w=8,  # EXACT from model_config.py line 871-872
        compute_with_storage_grid_size=ttnn.CoreCoord(7, 9),  # 63 cores
    )

    # Step 1: MatMul (should get ~754 µs like profiler)
    mm_output = ttnn.experimental.minimal_matmul(
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        config=minimal_config,
        compute_kernel_config=compute_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn.synchronize_device(mesh_device)

    print(f"MM output shape: {mm_output.shape}")
    print("✅ MatMul test completed!")

    # Cleanup
    ttnn.deallocate(input_tensor)
    ttnn.deallocate(weight_tensor)
    ttnn.deallocate(mm_output)


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["8x4"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_llama_70b_exact_rs_baseline(mesh_device, topology):
    """
    Replicate EXACT Llama 70B FF1/FF3 ReduceScatter baseline:
    - Input: 8192x3584 (output of FF1/FF3 MatMul)
    - Output: 8192x896 (3584/4 = 896)
    - cluster_axis=1, num_links=4, num_workers_per_link=4
    - Target: ~791 µs
    """

    M = 8192
    N = 3584  # FF1/FF3 output dim

    # Create input tensor matching MM output shape
    input_tensor = ttnn.from_torch(
        torch.randn([1, 1, M, N], dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print(f"RS Input shape: {input_tensor.shape}")

    # Create semaphores for reduce scatter (need cores and initial_value)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])  # From Llama profiler
    ccl_semaphore = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(3)]

    # EXACT Llama ReduceScatter config from profiler:
    # cluster_axis=1, num_links=4, num_workers_per_link=4, ring_size=4
    rs_output = ttnn.experimental.reduce_scatter_minimal_async(
        input_tensor,
        None,  # persistent_output_buffers
        3,  # dim
        ccl_semaphore,
        barrier_semaphore=None,
        num_links=4,  # EXACT from Llama profiler
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        cluster_axis=1,  # EXACT from Llama
        num_workers_per_link=4,  # EXACT from Llama profiler
        num_buffers_per_channel=None,
    )

    ttnn.synchronize_device(mesh_device)

    print(f"RS Output shape: {rs_output.shape}")
    print("✅ ReduceScatter test completed!")

    # Cleanup
    ttnn.deallocate(input_tensor)
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
def test_llama_70b_fused_mm_rs(mesh_device, topology):
    """
    Test FUSED MatMul + ReduceScatter using same params as Llama baseline.

    Baseline separate ops: MM=751µs + RS=712µs = 1,463µs total
    Goal: Beat 1,463µs with fused op by overlapping compute and communication.

    Using ttnn.experimental.minimal_matmul_strided_reduce_scatter_async
    """

    M, K, N = LLAMA_DIMS["M"], LLAMA_DIMS["K"], LLAMA_DIMS["N"]

    # Create input tensor (same as baseline)
    input_tensor = ttnn.from_torch(
        torch.randn([1, 1, M, K], dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create weight tensor (BFLOAT4_B like Llama)
    weight_tensor = ttnn.from_torch(
        torch.randn([1, 1, K, N], dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat4_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print(f"Input shape: {input_tensor.shape}")
    print(f"Weight shape: {weight_tensor.shape}")

    # EXACT Llama compute config
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    # Create semaphores for fused op
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
    ccl_semaphore = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(3)]
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    # MinimalMatmulConfig - use smaller grid to leave room for RS cores
    # Wormhole has 8x8 compute grid (rows 0-7, cols 0-7)
    # Use 6x6 for MM (36 cores), leave rows 6-7 for RS workers
    minimal_config = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=1,
        subblock_w=8,
        compute_with_storage_grid_size=ttnn.CoreCoord(6, 6),  # 36 cores for MM
    )

    # Fused-specific params
    chunk_width_in_mm_blocks = 2  # How many MM blocks per RS chunk
    num_workers_per_link = 2  # Reduced to fit within grid
    num_links = 4  # Same as Llama RS baseline
    cluster_axis = 1  # Same as Llama
    dim = 3  # Scatter dimension
    # RS cores placed below MM cores (rows 6-7 available)
    rs_core_grid_offset = ttnn.CoreCoord(0, 6)  # Below 6x6 MM grid

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
    )

    ttnn.synchronize_device(mesh_device)

    # fused_output is a list of tensors (one per device in the mesh)
    if isinstance(fused_output, list):
        print(f"Fused output: {len(fused_output)} tensors, first shape: {fused_output[0].shape}")
    else:
        print(f"Fused output shape: {fused_output.shape}")
    print("Fused MM+RS test completed!")

    # Cleanup
    ttnn.deallocate(input_tensor)
    ttnn.deallocate(weight_tensor)
    if isinstance(fused_output, list):
        for t in fused_output:
            ttnn.deallocate(t)
    else:
        ttnn.deallocate(fused_output)


# =============================================================================
# FUSED OP BLOCK SIZE SWEEP (7x6 MM grid, best fusion params: chunk=1, workers=2, links=4)
# =============================================================================
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
    "m_block, k_block, n_block, subblock_h, subblock_w",
    [
        # Original configs
        (8, 8, 8, 1, 8),  # Original baseline
        (4, 8, 16, 1, 8),  # ~1411 µs - GREAT
        (4, 8, 32, 1, 8),  # ~1407 µs - BEST
        # Explore around best
        (4, 8, 24, 1, 8),  # N between 16 and 32
        (3, 8, 32, 1, 8),  # Smaller M
        (6, 8, 32, 1, 8),  # M between 4 and 8
        (4, 4, 32, 1, 8),  # Smaller K
        (4, 16, 32, 1, 8),  # Larger K
        (4, 8, 32, 2, 4),  # Different subblock
        (4, 8, 32, 1, 4),  # Smaller subblock_w
        (4, 8, 48, 1, 8),  # Even larger N (might L1 overflow)
        (4, 4, 16, 1, 8),  # Small K with N=16
        (6, 8, 16, 1, 8),  # M=6 with N=16
        # Final exploration
        (5, 8, 32, 1, 8),  # M=5 (between 4 and 6) - ~1399 µs BEST
        (4, 8, 32, 1, 16),  # Wider subblock - ~1401 µs
        # Fine-tune around M=5
        (5, 8, 32, 1, 16),  # M=5 + wider subblock - ~1392 µs BEST
        (5, 8, 48, 1, 8),  # M=5 + larger N - L1 overflow
        (5, 4, 32, 1, 8),  # M=5 + smaller K
        (5, 8, 32, 2, 4),  # M=5 + different subblock
        (5, 8, 24, 1, 8),  # M=5 + N=24
        (5, 8, 16, 1, 8),  # M=5 + N=16
        # Ultra fine-tune around 5_8_32_1x16
        (5, 8, 32, 1, 32),  # Even wider subblock
        (5, 6, 32, 1, 16),  # K=6
        (5, 10, 32, 1, 16),  # K=10
        (5, 8, 28, 1, 16),  # N=28
        (5, 8, 36, 1, 16),  # N=36
        (4, 8, 32, 1, 32),  # M=4 with wider subblock
    ],
    ids=[
        "8_8_8_1x8",
        "4_8_16_1x8",
        "4_8_32_1x8",
        "4_8_24_1x8",
        "3_8_32_1x8",
        "6_8_32_1x8",
        "4_4_32_1x8",
        "4_16_32_1x8",
        "4_8_32_2x4",
        "4_8_32_1x4",
        "4_8_48_1x8",
        "4_4_16_1x8",
        "6_8_16_1x8",
        "5_8_32_1x8",
        "4_8_32_1x16",
        "5_8_32_1x16",
        "5_8_48_1x8",
        "5_4_32_1x8",
        "5_8_32_2x4",
        "5_8_24_1x8",
        "5_8_16_1x8",
        "5_8_32_1x32",
        "5_6_32_1x16",
        "5_10_32_1x16",
        "5_8_28_1x16",
        "5_8_36_1x16",
        "4_8_32_1x32",
    ],
)
def test_fused_block_sweep(
    mesh_device,
    topology,
    m_block,
    k_block,
    n_block,
    subblock_h,
    subblock_w,
):
    """
    Sweep MM block sizes with fixed 7x6 grid and best fusion params.

    Best so far: ~1,689 µs with M=8, K=8, N=8, subblock 1x8
    Baseline separate ops: ~1,463 µs
    """
    M = LLAMA_DIMS["M"]
    K = LLAMA_DIMS["K"]
    N = LLAMA_DIMS["N"]

    # Fixed best params
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

    # Llama compute config (LoFi)
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    # Semaphores
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
    ccl_semaphore = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(3)]
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    # Variable block sizes
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

    print(f"\n=== Block sizes: M={m_block}, K={k_block}, N={n_block}, subblock={subblock_h}x{subblock_w} ===")

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
    )

    ttnn.synchronize_device(mesh_device)

    print(f"Completed: M={m_block}, K={k_block}, N={n_block}")

    # Cleanup
    ttnn.deallocate(input_tensor)
    ttnn.deallocate(weight_tensor)
    if isinstance(fused_output, list):
        for t in fused_output:
            ttnn.deallocate(t)
    else:
        ttnn.deallocate(fused_output)


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
    "num_buffers",
    [1, 2, 4, 8, 16, 32],
    ids=["buf1", "buf2", "buf4", "buf8", "buf16", "buf32"],
)
def test_fused_buffers_sweep(
    mesh_device,
    topology,
    num_buffers,
):
    """
    Sweep num_buffers_per_channel with best block config (5_8_36_1x16)
    """
    M = LLAMA_DIMS["M"]
    K = LLAMA_DIMS["K"]
    N = LLAMA_DIMS["N"]

    # Best params from previous sweeps
    mm_grid_x, mm_grid_y = 7, 6
    rs_offset_y = 6
    chunk_width_in_mm_blocks = 1
    num_workers_per_link = 2
    num_links = 4
    m_block, k_block, n_block = 5, 8, 36
    subblock_h, subblock_w = 1, 16

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

    # Llama compute config (LoFi)
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
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

    print(f"\n=== num_buffers_per_channel={num_buffers} ===")

    # Run fused op with num_buffers_per_channel
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

    print(f"Completed: num_buffers={num_buffers}")

    # Cleanup
    ttnn.deallocate(input_tensor)
    ttnn.deallocate(weight_tensor)
    if isinstance(fused_output, list):
        for t in fused_output:
            ttnn.deallocate(t)
    else:
        ttnn.deallocate(fused_output)


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
        # Best block configs with buf16
        (5, 8, 36, 1, 16, 16),  # Current best
        (5, 8, 32, 1, 16, 16),  # N=32 with buf16
        (5, 8, 32, 1, 8, 16),  # N=32, subblock 1x8 with buf16
        (4, 8, 32, 1, 16, 16),  # M=4 with buf16
        (5, 8, 36, 1, 16, 32),  # Best block with buf32
        (5, 8, 32, 1, 16, 32),  # N=32 with buf32
    ],
    ids=[
        "5_8_36_1x16_buf16",
        "5_8_32_1x16_buf16",
        "5_8_32_1x8_buf16",
        "4_8_32_1x16_buf16",
        "5_8_36_1x16_buf32",
        "5_8_32_1x16_buf32",
    ],
)
def test_fused_combined_sweep(
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
    Combined sweep: best block configs with best buffer counts
    """
    M = LLAMA_DIMS["M"]
    K = LLAMA_DIMS["K"]
    N = LLAMA_DIMS["N"]

    # Best params from previous sweeps
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

    # Llama compute config (LoFi)
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
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

    print(f"\n=== M={m_block}, K={k_block}, N={n_block}, sub={subblock_h}x{subblock_w}, buf={num_buffers} ===")

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

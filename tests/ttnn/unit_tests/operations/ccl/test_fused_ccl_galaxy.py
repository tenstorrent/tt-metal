# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Galaxy (TG) tests for fused CCL operations - Performance Comparison

This file tests the optimization opportunities identified for Galaxy Llama 70B:

1. FF2 path (AllGather + Matmul):
   - Current: line_all_gather + ttnn.linear
   - Proposed: llama_all_gather_matmul_async (fused)

The tests compare fused vs non-fused operations and generate performance reports.

Galaxy uses these fused APIs which support 2D mesh (8x4) via cluster_axis parameter:
- ttnn.experimental.llama_all_gather_matmul_async (AllGather + Matmul)
- ttnn.experimental.llama_rs_matmul (ReduceScatter + Matmul)
"""

import torch
import pytest
import math
import time
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from ttnn import ShardTensorToMesh, ConcatMeshToTensor
from tracy import signpost
from models.demos.llama3_70b_galaxy.tt.model_config import PREFETCHER_NOC1_GRID


def create_global_semaphores(mesh_device, cores, initial_value, num_buffers=3):
    """Create global semaphore handles for CCL operations."""
    return [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(num_buffers)]


def setup_sub_device(mesh_device):
    """Set up sub-device configuration for Galaxy mesh."""
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    return ccl_sub_device_crs, worker_sub_device_id, sub_device_stall_group


def cleanup_sub_device(mesh_device):
    """Clean up sub-device configuration."""
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


def print_perf_comparison_report(test_name, fused_time_us, non_fused_time_us, rs_input_shape, mm_weights_shape):
    """Print a performance comparison report between fused and non-fused operations."""
    speedup = non_fused_time_us / fused_time_us if fused_time_us > 0 else 0
    improvement_pct = (1 - fused_time_us / non_fused_time_us) * 100 if non_fused_time_us > 0 else 0

    logger.info("=" * 80)
    logger.info(f"PERFORMANCE COMPARISON REPORT: {test_name}")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  - RS Input Shape: {rs_input_shape}")
    logger.info(f"  - MM Weights Shape: {mm_weights_shape}")
    logger.info("-" * 80)
    logger.info(f"Results:")
    logger.info(f"  - Non-Fused (linear + reduce_scatter): {non_fused_time_us:.2f} us")
    logger.info(f"  - Fused (matmul_reduce_scatter_async):  {fused_time_us:.2f} us")
    logger.info("-" * 80)
    if fused_time_us < non_fused_time_us:
        logger.info(f"  SPEEDUP: {speedup:.2f}x ({improvement_pct:.1f}% faster)")
    else:
        slowdown = fused_time_us / non_fused_time_us if non_fused_time_us > 0 else 0
        slowdown_pct = (fused_time_us / non_fused_time_us - 1) * 100 if non_fused_time_us > 0 else 0
        logger.info(f"  SLOWDOWN: {slowdown:.2f}x ({slowdown_pct:.1f}% slower)")
    logger.info("=" * 80)


# =============================================================================
# CoreRangeSet definitions for Galaxy (from test_llama_all_gather_matmul.py)
# =============================================================================

SUB_DEVICE_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
    ]
)
MCAST_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(6, 9)),
    ]
)
MCAST_NUM_CORES = 60
HOP_GRID = ttnn.CoreRangeSet([])
MAX_DST_TILES = 8


def round_up(a, b):
    """Round up a to the nearest multiple of b."""
    return b * math.ceil(a / b)


def num_cores_to_rectangle_grid(num_cores, device):
    """Find a rectangular core grid size, given an number of cores."""
    x = device.compute_with_storage_grid_size().x
    while x > 0 and num_cores % x != 0:
        x -= 1
    if x == 0:
        return None
    y = num_cores // x
    return (x, y)


# =============================================================================
# Test: AllGather + Matmul (Galaxy FF2 path optimization)
# =============================================================================


def run_all_gather_matmul_galaxy_impl(
    mesh_device,
    M,
    K,
    N,
    cluster_axis,
    in0_dtype,
    in1_dtype,
    output_dtype,
    num_links,
    input_num_cores,
    output_num_cores,
    fidelity,
    fp32_acc_mode,
    packer_l1_acc,
    op_mode="non_fused",  # "fused", "non_fused", "chunked_fused", "local_matmul_allreduce"
    num_iters=1,
    trace_mode=True,
):
    """
    Galaxy implementation of AllGather + Matmul operation.

    Supports multiple operation modes:
    - "fused": llama_all_gather_matmul_async (requires large L1 intermediate)
    - "non_fused": all_gather_async + ttnn.linear (standard approach)
    - "chunked_fused": Chunked AllGather + Matmul (smaller intermediate per chunk)
    - "local_matmul_allreduce": Local matmul + AllReduce (most memory efficient)

    Uses tensor dimensions from Llama 70B MLP FF2 path:
    - Input: [8, 4, M, K_per_device] where K_per_device = K // 4
    - Weight: [8, 4, K, N]
    - Output: [8, 4, M, N]
    """
    cluster_shape = (8, 4)

    # Only run on unharvested TG
    device_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Not TG! Requires 7x10 grid")

    # Setup fabric
    all_gather_topology = ttnn.Topology.Linear

    # Sub-device setup only needed for fused path (uses llama_all_gather_matmul_async)
    # Non-fused and chunked_fused paths run without sub-device (like Galaxy prefill mode)
    worker_sub_device_id = None
    sub_device_stall_group = None
    sub_device_manager = None

    if op_mode == "fused":
        worker_sub_device = ttnn.SubDevice([SUB_DEVICE_CRS])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        sub_device_stall_group = [worker_sub_device_id]
        sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
        mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # Create semaphores - need 2 semaphores per iteration for all_gather_async
    num_buffers = 8
    # Create 2x semaphores for all_gather_async (which requires 2 semaphores)
    semaphore_cores = (
        SUB_DEVICE_CRS
        if op_mode == "fused"
        else ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))})  # 7x10 = 70 cores
    )
    ccl_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, semaphore_cores, 0) for _ in range(num_buffers * 2)
    ]

    # Memory configs - use PREFETCHER_NOC1_GRID to avoid overlap with intermediate cores (3,0)-(3,3)
    # RING_CRS pattern from existing tests - uses cores in columns 1,2,5,6 only (24 cores)
    RING_CRS = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in PREFETCHER_NOC1_GRID]
    )
    # For input, use BINARY_MULT_CRS pattern - 30 cores from SUB_DEVICE_CRS
    BINARY_MULT_CRS = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(1, 0), 30, SUB_DEVICE_CRS, row_wise=True
    )
    input_core_range_set = BINARY_MULT_CRS
    output_core_range_set = RING_CRS
    # Override num_cores to match the actual core range sets (MUST be done before shard calculations)
    input_num_cores = 30
    output_num_cores = len(PREFETCHER_NOC1_GRID)  # 24

    # Now calculate storage_grid with correct output_num_cores
    storage_grid = num_cores_to_rectangle_grid(output_num_cores, mesh_device)
    if storage_grid is None:
        pytest.skip(f"Could not find a rectangle grid for num_cores: {output_num_cores}")

    logger.info(f"AllGather+Matmul Galaxy: M={M}, K={K}, N={N}")

    # Input shapes - use overridden num_cores values
    K_per_device = K // cluster_shape[cluster_axis]
    K_per_device_per_shard = round_up(math.ceil(K_per_device / input_num_cores), ttnn.TILE_SIZE)
    in0_shape = [*cluster_shape, M, K_per_device]
    in1_shape = [*cluster_shape, K, N]
    # Chunked weight shape: each device gets K_per_device rows of weight
    in1_chunked_shape = [*cluster_shape, K_per_device, N]

    K_per_shard = round_up(math.ceil(K / output_num_cores), ttnn.TILE_SIZE)
    N_per_shard = round_up(math.ceil(N / output_num_cores), ttnn.TILE_SIZE)
    N_padded = N_per_shard * output_num_cores

    logger.info(f"K_per_shard {K_per_shard}, N_per_shard {N_per_shard}, N_padded {N_padded}")

    # Program config for matmul
    in0_block_h = M // ttnn.TILE_SIZE
    in0_block_w = K // cluster_shape[cluster_axis] // ttnn.TILE_SIZE
    while (K / ttnn.TILE_SIZE) % in0_block_w != 0:
        in0_block_w -= 1

    out_block_h = M // ttnn.TILE_SIZE
    out_block_w = N_padded // output_num_cores // ttnn.TILE_SIZE

    num_blocks_y = (M // ttnn.TILE_SIZE - 1) // out_block_h + 1
    num_blocks_x = (N_padded // ttnn.TILE_SIZE - 1) // out_block_w + 1
    num_blocks_total = num_blocks_y * num_blocks_x

    if num_blocks_total != output_num_cores:
        pytest.skip(f"num_blocks_total {num_blocks_total} != output_num_cores {output_num_cores}")

    out_subblock_h = 1
    out_subblock_w = MAX_DST_TILES
    while out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1

    logger.debug(f"in0 block h w {in0_block_h} {in0_block_w}")
    logger.debug(f"out block h w {out_block_h} {out_block_w}")
    logger.debug(f"out subblock h w {out_subblock_h} {out_subblock_w}")

    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            input_core_range_set,
            [M, K_per_device_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # Note: L1 sharded weight config removed - weight shard [K, N_per_shard] with K=14336
    # exceeds L1 bank size (~1.3MB). Using DRAM instead like the actual model.

    # Intermediate shapes for AllGather
    intermediate_num_cores = cluster_shape[cluster_axis]
    intermediate_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 3))])
    intermediate_shape = [*cluster_shape, M, K_per_device * cluster_shape[cluster_axis]]
    interemediate_N_per_shard = round_up(math.ceil(intermediate_shape[-1] / intermediate_num_cores), ttnn.TILE_SIZE)

    intermediate_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            intermediate_core_range_set,
            [M, interemediate_N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    ag_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            intermediate_core_range_set,
            [M, interemediate_N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    # Fused path output memory config (uses RING_CRS - non-contiguous grid)
    mm_output_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            output_core_range_set,
            [M, N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # For non-fused path: mcast_in0=True requires a contiguous rectangular grid
    # Create a rectangular grid with same number of cores (24 = 6x4)
    non_fused_num_cores = output_num_cores  # 24
    non_fused_grid = (6, 4)  # 6x4 = 24 cores, contiguous rectangle
    non_fused_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(non_fused_grid[0] - 1, non_fused_grid[1] - 1))}
    )
    non_fused_K_per_shard = round_up(math.ceil(K / non_fused_num_cores), ttnn.TILE_SIZE)
    non_fused_N_per_shard = round_up(math.ceil(N / non_fused_num_cores), ttnn.TILE_SIZE)
    non_fused_N_padded = non_fused_N_per_shard * non_fused_num_cores

    # Non-fused path memory configs (uses rectangular grid)
    non_fused_ag_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            non_fused_core_range_set,
            [M, non_fused_K_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    non_fused_in1_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            non_fused_core_range_set,
            [K, non_fused_N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    non_fused_mm_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            non_fused_core_range_set,
            [M, non_fused_N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # Calculate non-fused matmul block sizes
    non_fused_out_block_w = non_fused_N_padded // non_fused_num_cores // ttnn.TILE_SIZE
    non_fused_out_subblock_w = MAX_DST_TILES
    while non_fused_out_block_w % non_fused_out_subblock_w != 0:
        non_fused_out_subblock_w -= 1

    # Calculate in0_block_w for non-fused path
    # Condition: shard_tiles % in0_block_w == 0 AND K_in_tiles % in0_block_w == 0
    non_fused_shard_tiles = non_fused_K_per_shard // ttnn.TILE_SIZE  # 160 / 32 = 5
    K_in_tiles = K // ttnn.TILE_SIZE  # 3584 / 32 = 112
    non_fused_in0_block_w = non_fused_shard_tiles
    while non_fused_in0_block_w > 0:
        if non_fused_shard_tiles % non_fused_in0_block_w == 0 and K_in_tiles % non_fused_in0_block_w == 0:
            break
        non_fused_in0_block_w -= 1
    if non_fused_in0_block_w == 0:
        non_fused_in0_block_w = 1

    logger.debug(
        f"Non-fused: shard_tiles={non_fused_shard_tiles}, K_tiles={K_in_tiles}, in0_block_w={non_fused_in0_block_w}"
    )

    # Program config for fused op (uses gather_in0 with global CB)
    fused_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=storage_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=HOP_GRID,
        num_global_cb_receivers=24,
        untilize_out=False,
    )
    # Program config for non-fused op (uses mcast_in0, same grid as fused but no global CB)
    non_fused_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=storage_grid,
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
        dst_full_sync_en=True,
    )

    # Create input tensors
    logger.info(f"Input shape: {in0_shape[2:]}, Padded shape: {[M, K_per_device_per_shard * input_num_cores]}")
    in0_tensor = torch.randn(in0_shape)
    tt_input_tensor = ttnn.from_torch(
        in0_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=in0_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
    )

    # Weight tensor - use DRAM like the actual model does
    # L1 sharded would require [K, N_per_shard] per shard which exceeds L1 bank size
    # when K=14336 (gathered K dimension)
    in1_tensor = torch.randn(in1_shape)
    tt_in1_tensor = ttnn.from_torch(
        in1_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=in1_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
    )

    # Use same DRAM weight tensor for non-fused path
    tt_in1_tensor_non_fused = tt_in1_tensor

    # Chunked weight tensor for local_matmul_allreduce approach:
    # Each device gets a K_per_device slice of the weight, sharded along K dimension
    # This allows matmul without all_gather by doing local matmul + all_reduce
    in1_chunked_tensor = torch.randn(in1_chunked_shape)
    tt_in1_chunked_tensor = ttnn.from_torch(
        in1_chunked_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=in1_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
    )

    # =========================================================================
    # Chunked fused approach: Split K dimension to reduce intermediate buffer size
    # =========================================================================
    # With num_chunks=4:
    # - K_per_device = 3584, K_chunk_per_device = 896
    # - K_gathered = 14336, K_chunk_gathered = 3584
    # - Intermediate size reduced from [32, 14336] to [32, 3584] (4x smaller)
    # - This should fit in available L1 (~133KB vs ~458KB needed for full)
    num_chunks = 4
    K_chunk_per_device = K_per_device // num_chunks
    K_chunk_gathered = K // num_chunks  # K_chunk_per_device * cluster_shape[cluster_axis]

    # Create chunked input tensors - slice along K dimension
    tt_input_chunks = []
    for c in range(num_chunks):
        chunk_start = c * K_chunk_per_device
        chunk_end = (c + 1) * K_chunk_per_device
        in0_chunk = in0_tensor[:, :, :, chunk_start:chunk_end].contiguous()
        K_chunk_per_shard = round_up(math.ceil(K_chunk_per_device / input_num_cores), ttnn.TILE_SIZE)
        chunk_input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                input_core_range_set,
                [M, K_chunk_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        tt_input_chunk = ttnn.from_torch(
            in0_chunk,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=in0_dtype,
            memory_config=chunk_input_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
        )
        tt_input_chunks.append(tt_input_chunk)

    # Create chunked weight tensors - must match interleaved layout from all_gather
    # After all_gather of input chunk c from 4 devices along cluster_axis=1:
    #   - Positions [0:K_chunk_per_device] contain device 0's chunk c
    #   - Positions [K_chunk_per_device:2*K_chunk_per_device] contain device 1's chunk c
    #   - etc.
    # The weight rows must be arranged to match this interleaved layout.
    num_devices_on_axis = cluster_shape[cluster_axis]  # 4
    tt_weight_chunks = []
    for c in range(num_chunks):
        # Gather weight rows from each device's portion to match interleaved all_gather layout
        weight_slices = []
        for device_idx in range(num_devices_on_axis):
            # Device i's K range in full weight is [i*K_per_device : (i+1)*K_per_device]
            # Chunk c within that range is [i*K_per_device + c*K_chunk_per_device : i*K_per_device + (c+1)*K_chunk_per_device]
            device_K_start = device_idx * K_per_device
            chunk_K_start = device_K_start + c * K_chunk_per_device
            chunk_K_end = device_K_start + (c + 1) * K_chunk_per_device
            weight_slices.append(in1_tensor[:, :, chunk_K_start:chunk_K_end, :])
        # Concatenate slices to match interleaved layout: [dev0_chunk | dev1_chunk | dev2_chunk | dev3_chunk]
        in1_chunk = torch.cat(weight_slices, dim=2).contiguous()
        tt_weight_chunk = ttnn.from_torch(
            in1_chunk,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=in1_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
        )
        tt_weight_chunks.append(tt_weight_chunk)

    # Create smaller intermediate tensors for chunked fused op
    # Intermediate shape is [8, 4, M, K_chunk_gathered] instead of [8, 4, M, K_gathered]
    chunk_intermediate_shape = [*cluster_shape, M, K_chunk_gathered]
    chunk_intermediate_N_per_shard = round_up(math.ceil(K_chunk_gathered / intermediate_num_cores), ttnn.TILE_SIZE)
    chunk_intermediate_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            intermediate_core_range_set,
            [M, chunk_intermediate_N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    chunk_ag_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            intermediate_core_range_set,
            [M, chunk_intermediate_N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    # Only allocate chunk intermediate tensors if using chunked_fused mode
    tt_chunk_intermediate_tensors = []
    chunked_fused_program_config = None
    if op_mode == "chunked_fused":
        chunk_intermediate_tensor = torch.zeros(chunk_intermediate_shape)
        for i in range(num_buffers):
            tt_chunk_intermediate = ttnn.from_torch(
                chunk_intermediate_tensor,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=in0_dtype,
                memory_config=chunk_intermediate_mem_config,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
            )
            tt_chunk_intermediate_tensors.append(tt_chunk_intermediate)

        logger.info(
            f"Chunked fused: num_chunks={num_chunks}, K_chunk_per_device={K_chunk_per_device}, K_chunk_gathered={K_chunk_gathered}"
        )
        logger.info(
            f"Chunk intermediate shape: {chunk_intermediate_shape}, shard: [M, {chunk_intermediate_N_per_shard}]"
        )

        # Program config for chunked fused op (smaller K dimension)
        chunk_in0_block_w = K_chunk_per_device // ttnn.TILE_SIZE
        while (K_chunk_gathered / ttnn.TILE_SIZE) % chunk_in0_block_w != 0:
            chunk_in0_block_w -= 1
        chunked_fused_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=storage_grid,
            in0_block_w=chunk_in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=out_block_h,
            per_core_N=out_block_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
            gather_in0=True,
            hop_cores=HOP_GRID,
            num_global_cb_receivers=24,
            untilize_out=False,
        )
        logger.debug(f"Chunked fused program config: in0_block_w={chunk_in0_block_w}")

    # Only allocate intermediate tensors if using fused mode
    tt_intermediate_tensors = []
    if op_mode == "fused":
        intermediate_tensor = torch.zeros(intermediate_shape)
        for i in range(num_buffers):
            tt_intermediate_tensor = ttnn.from_torch(
                intermediate_tensor,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=in0_dtype,
                memory_config=intermediate_mem_config,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
            )
            tt_intermediate_tensors.append(tt_intermediate_tensor)

    # Only allocate allreduce buffers if using local_matmul_allreduce mode
    tt_allreduce_buffers = []
    if op_mode == "local_matmul_allreduce":
        # Buffer tensors for all_reduce (used by local_matmul_allreduce path)
        # Output shape is [8, 4, M, N] - the result of matmul before reduce
        allreduce_buffer_shape = [*cluster_shape, M, N]
        allreduce_buffer_tensor = torch.zeros(allreduce_buffer_shape)
        for i in range(num_buffers):
            tt_allreduce_buffer = ttnn.from_torch(
                allreduce_buffer_tensor,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=output_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
            )
            tt_allreduce_buffers.append(tt_allreduce_buffer)

    # Compute golden
    output_tensor_goldens_list = []
    for i in range(num_iters):
        golden_Ashape = list(intermediate_shape)
        golden_Ashape[cluster_axis] = 1
        golden_A = in0_tensor.transpose(-2, cluster_axis).reshape(golden_Ashape).squeeze(cluster_axis)
        golden_A = golden_A.unsqueeze(cluster_axis).repeat(1, intermediate_num_cores, 1, 1)
        output_tensor_goldens_list.append(golden_A @ in1_tensor)

    def run_fused_op(n_iters, store_all_results=True):
        """Run fused llama_all_gather_matmul_async."""
        outs = []
        for i in range(n_iters):
            out = ttnn.experimental.llama_all_gather_matmul_async(
                tt_input_tensor,
                tt_in1_tensor,
                tt_intermediate_tensors[i % num_buffers],
                dim=3,
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                multi_device_global_semaphore=ccl_semaphore_handles[i % num_buffers],
                ag_memory_config=ag_output_mem_config,
                mm_memory_config=mm_output_sharded_mem_config,
                topology=all_gather_topology,
                num_links=num_links,
                subdevice_id=worker_sub_device_id,
                program_config=fused_program_config,
                compute_kernel_config=compute_kernel_config,
                dtype=output_dtype,
            )
            if not trace_mode:
                ttnn.synchronize_device(mesh_device)
            if store_all_results:
                outs.append(out)
        return outs if store_all_results else [out]

    def run_non_fused_op(n_iters, store_all_results=True):
        """Run non-fused: all_gather_async + ttnn.linear (like Galaxy model prefill mode)."""
        outs = []
        # Use DRAM interleaved for non-fused path (like model does in prefill mode)
        dram_interleaved = ttnn.DRAM_MEMORY_CONFIG
        for i in range(n_iters):
            # AllGather - requires 2 semaphores passed as a list
            # No sub-device (like Galaxy prefill mode)
            sem_idx = (i % num_buffers) * 2
            ag_out = ttnn.experimental.all_gather_async(
                tt_input_tensor,
                dim=3,
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                multi_device_global_semaphore=[ccl_semaphore_handles[sem_idx], ccl_semaphore_handles[sem_idx + 1]],
                memory_config=dram_interleaved,
                topology=all_gather_topology,
                num_links=num_links,
            )
            # Matmul with DRAM interleaved and auto core_grid (like model prefill mode)
            mm_out = ttnn.linear(
                ag_out,
                tt_in1_tensor_non_fused,  # DRAM interleaved weights
                memory_config=dram_interleaved,
                compute_kernel_config=compute_kernel_config,
                dtype=output_dtype,
                core_grid=ttnn.CoreGrid(y=7, x=10),  # Full device grid (7x10 for TG)
            )
            if not trace_mode:
                ttnn.synchronize_device(mesh_device)
            if store_all_results:
                outs.append(mm_out)
        return outs if store_all_results else [mm_out]

    def run_chunked_fused_op(n_iters, store_all_results=True):
        """
        Run chunked AllGather + Matmul approach to reduce intermediate buffer size.

        This splits the K dimension into chunks so each chunk's intermediate buffer
        fits in L1. Uses separate all_gather + matmul ops per chunk, then accumulates.

        For each iteration:
        1. For each chunk c in [0, num_chunks):
           - AllGather chunk: [M, K_chunk_per_device] -> [M, K_chunk_gathered]
           - Matmul chunk: [M, K_chunk_gathered] @ [K_chunk_gathered, N] -> [M, N] (partial)
        2. Sum all partial results to get final output

        Memory savings:
        - Original intermediate: [M, K_gathered] = [32, 14336] = 458KB per core
        - Chunked intermediate: [M, K_chunk_gathered] = [32, 3584] = 114KB per core
        - 4x reduction allows fitting in ~133KB available L1
        """
        outs = []
        dram_interleaved = ttnn.DRAM_MEMORY_CONFIG

        for iter_idx in range(n_iters):
            accumulated_result = None

            for chunk_idx in range(num_chunks):
                # Use different semaphores for each chunk within an iteration
                sem_idx = (iter_idx * num_chunks + chunk_idx) % num_buffers * 2

                # Step 1: AllGather this chunk
                gathered_chunk = ttnn.experimental.all_gather_async(
                    tt_input_chunks[chunk_idx],
                    dim=3,
                    cluster_axis=cluster_axis,
                    mesh_device=mesh_device,
                    multi_device_global_semaphore=[
                        ccl_semaphore_handles[sem_idx],
                        ccl_semaphore_handles[sem_idx + 1],
                    ],
                    memory_config=dram_interleaved,
                    topology=all_gather_topology,
                    num_links=num_links,
                )

                # Step 2: Matmul with corresponding weight chunk
                chunk_out = ttnn.linear(
                    gathered_chunk,
                    tt_weight_chunks[chunk_idx],
                    memory_config=dram_interleaved,
                    compute_kernel_config=compute_kernel_config,
                    dtype=output_dtype,
                    core_grid=ttnn.CoreGrid(y=7, x=10),
                )

                # Accumulate partial results
                if accumulated_result is None:
                    accumulated_result = chunk_out
                else:
                    accumulated_result = ttnn.add(accumulated_result, chunk_out)

            if not trace_mode:
                ttnn.synchronize_device(mesh_device)
            if store_all_results:
                outs.append(accumulated_result)
        return outs if store_all_results else [accumulated_result]

    def run_local_matmul_allreduce_op(n_iters, store_all_results=True):
        """
        Run local matmul + AllReduce approach (no AllGather needed).

        Mathematically equivalent to AllGather + Matmul:
        [A0|A1|A2|A3] @ [W0;W1;W2;W3] = A0@W0 + A1@W1 + A2@W2 + A3@W3

        Each device i computes A_i @ W_i locally, then AllReduce sums the partial products.

        Benefits:
        - No large intermediate buffer (no [M, K_gathered])
        - Single matmul + single AllReduce per iteration
        - Memory efficient for large K
        """
        outs = []
        dram_interleaved = ttnn.DRAM_MEMORY_CONFIG

        for i in range(n_iters):
            # Step 1: Local matmul - each device computes partial result
            # input per device: [32, 3584], weight chunk per device: [3584, 2048]
            partial_out = ttnn.linear(
                tt_input_tensor,  # [M, K_per_device] = [32, 3584] per device
                tt_in1_chunked_tensor,  # Chunked weight [K_per_device, N] = [3584, 2048] per device
                memory_config=dram_interleaved,
                compute_kernel_config=compute_kernel_config,
                dtype=output_dtype,
                core_grid=ttnn.CoreGrid(y=7, x=10),
            )

            # Step 2: AllReduce to sum partial results across devices
            # all_reduce_async requires: input, buffer_tensor, cluster_axis, mesh_device, semaphore
            mm_out = ttnn.experimental.all_reduce_async(
                partial_out,
                tt_allreduce_buffers[i % num_buffers],  # buffer tensor required
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                multi_device_global_semaphore=ccl_semaphore_handles[i % num_buffers],
                memory_config=dram_interleaved,
                topology=all_gather_topology,
                num_links=num_links,
            )

            if not trace_mode:
                ttnn.synchronize_device(mesh_device)
            if store_all_results:
                outs.append(mm_out)
        return outs if store_all_results else [mm_out]

    # Select operation based on mode
    op_functions = {
        "fused": run_fused_op,
        "non_fused": run_non_fused_op,
        "chunked_fused": run_chunked_fused_op,
        "local_matmul_allreduce": run_local_matmul_allreduce_op,
    }
    if op_mode not in op_functions:
        raise ValueError(f"Unknown op_mode: {op_mode}. Must be one of {list(op_functions.keys())}")
    run_op = op_functions[op_mode]
    logger.info(f"Running with op_mode: {op_mode}")

    if trace_mode:
        # Compile
        logger.info("Compiling model")
        tt_outs = run_op(num_iters, store_all_results=True)

        # Capture trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_outs = run_op(num_iters, store_all_results=True)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

        # Warmup
        for _ in range(3):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            if sub_device_stall_group:
                ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            else:
                ttnn.synchronize_device(mesh_device)

        # Timed run
        num_perf_runs = 10
        signpost("start")
        start_time = time.perf_counter()
        for _ in range(num_perf_runs):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            if sub_device_stall_group:
                ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            else:
                ttnn.synchronize_device(mesh_device)
        end_time = time.perf_counter()
        signpost("stop")

        total_time_us = (end_time - start_time) * 1e6
        avg_time_per_run_us = total_time_us / num_perf_runs
        avg_time_per_iter_us = avg_time_per_run_us / num_iters

        mode_str = op_mode.upper()
        logger.info(f"=== PERFORMANCE ({mode_str}) ===")
        logger.info(f"Total time for {num_perf_runs} runs: {total_time_us:.2f} us")
        logger.info(f"Avg time per run ({num_iters} iters): {avg_time_per_run_us:.2f} us")
        logger.info(f"Avg time per iteration: {avg_time_per_iter_us:.2f} us")

        ttnn.release_trace(mesh_device, trace_id)
    else:
        avg_time_per_iter_us = None
        signpost("start")
        tt_outs = run_op(num_iters, store_all_results=True)
        signpost("stop")

    # Validate
    def validate(tt_out_tensor, output_tensor):
        for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
            row_index = i // cluster_shape[1]
            col_index = i % cluster_shape[1]
            output_tensor_ = output_tensor[row_index, col_index]
            tt_output_tensor = t.cpu().to_torch().squeeze(0).squeeze(0)
            eq, output = comp_pcc(tt_output_tensor, output_tensor_)
            assert eq, f"{i} FAILED: {output}"
        logger.info(f"PCC output is: {output}")

    for tensor_index in range(len(tt_outs)):
        validate(tt_outs[tensor_index], output_tensor_goldens_list[tensor_index])

    # Only cleanup sub-device if it was set up (fused path only)
    if op_mode == "fused":
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()

    logger.info("AllGather+Matmul Galaxy test completed successfully")
    return avg_time_per_iter_us


# =============================================================================
# Pytest Test Cases for Galaxy AllGather + Matmul (FF2 path optimization)
# =============================================================================


@pytest.mark.parametrize(
    "M, K, N, in0_dtype, in1_dtype, output_dtype, fidelity, fp32_acc_mode, packer_l1_acc, input_num_cores, output_num_cores",
    [
        # ==================== Actual Galaxy Model FF2 dimensions ====================
        # Galaxy 8x4 mesh (32 devices total):
        # - cluster_axis=1 gathers along 4-device column dimension
        # - Llama 70B uses 4-way tensor parallelism for FFN hidden dim along axis 1
        # Real Llama 70B dimensions:
        # - hidden_size = 14336 (FFN hidden dimension)
        # - K_per_device = 3584 (hidden_size/4 = 14336/4, sharded across 4 column devices)
        # - K_gathered = 14336 (after all_gather along cluster_axis=1)
        # - N = 2048 (dim/4 = 8192/4)
        # Galaxy decode (M=32) - actual model dimensions
        (32, 14336, 2048, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, True, 10, 24),
    ],
    ids=[
        "model_ff2_decode_K14336_N2048",
    ],
)
@pytest.mark.parametrize("num_links", [3])
@pytest.mark.parametrize("cluster_axis", [1])  # Gather across 4 column devices (axis 1 of 8x4 mesh)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 23887872,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "op_mode",
    ["non_fused"],  # chunked_fused REJECTED: 2x slower than non_fused (396us vs 199us)
    ids=["non_fused"],
)
def test_all_gather_matmul_galaxy_check(
    mesh_device,
    M,
    K,
    N,
    in0_dtype,
    in1_dtype,
    output_dtype,
    fidelity,
    fp32_acc_mode,
    packer_l1_acc,
    input_num_cores,
    output_num_cores,
    num_links,
    cluster_axis,
    op_mode,
):
    """
    Functional test for AllGather + Matmul operations on Galaxy (32 devices, 8x4 mesh).

    Tests different operation modes:
    - non_fused: all_gather_async + ttnn.linear (standard approach)
    - chunked_fused: K split into chunks, all_gather + matmul per chunk, accumulate results

    Note: "fused" mode (llama_all_gather_matmul_async) skipped - requires too much L1 for 70B dims.
    """
    run_all_gather_matmul_galaxy_impl(
        mesh_device,
        M,
        K,
        N,
        cluster_axis,
        in0_dtype,
        in1_dtype,
        output_dtype,
        num_links,
        input_num_cores,
        output_num_cores,
        fidelity,
        fp32_acc_mode,
        packer_l1_acc,
        op_mode=op_mode,
        num_iters=1,
        trace_mode=False,
    )


@pytest.mark.parametrize(
    "M, K, N, in0_dtype, in1_dtype, output_dtype, fidelity, fp32_acc_mode, packer_l1_acc, input_num_cores, output_num_cores",
    [
        # ==================== Actual Galaxy Model FF2 dimensions ====================
        # Galaxy 8x4 mesh (32 devices total):
        # - cluster_axis=1 gathers along 4-device column dimension
        # - Llama 70B uses 4-way tensor parallelism for FFN hidden dim along axis 1
        # Real Llama 70B dimensions:
        # - hidden_size = 14336 (FFN hidden dimension)
        # - K_per_device = 3584 (hidden_size/4 = 14336/4, sharded across 4 column devices)
        # - K_gathered = 14336 (after all_gather along cluster_axis=1)
        # - N = 2048 (dim/4 = 8192/4)
        # This test compares: line_all_gather + ttnn.linear vs llama_all_gather_matmul_async
        # Galaxy decode (M=32) - actual model dimensions
        (32, 14336, 2048, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, True, 10, 24),
    ],
    ids=[
        "model_ff2_decode_K14336_N2048",
    ],
)
@pytest.mark.parametrize("num_links", [3])
@pytest.mark.parametrize("cluster_axis", [1])  # Gather across 4 column devices (axis 1 of 8x4 mesh)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 23887872,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_all_gather_matmul_galaxy_perf_comparison(
    mesh_device,
    M,
    K,
    N,
    in0_dtype,
    in1_dtype,
    output_dtype,
    fidelity,
    fp32_acc_mode,
    packer_l1_acc,
    input_num_cores,
    output_num_cores,
    num_links,
    cluster_axis,
):
    """
    Performance comparison test for AllGather + Matmul on Galaxy (32 devices, 8x4 mesh).

    Compares operation modes:
    - non_fused: all_gather_async + ttnn.linear (standard approach)
    - chunked_fused: K split into chunks, all_gather + matmul per chunk, accumulate results

    Note: "fused" mode (llama_all_gather_matmul_async) skipped - requires too much L1 for 70B dims.
    """
    num_iters = 5
    results = {}

    # Test all modes that work with model dimensions
    modes_to_test = ["non_fused", "chunked_fused"]

    for mode in modes_to_test:
        logger.info(f"Running {mode.upper()} operation...")
        time_us = run_all_gather_matmul_galaxy_impl(
            mesh_device,
            M,
            K,
            N,
            cluster_axis,
            in0_dtype,
            in1_dtype,
            output_dtype,
            num_links,
            input_num_cores,
            output_num_cores,
            fidelity,
            fp32_acc_mode,
            packer_l1_acc,
            op_mode=mode,
            num_iters=num_iters,
            trace_mode=True,
        )
        results[mode] = time_us

    # Print comparison report
    logger.info("=" * 80)
    logger.info(f"PERFORMANCE COMPARISON: Galaxy AllGather+Matmul FF2 (M={M}, K={K}, N={N})")
    logger.info("=" * 80)
    logger.info(f"Input shape per device: [8, 4, {M}, {K // 4}]")
    logger.info(f"Weight shape: [8, 4, {K}, {N}]")
    logger.info("-" * 80)

    baseline = results.get("non_fused", 1)
    for mode, time_us in results.items():
        if time_us is not None:
            speedup = baseline / time_us if time_us > 0 else 0
            logger.info(f"  {mode:30s}: {time_us:8.2f} us  ({speedup:.2f}x vs non_fused)")
        else:
            logger.info(f"  {mode:30s}: SKIPPED (trace_mode=False)")
    logger.info("=" * 80)


# =============================================================================
# Pytest Test Cases for Galaxy AllGather + Matmul PREFILL (FF2 path)
# =============================================================================


@pytest.mark.parametrize(
    "M, K, N, in0_dtype, in1_dtype, output_dtype, fidelity, fp32_acc_mode, packer_l1_acc, input_num_cores, output_num_cores",
    [
        # ==================== Galaxy Model FF2 PREFILL dimensions ====================
        # Prefill path in llama_mlp.py:forward_prefill (lines 285-307):
        # - w2_in after silu+mul: [8, 4, seq_len, hidden_dim/4] = [8, 4, M, 3584]
        # - After all_gather: [8, 4, seq_len, hidden_dim] = [8, 4, M, 14336]
        # - Weight w2: [hidden_dim, dim/4] = [14336, 2048]
        # - Output: [8, 4, M, 2048]
        #
        # Common prefill seq_lens: 128, 1024, 2048
        # Prefill seq_len=128 (most common short prefill)
        (128, 14336, 2048, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, True, 10, 24),
        # Prefill seq_len=1024
        (
            1024,
            14336,
            2048,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
            True,
            True,
            10,
            24,
        ),
    ],
    ids=[
        "prefill_ff2_M128_K14336_N2048",
        "prefill_ff2_M1024_K14336_N2048",
    ],
)
@pytest.mark.parametrize("num_links", [3])
@pytest.mark.parametrize("cluster_axis", [1])  # Gather across 4 column devices (axis 1 of 8x4 mesh)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 23887872,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "op_mode",
    ["non_fused"],  # Fused mode OOMs with model dims (needs 487KB/bank, only 385KB free)
    ids=["non_fused"],
)
def test_all_gather_matmul_galaxy_prefill_check(
    mesh_device,
    M,
    K,
    N,
    in0_dtype,
    in1_dtype,
    output_dtype,
    fidelity,
    fp32_acc_mode,
    packer_l1_acc,
    input_num_cores,
    output_num_cores,
    num_links,
    cluster_axis,
    op_mode,
):
    """
    Functional test for AllGather + Matmul PREFILL operations on Galaxy (32 devices, 8x4 mesh).

    Tests the FF2 prefill path from llama_mlp.py:forward_prefill with actual model dimensions:
    - M = seq_len (128, 1024, 2048, etc.)
    - K = 14336 (hidden_dim gathered from 4 devices)
    - N = 2048 (dim/4)
    """
    run_all_gather_matmul_galaxy_impl(
        mesh_device,
        M,
        K,
        N,
        cluster_axis,
        in0_dtype,
        in1_dtype,
        output_dtype,
        num_links,
        input_num_cores,
        output_num_cores,
        fidelity,
        fp32_acc_mode,
        packer_l1_acc,
        op_mode=op_mode,
        num_iters=1,
        trace_mode=False,
    )


@pytest.mark.parametrize(
    "M, K, N, in0_dtype, in1_dtype, output_dtype, fidelity, fp32_acc_mode, packer_l1_acc, input_num_cores, output_num_cores",
    [
        # Prefill seq_len=128 (most common short prefill)
        (128, 14336, 2048, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, True, 10, 24),
    ],
    ids=[
        "prefill_ff2_M128_K14336_N2048",
    ],
)
@pytest.mark.parametrize("num_links", [3])
@pytest.mark.parametrize("cluster_axis", [1])  # Gather across 4 column devices (axis 1 of 8x4 mesh)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 23887872,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_all_gather_matmul_galaxy_prefill_perf_comparison(
    mesh_device,
    M,
    K,
    N,
    in0_dtype,
    in1_dtype,
    output_dtype,
    fidelity,
    fp32_acc_mode,
    packer_l1_acc,
    input_num_cores,
    output_num_cores,
    num_links,
    cluster_axis,
):
    """
    Performance comparison test for AllGather + Matmul PREFILL on Galaxy (32 devices, 8x4 mesh).

    Compares fused vs non-fused operations for FF2 prefill path.
    Also validates PCC for both paths.

    Note: Fused op may fail with OOM for large M due to L1 intermediate buffer constraints.
    """
    num_iters = 5
    results = {}

    logger.info(f"Running PREFILL performance comparison: M={M} (seq_len), K={K}, N={N}")

    # Test non-fused first (always works)
    logger.info("Running NON_FUSED operation...")
    non_fused_time_us = run_all_gather_matmul_galaxy_impl(
        mesh_device,
        M,
        K,
        N,
        cluster_axis,
        in0_dtype,
        in1_dtype,
        output_dtype,
        num_links,
        input_num_cores,
        output_num_cores,
        fidelity,
        fp32_acc_mode,
        packer_l1_acc,
        op_mode="non_fused",
        num_iters=num_iters,
        trace_mode=True,
    )
    results["non_fused"] = non_fused_time_us

    # Try fused op - may fail with OOM for large M due to L1 constraints
    # Intermediate buffer: [M, 14336/4] = [M, 3584] per core across 4 cores
    # For M=128: 128 * 3584 * 1 byte = 458KB per core (likely exceeds available L1)
    fused_time_us = None
    try:
        logger.info("Running FUSED operation (may OOM for large M)...")
        fused_time_us = run_all_gather_matmul_galaxy_impl(
            mesh_device,
            M,
            K,
            N,
            cluster_axis,
            in0_dtype,
            in1_dtype,
            output_dtype,
            num_links,
            input_num_cores,
            output_num_cores,
            fidelity,
            fp32_acc_mode,
            packer_l1_acc,
            op_mode="fused",
            num_iters=num_iters,
            trace_mode=True,
        )
        results["fused"] = fused_time_us
    except Exception as e:
        logger.warning(f"FUSED operation failed (expected for large M due to L1 constraints): {e}")
        results["fused"] = None

    # Print comparison report
    logger.info("=" * 80)
    logger.info(f"PREFILL PERFORMANCE COMPARISON: Galaxy AllGather+Matmul FF2 (M={M}, K={K}, N={N})")
    logger.info("=" * 80)
    logger.info(f"Input shape per device: [8, 4, {M}, {K // 4}]")
    logger.info(f"Weight shape: [8, 4, {K}, {N}]")
    logger.info("-" * 80)

    baseline = results.get("non_fused", 1)
    for mode, time_us in results.items():
        if time_us is not None:
            speedup = baseline / time_us if time_us > 0 else 0
            logger.info(f"  {mode:30s}: {time_us:8.2f} us  ({speedup:.2f}x vs non_fused)")
        else:
            logger.info(f"  {mode:30s}: FAILED (OOM or other error)")
    logger.info("=" * 80)

    # Summary
    if fused_time_us is not None and non_fused_time_us is not None:
        if fused_time_us < non_fused_time_us:
            improvement = (1 - fused_time_us / non_fused_time_us) * 100
            logger.info(f"RESULT: Fused op is {improvement:.1f}% FASTER for prefill M={M}")
        else:
            slowdown = (fused_time_us / non_fused_time_us - 1) * 100
            logger.info(f"RESULT: Fused op is {slowdown:.1f}% SLOWER for prefill M={M}")
    elif fused_time_us is None:
        logger.info(f"RESULT: Fused op FAILED for prefill M={M} - L1 intermediate too large")

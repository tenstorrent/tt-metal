# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Galaxy (TG) tests for fused CCL operations (Matmul+ReduceScatter)

These tests validate fused CCL operations on Galaxy (32 devices, 8x4 mesh) using tensor dimensions
that match the Llama 70B model patterns.

Fused ops tested:
- ttnn.experimental.matmul_reduce_scatter_async (Matmul + ReduceScatter)

Performance comparison mode:
- Runs both fused and non-fused operations
- Reports speedup/slowdown of fused vs non-fused
"""

import torch
import pytest
import math
import time
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from ttnn import ShardTensorToMesh, ConcatMeshToTensor


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
# Test: Matmul + ReduceScatter (Galaxy version of ff1/ff3 path)
# =============================================================================


def run_matmul_reduce_scatter_galaxy_impl(
    mesh_device,
    num_devices,
    rs_input_shape,
    mm_shard_dim,
    rs_scatter_dim,
    num_links,
    mm_weights_shape,
    rs_input_dtype,
    layout,
    matmul_weights_dtype,
    max_in0_block_w,
    use_bias,
    mem_config_input,
    mem_config_rs,
    mem_config_mm,
    rs_topology,
    use_non_fused,
    mem_config_weights=None,
    num_iters=1,
    enable_trace=True,
):
    """
    Galaxy implementation of Matmul + ReduceScatter fused operation.

    Uses tensor dimensions from Llama 70B MLP ff1/ff3 path.
    Galaxy mesh: (8, 4) = 32 devices with FABRIC_1D topology.
    """
    torch.manual_seed(0)

    tile = (32, 32)

    # Set the default config
    if mem_config_weights is None:
        mem_config_weights = mem_config_rs

    # Set up sub-device
    ccl_sub_device_crs, worker_sub_device_id, sub_device_stall_group = setup_sub_device(mesh_device)

    # Create semaphores
    ccl_semaphore_handles = [create_global_semaphores(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)]

    # Create persistent output buffers
    logger.info("Creating persistent buffers")
    rs_num_batches = rs_input_shape[0]
    single_batch_input_shape = rs_input_shape[:]
    single_batch_input_shape[2] //= rs_num_batches
    persistent_intermediate_buffers = [
        ttnn.from_torch(
            torch.zeros(single_batch_input_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=rs_input_dtype,
            memory_config=mem_config_rs,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(num_iters)
    ]
    rs_output_shape = rs_input_shape[:]
    rs_output_shape[3] //= num_devices
    persistent_output_buffers = [
        ttnn.from_torch(
            torch.zeros(rs_output_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=rs_input_dtype,
            memory_config=mem_config_rs,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(num_iters)
    ]
    logger.info("Done creating persistent buffers")

    # Matmul weight setup
    weights_tensor = torch.randn(mm_weights_shape).bfloat16()
    weight_tt = ttnn.from_torch(
        weights_tensor,
        dtype=matmul_weights_dtype,
        layout=layout,
        device=mesh_device,
        memory_config=mem_config_weights,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=mm_shard_dim),
    )

    if use_bias:
        bias_tensor_padded = torch.randn([1, 1, 1, rs_input_shape[3]]).float()
        bias_tensor_scaled = bias_tensor_padded * (1 / num_devices)
        bias_tt = ttnn.from_torch(
            bias_tensor_scaled,
            dtype=matmul_weights_dtype,
            layout=layout,
            device=mesh_device,
            memory_config=mem_config_weights,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            tile=ttnn.Tile(tile),
        )
    else:
        bias_tt = None
        bias_tensor_padded = None

    # Configs for ttnn.matmul
    # For large tensors, we need to adjust the grid to avoid L1 overflow
    M_tiles = rs_input_shape[2] // 32
    N_tiles = rs_input_shape[3] // 32

    # Galaxy has 7x10 grid on unharvested devices
    if N_tiles > 400:  # Very large N (like 28672 -> 896 tiles)
        core_grid = (7, 4)
        per_core_M = max(1, math.ceil(M_tiles / core_grid[1]))
        per_core_N = max(1, math.ceil(N_tiles / core_grid[0]))
        if per_core_M * per_core_N > 64:
            logger.info(
                f"Large tensor detected: M_tiles={M_tiles}, N_tiles={N_tiles}, per_core_M*N={per_core_M * per_core_N}"
            )
            logger.info(f"Skipping this test configuration - L1 memory would overflow")
            cleanup_sub_device(mesh_device)
            pytest.skip(f"Tensor too large for L1: per_core_M={per_core_M}, per_core_N={per_core_N}")
        else:
            in0_block_w = min(max_in0_block_w, mm_weights_shape[2] // num_devices // 32 // core_grid[0])
            in0_block_w = max(1, in0_block_w)
            out_block_w = max(1, per_core_N // 2)
            program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=core_grid,
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=per_core_M,
                per_core_N=per_core_N,
                out_block_w=out_block_w,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )
    else:
        core_grid = (7, 6)
        in0_block_w = min(max_in0_block_w, mm_weights_shape[2] // num_devices // 32 // core_grid[0])
        in0_block_w = max(1, in0_block_w)
        per_core_M = max(1, math.ceil(M_tiles / core_grid[1]))
        per_core_N = max(1, math.ceil(N_tiles / core_grid[0]))
        out_block_w = max(1, per_core_N // 2)

        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=core_grid,
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            out_block_w=out_block_w,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # MM input setup
    logger.info(f"Matmul+ReduceScatter Galaxy: rs_input_shape={rs_input_shape}, mm_weights_shape={mm_weights_shape}")
    logger.info(f"Num devices: {num_devices}, Scatter dim: {rs_scatter_dim}")

    # Get mesh shape from device
    mesh_shape = mesh_device.shape
    num_rows = mesh_shape[0]  # 8 for Galaxy
    num_cols = mesh_shape[1]  # 4 for Galaxy

    tt_input_tensor_mesh_list = []
    torch_input_tensor_list = []

    for i in range(num_iters):
        mm_input_shape = [rs_input_shape[0], 1, rs_input_shape[2], mm_weights_shape[2]]
        mm_input_tensor = torch.rand(mm_input_shape).bfloat16()
        input_tensors = torch.chunk(mm_input_tensor, num_devices, 3)
        torch_input_tensor_list.append(input_tensors)

        # For Galaxy (8x4 mesh):
        # - Replicate across rows (dim 0) - same data on all 8 rows
        # - Shard across columns (dim 1) - scatter across 4 devices per column
        input_tensor_mesh = ttnn.from_torch(
            mm_input_tensor,
            device=mesh_device,
            layout=layout,
            dtype=rs_input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(
                    [ttnn.PlacementReplicate(), ttnn.PlacementShard(3)], ttnn.MeshShape(num_rows, num_cols)
                ),
            ),
        )
        tt_input_tensor_mesh_list.append(input_tensor_mesh)

    # Compute golden outputs
    torch_reduce_scatter_output_list = []
    torch_matmul_output_list = []
    for i in range(num_iters):
        matmul_input = torch.cat(torch_input_tensor_list[i], dim=3)
        if use_bias:
            matmul_output = torch.matmul(matmul_input, weights_tensor) + bias_tensor_padded
        else:
            matmul_output = torch.matmul(matmul_input, weights_tensor)
        scatter_output = torch.chunk(matmul_output, num_devices, rs_scatter_dim)
        torch_reduce_scatter_output_list.append(scatter_output)
        torch_matmul_output_list.append(matmul_output)

    # Run operation
    tt_reduce_scatter_output_list = []
    tt_matmul_output_list = []

    def run_op(i):
        if use_non_fused:
            tt_matmul_out_tensor = ttnn.linear(
                tt_input_tensor_mesh_list[i],
                weight_tt,
                bias=bias_tt,
                memory_config=mem_config_mm,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )
            tt_reduce_scatter_output_tensor = ttnn.experimental.reduce_scatter_minimal_async(
                tt_matmul_out_tensor,
                persistent_output_buffers=[persistent_intermediate_buffers[i], persistent_output_buffers[i]],
                dim=rs_scatter_dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                num_links=num_links,
                memory_config=mem_config_rs,
                topology=rs_topology,
                subdevice_id=worker_sub_device_id,
            )
        else:
            tt_matmul_out_tensor, tt_reduce_scatter_output_tensor = ttnn.experimental.matmul_reduce_scatter_async(
                tt_input_tensor_mesh_list[i],
                weight_tt,
                persistent_intermediate_buffer=persistent_intermediate_buffers[i],
                persistent_output_buffer=persistent_output_buffers[i],
                dim=rs_scatter_dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                reduce_scatter_core_grid_offset=(0, 6),
                bias=bias_tt,
                num_links=num_links,
                memory_config_rs=mem_config_rs,
                topology=rs_topology,
                subdevice_id=worker_sub_device_id,
                memory_config_mm=mem_config_mm,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )
        return tt_matmul_out_tensor, tt_reduce_scatter_output_tensor

    if enable_trace:
        # Compile the op
        for i in range(num_iters):
            tt_matmul_out_tensor, tt_reduce_scatter_output_tensor = run_op(i)
        logger.info("Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(num_iters):
            tt_matmul_out_tensor, tt_reduce_scatter_output_tensor = run_op(i)
            tt_reduce_scatter_output_list.append(tt_reduce_scatter_output_tensor)
            tt_matmul_output_list.append(tt_matmul_out_tensor)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        logger.info("Done capturing trace")

        # Warmup trace execution
        for _ in range(3):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

        # Timed trace execution
        num_perf_runs = 10
        start_time = time.perf_counter()
        for _ in range(num_perf_runs):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        end_time = time.perf_counter()

        total_time_us = (end_time - start_time) * 1e6
        avg_time_per_run_us = total_time_us / num_perf_runs
        avg_time_per_iter_us = avg_time_per_run_us / num_iters

        mode_str = "NON-FUSED (linear + reduce_scatter)" if use_non_fused else "FUSED (matmul_reduce_scatter_async)"
        logger.info(f"=== PERFORMANCE ({mode_str}) ===")
        logger.info(f"Total time for {num_perf_runs} runs: {total_time_us:.2f} us")
        logger.info(f"Avg time per run ({num_iters} iters): {avg_time_per_run_us:.2f} us")
        logger.info(f"Avg time per iteration: {avg_time_per_iter_us:.2f} us")
        logger.info(f"rs_input_shape={rs_input_shape}, mm_weights_shape={mm_weights_shape}")

        # Release trace
        ttnn.release_trace(mesh_device, trace_id)
    else:
        avg_time_per_iter_us = None
        for i in range(num_iters):
            tt_matmul_out_tensor, tt_reduce_scatter_output_tensor = run_op(i)
            tt_reduce_scatter_output_list.append(tt_reduce_scatter_output_tensor)
            tt_matmul_output_list.append(tt_matmul_out_tensor)

            logger.info("Waiting for op")
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info("Done op")
            logger.info(f"Done iteration {i}")

    # Validate
    # For Galaxy (8x4 mesh): data is replicated across 8 rows, scattered across 4 columns
    # We validate by checking one row's worth of data (4 devices in a column)
    total_devices = num_rows * num_cols
    for i in range(num_iters):
        tt_mm_out_tensor = tt_matmul_output_list[i]
        torch_mm_out_tensor = torch_matmul_output_list[i]

        tt_mm_out = ttnn.from_device(tt_mm_out_tensor)
        tt_mm_out = ttnn.to_torch(tt_mm_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
        # Output has all 32 devices concatenated. Extract just one row (4 devices) worth
        # Each row has num_cols (4) devices, so take the first row's data
        tt_mm_out_per_row = torch.chunk(tt_mm_out, num_rows, 3)[0]  # First row's 4 devices
        # Sum the 4 partial results from each device in the row (column scatter)
        tt_mm_out_summed = torch.sum(torch.stack(torch.chunk(tt_mm_out_per_row, num_cols, 3)), dim=0)
        eq, output = comp_pcc(tt_mm_out_summed, torch_mm_out_tensor)
        logger.info(f"Matmul PCC: {output}, iteration {i}")
        assert eq, f"{i} FAILED mm: {output}"

        tt_rs_out_tensor = tt_reduce_scatter_output_list[i]
        torch_rs_out_tensor = torch_reduce_scatter_output_list[i]
        torch_rs_out = torch.cat(torch_rs_out_tensor, 3)

        tt_rs_out = ttnn.from_device(tt_rs_out_tensor)
        tt_rs_out = ttnn.to_torch(tt_rs_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
        # Extract one row's worth of reduce_scatter output (4 devices)
        tt_rs_out_per_row = torch.chunk(tt_rs_out, num_rows, 3)[0]
        eq, output = comp_pcc(tt_rs_out_per_row, torch_rs_out)
        logger.info(f"ReduceScatter PCC: {output}, iteration {i}")
        assert eq, f"{i} FAILED rs: {output}"

    cleanup_sub_device(mesh_device)
    logger.info("Matmul+ReduceScatter Galaxy test completed successfully")

    return avg_time_per_iter_us


# =============================================================================
# Pytest Test Cases for Galaxy (8x4 mesh, 32 devices)
# =============================================================================


@pytest.mark.parametrize(
    "num_devices, num_links, mm_weights_shape, rs_input_shape, mm_shard_dim, rs_scatter_dim, layout, max_in0_block_w, matmul_weights_dtype, rs_input_dtype, use_bias",
    [
        # ==================== Llama 70B dimensions for Galaxy ====================
        # Galaxy: 8x4 mesh, scatter along columns (4 devices per column)
        # Llama 70B: dim=8192, hidden_dim=28672
        # Per-column: hidden_dim/4 = 7168 per device after scatter
        # Galaxy 70B decode (M=32) - scatter across 4 devices in column
        (4, 3, [1, 1, 8192, 7168], [1, 1, 32, 7168], 2, 3, ttnn.TILE_LAYOUT, 5, ttnn.bfloat16, ttnn.bfloat16, False),
        # Galaxy 70B prefill (M=128)
        (4, 3, [1, 1, 8192, 7168], [1, 1, 128, 7168], 2, 3, ttnn.TILE_LAYOUT, 5, ttnn.bfloat16, ttnn.bfloat16, False),
        # ==================== Scaled dimensions ====================
        # Scaled: dim=4096, hidden_dim=3584 per device (14336/4)
        (4, 3, [1, 1, 4096, 3584], [1, 1, 32, 3584], 2, 3, ttnn.TILE_LAYOUT, 5, ttnn.bfloat16, ttnn.bfloat16, False),
        (4, 3, [1, 1, 4096, 3584], [1, 1, 128, 3584], 2, 3, ttnn.TILE_LAYOUT, 5, ttnn.bfloat16, ttnn.bfloat16, False),
        (4, 3, [1, 1, 4096, 3584], [1, 1, 256, 3584], 2, 3, ttnn.TILE_LAYOUT, 5, ttnn.bfloat16, ttnn.bfloat16, False),
        # ==================== Smaller test dimensions ====================
        (4, 3, [1, 1, 8192, 2048], [1, 1, 32, 2048], 2, 3, ttnn.TILE_LAYOUT, 5, ttnn.bfloat16, ttnn.bfloat16, False),
        (4, 3, [1, 1, 8192, 2048], [1, 1, 128, 2048], 2, 3, ttnn.TILE_LAYOUT, 5, ttnn.bfloat16, ttnn.bfloat16, False),
        (4, 3, [1, 1, 8192, 2048], [1, 1, 512, 2048], 2, 3, ttnn.TILE_LAYOUT, 5, ttnn.bfloat16, ttnn.bfloat16, False),
    ],
    ids=[
        "llama70b_decode_32",
        "llama70b_prefill_128",
        "scaled_decode_32",
        "scaled_prefill_128",
        "scaled_prefill_256",
        "small_decode_32",
        "small_prefill_128",
        "small_prefill_512",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_mm, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "trace_region_size": 23887872,
            },
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_matmul_reduce_scatter_galaxy_check(
    mesh_device,
    num_devices,
    num_links,
    mm_weights_shape,
    rs_input_shape,
    mm_shard_dim,
    rs_scatter_dim,
    layout,
    use_bias,
    matmul_weights_dtype,
    max_in0_block_w,
    rs_input_dtype,
    mem_config_mm,
    mem_config_input,
    mem_config_rs,
    rs_topology,
):
    """
    Functional test for Matmul + ReduceScatter fused operation on Galaxy (32 devices, 8x4 mesh).

    Validates correctness of fused op by comparing against golden PyTorch outputs.
    Uses tensor dimensions from Llama 70B MLP ff1/ff3 path.
    """
    # Run fused op (correctness check only)
    run_matmul_reduce_scatter_galaxy_impl(
        mesh_device,
        num_devices,
        rs_input_shape,
        mm_shard_dim,
        rs_scatter_dim,
        num_links,
        mm_weights_shape,
        rs_input_dtype,
        layout,
        matmul_weights_dtype,
        max_in0_block_w,
        use_bias,
        mem_config_input,
        mem_config_rs,
        mem_config_mm,
        rs_topology=rs_topology,
        enable_trace=False,
        num_iters=1,
        use_non_fused=False,
    )


@pytest.mark.parametrize(
    "num_devices, num_links, mm_weights_shape, rs_input_shape, mm_shard_dim, rs_scatter_dim, layout, max_in0_block_w, matmul_weights_dtype, rs_input_dtype, use_bias",
    [
        # ==================== Llama 70B dimensions for Galaxy ====================
        # Galaxy: 8x4 mesh, scatter along columns (4 devices per column)
        # Galaxy 70B decode (M=32) - scatter across 4 devices
        (4, 3, [1, 1, 8192, 7168], [1, 1, 32, 7168], 2, 3, ttnn.TILE_LAYOUT, 5, ttnn.bfloat16, ttnn.bfloat16, False),
        # Galaxy 70B prefill (M=128)
        (4, 3, [1, 1, 8192, 7168], [1, 1, 128, 7168], 2, 3, ttnn.TILE_LAYOUT, 5, ttnn.bfloat16, ttnn.bfloat16, False),
        # ==================== Scaled dimensions ====================
        (4, 3, [1, 1, 4096, 3584], [1, 1, 32, 3584], 2, 3, ttnn.TILE_LAYOUT, 5, ttnn.bfloat16, ttnn.bfloat16, False),
        (4, 3, [1, 1, 4096, 3584], [1, 1, 128, 3584], 2, 3, ttnn.TILE_LAYOUT, 5, ttnn.bfloat16, ttnn.bfloat16, False),
        # ==================== Smaller test dimensions ====================
        (4, 3, [1, 1, 8192, 2048], [1, 1, 32, 2048], 2, 3, ttnn.TILE_LAYOUT, 5, ttnn.bfloat16, ttnn.bfloat16, False),
        (4, 3, [1, 1, 8192, 2048], [1, 1, 128, 2048], 2, 3, ttnn.TILE_LAYOUT, 5, ttnn.bfloat16, ttnn.bfloat16, False),
    ],
    ids=[
        "llama70b_decode_32",
        "llama70b_prefill_128",
        "scaled_decode_32",
        "scaled_prefill_128",
        "small_decode_32",
        "small_prefill_128",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_mm, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "trace_region_size": 23887872,
            },
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_matmul_reduce_scatter_galaxy_perf_comparison(
    mesh_device,
    num_devices,
    num_links,
    mm_weights_shape,
    rs_input_shape,
    mm_shard_dim,
    rs_scatter_dim,
    layout,
    use_bias,
    matmul_weights_dtype,
    max_in0_block_w,
    rs_input_dtype,
    mem_config_mm,
    mem_config_input,
    mem_config_rs,
    rs_topology,
):
    """
    Performance comparison test for Matmul + ReduceScatter on Galaxy (32 devices, 8x4 mesh).

    Runs both fused and non-fused operations and generates a comparison report showing
    the speedup/slowdown of the fused operation.
    """
    num_iters = 5

    # Run non-fused op first
    logger.info("Running NON-FUSED operation (linear + reduce_scatter)...")
    non_fused_time_us = run_matmul_reduce_scatter_galaxy_impl(
        mesh_device,
        num_devices,
        rs_input_shape,
        mm_shard_dim,
        rs_scatter_dim,
        num_links,
        mm_weights_shape,
        rs_input_dtype,
        layout,
        matmul_weights_dtype,
        max_in0_block_w,
        use_bias,
        mem_config_input,
        mem_config_rs,
        mem_config_mm,
        rs_topology=rs_topology,
        enable_trace=True,
        num_iters=num_iters,
        use_non_fused=True,
    )

    # Run fused op
    logger.info("Running FUSED operation (matmul_reduce_scatter_async)...")
    fused_time_us = run_matmul_reduce_scatter_galaxy_impl(
        mesh_device,
        num_devices,
        rs_input_shape,
        mm_shard_dim,
        rs_scatter_dim,
        num_links,
        mm_weights_shape,
        rs_input_dtype,
        layout,
        matmul_weights_dtype,
        max_in0_block_w,
        use_bias,
        mem_config_input,
        mem_config_rs,
        mem_config_mm,
        rs_topology=rs_topology,
        enable_trace=True,
        num_iters=num_iters,
        use_non_fused=False,
    )

    # Print comparison report
    test_name = f"Galaxy Matmul+ReduceScatter (M={rs_input_shape[2]}, N={rs_input_shape[3]})"
    print_perf_comparison_report(test_name, fused_time_us, non_fused_time_us, rs_input_shape, mm_weights_shape)

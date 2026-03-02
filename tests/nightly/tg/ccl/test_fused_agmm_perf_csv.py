# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance test for fused AllGather+MatMul vs baseline separate operations
Replicates the CSV format from profiler results for comparison with baseline numbers
"""

import torch
import pytest
import math
import time
import csv
from pathlib import Path
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from ttnn import ShardTensorToMesh, ConcatMeshToTensor


def measure_operation_time(func, num_iterations=10):
    """Measure average execution time of an operation"""
    # Warmup
    func()

    # Measure
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        func()
    end_time = time.perf_counter()

    return (end_time - start_time) * 1000000 / num_iterations  # Return in microseconds


def run_baseline_separate_ops(mesh_device, seq_len=8192):
    """Run baseline separate AllGather + MatMul operations"""

    # Setup
    num_devices = 4
    dim = 3
    hidden_per_device = 896  # 3584 / 4 = 896 per device (from your CSV)
    hidden_full = 3584
    output_dim = 2048

    # Create input tensor (sharded per device)
    input_tensor = torch.rand([1, 1, seq_len, hidden_per_device]).bfloat16()
    input_tt = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Create weight tensor
    weight_tensor = torch.rand([1, 1, hidden_full, output_dim]).bfloat16()
    weight_tt = ttnn.from_torch(
        weight_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Baseline operations
    def run_allgather():
        return ttnn.all_gather(input_tt, dim=dim, num_links=1, topology=ttnn.Topology.Ring)

    def run_separate_ops():
        # AllGather
        gathered = ttnn.all_gather(input_tt, dim=dim, num_links=1, topology=ttnn.Topology.Ring)
        # MatMul
        result = ttnn.linear(gathered, weight_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return result

    # Measure AllGather time
    ag_time = measure_operation_time(run_allgather, 10)

    # Measure total time
    total_time = measure_operation_time(run_separate_ops, 10)
    mm_time = total_time - ag_time

    return {
        "allgather_time_us": ag_time,
        "matmul_time_us": mm_time,
        "total_time_us": total_time,
        "allgather_cores": 40,  # From your CSV
        "matmul_cores": 56,  # From your CSV
    }


def run_fused_operation(mesh_device, core_grid, seq_len=8192):
    """Run fused AllGather+MatMul with specified grid"""

    # Setup
    num_devices = 4
    dim = 3
    hidden_per_device = 896  # 3584 / 4 = 896 per device
    hidden_full = 3584
    output_dim = 2048

    # Setup sub-devices for CCL
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

    # Create semaphores
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)]

    # Create input tensor (sharded per device)
    input_tensor = torch.rand([1, 1, seq_len, hidden_per_device]).bfloat16()
    input_tt = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
    )

    # Create weight tensor (sharded per device)
    weight_tensor = torch.rand([1, 1, hidden_per_device, output_dim]).bfloat16()
    weight_tt = ttnn.from_torch(
        weight_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
    )

    # Program config with custom grid
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=core_grid,
        in0_block_w=min(2, hidden_per_device // 32 // core_grid[0]),
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=max(1, math.ceil(seq_len / 32 / core_grid[1])),
        per_core_N=max(1, math.ceil(output_dim / 32 / core_grid[0])),
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

    # Fused operation
    def run_fused():
        return ttnn.experimental.all_gather_matmul_async(
            input_tt,
            weight_tt,
            dim=dim,
            multi_device_global_semaphore=ccl_semaphore_handles,
            all_gather_core_grid_offset=(0, core_grid[1]),
            num_links=1,
            memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            subdevice_id=worker_sub_device_id,
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

    # Compile first
    try:
        _, result = run_fused()
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

        # Measure execution time
        fused_time = measure_operation_time(lambda: run_fused()[1], 10)

        # Cleanup
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()

        return {
            "fused_time_us": fused_time,
            "fused_cores": core_grid[0] * core_grid[1],
            "grid": core_grid,
            "success": True,
            "error": None,
        }

    except Exception as e:
        # Cleanup on error
        try:
            mesh_device.reset_sub_device_stall_group()
            mesh_device.clear_loaded_sub_device_manager()
        except:
            pass

        return {
            "fused_time_us": None,
            "fused_cores": core_grid[0] * core_grid[1],
            "grid": core_grid,
            "success": False,
            "error": str(e),
        }


@pytest.mark.parametrize("mesh_device", [(4, 1)], indirect=True)
def test_fused_agmm_perf_comparison(mesh_device):
    """Compare fused vs baseline performance with CSV output"""

    seq_len = 8192

    # Test configurations
    grid_configs = [
        (4, 8),  # Current working: 32 cores
        (8, 8),  # Target: 64 cores
        (7, 8),  # Galaxy max width: 56 cores
        (8, 6),  # Original test: 48 cores
    ]

    logger.info(f"=== Performance Comparison: Seq Length {seq_len} ===")

    # Run baseline (separate AllGather + MatMul)
    logger.info("Running baseline separate operations...")
    try:
        baseline = run_baseline_separate_ops(mesh_device, seq_len)
        logger.info(
            f"✅ Baseline: AG={baseline['allgather_time_us']:.2f}μs ({baseline['allgather_cores']} cores), "
            f"MM={baseline['matmul_time_us']:.2f}μs ({baseline['matmul_cores']} cores), "
            f"Total={baseline['total_time_us']:.2f}μs"
        )
    except Exception as e:
        logger.error(f"❌ Baseline failed: {e}")
        baseline = None

    # Run fused operations with different grids
    results = []
    for grid in grid_configs:
        logger.info(f"Testing fused operation with grid {grid}...")
        result = run_fused_operation(mesh_device, grid, seq_len)
        results.append(result)

        if result["success"]:
            speedup = baseline["total_time_us"] / result["fused_time_us"] if baseline else "N/A"
            logger.info(
                f"✅ Grid {grid} ({result['fused_cores']} cores): {result['fused_time_us']:.2f}μs, "
                f"Speedup: {speedup:.2f}x"
                if speedup != "N/A"
                else f"Speedup: {speedup}"
            )
        else:
            logger.error(f"❌ Grid {grid}: {result['error']}")

    # Generate CSV report
    output_dir = Path("fused_agmm_perf_results")
    output_dir.mkdir(exist_ok=True)

    csv_file = output_dir / f"perf_comparison_{seq_len}.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["OPERATION", "GRID", "CORES", "TIME_us", "SPEEDUP_vs_BASELINE", "SUCCESS", "ERROR", "NOTES"])

        # Baseline
        if baseline:
            writer.writerow(
                [
                    "Baseline_AllGather",
                    "N/A",
                    baseline["allgather_cores"],
                    f"{baseline['allgather_time_us']:.2f}",
                    "1.00",
                    "True",
                    "",
                    f"Shape: 8192x896->8192x3584",
                ]
            )
            writer.writerow(
                [
                    "Baseline_MatMul",
                    "N/A",
                    baseline["matmul_cores"],
                    f"{baseline['matmul_time_us']:.2f}",
                    "1.00",
                    "True",
                    "",
                    f"Shape: 8192x3584->8192x2048",
                ]
            )
            writer.writerow(
                [
                    "Baseline_Total",
                    "N/A",
                    baseline["allgather_cores"] + baseline["matmul_cores"],
                    f"{baseline['total_time_us']:.2f}",
                    "1.00",
                    "True",
                    "",
                    "AllGather + MatMul separate",
                ]
            )

        # Fused results
        for result in results:
            if result["success"]:
                speedup = baseline["total_time_us"] / result["fused_time_us"] if baseline else "N/A"
                writer.writerow(
                    [
                        "Fused_AllGather_MatMul",
                        f"{result['grid'][0]}x{result['grid'][1]}",
                        result["fused_cores"],
                        f"{result['fused_time_us']:.2f}",
                        f"{speedup:.2f}" if speedup != "N/A" else "N/A",
                        "True",
                        "",
                        f"Single fused operation",
                    ]
                )
            else:
                writer.writerow(
                    [
                        "Fused_AllGather_MatMul",
                        f"{result['grid'][0]}x{result['grid'][1]}",
                        result["fused_cores"],
                        "N/A",
                        "N/A",
                        "False",
                        result["error"][:50],
                        "Failed to execute",
                    ]
                )

    logger.info(f"📊 Results saved to: {csv_file}")

    # Print summary
    logger.info(f"\n=== SUMMARY ===")
    if baseline:
        logger.info(f"Baseline (separate): {baseline['total_time_us']:.2f}μs")

    successful_results = [r for r in results if r["success"]]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x["fused_time_us"])
        speedup = baseline["total_time_us"] / best_result["fused_time_us"] if baseline else "N/A"
        logger.info(
            f"Best fused: Grid {best_result['grid']} - {best_result['fused_time_us']:.2f}μs "
            f"({speedup:.2f}x speedup)"
            if speedup != "N/A"
            else f"({speedup} speedup)"
        )

        # Check if 8x8 worked
        grid_8x8 = next((r for r in results if r["grid"] == (8, 8)), None)
        if grid_8x8 and grid_8x8["success"]:
            logger.info(f"🎉 8x8 grid (64 cores) WORKS! Time: {grid_8x8['fused_time_us']:.2f}μs")
        else:
            logger.warning("⚠️ 8x8 grid failed - investigate constraints")

    # Ensure at least one configuration worked
    assert len(successful_results) > 0, "No fused configurations worked!"

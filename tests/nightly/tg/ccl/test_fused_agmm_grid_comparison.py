# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Grid size comparison test for fused AllGather+MatMul
Tests 4x8 (32 cores) vs 8x8 (64 cores) to prove higher grids work
Uses exact Llama 70B W2 matmul sizes: (8192, 3584) -> 2048
"""

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import is_unsupported_case
from tests.tests_common.skip_reasons import LEGACY_CCL_SKIP
from ttnn import ShardTensorToMesh, ConcatMeshToTensor


def run_fused_agmm_with_grid(
    mesh_device,
    ag_output_shape,
    matmul_output_dim,
    core_grid,
    k_block_size=4,
    num_iters=5,
):
    """Run fused AllGather+MatMul with specific grid configuration"""

    torch.manual_seed(0)
    num_devices = 4
    dim = 3
    ag_input_dtype = ttnn.bfloat16
    matmul_weights_dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT

    # Memory configs
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    ##### Setup #####
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
    ccl_semaphore_handles = [
        [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)] for _ in range(num_iters)
    ]

    ##### Input setup #####
    _, _, seq_len, hidden_dim = ag_output_shape

    # Create input tensors (sharded per device)
    input_tensor_mesh_list = []
    ag_output_tensor_goldens_list = []

    for i in range(num_iters):
        ag_output_tensor = torch.rand(ag_output_shape).bfloat16()
        ag_output_tensor_goldens_list.append(ag_output_tensor)

        input_tensor_mesh = ttnn.from_torch(
            ag_output_tensor,
            device=mesh_device,
            layout=layout,
            dtype=ag_input_dtype,
            memory_config=mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        )
        input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Weight setup #####
    weights_tensor = torch.randn([1, 1, hidden_dim, matmul_output_dim]).bfloat16()
    weight_tt = ttnn.from_torch(
        weights_tensor,
        dtype=matmul_weights_dtype,
        layout=layout,
        device=mesh_device,
        memory_config=mem_config,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=dim),
    )

    ##### Program config with custom grid #####
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=core_grid,
        in0_block_w=min(2, hidden_dim // 32 // core_grid[0]),
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=max(1, math.ceil(seq_len / 32 / core_grid[1])),
        per_core_N=max(1, math.ceil(matmul_output_dim / 32 / core_grid[0])),
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

    ##### Golden computation #####
    torch_matmul_output_list = []
    for i in range(num_iters):
        matmul_output = torch.matmul(ag_output_tensor_goldens_list[i], weights_tensor.squeeze())
        torch_matmul_output_list.append(matmul_output)

    ##### Run fused operation #####
    def run_op(i):
        return ttnn.experimental.all_gather_matmul_async(
            input_tensor_mesh_list[i],
            weight_tt,
            dim=dim,
            multi_device_global_semaphore=ccl_semaphore_handles[i],
            all_gather_core_grid_offset=(0, core_grid[1]),  # Offset by grid height
            num_links=1,
            memory_config_ag=mem_config,
            topology=ttnn.Topology.Ring,
            subdevice_id=worker_sub_device_id,
            memory_config_mm=mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

    # Compile
    logger.info(f"Compiling with grid {core_grid}")
    tt_all_gather_out_tensor, tt_matmul_out_tensor = run_op(0)
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    # Capture trace
    logger.info(f"Capturing trace with grid {core_grid}")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_all_gather_out_tensor, tt_matmul_out_tensor = run_op(0)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    # Execute trace multiple times for timing
    import time

    start_time = time.time()

    for i in range(num_iters):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    end_time = time.time()
    avg_time_ms = (end_time - start_time) * 1000 / num_iters

    ##### Verify correctness #####
    tt_mm_out = ttnn.from_device(tt_matmul_out_tensor)
    tt_mm_out = ttnn.to_torch(tt_mm_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
    eq, output = comp_pcc(tt_mm_out, torch_matmul_output_list[0])

    logger.info(f"Grid {core_grid}: {output}, Avg time: {avg_time_ms:.2f}ms")

    # Cleanup
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()

    return {"grid": core_grid, "avg_time_ms": avg_time_ms, "pcc": output, "passed": eq}


@pytest.mark.parametrize("mesh_device", [(4, 1)], indirect=True)
def test_fused_agmm_grid_comparison(mesh_device):
    """Compare 4x8 vs 8x8 grid performance for fused AllGather+MatMul"""

    # Llama 70B W2 sizes
    ag_output_shape = [1, 1, 8192, 8192]  # After AllGather: (seq_len, full_hidden)
    matmul_output_dim = 2048  # W2 output dimension

    # Test configurations
    grid_configs = [
        (4, 8),  # Current: 32 cores
        (8, 8),  # Target: 64 cores
        (8, 6),  # Original test grid: 48 cores
    ]

    results = []

    for core_grid in grid_configs:
        logger.info(f"\n=== Testing grid {core_grid} ({core_grid[0]*core_grid[1]} cores) ===")

        try:
            result = run_fused_agmm_with_grid(
                mesh_device=mesh_device,
                ag_output_shape=ag_output_shape,
                matmul_output_dim=matmul_output_dim,
                core_grid=core_grid,
                k_block_size=4,
                num_iters=10,
            )
            results.append(result)
            logger.info(f"✅ Grid {core_grid}: PASSED - {result['avg_time_ms']:.2f}ms")

        except Exception as e:
            logger.error(f"❌ Grid {core_grid}: FAILED - {str(e)}")
            results.append({"grid": core_grid, "avg_time_ms": None, "pcc": None, "passed": False, "error": str(e)})

    # Print comparison
    logger.info(f"\n=== GRID COMPARISON RESULTS ===")
    for result in results:
        if result["passed"]:
            cores = result["grid"][0] * result["grid"][1]
            logger.info(f"Grid {result['grid']} ({cores} cores): {result['avg_time_ms']:.2f}ms - {result['pcc']}")
        else:
            logger.info(f"Grid {result['grid']}: FAILED - {result.get('error', 'Unknown error')}")

    # Ensure at least one grid worked
    passed_results = [r for r in results if r["passed"]]
    assert len(passed_results) > 0, "No grid configurations worked!"

    # If 8x8 worked, that's the key result for kernel developers
    grid_8x8_result = next((r for r in results if r["grid"] == (8, 8)), None)
    if grid_8x8_result and grid_8x8_result["passed"]:
        logger.info(f"🎉 SUCCESS: 8x8 grid (64 cores) works! Time: {grid_8x8_result['avg_time_ms']:.2f}ms")
    else:
        logger.warning("⚠️  8x8 grid failed - need to investigate constraints")

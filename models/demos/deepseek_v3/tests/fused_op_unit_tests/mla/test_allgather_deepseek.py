# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


def run_allgather_deepseek_with_trace(
    mesh_device,
    all_gather_topology,
    input_tensor_mesh,
    dim,
    num_links,
    output_mem_config,
    ccl_semaphore_handles,
    num_iter=100,
    warmup_iters=10,
    subdevice_id=None,
    profiler=BenchmarkProfiler(),
):
    """Run all-gather with trace mode for performance measurement."""
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.experimental.all_gather_async(
        input_tensor_mesh,
        dim,
        multi_device_global_semaphore=[ccl_semaphore_handles[0], ccl_semaphore_handles[1]],
        num_links=num_links,
        memory_config=output_mem_config,
        topology=all_gather_topology,
        subdevice_id=subdevice_id,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(warmup_iters):
        tt_out_tensor = ttnn.experimental.all_gather_async(
            input_tensor_mesh,
            dim,
            multi_device_global_semaphore=[ccl_semaphore_handles[2 * i], ccl_semaphore_handles[2 * i + 1]],
            num_links=num_links,
            memory_config=output_mem_config,
            topology=all_gather_topology,
            subdevice_id=subdevice_id,
        )
        tt_out_tensor.deallocate(True)
    ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iter} iterations")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.experimental.all_gather_async(
            input_tensor_mesh,
            dim,
            multi_device_global_semaphore=[ccl_semaphore_handles[2 * i], ccl_semaphore_handles[2 * i + 1]],
            num_links=num_links,
            memory_config=output_mem_config,
            topology=all_gather_topology,
            subdevice_id=subdevice_id,
        )
        if i != num_iter - 1:
            tt_out_tensor.deallocate(True)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    profiler.start("all-gather-trace-warmup")
    ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
    ttnn.release_trace(mesh_device, trace_id_warmup)
    profiler.end("all-gather-trace-warmup")

    # Execute main trace with signposts
    logger.info("Executing main trace")
    signpost("start")
    profiler.start("all-gather-trace")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    profiler.end("all-gather-trace")
    signpost("stop")

    return tt_out_tensor


@pytest.mark.parametrize(
    "op_name, num_devices, input_shape, output_shape, dim, layout",
    [
        (
            "wq_kv_a_ag_decode",
            8,  # 8 devices in a row for TG
            [1, 1, 32, 2112],  # Input per device: [1, 1, bsz, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
            [1, 8, 32, 2112],  # Output shape after all-gather: [1, num_devices, bsz, hidden]
            1,  # Gather along dim 1
            ttnn.TILE_LAYOUT,
        ),
        (
            "wo_ag_decode",
            8,  # 8 devices in a row for TG
            [1, 4, 128, 128],  # Input per device: [1, bsz_local, num_heads, v_head_dim]
            [1, 32, 128, 128],  # Output after all-gather: [1, bsz, num_heads, v_head_dim] (4*8=32)
            1,  # Gather along dim 1
            ttnn.TILE_LAYOUT,
        ),
    ],
    ids=["wq_kv_a_ag_decode", "wo_ag_decode"],
)
@pytest.mark.parametrize("num_links", [4])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("warmup_iters, num_iters", [(10, 100)])
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 901200,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_all_gather_trace_mode(
    mesh_device,
    op_name,
    num_devices,
    input_shape,
    output_shape,
    dim,
    num_links,
    trace_mode,
    input_dtype,
    layout,
    warmup_iters,
    num_iters,
    function_level_defaults,
):
    """
    Test all-gather operations from mla1d.py decode path with trace mode for performance measurement.

    This test captures traces of all-gather operations and executes them multiple times to measure performance.
    Uses signposts for Tracy profiling integration.

    Operations tested:
    1. wq_kv_a_ag_decode (line 1109): All-gather after fused wq_kv_a linear
       - Input: [1, 1, 32, 2112] per device
       - Output: [1, 8, 32, 2112] (gather along dim=1)

    2. wo_ag_decode (line 1325): All-gather for v_out before wo linear
       - Input: [1, 4, 128, 128] per device
       - Output: [1, 32, 128, 128] (gather along dim=1, 4*8=32)

    Configuration:
    - All-gather on dim=1 (across 8 devices in a row)
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Topology: Linear
    - Interleaved L1 memory
    """
    torch.manual_seed(0)

    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    # Set up sub-devices and semaphores for async operation
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

    # Create global semaphore handles (need 2 per iteration)
    ccl_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters * 2)
    ]

    # Memory config - interleaved L1 (matching mla1d.py line 1108)
    input_mem_config = ttnn.L1_MEMORY_CONFIG
    output_mem_config = ttnn.L1_MEMORY_CONFIG

    # Create golden output tensor and input
    logger.info(f"Running all-gather test: {op_name}")
    logger.info(f"Running on {mesh_device.get_num_devices()} devices")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Output shape: {output_shape}, dim: {dim}, num_devices: {num_devices}")

    output_tensor = torch.rand(output_shape, dtype=torch.bfloat16)

    # Create mesh tensor using MeshMapperConfig with PlacementShard on dim
    # This will automatically shard the tensor along the specified dimension
    input_tensor_mesh = ttnn.from_torch(
        output_tensor,
        device=mesh_device,
        layout=layout,
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig(
                [ttnn.PlacementReplicate(), ttnn.PlacementShard(dim)], ttnn.MeshShape(1, num_devices)
            ),
        ),
    )

    all_gather_topology = ttnn.Topology.Ring
    profiler = BenchmarkProfiler()

    try:
        if trace_mode:
            # Run all-gather with trace
            tt_out_tensor = run_allgather_deepseek_with_trace(
                mesh_device,
                all_gather_topology,
                input_tensor_mesh,
                dim,
                num_links,
                output_mem_config,
                ccl_semaphore_handles=ccl_semaphore_handles,
                num_iter=num_iters,
                warmup_iters=warmup_iters,
                subdevice_id=worker_sub_device_id,
                profiler=profiler,
            )
            tt_out_tensor_list = [tt_out_tensor]
        else:
            # Run without trace (for debugging)
            tt_out_tensor = ttnn.experimental.all_gather_async(
                input_tensor_mesh,
                dim,
                multi_device_global_semaphore=[ccl_semaphore_handles[0], ccl_semaphore_handles[1]],
                num_links=num_links,
                memory_config=output_mem_config,
                topology=all_gather_topology,
                subdevice_id=worker_sub_device_id,
            )
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            tt_out_tensor_list = [tt_out_tensor]

        # Verify correctness
        logger.info("Verifying correctness")
        passed = True
        for tensor_index in range(len(tt_out_tensor_list)):
            tt_out_tensor = tt_out_tensor_list[tensor_index]
            for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
                tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
                logger.info(f"Checking for device {t.device().id()}")

                eq, output = comp_equal(tt_output_tensor, output_tensor)
                if not eq:
                    logger.error(f"output mismatch for tensor {i}")
                    passed = False

        assert (
            mesh_device.num_program_cache_entries() == 1 or mesh_device.num_program_cache_entries() == num_iters
        ), f"Device has {mesh_device.num_program_cache_entries()} program cache entries"

        assert passed, f"Output verification failed"

        logger.info("✓ Trace mode all-gather test passed with correct output")

    finally:
        # Clean up sub-device configuration
        mesh_device.reset_sub_device_stall_group()

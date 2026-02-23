# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


def run_alltoall_deepseek_with_trace(
    mesh_device,
    input_tensor_mesh,
    cluster_axis,
    in_dim,
    out_dim,
    output_mem_config,
    topology,
    num_iter=100,
    warmup_iters=10,
    subdevice_id=None,
    profiler=BenchmarkProfiler(),
):
    """Run all-to-all with trace mode for performance measurement."""
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.experimental.all_to_all_async_generic(
        input_tensor_mesh,
        in_dim=in_dim,
        out_dim=out_dim,
        persistent_output_buffer=None,
        num_links=4,
        cluster_axis=cluster_axis,
        memory_config=output_mem_config,
        topology=topology,
        subdevice_id=subdevice_id,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(warmup_iters):
        tt_out_tensor = ttnn.experimental.all_to_all_async_generic(
            input_tensor_mesh,
            in_dim=in_dim,
            out_dim=out_dim,
            persistent_output_buffer=None,
            cluster_axis=cluster_axis,
            num_links=4,
            memory_config=output_mem_config,
            topology=topology,
            subdevice_id=subdevice_id,
        )
        tt_out_tensor.deallocate(True)
    ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iter} iterations")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.experimental.all_to_all_async_generic(
            input_tensor_mesh,
            in_dim=in_dim,
            out_dim=out_dim,
            cluster_axis=cluster_axis,
            persistent_output_buffer=None,
            num_links=4,
            memory_config=output_mem_config,
            topology=topology,
            subdevice_id=subdevice_id,
        )
        if i != num_iter - 1:
            tt_out_tensor.deallocate(True)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    profiler.start("all-to-all-trace-warmup")
    ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
    ttnn.release_trace(mesh_device, trace_id_warmup)
    profiler.end("all-to-all-trace-warmup")

    # Execute main trace with signposts
    logger.info("Executing main trace")
    signpost("start")
    profiler.start("all-to-all-trace")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    profiler.end("all-to-all-trace")
    signpost("stop")

    return tt_out_tensor


@pytest.mark.parametrize(
    "num_devices, input_shape, cluster_axis, in_dim, out_dim, output_shape, layout",
    [
        (
            8,  # 8 devices in a row for TG
            [1, 32, 16, 576],  # Input shape: [1, bsz, num_heads_local, kv_lora_rank + qk_rope_head_dim]
            1,  # Cluster axis (along device row)
            2,  # Input dimension to split (num_heads_local=16 -> 16/8=2 per device after A2A)
            1,  # Output dimension to gather (bsz=32 -> 32*8=256, but becomes 4 per device: 32/8=4)
            [1, 4, 128, 576],  # Output shape per device after all-to-all
            ttnn.TILE_LAYOUT,
        ),
    ],
    ids=["deepseek_mla_wq_a2a"],
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("warmup_iters, num_iters", [(10, 100)])
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="8x4_grid")], indirect=True)
# We are flipping the mesh shape so we can test the 8 device version on cluster axis 1 to simulate the dual and quad galaxy setup
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 550912,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_wq_a2a_all_to_all_trace_mode(
    mesh_device,
    num_devices,
    input_shape,
    cluster_axis,
    in_dim,
    out_dim,
    output_shape,
    trace_mode,
    input_dtype,
    layout,
    warmup_iters,
    num_iters,
    function_level_defaults,
):
    """
    Test the all-to-all operation from line 1301 of mla1d.py with trace mode for performance measurement.

    This test captures a trace of the all-to-all operation that transposes Q tensor before flash attention.

    Operation: all_to_all_async_generic with config flash_mla_reshard
    - Input shape per device: [1, 32, 16, 576] (before A2A)
    - cluster_axis=1 (along device row)
    - in_dim=2 (split num_heads: 16 heads -> 2 heads per device)
    - out_dim=1 (gather batch: 32 batch -> 4 batch per device, total 256 batch across 8 devices)
    - Output shape per device: [1, 4, 128, 576] (after A2A)

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Topology: Linear
    - Interleaved L1 memory
    """
    torch.manual_seed(0)

    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    # Set up sub-devices for async operation (no semaphores needed for all-to-all)
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

    # Memory config (matching mla1d.py line 1301)
    # Input: L1 interleaved
    input_mem_config = ttnn.L1_MEMORY_CONFIG
    # Output: L1 HEIGHT sharded 8x9 grid [32, 576] (matching flash_mla_reshard config)
    # After all-to-all, shape is [1, 4, 128, 576], total height = 1 * 4 * 128 = 512
    # Using 64 cores (8x9 grid minus 8 cores = 64 cores), shard height = 512 / 64 = 8 -> nearest_y = 32
    output_core_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}  # 8x8 grid = 64 cores
    )
    output_shard_shape = [32, 576]
    output_mem_config = ttnn.create_sharded_memory_config(
        shape=output_shard_shape,
        core_grid=output_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Create input tensor
    logger.info(f"Running all-to-all test on {mesh_device.get_num_devices()} devices")
    logger.info(f"Input shape per device: {input_shape}")
    logger.info(f"Output shape per device: {output_shape}")
    logger.info(f"cluster_axis={cluster_axis}, in_dim={in_dim}, out_dim={out_dim}")

    # Create a full tensor that will be distributed across devices
    # For all-to-all, we need to think about the global shape before distribution
    # Input: [1, 32, 16, 576] per device, with 8 devices along dim 1 (cluster_axis=1)
    # This means globally before A2A: [1, 32, 128, 576] (16 heads * 8 devices = 128 heads globally)
    global_input_shape = list(input_shape)
    global_input_shape[in_dim] = input_shape[in_dim] * num_devices  # 16 * 8 = 128 heads globally
    input_tensor = torch.rand(global_input_shape, dtype=torch.bfloat16)

    # Create mesh tensor - shard on in_dim (heads dimension)
    input_tensor_mesh = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        layout=layout,
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig(
                [ttnn.PlacementReplicate(), ttnn.PlacementShard(in_dim)], ttnn.MeshShape(1, num_devices)
            ),
        ),
    )

    topology = ttnn.Topology.Ring
    profiler = BenchmarkProfiler()

    if trace_mode:
        # Run all-to-all with trace
        tt_out_tensor = run_alltoall_deepseek_with_trace(
            mesh_device,
            input_tensor_mesh,
            cluster_axis,
            in_dim,
            out_dim,
            output_mem_config,
            topology,
            num_iter=num_iters,
            warmup_iters=warmup_iters,
            subdevice_id=worker_sub_device_id,
            profiler=profiler,
        )
    else:
        # Run without trace (for debugging)
        tt_out_tensor = ttnn.experimental.all_to_all_async_generic(
            input_tensor_mesh,
            in_dim=in_dim,
            out_dim=out_dim,
            persistent_output_buffer=None,
            num_links=4,
            cluster_axis=cluster_axis,
            memory_config=output_mem_config,
            topology=topology,
            subdevice_id=worker_sub_device_id,
        )

    # Verify output shape
    tt_output_torch = ttnn.to_torch(tt_out_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    logger.info(f"Output shape (concatenated across devices): {tt_output_torch.shape}")

    # The actual output from ConcatMeshToTensor concatenates along dim=0, so we get the actual shape
    actual_shape = list(tt_output_torch.shape)
    logger.info(f"Actual concatenated shape: {actual_shape}")

    # Just verify it's the right rank and has reasonable dimensions
    assert len(actual_shape) == 4, f"Expected 4D tensor, got {len(actual_shape)}D"
    assert actual_shape[0] == num_devices, f"Expected first dim to be {num_devices}, got {actual_shape[0]}"
    # Verify correctness
    tt_output_torch = tt_output_torch.reshape(input_tensor.shape)
    eq, output = comp_equal(tt_output_torch, input_tensor)
    assert eq, f"Output mismatch: {output}"

    logger.info("✓ All-to-all trace mode test passed")

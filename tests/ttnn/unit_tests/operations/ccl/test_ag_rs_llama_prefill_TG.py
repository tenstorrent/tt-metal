# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from ttnn import ShardTensor2dMesh, ConcatMesh2dToTensor
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_all_gather_TG_post_commit import (
    report_mismatches,
    print_tile_corners_of_tensor,
)
from models.perf.benchmarking_utils import BenchmarkProfiler
from tracy import signpost

NUM_BUFFERS = 16


def run_ag_with_trace(
    mesh_device,
    all_gather_topology,
    input_tensor,
    dim,
    persistent_output_tensor,
    num_links,
    cluster_axis,
    output_mem_config,
    ccl_semaphore_handles,
    worker_sub_device_id,
    n_worker=None,
    n_buffer=None,
    num_iter=20,
    warmup_iters=10,
    profiler=BenchmarkProfiler(),
):
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.experimental.all_gather_async(
        input_tensor,
        dim=dim,
        cluster_axis=cluster_axis,
        multi_device_global_semaphore=ccl_semaphore_handles[NUM_BUFFERS - 1],
        persistent_output_buffer=persistent_output_tensor,
        num_links=num_links,
        memory_config=output_mem_config,
        topology=all_gather_topology,
        subdevice_id=worker_sub_device_id,
    )

    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")

    def capture_trace(n_iters):
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(n_iters):
            tt_out_tensor = ttnn.experimental.all_gather_async(
                input_tensor,
                dim=dim,
                cluster_axis=cluster_axis,
                multi_device_global_semaphore=ccl_semaphore_handles[i % NUM_BUFFERS],
                persistent_output_buffer=persistent_output_tensor,
                num_links=num_links,
                memory_config=output_mem_config,
                topology=all_gather_topology,
                subdevice_id=worker_sub_device_id,
            )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        return trace_id

    if warmup_iters > 0:
        trace_id_warmup = capture_trace(warmup_iters)
    trace_id = capture_trace(num_iter)

    # Run the op
    logger.info("Starting Trace perf test...")
    profiler.start("all-gather-async-trace-warmup")
    if warmup_iters > 0:
        ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
        ttnn.release_trace(mesh_device, trace_id_warmup)
        ttnn.synchronize_device(mesh_device)
    profiler.end("all-gather-async-trace-warmup")

    profiler.start("all-gather-async-trace")
    signpost("start")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    profiler.end("all-gather-async-trace")
    time_taken = profiler.get_duration("all-gather-async-trace") - profiler.get_duration(
        "all-gather-async-trace-warmup"
    )
    effective_iter = num_iter - warmup_iters
    logger.info(f"Time taken e2e: {time_taken} s")
    logger.info(f"Time per iter e2e: {time_taken / effective_iter} s")
    logger.info(f"Time per iter e2e: {time_taken / effective_iter * 1e6} us")

    return tt_out_tensor


def run_all_gather_on_TG(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    tensor_memory_layout,
    dim,
    num_links,
    input_dtype,
    layout,
    buffer_type: ttnn.BufferType,
    function_level_defaults,
    input_shard_spec: ttnn.ShardSpec = None,
    output_shard_spec: ttnn.ShardSpec = None,
    num_all_gather_instances: int = 1,
    num_iters: int = 1,
    warmup_iters: int = 0,
    cluster_axis: int = 0,
    tile=(32, 32),
    trace_mode=True,
    debug=False,
    profiler=BenchmarkProfiler(),
):
    input_shape_per_chip = list(per_chip_output_shape)
    input_shape_per_chip[dim] //= num_devices
    tensor_height_per_all_gather = per_chip_output_shape[-2]

    full_mesh_input_shape = list(per_chip_output_shape)
    ## The `all_gather_instances_concat_dim` is the dimension we will split the cluster spanning tensor along in order to split it
    ## off into per-all-gather tensors
    all_gather_instances_concat_dim = 1 if dim == 0 else 0
    full_mesh_input_shape[all_gather_instances_concat_dim] *= num_all_gather_instances
    logger.info(
        f"per_chip_output_shape: {full_mesh_input_shape}, dim: {dim}, all_gather_instances_concat_dim: {all_gather_instances_concat_dim}, num_devices: {num_devices}"
    )

    all_gather_instances_goldens = []
    full_input_tensor_unfractured = torch.rand(full_mesh_input_shape, dtype=torch.bfloat16)

    input_mem_config = ttnn.MemoryConfig(tensor_memory_layout, buffer_type=buffer_type, shard_spec=input_shard_spec)
    shard_dims = (dim, all_gather_instances_concat_dim) if cluster_axis == 0 else (all_gather_instances_concat_dim, dim)
    concat_dims = shard_dims

    mesh_shape = (
        (num_devices, num_all_gather_instances) if cluster_axis == 0 else (num_all_gather_instances, num_devices)
    )

    if input_shard_spec is not None and output_shard_spec is None:
        output_shard_shape = list(input_shard_spec.shape)
        if dim == len(per_chip_output_shape) - 1:
            output_shard_shape[1] *= num_devices
        else:
            output_shard_shape[0] *= num_devices
        output_shard_spec = ttnn.ShardSpec(
            input_shard_spec.grid,
            output_shard_shape,
            input_shard_spec.orientation,
        )
    output_mem_config = ttnn.MemoryConfig(tensor_memory_layout, buffer_type=buffer_type, shard_spec=output_shard_spec)
    ttnn_tensor = ttnn.from_torch(
        full_input_tensor_unfractured,
        tile=ttnn.Tile(tile),
        dtype=input_dtype,
        device=mesh_device,
        layout=layout,
        memory_config=input_mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=shard_dims),
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
    ttnn_tensor = ttnn.to_memory_config(ttnn_tensor, input_mem_config)
    # TODO: Take as an arg
    linear = False
    all_gather_topology = ttnn.Topology.Ring

    ttnn_persistent_output_tensor = ttnn.from_torch(
        torch.zeros(per_chip_output_shape),
        tile=ttnn.Tile(tile),
        dtype=input_dtype,
        device=mesh_device,
        layout=layout,
        memory_config=output_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    sub_device_stall_group = []
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)
    # create global semaphore handles
    ccl_semaphore_handles = [
        [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for nsem in range(2)]
        for _ in range(NUM_BUFFERS)
    ]
    try:
        if trace_mode:
            ttnn_tensor_out = run_ag_with_trace(
                input_tensor=ttnn_tensor,
                dim=dim,
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                persistent_output_tensor=ttnn_persistent_output_tensor,
                num_links=num_links,
                output_mem_config=output_mem_config,
                ccl_semaphore_handles=ccl_semaphore_handles,
                worker_sub_device_id=worker_sub_device_id,
                all_gather_topology=all_gather_topology,
                num_iter=num_iters,
                warmup_iters=warmup_iters,
                profiler=profiler,
            )

        else:
            signpost("start")
            for i in range(num_iters):
                logger.info("Running all-gather async")
                ttnn_tensor_out = ttnn.experimental.all_gather_async(
                    ttnn_tensor,
                    dim=dim,
                    cluster_axis=cluster_axis,
                    # mesh_device=mesh_device,
                    persistent_output_buffer=ttnn_persistent_output_tensor,
                    multi_device_global_semaphore=ccl_semaphore_handles[i % NUM_BUFFERS],
                    num_links=num_links,
                    memory_config=output_mem_config,
                    topology=all_gather_topology,
                    subdevice_id=worker_sub_device_id,
                )
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            signpost("stop")
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise e
    finally:
        mesh_device.reset_sub_device_stall_group()

    # ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor_out)
    tt_output_tensor = ttnn.to_torch(
        ttnn_tensor_out, mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=concat_dims)
    )
    output_tensors_list = torch.chunk(tt_output_tensor, num_all_gather_instances, dim=all_gather_instances_concat_dim)
    output_golden = torch.zeros(tt_output_tensor.shape)

    # Check the tensor addresses
    persistent_output_tensors = ttnn.get_device_tensors(ttnn_persistent_output_tensor)
    output_tensors = ttnn.get_device_tensors(ttnn_tensor_out)

    for persistent_tensor, output_tensor in zip(persistent_output_tensors, output_tensors):
        assert (
            persistent_tensor.buffer_address() == output_tensor.buffer_address()
        ), "Persistent tensor address mismatch"

    repeat_factor = [1] * len(output_golden.shape)
    repeat_factor[dim] = num_devices
    output_golden[:, :, :, :] = full_input_tensor_unfractured.repeat(repeat_factor)

    eq = True
    if input_dtype == ttnn.bfloat16:
        eq, output = comp_equal(tt_output_tensor, output_golden)
        if not eq and debug is True:
            logger.error(f"found mismatches")
            report_mismatches(tt_output_tensor, output_golden, 100)
            print_tile_corners_of_tensor(tt_output_tensor)
    else:
        eq, output = comp_pcc(tt_output_tensor, output_golden)
    if not eq:
        logger.error(f"output mismatch for tensor: {output}")

    assert eq, f"FAILED: {output}"


def run_rs_with_trace(
    mesh_device,
    all_gather_topology,
    input_tensor,
    dim,
    num_links,
    math_op,
    cluster_axis,
    output_mem_config,
    n_worker=None,
    n_buffer=None,
    num_iter=20,
    warmup_iters=10,
    profiler=BenchmarkProfiler(),
    worker_sub_device_id=None,
    from_remote_semaphore_handles=None,
    persistent_buffers=None,
):
    # Compile Run
    logger.info("Compiling model")
    ttnn_tensor_out = ttnn.experimental.reduce_scatter_minimal_async(
        input_tensor,
        dim=dim,
        persistent_output_buffers=persistent_buffers,
        multi_device_global_semaphore=from_remote_semaphore_handles,
        num_links=num_links,
        memory_config=output_mem_config,
        topology=ttnn.Topology.Ring,
        subdevice_id=worker_sub_device_id,
        cluster_axis=cluster_axis,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")

    def capture_trace(n_iters):
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(n_iters):
            ttnn_tensor_out = ttnn.experimental.reduce_scatter_minimal_async(
                input_tensor,
                dim=dim,
                persistent_output_buffers=persistent_buffers,
                multi_device_global_semaphore=from_remote_semaphore_handles,
                num_links=num_links,
                memory_config=output_mem_config,
                topology=ttnn.Topology.Ring,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
            )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        return trace_id

    if warmup_iters > 0:
        trace_id_warmup = capture_trace(warmup_iters)
    trace_id = capture_trace(num_iter)

    # Run the op
    logger.info("Starting Trace perf test...")
    profiler.start("reduce-scatter-minimal-trace-warmup")
    if warmup_iters > 0:
        ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
        ttnn.release_trace(mesh_device, trace_id_warmup)
        ttnn.synchronize_device(mesh_device)
    profiler.end("reduce-scatter-minimal-trace-warmup")

    profiler.start("reduce-scatter-minimal-trace")
    signpost("start")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    profiler.end("reduce-scatter-minimal-trace")
    time_taken = profiler.get_duration("reduce-scatter-minimal-trace") - profiler.get_duration(
        "reduce-scatter-minimal-trace-warmup"
    )
    effective_iter = num_iter - warmup_iters
    logger.info(f"Time taken e2e: {time_taken} s")
    logger.info(f"Time per iter e2e: {time_taken / effective_iter} s")
    logger.info(f"Time per iter e2e: {time_taken / effective_iter * 1e6} us")

    return ttnn_tensor_out


def run_reduce_scatter_on_TG(
    mesh_device,
    num_devices,
    per_chip_input_shape,
    tensor_memory_layout,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type: ttnn.BufferType,
    function_level_defaults,
    input_shard_spec: ttnn.ShardSpec = None,
    num_reduce_scatter_instances: int = 1,
    num_iters: int = 1,
    warmup_iters: int = 0,
    cluster_axis: int = 0,
    trace_mode=False,
):
    per_reduce_scatter_output_shape = list(per_chip_input_shape)
    per_reduce_scatter_output_shape[dim] *= num_devices
    full_mesh_input_shape = list(per_reduce_scatter_output_shape)
    ## The `reduce_scatter_instances_concat_dim` is the dimension we will split the cluster spanning tensor along in order to split it
    ## off into per-all-gather tensors
    reduce_scatter_instances_concat_dim = 1 if dim == 0 else 0
    full_mesh_input_shape[reduce_scatter_instances_concat_dim] *= num_reduce_scatter_instances
    logger.info(
        f"full_mesh_input_shape: {full_mesh_input_shape}, dim: {dim}, reduce_scatter_instances_concat_dim: {reduce_scatter_instances_concat_dim}, num_devices: {num_devices}"
    )

    sub_device_stall_group = []
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
    # create global semaphore handles
    from_remote_semaphore_handles = []
    for _ in range(num_links):
        from_remote_semaphore_handles.append(ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0))
        from_remote_semaphore_handles.append(ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0))
        from_remote_semaphore_handles.append(ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0))

    ##
    ## Compute golden
    ##
    per_chip_output_shape = list(per_chip_input_shape)
    per_chip_output_shape[dim] //= num_devices
    per_reduce_scatter_inputs = []
    per_reduce_scatter_goldens = []
    for i in range(num_reduce_scatter_instances):
        per_chip_inputs = [torch.rand(per_chip_input_shape).bfloat16() for _ in range(num_devices)]
        per_reduce_scatter_inputs.append(per_chip_inputs)

        golden_canonical_out_tensor = torch.zeros(per_chip_input_shape).bfloat16()
        for t in per_chip_inputs:
            golden_canonical_out_tensor = torch.add(golden_canonical_out_tensor, t).bfloat16()
        per_reduce_scatter_goldens.append(golden_canonical_out_tensor)

    per_reduce_scatter_concatenated_inputs = [
        torch.cat(per_reduce_scatter_inputs[i], dim=dim) for i in range(num_reduce_scatter_instances)
    ]

    full_input_tensor_unfractured = torch.cat(
        per_reduce_scatter_concatenated_inputs, dim=reduce_scatter_instances_concat_dim
    )

    input_mem_config = ttnn.MemoryConfig(tensor_memory_layout, buffer_type=buffer_type, shard_spec=input_shard_spec)
    shard_dims = (
        (dim, reduce_scatter_instances_concat_dim) if cluster_axis == 0 else (reduce_scatter_instances_concat_dim, dim)
    )
    concat_dims = shard_dims

    mesh_shape = (
        (num_devices, num_reduce_scatter_instances)
        if cluster_axis == 0
        else (num_reduce_scatter_instances, num_devices)
    )

    output_shard_spec = None
    if input_shard_spec is not None:
        output_shard_shape = list(input_shard_spec.shape)
        if dim == 3:
            output_shard_shape[1] //= num_devices
        else:
            output_shard_shape[0] //= num_devices
        output_shard_spec = ttnn.ShardSpec(
            input_shard_spec.grid,
            output_shard_shape,
            input_shard_spec.orientation,
        )
    output_mem_config = ttnn.MemoryConfig(tensor_memory_layout, buffer_type=buffer_type, shard_spec=output_shard_spec)
    ttnn_tensor = ttnn.from_torch(
        full_input_tensor_unfractured,
        dtype=input_dtype,
        device=mesh_device,
        layout=layout,
        memory_config=input_mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=shard_dims),
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)

    persistent_buffers = None
    tile = (32, 32)
    persistent_buffers = [
        ttnn.from_torch(
            torch.zeros(per_chip_input_shape),
            tile=ttnn.Tile(tile),
            dtype=input_dtype,
            device=mesh_device,
            layout=layout,
            memory_config=output_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        ttnn.from_torch(
            torch.zeros(per_chip_output_shape),
            tile=ttnn.Tile(tile),
            dtype=input_dtype,
            device=mesh_device,
            layout=layout,
            memory_config=output_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
    ]

    if trace_mode:
        ttnn_tensor_out = run_rs_with_trace(
            input_tensor=ttnn_tensor,
            dim=dim,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            math_op=math_op,
            output_mem_config=output_mem_config,
            all_gather_topology=ttnn.Topology.Linear,
            num_links=num_links,
            num_iter=num_iters,
            warmup_iters=warmup_iters,
            worker_sub_device_id=worker_sub_device_id,
            from_remote_semaphore_handles=from_remote_semaphore_handles,
            persistent_buffers=persistent_buffers,
        )
    else:
        logger.info(f"Running {num_iters} iterations of reduce scatter")
        for _ in range(num_iters):
            ttnn_tensor_out = ttnn.experimental.reduce_scatter_minimal_async(
                ttnn_tensor,
                dim=dim,
                persistent_output_buffers=persistent_buffers,
                multi_device_global_semaphore=from_remote_semaphore_handles,
                num_links=num_links,
                memory_config=output_mem_config,
                topology=ttnn.Topology.Ring,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
            )

            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    mesh_device.reset_sub_device_stall_group()

    # ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor_out)
    tt_output_tensor = ttnn.to_torch(
        ttnn_tensor_out, mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=concat_dims)
    )
    output_tensors_list = torch.chunk(
        tt_output_tensor, num_reduce_scatter_instances, dim=reduce_scatter_instances_concat_dim
    )

    passed = True
    for i in range(num_reduce_scatter_instances):
        # The result of all-chips in the reduce scatter line having their outputs concatenated
        reduce_scatter_outputs_concatenated = output_tensors_list[i]
        per_chip_outputs = torch.chunk(reduce_scatter_outputs_concatenated, num_devices, dim=dim)
        per_chip_goldens = torch.chunk(per_reduce_scatter_goldens[i], num_devices, dim=dim)

        assert len(per_chip_outputs) == len(per_chip_goldens)
        # compare the output and golden (zip)
        for d, (output, golden) in enumerate(zip(per_chip_outputs, per_chip_goldens)):
            eq, output = comp_pcc(output, golden)

            if not eq:
                passed = False
                logger.error(f"output mismatch for tensor on reduce_scatter {i}, device {d}: {output}")

    assert passed, f"FAILED: {output}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("seq_len", [128, 4096, 8192], ids=["128_seq", "4k_seq", "8k_seq"])
@pytest.mark.parametrize(
    "num_devices, num_links, width, dim, layout, input_dtype, cluster_axis, replication_factor",
    [
        (8, 4, 256, 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b, 0, 4),
        (4, 4, 320, 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b, 1, 8),
        (4, 4, 896, 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b, 1, 8),
        (4, 4, 32, 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 8),
    ],
    ids=["sh-256w", "sh-320w", "sh-896w", "sh-32w"],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
    ],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 5554176}], indirect=True
)
@pytest.mark.parametrize(
    "trace_mode, warmup_iters, num_iters", [(False, 0, 1), (True, 15, 100)], ids=["no-trace", "yes-trace"]
)
def test_all_gather_TG(
    mesh_device,
    num_devices,
    num_links,
    seq_len,
    width,
    dim,
    input_dtype,
    layout,
    buffer_type,
    cluster_axis,
    function_level_defaults,
    replication_factor,
    num_iters,
    warmup_iters,
    trace_mode,
):
    per_chip_output_shape = [1, 1, seq_len, width * num_devices]

    run_all_gather_on_TG(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,
        dim,
        num_links,
        input_dtype,
        layout,
        buffer_type,
        function_level_defaults,
        warmup_iters=warmup_iters,
        num_iters=num_iters,
        num_all_gather_instances=replication_factor,
        cluster_axis=cluster_axis,
        trace_mode=trace_mode,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("seq_len", [128, 4096, 8192], ids=["128_seq", "4k_seq", "8k_seq"])
@pytest.mark.parametrize(
    "num_devices, num_links, width, dim, layout, input_dtype, cluster_axis, replication_factor",
    [
        (8, 4, 2048, 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b, 0, 4),
        (4, 4, 1280, 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b, 1, 8),
        (4, 4, 3584, 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b, 1, 8),
    ],
    ids=["sh-2048w", "sh-1280w", "sh-3584w"],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
    ],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 5554176}], indirect=True
)
@pytest.mark.parametrize(
    "trace_mode, warmup_iters, num_iters", [(False, 0, 1), (True, 15, 100)], ids=["no-trace", "yes-trace"]
)
def test_reduce_scatter_TG(
    mesh_device,
    num_devices,
    num_links,
    seq_len,
    width,
    dim,
    input_dtype,
    layout,
    buffer_type,
    cluster_axis,
    function_level_defaults,
    replication_factor,
    num_iters,
    warmup_iters,
    trace_mode,
):
    rs_input_shape = [1, 1, seq_len, width]

    run_reduce_scatter_on_TG(
        mesh_device,
        num_devices,
        rs_input_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,
        dim,
        num_links,
        ttnn.ReduceType.Sum,
        input_dtype,
        layout,
        buffer_type,
        function_level_defaults,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        num_reduce_scatter_instances=replication_factor,
        cluster_axis=cluster_axis,
        trace_mode=trace_mode,
    )

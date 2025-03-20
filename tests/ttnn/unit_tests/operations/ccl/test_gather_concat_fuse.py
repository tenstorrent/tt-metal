# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.utility_functions import skip_for_grayskull
from tracy import signpost
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc

from tests.ttnn.unit_tests.operations.ccl.test_all_gather_TG_post_commit import (
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows,
)
from tests.ttnn.unit_tests.operations.ccl.test_new_all_reduce import (
    run_all_reduce_impl,
)
from ttnn import ShardTensor2dMesh, ConcatMesh2dToTensor
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
    create_global_semaphore_with_same_address,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler


NUM_ITERATIONS = 10
NUM_BUFFERS = 8


def run_with_trace(
    mesh_device,
    all_gather_topology,
    input_tensor,
    dim,
    cluster_axis,
    num_links,
    output_mem_config,
    ccl_semaphore_handles,
    worker_sub_device_id,
    enable_persistent_fabric,
    n_worker=None,
    n_buffer=None,
    num_iter=20,
    warmup_iters=0,
    use_all_gather_async=False,
    profiler=BenchmarkProfiler(),
):
    # Compile Run
    logger.info("Compiling model")
    print("input tensor shape: \n", input_tensor.shape)

    tt_out_tensor = ttnn.experimental.all_gather_concat(
        input_tensor,
        dim,
        cluster_axis=cluster_axis,
        multi_device_global_semaphore=ccl_semaphore_handles[0],
        mesh_device=mesh_device,
        num_links=num_links,
        num_heads=8,
        memory_config=output_mem_config,
        topology=ttnn.Topology.Linear,
        subdevice_id=worker_sub_device_id,
        enable_persistent_fabric_mode=enable_persistent_fabric,
    )

    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")

    def capture_trace(n_iters):
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(n_iters):
            tt_out_tensor = ttnn.experimental.all_gather_concat(
                input_tensor,
                dim,
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                multi_device_global_semaphore=ccl_semaphore_handles[i % NUM_BUFFERS],
                num_links=num_links,
                num_heads=8,
                memory_config=output_mem_config,
                topology=ttnn.Topology.Linear,
                subdevice_id=worker_sub_device_id,
                enable_persistent_fabric_mode=enable_persistent_fabric,
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


def run_line_all_gather_concat_on_TG_with_mesh_tensor_along_rows(
    mesh_device,
    num_devices_per_line,
    per_chip_output_shape,
    tensor_memory_layout,
    dim,
    num_links,
    input_dtype,
    layout,
    buffer_type: ttnn.BufferType,
    # use_program_cache,
    function_level_defaults,
    enable_async,
    input_shard_spec: ttnn.ShardSpec = None,
    output_shard_spec: ttnn.ShardSpec = None,
    num_all_gather_instances: int = 1,
    num_iters: int = 1,
    warmup_iters: int = 0,
    cluster_axis: int = 0,
    tile=(32, 32),
    trace_mode=False,
    debug=False,
    profiler=BenchmarkProfiler(),
    # New all-gather-async and persistent fabric params
    use_all_gather_async=False,
    enable_persistent_fabric=False,
    create_persistent_fabric=False,
    teardown_persistent_fabric=False,
):
    if create_persistent_fabric:
        assert use_all_gather_async
        assert enable_persistent_fabric
    if teardown_persistent_fabric:
        assert use_all_gather_async
        assert enable_persistent_fabric
    if not use_all_gather_async:
        assert not create_persistent_fabric
        assert not teardown_persistent_fabric
        assert not enable_persistent_fabric

    mesh_device.enable_async(enable_async)

    input_shape_per_chip = list(per_chip_output_shape)
    input_shape_per_chip[dim] //= num_devices_per_line
    tensor_height_per_all_gather = per_chip_output_shape[-2]

    full_mesh_input_shape = list(per_chip_output_shape)
    ## The `all_gather_instances_concat_dim` is the dimension we will split the cluster spanning tensor along in order to split it
    ## off into per-all-gather tensors
    all_gather_instances_concat_dim = 1 if dim == 0 else 0
    print("all_gather_instances_concat_dim:", all_gather_instances_concat_dim)
    print("per_chip_output_shape: ", full_mesh_input_shape)
    print("all_gather_instances_concat_dim: ", all_gather_instances_concat_dim)
    print("num_devices_per_line: ", num_devices_per_line)

    full_mesh_input_shape[all_gather_instances_concat_dim] *= num_all_gather_instances
    print("full_mesh_input_shape: ", full_mesh_input_shape)
    logger.info(
        f"per_chip_output_shape: {full_mesh_input_shape}, dim: {dim}, all_gather_instances_concat_dim: {all_gather_instances_concat_dim}, num_devices_per_line: {num_devices_per_line}"
    )

    all_gather_instances_goldens = []
    full_input_tensor_unfractured = torch.rand(full_mesh_input_shape, dtype=torch.bfloat16)

    input_mem_config = ttnn.MemoryConfig(tensor_memory_layout, buffer_type=buffer_type, shard_spec=input_shard_spec)
    shard_dims = (dim, all_gather_instances_concat_dim) if cluster_axis == 0 else (all_gather_instances_concat_dim, dim)
    concat_dims = shard_dims

    mesh_shape = (
        (num_devices_per_line, num_all_gather_instances)
        if cluster_axis == 0
        else (num_all_gather_instances, num_devices_per_line)
    )

    if input_shard_spec is not None and output_shard_spec is None:
        output_shard_shape = list(input_shard_spec.shape)
        if dim == len(per_chip_output_shape) - 1:
            output_shard_shape[1] *= num_devices_per_line
        else:
            output_shard_shape[0] *= num_devices_per_line
        output_shard_spec = ttnn.ShardSpec(
            input_shard_spec.grid,
            output_shard_shape,
            input_shard_spec.orientation,
        )

    output_mem_config = ttnn.MemoryConfig(tensor_memory_layout, buffer_type=buffer_type, shard_spec=output_shard_spec)
    logger.info(f"input_shard_shape: {input_shard_spec.shape}, output_shard_shape: {output_shard_spec.shape}")
    ttnn_tensor = ttnn.from_torch(
        full_input_tensor_unfractured,
        tile=ttnn.Tile(tile),
        dtype=input_dtype,
        device=mesh_device,
        layout=layout,
        memory_config=input_mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=shard_dims),
    )
    logger.info(f"ttnn_tensor_shape: {ttnn_tensor.shape}, mesh_shape: {mesh_shape}, shard_dims: {shard_dims}")
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
    ttnn_tensor = ttnn.to_memory_config(ttnn_tensor, input_mem_config)

    sub_device_stall_group = []
    if use_all_gather_async:
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
        if create_persistent_fabric:
            logger.info("Create persistent fabric interface")
            mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
                mesh_device, [worker_sub_device], 0, 0, enable_persistent_fabric
            )
            logger.info("Done Create persistent fabric interface")
            mesh_device.set_sub_device_stall_group(sub_device_stall_group)

        # create global semaphore handles
        ccl_semaphore_handles = [
            create_global_semaphore_with_same_address(mesh_device, ccl_sub_device_crs, 0) for _ in range(NUM_BUFFERS)
        ]
    try:
        # ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor)
        if trace_mode:
            ttnn_tensor_out = run_with_trace(
                input_tensor=ttnn_tensor,
                dim=dim,
                mesh_device=mesh_device,
                cluster_axis=cluster_axis,
                num_links=num_links,
                output_mem_config=output_mem_config,
                ccl_semaphore_handles=ccl_semaphore_handles,
                worker_sub_device_id=worker_sub_device_id,
                enable_persistent_fabric=enable_persistent_fabric,
                all_gather_topology=ttnn.Topology.Linear,
                num_iter=num_iters,
                warmup_iters=warmup_iters,
                use_all_gather_async=use_all_gather_async,
                profiler=profiler,
            )

        else:
            signpost("start")
            for i in range(num_iters):
                logger.info("Running all-gather async")
                ttnn_tensor_out = ttnn.experimental.all_gather_concat(
                    ttnn_tensor,
                    dim,
                    cluster_axis=cluster_axis,
                    multi_device_global_semaphore=ccl_semaphore_handles[i % NUM_BUFFERS],
                    mesh_device=mesh_device,
                    num_links=num_links,
                    num_heads=8,
                    memory_config=output_mem_config,
                    topology=ttnn.Topology.Linear,
                    subdevice_id=worker_sub_device_id,
                    enable_persistent_fabric_mode=enable_persistent_fabric,
                )

            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            signpost("stop")
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise e
    finally:
        if enable_persistent_fabric and teardown_persistent_fabric:
            logger.info("Tearing down persistent fabric interface")
            mesh_device.reset_sub_device_stall_group()
            teardown_fabric_interface(mesh_device)
            logger.info("Done tearing down persistent fabric interface")

    # ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor_out)
    tt_output_tensor = ttnn.to_torch(
        ttnn_tensor_out, mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=concat_dims)
    )
    output_tensors_list = torch.chunk(tt_output_tensor, num_all_gather_instances, dim=all_gather_instances_concat_dim)
    output_golden = torch.zeros(tt_output_tensor.shape)

    # Repeat the input tensor to represent the fact that the full concatenated input tensor lives across every
    # device in the line
    repeat_factor = [1] * len(output_golden.shape)
    repeat_factor[dim] = num_devices_per_line

    # here
    output_tensor_concat = full_input_tensor_unfractured[:, :, :8, :]
    print("output_tensor_concat1:", output_tensor_concat.shape)
    output_tensor_concat = output_tensor_concat.reshape(num_all_gather_instances, 1, 32, 1024)
    print("output_tensor_concat2:", output_tensor_concat.shape)

    output_golden[:, :, :, :] = output_tensor_concat.repeat(repeat_factor)
    print("output_golden:", output_golden.shape)

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


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 3),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "num_iters, warmup_iters",
    [
        (NUM_ITERATIONS, 0),
    ],
)
@pytest.mark.parametrize("shard_grid_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "tensor_mem_layout, output_shape, dim, input_shard_shape,input_shard_grid,output_shard_shape, output_shard_grid, layout",
    (
        (  # AllGather after SDPA
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (1, 32, 32, 128),
            1,
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            ttnn.TILE_LAYOUT,
        ),
    ),
    ids=[
        "sdpa",
    ],
)
@pytest.mark.parametrize("replication_factor", [8])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 17068032}], indirect=True)
def test_all_gather_concat_fuse_llama(
    mesh_device,
    num_devices,
    output_shape,
    input_shard_shape,
    input_shard_grid,
    output_shard_shape,
    output_shard_grid,
    shard_grid_orientation,
    tensor_mem_layout,
    dim,
    num_links,
    input_dtype,
    layout,
    # use_program_cache,
    function_level_defaults,
    enable_async,
    replication_factor,
    num_iters,
    warmup_iters,
):
    if len(mesh_device.get_devices()) != 32:
        pytest.skip("Not TG!")
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        shard_grid_orientation,
    )

    if output_shard_grid is not None and output_shard_shape is not None:
        output_shard_spec = ttnn.ShardSpec(
            output_shard_grid,
            output_shard_shape,
            shard_grid_orientation,
        )
    else:
        output_shard_spec = None

    profiler = BenchmarkProfiler()

    run_line_all_gather_concat_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        output_shape,
        tensor_mem_layout,
        dim,
        num_links,
        input_dtype,
        layout,
        ttnn.BufferType.L1,
        # use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        input_shard_spec=input_shard_spec,
        output_shard_spec=output_shard_spec,
        num_all_gather_instances=replication_factor,
        cluster_axis=1,
        profiler=profiler,
        trace_mode=False,
        use_all_gather_async=True,
        enable_persistent_fabric=True,
        create_persistent_fabric=True,
        teardown_persistent_fabric=True,
    )

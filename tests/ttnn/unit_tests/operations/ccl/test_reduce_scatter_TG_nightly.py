# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from ttnn import ShardTensor2dMesh, ConcatMesh2dToTensor


def report_mismatches(golden, actual, max_printable=None):
    printed = 0
    for w in range(golden.shape[0]):
        for z in range(golden.shape[1]):
            for y in range(0, golden.shape[2], 32):
                for x in range(0, golden.shape[3], 32):
                    print_it = (max_printable is None or printed < max_printable) and golden[w, z, y, x] != actual[
                        w, z, y, x
                    ]
                    if print_it:
                        printed += 1
                        print(
                            f"output mismatch for tensor at [{w}, {z}, {y}, {x}]: expected {golden[w, z, y, x]} != actual {actual[w, z, y, x]}"
                        )


def print_tile_corners_of_tensor(t):
    for w in range(t.shape[0]):
        for z in range(t.shape[1]):
            str = ""
            for x in range(0, t.shape[3], 32):
                str += f"{x:<5} "[:5]
            print(f"     {str}")
            for y in range(0, t.shape[2], 32):
                str_vals = f"y={y:<3} "[:5]
                for x in range(0, t.shape[3], 32):
                    yy = 0
                    xx = 0
                    val = int(t[w, z, y + yy, x + xx].item())
                    str_vals += f"{val:<5} "[:5]
                print(f"{str_vals}")


def run_with_trace(
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
):
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.reduce_scatter(
        input_tensor,
        dim=dim,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        math_op=math_op,
        num_links=num_links,
        memory_config=output_mem_config,
        topology=ttnn.Topology.Linear,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.reduce_scatter(
            input_tensor,
            dim=dim,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            math_op=math_op,
            num_links=num_links,
            memory_config=output_mem_config,
            topology=ttnn.Topology.Linear,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    return tt_out_tensor


def run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows(
    mesh_device,
    num_devices_per_line,
    per_chip_input_shape,
    tensor_memory_layout,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type: ttnn.BufferType,
    use_program_cache,
    function_level_defaults,
    input_shard_spec: ttnn.ShardSpec = None,
    num_reduce_scatter_instances: int = 1,
    num_iters: int = 1,
    cluster_axis: int = 0,
    trace_mode=False,
    # New all-gather-async and persistent fabric params
    use_reduce_scatter_async=False,
    use_persistent_output=False,
):
    mesh_device.enable_program_cache()

    per_reduce_scatter_output_shape = list(per_chip_input_shape)
    per_reduce_scatter_output_shape[dim] *= num_devices_per_line
    full_mesh_input_shape = list(per_reduce_scatter_output_shape)
    ## The `reduce_scatter_instances_concat_dim` is the dimension we will split the cluster spanning tensor along in order to split it
    ## off into per-all-gather tensors
    reduce_scatter_instances_concat_dim = 1 if dim == 0 else 0
    full_mesh_input_shape[reduce_scatter_instances_concat_dim] *= num_reduce_scatter_instances
    logger.info(
        f"full_mesh_input_shape: {full_mesh_input_shape}, dim: {dim}, reduce_scatter_instances_concat_dim: {reduce_scatter_instances_concat_dim}, num_devices_per_line: {num_devices_per_line}"
    )

    sub_device_stall_group = []
    if use_reduce_scatter_async:
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
        from_remote_semaphore_handles = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)
        to_remote_semaphore_handles = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)
    else:
        worker_sub_device_id = None
    ##
    ## Compute golden
    ##

    per_chip_output_shape = list(per_chip_input_shape)
    per_chip_output_shape[dim] //= num_devices_per_line
    per_reduce_scatter_inputs = []
    per_reduce_scatter_goldens = []
    for i in range(num_reduce_scatter_instances):
        per_chip_inputs = [torch.rand(per_chip_input_shape).bfloat16() for _ in range(num_devices_per_line)]
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
        (num_devices_per_line, num_reduce_scatter_instances)
        if cluster_axis == 0
        else (num_reduce_scatter_instances, num_devices_per_line)
    )

    output_shard_spec = None
    if input_shard_spec is not None:
        output_shard_shape = list(input_shard_spec.shape)
        if dim == 3:
            output_shard_shape[1] //= num_devices_per_line
        else:
            output_shard_shape[0] //= num_devices_per_line
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

    if trace_mode:
        ttnn_tensor_out = run_with_trace(
            input_tensor=ttnn_tensor,
            dim=dim,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            math_op=math_op,
            output_mem_config=output_mem_config,
            all_gather_topology=ttnn.Topology.Linear,
            num_links=num_links,
            num_iter=num_iters,
        )
    else:
        logger.info(f"Running {num_iters} iterations of reduce scatter")
        persistent_buffers = None
        if use_persistent_output:
            tile = (32, 32)
            persistent_buffers = [
                ttnn.from_torch(
                    torch.zeros(per_chip_output_shape),
                    tile=ttnn.Tile(tile),
                    dtype=input_dtype,
                    device=mesh_device,
                    layout=layout,
                    memory_config=output_mem_config,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                ),
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
        for _ in range(num_iters):
            if use_reduce_scatter_async:
                ttnn_tensor_out = ttnn.experimental.reduce_scatter_async(
                    ttnn_tensor,
                    dim=dim,
                    cluster_axis=cluster_axis,
                    mesh_device=mesh_device,
                    from_remote_multi_device_global_semaphore=from_remote_semaphore_handles,
                    to_remote_multi_device_global_semaphore=to_remote_semaphore_handles,
                    math_op=math_op,
                    persistent_output_tensors=persistent_buffers,
                    memory_config=output_mem_config,
                    topology=ttnn.Topology.Linear,
                    num_links=num_links,
                    subdevice_id=worker_sub_device_id,
                )
            else:
                ttnn_tensor_out = ttnn.reduce_scatter(
                    ttnn_tensor,
                    dim=dim,
                    cluster_axis=cluster_axis,
                    mesh_device=mesh_device,
                    math_op=math_op,
                    num_links=num_links,
                    memory_config=output_mem_config,
                    topology=ttnn.Topology.Linear,
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
    # Check the tensor addresses
    if use_persistent_output:
        persistent_output_tensors = ttnn.get_device_tensors(persistent_buffers[0])
        output_tensors = ttnn.get_device_tensors(ttnn_tensor_out)

        for persistent_tensor, output_tensor in zip(persistent_output_tensors, output_tensors):
            assert (
                persistent_tensor.buffer_address() == output_tensor.buffer_address()
            ), "Persistent tensor address mismatch"

    passed = True
    for i in range(num_reduce_scatter_instances):
        # The result of all-chips in the reduce scatter line having their outputs concatenated
        reduce_scatter_outputs_concatenated = output_tensors_list[i]
        per_chip_outputs = torch.chunk(reduce_scatter_outputs_concatenated, num_devices_per_line, dim=dim)
        per_chip_goldens = torch.chunk(per_reduce_scatter_goldens[i], num_devices_per_line, dim=dim)

        assert len(per_chip_outputs) == len(per_chip_goldens)
        # compare the output and golden (zip)
        for d, (output, golden) in enumerate(zip(per_chip_outputs, per_chip_goldens)):
            eq, output = comp_pcc(output, golden)

            if not eq:
                passed = False
                logger.error(f"output mismatch for tensor on reduce_scatter {i}, device {d}: {output}")

    assert passed, f"FAILED: {output}"


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, dim, layout",
    [
        (4, 2, [1, 4, 32, 2304], 1, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("replication_factor", [8])  # 1, 8])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10281600, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_line_reduce_scatter_on_TG_rows_post_commit(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    use_program_cache,
    function_level_defaults,
    replication_factor,
    num_iters=16,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")
    run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
        num_reduce_scatter_instances=replication_factor,
        cluster_axis=1,
        use_reduce_scatter_async=True,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, dim, layout",
    [
        (8, 2, [1, 8, 32, 1280], 1, ttnn.TILE_LAYOUT),
        (8, 2, [8, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
    ],
)
@pytest.mark.parametrize("replication_factor", [4])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_line_reduce_scatter_on_TG_cols_post_commit(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    use_program_cache,
    function_level_defaults,
    replication_factor,
    num_iters=16,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")

    run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
        num_reduce_scatter_instances=replication_factor,
        cluster_axis=0,
        use_reduce_scatter_async=True,
    )

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
    enable_async,
    input_shard_spec: ttnn.ShardSpec = None,
    num_reduce_scatter_instances: int = 1,
    num_iters: int = 1,
    cluster_axis: int = 0,
):
    if len(mesh_device.get_devices()) != 32:
        pytest.skip("Not TG!")
    for d in mesh_device.get_devices():
        ttnn.enable_program_cache(d)
    mesh_device.enable_async(enable_async)

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
            output_shard_shape[1] *= num_devices_per_line
        else:
            output_shard_shape[0] *= num_devices_per_line
        output_shard_spec = ttnn.ShardSpec(
            input_shard_spec.grid,
            output_shard_shape,
            input_shard_spec.orientation,
            False,
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

    # ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor)
    ttnn_tensor_out = ttnn.reduce_scatter(
        ttnn_tensor,
        scatter_dim=dim,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        math_op=math_op,
        num_links=num_links,
        memory_config=output_mem_config,
        topology=ttnn.Topology.Linear,
    )
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    # ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor)
    for _ in range(num_iters):
        ttnn_tensor_out = ttnn.reduce_scatter(
            ttnn_tensor,
            scatter_dim=dim,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            math_op=math_op,
            num_links=num_links,
            memory_config=output_mem_config,
            topology=ttnn.Topology.Linear,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    for d in mesh_device.get_devices():
        ttnn.synchronize_device(d)

    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    for d in mesh_device.get_devices():
        ttnn.synchronize_device(d)

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
        (4, 1, [1, 4, 32, 2304], 1, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 10281600}], indirect=True)
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
    enable_async,
    replication_factor,
    num_iters=16,
):
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
        enable_async=enable_async,
        num_iters=num_iters,
        num_reduce_scatter_instances=replication_factor,
        cluster_axis=1,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, dim, layout",
    [
        (8, 1, [1, 8, 32, 1280], 1, ttnn.TILE_LAYOUT),
        (8, 1, [8, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("replication_factor", [4])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 10281600}], indirect=True)
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
    enable_async,
    replication_factor,
    num_iters=16,
):
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
        enable_async=enable_async,
        num_iters=num_iters,
        num_reduce_scatter_instances=replication_factor,
        cluster_axis=0,
    )

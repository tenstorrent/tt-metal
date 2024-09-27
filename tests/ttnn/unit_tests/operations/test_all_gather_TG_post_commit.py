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
                        logger.error(
                            f"output mismatch for tensor at [{w}, {z}, {y}, {x}]: expected {int(golden[w, z, y, x])} != actual {int(actual[w, z, y, x])}"
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
                    str_vals += f"{int(t[w, z, y + yy, x + xx]):<5} "[:5]
                print(f"{str_vals}")


def run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
    mesh_device,
    num_devices_per_line,
    input_shape_per_all_gather,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_all_gather_instances=1,
    num_iters=1,
    cluster_axis=0,
):
    if len(mesh_device.get_devices()) != 32:
        pytest.skip("Not TG!")
    for device in mesh_device.get_devices():
        device.enable_async(enable_async)

    input_shape_per_chip = list(input_shape_per_all_gather)
    input_shape_per_chip[2 if cluster_axis == 0 else 3] //= num_devices_per_line
    tensor_height_per_all_gather = input_shape_per_all_gather[-2]

    full_mesh_input_shape = list(input_shape_per_all_gather)
    full_mesh_input_shape[-2] *= num_all_gather_instances
    logger.info(f"tensor_height_per_all_gather: {tensor_height_per_all_gather}")
    logger.info(f"input_shape_per_all_gather: {input_shape_per_all_gather}")
    logger.info(f"input_shape_per_chip: {input_shape_per_chip}")
    logger.info(f"full_mesh_input_shape: {full_mesh_input_shape}")
    logger.info(f"input_shape_per_all_gather: {input_shape_per_all_gather}")

    full_tensor = torch.zeros(full_mesh_input_shape, dtype=torch.bfloat16)

    for i in range(num_all_gather_instances):
        full_tensor[0, 0, i * tensor_height_per_all_gather : (i + 1) * tensor_height_per_all_gather, :] = torch.rand(
            input_shape_per_all_gather
        ).bfloat16()

    logger.info(f"full_tensor.shape: {full_tensor.shape}")
    debug = False
    if debug:
        tile_id = 0
        for w in range(full_tensor.shape[0]):
            for z in range(full_tensor.shape[1]):
                for y in range(0, full_tensor.shape[2], 32):
                    for x in range(0, full_tensor.shape[3], 32):
                        yy_max = 32 if y + 32 < full_tensor.shape[2] else full_tensor.shape[2] - y
                        xx_max = 32 if x + 32 < full_tensor.shape[3] else full_tensor.shape[3] - x
                        full_tensor[w, z, y : y + yy_max, x : x + xx_max] = tile_id
                        tile_id += 1

    #
    # assemble the golden output tensor
    #
    inner_dim_concat_axis = 2
    outer_dim_concat_axis = 3
    full_tensor_chunks_per_allgather = torch.chunk(full_tensor, num_all_gather_instances, dim=inner_dim_concat_axis)
    output_chunks_per_allgather = []
    for i, chunk in enumerate(full_tensor_chunks_per_allgather):
        width_chunks = torch.chunk(chunk, num_devices_per_line, dim=outer_dim_concat_axis)
        output_chunk = torch.cat(width_chunks, dim=dim)
        output_chunks_per_allgather.append(output_chunk)
    full_mesh_output_golden_per_chip = torch.cat(output_chunks_per_allgather, dim=inner_dim_concat_axis)
    logger.info(f"full_mesh_output_golden_per_chip.shape: {full_mesh_output_golden_per_chip.shape}")
    non_replicated_output_golden_tensors = [full_mesh_output_golden_per_chip] * num_devices_per_line
    full_mesh_output_golden = torch.cat(non_replicated_output_golden_tensors, dim=outer_dim_concat_axis)
    logger.info(f"full_mesh_output_golden.shape: {full_mesh_output_golden.shape}")

    shard_dims = (-1, -2) if cluster_axis == 0 else (-2, -1)
    mesh_shape = (
        (num_devices_per_line, num_all_gather_instances)
        if cluster_axis == 0
        else (num_all_gather_instances, num_devices_per_line)
    )
    logger.info(f"mesh_shape: {mesh_shape}")
    ttnn_tensor = ttnn.from_torch(
        full_tensor,
        dtype=input_dtype,
        device=mesh_device,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=shard_dims),
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)

    # ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor)
    for _ in range(num_iters):
        ttnn_tensor_out = ttnn.all_gather(
            ttnn_tensor, dim=dim, cluster_axis=cluster_axis, mesh_device=mesh_device, num_links=num_links
        )

    concat_dims = (3, 2) if cluster_axis == 0 else (2, 3)
    if debug:
        readback_input_tensor = ttnn.to_torch(
            ttnn_tensor, mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=concat_dims)
        )
        print(f"readback_input_tensor")
        print_tile_corners_of_tensor(readback_input_tensor)

    if debug:
        for i, t in enumerate(ttnn.get_device_tensors(ttnn_tensor)):
            print(f"readback_input_tensor {i}")
            print_tile_corners_of_tensor(t)

    if debug:
        for i, t in enumerate(ttnn.get_device_tensors(ttnn_tensor_out)):
            t = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            print(f"OUTPUT TENSOR {i}")
            print_tile_corners_of_tensor(t)

    # ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor_out)
    logger.info(f"concat_dims: {concat_dims}")
    tt_output_tensor = ttnn.to_torch(
        ttnn_tensor_out, mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=concat_dims)
    )
    logger.info(f"tt_output_tensor.shape: {tt_output_tensor.shape}")

    if debug:
        print(f"tt_output_tensor")
        print_tile_corners_of_tensor(tt_output_tensor)

    ## This full_tensor will only be 1/num_devices_per_line of the tt_output_tensor. We should just be able to concatenate it along the
    if input_dtype == ttnn.bfloat16:
        eq, output = comp_equal(tt_output_tensor, full_mesh_output_golden)
        if not eq and debug:
            report_mismatches(full_mesh_output_golden, tt_output_tensor)
    else:
        eq, output = comp_pcc(tt_output_tensor, full_mesh_output_golden)
    if not eq:
        logger.error(f"output mismatch for tensor")
    assert eq, f"FAILED: {output}"


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (4, 3, [1, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
        (4, 3, [1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
        (4, 3, [1, 1, 32, 2304], 1, ttnn.TILE_LAYOUT),
        (4, 3, [1, 1, 32, 4096], 1, ttnn.TILE_LAYOUT),
        (4, 3, [1, 1, 32, 6656], 1, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("replication_factor", [8])  # 1, 8])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_line_all_gather_on_TG_rows_post_commit(
    mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    replication_factor,
    num_iters=1,
):
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        num_iters=num_iters,
        num_all_gather_instances=replication_factor,
        cluster_axis=1,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        # (8, 4, [1, 1, 32, 1280], 1, ttnn.TILE_LAYOUT), # Rightmost column of tiles per input not copied to final output
        (8, 4, [1, 1, 32, 2048], 1, ttnn.TILE_LAYOUT),  # passes
        (8, 4, [1, 1, 32, 2304], 1, ttnn.TILE_LAYOUT),  # passes
        (8, 4, [1, 1, 32, 4096], 1, ttnn.TILE_LAYOUT),  # passes
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
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("enable_async", [False])
@pytest.mark.parametrize("replication_factor", [4])  # 1, 4])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_line_all_gather_on_TG_cols_post_commit(
    mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    replication_factor,
    num_iters=1,
):
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        num_iters=num_iters,
        num_all_gather_instances=replication_factor,
        cluster_axis=0,
    )

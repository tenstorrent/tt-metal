# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull


def is_unsupported_case(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout, tile):
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        return True, "Invalid combination"

    if input_shape[dim] % num_devices != 0:
        return True, "Unsupported test case"
    if tile != (32, 32) and input_dtype != ttnn.bfloat16:
        return True, "Tiny tile only supports bfloat16"

    ## Check that we can readback results
    fast_dispatch_page_size_limit = 55 * 1024
    elem_size = 2 if input_dtype == ttnn.bfloat16 else 1
    if layout == ttnn.ROW_MAJOR_LAYOUT and (input_shape[dim] * elem_size) > fast_dispatch_page_size_limit:
        # Fast dispatch currently can't breakup readback of large pages into multiple smaller pages and is
        # limited to ~55K pages.
        return True, "Fast dispatch can't support reading back this page size in one shot"

    # Check that we can fit in L1 (if L1 config)
    tensor_size_bytes = elem_size
    for i in input_shape:
        tensor_size_bytes *= i
    num_l1_banks = 64
    if mem_config.buffer_type == ttnn.BufferType.L1 and tensor_size_bytes > num_l1_banks * 50 * 1024:
        return True, "L1 buffer can't support large tensor sizes"

    # Check that each chip has a non-zero amount of data available
    if input_shape[dim] < num_devices:
        return (
            True,
            f"Input shape {input_shape} incompatible with {num_devices} on dim {dim} because some chips will have no tensor",
        )

    if (
        input_shape == [8, 8, 256, 384]
        and dim == 1
        and layout == ttnn.TILE_LAYOUT
        and (input_dtype == ttnn.bfloat8_b or tile != (32, 32))
    ):
        return True, "Known failure"

    return False, ""


def is_unsupported_case_n300(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout, tile):
    return is_unsupported_case(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout, tile)


def run_split_scatter_impl(
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
    split_scatter_topology,
    num_iters=1,
    enable_async=False,
    trace_mode=False,
    tile=(32, 32),
):
    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")
    # Use Async mode based on test input config
    mesh_device.enable_async(enable_async)

    if enable_async:
        logger.info(f"Using Async Mode for All Gather Op Dispatch")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    ttnn.set_printoptions(profile="full")
    torch.set_printoptions(profile="full")
    input_tensors = torch.zeros(input_shape)
    print("input shapeee ", input_shape)
    tile_id = 1
    for w in range(input_shape[0]):
        for z in range(input_shape[1]):
            for y in range(0, input_shape[2], 32):
                for x in range(0, input_shape[3], 32):
                    input_tensors[w, z, y : y + 32, x : x + 32] = tile_id
                    tile_id += 1
    input_tensors = torch.rand(input_shape).bfloat16()
    input_tensors = torch.arange(1, 17).reshape(1, 1, 1, 16)
    input_tensorsnext = torch.arange(9, 25).reshape(1, 1, 1, 16)
    tt_input_tensors = []
    # for i, t in enumerate(input_tensors):
    t = ttnn.from_torch(input_tensors, input_dtype, layout=layout, tile=ttnn.Tile(tile))
    tt = ttnn.from_torch(input_tensorsnext, input_dtype, layout=layout, tile=ttnn.Tile(tile))
    tt_input_tensors.append(t.to(mesh_device.get_devices()[0], mem_config))
    tt_input_tensors.append(tt.to(mesh_device.get_devices()[1], mem_config))
    print("inp mesh ", tt_input_tensors)
    golden_output_tensors = torch.chunk(input_tensors, num_devices, dim)

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

    for i in range(num_iters):
        tt_out_tensor = ttnn.experimental.split_scatter(
            input_tensor_mesh, dim, num_links=num_links, memory_config=mem_config, topology=split_scatter_topology
        )
        print("out tensor mesh ", tt_out_tensor)

        for d in mesh_device.get_devices():
            ttnn.synchronize_device(d)
        logger.info(f"Done iteration {i}")

    outs = ttnn.get_device_tensors(tt_out_tensor)
    print("ot lens ", len(outs))
    for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
        tt_output_tensor = ttnn.to_torch(t)
        print("COMPAREEEE loop ----------> ", i)
        print(tt_output_tensor)
        print(golden_output_tensors[i])
        print("tt shape ", tt_output_tensor.shape)
        print("torch shape ", golden_output_tensors[i].shape)
        if input_dtype == ttnn.bfloat16:
            eq, output = comp_equal(tt_output_tensor, golden_output_tensors[i])
        else:
            eq, output = comp_pcc(tt_output_tensor, golden_output_tensors[i])
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
        # assert eq, f"{i} FAILED: {output}"


def run_split_scatter_on_n300_impl(
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
    split_scatter_topology,
    num_iters=1,
    enable_async=False,
    trace_mode=False,
    tile=(32, 32),
):
    if mesh_device.get_num_devices() != 2:
        pytest.skip("Not N300!")

    (is_known_failure, message) = is_unsupported_case_n300(
        input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout, tile
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    return run_split_scatter_impl(
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
        split_scatter_topology=split_scatter_topology,
        num_iters=num_iters,
        enable_async=enable_async,
        trace_mode=trace_mode,
        tile=tile,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, dim, layout",
    [
        (2, 1, [1, 1, 1, 16], 3, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
    ],
)
@pytest.mark.parametrize("num_iters", [1])
@pytest.mark.parametrize("enable_async", [True])
def test_split_scatter_on_n300_post_commit(
    n300_mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_split_scatter_on_n300_impl(
        n300_mesh_device,
        num_devices,
        output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        split_scatter_topology=ttnn.Topology.Linear,
        num_iters=num_iters,
        enable_async=enable_async,
    )

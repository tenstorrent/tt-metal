# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import tt_lib as ttl
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000
import itertools


def is_unsupported_case(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout):
    if layout == ttl.tensor.Layout.ROW_MAJOR and input_dtype == ttl.tensor.DataType.BFLOAT8_B:
        return True, "Invalid combination"

    if num_devices < 2:
        return True, "Requires multiple devices to run"
    elif num_devices == 2 and num_links <= 2:
        return True, "Not enough links to run"

    if input_shape[dim] % num_devices != 0 or (dim == 3 and input_shape[dim] // num_devices % 32 != 0):
        return True, "Unsupported test case"

    ## Check that we can readback results
    fast_dispatch_page_size_limit = 55 * 1024
    elem_size = 2 if input_dtype == ttl.tensor.DataType.BFLOAT16 else 1
    if layout == ttl.tensor.Layout.ROW_MAJOR and (input_shape[dim] * elem_size) > fast_dispatch_page_size_limit:
        # Fast dispatch currently can't breakup readback of large pages into multiple smaller pages and is
        # limited to ~55K pages.
        return True, "Fast dispatch can't support reading back this page size in one shot"

    # Check that we can fit in L1 (if L1 config)
    tensor_size_bytes = elem_size
    for i in input_shape:
        tensor_size_bytes *= i
    num_l1_banks = 64
    if mem_config.buffer_type == ttl.tensor.BufferType.L1 and tensor_size_bytes > num_l1_banks * 50 * 1024:
        return True, "L1 buffer can't support large tensor sizes"

    # Check that each chip has a non-zero amount of data available
    min_sized_chunks_on_dim = input_shape[dim]
    if dim == 3:
        min_sized_chunks_on_dim //= 32
    if dim == 2:
        if layout == ttl.tensor.Layout.TILE:
            min_sized_chunks_on_dim //= 32
    if min_sized_chunks_on_dim < num_devices:
        return (
            True,
            f"Input shape {input_shape} incompatible with {num_devices} on dim {dim} because some chips will have no tensor",
        )

    if (
        input_shape == [8, 8, 256, 384]
        and dim == 1
        and layout == ttl.tensor.Layout.TILE
        and input_dtype == ttl.tensor.DataType.BFLOAT8_B
    ):
        return True, "Known failure"

    return False, ""


def run_line_all_gather(
    all_devices,
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
    num_iters=1,
):
    if len(all_devices) != 8:
        pytest.skip("Not T3000!")

    for device in all_devices:
        device.enable_async(enable_async)

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    (is_known_failure, message) = is_unsupported_case(
        input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    devices = get_devices_for_t3000(all_devices, num_devices)
    # for device in devices:
    #    device.disable_and_clear_program_cache()

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    input_tensor = torch.rand(input_shape).bfloat16()

    input_tensors = torch.chunk(input_tensor, num_devices, dim)
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        tt_input_tensors.append(ttl.tensor.Tensor(t, input_dtype).to(layout).to(devices[i], mem_config))

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)
    for i in range(num_iters):
        tt_out_tensor = ttnn.line_all_gather(input_tensor_mesh, dim, num_links=num_links, memory_config=mem_config)

        for d in devices:
            ttl.device.Synchronize(d)
        logger.info(f"Done iteration {i}")

    for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
        tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        if input_dtype == ttl.tensor.DataType.BFLOAT16:
            eq, output = comp_equal(tt_output_tensor, input_tensor)
        else:
            eq, output = comp_pcc(tt_output_tensor, input_tensor)
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
        assert eq, f"{i} FAILED: {output}"


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        # (4, 1, [4, 1, 33, 256], 0, ttl.tensor.Layout.ROW_MAJOR), # https://github.com/tenstorrent/tt-metal/issues/9686
        (8, 1, [8, 1, 33, 256], 0, ttl.tensor.Layout.ROW_MAJOR),  # https://github.com/tenstorrent/tt-metal/issues/9686
        (8, 1, [8, 1, 256, 32], 0, ttl.tensor.Layout.TILE),
        (8, 1, [8, 8, 256, 384], 1, ttl.tensor.Layout.ROW_MAJOR),  # https://github.com/tenstorrent/tt-metal/issues/9686
        (4, 2, [8, 8, 256, 384], 1, ttl.tensor.Layout.TILE),  # https://github.com/tenstorrent/tt-metal/issues/9686
        (8, 1, [8, 8, 256, 384], 1, ttl.tensor.Layout.TILE),  # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 1, [8, 5, 13, 384], 3, ttl.tensor.Layout.ROW_MAJOR), # https://github.com/tenstorrent/tt-metal/issues/9686
        (8, 1, [8, 5, 13, 512], 3, ttl.tensor.Layout.ROW_MAJOR),  # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 1, [8, 5, 32, 384], 3, ttl.tensor.Layout.TILE), # https://github.com/tenstorrent/tt-metal/issues/9686
        (8, 1, [8, 5, 32, 512], 3, ttl.tensor.Layout.TILE),  # https://github.com/tenstorrent/tt-metal/issues/9686
        (4, 1, [1, 1, 32, 16384], 3, ttl.tensor.Layout.TILE),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttl.tensor.DataType.BFLOAT16,
        # ttl.tensor.DataType.BFLOAT8_B, # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttl.tensor.MemoryConfig(
            buffer_type=ttl.tensor.BufferType.DRAM
        ),  # https://github.com/tenstorrent/tt-metal/issues/9686
        # ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.L1),
    ],
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_line_all_gather_on_t3000_post_commit(
    all_devices,
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
    num_iters=1,
):
    run_line_all_gather(
        all_devices,
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
        num_iters,
    )


def run_line_all_gather_instances(
    all_devices,
    num_devices,
    num_instances,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    if len(all_devices) != 8:
        pytest.skip("Not T3000!")

    for device in all_devices:
        device.enable_async(enable_async)

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    (is_known_failure, message) = is_unsupported_case(
        input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    # devices = get_devices_for_t3000(all_devices, num_devices)
    # for device in devices:
    #    device.disable_and_clear_program_cache()

    t3000_device_rows = [
        [all_devices[4], all_devices[0], all_devices[3], all_devices[7]],
        [all_devices[5], all_devices[1], all_devices[2], all_devices[6]],
    ]
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    input_mesh_tensors = []
    input_tensor_to_compare = []
    for devices in t3000_device_rows:
        input_tensor = torch.rand(input_shape).bfloat16()
        input_tensor_to_compare.append(input_tensor)

        input_tensors = torch.chunk(input_tensor, len(devices), dim)
        tt_input_tensors = []
        for i, t in enumerate(input_tensors):
            tt_input_tensors.append(ttl.tensor.Tensor(t, input_dtype).to(layout).to(devices[i], mem_config))

        input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)
        input_mesh_tensors.append(input_tensor_mesh)

    result_mesh_tensors = []
    for i, devices in enumerate(t3000_device_rows):
        tt_out_tensor = ttnn.line_all_gather(input_mesh_tensors[i], dim, num_links=num_links, memory_config=mem_config)
        result_mesh_tensors.append(tt_out_tensor)

    ## Wait for completion
    for i, devices in enumerate(t3000_device_rows):
        for d in devices:
            ttl.device.Synchronize(d)

    for count, tt_out_tensor in enumerate(result_mesh_tensors):
        for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
            tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
            if input_dtype == ttl.tensor.DataType.BFLOAT16:
                eq, output = comp_equal(tt_output_tensor, input_tensor_to_compare[count])
            else:
                eq, output = comp_pcc(tt_output_tensor, input_tensor_to_compare[count])
            if not eq:
                logger.error(f"output mismatch for tensor {i}")
            assert eq, f"{i} FAILED: {output}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_instances, num_links, input_shape, dim, layout",
    [
        # (4, 1, [4, 1, 33, 256], 0, ttl.tensor.Layout.ROW_MAJOR), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 1, 33, 256], 0, ttl.tensor.Layout.ROW_MAJOR), # https://github.com/tenstorrent/tt-metal/issues/9686
        # # (8, 1, [8, 1, 256, 32], 0, ttl.tensor.Layout.TILE),
        # (8, 1, [8, 8, 256, 384], 1, ttl.tensor.Layout.ROW_MAJOR), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 2, [8, 8, 256, 384], 1, ttl.tensor.Layout.TILE), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 8, 256, 384], 1, ttl.tensor.Layout.TILE), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 1, [8, 5, 13, 384], 3, ttl.tensor.Layout.ROW_MAJOR), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 5, 13, 512], 3, ttl.tensor.Layout.ROW_MAJOR), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 1, [8, 5, 32, 384], 3, ttl.tensor.Layout.TILE), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (8, 1, [8, 5, 32, 512], 3, ttl.tensor.Layout.TILE), # https://github.com/tenstorrent/tt-metal/issues/9686
        # (4, 1, 1, [1, 1, 32, 16384], 3, ttl.tensor.Layout.TILE),
        (4, 2, 1, [1, 1, 32, 16384], 3, ttl.tensor.Layout.TILE),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttl.tensor.DataType.BFLOAT16,
        # ttl.tensor.DataType.BFLOAT8_B, # https://github.com/tenstorrent/tt-metal/issues/9686
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttl.tensor.MemoryConfig(
            buffer_type=ttl.tensor.BufferType.DRAM
        ),  # https://github.com/tenstorrent/tt-metal/issues/9686
        # ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.L1),
    ],
)
@pytest.mark.parametrize("enable_async", [True, False])
def test_line_all_gather_on_t3000_post_commit_instances(
    all_devices,
    num_devices,
    num_instances,
    input_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=1,
):
    run_line_all_gather_instances(
        all_devices,
        num_devices,
        num_instances,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        enable_async,
        num_iters,
    )

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull


def is_unsupported_case(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout):
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        return True, "Invalid combination"

    if input_shape[dim] % num_devices != 0 or (dim == 3 and input_shape[dim] // num_devices % 32 != 0):
        return True, "Unsupported test case"

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
    min_sized_chunks_on_dim = input_shape[dim]
    if dim == 3:
        min_sized_chunks_on_dim //= 32
    if dim == 2:
        if layout == ttnn.TILE_LAYOUT:
            min_sized_chunks_on_dim //= 32
    if min_sized_chunks_on_dim < num_devices:
        return (
            True,
            f"Input shape {input_shape} incompatible with {num_devices} on dim {dim} because some chips will have no tensor",
        )

    if input_shape == [8, 8, 256, 384] and dim == 1 and layout == ttnn.TILE_LAYOUT and input_dtype == ttnn.bfloat8_b:
        return True, "Known failure"

    return False, ""


def run_all_gather_on_n300_impl(
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
    all_gather_operation,
    num_iters=1,
    enable_async=False,
):
    if len(all_devices) != 2:
        pytest.skip("Not N300!")

    # Use Async mode based on test input config
    for device in all_devices:
        device.enable_async(enable_async)
    if enable_async:
        logger.info(f"Using Async Mode for All Gather Op Dispatch")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    (is_known_failure, message) = is_unsupported_case(
        input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    devices = all_devices
    # for device in devices:
    #    device.disable_and_clear_program_cache()

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    input_tensor = torch.rand(input_shape).bfloat16()

    input_tensors = torch.chunk(input_tensor, num_devices, dim)
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        tt_input_tensors.append(ttnn.Tensor(t, input_dtype).to(layout).to(devices[i], mem_config))

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)
    for i in range(num_iters):
        tt_out_tensor = all_gather_operation(input_tensor_mesh, dim, num_links=num_links, memory_config=mem_config)

        for d in devices:
            ttnn.synchronize_device(d)
        logger.info(f"Done iteration {i}")

    for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
        tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if input_dtype == ttnn.bfloat16:
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
        (2, 1, [1, 1, 64, 16384], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("enable_async", [True, False])
def test_all_gather_on_n300_post_commit(
    all_devices,
    num_devices,
    input_shape,
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
    run_all_gather_on_n300_impl(
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
        all_gather_operation=ttnn.all_gather,
        num_iters=num_iters,
        enable_async=enable_async,
    )


def run_all_gather_n300_sharded(
    all_devices,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    all_gather_operation,
    enable_async,
    n_worker=None,
    n_buffer=None,
    num_iter=1,
    trace_mode=False,
):
    if len(all_devices) != 2:
        pytest.skip("Not N300!")

    for device in all_devices:
        device.enable_async(enable_async)

    numel = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * num_devices
    unchunked_input_shape = list(input_shape)
    unchunked_input_shape[dim] *= num_devices

    unchunked_input_tensor = torch.rand(unchunked_input_shape).bfloat16()

    debug = False
    if debug:
        tile_id = 0
        for w in range(unchunked_input_shape[0]):
            for z in range(unchunked_input_shape[1]):
                for y in range(0, unchunked_input_shape[2], 32):
                    for x in range(0, unchunked_input_shape[3], 32):
                        for yy in range(32):
                            for xx in range(32):
                                unchunked_input_tensor[w][z][y + yy][x + xx] = tile_id
                        tile_id += 1

    unchunked_input_tensor = unchunked_input_tensor.bfloat16()

    input_tensors = torch.chunk(unchunked_input_tensor, num_devices, dim)
    devices = all_devices

    # num_cores =
    # compute_grid_size = devices[0].compute_with_storage_grid_size()

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"unchunked_input_shape: {unchunked_input_shape}")
    logger.info(f"dim: {dim}")
    logger.info(f"num_devices: {num_devices}")
    logger.info(f"num_links: {num_links}")
    logger.info(f"input_dtype: {input_dtype}")
    logger.info(f"tensor_layout: {tensor_layout}")
    logger.info(f"tensor_mem_layout: {tensor_mem_layout}")
    logger.info(f"orientation: {orientation}")
    # logger.info(f"num_cores: {num_cores}")
    logger.info(f"shard_grid: {shard_grid}")
    logger.info(f"input_shard_shape: {input_shard_shape}")

    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        input_shard_shape,
        orientation,
        False,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)
    output_shard_shape = list(input_shard_shape)
    if dim == 3:
        output_shard_shape[1] *= num_devices
    else:
        output_shard_shape[0] *= num_devices
    output_shard_spec = ttnn.ShardSpec(
        shard_grid,
        output_shard_shape,
        orientation,
        False,
    )
    output_mem_config = ttnn.MemoryConfig(
        tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
    )

    if num_devices == 2 and num_links == 2:
        pytest.skip("Not enough links to run")

    if unchunked_input_shape[dim] % num_devices != 0 or (
        dim == 3 and unchunked_input_shape[dim] // num_devices % 32 != 0
    ):
        pytest.skip("Unsupported test case")

    tt_input_tensors_dups = []
    tt_input_tensors = []

    for i, t in enumerate(input_tensors):
        tt_input_tensors_dups.append(ttnn.Tensor(t, input_dtype).to(tensor_layout).to(devices[i], input_mem_config))
        tt_input_tensors.append(ttnn.Tensor(t, input_dtype).to(tensor_layout).to(devices[i], input_mem_config))

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

    ## Run the actual allgather operation
    for i in range(num_iter):
        tt_out_tensor = all_gather_operation(
            input_tensor_mesh,
            dim,
            num_links=num_links,
            memory_config=output_mem_config,
            num_workers=n_worker,
            num_buffers_per_channel=n_buffer,
        )
    ## Wait for completion
    for d in devices:
        ttnn.synchronize_device(d)

    torch.set_printoptions(sci_mode=False)
    all_eq = True
    reported_mismatch = False
    for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
        tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if input_dtype == ttnn.bfloat16:
            eq, output = comp_equal(tt_output_tensor, unchunked_input_tensor)
        else:
            eq, output = comp_pcc(tt_output_tensor, unchunked_input_tensor)
        if not eq:
            all_eq = False
            logger.error(f"output mismatch for tensor {i}")
            for w in range(input_shape[0]):
                for z in range(input_shape[1]):
                    for y in range(0, input_shape[2], 32):
                        for x in range(0, input_shape[3], 32):
                            xx = 0
                            yy = 0
                            # for yy in range(32):
                            #     for xx in range(32):
                            if tt_output_tensor[w, z, y + yy, x + xx] != unchunked_input_tensor[w, z, y + yy, x + xx]:
                                logger.error(
                                    f"mismatch at {w}, {z}, {y + yy}, {x + xx}: {tt_output_tensor[w, z, y + yy, x + xx]} != {unchunked_input_tensor[w, z, y + yy, x + xx]}"
                                )
                                # if not reported_mismatch:
                                #     reported_mismatch = True

    assert all_eq, f"{i} FAILED: {output}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", [2])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
)
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "input_shape, input_shard_shape,shard_grid",
    (
        (
            (1, 1, 512, 2048),
            (128, 256),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ),
)
@pytest.mark.parametrize("enable_async", [True])
def test_all_gather_sharded_n300_post_commit(
    all_devices,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_n300_sharded(
        all_devices,
        num_devices,
        input_shape,
        input_shard_shape,
        shard_grid,
        dim,
        num_links,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        # num_cores,
        use_program_cache,
        function_level_defaults,
        all_gather_operation=ttnn.all_gather,
        enable_async=enable_async,
    )

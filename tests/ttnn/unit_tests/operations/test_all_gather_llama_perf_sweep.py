# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000
import itertools
from ttnn import ShardTensorToMesh


def is_unsupported_case(input_shape, dim, mem_config, num_devices, num_links, input_dtype, layout):
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        return True, "Invalid combination"

    if num_devices < 2:
        return True, "Requires multiple devices to run"
    elif num_devices == 2 and num_links <= 2:
        return True, "Not enough links to run"

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


def run_all_gather_sharded(
    t3k_mesh_device,
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
    n_worker,
    n_buffer,
    use_program_cache,
    function_level_defaults,
    all_gather_operation,
    enable_async,
):
    if len(t3k_mesh_device.get_device_ids()) != 8:
        pytest.skip("Not T3000!")

    for device_id in t3k_mesh_device.get_device_ids():
        t3k_mesh_device.get_device(device_id).enable_async(enable_async)

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
    devices = [t3k_mesh_device.get_device(t3k_mesh_device.get_device_ids()[i]) for i in range(num_devices)]

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

    if num_devices < 2:
        pytest.skip("Requires multiple devices to run")
    elif num_devices == 2 and num_links == 2:
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

    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = all_gather_operation(
        input_tensor_mesh,
        dim,
        num_links=num_links,
        memory_config=output_mem_config,
        num_workers=n_worker,
        num_buffers_per_channel=n_buffer,
    )
    for d in devices:
        ttnn.synchronize_device(d)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
    tt_out_tensor = all_gather_operation(
        input_tensor_mesh,
        dim,
        num_links=num_links,
        memory_config=output_mem_config,
        num_workers=n_worker,
        num_buffers_per_channel=n_buffer,
    )
    ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)
    for d in devices:
        ttnn.synchronize_device(d)

    # Run the op
    logger.info("Starting Trace perf test...")
    for i in range(1):
        ttnn.execute_trace(t3k_mesh_device, trace_id, blocking=False)

        for d in devices:
            ttnn.synchronize_device(d)
        logger.info(f"Done iteration {i}")
    ttnn.release_trace(t3k_mesh_device, trace_id)

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


@pytest.mark.timeout(120)
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
# @pytest.mark.parametrize("num_cores", [1])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,  # https://github.com/tenstorrent/tt-metal/issues/9686
        # ttnn.bfloat8_b,
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
        # LLama
        (
            (1, 1, 32, 1024),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ),
)
@pytest.mark.parametrize(
    "n_worker_list,n_buffer_list",
    (
        # LLama
        (
            (2, 4, 8, 10, 12),
            (1, 2, 3, 4, 6, 8),
        ),
    ),
    ids=["sweep_params_1"],
)
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 17068032}], indirect=True)
def test_all_gather_sharded_post_commit(
    t3k_mesh_device,
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
    n_worker_list,
    n_buffer_list,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    for n_worker in n_worker_list:
        for n_buffer in n_buffer_list:
            logger.info(f"Running for n_worker={n_worker}, n_buffer={n_buffer}:")
            run_all_gather_sharded(
                t3k_mesh_device,
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
                n_worker,
                n_buffer,
                use_program_cache,
                function_level_defaults,
                all_gather_operation=ttnn.all_gather,
                enable_async=enable_async,
            )

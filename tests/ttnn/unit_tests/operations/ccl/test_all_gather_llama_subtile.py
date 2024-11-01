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
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import run_all_gather_sharded
from models.utility_functions import nearest_32


def run_all_gather_subtile(
    mesh_device,
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
    all_gather_topology,
    enable_async,
    n_worker=None,
    n_buffer=None,
    num_iter=1,
    trace_mode=False,
):
    numel = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * num_devices
    unchunked_input_shape = list(input_shape)
    unchunked_input_shape[dim] *= num_devices

    if unchunked_input_shape[dim] > 32:
        pytest.skip("Unsupported test case for gathered dim > 32")

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
        output_shard_shape[1] = nearest_32(input_shape[3] * num_devices)
    else:
        output_shard_shape[0] = nearest_32(input_shape[2] * num_devices)
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
        tt_input_tensors_dups.append(
            ttnn.from_torch(
                t,
                dtype=input_dtype,
                layout=tensor_layout,
                device=mesh_device.get_devices()[i],
                memory_config=input_mem_config,
            )
        )

        tt_input_tensors.append(
            ttnn.from_torch(
                t,
                dtype=input_dtype,
                layout=tensor_layout,
                device=mesh_device.get_devices()[i],
                memory_config=input_mem_config,
            )
        )

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

    if trace_mode:
        # Compile Run
        logger.info("Compiling model")
        tt_out_tensor = ttnn.all_gather(
            input_tensor_mesh,
            dim,
            num_links=num_links,
            memory_config=output_mem_config,
            topology=all_gather_topology,
        )
        for d in mesh_device.get_devices():
            ttnn.synchronize_device(d)

        # Capture trace
        logger.info("Capturing trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(num_iter):
            tt_out_tensor = ttnn.all_gather(
                input_tensor_mesh,
                dim,
                num_links=num_links,
                memory_config=output_mem_config,
                topology=all_gather_topology,
            )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        for d in mesh_device.get_devices():
            ttnn.synchronize_device(d)

        # Run the op
        logger.info("Starting Trace perf test...")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.release_trace(mesh_device, trace_id)
        for d in mesh_device.get_devices():
            ttnn.synchronize_device(d)

    else:
        ## Run the actual allgather operation
        for i in range(num_iter):
            tt_out_tensor = ttnn.all_gather(
                input_tensor_mesh,
                dim,
                num_links=num_links,
                memory_config=output_mem_config,
                num_workers=n_worker,
                num_buffers_per_channel=n_buffer,
                topology=all_gather_topology,
            )
        ## Wait for completion
        for d in mesh_device.get_devices():
            ttnn.synchronize_device(d)

    torch.set_printoptions(sci_mode=False)
    all_eq = True
    reported_mismatch = False
    for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
        tt_output_tensor = ttnn.to_torch(t)
        if input_dtype == ttnn.bfloat16:
            eq, output = comp_equal(tt_output_tensor, unchunked_input_tensor)
        else:
            eq, output = comp_pcc(tt_output_tensor, unchunked_input_tensor)

        if not eq:
            all_eq = False
            logger.error(f"output mismatch for tensor {i}")
            logger.error(f"output: {output}")
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
    # breakpoint()
    assert all_eq, f"{i} FAILED: {output}"


@pytest.mark.timeout(120)
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", [2, 4, 8])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
# @pytest.mark.parametrize("num_cores", [1])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.float32,
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
            (1, 1, 8, 128),
            (nearest_32(8), 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        ),
        # LLama
        (
            (1, 1, 1, 192),
            (nearest_32(1), 192),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ),
        # LLama
        (
            (1, 1, 3, 256),
            (nearest_32(3), 256),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ),
        # LLama
        (
            (1, 1, 6, 1056),
            (nearest_32(6), 1056),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ),
        # LLama
        (
            (1, 1, 7, 1280),
            (nearest_32(7), 1280),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ),
    ),
)
@pytest.mark.parametrize("enable_async", [True])
def test_all_gather_subtile_sharded_post_commit(
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
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_subtile(
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
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Linear,
        enable_async=enable_async,
        trace_mode=False,
    )


@pytest.mark.timeout(120)
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", [4])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
# @pytest.mark.parametrize("num_cores", [1])
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
        # LLama
        (
            (1, 1, 8, 128),
            (nearest_32(8), 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        ),
    ),
)
@pytest.mark.parametrize("num_iter", [1000])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 17068032}], indirect=True)
@pytest.mark.parametrize("enable_async", [True])
def test_all_gather_llama_perf(
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
    num_iter,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_subtile(
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
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Linear,
        enable_async=enable_async,
        trace_mode=True,
        num_iter=num_iter,
    )

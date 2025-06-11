# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import is_unsupported_case_t3k
from ttnn.distributed.distributed import ShardTensorToMesh


def sharded_impl(
    device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    use_program_cache,
    function_level_defaults,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    all_gather_topology,
    trace_mode,
    num_iter,
    tile=(32, 32),
):
    if device.get_num_devices() < num_devices:
        pytest.skip("Not T3000!")
    n_worker = None
    n_buffer = None
    unchunked_input_shape = list(input_shape)
    unchunked_input_shape[dim] *= num_devices

    input_tensor = torch.rand(unchunked_input_shape).bfloat16()

    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        input_shard_shape,
        orientation,
    )
    mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)
    # Check if the case is supported for all gather
    (is_known_failure, message) = is_unsupported_case_t3k(
        input_shape, dim, mem_config, num_devices, num_links, input_dtype, tensor_mem_layout, tile
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")
    output_shard_shape = list(input_shard_shape)
    if dim == len(input_shape) - 1:
        output_shard_shape[1] *= num_devices
    else:
        output_shard_shape[0] *= num_devices
    output_shard_spec = ttnn.ShardSpec(
        shard_grid,
        output_shard_shape,
        orientation,
    )

    if num_devices < 2:
        pytest.skip("Requires multiple devices to run")

    input_tensor_mesh = ttnn.from_torch(
        input_tensor,
        device=device,
        dtype=input_dtype,
        layout=tensor_layout,
        mesh_mapper=ShardTensorToMesh(mesh_device=device, dim=dim),
        tile=ttnn.Tile(tile),
    )

    if trace_mode:
        tt_out_tensor = run_with_trace(
            device,
            all_gather_topology,
            input_tensor_mesh,
            dim,
            num_links,
            mem_config,
            n_worker,
            n_buffer,
            num_iter,
        )
    else:
        ## Alternate between barrier and all gather in a loop
        for i in range(num_iter):
            tt_out_tensor = ttnn.all_gather(
                input_tensor_mesh,
                dim,
                num_links=num_links,
                memory_config=mem_config,
                num_workers=n_worker,
                num_buffers_per_channel=n_buffer,
                topology=all_gather_topology,
            )
            ttnn.barrier(
                input_tensor_mesh,
                memory_config=mem_config,
                topology=all_gather_topology,
            )
        ## Wait for completion
        ttnn.synchronize_device(device)


def run_normal(
    device,
    num_devices,
    input_shape,
    dim,
    input_dtype,
    mem_config,
    layout,
    num_iters,
    all_gather_topology,
    tile=(32, 32),
):
    print("Running barrier test")
    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")
    input_tensor = torch.rand(input_shape).bfloat16()
    # Use Async mode based on test input config
    input_tensor_mesh = ttnn.from_torch(
        input_tensor,
        device=device,
        dtype=input_dtype,
        layout=layout,
        mesh_mapper=ShardTensorToMesh(mesh_device=device, dim=dim),
        tile=ttnn.Tile(tile),
    )
    for i in range(num_iters):
        # Run barrier many times in a loop
        ttnn.barrier(
            input_tensor_mesh,
            memory_config=mem_config,
            topology=all_gather_topology,
        )


def run_with_trace(
    device, all_gather_topology, input_tensor_mesh, dim, num_links, output_mem_config, n_worker, n_buffer, num_iter
):
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.all_gather(
        input_tensor_mesh,
        dim,
        num_links=num_links,
        memory_config=output_mem_config,
        num_workers=n_worker,
        num_buffers_per_channel=n_buffer,
        topology=all_gather_topology,
    )
    ttnn.barrier(
        input_tensor_mesh,
        memory_config=output_mem_config,
        topology=all_gather_topology,
    )
    ttnn.synchronize_device(device)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iter):
        # Alternate between barrier and all gather in a loop
        tt_out_tensor = ttnn.all_gather(
            input_tensor_mesh,
            dim,
            num_links=num_links,
            memory_config=output_mem_config,
            num_workers=n_worker,
            num_buffers_per_channel=n_buffer,
            topology=all_gather_topology,
        )
        ttnn.barrier(
            input_tensor_mesh,
            memory_config=output_mem_config,
            topology=all_gather_topology,
        )
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(device, trace_id, blocking=False)
    ttnn.release_trace(device, trace_id)
    ttnn.synchronize_device(device)
    return tt_out_tensor


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices",
    [
        (8),
    ],
)
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "input_shape, output_shard_spec,shard_grid",
    (
        # LLama
        (
            (1, 1, 32, 8192),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ),
)
@pytest.mark.parametrize("all_gather_topology", [ttnn.Topology.Ring])
@pytest.mark.parametrize("num_iters", [1000])
@pytest.mark.parametrize("mem_config", [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)])
def test_run_barrier_impl(
    t3k_mesh_device,
    num_devices,
    input_shape,
    output_shard_spec,
    shard_grid,
    dim,
    input_dtype,
    mem_config,
    layout,
    num_iters,
    all_gather_topology,
):
    if t3k_mesh_device.get_num_devices() < num_devices:
        pytest.skip("Not T3000!")
    run_normal(
        t3k_mesh_device,
        num_devices,
        input_shape,
        dim,
        input_dtype,
        mem_config,
        layout,
        num_iters,
        all_gather_topology,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices",
    [
        (4),
    ],
)
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "input_shape, output_shard_spec,shard_grid",
    (
        # LLama
        (
            (1, 1, 32, 8192),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ),
)
@pytest.mark.parametrize("all_gather_topology", [ttnn.Topology.Ring])
@pytest.mark.parametrize("num_iters", [1000])
@pytest.mark.parametrize("mem_config", [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)])
def test_run_barrier_impl_pcie(
    pcie_mesh_device,
    num_devices,
    input_shape,
    output_shard_spec,
    shard_grid,
    dim,
    input_dtype,
    mem_config,
    layout,
    num_iters,
    all_gather_topology,
):
    if pcie_mesh_device.get_num_devices() < num_devices:
        pytest.skip("Not T3000!")
    run_normal(
        pcie_mesh_device,
        num_devices,
        input_shape,
        dim,
        input_dtype,
        mem_config,
        layout,
        num_iters,
        all_gather_topology,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", [8])
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
    [ttnn.TensorMemoryLayout.WIDTH_SHARDED],
)
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "input_shape, input_shard_shape,shard_grid",
    (
        # LLama
        (
            (1, 1, 32, 1024),
            (32, 256),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ),
)
@pytest.mark.parametrize("trace_mode", [True, False])
@pytest.mark.parametrize("num_iter", [1000])
@pytest.mark.parametrize("all_gather_topology", [ttnn.Topology.Ring])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 7840768}], indirect=True)
def test_barrier_sharded(
    t3k_mesh_device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    use_program_cache,
    function_level_defaults,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    all_gather_topology,
    trace_mode,
    num_iter,
):
    sharded_impl(
        t3k_mesh_device,
        num_devices,
        input_shape,
        input_shard_shape,
        shard_grid,
        dim,
        num_links,
        use_program_cache,
        function_level_defaults,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        all_gather_topology,
        trace_mode,
        num_iter,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices",
    [
        (8),
    ],
)
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "input_shape, output_shard_spec,shard_grid",
    (
        # LLama
        (
            (1, 1, 32, 8192),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ),
)
@pytest.mark.parametrize("all_gather_topology", [ttnn.Topology.Ring])
@pytest.mark.parametrize("num_iters", [1000])
@pytest.mark.parametrize("mem_config", [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)])
@pytest.mark.parametrize("tile_h", [8])
def test_run_barrier_tiny_tile(
    t3k_mesh_device,
    num_devices,
    input_shape,
    output_shard_spec,
    shard_grid,
    dim,
    input_dtype,
    mem_config,
    layout,
    num_iters,
    all_gather_topology,
    tile_h,
):
    if t3k_mesh_device.get_num_devices() < num_devices:
        pytest.skip("Not T3000!")
    run_normal(
        t3k_mesh_device,
        num_devices,
        input_shape,
        dim,
        input_dtype,
        mem_config,
        layout,
        num_iters,
        all_gather_topology,
        tile=(tile_h, 32),
    )

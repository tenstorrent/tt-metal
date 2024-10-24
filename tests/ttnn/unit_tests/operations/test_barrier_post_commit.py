# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.utility_functions import skip_for_grayskull


def is_unsupported_case(topology):
    if topology != ttnn.Topology.Ring:
        return True, "We currently only support RING topology for barrier"
    # if input_dtype == ttnn.bfloat8_b and tuple(input_shape) == (1, 1, 2048, 1024) and scatter_dim == 3:
    #     return True, "Known failure with bfp8_b data format"

    return False, ""


def run_with_trace(
    t3k_mesh_device,
    input_tensor_mesh,
    output_mem_config,
    num_iters,
):
    # Compile Run
    logger.info("Compiling model")
    output_tensor_mesh = ttnn.barrier(
        input_tensor_mesh,
        memory_config=output_mem_config,
    )
    for device_id in t3k_mesh_device.get_device_ids():
        ttnn.synchronize_device(t3k_mesh_device.get_device(device_id))

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
    for i in range(num_iters):
        ttnn.barrier(
            input_tensor_mesh,
            memory_config=output_mem_config,
        )
    ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)
    for device_id in t3k_mesh_device.get_device_ids():
        ttnn.synchronize_device(t3k_mesh_device.get_device(device_id))

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(t3k_mesh_device, trace_id, blocking=False)
    ttnn.release_trace(t3k_mesh_device, trace_id)
    for device_id in t3k_mesh_device.get_device_ids():
        ttnn.synchronize_device(t3k_mesh_device.get_device(device_id))

    return output_tensor_mesh


def run_barrier_test(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    scatter_dim,
    input_dtype,
    layout,
    mem_config,
    enable_async=True,
    num_iters=1,
    topology=ttnn.Topology.Ring,
):
    if len(mesh_device.get_device_ids()) < num_devices:
        pytest.skip(
            f"Not enough devices on machine to implement test case. Wanted {num_devices} but found {len(mesh_device.get_device_ids())}"
        )

    debug = False

    (is_known_failure, message) = is_unsupported_case(
        topology,
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    mesh_device.enable_async(enable_async)
    if enable_async:
        logger.info(f"Using Async Mode for Barrier Op Dispatch")

    logger.info(f"Per chip output shape: {per_chip_output_shape}, devices: {num_devices}, scatter_dim: {scatter_dim}")

    # Generate input tensors
    canonical_input_shape = per_chip_output_shape.copy()
    canonical_input_shape[scatter_dim] *= num_devices
    tt_input_tensors = []

    numel = canonical_input_shape[0] * canonical_input_shape[1] * canonical_input_shape[2] * canonical_input_shape[3]
    input_tensors = [
        torch.rand(canonical_input_shape).bfloat16() if not debug else torch.ones(canonical_input_shape).bfloat16()
        for _ in range(num_devices)
    ]
    if debug:
        input_tensors[-1] = torch.arange(numel).reshape(canonical_input_shape).bfloat16()
    for i, canonical_input_tensor in enumerate(input_tensors):
        tt_input_tensors.append(
            ttnn.Tensor(canonical_input_tensor, input_dtype)
            .to(layout)
            .to(mesh_device.get_device(mesh_device.get_device_ids()[i]), mem_config)
        )

    assert len(tt_input_tensors) == num_devices

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)
    # Run the op
    for i in range(num_iters):
        ttnn.barrier(
            input_tensor_mesh,
            memory_config=mem_config,
            topology=topology,
        )

        for device_id in mesh_device.get_device_ids():
            ttnn.synchronize_device(mesh_device.get_device(device_id))
        logger.info(f"Done iteration {i}")

    # ttnn.visualize_mesh_device(t3k_mesh_device, tensor=output_tensor_mesh)
    # Compute golden
    # TODO: Make it model how reduce scatter actually works for numerical correctness/ordering


# ~2:45 extra time in the current state
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        # (4, 1),
        (8, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, scatter_dim, layout",
    [
        ([1, 2, 256, 32 * 8], 3, ttnn.TILE_LAYOUT),  # Input tensor is (16*32) x (64*32) = 8 * input tensor shape
        ([1, 1, 32, 32 * 8], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("enable_async", [True])
def test_ring_barrier_post_commit(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    scatter_dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    enable_async,
    num_iters=1,
):
    run_barrier_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        scatter_dim,
        input_dtype,
        layout,
        mem_config,
        num_iters=num_iters,
        enable_async=enable_async,
    )


# ~2:45 extra time in the current state
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (8, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, scatter_dim, layout",
    [
        ([1, 1, 32, 32 * 8], 3, ttnn.TILE_LAYOUT),
        ([1, 2, 224, 32 * 8], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("enable_async", [True])
def test_line_barrier_post_commit(
    t3k_mesh_device,
    num_devices,
    num_links,
    per_chip_output_shape,
    scatter_dim,
    input_dtype,
    layout,
    mem_config,
    enable_async,
    num_iters=1,
):
    run_barrier_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        scatter_dim,
        input_dtype,
        layout,
        mem_config,
        num_iters=num_iters,
        enable_async=enable_async,
        topology=ttnn.Topology.Ring,
    )


def run_barrier_sharded_test(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    output_shard_shape,
    scatter_dim,
    shard_grid,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    enable_async=True,
    num_iters=1,
    trace_mode=False,
    topology=ttnn.Topology.Ring,
):
    if len(t3k_mesh_device.get_device_ids()) < num_devices:
        pytest.skip(
            f"Not enough devices on machine to implement test case. Wanted {num_devices} but found {len(t3k_mesh_device.get_device_ids())}"
        )

    debug = False

    t3k_mesh_device.enable_async(enable_async)

    # Generate input tensors
    input_shard_shape = list(output_shard_shape)
    if scatter_dim == 3:
        input_shard_shape[1] *= num_devices
    else:
        input_shard_shape[0] *= num_devices
    tt_input_tensors = []

    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        tuple(input_shard_shape),
        orientation,
        False,
    )

    output_shard_spec = ttnn.ShardSpec(
        shard_grid,
        output_shard_shape,
        orientation,
        False,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)
    output_mem_config = ttnn.MemoryConfig(
        tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
    )

    canonical_input_shape = list(per_chip_output_shape)
    canonical_input_shape[scatter_dim] *= num_devices

    numel = canonical_input_shape[0] * canonical_input_shape[1] * canonical_input_shape[2] * canonical_input_shape[3]
    input_tensors = [
        # torch.rand(canonical_input_shape).bfloat16() if not debug else torch.arange(numel).reshape(canonical_input_shape).bfloat16()
        torch.rand(canonical_input_shape).bfloat16() if not debug else torch.ones(canonical_input_shape).bfloat16()
        for _ in range(num_devices)
    ]
    if debug:
        input_tensors[-1] = torch.arange(numel).reshape(canonical_input_shape).bfloat16()
    for i, canonical_input_tensor in enumerate(input_tensors):
        tt_input_tensors.append(
            ttnn.Tensor(canonical_input_tensor, input_dtype)
            .to(tensor_layout)
            .to(t3k_mesh_device.get_device(t3k_mesh_device.get_device_ids()[i]), input_mem_config)
        )

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)
    trace_mode = False
    # Run the op
    if trace_mode:
        output_tensor_mesh = run_with_trace(
            t3k_mesh_device,
            input_tensor_mesh,
            output_mem_config,
            num_iters,
        )
    else:
        for i in range(num_iters):
            output_tensor_mesh = ttnn.barrier(
                input_tensor_mesh,
                memory_config=output_mem_config,
                topology=topology,
            )

            for device_id in t3k_mesh_device.get_device_ids():
                ttnn.synchronize_device(t3k_mesh_device.get_device(device_id))
            logger.info(f"Done iteration {i}")

    # TODO verify that the tensor is unmodified


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        # (4, 1),
        (8, 1),
    ],
)
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        # ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        # ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape,output_shard_shape,shard_grid",
    (
        # LLama
        (
            (1, 1, 32, 1024),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 4096),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 2048),
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 1792),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
        ),
    ),
)
@pytest.mark.parametrize("enable_async", [True])
def test_width_sharded_barrier_post_commit(
    t3k_mesh_device,
    num_devices,
    num_links,
    per_chip_output_shape,
    output_shard_shape,
    dim,
    shard_grid,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    enable_async,
    num_iters=1,
):
    run_barrier_sharded_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        output_shard_shape,
        dim,
        shard_grid,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        enable_async=enable_async,
        num_iters=num_iters,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.skip("Hangs")
@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (8, 1),
        (4, 2),
        (4, 1),
    ],
)
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ],
)
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.COL_MAJOR])  # Hangs with ROW_MAJOR
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape,output_shard_shape,shard_grid",
    (
        # LLama
        (
            (1, 1, 1024, 32),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ),
)
@pytest.mark.parametrize("enable_async", [True])
def test_height_sharded_barrier_post_commit(
    t3k_mesh_device,
    num_devices,
    num_links,
    per_chip_output_shape,
    output_shard_shape,
    dim,
    shard_grid,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    enable_async,
    num_iters=1,
):
    run_barrier_sharded_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        output_shard_shape,
        dim,
        shard_grid,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        enable_async=enable_async,
        num_iters=num_iters,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (8, 1),
    ],
)
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape,output_shard_shape,shard_grid",
    (
        # LLama
        (
            (1, 1, 256, 512),
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ),
)
@pytest.mark.parametrize("enable_async", [True])
def test_block_sharded_barrier_post_commit(
    t3k_mesh_device,
    num_devices,
    num_links,
    per_chip_output_shape,
    output_shard_shape,
    dim,
    shard_grid,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    enable_async,
    num_iters=1,
):
    run_barrier_sharded_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        output_shard_shape,
        dim,
        shard_grid,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        enable_async=enable_async,
        num_iters=num_iters,
    )

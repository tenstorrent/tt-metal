# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.utility_functions import skip_for_grayskull


def is_unsupported_case(input_shape, dim, math_op, mem_config, num_devices, num_links, input_dtype, layout):
    elem_size = 2 if input_dtype == ttnn.bfloat16 else 1
    tensor_size_bytes = elem_size
    for i in input_shape:
        tensor_size_bytes *= i
    num_l1_banks = 64
    if mem_config.buffer_type == ttnn.BufferType.L1 and tensor_size_bytes > num_l1_banks * 50 * 1024:
        return True, "L1 buffer can't support large tensor sizes"

    # if input_dtype == ttnn.bfloat8_b and tuple(input_shape) == (1, 1, 2048, 1024) and dim == 3:
    #     return True, "Known failure with bfp8_b data format"

    return False, ""


def run_with_trace(
    t3k_mesh_device,
    input_tensor_mesh,
    dim,
    num_links,
    math_op,
    output_mem_config,
    n_worker=None,
    n_buffer=None,
    num_iters=40,
    topology=ttnn.Topology.Ring,
):
    # Compile Run
    logger.info("Compiling model")
    output_tensor_mesh = ttnn.reduce_scatter(
        input_tensor_mesh,
        dim=dim,
        math_op=math_op,
        num_links=num_links,
        memory_config=output_mem_config,
        num_workers=n_worker,
        num_buffers_per_channel=n_buffer,
        topology=topology,
    )
    ttnn.synchronize_device(t3k_mesh_device)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
    for i in range(num_iters):
        output_tensor_mesh = ttnn.reduce_scatter(
            input_tensor_mesh,
            dim=dim,
            math_op=math_op,
            num_links=num_links,
            memory_config=output_mem_config,
            num_workers=n_worker,
            num_buffers_per_channel=n_buffer,
            topology=topology,
        )
    ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(t3k_mesh_device)

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(t3k_mesh_device, trace_id, blocking=False)
    ttnn.release_trace(t3k_mesh_device, trace_id)
    ttnn.synchronize_device(t3k_mesh_device)

    return output_tensor_mesh


def run_reduce_scatter_test(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    num_iters=1,
    topology=ttnn.Topology.Ring,
    trace_mode=False,
):
    if len(mesh_device.get_device_ids()) < num_devices:
        pytest.skip(
            f"Not enough devices on machine to implement test case. Wanted {num_devices} but found {len(mesh_device.get_device_ids())}"
        )

    debug = False

    (is_known_failure, message) = is_unsupported_case(
        per_chip_output_shape, dim, math_op, mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    logger.info(f"Per chip output shape: {per_chip_output_shape}, devices: {num_devices}, dim: {dim}")

    # Generate input tensors
    canonical_input_shape = per_chip_output_shape.copy()
    canonical_input_shape[dim] *= num_devices
    torch_tensor_shape = canonical_input_shape.copy()
    torch_tensor_shape[dim] *= num_devices

    torch_tensor = torch.rand(torch_tensor_shape).bfloat16()
    if debug:
        torch_tensor = torch.arange(torch_tensor.numel()).reshape(canonical_input_shape).bfloat16()
    input_tensors = torch.chunk(torch_tensor, num_devices, dim)
    input_tensor_mesh = ttnn.from_torch(
        torch_tensor,
        dtype=input_dtype,
        layout=layout,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim),
        device=mesh_device,
    )
    # Run the op
    if trace_mode:
        output_tensor_mesh = run_with_trace(
            mesh_device,
            input_tensor_mesh,
            dim,
            num_links,
            math_op,
            mem_config,
            num_iters=num_iters,
            topology=topology,
        )
    else:
        for i in range(num_iters):
            output_tensor_mesh = ttnn.reduce_scatter(
                input_tensor_mesh,
                dim=dim,
                math_op=math_op,
                num_links=num_links,
                memory_config=mem_config,
                topology=topology,
            )

            ttnn.synchronize_device(mesh_device)
            logger.info(f"Done iteration {i}")

    # ttnn.visualize_mesh_device(t3k_mesh_device, tensor=output_tensor_mesh)
    # Compute golden
    # TODO: Make it model how reduce scatter actually works for numerical correctness/ordering
    golden_canonical_out_tensor = torch.zeros(canonical_input_shape).bfloat16()
    for i, t in enumerate(input_tensors):
        golden_canonical_out_tensor = torch.add(golden_canonical_out_tensor, t).bfloat16()

    golden_output_tensors = torch.chunk(golden_canonical_out_tensor, num_devices, dim)

    tt_out_tensors = ttnn.get_device_tensors(output_tensor_mesh)
    logger.info(f"Compare")
    # Compare
    assert len(golden_output_tensors) == len(tt_out_tensors)
    mismatch = False
    for i, t in enumerate(tt_out_tensors):
        tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        eq, output = comp_pcc(tt_output_tensor, golden_output_tensors[i])
        mismatch = mismatch or not eq
        if not eq:
            logger.error(f"output mismatch for tensor {i}. Mesh device ID: {mesh_device.get_device_ids()[i]}")
            if debug:
                for w in range(tt_output_tensor.shape[0]):
                    for z in range(tt_output_tensor.shape[1]):
                        for y in range(tt_output_tensor.shape[2]):
                            for x in range(tt_output_tensor.shape[3]):
                                if tt_output_tensor[w, z, y, x] != golden_output_tensors[i][w, z, y, x]:
                                    logger.error(
                                        f"mismatch at {w}, {z}, {y}, {x}: {tt_output_tensor[w, z, y, x]} != {golden_output_tensors[i][w, z, y, x]}"
                                    )

        else:
            logger.info(f"output match for tensor {i}")
    assert not mismatch, f"{i} FAILED: {output}"


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
    "per_chip_output_shape, dim, layout",
    [
        ([1, 2, 256, 32 * 8], 3, ttnn.TILE_LAYOUT),  # Input tensor is (16*32) x (64*32) = 8 * input tensor shape
        ([1, 1, 32, 32 * 8], 3, ttnn.TILE_LAYOUT),
        ([1, 8, 1024, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 4, 2048, 1024], 3, ttnn.TILE_LAYOUT),
        # # # Has worker slice size warning - defaults to 1x1
        ([1, 1, 128, 8192], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
def test_ring_reduce_scatter_post_commit(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    num_iters=1,
):
    run_reduce_scatter_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
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
    "per_chip_output_shape, dim, layout",
    [
        ([1, 1, 32, 32 * 8], 3, ttnn.TILE_LAYOUT),
        ([1, 2, 224, 32 * 8], 3, ttnn.TILE_LAYOUT),
        ([1, 8, 1024, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 4, 2048, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 8192], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
def test_line_reduce_scatter_post_commit(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    num_iters=1,
):
    run_reduce_scatter_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
        topology=ttnn.Topology.Linear,
    )


# ~2:45 extra time in the current state
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 2),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, dim, layout",
    [
        ([1, 1, 32, 1280], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 1024], 1, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
def test_line_reduce_scatter_post_commit_4chip(
    pcie_mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    num_iters=1,
):
    run_reduce_scatter_test(
        pcie_mesh_device,
        num_devices,
        per_chip_output_shape,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
        topology=ttnn.Topology.Linear,
    )


def run_reduce_scatter_sharded_test(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    output_shard_shape,
    dim,
    num_links,
    math_op,
    shard_grid,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    use_program_cache,
    function_level_defaults,
    in_shard_override=None,
    in_shard_grid_override=None,
    topology=ttnn.Topology.Ring,
    num_iters=1,
    n_worker=None,
    n_buffer=None,
    trace_mode=False,
):
    if len(t3k_mesh_device.get_device_ids()) < num_devices:
        pytest.skip(
            f"Not enough devices on machine to implement test case. Wanted {num_devices} but found {len(t3k_mesh_device.get_device_ids())}"
        )

    logger.info(f"Per chip output shape: {per_chip_output_shape}, devices: {num_devices}, dim: {dim}")

    debug = False

    # Generate input tensors
    if in_shard_grid_override is None:
        assert in_shard_override is None
        in_shard_grid = shard_grid
        input_shard_shape = list(output_shard_shape)
        if dim == 3:
            input_shard_shape[1] *= num_devices
        else:
            input_shard_shape[0] *= num_devices
    else:
        assert in_shard_override is not None
        input_shard_shape = list(in_shard_override)
        in_shard_grid = in_shard_grid_override

    tt_input_tensors = []

    input_shard_spec = ttnn.ShardSpec(
        in_shard_grid,
        tuple(input_shard_shape),
        orientation,
    )

    output_shard_spec = ttnn.ShardSpec(
        shard_grid,
        output_shard_shape,
        orientation,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)
    output_mem_config = ttnn.MemoryConfig(
        tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
    )

    canonical_input_shape = list(per_chip_output_shape)
    canonical_input_shape[dim] *= num_devices

    numel = canonical_input_shape[0] * canonical_input_shape[1] * canonical_input_shape[2] * canonical_input_shape[3]
    input_tensors = [
        # torch.rand(canonical_input_shape).bfloat16() if not debug else torch.arange(numel).reshape(canonical_input_shape).bfloat16()
        torch.rand(canonical_input_shape).bfloat16() if not debug else torch.ones(canonical_input_shape).bfloat16()
        for _ in range(num_devices)
    ]
    if debug:
        input_tensors[-1] = torch.arange(numel).reshape(canonical_input_shape).bfloat16()
    for i, canonical_input_tensor in enumerate(input_tensors):
        tt_input_tensors.append(ttnn.Tensor(canonical_input_tensor, input_dtype).to(tensor_layout))

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors).to(t3k_mesh_device, input_mem_config)

    # Run the op
    if trace_mode:
        output_tensor_mesh = run_with_trace(
            t3k_mesh_device,
            input_tensor_mesh,
            dim,
            num_links,
            math_op,
            output_mem_config,
            n_worker,
            n_buffer,
            num_iters,
        )
    else:
        for i in range(num_iters):
            output_tensor_mesh = ttnn.reduce_scatter(
                input_tensor_mesh,
                dim=dim,
                math_op=math_op,
                num_links=num_links,
                memory_config=output_mem_config,
                topology=topology,
            )

            ttnn.synchronize_device(t3k_mesh_device)
            logger.info(f"Done iteration {i}")

    # Compute golden
    # TODO: Make it model how reduce scatter actually works for numerical correctness/ordering
    golden_canonical_out_tensor = torch.zeros(canonical_input_shape).bfloat16()
    for i, t in enumerate(input_tensors):
        golden_canonical_out_tensor = torch.add(golden_canonical_out_tensor, t).bfloat16()

    golden_output_tensors = torch.chunk(golden_canonical_out_tensor, num_devices, dim)

    tt_out_tensors = ttnn.get_device_tensors(output_tensor_mesh)
    logger.info(f"Compare")
    # Compare
    assert len(golden_output_tensors) == len(tt_out_tensors)
    mismatch = False
    for i, t in enumerate(tt_out_tensors):
        tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        eq, output = comp_pcc(tt_output_tensor, golden_output_tensors[i])
        mismatch = mismatch or not eq
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
        else:
            logger.info(f"output match for tensor {i}")
    assert not mismatch, f"{i} FAILED: {output}"


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
        ttnn.bfloat8_b,
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
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
def test_width_sharded_reduce_scatter_post_commit(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    output_shard_shape,
    dim,
    num_links,
    math_op,
    shard_grid,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    use_program_cache,
    function_level_defaults,
    num_iters=1,
):
    run_reduce_scatter_sharded_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        output_shard_shape,
        dim,
        num_links,
        math_op,
        shard_grid,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        use_program_cache=use_program_cache,
        function_level_defaults=function_level_defaults,
        num_iters=num_iters,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 2),
    ],
)
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
)
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape,output_shard_shape,shard_grid,in_shard_override,in_shard_grid_override",
    (
        # LLama
        (
            (1, 1, 32, 1280),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 1))}),
            (32, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4))}),
        ),
        (
            (1, 1, 32, 1280),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 1))}),
            None,
            None,
        ),
    ),
)
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring, ttnn.Topology.Linear])
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
def test_width_sharded_reduce_scatter_post_commit_4chip(
    pcie_mesh_device,
    num_devices,
    per_chip_output_shape,
    output_shard_shape,
    dim,
    num_links,
    math_op,
    topology,
    shard_grid,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    use_program_cache,
    function_level_defaults,
    in_shard_override,
    in_shard_grid_override,
    num_iters=1,
):
    run_reduce_scatter_sharded_test(
        pcie_mesh_device,
        num_devices,
        per_chip_output_shape,
        output_shard_shape,
        dim,
        num_links,
        math_op,
        shard_grid,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        use_program_cache=use_program_cache,
        function_level_defaults=function_level_defaults,
        in_shard_override=in_shard_override,
        in_shard_grid_override=in_shard_grid_override,
        topology=topology,
        num_iters=num_iters,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.skip("Hangs")
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
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
def test_height_sharded_reduce_scatter_post_commit(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    output_shard_shape,
    dim,
    num_links,
    math_op,
    shard_grid,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    use_program_cache,
    function_level_defaults,
    num_iters=1,
):
    run_reduce_scatter_sharded_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        output_shard_shape,
        dim,
        num_links,
        math_op,
        shard_grid,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        use_program_cache=use_program_cache,
        function_level_defaults=function_level_defaults,
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
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
def test_block_sharded_reduce_scatter_post_commit(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    output_shard_shape,
    dim,
    num_links,
    math_op,
    shard_grid,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    use_program_cache,
    function_level_defaults,
    num_iters=1,
):
    run_reduce_scatter_sharded_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        output_shard_shape,
        dim,
        num_links,
        math_op,
        shard_grid,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        use_program_cache=use_program_cache,
        function_level_defaults=function_level_defaults,
        num_iters=num_iters,
    )

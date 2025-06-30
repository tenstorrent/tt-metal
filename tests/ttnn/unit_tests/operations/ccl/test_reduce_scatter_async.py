# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_reduce_scatter_TG_nightly import (
    run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows,
)


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
    num_iters=40,
    topology=ttnn.Topology.Ring,
    from_remote_semaphore_handles=None,
    to_remote_semaphore_handles=None,
    worker_sub_device_id=None,
):
    # Compile Run
    logger.info("Compiling model")
    output_tensor_mesh = ttnn.experimental.reduce_scatter_async(
        input_tensor_mesh,
        dim=dim,
        from_remote_multi_device_global_semaphore=from_remote_semaphore_handles[0],
        to_remote_multi_device_global_semaphore=to_remote_semaphore_handles[0],
        math_op=math_op,
        num_links=num_links,
        memory_config=output_mem_config,
        topology=topology,
        subdevice_id=worker_sub_device_id,
    )
    ttnn.synchronize_device(t3k_mesh_device)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
    for i in range(num_iters):
        output_tensor_mesh = ttnn.experimental.reduce_scatter_async(
            input_tensor_mesh,
            dim=dim,
            from_remote_multi_device_global_semaphore=from_remote_semaphore_handles[
                i % len(from_remote_semaphore_handles)
            ],
            to_remote_multi_device_global_semaphore=to_remote_semaphore_handles[i % len(to_remote_semaphore_handles)],
            math_op=math_op,
            num_links=num_links,
            memory_config=output_mem_config,
            topology=topology,
            subdevice_id=worker_sub_device_id,
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
    function_level_defaults,
    num_iters,
    input_shard_shape=None,
    shard_grid=None,
    tensor_mem_layout=None,
    topology=ttnn.Topology.Ring,
    trace_mode=False,
):
    assert num_iters > 0
    if len(mesh_device.get_device_ids()) < num_devices:
        pytest.skip(
            f"Not enough devices on machine to implement test case. Wanted {num_devices} but found {len(mesh_device.get_device_ids())}"
        )

    if input_shard_shape and shard_grid:
        input_shard_spec = ttnn.ShardSpec(
            shard_grid,
            input_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        input_mem_config = ttnn.MemoryConfig(
            tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec
        )
        output_shard_shape = list(input_shard_shape)
        if dim == len(per_chip_output_shape) - 1:
            output_shard_shape[1] *= num_devices
        else:
            output_shard_shape[0] *= num_devices
        output_shard_spec = ttnn.ShardSpec(
            shard_grid,
            output_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        output_mem_config = ttnn.MemoryConfig(
            tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
        )
    else:
        assert mem_config is not None
        input_mem_config = mem_config
        output_mem_config = mem_config

    (is_known_failure, message) = is_unsupported_case(
        per_chip_output_shape, dim, math_op, input_mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    # create global semaphore handles
    from_remote_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]
    to_remote_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])
    debug = False

    logger.info(f"Per chip output shape: {per_chip_output_shape}, devices: {num_devices}, dim: {dim}")

    # Generate input tensors
    canonical_input_shape = per_chip_output_shape.copy()
    canonical_input_shape[dim] *= num_devices

    numel = canonical_input_shape[0] * canonical_input_shape[1] * canonical_input_shape[2] * canonical_input_shape[3]
    input_tensors = [
        torch.rand(canonical_input_shape).bfloat16() if not debug else torch.ones(canonical_input_shape).bfloat16()
        for _ in range(num_devices)
    ]
    if debug:
        tile_id = 0
        for w in range(input_tensors[-1].shape[0]):
            for z in range(input_tensors[-1].shape[1]):
                for y in range(0, input_tensors[-1].shape[2], 32):
                    for x in range(0, input_tensors[-1].shape[3], 32):
                        for yy in range(32):
                            for xx in range(32):
                                input_tensors[-1][w, z, y + yy, x + xx] = tile_id
                        tile_id += 1

    input_tensor_mesh = ttnn.from_torch(
        torch.cat(input_tensors),
        dtype=input_dtype,
        layout=layout,
        device=mesh_device,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(0)], ttnn.MeshShape(1, num_devices)),
        ),
    )

    # Run the op
    if trace_mode:
        output_tensor_mesh = run_with_trace(
            mesh_device,
            input_tensor_mesh,
            dim,
            num_links,
            math_op,
            output_mem_config,
            num_iters=num_iters,
            topology=topology,
            from_remote_semaphore_handles=from_remote_semaphore_handles,
            to_remote_semaphore_handles=to_remote_semaphore_handles,
            worker_sub_device_id=worker_sub_device_id,
        )
    else:
        logger.info(f"Running {num_iters} iterations of reduce scatter")
        for i in range(num_iters):
            output_tensor_mesh = ttnn.experimental.reduce_scatter_async(
                input_tensor_mesh,
                dim=dim,
                from_remote_multi_device_global_semaphore=from_remote_semaphore_handles[
                    i % len(from_remote_semaphore_handles)
                ],
                to_remote_multi_device_global_semaphore=to_remote_semaphore_handles[
                    i % len(to_remote_semaphore_handles)
                ],
                math_op=math_op,
                num_links=num_links,
                memory_config=output_mem_config,
                topology=topology,
                subdevice_id=worker_sub_device_id,
            )

            logger.info(f"Waiting for op to finish all iterations")
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done iterations")

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
        logger.info(f"DEVICE {i}")
        logger.info(f"Checking output from device {t.device().id()}")
        tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        eq, output = comp_pcc(tt_output_tensor, golden_output_tensors[i])
        mismatch = mismatch or not eq
        if not eq:
            logger.error(f"output mismatch for tensor {i}. Mesh device ID: {mesh_device.get_device_ids()[i]}")
            if debug:
                logger.info(f"FINAL OUTPUT TENSOR {tt_output_tensor}")
                mismatch_tensor_shape = [
                    tt_output_tensor.shape[0],
                    tt_output_tensor.shape[1],
                    tt_output_tensor.shape[2] // 32,
                    tt_output_tensor.shape[3] // 32,
                ]
                mismatch_tensor = torch.zeros(mismatch_tensor_shape).bfloat16()
                for w in range(tt_output_tensor.shape[0]):
                    for z in range(tt_output_tensor.shape[1]):
                        for y in range(0, tt_output_tensor.shape[2], 32):
                            for x in range(0, tt_output_tensor.shape[3], 32):
                                if tt_output_tensor[w, z, y, x] != golden_output_tensors[i][w, z, y, x]:
                                    mismatch_tensor[w, z, y // 32, x // 32] = 1
                                    logger.error(
                                        f"mismatch at {w}, {z}, {y}, {x}: {tt_output_tensor[w, z, y, x]} != {golden_output_tensors[i][w, z, y, x]}"
                                    )
                logger.error(f"MISMATCH TENSOR {mismatch_tensor}")

        else:
            logger.info(f"output match for tensor {i}")
    mesh_device.reset_sub_device_stall_group()

    assert not mismatch, f"{i} FAILED: {output}"


# ~2:45 extra time in the current state
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, dim, layout",
    [
        ([1, 1, 32, 32], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 32 * 2], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 64, 32], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 64, 64], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 128], 0, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 128], 1, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 128], 2, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 128], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 32], 2, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 64], 2, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 32 * 4], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 128, 4096], 3, ttnn.TILE_LAYOUT),
        ([1, 4, 32, 2304], 2, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 27648, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4_grid")], indirect=True)
def test_line_reduce_scatter_async_post_commit(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    function_level_defaults,
    trace_mode,
    num_iters=16,
):
    run_reduce_scatter_test(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        function_level_defaults,
        num_iters=num_iters,
        topology=ttnn.Topology.Linear,
        trace_mode=trace_mode,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_input_shape, dim, layout",
    [
        (2, 2, [1, 2, 32, 1280], 1, ttnn.TILE_LAYOUT),
        (2, 2, [2, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
        (2, 1, [1, 2, 32, 1280], 1, ttnn.TILE_LAYOUT),
        (2, 1, [2, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
        (2, 2, [1, 1, 32, 1280], 3, ttnn.TILE_LAYOUT),
        (2, 1, [1, 1, 32, 1280], 3, ttnn.TILE_LAYOUT),
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
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("replication_factor", [4])
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4_grid")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_line_reduce_scatter_async_on_T3K_cols_post_commit(
    mesh_device,
    num_devices,
    per_chip_input_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    function_level_defaults,
    replication_factor,
    num_iters=1,
):
    if mesh_device.get_num_devices() < 8:
        pytest.skip("Not T3K!")

    run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        per_chip_input_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        function_level_defaults,
        num_iters=num_iters,
        num_reduce_scatter_instances=replication_factor,
        cluster_axis=0,
        use_reduce_scatter_async=True,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_input_shape, dim, layout",
    [
        (4, 1, [1, 4, 32, 1280], 1, ttnn.TILE_LAYOUT),
        (4, 1, [4, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
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
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("replication_factor", [2])
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4_grid")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_line_reduce_scatter_async_on_T3K_rows_post_commit(
    mesh_device,
    num_devices,
    per_chip_input_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    function_level_defaults,
    replication_factor,
    num_iters=1,
):
    if mesh_device.get_num_devices() < 8:
        pytest.skip("Not T3K!")

    run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        per_chip_input_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        function_level_defaults,
        num_iters=num_iters,
        num_reduce_scatter_instances=replication_factor,
        cluster_axis=1,
        use_reduce_scatter_async=True,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 1),
    ],
)
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
)
@pytest.mark.parametrize("buffer_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize(
    "per_chip_input_shape,input_shard_shape,shard_grid",
    (
        (
            (1, 1, 32, 256),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
        ),
        (
            (1, 1, 32, 128),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ),
        (
            (1, 1, 32, 256),
            (32, 256),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ),
        (
            (1, 1, 32, 1024),
            (32, 1024),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ),
        # LLama
        (
            (1, 1, 32, 4096),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 16384),
            (32, 512),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 8192),
            (32, 256),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (  # https://github.com/tenstorrent/tt-metal/issues/9686
            (1, 1, 32, 7168),
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
        ),
    ),
)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("replication_factor", [1])
@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4_grid")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_line_reduce_scatter_cluster_axis_on_T3K_width_sharded_reduce_scatter_post_commit(
    mesh_device,
    num_devices,
    per_chip_input_shape,
    input_shard_shape,
    dim,
    num_links,
    buffer_type,
    math_op,
    shard_grid,
    orientation,
    input_dtype,
    buffer_layout,
    tensor_mem_layout,
    function_level_defaults,
    replication_factor,
    num_iters=1,
    trace_mode=False,
):
    if mesh_device.get_num_devices() < 8:
        pytest.skip("Not T3K!")

    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        tuple(input_shard_shape),
        orientation,
    )

    run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        per_chip_input_shape,
        tensor_mem_layout,
        dim,
        num_links,
        math_op,
        input_dtype,
        buffer_layout,
        buffer_type,
        function_level_defaults,
        num_iters=num_iters,
        input_shard_spec=input_shard_spec,
        num_reduce_scatter_instances=replication_factor,
        cluster_axis=1,
        trace_mode=trace_mode,
        use_reduce_scatter_async=True,
    )

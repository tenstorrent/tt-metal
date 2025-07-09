# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import is_unsupported_case
from models.utility_functions import skip_for_grayskull, skip_for_blackhole

from ttnn import ShardTensorToMesh, ConcatMeshToTensor


def create_global_semaphores(mesh_device, num_devices, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(3)]
    return ccl_semaphore_handles


def run_reduce_scatter_impl(
    t3k_mesh_device,
    num_devices,
    rs_input_shape,
    dim,
    num_links,
    rs_input_dtype,
    layout,
    mem_config_input,
    mem_config_rs,
    rs_topology,
    num_iters=1,
    enable_trace=True,
    ones_tensor=False,
    mem_config_intermediate=None,
):
    torch.manual_seed(0)

    tile = (32, 32)

    # Set the default config
    if mem_config_intermediate is None:
        mem_config_intermediate = mem_config_rs

    ##### Fabric setup #####
    compute_grid_size = t3k_mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = t3k_mesh_device.create_sub_device_manager([worker_sub_device], 0)
    t3k_mesh_device.load_sub_device_manager(sub_device_manager)
    t3k_mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [
        create_global_semaphores(t3k_mesh_device, num_devices, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    ### Create persistent output buffers
    logger.info("Creating persistent buffers")
    rs_num_batches = rs_input_shape[0]
    single_batch_input_shape = rs_input_shape[:]
    single_batch_input_shape[2] //= rs_num_batches
    persistent_intermediate_buffers = [
        ttnn.from_torch(
            torch.zeros(single_batch_input_shape),
            device=t3k_mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=rs_input_dtype,
            memory_config=mem_config_intermediate,
            mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
        )
        for _ in range(num_iters)
    ]
    rs_output_shape = rs_input_shape[:]
    rs_output_shape[3] //= num_devices
    persistent_output_buffers = [
        ttnn.from_torch(
            torch.zeros(rs_output_shape),
            device=t3k_mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=rs_input_dtype,
            memory_config=mem_config_rs,
            mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
        )
        for _ in range(num_iters)
    ]

    logger.info("Done creating persistent buffers")

    ##### All gather input setup #####
    logger.info(f"Reduce scatter shape: {rs_input_shape}")
    logger.info(f"Reduce scatter dim: {dim}")

    tt_input_tensor_mesh_list = []
    torch_input_tensor_list = []

    for i in range(num_iters):
        rs_global_input_shape = rs_input_shape[:]
        rs_global_input_shape[3] *= num_devices
        if ones_tensor:
            rs_input_tensor = torch.ones(rs_global_input_shape).bfloat16()
        else:
            rs_input_tensor = torch.rand(rs_global_input_shape).bfloat16()
        input_tensors = torch.chunk(rs_input_tensor, num_devices, dim)
        torch_input_tensor_list.append(input_tensors)

        input_tensor_mesh = ttnn.from_torch(
            rs_input_tensor,
            device=t3k_mesh_device,
            layout=layout,
            dtype=rs_input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=ttnn.create_mesh_mapper(
                t3k_mesh_device,
                ttnn.MeshMapperConfig(
                    [ttnn.PlacementReplicate(), ttnn.PlacementShard(dim)], ttnn.MeshShape(1, num_devices)
                ),
            ),
        )

        tt_input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Perform torch ops #####
    torch_reduce_scatter_output_list = []
    for i in range(num_iters):
        reduce_output = torch.sum(torch.stack(torch_input_tensor_list[i]), dim=0)
        scatter_output = torch.chunk(reduce_output, num_devices, dim)
        torch_reduce_scatter_output_list.append(scatter_output)

    ##### Perform the TT ops #####
    tt_reduce_scatter_output_list = []

    def run_op(i):
        tt_reduce_scatter_output_tensor = ttnn.experimental.reduce_scatter_minimal_async(
            tt_input_tensor_mesh_list[i],
            persistent_intermediate_buffer=persistent_intermediate_buffers[i],
            persistent_output_buffer=persistent_output_buffers[i],
            dim=dim,
            multi_device_global_semaphore=ccl_semaphore_handles[i],
            num_links=num_links,
            memory_config=mem_config_rs,
            topology=rs_topology,
            subdevice_id=worker_sub_device_id,
        )

        return tt_reduce_scatter_output_tensor

    if enable_trace:
        # Compile the op
        for i in range(num_iters):
            tt_reduce_scatter_output_tensor = run_op(i)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
        for i in range(num_iters):
            tt_reduce_scatter_output_tensor = run_op(i)
            tt_reduce_scatter_output_list.append(tt_reduce_scatter_output_tensor)
        ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)
        logger.info(f"Done capturing trace")

        # Execute trace
        ttnn.execute_trace(t3k_mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")

        # Synchronize the devices
        ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
    else:
        for i in range(num_iters):
            tt_reduce_scatter_output_tensor = run_op(i)
            tt_reduce_scatter_output_list.append(tt_reduce_scatter_output_tensor)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    for i in range(num_iters):
        tt_rs_out_tensor = tt_reduce_scatter_output_list[i]
        torch_rs_out_tensor = torch_reduce_scatter_output_list[i]

        torch_rs_out = torch.cat(torch_rs_out_tensor, 3)

        tt_rs_out = ttnn.from_device(tt_rs_out_tensor)
        tt_rs_out = ttnn.to_torch(tt_rs_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=3))

        if ones_tensor:
            eq, output = comp_equal(tt_rs_out, torch_rs_out)
        else:
            eq, output = comp_pcc(tt_rs_out, torch_rs_out)

        logger.info(f"{output}, iteration {i}")
        assert eq, f"{i} FAILED ag: {output}"

    t3k_mesh_device.reset_sub_device_stall_group()
    t3k_mesh_device.clear_loaded_sub_device_manager()


@skip_for_grayskull("Requires wormhole_b0 to run")
@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "num_devices, num_links, rs_input_shape, dim, layout, rs_input_dtype",
    [
        # (
        #     8,
        #     1,
        #     [1, 1, 4096, 2560],
        #     3,
        #     ttnn.TILE_LAYOUT,
        #     ttnn.bfloat16,
        #     1,
        # ),  # Full SD3.5 shape, when reduce scatter unfused
        (8, 1, [8, 1, 512, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (8, 1, [4, 1, 1024, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (8, 1, [2, 1, 2048, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (8, 1, [1, 1, 4096, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (8, 1, [1, 1, 512, 256], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, 1, [1, 1, 512, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
    ids=[
        "batch_8",
        "batch_4",
        "batch_2",
        "batch_1",
        "batch_1_slice_wt_1",
        "batch_1_slice_wt_2",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "ones_tensor",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_reduce_scatter_async(
    t3k_mesh_device,
    num_devices,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    num_iters,
    ones_tensor,
    rs_topology,
):
    if t3k_mesh_device.get_num_devices() != 8:
        pytest.skip("Not T3K!")

    run_reduce_scatter_impl(
        t3k_mesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        ones_tensor=ones_tensor,
    )


@skip_for_grayskull("Requires wormhole_b0 to run")
@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "num_devices, num_links, layout, rs_input_dtype",
    [
        (8, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "rs_input_shape, dim, input_shard_shape, input_shard_grid, input_mem_layout, intermediate_shard_shape, intermediate_shard_grid, intermediate_mem_layout, output_shard_shape, output_shard_grid, output_mem_layout",
    [
        (
            [1, 1, 32, 3072],
            3,
            (32, 512),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (32, 512),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            [4, 1, 384, 1024],
            3,
            (256, 1024),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 1024),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (256, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        (
            [4, 1, 384, 3072],
            3,
            (256, 3072),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (384, 512),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (1536, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "ones_tensor",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_reduce_scatter_async_sharded_to_sharded(
    t3k_mesh_device,
    num_devices,
    num_links,
    rs_input_dtype,
    layout,
    rs_input_shape,
    dim,
    input_shard_shape,
    input_shard_grid,
    input_mem_layout,
    intermediate_shard_shape,
    intermediate_shard_grid,
    intermediate_mem_layout,
    output_shard_shape,
    output_shard_grid,
    output_mem_layout,
    enable_trace,
    num_iters,
    ones_tensor,
    rs_topology,
):
    if t3k_mesh_device.get_num_devices() != 8:
        pytest.skip("Not T3K!")

    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    intermediate_shard_spec = ttnn.ShardSpec(
        intermediate_shard_grid,
        intermediate_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_shard_spec = ttnn.ShardSpec(
        output_shard_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config_input = ttnn.MemoryConfig(
        input_mem_layout, buffer_type=ttnn.BufferType.DRAM, shard_spec=input_shard_spec
    )
    mem_config_intermediate = ttnn.MemoryConfig(
        intermediate_mem_layout, buffer_type=ttnn.BufferType.DRAM, shard_spec=intermediate_shard_spec
    )
    mem_config_rs = ttnn.MemoryConfig(output_mem_layout, buffer_type=ttnn.BufferType.DRAM, shard_spec=output_shard_spec)

    run_reduce_scatter_impl(
        t3k_mesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        ones_tensor=ones_tensor,
        mem_config_intermediate=mem_config_intermediate,
    )


@skip_for_grayskull("Requires wormhole_b0 to run")
@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "num_devices, num_links, layout, rs_input_dtype",
    [
        (8, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "rs_input_shape, dim, intermediate_shard_shape, intermediate_shard_grid, intermediate_mem_layout, output_shard_shape, output_shard_grid, output_mem_layout",
    [
        (
            [4, 1, 256, 3072],
            3,
            (256, 512),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (1024, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            [4, 1, 384, 1024],
            3,
            (64, 1024),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (256, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "ones_tensor",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_reduce_scatter_async_interleaved_to_sharded(
    t3k_mesh_device,
    num_devices,
    num_links,
    rs_input_dtype,
    layout,
    rs_input_shape,
    dim,
    intermediate_shard_shape,
    intermediate_shard_grid,
    intermediate_mem_layout,
    output_shard_shape,
    output_shard_grid,
    output_mem_layout,
    enable_trace,
    num_iters,
    ones_tensor,
    rs_topology,
):
    if t3k_mesh_device.get_num_devices() != 8:
        pytest.skip("Not T3K!")

    intermediate_shard_spec = ttnn.ShardSpec(
        intermediate_shard_grid,
        intermediate_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_shard_spec = ttnn.ShardSpec(
        output_shard_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config_input = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    mem_config_intermediate = ttnn.MemoryConfig(
        intermediate_mem_layout, buffer_type=ttnn.BufferType.DRAM, shard_spec=intermediate_shard_spec
    )
    mem_config_rs = ttnn.MemoryConfig(output_mem_layout, buffer_type=ttnn.BufferType.DRAM, shard_spec=output_shard_spec)

    run_reduce_scatter_impl(
        t3k_mesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        ones_tensor=ones_tensor,
        mem_config_intermediate=mem_config_intermediate,
    )


@skip_for_grayskull("Requires wormhole_b0 to run")
@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "num_devices, num_links, layout, rs_input_dtype",
    [
        (8, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "rs_input_shape, dim, input_shard_shape, input_shard_grid, input_mem_layout",
    [
        (
            [4, 1, 256, 3072],
            3,
            (1024, 512),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            [4, 1, 384, 1024],
            3,
            (256, 1024),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "ones_tensor",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_reduce_scatter_async_sharded_to_interleaved(
    t3k_mesh_device,
    num_devices,
    num_links,
    rs_input_dtype,
    layout,
    rs_input_shape,
    dim,
    input_shard_shape,
    input_shard_grid,
    input_mem_layout,
    enable_trace,
    num_iters,
    ones_tensor,
    rs_topology,
):
    if t3k_mesh_device.get_num_devices() != 8:
        pytest.skip("Not T3K!")

    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config_input = ttnn.MemoryConfig(
        input_mem_layout, buffer_type=ttnn.BufferType.DRAM, shard_spec=input_shard_spec
    )
    mem_config_intermediate = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    mem_config_rs = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    run_reduce_scatter_impl(
        t3k_mesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        ones_tensor=ones_tensor,
        mem_config_intermediate=mem_config_intermediate,
    )

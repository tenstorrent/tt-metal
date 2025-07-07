# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import is_unsupported_case

from ttnn import ShardTensorToMesh, ConcatMeshToTensor


def create_global_semaphores(t3k_mesh_device, num_devices, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(t3k_mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


def run_all_gather_impl(
    t3k_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    all_gather_topology,
    num_iters=1,
    enable_trace=True,
):
    torch.manual_seed(0)

    tile = (32, 32)

    # Skip unsupported cases
    (is_known_failure, message) = is_unsupported_case(
        ag_output_shape, dim, mem_config_ag, num_devices, num_links, ag_input_dtype, layout, tile
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    ##### All gather setup #####
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
    persistent_output_buffers = [
        ttnn.from_torch(
            torch.zeros(ag_output_shape),
            device=t3k_mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ag_input_dtype,
            memory_config=mem_config_ag,
            mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
        )
        for _ in range(num_iters)
    ]

    logger.info("Done creating persistent buffers")

    ##### All gather input setup #####
    logger.info(f"All gather output shape: {ag_output_shape}")
    logger.info(f"All gather dim: {dim}")

    input_tensor_mesh_list = []
    ag_output_tensor_goldens_list = []

    for i in range(num_iters):
        ag_output_tensor = torch.rand(ag_output_shape).bfloat16()
        ag_output_tensor_goldens_list.append(ag_output_tensor)

        input_tensor_mesh = ttnn.from_torch(
            ag_output_tensor,
            device=t3k_mesh_device,
            layout=layout,
            dtype=ag_input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=ttnn.ShardTensorToMesh(t3k_mesh_device, dim=dim),
        )

        input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Perform the TT ops #####
    tt_all_gather_out_tensor_list = []

    def run_op(i):
        tt_all_gather_out_tensor = ttnn.experimental.all_gather_async(
            input_tensor_mesh_list[i],
            persistent_output_buffer=persistent_output_buffers[i],
            dim=dim,
            multi_device_global_semaphore=ccl_semaphore_handles[i],
            num_links=num_links,
            memory_config=mem_config_ag,
            topology=all_gather_topology,
            subdevice_id=worker_sub_device_id,
        )

        return tt_all_gather_out_tensor

    if enable_trace:
        # Compile the op
        tt_all_gather_out_tensor = run_op(0)
        ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
        tt_all_gather_out_tensor = run_op(0)
        ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done capturing trace")

        # Execute trace
        for i in range(num_iters):
            ttnn.execute_trace(t3k_mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensor)
        logger.info(f"Done executing trace")
    else:
        for i in range(num_iters):
            tt_all_gather_out_tensor = run_op(i)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensor)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    for i in range(num_iters):
        tt_ag_out_tensor = tt_all_gather_out_tensor_list[i]
        torch_ag_out_tensor = ag_output_tensor_goldens_list[i if not enable_trace else 0]

        tt_ag_out = ttnn.from_device(tt_ag_out_tensor)
        tt_ag_out = ttnn.to_torch(tt_ag_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=3))
        tt_ag_out = tt_ag_out[:, :, :, 0 : torch_ag_out_tensor.shape[3]]
        eq, output = comp_pcc(tt_ag_out, torch_ag_out_tensor, 1)
        logger.info(f"{output}, iteration {i}")
        assert eq, f"{i} FAILED ag: {output}"

    t3k_mesh_device.reset_sub_device_stall_group()
    t3k_mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.parametrize(
    "num_devices, num_links, ag_output_shape, dim, layout, ag_input_dtype",
    [
        (8, 1, [1, 1, 4096, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace,num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_async_interleaved_to_interleaved(
    t3k_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    all_gather_topology,
    num_iters,
):
    run_all_gather_impl(
        t3k_mesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
    )


@pytest.mark.parametrize(
    "num_devices, num_links, layout, ag_input_dtype",
    [
        (8, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "ag_output_shape, dim, input_shard_shape, input_shard_grid, input_mem_layout, output_shard_shape, output_shard_grid, output_mem_layout",
    [
        (
            [1, 1, 32, 3072],
            3,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (32, 512),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            [1, 1, 384, 1024],
            3,
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 1024),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        (
            [1, 1, 384, 3072],
            3,
            (64, 384),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (384, 512),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize(
    "enable_trace,num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_async_sharded_to_sharded(
    t3k_mesh_device,
    num_devices,
    num_links,
    layout,
    ag_input_dtype,
    ag_output_shape,
    dim,
    input_shard_shape,
    input_shard_grid,
    input_mem_layout,
    output_shard_shape,
    output_shard_grid,
    output_mem_layout,
    enable_trace,
    all_gather_topology,
    num_iters,
):
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
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
    mem_config_ag = ttnn.MemoryConfig(output_mem_layout, buffer_type=ttnn.BufferType.DRAM, shard_spec=output_shard_spec)

    run_all_gather_impl(
        t3k_mesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
    )


@pytest.mark.parametrize(
    "num_devices, num_links, layout, ag_input_dtype",
    [
        (8, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "ag_output_shape, dim, input_shard_shape, input_shard_grid, input_mem_layout",
    [
        (
            [1, 1, 32, 3072],
            3,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            [1, 1, 384, 1024],
            3,
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize(
    "enable_trace,num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_async_sharded_to_interleaved(
    t3k_mesh_device,
    num_devices,
    num_links,
    layout,
    ag_input_dtype,
    ag_output_shape,
    dim,
    input_shard_shape,
    input_shard_grid,
    input_mem_layout,
    enable_trace,
    all_gather_topology,
    num_iters,
):
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config_input = ttnn.MemoryConfig(
        input_mem_layout, buffer_type=ttnn.BufferType.DRAM, shard_spec=input_shard_spec
    )
    mem_config_ag = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    run_all_gather_impl(
        t3k_mesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
    )


@pytest.mark.parametrize(
    "num_devices, num_links, layout, ag_input_dtype",
    [
        (8, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "ag_output_shape, dim, output_shard_shape, output_shard_grid, output_mem_layout",
    [
        (
            [1, 1, 32, 3072],
            3,
            (32, 512),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            [1, 1, 384, 1024],
            3,
            (64, 1024),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize(
    "enable_trace,num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_async_interleaved_to_sharded(
    t3k_mesh_device,
    num_devices,
    num_links,
    layout,
    ag_input_dtype,
    ag_output_shape,
    dim,
    output_shard_shape,
    output_shard_grid,
    output_mem_layout,
    enable_trace,
    all_gather_topology,
    num_iters,
):
    output_shard_spec = ttnn.ShardSpec(
        output_shard_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config_input = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    mem_config_ag = ttnn.MemoryConfig(output_mem_layout, buffer_type=ttnn.BufferType.DRAM, shard_spec=output_shard_spec)

    run_all_gather_impl(
        t3k_mesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
    )

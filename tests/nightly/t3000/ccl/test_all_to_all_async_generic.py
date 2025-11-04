# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc


def run_with_trace(
    mesh_device,
    topology,
    input_tensor,
    persistent_output_buffer,
    in_dim,
    out_dim,
    num_links,
    output_mem_config,
    num_iter=20,
    subdevice_id=None,
    cluster_axis=None,
):
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.experimental.all_to_all_async_generic(
        input_tensor,
        in_dim=in_dim,
        out_dim=out_dim,
        persistent_output_buffer=persistent_output_buffer,
        num_links=num_links,
        memory_config=output_mem_config,
        topology=topology,
        subdevice_id=subdevice_id,
        cluster_axis=cluster_axis,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.experimental.all_to_all_async_generic(
            input_tensor,
            in_dim=in_dim,
            out_dim=out_dim,
            persistent_output_buffer=persistent_output_buffer,
            num_links=num_links,
            memory_config=output_mem_config,
            topology=topology,
            subdevice_id=subdevice_id,
            cluster_axis=cluster_axis,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    return tt_out_tensor


def run_all_to_all_impl(
    mesh_device,
    num_devices,
    logical_shape,
    in_dim,
    out_dim,
    num_links,
    dtype,
    layout,
    topology,
    num_iters=1,
    mem_config=None,
    trace_mode=False,
    do_check=True,
    reuse_inputs=False,
    cluster_axis=None,
):
    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
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
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    logger.info(f"Logical shape: {logical_shape}")
    logger.info(f"in_dim: {in_dim}")
    logger.info(f"out_dim: {out_dim}")

    ###

    ### Create persistent output buffers
    logger.info("Creating persistent buffers")
    output_shape = list(logical_shape)
    output_shape[out_dim] //= num_devices

    persistent_output_buffers = [
        ttnn.from_torch(
            torch.zeros(output_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(num_iters)
    ]

    logger.info("Done creating persistent buffers")

    input_tensor_mesh_list = []
    output_tensor_goldens_list = []

    for i in range(num_iters if not reuse_inputs else 1):
        output_tensor = torch.rand(logical_shape).bfloat16()
        output_tensor_golden = torch.chunk(output_tensor, num_devices, out_dim)

        if cluster_axis == 0:
            output_tensor_golden_transposed = []
            for i in range(len(output_tensor_golden)):
                col = i // mesh_device.shape[1]
                row = i % mesh_device.shape[1]
                index = row * mesh_device.shape[0] + col
                output_tensor_golden_transposed.append(output_tensor_golden[index])
            output_tensor_golden = output_tensor_golden_transposed

        output_tensor_goldens_list.append(output_tensor_golden)
        if cluster_axis == 0:
            mesh_config = ttnn.MeshMapperConfig(
                [ttnn.PlacementShard(in_dim), ttnn.PlacementShard(out_dim)], mesh_device.shape
            )
        else:
            mesh_config = ttnn.MeshMapperConfig(
                [ttnn.PlacementShard(out_dim), ttnn.PlacementShard(in_dim)], mesh_device.shape
            )
        input_tensor_mesh = ttnn.from_torch(
            output_tensor,
            device=mesh_device,
            layout=layout,
            dtype=dtype,
            memory_config=mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_config),
        )

        input_tensor_mesh_list.append(input_tensor_mesh)

    tt_out_tensor_list = []
    if trace_mode:
        tt_out_tensor = run_with_trace(
            mesh_device,
            topology,
            input_tensor_mesh_list[0],
            persistent_output_buffers[0],
            in_dim,
            out_dim,
            num_links,
            mem_config,
            num_iter=num_iters,
            subdevice_id=worker_sub_device_id,
            cluster_axis=cluster_axis,
        )
        tt_out_tensor_list.append(tt_out_tensor)
    else:
        for i in range(num_iters):
            tt_out_tensor = ttnn.experimental.all_to_all_async_generic(
                input_tensor_mesh_list[i if not reuse_inputs else 0],
                in_dim=in_dim,
                out_dim=out_dim,
                num_links=num_links,
                memory_config=mem_config,
                topology=topology,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
            )
            tt_out_tensor_list.append(tt_out_tensor)

        logger.info(f"Waiting for op")
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done op")

    passed = True
    if do_check:
        for tensor_index in range(len(tt_out_tensor_list)):
            tt_out_tensor = tt_out_tensor_list[tensor_index]
            output_tensors = output_tensor_goldens_list[tensor_index if not reuse_inputs else 0]
            tt_output_tensor = ttnn.to_torch(
                tt_out_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=out_dim)
            )
            output_tensor = torch.cat(output_tensors, dim=out_dim)
            logger.info(f"Checking output for iteration {tensor_index}")
            if dtype == ttnn.bfloat16:
                eq, output = comp_equal(tt_output_tensor, output_tensor)
            else:
                eq, output = comp_pcc(tt_output_tensor, output_tensor)
            if not eq:
                logger.error(f"output mismatch for tensor {tensor_index}: {output}")
                passed = False

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    if do_check:
        assert passed, f"FAILED: output mismatch"


@pytest.mark.parametrize(
    "mesh_device, cluster_axis, logical_shape, in_dim, out_dim",
    [
        ((1, 8), None, [1, 128, 128, 512], 1, 2),  # test 1-2
        ((2, 4), 1, [1, 32, 128, 576], 2, 1),  # test 2-1
        ((2, 4), 1, [3, 1, 256, 768], 2, 3),  # test 2-3
        ((2, 4), 0, [1, 2, 256, 768], 3, 2),  # test 3-2
    ],
    ids=["test1-2:mesh1x8", "test2-1:mesh2x4:1", "test2-3:mesh2x4:1", "test3-2:mesh2x4:0"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 100000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_all_to_all(
    mesh_device,
    logical_shape,
    in_dim,
    out_dim,
    function_level_defaults,
    is_ci_env,
    cluster_axis,
):
    run_all_to_all_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        logical_shape,
        in_dim,
        out_dim,
        num_links=1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        topology=ttnn.Topology.Linear,
        num_iters=2,
        mem_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        do_check=True,
        trace_mode=False,
        reuse_inputs=False,
        cluster_axis=cluster_axis,
    )

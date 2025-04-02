# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
    create_global_semaphore_with_same_address,
)

from tests.ttnn.unit_tests.operations.ccl.test_new_all_reduce import (
    SUB_DEVICE_CRS,
    QKV_CRS,
    RING_CRS,
    FF1_CRS,
    FF1_CRS_RS_OUT,
    NORM_CRS,
    check_mesh_tensor_alloc,
)

PACKET_WORKER_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
        ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
    ]
)


def gen_tensor(dim, shard_height, shard_width, num_devices_scatter, num_devices_fracture, num_cores, scheme="random"):
    factor = 0
    torch_fracture_tensors = []
    for _ in range(num_devices_fracture):
        torch_scatter_tensors = []
        for _ in range(num_devices_scatter):
            torch_input_tensors = []
            for _ in range(num_cores):
                for _ in range(shard_width // 32):
                    if scheme == "random":
                        torch_input_tensors.append(torch.rand(1, 1, shard_height, 32))
                    elif scheme == "sequential":
                        torch_input_tensors.append(torch.ones(1, 1, shard_height, 32) * factor)
                        factor += 1
                    else:
                        raise ValueError(f"Invalid scheme: {scheme}")
            torch_scatter_tensors.append(torch.cat(torch_input_tensors, dim=dim))

        torch_fracture_tensors.append(torch.cat(torch_scatter_tensors, dim=1))

    return torch.cat(torch_fracture_tensors, dim=0)


def run_reduce_scatter_test(
    mesh_device,
    dim,
    shard_height,
    shard_width,
    num_devices_scatter,
    num_devices_fracture,
    num_cores,
    num_iters,
    trace_mode,
    num_links=3,
    scheme="random",
):
    mesh_device.enable_async(True)
    mesh_device.enable_program_cache()
    num_pages_per_packet = 4

    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            RING_CRS,
            [shard_height, shard_width],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            FF1_CRS_RS_OUT,
            [shard_height, 32],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # have to allocate a tensor for buffers that are written to across devices as there's no way to synchronize lifetime of buffers across devices
    packet_workers_persistent_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            SUB_DEVICE_CRS,
            [shard_height, num_devices_scatter * num_pages_per_packet * 32],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    output_tensor_goldens_list = []
    tt_input_tensors_list = []
    tt_intermediate_tensors_list = []
    for _ in range(num_iters):
        input = gen_tensor(
            dim, shard_height, shard_width, num_devices_scatter, num_devices_fracture, num_cores, scheme=scheme
        )

        intermediate_tensor = torch.zeros(
            [
                num_devices_fracture,
                num_devices_scatter,
                shard_height,
                num_devices_scatter
                * num_pages_per_packet
                * 32
                * packet_workers_persistent_mem_config.shard_spec.num_cores(),
            ]
        )

        intermediate_outputs = torch.chunk(input, chunks=num_devices_scatter, dim=1)
        output = torch.zeros(intermediate_outputs[0].shape)

        for i in range(0, len(intermediate_outputs)):
            output += intermediate_outputs[i]

        scattered_output = torch.chunk(output, chunks=num_devices_scatter, dim=dim)
        scattered_output = torch.cat(scattered_output, dim=1)

        output_tensor_goldens_list.append(scattered_output)

        tt_input = ttnn.from_torch(
            input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=sharded_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, dims=(0, 1), mesh_shape=[num_devices_fracture, num_devices_scatter]
            ),
        )
        tt_intermediate = ttnn.from_torch(
            intermediate_tensor,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=packet_workers_persistent_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, dims=(0, 1), mesh_shape=[num_devices_fracture, num_devices_scatter]
            ),
        )
        check_mesh_tensor_alloc(tt_input)
        check_mesh_tensor_alloc(tt_intermediate)
        tt_input_tensors_list.append(tt_input)
        tt_intermediate_tensors_list.append(tt_intermediate)

    enable_persistent_fabric = True
    ccl_sub_device_crs = SUB_DEVICE_CRS
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
        mesh_device,
        [worker_sub_device],
        0,
        0,
        enable_persistent_fabric,
    )
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [
        create_global_semaphore_with_same_address(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    tt_out_tensor_list = []
    if trace_mode:
        tt_out_tensor = ttnn.experimental.llama_reduce_scatter(
            tt_input_tensors_list[0],
            tt_intermediate_tensors_list[0],
            dim,
            ccl_semaphore_handles[0],
            worker_sub_device_id,
            cluster_axis=1,
            mesh_device=mesh_device,
            num_links=num_links,
            memory_config=output_mem_config,
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for iter in range(num_iters):
            tt_out_tensor = ttnn.experimental.llama_reduce_scatter(
                tt_input_tensors_list[0],
                tt_intermediate_tensors_list[0],
                dim,
                ccl_semaphore_handles[0],
                worker_sub_device_id,
                cluster_axis=1,
                mesh_device=mesh_device,
                num_links=num_links,
                memory_config=output_mem_config,
            )

        tt_out_tensor_list.append(tt_out_tensor)

        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.synchronize_device(mesh_device)
    else:
        for i in range(num_iters):
            tt_out_tensor = ttnn.experimental.llama_reduce_scatter(
                tt_input_tensors_list[i],
                tt_intermediate_tensors_list[i],
                dim,
                ccl_semaphore_handles[i],
                worker_sub_device_id,
                cluster_axis=1,
                mesh_device=mesh_device,
                num_links=num_links,
                memory_config=output_mem_config,
            )
            tt_out_tensor_list.append(tt_out_tensor)
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    mesh_device.reset_sub_device_stall_group()
    teardown_fabric_interface(mesh_device)

    passed = True
    first_failed_tensor_index = None
    failed_indices = []
    for tensor_index in range(len(tt_out_tensor_list)):
        tt_torch_tensor = ttnn.to_torch(
            tt_out_tensor_list[tensor_index],
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, mesh_shape=[num_devices_fracture, num_devices_scatter], dims=(0, 1)
            ),
        )
        eq, output_results = comp_pcc(tt_torch_tensor, output_tensor_goldens_list[tensor_index], 0.999)
        logger.info(f"Output tensor {tensor_index} has result {output_results}")
        if not eq:
            passed = False
            first_failed_tensor_index = tensor_index
            failed_indices = torch.where(tt_torch_tensor != output_tensor_goldens_list[tensor_index])
            break

    for i in range(num_devices_scatter * num_devices_fracture):
        logger.info(f"Device {i} has {mesh_device.get_devices()[i].num_program_cache_entries()} program cache entries")
        assert (
            mesh_device.get_devices()[i].num_program_cache_entries() == 1
            or mesh_device.get_devices()[i].num_program_cache_entries() == num_iters
        ), f"Device {i} has {mesh_device.get_devices()[i].num_program_cache_entries()} program cache entries"

    if not passed:
        logger.info(f"Failed indices: {failed_indices}")
        assert eq, f"{first_failed_tensor_index} FAILED: {output_results}"


@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 90000, "dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_fabric_reduce_scatter_tg_trace(mesh_device, trace_mode):
    dim = 3
    shard_height = 32
    shard_width = 160
    num_devices_scatter = 4
    num_devices_fracture = 8
    num_cores = 24
    num_iters = 30
    trace_mode = trace_mode

    run_reduce_scatter_test(
        mesh_device,
        dim,
        shard_height,
        shard_width,
        num_devices_scatter,
        num_devices_fracture,
        num_cores,
        num_iters,
        trace_mode,
        num_links=3,
        scheme="random",
    )


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_fabric_reduce_scatter_tg_no_trace(mesh_device, trace_mode):
    dim = 3
    shard_height = 32
    shard_width = 160
    num_devices_scatter = 4
    num_devices_fracture = 8
    num_cores = 24
    num_iters = 30
    trace_mode = trace_mode

    run_reduce_scatter_test(
        mesh_device,
        dim,
        shard_height,
        shard_width,
        num_devices_scatter,
        num_devices_fracture,
        num_cores,
        num_iters,
        trace_mode,
        num_links=3,
        scheme="random",
    )

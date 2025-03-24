# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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
)


def gen_tensor(dim, shard_height, shard_width, num_devices, num_cores, scheme="random"):
    torch_input_tensors = []
    factor = 0
    for _ in range(num_devices):
        for _ in range(num_cores):
            for _ in range(shard_width // 32):
                if scheme == "random":
                    torch_input_tensors.append(torch.rand(1, 1, shard_height, 32))
                elif scheme == "sequential":
                    torch_input_tensors.append(torch.ones(1, 1, shard_height, 32) * factor)
                    factor += 1
                else:
                    raise ValueError(f"Invalid scheme: {scheme}")
                # factor += 1

    return torch.cat(torch_input_tensors, dim=dim)


def run_reduce_scatter_test(
    mesh_device,
    dim,
    shard_height,
    shard_width,
    num_devices,
    num_cores,
    num_iters,
    trace_mode,
    num_links=1,
    scheme="random",
):
    mesh_device.enable_async(True)
    mesh_device.enable_program_cache()

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

    # print(sharded_mem_config)
    # print("--------------------------------")
    # print(output_mem_config)

    output_tensor_goldens_list = []
    tt_input_tensors_list = []
    for _ in range(num_iters):
        input = gen_tensor(dim, shard_height, shard_width, num_devices, num_cores, scheme=scheme)

        intermediate_outputs = torch.chunk(input, chunks=num_devices, dim=3)
        output = torch.zeros(intermediate_outputs[0].shape)
        for i in range(0, len(intermediate_outputs)):
            output += intermediate_outputs[i]

        output_tensor_goldens_list.append(output)
        input_tensors_per_device = []
        for j, t in enumerate(intermediate_outputs):
            input_tensors_per_device.append(
                ttnn.Tensor(t, ttnn.bfloat8_b)
                .to(ttnn.TILE_LAYOUT)
                .to(mesh_device.get_devices()[j], mem_config=sharded_mem_config)
            )

        tt_input = ttnn.aggregate_as_tensor(input_tensors_per_device)
        tt_input_tensors_list.append(tt_input)

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
            dim,
            ccl_semaphore_handles[0],
            worker_sub_device_id,
            cluster_axis=1,
            num_links=num_links,
            memory_config=output_mem_config,
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(num_iters):
            tt_out_tensor = ttnn.experimental.llama_reduce_scatter(
                tt_input_tensors_list[0],
                dim,
                ccl_semaphore_handles[0],
                worker_sub_device_id,
                cluster_axis=1,
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
                dim,
                ccl_semaphore_handles[i],
                worker_sub_device_id,
                cluster_axis=1,
                num_links=num_links,
                memory_config=output_mem_config,
            )
            tt_out_tensor_list.append(tt_out_tensor)
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    mesh_device.reset_sub_device_stall_group()
    teardown_fabric_interface(mesh_device)

    passed = True
    for tensor_index in range(len(tt_out_tensor_list)):
        tt_torch_tensor = ttnn.to_torch(
            tt_out_tensor_list[tensor_index], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim)
        )
        eq, output_results = comp_pcc(tt_torch_tensor, output_tensor_goldens_list[tensor_index])
        logger.info(f"Output tensor {tensor_index} has result {output_results}")
        if not eq:
            passed = False
            break

    for i in range(num_devices):
        assert (
            mesh_device.get_devices()[i].num_program_cache_entries() == 1
            or mesh_device.get_devices()[i].num_program_cache_entries() == num_iters
        ), f"Device {i} has {mesh_device.get_devices()[i].num_program_cache_entries()} program cache entries"

    if not passed:
        assert eq, f"{i} FAILED: {output_results}"


@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 40960, "dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True
)
def test_fabric_reduce_scatter_tg_trace(mesh_device):
    torch.manual_seed(2005)
    dim = 3
    shard_height = 32
    shard_width = 160
    num_devices = 4
    num_cores = 24
    num_iters = 5
    trace_mode = True

    run_reduce_scatter_test(
        mesh_device, dim, shard_height, shard_width, num_devices, num_cores, num_iters, trace_mode, num_links=3
    )


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_fabric_reduce_scatter_tg_no_trace(mesh_device):
    torch.manual_seed(2005)
    dim = 3
    shard_height = 32
    shard_width = 160
    num_devices = 4
    num_cores = 24
    num_iters = 5
    trace_mode = False

    run_reduce_scatter_test(
        mesh_device,
        dim,
        shard_height,
        shard_width,
        num_devices,
        num_cores,
        num_iters,
        trace_mode,
        num_links=3,
        scheme="sequential",
    )

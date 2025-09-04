# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
import os
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_blackhole

from ttnn import ShardTensorToMesh, ConcatMeshToTensor


def run_neighbor_pad_impl(
    t3k_mesh_device,
    num_devices,
    input_shape,
    dim,
    padding_left,
    padding_right,
    num_links,
    input_dtype,
    layout,
    mem_config_input,
    mem_config_output,
    enable_trace,
    neighbor_pad_topology,
    num_iters,
):
    torch.manual_seed(0)

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
        ttnn.create_global_semaphore(t3k_mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    barrier_semaphore_handles = [
        ttnn.create_global_semaphore(t3k_mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    ##### Neighbor pad input setup #####
    logger.info(f"Neighbor pad input shape: {input_shape}")
    logger.info(f"Neighbor pad dim: {dim}")

    input_tensor_mesh_list = []
    np_output_tensor_goldens_list = []

    for i in range(num_iters):
        input_tensor = torch.rand(input_shape).bfloat16()
        chunks = torch.chunk(input_tensor, num_devices, dim)
        np_output_tensor = []
        # pad left
        first_slice_front = torch.narrow(chunks[0], dim, 0, 1)
        first_slice = torch.cat((first_slice_front, chunks[0]), dim=dim)
        np_output_tensor.append(first_slice)
        for p in range(padding_left - 1):
            np_output_tensor[0] = torch.cat((first_slice_front, np_output_tensor[0]), dim=dim)
        for k in range(1, num_devices):
            prev_halo = torch.narrow(chunks[k - 1], dim, chunks[k - 1].shape[dim] - padding_left, padding_left)
            np_output_tensor.append(torch.cat((prev_halo, chunks[k]), dim=dim))

        # pad right
        last_slice_size = np_output_tensor[num_devices - 1].shape[dim]
        last_slice_back = torch.narrow(np_output_tensor[num_devices - 1], dim, last_slice_size - 1, 1)
        for p in range(padding_right):
            np_output_tensor[num_devices - 1] = torch.cat((np_output_tensor[num_devices - 1], last_slice_back), dim=dim)
        for k in range(0, num_devices - 1):
            next_halo = torch.narrow(chunks[k + 1], dim, 0, padding_right)
            np_output_tensor[k] = torch.cat((np_output_tensor[k], next_halo), dim=dim)
        np_output_tensor_goldens_list.append(torch.cat(np_output_tensor, dim=dim))

        input_tensor_mesh = ttnn.from_torch(
            input_tensor,
            device=t3k_mesh_device,
            layout=layout,
            dtype=input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=ttnn.ShardTensorToMesh(t3k_mesh_device, dim=dim),
        )

        input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Perform the TT ops #####
    tt_neighbor_pad_out_tensor_list = []

    def run_op(i):
        tt_neighbor_pad_out_tensor = ttnn.experimental.neighbor_pad_async(
            input_tensor_mesh_list[i],
            dim=dim,
            padding_left=padding_left,
            padding_right=padding_right,
            padding_mode="replicate",
            cluster_axis=1,
            final_semaphore=ccl_semaphore_handles[i],
            barrier_semaphore=barrier_semaphore_handles[i],
            mesh_device=t3k_mesh_device,
            num_links=num_links,
            memory_config=mem_config_output,
            topology=neighbor_pad_topology,
        )

        return tt_neighbor_pad_out_tensor

    if enable_trace:
        # Compile the op
        tt_neighbor_pad_out_tensor = run_op(0)
        ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
        tt_neighbor_pad_out_tensor = run_op(0)
        ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done capturing trace")

        # Execute trace
        for i in range(num_iters):
            ttnn.execute_trace(t3k_mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
            tt_neighbor_pad_out_tensor_list.append(tt_neighbor_pad_out_tensor)
        logger.info(f"Done executing trace")
    else:
        for i in range(num_iters):
            tt_neighbor_pad_out_tensor = run_op(i)
            tt_neighbor_pad_out_tensor_list.append(tt_neighbor_pad_out_tensor)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    for i in range(num_iters):
        tt_np_out_tensor = tt_neighbor_pad_out_tensor_list[i]
        torch_np_out_tensor = np_output_tensor_goldens_list[i if not enable_trace else 0]

        tt_np_out = ttnn.from_device(tt_np_out_tensor)
        tt_np_out = ttnn.to_torch(tt_np_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=dim))
        eq, output = comp_pcc(tt_np_out, torch_np_out_tensor, 1)
        logger.info(f"{output}, iteration {i}")
        assert eq, f"{i} FAILED np: {output}"

    t3k_mesh_device.reset_sub_device_stall_group()
    t3k_mesh_device.clear_loaded_sub_device_manager()


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "num_devices, input_shape, dim, layout, input_dtype, padding_left, padding_right",
    [
        (8, [32, 60, 106, 768], 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0),
        (8, [88, 120, 212, 512], 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0),
        (8, [168, 240, 424, 256], 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0),
        (8, [168, 480, 848, 128], 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0),
    ],
    ids=[
        "mochi_vae_1",
        "mochi_vae_2",
        "mochi_vae_3",
        "mochi_vae_4",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_output",
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
    "device_params, neighbor_pad_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_neighbor_pad_async(
    mesh_device,
    num_devices,
    input_shape,
    dim,
    padding_left,
    padding_right,
    num_links,
    input_dtype,
    layout,
    mem_config_input,
    mem_config_output,
    enable_trace,
    neighbor_pad_topology,
    num_iters,
):
    run_neighbor_pad_impl(
        mesh_device,
        num_devices=num_devices,
        input_shape=input_shape,
        dim=dim,
        padding_left=padding_left,
        padding_right=padding_right,
        num_links=num_links,
        input_dtype=input_dtype,
        layout=layout,
        mem_config_input=mem_config_input,
        mem_config_output=mem_config_output,
        enable_trace=enable_trace,
        neighbor_pad_topology=neighbor_pad_topology,
        num_iters=num_iters,
    )

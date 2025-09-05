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

from ttnn import ShardTensor2dMesh, ConcatMesh2dToTensor


def run_neighbor_pad_impl(
    t3k_mesh_device,
    input_shape,
    dim,
    padding_left,
    padding_right,
    padding_mode,
    cluster_axis,
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
        num_chunks = t3k_mesh_device.shape[cluster_axis]
        chunks = torch.chunk(input_tensor, num_chunks, dim)
        np_output_tensor = []
        # pad left
        if padding_mode == "zeros":
            slice_shape = list(chunks[0].shape)
            slice_shape[dim] = 1
            first_slice_front = torch.zeros(slice_shape)
        else:
            first_slice_front = torch.narrow(chunks[0], dim, 0, 1)
        first_slice = torch.cat((first_slice_front, chunks[0]), dim=dim)
        np_output_tensor.append(first_slice)
        for p in range(padding_left - 1):
            np_output_tensor[0] = torch.cat((first_slice_front, np_output_tensor[0]), dim=dim)
        for k in range(1, num_chunks):
            prev_halo = torch.narrow(chunks[k - 1], dim, chunks[k - 1].shape[dim] - padding_left, padding_left)
            np_output_tensor.append(torch.cat((prev_halo, chunks[k]), dim=dim))

        # pad right
        if padding_mode == "zeros":
            slice_shape = list(np_output_tensor[num_chunks - 1].shape)
            slice_shape[dim] = 1
            last_slice_back = torch.zeros(slice_shape)
        else:
            last_slice_size = np_output_tensor[num_chunks - 1].shape[dim]
            last_slice_back = torch.narrow(np_output_tensor[num_chunks - 1], dim, last_slice_size - 1, 1)
        for p in range(padding_right):
            np_output_tensor[num_chunks - 1] = torch.cat((np_output_tensor[num_chunks - 1], last_slice_back), dim=dim)
        for k in range(0, num_chunks - 1):
            next_halo = torch.narrow(chunks[k + 1], dim, 0, padding_right)
            np_output_tensor[k] = torch.cat((np_output_tensor[k], next_halo), dim=dim)
        np_output_tensor_goldens_list.append(torch.cat(np_output_tensor, dim=dim))

        dims = [None, None]
        dims[cluster_axis] = dim
        input_tensor_mesh = ttnn.from_torch(
            input_tensor,
            device=t3k_mesh_device,
            layout=layout,
            dtype=input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=ttnn.ShardTensor2dMesh(t3k_mesh_device, mesh_shape=tuple(t3k_mesh_device.shape), dims=dims),
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
            padding_mode=padding_mode,
            cluster_axis=cluster_axis,
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
        dims[cluster_axis] = dim
        other_dim = (dim + 1) % len(tt_np_out.shape)
        dims[1 - cluster_axis] = other_dim
        tt_np_out = ttnn.to_torch(
            tt_np_out,
            mesh_composer=ConcatMesh2dToTensor(t3k_mesh_device, mesh_shape=tuple(t3k_mesh_device.shape), dims=dims),
        )
        tt_np_out = torch.narrow(tt_np_out, other_dim, 0, torch_np_out_tensor.shape[other_dim])
        eq, output = comp_pcc(tt_np_out, torch_np_out_tensor, 1)
        logger.info(f"{output}, iteration {i}")
        assert eq, f"{i} FAILED np: {output}"

    t3k_mesh_device.reset_sub_device_stall_group()
    t3k_mesh_device.clear_loaded_sub_device_manager()


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (2, 4), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "input_shape, dim, layout, input_dtype, padding_left, padding_right, padding_mode, cluster_axis",
    [
        ([32, 60, 106, 768], 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0, "replicate", 1),
        ([88, 120, 212, 512], 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0, "replicate", 1),
        ([168, 240, 424, 256], 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0, "replicate", 1),
        ([168, 480, 848, 128], 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0, "replicate", 1),
        ([32, 60, 106, 768], 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 2, "replicate", 1),
        ([32, 60, 106, 768], 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 2, "replicate", 0),
        ([32, 60, 106, 768], 2, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 2, "replicate", 0),
        ([1, 1, 106, 768], 2, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 2, "zeros", 0),
    ],
    ids=[
        "mochi_vae_1",
        "mochi_vae_2",
        "mochi_vae_3",
        "mochi_vae_4",
        "left_and_right",
        "cluster_axis",
        "replicate_width_dim",
        "zeros_width_dim",
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
    input_shape,
    dim,
    padding_left,
    padding_right,
    padding_mode,
    cluster_axis,
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
        input_shape=input_shape,
        dim=dim,
        padding_left=padding_left,
        padding_right=padding_right,
        padding_mode=padding_mode,
        cluster_axis=cluster_axis,
        num_links=num_links,
        input_dtype=input_dtype,
        layout=layout,
        mem_config_input=mem_config_input,
        mem_config_output=mem_config_output,
        enable_trace=enable_trace,
        neighbor_pad_topology=neighbor_pad_topology,
        num_iters=num_iters,
    )

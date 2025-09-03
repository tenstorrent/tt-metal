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


def run_slice_reshard_impl(
    t3k_mesh_device,
    num_devices,
    input_shape,
    dim,
    output_offset,
    output_shape,
    num_links,
    input_dtype,
    layout,
    mem_config_input,
    mem_config_output,
    enable_trace,
    slice_reshard_topology,
    num_iters,
):
    torch.manual_seed(0)

    ##### Slice Reshard setup #####
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

    ##### Slice Reshard input setup #####
    logger.info(f"Slice reshard input shape: {input_shape}")
    logger.info(f"Slice reshard dim: {dim}")

    input_tensor_mesh_list = []
    sr_output_tensor_goldens_list = []

    for i in range(num_iters):
        input_tensor = torch.rand(input_shape).bfloat16()
        input_tensor_sliced = input_tensor[output_offset : output_offset + output_shape, :, :, :]
        sr_output_tensor_goldens_list.append(input_tensor_sliced)

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
    tt_slice_reshard_out_tensor_list = []

    def run_op(i):
        tt_slice_reshard_out_tensor = ttnn.experimental.slice_reshard_async(
            input_tensor_mesh_list[i],
            dim=dim,
            output_dim_shape=output_shape,
            output_dim_offset=output_offset,
            cluster_axis=1,
            final_semaphore=ccl_semaphore_handles[i],
            barrier_semaphore=barrier_semaphore_handles[i],
            mesh_device=t3k_mesh_device,
            num_links=num_links,
            memory_config=mem_config_output,
            topology=slice_reshard_topology,
        )

        return tt_slice_reshard_out_tensor

    if enable_trace:
        # Compile the op
        tt_slice_reshard_out_tensor = run_op(0)
        ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
        tt_slice_reshard_out_tensor = run_op(0)
        ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done capturing trace")

        # Execute trace
        for i in range(num_iters):
            ttnn.execute_trace(t3k_mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
            tt_slice_reshard_out_tensor_list.append(tt_slice_reshard_out_tensor)
        logger.info(f"Done executing trace")
    else:
        for i in range(num_iters):
            tt_slice_reshard_out_tensor = run_op(i)
            tt_slice_reshard_out_tensor_list.append(tt_slice_reshard_out_tensor)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    for i in range(num_iters):
        tt_sr_out_tensor = tt_slice_reshard_out_tensor_list[i]
        torch_sr_out_tensor = sr_output_tensor_goldens_list[i if not enable_trace else 0]

        tt_sr_out = ttnn.from_device(tt_sr_out_tensor)
        tt_sr_out = ttnn.to_torch(tt_sr_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=dim))
        eq, output = comp_pcc(tt_sr_out, torch_sr_out_tensor, 1)
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
    "num_devices, input_shape, dim, layout, input_dtype, output_offset, output_shape",
    [
        (8, [96, 120, 212, 512], 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 88),
        (8, [176, 240, 424, 256], 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 168),
    ],
    ids=[
        "mochi_vae_1",
        "mochi_vae_2",
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
    "device_params, slice_reshard_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_slice_reshard_async(
    mesh_device,
    num_devices,
    input_shape,
    dim,
    output_offset,
    output_shape,
    num_links,
    input_dtype,
    layout,
    mem_config_input,
    mem_config_output,
    enable_trace,
    slice_reshard_topology,
    num_iters,
):
    run_slice_reshard_impl(
        mesh_device,
        num_devices=num_devices,
        input_shape=input_shape,
        dim=dim,
        output_offset=output_offset,
        output_shape=output_shape,
        num_links=num_links,
        input_dtype=input_dtype,
        layout=layout,
        mem_config_input=mem_config_input,
        mem_config_output=mem_config_output,
        enable_trace=enable_trace,
        slice_reshard_topology=slice_reshard_topology,
        num_iters=num_iters,
    )

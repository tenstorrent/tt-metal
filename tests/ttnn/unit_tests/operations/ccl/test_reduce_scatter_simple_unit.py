# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
import os
from models.common.utility_functions import skip_for_blackhole
from models.tt_transformers.tt.generator import create_submeshes
import tracy


@pytest.mark.parametrize("chunks_per_sync", [10, 20, 40, 80])
@pytest.mark.parametrize("num_workers_per_link", [1, 2, 4, 8])
@pytest.mark.parametrize("num_buffers_per_channel", [1, 2, 4, 8])
# @pytest.mark.parametrize("chunks_per_sync", [10])
# @pytest.mark.parametrize("num_workers_per_link", [1])
# @pytest.mark.parametrize("num_buffers_per_channel", [2])
@pytest.mark.parametrize(
    "rs_input_shape, dim, layout, rs_input_dtype",
    [
        ([1, 1, 32, 4096], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
    ids=["shape_1_1_32_4096"],
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
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_reduce_scatter_1_1_32_4096(
    mesh_device,
    rs_input_shape,
    dim,
    rs_input_dtype,
    layout,
    mem_config_input,
    mem_config_rs,
    rs_topology,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
):
    """
    Unit test for reduce_scatter_minimal_async with shape (1, 1, 32, 4096) on 8 devices.
    Tests the CCL operation at models/tt_transformers/tt/ccl.py:102
    """
    mesh_device = create_submeshes(mesh_device, 4)[0]

    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(0)

    # Setup fabric
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # Create semaphores
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(3)]
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)

    # Create input tensor with random values
    logger.info(f"Reduce scatter shape: {rs_input_shape}")
    logger.info(f"Reduce scatter dim: {dim}")
    logger.info(f"Number of devices: {num_devices}")

    rs_global_input_shape = rs_input_shape[:]
    rs_global_input_shape[dim] *= num_devices

    # Random tensor
    rs_input_tensor = torch.rand(rs_global_input_shape).bfloat16()
    input_tensors = torch.chunk(rs_input_tensor, num_devices, dim)

    input_tensor_mesh = ttnn.from_torch(
        rs_input_tensor,
        device=mesh_device,
        layout=layout,
        dtype=rs_input_dtype,
        memory_config=mem_config_input,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig(
                [ttnn.PlacementReplicate(), ttnn.PlacementShard(dim)], ttnn.MeshShape(1, num_devices)
            ),
        ),
    )

    # Compute torch reference
    reduce_output = torch.sum(torch.stack(input_tensors), dim=0)
    rs_output_shape = rs_input_shape[:]
    rs_output_shape[dim] //= num_devices
    scatter_output = torch.chunk(reduce_output, num_devices, dim)
    torch_rs_out_tensor = torch.cat(scatter_output, dim)

    # Run reduce_scatter_minimal_async
    logger.info("Running reduce_scatter_minimal_async")

    message = f"{chunks_per_sync} {num_workers_per_link} {num_buffers_per_channel}"
    tracy.signpost(message[:10])
    tt_reduce_scatter_output_tensor = ttnn.experimental.reduce_scatter_minimal_async(
        input_tensor_mesh,
        persistent_output_buffers=None,
        dim=dim,
        multi_device_global_semaphore=ccl_semaphore_handles,
        barrier_semaphore=barrier_semaphore,
        num_links=4,
        memory_config=mem_config_rs,
        intermediate_memory_config=ttnn.L1_MEMORY_CONFIG,
        topology=rs_topology,
        subdevice_id=worker_sub_device_id,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )
    tt_reduce_scatter_output_tensor = ttnn.experimental.reduce_scatter_minimal_async(
        input_tensor_mesh,
        persistent_output_buffers=None,
        dim=dim,
        multi_device_global_semaphore=ccl_semaphore_handles,
        barrier_semaphore=barrier_semaphore,
        num_links=4,
        memory_config=mem_config_rs,
        intermediate_memory_config=ttnn.L1_MEMORY_CONFIG,
        topology=rs_topology,
        subdevice_id=worker_sub_device_id,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )
    tt_reduce_scatter_output_tensor = ttnn.experimental.reduce_scatter_minimal_async(
        input_tensor_mesh,
        persistent_output_buffers=None,
        dim=dim,
        multi_device_global_semaphore=ccl_semaphore_handles,
        barrier_semaphore=barrier_semaphore,
        num_links=4,
        memory_config=mem_config_rs,
        intermediate_memory_config=ttnn.L1_MEMORY_CONFIG,
        topology=rs_topology,
        subdevice_id=worker_sub_device_id,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )
    tracy.signpost("===")
    tracy.signpost("===")
    tracy.signpost("===")

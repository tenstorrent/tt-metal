# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Small-page Fabric 2D all-gather regression coverage.

The sparse MLA 4x2 path uses row-major bfloat16 pages of 2 KiB.  With the
14 KiB Fabric router payload configuration, the multicast implementation
batches four pages into each scatter packet. Keep this test intentionally
small so a Fabric routing failure can be diagnosed without model execution.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _fabric_router_config():
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = 14 * 1024
    return config


def _all_worker_cores(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 512,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
@pytest.mark.parametrize(
    "cluster_axis,gather_dim,global_shape,use_persistent_output",
    [
        pytest.param(1, 3, (1, 1, 1, 512), False, id="tp_axis_2x_half_page"),
        pytest.param(1, 3, (1, 1, 1, 2048), False, id="tp_axis_2x_single_page"),
        pytest.param(1, 3, (1, 1, 1, 2048), True, id="tp_axis_2x_single_page_persistent"),
        pytest.param(1, 3, (1, 1, 32, 2048), False, id="tp_axis_2x_32_pages"),
        pytest.param(0, 2, (1, 1, 4, 1024), False, id="sp_axis_4x_single_page"),
        pytest.param(0, 2, (1, 1, 128, 1024), False, id="sp_axis_4x_32_pages"),
    ],
)
def test_all_gather_fabric_2d_row_major_2k_pages(
    mesh_device, cluster_axis, gather_dim, global_shape, use_persistent_output
):
    """Gather 2 KiB row-major pages; 32-page cases match sparse MLA's geometry."""
    torch.manual_seed(0)
    torch_input = torch.rand(global_shape, dtype=torch.bfloat16)
    shard_dims = (None, gather_dim) if cluster_axis == 1 else (gather_dim, None)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
        device=mesh_device,
    )

    persistent_output = None
    if use_persistent_output:
        persistent_output = ttnn.from_torch(
            torch.zeros(global_shape, dtype=torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            device=mesh_device,
        )

    tt_output = ttnn.all_gather(
        tt_input,
        dim=gather_dim,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=cluster_axis,
        output_tensor=persistent_output,
    )

    for device_tensor in ttnn.get_device_tensors(tt_output):
        assert_with_pcc(ttnn.to_torch(device_tensor), torch_input, pcc=1.0)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _fabric_router_config(),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 512,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
def test_all_gather_async_fabric_2d_control_row_major_2k_page(mesh_device):
    """Control: the pre-multicast MLA all-gather with the same 4x2 TP geometry."""
    torch.manual_seed(0)
    global_shape = (1, 1, 1, 2048)
    torch_input = torch.rand(global_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=tuple(mesh_device.shape)),
        device=mesh_device,
    )

    worker_cores = _all_worker_cores(mesh_device)
    semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(2)]
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)
    tt_output = ttnn.experimental.all_gather_async(
        tt_input,
        dim=3,
        cluster_axis=1,
        multi_device_global_semaphore=semaphores,
        barrier_semaphore=barrier_semaphore,
        num_links=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
    )
    ttnn.synchronize_device(mesh_device)

    for device_tensor in ttnn.get_device_tensors(tt_output):
        assert_with_pcc(ttnn.to_torch(device_tensor), torch_input, pcc=1.0)

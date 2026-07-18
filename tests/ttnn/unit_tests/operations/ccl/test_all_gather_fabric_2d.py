# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse-MLA Fabric 2D all-gather regression coverage.

The sparse MLA 4x2 path gathers row-major KV-cache rows on the SP axis.  Its
BF16 cache has 1,152-byte rows; its scaled-FP8 cache packs each 656-byte row
into a 704-byte aligned DRAM page.  Keep these tests small so a Fabric routing
failure can be diagnosed without model execution.
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
    "cluster_axis,gather_dim,global_shape,dtype,pcc,use_persistent_output",
    [
        pytest.param(1, 3, (1, 1, 1, 512), ttnn.bfloat16, 1.0, False, id="tp_axis_2x_half_page"),
        pytest.param(1, 3, (1, 1, 1, 2048), ttnn.bfloat16, 1.0, False, id="tp_axis_2x_single_page"),
        pytest.param(1, 3, (1, 1, 1, 2048), ttnn.bfloat16, 1.0, True, id="tp_axis_2x_single_page_persistent"),
        pytest.param(1, 3, (1, 1, 32, 2048), ttnn.bfloat16, 1.0, False, id="tp_axis_2x_32_pages"),
        pytest.param(0, 2, (1, 1, 4, 1024), ttnn.bfloat16, 1.0, False, id="sp_axis_4x_single_page"),
        pytest.param(0, 2, (1, 1, 128, 1024), ttnn.bfloat16, 1.0, False, id="sp_axis_4x_32_pages"),
        # Sparse MLA KV cache: 576 BF16 values, one 1,152-byte physical row page.
        pytest.param(0, 2, (1, 1, 128, 576), ttnn.bfloat16, 1.0, True, id="sp_axis_4x_mla_kv_bf16"),
        # Packed scaled-FP8 cache: 656 logical bytes, rounded to a 704-byte DRAM row page.
        pytest.param(0, 2, (1, 1, 128, 656), ttnn.fp8_e4m3, 0.99, True, id="sp_axis_4x_mla_kv_scaled_fp8"),
    ],
)
def test_all_gather_fabric_2d_row_major_2k_pages(
    mesh_device, cluster_axis, gather_dim, global_shape, dtype, pcc, use_persistent_output
):
    """Gather row-major pages using the same mesh, output lifetime, and dtype as sparse MLA."""
    torch.manual_seed(0)
    host_dtype = torch.float32 if dtype == ttnn.fp8_e4m3 else torch.bfloat16
    torch_input = torch.rand(global_shape, dtype=host_dtype)
    shard_dims = (None, gather_dim) if cluster_axis == 1 else (gather_dim, None)
    # Mesh-mapped host construction currently forces FP8_E4M3 through TILE. Build
    # the row-major BF16 transport tensor first, then typecast on device; this is
    # the same sequence used for the sparse MLA packed-cache setup.
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16 if dtype == ttnn.fp8_e4m3 else dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
        device=mesh_device,
    )
    if dtype == ttnn.fp8_e4m3:
        tt_input = ttnn.typecast(tt_input, dtype)

    persistent_output = None
    if use_persistent_output:
        persistent_output = ttnn.from_torch(
            torch.zeros(global_shape, dtype=host_dtype),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16 if dtype == ttnn.fp8_e4m3 else dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            device=mesh_device,
        )
        if dtype == ttnn.fp8_e4m3:
            persistent_output = ttnn.typecast(persistent_output, dtype)

    tt_output = ttnn.all_gather(
        tt_input,
        dim=gather_dim,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=cluster_axis,
        output_tensor=persistent_output,
    )

    # FP8_E4M3 cannot be read back one device at a time. Convert after the
    # collective so we still validate every gathered replica numerically.
    check_output = ttnn.typecast(tt_output, ttnn.bfloat16) if dtype == ttnn.fp8_e4m3 else tt_output
    for device_tensor in ttnn.get_device_tensors(check_output):
        assert_with_pcc(ttnn.to_torch(device_tensor), torch_input, pcc=pcc)


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

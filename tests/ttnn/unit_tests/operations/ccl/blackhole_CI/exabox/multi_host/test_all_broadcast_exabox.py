# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
#  SPDX-License-Identifier: Apache-2.0

"""All-broadcast tests for multi-galaxy exabox mesh configurations (DUAL_BH / QUAD_BH).

Tests cover:
- Sync API (ttnn.all_broadcast) on 16x4 and 32x4 meshes
- Multiple fabric configs (FABRIC_1D, FABRIC_1D_RING) and topologies (Linear, Ring)
- DRAM and L1 memory configs
- TILE_LAYOUT and ROW_MAJOR_LAYOUT
- bfloat16 dtype
"""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _setup_sub_devices(mesh_device):
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
    return worker_sub_device_id, sub_device_stall_group, sub_device_manager


def _cleanup_sub_devices(mesh_device, sub_device_manager):
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(sub_device_manager)


def _run_all_broadcast_test(
    mesh_device,
    output_shape,
    num_links,
    input_dtype,
    layout,
    topology,
    cluster_axis,
    mem_config,
    num_iters=1,
):
    mesh_shape = tuple(mesh_device.shape)
    num_devices = mesh_shape[cluster_axis]
    worker_sub_device_id, sub_device_stall_group, sub_device_manager = _setup_sub_devices(mesh_device)

    try:
        # Seed before torch.rand so every MPI rank generates identical golden
        # data; otherwise rank-local goldens diverge and PCC drops to ~0.5.
        torch.manual_seed(0)
        output_tensors = []
        for k in range(num_devices):
            output_tensors.append(torch.rand(output_shape).bfloat16())

        temp_output_tensor = torch.cat(output_tensors, -1)

        # Input must be sharded along the broadcast axis so each device on that
        # axis contributes a distinct shard. Replicating along the broadcast axis
        # would make the broadcast a no-op and corrupt the per-source outputs.
        # Use the full mesh shape so xtensor_views == mesh_size, which the 2D
        # composer needs in distributed-env verification below.
        if cluster_axis == 0:
            placements = [ttnn.PlacementShard(-1), ttnn.PlacementReplicate()]
        else:
            placements = [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)]
        mapper_mesh_shape = ttnn.MeshShape(*mesh_shape)

        input_tensor_mesh = ttnn.from_torch(
            temp_output_tensor,
            device=mesh_device,
            layout=layout,
            dtype=input_dtype,
            memory_config=mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(placements, mapper_mesh_shape),
            ),
        )

        for i in range(num_iters):
            tt_out_tensors = ttnn.all_broadcast(
                input_tensor_mesh,
                num_links=num_links,
                memory_config=mem_config,
                topology=topology,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
            )

        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

        # Compose the broadcast output across both mesh axes and verify each
        # per-mesh-coord block equals output_tensors[k]. With full-mesh
        # placements above, xtensor_views == mesh_size so the 2D composer
        # works in both single-galaxy and multi-host MPI runs.
        mesh_shape_tt = ttnn.MeshShape(*mesh_shape)
        composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=mesh_shape_tt)
        for k in range(num_devices):
            output_tensor = output_tensors[k]
            composed = ttnn.to_torch(tt_out_tensors[k], mesh_composer=composer)
            expected = output_tensor.repeat([mesh_shape[0], mesh_shape[1]] + [1] * (output_tensor.ndim - 2))
            if input_dtype == ttnn.bfloat16:
                eq, output = comp_equal(composed, expected)
            else:
                eq, output = comp_pcc(composed, expected)
            assert eq, f"source {k} FAILED: {output}"
    finally:
        _cleanup_sub_devices(mesh_device, sub_device_manager)


# ---------------------------------------------------------------------------
# Fabric / topology parametrize combos
# ---------------------------------------------------------------------------

FABRIC_TOPOLOGY_COMBOS = [
    pytest.param(
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
        ttnn.Topology.Linear,
        id="fabric_1d-linear",
    ),
    pytest.param(
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
        ttnn.Topology.Ring,
        id="fabric_1d_ring-ring",
    ),
]


# ---------------------------------------------------------------------------
# Test: all_broadcast on 16x4 mesh (DUAL_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["DUAL_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
            ttnn.Topology.Linear,
            id="fabric_1d-linear",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((16, 4), id="16x4_grid")], indirect=True)
@pytest.mark.parametrize("cluster_axis", [pytest.param(0, id="axis0"), pytest.param(1, id="axis1")])
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "output_shape, layout, input_dtype, mem_config",
    [
        ([1, 1, 32, 1024], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)),
        ([2, 30], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)),
    ],
    ids=["tile_dram", "rm_l1_small"],
)
def test_all_broadcast_16x4(
    mesh_device,
    cluster_axis,
    topology,
    num_links,
    output_shape,
    layout,
    input_dtype,
    mem_config,
):
    _run_all_broadcast_test(
        mesh_device, output_shape, num_links, input_dtype, layout, topology, cluster_axis, mem_config
    )


# ---------------------------------------------------------------------------
# Test: all_broadcast on 32x4 mesh (QUAD_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    FABRIC_TOPOLOGY_COMBOS,
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((32, 4), id="32x4_grid")], indirect=True)
@pytest.mark.parametrize("cluster_axis", [pytest.param(0, id="axis0"), pytest.param(1, id="axis1")])
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "output_shape, layout, input_dtype, mem_config",
    [
        ([1, 1, 32, 1024], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)),
        ([2, 30], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)),
    ],
    ids=["tile_dram", "rm_l1_small"],
)
def test_all_broadcast_32x4(
    mesh_device,
    cluster_axis,
    topology,
    num_links,
    output_shape,
    layout,
    input_dtype,
    mem_config,
):
    _run_all_broadcast_test(
        mesh_device, output_shape, num_links, input_dtype, layout, topology, cluster_axis, mem_config
    )

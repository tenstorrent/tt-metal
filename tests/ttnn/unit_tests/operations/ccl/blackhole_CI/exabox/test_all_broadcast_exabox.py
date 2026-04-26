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
from loguru import logger

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
    num_devices,
    num_links,
    input_dtype,
    layout,
    topology,
    cluster_axis,
    mem_config,
    num_iters=1,
):
    mesh_shape = tuple(mesh_device.shape)
    worker_sub_device_id, sub_device_stall_group, sub_device_manager = _setup_sub_devices(mesh_device)

    try:
        output_tensors = []
        for k in range(num_devices):
            output_tensors.append(torch.rand(output_shape).bfloat16())

        temp_output_tensor = torch.cat(output_tensors, -1)

        mapper_mesh_shape = ttnn.MeshShape(mesh_shape[0], mesh_shape[1])
        input_tensor_mesh = ttnn.from_torch(
            temp_output_tensor,
            device=mesh_device,
            layout=layout,
            dtype=input_dtype,
            memory_config=mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(
                    [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)],
                    mapper_mesh_shape,
                ),
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

        view = mesh_device.get_view() if ttnn.using_distributed_env() else None

        for k in range(num_devices):
            output_tensor = output_tensors[k]
            coords = list(tt_out_tensors[k].tensor_topology().mesh_coords())
            device_tensors = ttnn.get_device_tensors(tt_out_tensors[k])
            coord_iter = coords
            if view is not None and len(device_tensors) != len(coords):
                coord_iter = [coord for coord in coords if view.is_local(coord)]
            for coord, t in zip(coord_iter, device_tensors):
                if view is not None and not view.is_local(coord):
                    continue
                tt_output_tensor = ttnn.to_torch(t)
                if input_dtype == ttnn.bfloat16:
                    eq, output = comp_equal(tt_output_tensor, output_tensor)
                else:
                    eq, output = comp_pcc(tt_output_tensor, output_tensor)
                assert eq, f"Device {coord}, source {k} FAILED: {output}"
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
@pytest.mark.parametrize("cluster_axis", [pytest.param(1, id="axis1")])
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "num_devices, output_shape, layout, input_dtype, mem_config",
    [
        (4, [1, 1, 32, 1024], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)),
        (4, [2, 30], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)),
        (4, [3, 122, 2042], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)),
    ],
    ids=["tile_dram", "rm_l1_small", "rm_l1_large"],
)
def test_all_broadcast_16x4(
    mesh_device,
    cluster_axis,
    topology,
    num_links,
    num_devices,
    output_shape,
    layout,
    input_dtype,
    mem_config,
):
    _run_all_broadcast_test(
        mesh_device, output_shape, num_devices, num_links, input_dtype, layout, topology, cluster_axis, mem_config
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
@pytest.mark.parametrize("cluster_axis", [pytest.param(1, id="axis1")])
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "num_devices, output_shape, layout, input_dtype, mem_config",
    [
        (4, [1, 1, 32, 1024], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)),
        (4, [2, 30], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)),
        (4, [3, 122, 2042], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)),
        (4, [256, 3328], ttnn.TILE_LAYOUT, ttnn.bfloat8_b, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)),
    ],
    ids=["tile_dram", "rm_l1_small", "rm_l1_large", "tile_l1_bfloat8"],
)
def test_all_broadcast_32x4(
    mesh_device,
    cluster_axis,
    topology,
    num_links,
    num_devices,
    output_shape,
    layout,
    input_dtype,
    mem_config,
):
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        pytest.skip("bfloat8_b not supported for row-major")

    _run_all_broadcast_test(
        mesh_device, output_shape, num_devices, num_links, input_dtype, layout, topology, cluster_axis, mem_config
    )

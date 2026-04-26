# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
#  SPDX-License-Identifier: Apache-2.0

"""Broadcast tests for multi-galaxy exabox mesh configurations (DUAL_BH / QUAD_BH).

Tests cover:
- Sync API (ttnn.broadcast) on 16x4 and 32x4 meshes
- Multiple fabric configs (FABRIC_1D, FABRIC_1D_RING) and topologies (Linear, Ring)
- Various sender coordinates
- DRAM and L1 memory configs
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


def _run_broadcast_test(
    mesh_device,
    output_shape,
    sender_coord_tuple,
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
    sender_coord = ttnn.MeshCoordinate(sender_coord_tuple)
    worker_sub_device_id, sub_device_stall_group, sub_device_manager = _setup_sub_devices(mesh_device)

    try:
        sender_tensor = torch.rand(output_shape, dtype=torch.bfloat16)

        device_tensors = []
        for device_idx in range(num_devices):
            if device_idx == sender_coord_tuple[cluster_axis]:
                device_tensors.append(sender_tensor)
            else:
                device_tensors.append(torch.zeros_like(sender_tensor))
        mesh_tensor_torch = torch.cat(device_tensors, dim=-1)

        mapper_mesh_shape = ttnn.MeshShape(mesh_shape[0], mesh_shape[1])
        input_tensor_mesh = ttnn.from_torch(
            mesh_tensor_torch,
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
            tt_out_tensor = ttnn.broadcast(
                input_tensor_mesh,
                sender_coord=sender_coord,
                num_links=num_links,
                memory_config=mem_config,
                topology=topology,
                subdevice_id=worker_sub_device_id,
            )

        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

        output_tensor_torch = ttnn.to_torch(
            tt_out_tensor,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=cluster_axis),
        )
        slice_size = output_shape[cluster_axis] if cluster_axis < len(output_shape) else 1
        for i in range(num_devices):
            start = i * slice_size
            end = start + slice_size
            slices = [slice(None)] * output_tensor_torch.dim()
            slices[cluster_axis] = slice(start, end)
            received = output_tensor_torch[tuple(slices)]
            assert (
                received.shape == sender_tensor.shape
            ), f"Shape mismatch: received {received.shape}, expected {sender_tensor.shape}"
            if input_dtype == ttnn.bfloat16:
                eq, output = comp_equal(received, sender_tensor)
            else:
                eq, output = comp_pcc(received, sender_tensor)
            assert eq, f"Device {i} FAILED: {output}"
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
# Test: broadcast on 16x4 mesh (DUAL_BH)
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
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "num_devices, sender_idx, cluster_axis, output_shape, layout, input_dtype, mem_config",
    [
        (
            4,
            0,
            1,
            [1, 1, 32, 1024],
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ),
        (4, 2, 1, [32, 32], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)),
        (
            4,
            1,
            1,
            [2, 90, 1042],
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
        ),
    ],
    ids=["tile_dram_sender0", "tile_l1_sender2", "rm_l1_sender1"],
)
def test_broadcast_16x4(
    mesh_device,
    topology,
    num_links,
    num_devices,
    sender_idx,
    cluster_axis,
    output_shape,
    layout,
    input_dtype,
    mem_config,
):
    sender_coord_tuple = (0, sender_idx)
    _run_broadcast_test(
        mesh_device,
        output_shape,
        sender_coord_tuple,
        num_devices,
        num_links,
        input_dtype,
        layout,
        topology,
        cluster_axis,
        mem_config,
    )


# ---------------------------------------------------------------------------
# Test: broadcast on 32x4 mesh (QUAD_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    FABRIC_TOPOLOGY_COMBOS,
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((32, 4), id="32x4_grid")], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "num_devices, sender_idx, cluster_axis, output_shape, layout, input_dtype, mem_config",
    [
        (
            4,
            0,
            1,
            [1, 1, 32, 1024],
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ),
        (4, 3, 1, [2, 64, 512], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)),
        (
            4,
            1,
            1,
            [3, 121, 2042],
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
        ),
        (4, 2, 1, [256, 3328], ttnn.TILE_LAYOUT, ttnn.bfloat8_b, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)),
    ],
    ids=["tile_dram_sender0", "tile_l1_sender3", "rm_l1_sender1", "tile_dram_bfloat8_sender2"],
)
def test_broadcast_32x4(
    mesh_device,
    topology,
    num_links,
    num_devices,
    sender_idx,
    cluster_axis,
    output_shape,
    layout,
    input_dtype,
    mem_config,
):
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        pytest.skip("bfloat8_b not supported for row-major")

    sender_coord_tuple = (0, sender_idx)
    _run_broadcast_test(
        mesh_device,
        output_shape,
        sender_coord_tuple,
        num_devices,
        num_links,
        input_dtype,
        layout,
        topology,
        cluster_axis,
        mem_config,
    )

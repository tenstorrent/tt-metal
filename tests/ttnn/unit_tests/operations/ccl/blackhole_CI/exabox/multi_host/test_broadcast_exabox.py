# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
#  SPDX-License-Identifier: Apache-2.0

"""Broadcast tests for multi-galaxy exabox mesh configurations (DUAL_BH / QUAD_BH).

Tests cover:
- Sync API (ttnn.broadcast) on 16x4 and 32x4 meshes
- FABRIC_2D fabric config (required for 2D-mesh broadcast; FABRIC_1D hangs the op)
- Both cluster axes (0 and 1)
- DRAM and L1 memory configs
- bfloat16 dtype
"""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.ttnn.utils_for_testing import maybe_trace

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_tensors(input_shape, sender_coord_tuple, cluster_axis, mesh_shape, dtype, layout, memory_config, device):
    # Single sender_tensor at the sender position along cluster_axis; zeros elsewhere.
    # Use the canonical mapper pattern from test_broadcast_2d / run_broadcast_impl:
    # MeshShape(num_along_axis, 1) or (1, num_along_axis) with [Shard, Replicate].
    # Earlier attempt with MeshShape(*mesh_shape) (full 2D) hung from_torch on both
    # single-galaxy 8x4 and multi-host 16x4.
    torch.manual_seed(0)
    sender_tensor = torch.rand(input_shape).bfloat16()

    sender_idx = sender_coord_tuple[cluster_axis]
    num_along_axis = mesh_shape[cluster_axis]
    shards = [sender_tensor if k == sender_idx else torch.zeros_like(sender_tensor) for k in range(num_along_axis)]
    torch_input = torch.cat(shards, dim=0)

    if cluster_axis == 0:
        placements = [ttnn.PlacementShard(0), ttnn.PlacementReplicate()]
        mapper_mesh_shape = ttnn.MeshShape(num_along_axis, 1)
    else:
        placements = [ttnn.PlacementReplicate(), ttnn.PlacementShard(0)]
        mapper_mesh_shape = ttnn.MeshShape(1, num_along_axis)

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
        device=device,
        mesh_mapper=ttnn.create_mesh_mapper(
            device,
            ttnn.MeshMapperConfig(placements, mapper_mesh_shape),
        ),
    )
    return tt_input, sender_tensor


def _verify_broadcast_output(tt_output_tensor, sender_tensor, cluster_axis, mesh_device):
    # The 1D-style mapper (MeshShape(1, num_along_axis) etc.) doesn't establish
    # per-device topology that get_device_tensors+to_torch can use directly, so
    # use a mesh composer along cluster_axis (same pattern as the working
    # tests/nightly/t3000/ccl/test_broadcast_op.py verification).
    output_tensor_torch = ttnn.to_torch(
        tt_output_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=cluster_axis),
    )
    slice_size = sender_tensor.shape[cluster_axis] if cluster_axis < sender_tensor.dim() else 1
    num_slices = output_tensor_torch.shape[cluster_axis] // slice_size
    for i in range(num_slices):
        slices = [slice(None)] * output_tensor_torch.dim()
        slices[cluster_axis] = slice(i * slice_size, (i + 1) * slice_size)
        received = output_tensor_torch[tuple(slices)]
        eq, mess = comp_equal(received, sender_tensor)
        assert eq, f"slice {i} along axis {cluster_axis} mismatch: {mess}"


def _run_broadcast_test(
    mesh_device,
    input_shape,
    sender_coord_tuple,
    cluster_axis,
    buffer_type,
    dtype,
    topology,
    enable_trace,
    num_links=None,
):
    mesh_shape = tuple(mesh_device.shape)
    memory_config = ttnn.MemoryConfig(buffer_type=buffer_type)

    # Sub-device setup: ttnn.broadcast requires a worker sub-device to drive the CCL workers,
    # otherwise the op hangs after fabric init. (Sync all_reduce / all_gather don't need this.)
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

    try:
        tt_input, sender_tensor = _get_tensors(
            input_shape,
            sender_coord_tuple,
            cluster_axis,
            mesh_shape,
            dtype,
            ttnn.TILE_LAYOUT,
            memory_config,
            mesh_device,
        )

        sender_coord = ttnn.MeshCoordinate(sender_coord_tuple)
        bc_kwargs = dict(
            cluster_axis=cluster_axis,
            topology=topology,
            memory_config=memory_config,
            subdevice_id=worker_sub_device_id,
        )
        if num_links is not None:
            bc_kwargs["num_links"] = num_links

        def run_op():
            return ttnn.broadcast(tt_input, sender_coord, **bc_kwargs)

        tt_output_tensor = maybe_trace(run_op, enable_trace=enable_trace, device=mesh_device)
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        _verify_broadcast_output(tt_output_tensor, sender_tensor, cluster_axis, mesh_device)
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(sub_device_manager)


# ---------------------------------------------------------------------------
# Fabric / topology parametrize combos (reused by multiple tests)
# ---------------------------------------------------------------------------

FABRIC_TOPOLOGY_COMBOS = [
    pytest.param(
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 90112},
        ttnn.Topology.Linear,
        id="fabric_2d-linear",
    ),
]


# ---------------------------------------------------------------------------
# Test: sync broadcast on 16x4 mesh (DUAL_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["DUAL_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 90112},
            ttnn.Topology.Linear,
            id="fabric_2d-linear",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((16, 4), id="16x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "cluster_axis, sender_coord_tuple",
    [
        pytest.param(0, (5, 0), id="axis0_sender_row5"),
        pytest.param(1, (0, 2), id="axis1_sender_col2"),
    ],
)
@pytest.mark.parametrize("num_links", [2], ids=["2links"])
@pytest.mark.parametrize("enable_trace", [False])
@pytest.mark.parametrize(
    "input_shape, dtype, buffer_type",
    [
        ([1, 1, 32, 224], ttnn.bfloat16, ttnn.BufferType.DRAM),
        ([1, 1, 32, 224], ttnn.bfloat16, ttnn.BufferType.L1),
    ],
    ids=["small_dram", "small_l1"],
)
def test_broadcast_16x4(
    mesh_device,
    cluster_axis,
    sender_coord_tuple,
    topology,
    enable_trace,
    input_shape,
    dtype,
    buffer_type,
    num_links,
):
    _run_broadcast_test(
        mesh_device,
        input_shape,
        sender_coord_tuple,
        cluster_axis,
        buffer_type,
        dtype,
        topology,
        enable_trace,
        num_links=num_links,
    )


# ---------------------------------------------------------------------------
# Test: sync broadcast on 32x4 mesh (QUAD_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    FABRIC_TOPOLOGY_COMBOS,
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((32, 4), id="32x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "cluster_axis, sender_coord_tuple",
    [
        pytest.param(0, (10, 0), id="axis0_sender_row10"),
        pytest.param(1, (0, 2), id="axis1_sender_col2"),
    ],
)
@pytest.mark.parametrize("num_links", [2], ids=["2links"])
@pytest.mark.parametrize("enable_trace", [False])
@pytest.mark.parametrize(
    "input_shape, dtype, buffer_type",
    [
        ([1, 1, 32, 224], ttnn.bfloat16, ttnn.BufferType.DRAM),
        ([1, 1, 32, 224], ttnn.bfloat16, ttnn.BufferType.L1),
    ],
    ids=["small_dram", "small_l1"],
)
def test_broadcast_32x4(
    mesh_device,
    cluster_axis,
    sender_coord_tuple,
    topology,
    enable_trace,
    input_shape,
    dtype,
    buffer_type,
    num_links,
):
    _run_broadcast_test(
        mesh_device,
        input_shape,
        sender_coord_tuple,
        cluster_axis,
        buffer_type,
        dtype,
        topology,
        enable_trace,
        num_links=num_links,
    )

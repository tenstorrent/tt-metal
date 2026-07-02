# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the self-contained Python point_to_point CCL op.

point_to_point sends one mesh device's interleaved shard to another device over
the Tenstorrent fabric. It is pure data movement (no arithmetic), so the oracle
is identity:

  * the receiver device's output shard == the sender device's input shard,
  * every other device's output shard == that device's input shard (unchanged).

This file is the immutable spec — the implementer must not modify it.

Verification topology (MUST match the multi-device sim's mesh-graph descriptor):
the op is verified on an 8-chip Blackhole mesh of shape ``(2, 4)`` with
``fabric_config = ttnn.FabricConfig.FABRIC_1D`` via
``scripts/run_multidevice_sim_pytest.py --op point_to_point``. Opening a
different mesh shape hangs fabric init ("Fabric Router Sync: Timeout"), so the
``mesh_device`` fixture is pinned to ``(2, 4)`` + ``FABRIC_1D``. Both Linear and
Ring topology are exercised through the op's ``topology`` kwarg on the same
FABRIC_1D fabric using adjacent coords ``(0,0) -> (0,1)`` (a single hop either
way), which is valid for both topologies.
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.point_to_point import point_to_point


# PCC tolerances keyed by dtype (same thresholds as the golden suite).
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

# Valid (dtype, layout) pairs. bfloat8_b is a tiled block-float format with no
# row-major representation, so it appears only with TILE_LAYOUT.
DTYPE_LAYOUTS = [
    (ttnn.bfloat16, ttnn.TILE_LAYOUT),
    (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    (ttnn.float32, ttnn.TILE_LAYOUT),
    (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
]

# Per-device shard shapes: single-tile, multi-tile, non-square, multi-batch,
# non-tile-aligned. Last dims are multiples of 8 so the row-major page size stays
# 16-byte aligned for every dtype.
SHARD_SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (1, 1, 96, 64),  # non-square, tile-aligned
    (2, 1, 32, 64),  # multi-batch
    (1, 1, 48, 64),  # non-tile-aligned (H not %32), 16B-aligned page
]

# The sim's mesh-graph descriptor is fixed to (2, 4) + FABRIC_1D. Both topologies
# run on this fabric; adjacent coords route one hop either way.
MESH_SHAPE = (2, 4)
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}
TOPOLOGIES = [ttnn.Topology.Linear, ttnn.Topology.Ring]


def _linear_index(coord, mesh_shape):
    """Row-major linear index of a MeshCoordinate in a (rows, cols) mesh."""
    mesh_shape = tuple(mesh_shape)
    return coord[0] * mesh_shape[1] + coord[1]


def _make_input(mesh_device, shard_shape, dtype, layout):
    """Shard a freshly-seeded tensor along dim 0 across the mesh.

    Each device receives exactly ``shard_shape``. Returns the ttnn input tensor
    and the list of per-device shards (torch) in linear mesh order, snapshotted
    before the op runs.
    """
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(42)
    torch_full = torch.randn(full_shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)

    input_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(input_tensor)]
    return input_tensor, input_shards


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("dtype, layout", DTYPE_LAYOUTS)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_point_to_point(mesh_device, topology, dtype, layout, shard_shape):
    """Receiver shard equals sender shard; sender's own shard is unchanged."""
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    sender_coord = ttnn.MeshCoordinate(0, 0)
    receiver_coord = ttnn.MeshCoordinate(0, 1)
    send_idx = _linear_index(sender_coord, mesh_device.shape)
    recv_idx = _linear_index(receiver_coord, mesh_device.shape)

    input_tensor, input_shards = _make_input(mesh_device, shard_shape, dtype, layout)

    output_tensor = point_to_point(input_tensor, sender_coord, receiver_coord, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    pcc = PCC[dtype]
    # Receiver shard now holds the sender's input shard.
    assert_with_pcc(input_shards[send_idx], output_shards[recv_idx], pcc)
    # Sender's own shard is unchanged.
    assert_with_pcc(input_shards[send_idx], output_shards[send_idx], pcc)
    logger.info(f"p2p {dtype} {layout} {shard_shape} {topology}: receiver shard matches sender shard")


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("dtype, layout", [(ttnn.bfloat16, ttnn.TILE_LAYOUT), (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)])
def test_point_to_point_nonparticipating_unchanged(mesh_device, topology, dtype, layout):
    """Every non-receiver shard stays equal to its input (only the receiver changes)."""
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    sender_coord = ttnn.MeshCoordinate(0, 0)
    receiver_coord = ttnn.MeshCoordinate(0, 1)
    send_idx = _linear_index(sender_coord, mesh_device.shape)
    recv_idx = _linear_index(receiver_coord, mesh_device.shape)

    input_tensor, input_shards = _make_input(mesh_device, (1, 1, 64, 64), dtype, layout)

    output_tensor = point_to_point(input_tensor, sender_coord, receiver_coord, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    pcc = PCC[dtype]
    # Receiver got the sender's data.
    assert_with_pcc(input_shards[send_idx], output_shards[recv_idx], pcc)
    # All other devices (including the sender) are untouched.
    for i in range(len(output_shards)):
        if i == recv_idx:
            continue
        assert_with_pcc(input_shards[i], output_shards[i], pcc)


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("dtype, layout", [(ttnn.bfloat16, ttnn.TILE_LAYOUT), (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)])
def test_point_to_point_output_tensor(mesh_device, topology, dtype, layout):
    """The output_tensor path writes into the supplied tensor and returns it."""
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    sender_coord = ttnn.MeshCoordinate(0, 0)
    receiver_coord = ttnn.MeshCoordinate(0, 1)
    send_idx = _linear_index(sender_coord, mesh_device.shape)
    recv_idx = _linear_index(receiver_coord, mesh_device.shape)

    input_tensor, input_shards = _make_input(mesh_device, (1, 1, 64, 128), dtype, layout)
    preallocated = ttnn.allocate_tensor_on_device(input_tensor.spec, mesh_device)

    returned = point_to_point(input_tensor, sender_coord, receiver_coord, topology=topology, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)

    # The supplied handle is returned.
    assert returned.buffer_address() == preallocated.buffer_address()

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(returned)]
    assert_with_pcc(input_shards[send_idx], output_shards[recv_idx], PCC[dtype])


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_point_to_point_program_cache(mesh_device, topology):
    """Second call (program-cache hit) still transfers correctly.

    The op-internal GlobalSemaphore must survive the cache hit (created once,
    not re-created per call).
    """
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    sender_coord = ttnn.MeshCoordinate(0, 0)
    receiver_coord = ttnn.MeshCoordinate(0, 1)
    send_idx = _linear_index(sender_coord, mesh_device.shape)
    recv_idx = _linear_index(receiver_coord, mesh_device.shape)

    for call in range(2):
        input_tensor, input_shards = _make_input(mesh_device, (1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        output_tensor = point_to_point(input_tensor, sender_coord, receiver_coord, topology=topology)
        ttnn.synchronize_device(mesh_device)
        output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
        assert_with_pcc(input_shards[send_idx], output_shards[recv_idx], PCC[ttnn.bfloat16])
        logger.info(f"program-cache call {call} ({topology}): receiver shard matches sender shard")

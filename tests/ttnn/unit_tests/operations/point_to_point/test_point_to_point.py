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

Verification topology (the mesh_device fixture MUST match the sim descriptor)
-----------------------------------------------------------------------------
The op is verified on a simulated 8-chip Blackhole mesh of shape ``(2, 4)`` with
``fabric_config = ttnn.FabricConfig.FABRIC_1D`` — the ``bh_8xP150_p2p`` topology
in ``scripts/multidevice_sim_topologies.yaml`` (``required: true``). The sim's
mesh-graph descriptor is FIXED to that shape, so the ``mesh_device`` fixture MUST
open exactly ``(2, 4)`` with ``FABRIC_1D``; opening a different shape (e.g.
``(1, 2)``) hangs fabric init with "Fabric Router Sync: Timeout" — a
test/topology mismatch, not a sim or op defect.

The mesh shape + fabric_config are read from the environment the multi-device sim
runner (``scripts/run_multidevice_sim_pytest.py``) exports per topology
(``MULTIDEV_SIM_MESH_SHAPE`` / ``MULTIDEV_SIM_FABRIC_CONFIG``), defaulting to
``(2, 4)`` / ``FABRIC_1D`` when run standalone. This lets
``--op point_to_point`` fan the same test across every p2p topology in the matrix
(2x4, 4x2, 8x4 — all FABRIC_1D) while a plain ``pytest`` invocation opens the
required ``(2, 4)`` / ``FABRIC_1D`` mesh.

``sender_coord = (0, 0)`` and ``receiver_coord = (0, 1)`` are adjacent on row 0
(1 fabric hop) and valid in every mesh in the matrix; on FABRIC_1D both the Linear
and the Ring topology kwarg route the same single hop (a true Ring wraparound
needs FABRIC_1D_RING + non-adjacent coords, covered by test_p2p_ring_confirm.py).

Tests auto-skip on machines with too few devices.
"""

import os
from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.point_to_point import point_to_point


# Mesh shape + fabric_config: read from the multi-device sim runner's per-topology
# env, defaulting to the required bh_8xP150_p2p topology (2x4, FABRIC_1D). This is
# the load-bearing fixture contract — the shape MUST match the sim's mesh-graph
# descriptor or fabric init hangs ("Fabric Router Sync: Timeout").
_MESH = tuple(int(x) for x in os.environ.get("MULTIDEV_SIM_MESH_SHAPE", "2,4").split(","))
_FABRIC = getattr(ttnn.FabricConfig, os.environ.get("MULTIDEV_SIM_FABRIC_CONFIG", "FABRIC_1D"))

# Adjacent endpoints on row 0 (1 hop), valid in every matrix mesh.
SENDER = (0, 0)
RECEIVER = (0, 1)

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
# 16-byte aligned for every dtype (the op rejects non-16B-aligned pages).
SHARD_SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (1, 1, 96, 64),  # non-square, tile-aligned
    (2, 1, 32, 64),  # multi-batch
    (1, 1, 48, 64),  # non-tile-aligned (H not %32), 16B-aligned page
]

# Fabric topology is an op kwarg (Linear primary; Ring may route the short way).
# On FABRIC_1D with the adjacent coords above, both route the same single hop.
TOPOLOGIES = [ttnn.Topology.Linear, ttnn.Topology.Ring]


def _linear_index(coord, mesh_shape):
    """Row-major linear index of a (row, col) coordinate in a (rows, cols) mesh."""
    mesh_shape = tuple(mesh_shape)
    return coord[0] * mesh_shape[1] + coord[1]


def _make_input(mesh_device, shard_shape, dtype, layout):
    """Shard a freshly-seeded tensor along dim 0 across the mesh.

    Each device receives exactly ``shard_shape``. Returns the ttnn input tensor
    and the list of per-device shards (torch) in linear (row-major) mesh order.
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


@pytest.mark.parametrize("device_params", [{"fabric_config": _FABRIC}], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("dtype, layout", DTYPE_LAYOUTS)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_point_to_point(mesh_device, topology, dtype, layout, shard_shape):
    """Receiver shard == sender shard; every other shard (incl. sender) unchanged."""
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    sender_coord = ttnn.MeshCoordinate(*SENDER)
    receiver_coord = ttnn.MeshCoordinate(*RECEIVER)
    send_idx = _linear_index(SENDER, mesh_device.shape)
    recv_idx = _linear_index(RECEIVER, mesh_device.shape)

    input_tensor, input_shards = _make_input(mesh_device, shard_shape, dtype, layout)

    output_tensor = point_to_point(input_tensor, sender_coord, receiver_coord, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    pcc = PCC[dtype]
    # Receiver shard now holds the sender's input shard (the transfer).
    assert_with_pcc(input_shards[send_idx], output_shards[recv_idx], pcc)
    # Every other device's shard is unchanged — including the sender's own shard
    # (only the receiver shard is written; all others are seeded from the input).
    for i in range(len(output_shards)):
        if i == recv_idx:
            continue
        assert_with_pcc(input_shards[i], output_shards[i], pcc)
    logger.info(
        f"p2p {dtype} {layout} {shard_shape} {topology} mesh={tuple(mesh_device.shape)}: "
        "receiver shard matches sender shard; all other shards unchanged"
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": _FABRIC}], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize(
    "dtype, layout",
    [(ttnn.bfloat16, ttnn.TILE_LAYOUT), (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)],
)
def test_point_to_point_output_tensor(mesh_device, topology, dtype, layout):
    """The output_tensor path writes into the supplied tensor and returns it."""
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    sender_coord = ttnn.MeshCoordinate(*SENDER)
    receiver_coord = ttnn.MeshCoordinate(*RECEIVER)
    send_idx = _linear_index(SENDER, mesh_device.shape)
    recv_idx = _linear_index(RECEIVER, mesh_device.shape)

    input_tensor, input_shards = _make_input(mesh_device, (1, 1, 64, 128), dtype, layout)
    preallocated = ttnn.allocate_tensor_on_device(input_tensor.spec, mesh_device)

    returned = point_to_point(input_tensor, sender_coord, receiver_coord, topology=topology, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)

    # The supplied tensor handle is returned (write-into-preallocated path).
    assert returned.buffer_address() == preallocated.buffer_address()

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(returned)]
    assert_with_pcc(input_shards[send_idx], output_shards[recv_idx], PCC[dtype])


@pytest.mark.parametrize("device_params", [{"fabric_config": _FABRIC}], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_point_to_point_program_cache(mesh_device):
    """Second call (program-cache hit) still transfers correctly.

    The op-internal GlobalSemaphore must survive the cache hit (created once, not
    re-created per call), and each device's semaphore must be re-armed cleanly so
    the second run neither hangs nor corrupts.
    """
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    sender_coord = ttnn.MeshCoordinate(*SENDER)
    receiver_coord = ttnn.MeshCoordinate(*RECEIVER)
    send_idx = _linear_index(SENDER, mesh_device.shape)
    recv_idx = _linear_index(RECEIVER, mesh_device.shape)

    for call in range(2):
        input_tensor, input_shards = _make_input(mesh_device, (1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        output_tensor = point_to_point(input_tensor, sender_coord, receiver_coord, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(mesh_device)
        output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
        assert_with_pcc(input_shards[send_idx], output_shards[recv_idx], PCC[ttnn.bfloat16])
        logger.info(f"program-cache call {call}: receiver shard matches sender shard")

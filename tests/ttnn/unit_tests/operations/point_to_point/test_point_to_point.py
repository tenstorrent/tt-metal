# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test (immutable spec) for the self-contained Python point_to_point CCL op.

point_to_point sends one mesh device's interleaved shard to another device over
the Tenstorrent fabric. It is pure data movement (no arithmetic), so the oracle
is IDENTITY:

  * the receiver device's output shard  == the sender device's input shard,
  * every other device's output shard   == that device's input shard (unchanged),
    including the sender's own output shard (no kernel writes it).

The op requires a multi-device ``ttnn.MeshDevice`` with the fabric enabled.

VERIFICATION-TOPOLOGY PIN (HARD): every test opens EXACTLY a ``(2, 4)`` mesh with
``device_params = {fabric_config: FABRIC_1D}`` (the sim's mesh-graph descriptor is
fixed to this shape). A different mesh shape hangs fabric init with
``Fabric Router Sync: Timeout`` — a test/topology mismatch, not an op defect. The
``topology`` axis (Linear / Ring) is an OP KWARG: with the pinned adjacent
``(0,0) -> (0,1)`` 1-hop coords, Ring routes the short (line) way, so both
topologies run correctly on FABRIC_1D. Drive via scripts/run_multidevice_sim_pytest.py
whose topology mesh_shape MUST equal (2, 4).

The implementer must not modify this file — it is the spec.
"""

from __future__ import annotations

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.point_to_point import point_to_point


# ---------------------------------------------------------------------------
# Verification-topology pin — applies to EVERY test in this module.
# ---------------------------------------------------------------------------
MESH_SHAPE = (2, 4)

pytestmark = [
    pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True),
    pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True),
]

# Adjacent, 1-hop coords on row 0 — valid for both Linear and (short-way) Ring on FABRIC_1D.
SENDER_COORD = ttnn.MeshCoordinate(0, 0)
RECEIVER_COORD = ttnn.MeshCoordinate(0, 1)

# PCC tolerances keyed by dtype (same thresholds as the golden suite). p2p is a
# pure byte copy, so PCC is exactly 1.0 in practice; these are safety bands.
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

TOPOLOGIES = [ttnn.Topology.Linear, ttnn.Topology.Ring]


def _linear_index(coord, mesh_shape):
    """Row-major linear index of a MeshCoordinate in a (rows, cols) mesh."""
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


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("dtype, layout", DTYPE_LAYOUTS)
def test_point_to_point(mesh_device, dtype, layout, topology):
    """Every supported dtype/layout, on both Linear and Ring topology.

    Receiver shard == sender shard; sender's own shard unchanged; every other
    (non-participating) shard unchanged.
    """
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    send_idx = _linear_index(SENDER_COORD, mesh_device.shape)
    recv_idx = _linear_index(RECEIVER_COORD, mesh_device.shape)

    input_tensor, input_shards = _make_input(mesh_device, (1, 1, 64, 128), dtype, layout)

    output_tensor = point_to_point(input_tensor, SENDER_COORD, RECEIVER_COORD, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    pcc = PCC[dtype]
    # Receiver shard now holds the sender's input shard.
    assert_with_pcc(input_shards[send_idx], output_shards[recv_idx], pcc)
    # Every non-receiver shard (incl. the sender's own) is unchanged.
    for i in range(len(output_shards)):
        if i == recv_idx:
            continue
        assert_with_pcc(input_shards[i], output_shards[i], pcc)
    logger.info(f"p2p {dtype} {layout} {topology}: receiver==sender, all others unchanged")


@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_point_to_point_shapes(mesh_device, shard_shape):
    """Shape coverage: single-tile, multi-tile, non-square, multi-batch, non-tile-aligned."""
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    dtype, layout, topology = ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.Topology.Linear
    send_idx = _linear_index(SENDER_COORD, mesh_device.shape)
    recv_idx = _linear_index(RECEIVER_COORD, mesh_device.shape)

    input_tensor, input_shards = _make_input(mesh_device, shard_shape, dtype, layout)

    output_tensor = point_to_point(input_tensor, SENDER_COORD, RECEIVER_COORD, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    pcc = PCC[dtype]
    assert_with_pcc(input_shards[send_idx], output_shards[recv_idx], pcc)
    assert_with_pcc(input_shards[send_idx], output_shards[send_idx], pcc)  # sender unchanged
    logger.info(f"p2p {shard_shape}: receiver shard matches sender shard")


@pytest.mark.parametrize("dtype, layout", [(ttnn.bfloat16, ttnn.TILE_LAYOUT), (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)])
def test_point_to_point_output_tensor(mesh_device, dtype, layout):
    """The output_tensor path writes into the supplied tensor and returns it."""
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    send_idx = _linear_index(SENDER_COORD, mesh_device.shape)
    recv_idx = _linear_index(RECEIVER_COORD, mesh_device.shape)

    input_tensor, input_shards = _make_input(mesh_device, (1, 1, 64, 128), dtype, layout)
    preallocated = ttnn.allocate_tensor_on_device(input_tensor.spec, mesh_device)

    returned = point_to_point(
        input_tensor, SENDER_COORD, RECEIVER_COORD, topology=ttnn.Topology.Linear, output_tensor=preallocated
    )
    ttnn.synchronize_device(mesh_device)

    # Same handle is returned.
    assert returned.buffer_address() == preallocated.buffer_address()

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(returned)]
    assert_with_pcc(input_shards[send_idx], output_shards[recv_idx], PCC[dtype])


def test_point_to_point_program_cache(mesh_device):
    """Second call (program-cache hit) still transfers correctly.

    The op-internal GlobalSemaphore must survive the cache hit (created once,
    not re-created per call).
    """
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    send_idx = _linear_index(SENDER_COORD, mesh_device.shape)
    recv_idx = _linear_index(RECEIVER_COORD, mesh_device.shape)

    for call in range(2):
        input_tensor, input_shards = _make_input(mesh_device, (1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        output_tensor = point_to_point(input_tensor, SENDER_COORD, RECEIVER_COORD, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(mesh_device)
        output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
        assert_with_pcc(input_shards[send_idx], output_shards[recv_idx], PCC[ttnn.bfloat16])
        logger.info(f"program-cache call {call}: receiver shard matches sender shard")

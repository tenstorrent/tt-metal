# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the self-contained Python point_to_point CCL op.

point_to_point sends one mesh device's interleaved shard to another device over
the Tenstorrent fabric. It is pure data movement (no arithmetic), so the oracle
is identity:

  * the receiver device's output shard == the sender device's input shard,
  * every other device's output shard == that device's input shard (unchanged).

This file is the immutable spec — the implementer must not modify it.

Verification topology (MUST match scripts/run_multidevice_sim_pytest.py's graded
`bh_8xP150_p2p` entry): an 8-chip Blackhole mesh of shape (2, 4) with
`fabric_config = ttnn.FabricConfig.FABRIC_1D` (a torus-x mesh-graph descriptor).
Opening any other mesh shape (e.g. (1, 2)) hangs fabric init with
"Fabric Router Sync: Timeout" — a test/topology mismatch, not an op defect. So
every test here opens exactly (2, 4) + FABRIC_1D and picks sender/receiver coords
inside it, (0, 0) -> (0, 1).

Both Linear and Ring op-topologies are exercised under FABRIC_1D. For an adjacent
sender/receiver pair, `ccl_dm_route` resolves the Ring route to the same 1-hop
line route as Linear (the ring wraparound is only shorter for distant coords), so
Ring is safely routable under FABRIC_1D here; a genuine wraparound ring (which
needs FABRIC_1D_RING) is out of scope for the graded topology.
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.point_to_point import point_to_point


# ---------------------------------------------------------------------------
# Verification topology (fixed to the graded sim entry).
# ---------------------------------------------------------------------------
MESH_SHAPE = (2, 4)
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}

# Adjacent sender/receiver on row 0 (1-D route valid for Linear and Ring).
SENDER = ttnn.MeshCoordinate(0, 0)
RECEIVER = ttnn.MeshCoordinate(0, 1)

# PCC tolerances keyed by dtype (same thresholds as the golden suite).
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

# Integer/passthrough dtypes are compared bit-exactly (no PCC).
_INT_DTYPES = (ttnn.uint16, ttnn.int32, ttnn.uint32)


# Identity-transfer matrix: (topology, dtype, layout, shard_shape). Collectively
# covers both topologies, every supported dtype, both layouts, and shape
# diversity (single-tile, multi-tile, non-square, multi-batch, non-tile-aligned).
# All last dims are multiples of 8 so the row-major page stays 16-byte aligned.
CASES = [
    # --- Linear, float / block-float (PCC) ---
    (ttnn.Topology.Linear, ttnn.bfloat16, ttnn.TILE_LAYOUT, (1, 1, 32, 32)),  # single tile
    (ttnn.Topology.Linear, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 96, 64)),  # non-square, RM
    (ttnn.Topology.Linear, ttnn.float32, ttnn.TILE_LAYOUT, (1, 1, 64, 128)),  # multi-tile
    (ttnn.Topology.Linear, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 48, 64)),  # non-tile-aligned, RM
    (ttnn.Topology.Linear, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, (2, 1, 32, 64)),  # multi-batch, bf8b
    # --- Ring (adjacent coords -> same 1-hop route under FABRIC_1D) ---
    (ttnn.Topology.Ring, ttnn.bfloat16, ttnn.TILE_LAYOUT, (1, 1, 64, 128)),  # ring, multi-tile
    (ttnn.Topology.Ring, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 96, 64)),  # ring, RM
    (ttnn.Topology.Ring, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, (1, 1, 32, 32)),  # ring, bf8b
    # --- integer passthrough (bit-exact) ---
    (ttnn.Topology.Linear, ttnn.uint16, ttnn.TILE_LAYOUT, (1, 1, 32, 64)),
    (ttnn.Topology.Linear, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 32, 64)),
    (ttnn.Topology.Ring, ttnn.uint32, ttnn.TILE_LAYOUT, (1, 1, 64, 64)),
]


def _linear_index(coord, mesh_shape):
    """Row-major linear index of a MeshCoordinate in a (rows, cols) mesh."""
    mesh_shape = tuple(mesh_shape)
    return coord[0] * mesh_shape[1] + coord[1]


def _make_input(mesh_device, shard_shape, dtype, layout):
    """Shard a freshly-seeded tensor along dim 0 across the mesh.

    Each device receives exactly ``shard_shape``. Returns the ttnn input tensor
    and the list of per-device shards (torch) in linear mesh order.
    """
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(42)
    if dtype in _INT_DTYPES:
        # Positive values that fit in uint16 (and hence every integer dtype here).
        torch_full = torch.randint(0, 30000, full_shape, dtype=torch.int32)
    else:
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


def _compare(expected, actual, dtype):
    """Bit-exact for integer dtypes; PCC for float / block-float dtypes."""
    if dtype in _INT_DTYPES:
        assert torch.equal(expected, actual), f"{dtype}: shard bytes differ"
    else:
        assert_with_pcc(expected, actual, PCC[dtype])


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("topology, dtype, layout, shard_shape", CASES)
def test_point_to_point(mesh_device, topology, dtype, layout, shard_shape):
    """Receiver shard equals sender shard; sender's own shard is unchanged."""
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    send_idx = _linear_index(SENDER, mesh_device.shape)
    recv_idx = _linear_index(RECEIVER, mesh_device.shape)

    input_tensor, input_shards = _make_input(mesh_device, shard_shape, dtype, layout)

    output_tensor = point_to_point(input_tensor, SENDER, RECEIVER, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    # Receiver shard now holds the sender's input shard.
    _compare(input_shards[send_idx], output_shards[recv_idx], dtype)
    # Sender's own shard is unchanged.
    _compare(input_shards[send_idx], output_shards[send_idx], dtype)
    logger.info(f"p2p {dtype} {layout} {shard_shape} {topology}: receiver shard matches sender shard")


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "topology, dtype, layout",
    [
        (ttnn.Topology.Linear, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        (ttnn.Topology.Ring, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
def test_point_to_point_nonparticipating_unchanged(mesh_device, topology, dtype, layout):
    """On the full mesh, every non-receiver shard stays equal to its input."""
    if prod(tuple(mesh_device.shape)) < 3:
        pytest.skip("this test needs at least 3 mesh devices to observe a bystander")

    send_idx = _linear_index(SENDER, mesh_device.shape)
    recv_idx = _linear_index(RECEIVER, mesh_device.shape)

    input_tensor, input_shards = _make_input(mesh_device, (1, 1, 64, 64), dtype, layout)

    output_tensor = point_to_point(input_tensor, SENDER, RECEIVER, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    # Receiver got the sender's shard.
    _compare(input_shards[send_idx], output_shards[recv_idx], dtype)
    # Every other device (including the sender) is untouched.
    for i in range(len(output_shards)):
        if i == recv_idx:
            continue
        _compare(input_shards[i], output_shards[i], dtype)


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "topology, dtype, layout",
    [
        (ttnn.Topology.Linear, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        (ttnn.Topology.Linear, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
def test_point_to_point_output_tensor(mesh_device, topology, dtype, layout):
    """The output_tensor path writes into the supplied tensor and returns it."""
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    send_idx = _linear_index(SENDER, mesh_device.shape)
    recv_idx = _linear_index(RECEIVER, mesh_device.shape)

    input_tensor, input_shards = _make_input(mesh_device, (1, 1, 64, 128), dtype, layout)
    preallocated = ttnn.allocate_tensor_on_device(input_tensor.spec, mesh_device)

    returned = point_to_point(input_tensor, SENDER, RECEIVER, topology=topology, output_tensor=preallocated)
    ttnn.synchronize_device(mesh_device)

    # Same handle is returned.
    assert returned.buffer_address() == preallocated.buffer_address()

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(returned)]
    _compare(input_shards[send_idx], output_shards[recv_idx], dtype)


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_point_to_point_program_cache(mesh_device):
    """Second call (program-cache hit) still transfers correctly.

    The op-internal GlobalSemaphore must survive the cache hit (created once,
    not re-created per call).
    """
    if prod(tuple(mesh_device.shape)) < 2:
        pytest.skip("point_to_point requires at least 2 mesh devices")

    send_idx = _linear_index(SENDER, mesh_device.shape)
    recv_idx = _linear_index(RECEIVER, mesh_device.shape)

    for call in range(2):
        input_tensor, input_shards = _make_input(mesh_device, (1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        output_tensor = point_to_point(input_tensor, SENDER, RECEIVER, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(mesh_device)
        output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
        assert_with_pcc(input_shards[send_idx], output_shards[recv_idx], PCC[ttnn.bfloat16])
        logger.info(f"program-cache call {call}: receiver shard matches sender shard")

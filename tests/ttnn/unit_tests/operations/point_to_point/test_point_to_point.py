# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the self-contained Python point_to_point CCL op.

``point_to_point`` copies one mesh device's interleaved shard to another device over
the TT-Fabric. It performs no arithmetic, so the PyTorch reference is identity:

  * the receiver device's output shard == the sender device's input shard,
  * every other device's output shard is untouched,
  * the input tensor is not mutated anywhere.

**This file is the immutable spec — the implementer must not modify it.**

Verification topology
---------------------
The op is graded on real Blackhole hardware via::

    scripts/run_multidevice_sim_pytest.py --runtime hardware --op point_to_point

whose topology is ``bh_quietbox_1x4_hw``: a 4-chip Blackhole mesh of shape ``(1, 4)``
with ``fabric_config = ttnn.FabricConfig.FABRIC_1D``. Every test below opens exactly
that mesh. Opening any other shape either hangs fabric init with
``Fabric Router Sync: Timeout`` or trips ``system_mesh.cpp: requested_size <= system_size``.
Do not change ``MESH_SHAPE`` or ``FABRIC``.

Ring-topology note: ``ttnn.Topology.Ring`` makes the router take the short way around,
which on a 4-device line means a coordinate pair more than 2 hops apart resolves to a
wraparound link that ``FABRIC_1D`` (non-ring) does not provide. Ring is therefore
exercised only with pairs at most 2 hops apart; a true wraparound needs
``FabricConfig.FABRIC_1D_RING``.
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.point_to_point import point_to_point

# --------------------------------------------------------------------------------------
# Verification topology — the contract with scripts/multidevice_sim_topologies.yaml
# --------------------------------------------------------------------------------------
MESH_SHAPE = (1, 4)
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


# PCC tolerances keyed by dtype (identical to the golden suite's thresholds).
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

# Per-device shard shapes. Every last dim is a multiple of 8 so the row-major page
# (last_dim * element_size) stays 16-byte aligned for every dtype, which is what the
# op requires. The last two entries deliberately produce row-major pages that are NOT
# a multiple of the 64-byte Blackhole DRAM alignment (96 B and 48 B for bfloat16) —
# the case that catches a TensorAccessor built with a raw page-size override.
SHARD_SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (1, 1, 96, 64),  # non-square, tile-aligned
    (2, 1, 32, 64),  # multi-batch
    (1, 1, 48, 64),  # non-tile-aligned H, 64B-aligned row
    (1, 1, 32, 48),  # non-tile-aligned W, 96 B row (not 64B-aligned)
    (1, 1, 24, 24),  # both dims non-tile-aligned, 48 B row (not 64B-aligned)
]

TOPOLOGIES = [ttnn.Topology.Linear, ttnn.Topology.Ring]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _linear_index(coord, mesh_shape):
    """Row-major linear index of a MeshCoordinate in a (rows, cols) mesh."""
    return coord[0] * tuple(mesh_shape)[1] + coord[1]


def _require_mesh(mesh_device):
    if prod(tuple(mesh_device.shape)) < prod(MESH_SHAPE):
        pytest.skip(f"point_to_point acceptance needs a {MESH_SHAPE} mesh")


def _shard_tensor(mesh_device, shard_shape, dtype, layout, seed=42):
    """Build a mesh-sharded tensor whose per-device shard is exactly ``shard_shape``.

    Returns ``(ttnn_tensor, [per-device torch shards])`` in linear mesh order.
    """
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(seed)
    torch_full = torch.randn(full_shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)

    tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return tensor, _read_shards(tensor)


def _read_shards(tensor):
    """Per-device shards of a mesh tensor as float32 torch tensors, in mesh order."""
    return [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(tensor)]


def torch_point_to_point(input_shards, send_idx, recv_idx):
    """PyTorch reference: pure data movement, no arithmetic.

    The receiver's shard becomes the sender's shard; every other shard is untouched.
    """
    expected = [s.clone() for s in input_shards]
    expected[recv_idx] = input_shards[send_idx].clone()
    return expected


def _assert_shards(actual_shards, expected_shards, pcc, label):
    assert len(actual_shards) == len(expected_shards)
    for actual, expected in zip(actual_shards, expected_shards):
        assert_with_pcc(expected, actual, pcc)
    logger.info(f"{label}: all {len(actual_shards)} device shards match the reference")


# --------------------------------------------------------------------------------------
# 1. Core correctness across dtype x layout x shape x topology
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("dtype, layout", DTYPE_LAYOUTS)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_point_to_point(mesh_device, topology, dtype, layout, shard_shape):
    """Receiver shard == sender shard; every other shard untouched; input unmutated."""
    _require_mesh(mesh_device)

    sender_coord = ttnn.MeshCoordinate(0, 0)
    receiver_coord = ttnn.MeshCoordinate(0, 1)
    send_idx = _linear_index(sender_coord, mesh_device.shape)
    recv_idx = _linear_index(receiver_coord, mesh_device.shape)

    input_tensor, input_shards = _shard_tensor(mesh_device, shard_shape, dtype, layout)

    output_tensor = point_to_point(input_tensor, sender_coord, receiver_coord, topology=topology)
    ttnn.synchronize_device(mesh_device)

    expected = torch_point_to_point(input_shards, send_idx, recv_idx)
    _assert_shards(_read_shards(output_tensor), expected, PCC[dtype], f"p2p {dtype} {layout} {shard_shape} {topology}")

    # The op must not mutate its input on any device.
    _assert_shards(_read_shards(input_tensor), input_shards, PCC[dtype], "input unchanged")

    # Shape / dtype / layout are preserved end to end.
    assert list(output_tensor.shape) == list(input_tensor.shape)
    assert output_tensor.dtype == input_tensor.dtype
    assert output_tensor.layout == input_tensor.layout


# --------------------------------------------------------------------------------------
# 2. Routing: hop count, direction, and both topologies
# --------------------------------------------------------------------------------------
# (sender, receiver, topology). Ring pairs are limited to <= 2 hops so the short way
# round is still the line way — a longer Ring pair would route over a wraparound link
# that FABRIC_1D does not provide.
ROUTES = [
    ((0, 0), (0, 1), ttnn.Topology.Linear),  # 1 hop, forward
    ((0, 2), (0, 1), ttnn.Topology.Linear),  # 1 hop, backward
    ((0, 0), (0, 2), ttnn.Topology.Linear),  # 2 hops, forward
    ((0, 3), (0, 0), ttnn.Topology.Linear),  # 3 hops, backward
    ((0, 0), (0, 3), ttnn.Topology.Linear),  # 3 hops, forward
    ((0, 0), (0, 1), ttnn.Topology.Ring),  # 1 hop, short way == line way
    ((0, 1), (0, 3), ttnn.Topology.Ring),  # 2 hops, short way == line way
]


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype, layout", [(ttnn.bfloat16, ttnn.TILE_LAYOUT), (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)])
@pytest.mark.parametrize("send, recv, topology", ROUTES)
def test_point_to_point_routes(mesh_device, send, recv, topology, dtype, layout):
    """Multi-hop and both fabric directions land the shard on the right device."""
    _require_mesh(mesh_device)

    sender_coord = ttnn.MeshCoordinate(*send)
    receiver_coord = ttnn.MeshCoordinate(*recv)
    send_idx = _linear_index(sender_coord, mesh_device.shape)
    recv_idx = _linear_index(receiver_coord, mesh_device.shape)

    input_tensor, input_shards = _shard_tensor(mesh_device, (1, 1, 64, 64), dtype, layout)

    output_tensor = point_to_point(input_tensor, sender_coord, receiver_coord, topology=topology)
    ttnn.synchronize_device(mesh_device)

    expected = torch_point_to_point(input_shards, send_idx, recv_idx)
    _assert_shards(_read_shards(output_tensor), expected, PCC[dtype], f"route {send}->{recv} {topology}")


# --------------------------------------------------------------------------------------
# 3. output_tensor path — writes into the supplied tensor and returns it
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype, layout", [(ttnn.bfloat16, ttnn.TILE_LAYOUT), (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)])
def test_point_to_point_output_tensor(mesh_device, dtype, layout):
    """The supplied output tensor is written in place, returned, and only the
    receiver device's shard changes."""
    _require_mesh(mesh_device)

    sender_coord = ttnn.MeshCoordinate(0, 0)
    receiver_coord = ttnn.MeshCoordinate(0, 2)
    send_idx = _linear_index(sender_coord, mesh_device.shape)
    recv_idx = _linear_index(receiver_coord, mesh_device.shape)

    shard_shape = (1, 1, 64, 128)
    input_tensor, input_shards = _shard_tensor(mesh_device, shard_shape, dtype, layout, seed=42)
    # A distinct pre-fill so "untouched" is observable (different seed => different values).
    preallocated, prefill_shards = _shard_tensor(mesh_device, shard_shape, dtype, layout, seed=1234)

    returned = point_to_point(
        input_tensor,
        sender_coord,
        receiver_coord,
        output_tensor=preallocated,
    )
    ttnn.synchronize_device(mesh_device)

    # The same buffer is returned, not a fresh allocation.
    assert returned.buffer_address() == preallocated.buffer_address()

    # Only the receiver's shard changed; every other shard keeps its pre-fill.
    expected = [s.clone() for s in prefill_shards]
    expected[recv_idx] = input_shards[send_idx].clone()
    _assert_shards(_read_shards(returned), expected, PCC[dtype], "output_tensor path")


# --------------------------------------------------------------------------------------
# 4. Program cache — repeated calls with identical shape/dtype/coords/topology
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_point_to_point_program_cache(mesh_device):
    """Calls 2 and 3 are program-cache hits and must still transfer correctly.

    The op-internal GlobalSemaphore is created once and must survive the cache hit; a
    re-created or dead semaphore shows up here as a hang or a stale-data mismatch.
    """
    _require_mesh(mesh_device)

    sender_coord = ttnn.MeshCoordinate(0, 0)
    receiver_coord = ttnn.MeshCoordinate(0, 1)
    send_idx = _linear_index(sender_coord, mesh_device.shape)
    recv_idx = _linear_index(receiver_coord, mesh_device.shape)

    for call in range(3):
        input_tensor, input_shards = _shard_tensor(
            mesh_device, (1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, seed=42 + call
        )
        output_tensor = point_to_point(input_tensor, sender_coord, receiver_coord)
        ttnn.synchronize_device(mesh_device)

        expected = torch_point_to_point(input_shards, send_idx, recv_idx)
        _assert_shards(_read_shards(output_tensor), expected, PCC[ttnn.bfloat16], f"program-cache call {call}")


# --------------------------------------------------------------------------------------
# 5. Validation
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_point_to_point_rejects_self_send(mesh_device):
    """sender_coord == receiver_coord is an error."""
    _require_mesh(mesh_device)
    input_tensor, _ = _shard_tensor(mesh_device, (1, 1, 32, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    coord = ttnn.MeshCoordinate(0, 1)
    with pytest.raises(ValueError):
        point_to_point(input_tensor, coord, coord)


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_point_to_point_rejects_out_of_mesh_coord(mesh_device):
    """A coordinate outside the mesh is an error, for either endpoint."""
    _require_mesh(mesh_device)
    input_tensor, _ = _shard_tensor(mesh_device, (1, 1, 32, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    inside = ttnn.MeshCoordinate(0, 0)
    outside = ttnn.MeshCoordinate(0, tuple(mesh_device.shape)[1] + 3)

    with pytest.raises(ValueError):
        point_to_point(input_tensor, inside, outside)
    with pytest.raises(ValueError):
        point_to_point(input_tensor, outside, inside)


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_point_to_point_rejects_mismatched_output_tensor(mesh_device):
    """An output_tensor whose spec differs from the resolved output spec is an error."""
    _require_mesh(mesh_device)
    input_tensor, _ = _shard_tensor(mesh_device, (1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)

    wrong_shape, _ = _shard_tensor(mesh_device, (1, 1, 64, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    with pytest.raises(ValueError):
        point_to_point(
            input_tensor,
            ttnn.MeshCoordinate(0, 0),
            ttnn.MeshCoordinate(0, 1),
            output_tensor=wrong_shape,
        )

    wrong_dtype, _ = _shard_tensor(mesh_device, (1, 1, 32, 64), ttnn.float32, ttnn.TILE_LAYOUT)
    with pytest.raises(ValueError):
        point_to_point(
            input_tensor,
            ttnn.MeshCoordinate(0, 0),
            ttnn.MeshCoordinate(0, 1),
            output_tensor=wrong_dtype,
        )

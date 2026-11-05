# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.from_local_host_shards`` assembles a mesh host tensor from this host's shards under a given label.

``from_host_shards`` needs one shard for every coordinate of the mesh and spells a 1-D label of its own. A loader
that places a checkpoint shard by shard wants to hand in the coordinates this host holds, no more, with the exact
N-D ``TensorTopology`` the live tensor carries. On a single-host mesh every coordinate is local, which is what this
file can test; the remote-coordinate case needs a multi-host mesh.
"""

import pytest
import torch

import ttnn


def _names(topology):
    return [f"Shard({p.dim})" if isinstance(p, ttnn.PlacementShard) else "Replicate" for p in topology.placements()]


def _two_by_two(mesh_device):
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape != (2, 2):
        pytest.skip(f"needs a [2, 2] mesh, the fixture opened {mesh_shape}")
    full = torch.randn(1, 1, 64, 64).to(torch.bfloat16)
    coords = [ttnn.MeshCoordinate(r, c) for r in range(2) for c in range(2)]
    # Rows split across mesh rows, columns across mesh columns: the label is [Shard(2), Shard(3)].
    pieces = {
        ttnn.MeshCoordinate(r, c): full[:, :, 32 * r : 32 * (r + 1), 32 * c : 32 * (c + 1)].contiguous()
        for r in range(2)
        for c in range(2)
    }
    shards = {
        coord: ttnn.from_torch(piece, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for coord, piece in pieces.items()
    }
    topology = ttnn.TensorTopology(ttnn.MeshShape(2, 2), [ttnn.PlacementShard(2), ttnn.PlacementShard(3)], coords)
    return full, pieces, shards, topology


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_from_local_host_shards_places_each_shard_at_its_coordinate_under_the_given_label(mesh_device):
    full, pieces, shards, topology = _two_by_two(mesh_device)

    host = ttnn.from_local_host_shards(shards, mesh_device, topology)

    assert _names(host.tensor_topology()) == ["Shard(2)", "Shard(3)"]
    assert list(host.tensor_topology().distribution_shape()) == [2, 2]
    assert list(host.shape) == [1, 1, 32, 32]
    for coord in pieces:
        assert host.host_buffer().get_shard(coord) is not None, f"no shard at {coord}"

    on_device = host.to(mesh_device)
    assert _names(on_device.tensor_topology()) == ["Shard(2)", "Shard(3)"], "the label survives the move"
    for shard, coord in zip(ttnn.get_device_tensors(on_device), on_device.device_coords()):
        torch.testing.assert_close(ttnn.to_torch(shard), pieces[coord], rtol=0, atol=0)
    # The composer rebuilds the whole tensor from the label, so a misplaced shard would not round-trip.
    composed = ttnn.to_torch(
        on_device, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(2, 2), dims=(2, 3))
    )
    torch.testing.assert_close(composed, full, rtol=0, atol=0)


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_from_local_host_shards_rejects_incomplete_or_foreign_shards(mesh_device, expect_error):
    _, _, shards, topology = _two_by_two(mesh_device)
    origin = ttnn.MeshCoordinate(0, 0)

    with expect_error(RuntimeError, "at least one shard"):
        ttnn.from_local_host_shards({}, mesh_device, topology)

    missing = dict(shards)
    missing.pop(ttnn.MeshCoordinate(1, 1))
    with expect_error(RuntimeError, "no shard was given"):
        ttnn.from_local_host_shards(missing, mesh_device, topology)

    foreign = dict(shards)
    foreign[ttnn.MeshCoordinate(5, 5)] = shards[origin]
    with expect_error(RuntimeError, "not inside the mesh"):
        ttnn.from_local_host_shards(foreign, mesh_device, topology)

    mismatched = dict(shards)
    mismatched[origin] = ttnn.from_torch(torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "different tensor spec"):
        ttnn.from_local_host_shards(mismatched, mesh_device, topology)

    on_device = dict(shards)
    on_device[origin] = shards[origin].to(mesh_device)
    with expect_error(RuntimeError, "not on host"):
        ttnn.from_local_host_shards(on_device, mesh_device, topology)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.from_local_host_shards`` assembles a mesh host tensor from this host's shards under a given label.

``from_host_shards`` needs one shard for every coordinate of the mesh and spells a 1-D label of its own. A loader
that places a checkpoint shard by shard wants to hand in the coordinates this host holds, no more, with the exact
N-D ``TensorTopology`` the live tensor carries. On a single-host mesh every coordinate is local, which is what this
file can test; the remote-coordinate case needs a multi-host mesh.

The round trip and the refusals run on a ``[1, 1]`` mesh, so single-chip CI runs them, and on a ``[2, 2]`` mesh, where
the refusals that need a second coordinate run too; a label covering part of the mesh needs ``[2, 2]``.
"""

import pytest
import torch

import ttnn


def _names(topology):
    return [f"Shard({p.dim})" if isinstance(p, ttnn.PlacementShard) else "Replicate" for p in topology.placements()]


def _whole_mesh(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    full = torch.randn(1, 1, 32 * rows, 32 * cols).to(torch.bfloat16)
    coords = [ttnn.MeshCoordinate(r, c) for r in range(rows) for c in range(cols)]
    # Rows split across mesh rows, columns across mesh columns: the label is [Shard(2), Shard(3)].
    pieces = {
        ttnn.MeshCoordinate(r, c): full[:, :, 32 * r : 32 * (r + 1), 32 * c : 32 * (c + 1)].contiguous()
        for r in range(rows)
        for c in range(cols)
    }
    shards = {
        coord: ttnn.from_torch(piece, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for coord, piece in pieces.items()
    }
    topology = ttnn.TensorTopology(ttnn.MeshShape(rows, cols), [ttnn.PlacementShard(2), ttnn.PlacementShard(3)], coords)
    return full, pieces, shards, topology


@pytest.mark.parametrize("mesh_device", [(1, 1), (2, 2)], indirect=True)
def test_from_local_host_shards_places_each_shard_at_its_coordinate_under_the_given_label(mesh_device):
    mesh_shape = tuple(mesh_device.shape)
    full, pieces, shards, topology = _whole_mesh(mesh_device)

    host = ttnn.from_local_host_shards(shards, mesh_device, topology)

    assert _names(host.tensor_topology()) == ["Shard(2)", "Shard(3)"]
    assert list(host.tensor_topology().distribution_shape()) == list(mesh_shape)
    assert list(host.shape) == [1, 1, 32, 32]
    for coord in pieces:
        assert host.host_buffer().get_shard(coord) is not None, f"no shard at {coord}"

    on_device = host.to(mesh_device)
    assert _names(on_device.tensor_topology()) == ["Shard(2)", "Shard(3)"], "the label survives the move"
    for shard, coord in zip(ttnn.get_device_tensors(on_device), on_device.device_coords()):
        torch.testing.assert_close(ttnn.to_torch(shard), pieces[coord], rtol=0, atol=0)
    # The composer rebuilds the whole tensor from the label, so a misplaced shard would not round-trip.
    composed = ttnn.to_torch(
        on_device, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(2, 3))
    )
    torch.testing.assert_close(composed, full, rtol=0, atol=0)


@pytest.mark.parametrize("mesh_device", [(1, 1), (2, 2)], indirect=True)
def test_from_local_host_shards_rejects_incomplete_or_foreign_shards(mesh_device, expect_error):
    mesh_shape = tuple(mesh_device.shape)
    _, _, shards, topology = _whole_mesh(mesh_device)
    coords = list(shards)
    origin = ttnn.MeshCoordinate(0, 0)

    with expect_error(RuntimeError, "at least one shard"):
        ttnn.from_local_host_shards({}, mesh_device, topology)

    foreign = dict(shards)
    foreign[ttnn.MeshCoordinate(5, 5)] = shards[origin]
    with expect_error(RuntimeError, "not inside the mesh"):
        ttnn.from_local_host_shards(foreign, mesh_device, topology)

    on_device = dict(shards)
    on_device[origin] = shards[origin].to(mesh_device)
    with expect_error(RuntimeError, "not on host"):
        ttnn.from_local_host_shards(on_device, mesh_device, topology)

    # A coordinate with the wrong number of dimensions. The dict holds only that key: coordinates of different
    # dimensions cannot be ordered, so a mixed dict fails in the dict conversion before the function runs.
    with expect_error(RuntimeError, "not inside the mesh"):
        ttnn.from_local_host_shards({ttnn.MeshCoordinate(0, 0, 0): shards[origin]}, mesh_device, topology)

    # A label that lists a coordinate outside the mesh, in place of one of the shards' coordinates.
    outside = ttnn.TensorTopology(
        ttnn.MeshShape(*mesh_shape),
        [ttnn.PlacementShard(2), ttnn.PlacementShard(3)],
        coords[:-1] + [ttnn.MeshCoordinate(5, 5)],
    )
    with expect_error(RuntimeError, "not inside the mesh"):
        ttnn.from_local_host_shards(shards, mesh_device, outside)

    if len(coords) == 1:
        # The rest need a second coordinate: a missing shard, a spec that differs from another shard's, a mesh host
        # tensor (from a [1, 1] mesh it holds one shard, so it is a valid one), and a label that omits a shard.
        return

    missing = dict(shards)
    missing.pop(coords[-1])
    with expect_error(RuntimeError, "no shard was given"):
        ttnn.from_local_host_shards(missing, mesh_device, topology)

    mismatched = dict(shards)
    mismatched[origin] = ttnn.from_torch(torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "different tensor spec"):
        ttnn.from_local_host_shards(mismatched, mesh_device, topology)

    # A mesh host tensor is on host too, but is not a single shard.
    mesh_host = dict(shards)
    mesh_host[origin] = shards[origin].to(mesh_device).cpu()
    with expect_error(RuntimeError, "multi-device"):
        ttnn.from_local_host_shards(mesh_host, mesh_device, topology)

    # The label is the caller's; one that does not list a populated coordinate is refused at assembly rather than
    # when the tensor is composed or dumped.
    wrong_label = ttnn.TensorTopology(
        ttnn.MeshShape(*mesh_shape), [ttnn.PlacementShard(2), ttnn.PlacementShard(3)], [origin] * len(coords)
    )
    with expect_error(RuntimeError, "does not cover"):
        ttnn.from_local_host_shards(shards, mesh_device, wrong_label)


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_from_local_host_shards_assembles_a_tensor_that_lives_on_part_of_the_mesh(mesh_device, expect_error):
    """A label can cover part of the mesh, as a mapper's ``mesh_shape_override`` makes one: only the coordinates it
    lists take a shard, and the rest of the mesh stays unpopulated."""
    _, pieces, shards, _ = _whole_mesh(mesh_device)
    row = [ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 1)]
    # The top half of the tensor, its columns split across mesh row 0: [Replicate, Shard(3)] over a [1, 2] distribution.
    topology = ttnn.TensorTopology(ttnn.MeshShape(1, 2), [ttnn.PlacementReplicate(), ttnn.PlacementShard(3)], row)
    on_row = {coord: shards[coord] for coord in row}

    host = ttnn.from_local_host_shards(on_row, mesh_device, topology)

    assert _names(host.tensor_topology()) == ["Replicate", "Shard(3)"]
    assert list(host.tensor_topology().distribution_shape()) == [1, 2]
    for coord in pieces:
        populated = host.host_buffer().get_shard(coord) is not None
        assert populated == (coord in row), f"{coord}: populated {populated}"
    on_device = host.to(mesh_device)
    assert sorted(map(str, on_device.device_coords())) == sorted(map(str, row)), "only the labelled devices hold it"
    for shard, coord in zip(ttnn.get_device_tensors(on_device), on_device.device_coords()):
        torch.testing.assert_close(ttnn.to_torch(shard), pieces[coord], rtol=0, atol=0)

    with expect_error(RuntimeError, "no shard was given"):
        ttnn.from_local_host_shards({row[0]: shards[row[0]]}, mesh_device, topology)
    unlabelled = ttnn.MeshCoordinate(1, 0)
    with expect_error(RuntimeError, "does not cover"):
        ttnn.from_local_host_shards({**on_row, unlabelled: shards[unlabelled]}, mesh_device, topology)
    outside = ttnn.TensorTopology(
        ttnn.MeshShape(1, 2), [ttnn.PlacementReplicate(), ttnn.PlacementShard(3)], [row[0], ttnn.MeshCoordinate(5, 5)]
    )
    with expect_error(RuntimeError, "not inside the mesh"):
        ttnn.from_local_host_shards(on_row, mesh_device, outside)

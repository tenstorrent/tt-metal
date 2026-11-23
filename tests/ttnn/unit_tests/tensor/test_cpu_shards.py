# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``Tensor.cpu_shards(coords)``, or ``ttnn.cpu_shards(tensor, coords)``, reads a subset of a mesh tensor's shards,
one single-device host tensor each.

A checkpoint writer that deduplicates replicas wants one shard per replica group off the device, not the whole
mesh. Building that subset through ``combine_device_tensors`` relabels the mesh tensor the live parameter shares
(``DeviceStorage::combine_device_storages`` stamps a 1-D ``Shard(0)`` label on it), so the subset read is its own
path: one command-queue batch, returned as one single-device host tensor per coordinate, as ``get_device_tensors``
returns a host tensor's shards. No result lists shards it does not hold, and the device tensor keeps its label.

The round trips run on a ``[1, 1]`` mesh, so single-chip CI runs them, and on a ``[2, 2]`` mesh; the refusals use
whatever mesh the fixture opens.
"""

import numpy as np
import pytest
import torch

import ttnn


def _label(topology):
    names = [f"Shard({p.dim})" if isinstance(p, ttnn.PlacementShard) else "Replicate" for p in topology.placements()]
    return names, list(topology.distribution_shape()), list(topology.mesh_coords())


def _bytes(host_tensor, coord=None):
    """The bytes of a mesh host tensor's shard at `coord`, or of a single-device host tensor's one shard."""
    shard = host_tensor.host_buffer().get_shard(coord if coord is not None else ttnn.MeshCoordinate(0, 0))
    assert shard is not None, f"no shard at {coord}"
    return np.from_dlpack(shard).tobytes()


def _all_coords(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    return [ttnn.MeshCoordinate(r, c) for r in range(rows) for c in range(cols)]


@pytest.mark.parametrize("mesh_device", [(1, 1), (2, 2)], indirect=True)
def test_cpu_shards_returns_one_host_tensor_per_requested_coordinate_in_order(mesh_device):
    rows, cols = mesh_shape = tuple(mesh_device.shape)

    # [Replicate, Shard(2)]: mesh column c holds rows [32c, 32c + 32); every mesh row holds the same shards.
    full = torch.randn(1, 1, 32 * cols, 64).to(torch.bfloat16)
    tensor = ttnn.from_torch(
        full,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=list(mesh_shape), dims=(None, 2)),
    )
    before = _label(tensor.tensor_topology())
    assert before[0] == ["Replicate", "Shard(2)"], "precondition: a 2-D label"
    reference = tensor.cpu()

    # One replica of each column, last column first: on [2, 2], (1, 1) then (0, 0), not in mesh order.
    owned = [ttnn.MeshCoordinate(c % rows, c) for c in reversed(range(cols))]
    shards = tensor.cpu_shards(owned)

    assert len(shards) == len(owned)
    assert [_bytes(s) for s in ttnn.cpu_shards(tensor, owned)] == [_bytes(s) for s in shards], "the function form"
    for shard, coord in zip(shards, owned):
        assert _bytes(shard) == _bytes(reference, coord), f"the result for {coord} holds that coordinate's shard"
        assert len(shard.tensor_topology().mesh_coords()) == 1, "a single-device tensor, labelled as one device"
        # Each result is a whole tensor of its own: it converts without a composer.
        col = coord[1]
        torch.testing.assert_close(ttnn.to_torch(shard), full[:, :, 32 * col : 32 * (col + 1), :], rtol=0, atol=0)
    assert _label(tensor.tensor_topology()) == before, "the device tensor's label is untouched"
    whole = tensor.cpu()
    assert all(whole.host_buffer().get_shard(c) is not None for c in _all_coords(mesh_device)), "a later read is whole"


@pytest.mark.parametrize("mesh_device", [(1, 1), (2, 2)], indirect=True)
def test_cpu_shards_reads_one_replica_of_a_replicated_tensor(mesh_device):
    rows = tuple(mesh_device.shape)[0]

    expected = torch.randn(1, 1, 32, 32).to(torch.bfloat16)
    tensor = ttnn.from_torch(
        expected, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    before = _label(tensor.tensor_topology())

    (shard,) = tensor.cpu_shards([ttnn.MeshCoordinate(rows - 1, 0)])

    torch.testing.assert_close(ttnn.to_torch(shard), expected, rtol=0, atol=0)
    assert _label(tensor.tensor_topology()) == before


def test_cpu_shards_rejects_what_the_tensor_does_not_hold(mesh_device, expect_error):
    tensor = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    first = tensor.device_coords()[0]

    with expect_error(RuntimeError, "at least one coordinate"):
        tensor.cpu_shards([])
    with expect_error(RuntimeError, "listed twice"):
        tensor.cpu_shards([first, first])
    with expect_error(RuntimeError, "listed twice"):
        ttnn.cpu_shards(tensor, [first, first])
    with expect_error(RuntimeError, "holds no shard"):
        tensor.cpu_shards([ttnn.MeshCoordinate(99, 99)])
    with expect_error(RuntimeError, "needs a device tensor"):
        tensor.cpu().cpu_shards([first])

    if len(tensor.device_coords()) > 1:
        # A shard taken off the tensor holds one coordinate; asking it for another is refused.
        shard = ttnn.get_device_tensors(tensor)[0]
        with expect_error(RuntimeError, "holds no shard"):
            shard.cpu_shards([tensor.device_coords()[-1]])
        (host,) = shard.cpu_shards([first])
        assert _bytes(host) == _bytes(shard.cpu())

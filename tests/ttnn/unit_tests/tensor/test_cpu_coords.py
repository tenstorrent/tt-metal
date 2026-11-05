# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``Tensor.cpu(coords=...)`` reads a subset of a mesh tensor's shards without touching its topology label.

A checkpoint writer that deduplicates replicas wants one shard per replica group off the device, not the whole
mesh. Building that subset through ``combine_device_tensors`` relabels the mesh tensor the live parameter shares
(``DeviceStorage::combine_device_storages`` stamps a 1-D ``Shard(0)`` label on it), so the subset read is its own
path: the host tensor is populated at the requested coordinates only and carries the device tensor's own label,
which the device tensor keeps.

The 2-D cases open a ``[2, 2]`` mesh; the rest use whatever mesh the fixture opens.
"""

import numpy as np
import pytest
import torch

import ttnn


def _label(topology):
    names = [f"Shard({p.dim})" if isinstance(p, ttnn.PlacementShard) else "Replicate" for p in topology.placements()]
    return names, list(topology.distribution_shape()), list(topology.mesh_coords())


def _shard_bytes(host_tensor, coord):
    shard = host_tensor.host_buffer().get_shard(coord)
    assert shard is not None, f"no shard at {coord}"
    return np.from_dlpack(shard).tobytes()


def _all_coords(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    return [ttnn.MeshCoordinate(r, c) for r in range(rows) for c in range(cols)]


def _populated(host_tensor, mesh_device):
    return [c for c in _all_coords(mesh_device) if host_tensor.host_buffer().get_shard(c) is not None]


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_cpu_coords_reads_only_the_requested_shards_and_keeps_the_label(mesh_device):
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape != (2, 2):
        pytest.skip(f"needs a [2, 2] mesh, the fixture opened {mesh_shape}")

    # [Replicate, Shard(2)]: mesh column c holds rows [32c, 32c + 32); both mesh rows hold the same two shards.
    tensor = ttnn.from_torch(
        torch.randn(1, 1, 64, 64).to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=list(mesh_shape), dims=(None, 2)),
    )
    before = _label(tensor.tensor_topology())
    assert before[0] == ["Replicate", "Shard(2)"], "precondition: a 2-D label"
    reference = tensor.cpu()

    owned = [ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(1, 1)]  # one replica of each column's shard
    host = tensor.cpu(coords=owned)

    assert _populated(host, mesh_device) == owned
    for coord in owned:
        assert _shard_bytes(host, coord) == _shard_bytes(reference, coord)
    assert _shard_bytes(host, owned[0]) != _shard_bytes(host, owned[1]), "the two columns hold different shards"
    assert _label(host.tensor_topology()) == before, "the subset carries the device tensor's label"
    assert _label(tensor.tensor_topology()) == before, "the device tensor's label is untouched"
    assert _populated(tensor.cpu(), mesh_device) == _all_coords(mesh_device), "a later whole read is unaffected"


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_cpu_coords_reads_one_replica_of_a_replicated_tensor(mesh_device):
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape != (2, 2):
        pytest.skip(f"needs a [2, 2] mesh, the fixture opened {mesh_shape}")

    expected = torch.randn(1, 1, 32, 32).to(torch.bfloat16)
    tensor = ttnn.from_torch(
        expected, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    before = _label(tensor.tensor_topology())
    one = ttnn.MeshCoordinate(1, 0)

    host = tensor.cpu(coords=[one])

    assert _populated(host, mesh_device) == [one]
    assert _shard_bytes(host, one) == _shard_bytes(tensor.cpu(), one)
    assert _label(host.tensor_topology()) == before
    assert _label(tensor.tensor_topology()) == before


def test_cpu_coords_rejects_what_the_tensor_does_not_hold(mesh_device, expect_error):
    tensor = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    first = tensor.device_coords()[0]

    with expect_error(RuntimeError, "at least one coordinate"):
        tensor.cpu(coords=[])
    with expect_error(RuntimeError, "listed twice"):
        tensor.cpu(coords=[first, first])
    with expect_error(RuntimeError, "holds no shard"):
        tensor.cpu(coords=[ttnn.MeshCoordinate(99, 99)])
    with expect_error(RuntimeError, "needs a device tensor"):
        tensor.cpu().cpu(coords=[first])

    if len(tensor.device_coords()) > 1:
        # A shard taken off the tensor holds one coordinate; asking it for another is refused.
        shard = ttnn.get_device_tensors(tensor)[0]
        with expect_error(RuntimeError, "holds no shard"):
            shard.cpu(coords=[tensor.device_coords()[-1]])
        assert _populated(shard.cpu(coords=[first]), mesh_device) == [first]

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh-topology labels an op without its own compute_output_topologies gives its outputs.

An output the op allocates takes the union of the input labels. An output that is the caller's tensor (passed as
``output_tensor=``, or an input updated in place) keeps that tensor's label unless another input is sharded along a
mesh axis on which the label is replicated: every device then holds a different result, and the output takes the
union too, on the caller's handle as well. The label is not part of the program hash.
"""

import pytest
import torch

import ttnn

pytestmark = pytest.mark.parametrize("mesh_device", [2], indirect=True)


def _from_torch(tensor, mesh_device, mesh_mapper):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mesh_mapper
    )


def _replicated(tensor, mesh_device):
    return _from_torch(tensor, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))


def _sharded(tensor, mesh_device):
    """Collapsed 1-D label ``{N},[Shard(3)]``."""
    return _from_torch(tensor, mesh_device, ttnn.ShardTensorToMesh(mesh_device, dim=3))


def _sharded_nd(tensor, mesh_device):
    """The same split with one placement per mesh axis: ``[Replicate, Shard(3)]`` on the 1xN mesh."""
    config = ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(3)])
    return _from_torch(tensor, mesh_device, ttnn.create_mesh_mapper(mesh_device, config))


def _local():
    return torch.randn([1, 1, 32, 64], dtype=torch.bfloat16)


def _wide(mesh_device):
    return torch.randn([1, 1, 32, 64 * mesh_device.get_num_devices()], dtype=torch.bfloat16)


def _device_slices(tensor):
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def _assert_sum(out, a, b):
    """Per device, out holds that device's a + b."""
    for got, a_dev, b_dev in zip(_device_slices(out), _device_slices(a), _device_slices(b)):
        assert torch.allclose(got.float(), a_dev.float() + b_dev.float(), rtol=0.02, atol=0.02)


def test_fresh_output_takes_the_union(mesh_device):
    a, b = _sharded_nd(_wide(mesh_device), mesh_device), _replicated(_local(), mesh_device)

    out = ttnn.add(a, b)

    assert out.tensor_topology() == a.tensor_topology()
    _assert_sum(out, a, b)


def test_preallocated_output_keeps_its_label(mesh_device):
    """A ``{N},[Shard(3)]`` output written from an input sharded along the same mesh axis keeps its collapsed label,
    where the union is the input's N-D one."""
    a, b = _sharded_nd(_wide(mesh_device), mesh_device), _replicated(_local(), mesh_device)
    out = _sharded(torch.zeros([1, 1, 32, 64 * mesh_device.get_num_devices()], dtype=torch.bfloat16), mesh_device)
    label = out.tensor_topology()
    assert label != a.tensor_topology(), "precondition: the output's label differs from the union"

    returned = ttnn.add(a, b, output_tensor=out)

    assert returned.buffer_address() == out.buffer_address()
    assert returned.tensor_topology() == label
    assert out.tensor_topology() == label
    _assert_sum(out, a, b)


def test_preallocated_replicated_output_follows_the_data(mesh_device):
    """A replicated output written from a sharded input differs per device afterwards: it takes the input's label."""
    a, b = _sharded(_wide(mesh_device), mesh_device), _replicated(_local(), mesh_device)
    out = _replicated(torch.zeros([1, 1, 32, 64], dtype=torch.bfloat16), mesh_device)

    returned = ttnn.add(a, b, output_tensor=out)

    assert returned.tensor_topology() == a.tensor_topology()
    assert out.tensor_topology() == a.tensor_topology()
    _assert_sum(out, a, b)


@pytest.mark.parametrize("sharded_a", [True, False], ids=["sharded_a_keeps_label", "replicated_a_takes_union"])
def test_in_place_label_follows_the_data(mesh_device, sharded_a):
    """a += b with b sharded along the mesh: a sharded along the same axis keeps its collapsed label, a replicated a
    now differs per device and takes b's label."""
    a = _sharded(_wide(mesh_device), mesh_device) if sharded_a else _replicated(_local(), mesh_device)
    b = _sharded_nd(_wide(mesh_device), mesh_device)
    a_label = a.tensor_topology()
    expected = a_label if sharded_a else b.tensor_topology()
    before = [s.clone() for s in _device_slices(a)]

    returned = ttnn.add(a, b, output_tensor=a)

    assert returned.buffer_address() == a.buffer_address()
    assert returned.tensor_topology() == expected
    assert a.tensor_topology() == expected
    for got, a_dev, b_dev in zip(_device_slices(a), before, _device_slices(b)):
        assert torch.allclose(got.float(), a_dev.float() + b_dev.float(), rtol=0.02, atol=0.02)


def test_label_is_not_part_of_the_program_hash(mesh_device):
    """Outputs that differ only in their labels reuse one cached program and still get their own labels."""
    b = _replicated(_local(), mesh_device)
    runs = [
        (_sharded(_wide(mesh_device), mesh_device), _sharded(torch.zeros_like(_wide(mesh_device)), mesh_device)),
        (_sharded(_wide(mesh_device), mesh_device), _replicated(torch.zeros_like(_local()), mesh_device)),
    ]

    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    try:
        entries = []
        for a, out in runs:
            ttnn.add(a, b, output_tensor=out)
            entries.append(mesh_device.num_program_cache_entries())
            assert out.tensor_topology() == a.tensor_topology()
        assert entries[0] > 0
        assert entries[0] == entries[1], f"program cache entries changed across labels: {entries}"
    finally:
        mesh_device.disable_and_clear_program_cache()

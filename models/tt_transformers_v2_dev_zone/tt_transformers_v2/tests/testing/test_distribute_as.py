#!/usr/bin/env python3
"""
Tests for distributing a torch.Tensor using the topology of an existing TTNN tensor.

Follows the same style as test_auto_compose.py, but in the opposite
direction: given a reference distributed tensor, verify that
`from_torch_dist_as(torch_tensor, ref_tensor)` produces a TTNN tensor
with the same topology and data that composes back to the original
torch input.
"""

import pytest
import torch
from tt_transformers_v2.src.testing.auto_compose import extract_tensor_topology_info, to_torch_auto_compose
from tt_transformers_v2.src.testing.distribute_as import from_torch_dist_as

import ttnn

# ======================================================================================
# Test Parameters
# ======================================================================================


pytestmark = [
    pytest.mark.parametrize(
        "ttnn_mesh_device",
        [
            # (1, 1),  # single device
            # (1, 2),  # 1D mesh, 2 devices
            # (1, 8),  # 1D mesh, 8 devices
            (2, 4),  # 2D mesh, 8 devices
        ],
        ids=[
            # "1x1",
            # "1x2",
            # "1x8",
            "2x4",
        ],
        indirect=True,
    ),
    pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["row_major", "tile"]),
]


# ======================================================================================
# Helpers
# ======================================================================================


def _make_known_pattern(num_chunks: int) -> torch.Tensor:
    rows = []
    for i in range(num_chunks):
        rows.append(torch.tensor([[[i * 1.0], [i * 2.0], [i * 3.0]]]).transpose(0, 1))  # [1,3,1]
    data = torch.stack(rows, dim=0)  # [num_chunks,1,3,1]
    return data.to(torch.bfloat16)


def _make_arange_bf16(shape: tuple[int, ...]) -> torch.Tensor:
    numel = 1
    for s in shape:
        numel *= s
    data = torch.arange(numel, dtype=torch.float32).reshape(shape)
    return data.to(torch.bfloat16)


def _pos_dim(dim: int, rank: int) -> int:
    return dim % rank


def _topology_signature(tt: ttnn.Tensor) -> tuple[tuple[str, tuple[int, ...]], tuple[int, ...]]:
    placements, dist_shape = extract_tensor_topology_info(tt)
    sig_p = []
    for p in placements:
        if isinstance(p, ttnn.PlacementShard):
            sig_p.append(("shard", (p.dim,)))
        else:
            assert isinstance(p, ttnn.PlacementReplicate)
            sig_p.append(("replicate", ()))
    return tuple(sig_p), tuple(dist_shape)


# ======================================================================================
# Tests
# ======================================================================================


def test_as_host_sharded_1d(ttnn_mesh_device: ttnn.MeshDevice, layout) -> None:
    num_devices = ttnn_mesh_device.get_num_devices()

    # Reference topology: host-sharded along dim 0
    ref_input = _make_known_pattern(num_devices)
    ref_tt = ttnn.from_torch(
        ref_input,
        device=None,
        dtype=ttnn.bfloat16,
        layout=layout,
        mesh_mapper=ttnn.ShardTensorToMesh(ttnn_mesh_device, dim=0),
    )

    # Distribute a different tensor using the same topology
    new_input = _make_known_pattern(num_devices) + 1  # change values to avoid accidental equality
    tt_as = from_torch_dist_as(new_input, ref_tt, device=ttnn_mesh_device)

    # Verify topology matches reference
    assert _topology_signature(tt_as) == _topology_signature(ref_tt)

    # Verify round-trip via auto-composition matches the source
    torch_auto = to_torch_auto_compose(tt_as, device=ttnn_mesh_device)
    assert torch.equal(torch_auto, new_input)


def test_as_device_sharded_1d(ttnn_mesh_device: ttnn.MeshDevice, layout) -> None:
    num_devices = ttnn_mesh_device.get_num_devices()

    ref_input = _make_known_pattern(num_devices)
    ref_tt = ttnn.from_torch(
        ref_input,
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=layout,
        mesh_mapper=ttnn.ShardTensorToMesh(ttnn_mesh_device, dim=0),
    )

    new_input = _make_known_pattern(num_devices) + 2
    tt_as = from_torch_dist_as(new_input, ref_tt)

    if ttnn_mesh_device.get_num_devices() > 1:
        assert _topology_signature(tt_as) == _topology_signature(ref_tt)

    torch_auto = to_torch_auto_compose(tt_as)
    assert torch.equal(torch_auto, new_input)


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_as_host_sharded_various_dims(ttnn_mesh_device: ttnn.MeshDevice, layout, dim: int) -> None:
    num_devices = ttnn_mesh_device.get_num_devices()

    rank = 4
    axis = _pos_dim(dim, rank)
    shape = [2, 3, 4, 1]
    shape[axis] = num_devices
    ref_input = _make_arange_bf16(tuple(shape))

    ref_tt = ttnn.from_torch(
        ref_input,
        device=None,
        dtype=ttnn.bfloat16,
        layout=layout,
        mesh_mapper=ttnn.ShardTensorToMesh(ttnn_mesh_device, dim=dim),
    )

    new_input = ref_input + 7
    tt_as = from_torch_dist_as(new_input, ref_tt, device=ttnn_mesh_device)

    assert _topology_signature(tt_as) == _topology_signature(ref_tt)
    torch_auto = to_torch_auto_compose(tt_as, device=ttnn_mesh_device)
    assert torch.equal(torch_auto, new_input)


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_as_device_sharded_various_dims(ttnn_mesh_device: ttnn.MeshDevice, layout, dim: int) -> None:
    num_devices = ttnn_mesh_device.get_num_devices()

    rank = 4
    axis = _pos_dim(dim, rank)
    shape = [2, 3, 4, 1]
    shape[axis] = num_devices
    ref_input = _make_arange_bf16(tuple(shape))

    ref_tt = ttnn.from_torch(
        ref_input,
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=layout,
        mesh_mapper=ttnn.ShardTensorToMesh(ttnn_mesh_device, dim=dim),
    )

    new_input = ref_input + 11
    tt_as = from_torch_dist_as(new_input, ref_tt)

    if ttnn_mesh_device.get_num_devices() > 1:
        assert _topology_signature(tt_as) == _topology_signature(ref_tt)
    torch_auto = to_torch_auto_compose(tt_as)
    assert torch.equal(torch_auto, new_input)


@pytest.mark.parametrize("dims_pair", [(0, 1), (0, -1), (1, -1)])
def test_as_2d_shard_shard(ttnn_mesh_device: ttnn.MeshDevice, layout, dims_pair: tuple[int, int]) -> None:
    mesh_shape = tuple(ttnn_mesh_device.shape)
    if len(mesh_shape) != 2 or mesh_shape[0] <= 1 or mesh_shape[1] <= 1:
        pytest.skip("Requires a 2D mesh with both dims > 1")

    rank = 4
    d0 = _pos_dim(dims_pair[0], rank)
    d1 = _pos_dim(dims_pair[1], rank)
    assert d0 != d1

    shape = [2, 3, 4, 5]
    shape[d0] = mesh_shape[0]
    shape[d1] = mesh_shape[1]
    ref_input = _make_arange_bf16(tuple(shape))

    mapper = ttnn.ShardTensor2dMesh(ttnn_mesh_device, mesh_shape=mesh_shape, dims=(dims_pair[0], dims_pair[1]))
    ref_tt = ttnn.from_torch(ref_input, device=None, dtype=ttnn.bfloat16, layout=layout, mesh_mapper=mapper)

    new_input = ref_input + 23
    tt_as = from_torch_dist_as(new_input, ref_tt, device=ttnn_mesh_device)

    assert _topology_signature(tt_as) == _topology_signature(ref_tt)
    torch_auto = to_torch_auto_compose(tt_as, device=ttnn_mesh_device)
    assert torch.equal(torch_auto, new_input)


# todo)) add test cases along the dtype axis!

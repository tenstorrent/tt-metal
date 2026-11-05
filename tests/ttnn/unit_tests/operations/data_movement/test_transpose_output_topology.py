# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh-topology labels of ttnn.transpose, ttnn.permute and ttnn.stack outputs.

A Shard follows its tensor dim to the place the transpose or permute moves it, so the output gathered along the dim its
label names is the transposed or permuted gathered input. A Shard keeps its spelling (a negative dim stays negative),
and a Shard on a dim that does not move keeps its label. A rank-2 or rank-3 input is transposed or permuted as a 4-D
view of it, and ttnn.stack on a new last dim goes through a view with a new leading dim; a Shard counted from the front
is renumbered for the view and back. When the dims a transpose or permute moves hold one element per device, the
wrappers return the input (or a copy of it) without running the operation, and the output keeps the input's label.
"""

import pytest
import torch

import ttnn

pytestmark = pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)

REPLICATE = "PlacementReplicate()"


def _sharded(tensor, mesh_device, dim, layout=ttnn.TILE_LAYOUT, nd=True):
    """`tensor` split on `dim` over the 1x2 mesh, labelled [Replicate, Shard(dim)] (one placement per mesh axis) or,
    with nd=False, with the collapsed 1-D label {2},[Shard(dim)]."""
    if nd:
        config = ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(dim)])
        mapper = ttnn.create_mesh_mapper(mesh_device, config)
    else:
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=dim)
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=layout, device=mesh_device, mesh_mapper=mapper)


def _placements(tensor):
    return [repr(placement) for placement in tensor.tensor_topology().placements()]


def _gathered(tensor):
    """The tensor a 1xN mesh tensor's label describes: its device tensors concatenated along the Shard's dim."""
    pieces = [ttnn.to_torch(device_tensor) for device_tensor in ttnn.get_device_tensors(tensor)]
    dims = [p.dim for p in tensor.tensor_topology().placements() if isinstance(p, ttnn.PlacementShard)]
    if not dims:
        return pieces[0]
    (dim,) = dims
    return torch.cat(pieces, dim=dim)


@pytest.mark.parametrize("nd", [True, False], ids=["nd", "collapsed"])
@pytest.mark.parametrize(
    "shape, split_dim, dims, expected_dim",
    [
        ((1, 1, 64, 128), 2, (-2, -1), 3),  # WH
        ((1, 1, 64, 128), -1, (-2, -1), -2),  # WH, the Shard spelled from the end
        ((1, 4, 64, 64), 1, (1, 2), 2),  # HC
        ((4, 2, 64, 64), 0, (0, 1), 1),  # CN
        ((4, 1, 64, 64), 0, (0, 2), 2),  # NH, through ttnn.permute
        ((1, 2, 64, 128), 1, (-2, -1), 1),  # a dim the transpose does not move
    ],
    ids=["wh", "wh-negative", "hc", "cn", "nh", "untouched"],
)
def test_transpose_moves_the_shard_with_its_dim(mesh_device, nd, shape, split_dim, dims, expected_dim):
    torch_input = torch.randn(shape).bfloat16()

    out = ttnn.transpose(_sharded(torch_input, mesh_device, split_dim, nd=nd), *dims)

    shard = f"PlacementShard({expected_dim})"
    assert _placements(out) == ([REPLICATE, shard] if nd else [shard])
    assert torch.equal(_gathered(out), torch_input.transpose(*dims))


def test_row_major_transpose(mesh_device):
    """An interleaved row-major transpose runs as a permute."""
    torch_input = torch.randn(1, 1, 64, 128).bfloat16()

    out = ttnn.transpose(_sharded(torch_input, mesh_device, 2, layout=ttnn.ROW_MAJOR_LAYOUT), -2, -1)

    assert _placements(out) == [REPLICATE, "PlacementShard(3)"]
    assert torch.equal(_gathered(out), torch_input.transpose(-2, -1))


@pytest.mark.parametrize(
    "shape, split_dim, dims, expected_dim",
    [
        ((64, 128), -1, (-2, -1), -2),  # counted from the end: the same dim of the view
        ((128, 64), 0, (-2, -1), 1),
        ((2, 64, 128), 2, (-2, -1), 1),
        ((2, 64, 128), 2, (0, 1), 2),  # a dim the transpose does not move
        ((2, 128, 64), 1, (0, 2), 1),  # a dim the transpose does not move, through ttnn.permute
        ((2, 64, 128), -1, (0, 1), -1),
    ],
    ids=["rank2-negative", "rank2", "rank3", "rank3-untouched", "rank3-untouched-permute", "rank3-negative-untouched"],
)
def test_lower_rank_transpose(mesh_device, shape, split_dim, dims, expected_dim):
    """A rank-2 or rank-3 input is transposed as a 4-D view of it."""
    torch_input = torch.randn(shape).bfloat16()

    out = ttnn.transpose(_sharded(torch_input, mesh_device, split_dim), *dims)

    assert _placements(out) == [REPLICATE, f"PlacementShard({expected_dim})"]
    assert torch.equal(_gathered(out), torch_input.transpose(*dims))


def test_higher_rank_transpose(mesh_device):
    """A 5-D transpose runs as a permute."""
    torch_input = torch.randn(4, 1, 2, 64, 64).bfloat16()

    out = ttnn.transpose(_sharded(torch_input, mesh_device, 0), 0, 2)

    assert _placements(out) == [REPLICATE, "PlacementShard(2)"]
    assert torch.equal(_gathered(out), torch_input.transpose(0, 2))


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row-major"])
def test_permute_moves_the_shard_with_its_dim(mesh_device, layout):
    """Output dim i is input dim dims[i]: the input's dim 1 is the output's dim 3."""
    torch_input = torch.randn(1, 4, 64, 96).bfloat16()
    dims = (0, 2, 3, 1)

    out = ttnn.permute(_sharded(torch_input, mesh_device, 1, layout=layout), dims)

    assert _placements(out) == [REPLICATE, "PlacementShard(3)"]
    assert torch.equal(_gathered(out), torch_input.permute(dims))


@pytest.mark.parametrize("dims, expected_dim", [((1, 0, 2), 2), ((2, 0, 1), 0)], ids=["untouched", "moved"])
def test_lower_rank_permute(mesh_device, dims, expected_dim):
    """A rank-3 input is permuted as a 4-D view of it."""
    torch_input = torch.randn(2, 64, 128).bfloat16()

    out = ttnn.permute(_sharded(torch_input, mesh_device, 2), dims)

    assert _placements(out) == [REPLICATE, f"PlacementShard({expected_dim})"]
    assert torch.equal(_gathered(out), torch_input.permute(dims))


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row-major"])
def test_stack_on_a_new_last_dim(mesh_device, layout):
    """ttnn.stack on a new last dim stacks on a new leading dim and permutes it to the end: the Shard keeps its dim."""
    a, b = torch.randn(64, 128).bfloat16(), torch.randn(64, 128).bfloat16()

    out = ttnn.stack([_sharded(a, mesh_device, 1, layout=layout), _sharded(b, mesh_device, 1, layout=layout)], dim=-1)

    assert _placements(out) == [REPLICATE, "PlacementShard(1)"]
    assert torch.equal(_gathered(out), torch.stack([a, b], dim=-1))

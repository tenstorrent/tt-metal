# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh-topology labels of ttnn.matmul and ttnn.linear outputs with a transposed operand.

a's M dim and b's N dim become the output's last two dims, but transpose_a / transpose_b store them on the other side
of K in the operand. A Shard on a transposed operand's M or N follows it to the output dim it becomes, so the output
gathered along the dim its label names is the product of the gathered operands. A Shard keeps its spelling (a negative
dim stays negative). Otherwise the output keeps the default label, the union of the operands' labels: without a
transpose flag, when an operand is split on the contracted dim (each device then holds a partial sum), and when a
preallocated output has another shape than the operands give.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)

S, K, N = 64, 64, 128  # rows, contracted dim, columns
REPLICATE = "PlacementReplicate()"


def _put(tensor, mesh_device, placement, nd=True):
    """`tensor` on the 1x2 mesh, labelled [Replicate, placement] (one placement per mesh axis) or, with nd=False, with
    the collapsed 1-D label ShardTensorToMesh / ReplicateTensorToMesh give."""
    if nd:
        mapper = ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), placement]))
    elif isinstance(placement, ttnn.PlacementShard):
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=placement.dim)
    else:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)


def _replicated(tensor, mesh_device, nd=True):
    return _put(tensor, mesh_device, ttnn.PlacementReplicate(), nd)


def _sharded(tensor, mesh_device, dim, nd=True):
    return _put(tensor, mesh_device, ttnn.PlacementShard(dim), nd)


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
def test_transposed_weight_split_on_out(mesh_device, nd):
    """A [out, in] weight, as torch.nn.Linear stores it, split on out: with transpose_b each device computes its slice
    of the output's columns."""
    x, w = torch.randn(1, 1, S, K), torch.randn(1, 1, N, K)

    out = ttnn.linear(_replicated(x, mesh_device, nd), _sharded(w, mesh_device, 2, nd), transpose_b=True)

    assert _placements(out) == ([REPLICATE, "PlacementShard(3)"] if nd else ["PlacementShard(3)"])
    assert_with_pcc(x @ w.mT, _gathered(out), 0.999)


@pytest.mark.parametrize("dim, expected", [(0, "PlacementShard(3)"), (-2, "PlacementShard(-1)")])
def test_transposed_lower_rank_weight_split_on_out(mesh_device, dim, expected):
    """A 2-D [out, in] weight split on out against a 4-D activation: out is the output's last dim, in either spelling."""
    x, w = torch.randn(1, 1, S, K), torch.randn(N, K)

    out = ttnn.linear(_replicated(x, mesh_device), _sharded(w, mesh_device, dim), transpose_b=True)

    assert _placements(out) == [REPLICATE, expected]
    assert_with_pcc(x @ w.mT, _gathered(out), 0.999)


def test_bias_split_on_out(mesh_device):
    """A [1, out] bias split with the transposed weight it is added to."""
    x, w, bias = torch.randn(1, 1, S, K), torch.randn(1, 1, N, K), torch.randn(1, N)

    out = ttnn.linear(
        _replicated(x, mesh_device),
        _sharded(w, mesh_device, 2),
        bias=_sharded(bias, mesh_device, 1),
        transpose_b=True,
    )

    assert _placements(out) == [REPLICATE, "PlacementShard(3)"]
    assert_with_pcc(x @ w.mT + bias, _gathered(out), 0.999)


def test_transposed_activation_split_on_rows(mesh_device):
    """An a stored [K, M] and split on M: with transpose_a each device computes its slice of the output's rows."""
    a, b = torch.randn(1, 1, K, S), torch.randn(1, 1, K, N)

    out = ttnn.matmul(_sharded(a, mesh_device, 3), _replicated(b, mesh_device), transpose_a=True)

    assert _placements(out) == [REPLICATE, "PlacementShard(2)"]
    assert_with_pcc(a.mT @ b, _gathered(out), 0.999)


@pytest.mark.parametrize("dim", [3, -1])
def test_untransposed_weight_label_unchanged(mesh_device, dim):
    """A [in, out] weight split on out: without a transpose flag the default label already names the output's columns."""
    x, w = torch.randn(1, 1, S, K), torch.randn(1, 1, K, N)

    out = ttnn.matmul(_replicated(x, mesh_device), _sharded(w, mesh_device, dim))

    assert _placements(out) == [REPLICATE, f"PlacementShard({dim})"]
    assert_with_pcc(x @ w, _gathered(out), 0.999)


@pytest.mark.parametrize("transpose_b", [False, True])
def test_contracted_dim_split_keeps_default_label(mesh_device, transpose_b):
    """Both operands split on K: each device holds a partial sum. The output keeps the union of the operands' labels,
    a's Shard(3), so it is not labelled Replicate."""
    x, w = torch.randn(1, 1, S, K), torch.randn(1, 1, K, N)
    w_stored = w.mT.contiguous() if transpose_b else w

    out = ttnn.matmul(
        _sharded(x, mesh_device, 3), _sharded(w_stored, mesh_device, 3 if transpose_b else 2), transpose_b=transpose_b
    )

    assert _placements(out) == [REPLICATE, "PlacementShard(3)"]
    partial_sums = [ttnn.to_torch(device_tensor) for device_tensor in ttnn.get_device_tensors(out)]
    assert_with_pcc(x @ w, sum(partial_sums), 0.999)


def test_preallocated_output_of_another_shape_keeps_default_label(mesh_device):
    """A preallocated output may fold leading dims into its rows: a 3-D out for a 4-D product. Its dims are not the
    operands', so it keeps the default label, which here names its columns."""
    x, w = torch.randn(1, 1, S, K), torch.randn(1, 1, N, K)
    out = _replicated(torch.zeros(1, S, N // 2), mesh_device)

    ttnn.linear(_replicated(x, mesh_device), _sharded(w, mesh_device, 2), transpose_b=True, optional_output_tensor=out)

    assert _placements(out) == [REPLICATE, "PlacementShard(2)"]
    assert_with_pcc((x @ w.mT)[0], _gathered(out), 0.999)

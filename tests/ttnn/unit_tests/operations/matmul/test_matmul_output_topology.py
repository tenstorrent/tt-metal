# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh-topology labels of ttnn.matmul and ttnn.linear outputs.

a's M dim and b's N dim become the output's last two dims wherever transpose_a / transpose_b put them in the operand,
and the batch dims of a lower-rank operand align with the output's from the right. An operand's Shard follows its dim
there, so the output gathered along the dim its label names is the product of the gathered operands. A Shard keeps its
spelling (a negative dim stays negative), so a label that already names the right dim comes out unchanged. A Shard on
the contracted dim leaves partial sums on the devices; the output keeps the default label, the union of the operands'.
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


def test_bias_split_on_out(mesh_device):
    """A [1, out] bias split with the weight it is added to."""
    x, w, bias = torch.randn(1, 1, S, K), torch.randn(1, 1, N, K), torch.randn(1, N)

    out = ttnn.linear(
        _replicated(x, mesh_device),
        _sharded(w, mesh_device, 2),
        bias=_sharded(bias, mesh_device, 1),
        transpose_b=True,
    )

    assert _placements(out) == [REPLICATE, "PlacementShard(3)"]
    assert_with_pcc(x @ w.mT + bias, _gathered(out), 0.999)


def test_bias_label_of_higher_distribution_rank(mesh_device):
    """A weight with the collapsed label {2},[Shard(2)] next to a bias labelled [Replicate, Shard(1)]: the union counts
    only labels of the highest distribution rank, so the output's label is the bias's, renumbered."""
    x, w, bias = torch.randn(1, 1, S, K), torch.randn(1, 1, N, K), torch.randn(1, N)

    out = ttnn.linear(
        _replicated(x, mesh_device),
        _sharded(w, mesh_device, 2, nd=False),
        bias=_sharded(bias, mesh_device, 1),
        transpose_b=True,
    )

    assert _placements(out) == [REPLICATE, "PlacementShard(3)"]
    assert_with_pcc(x @ w.mT + bias, _gathered(out), 0.999)


def test_preallocated_output_of_lower_rank(mesh_device):
    """A preallocated output may fold the leading dims into its rows: a 2-D out for a 4-D product. Its dim 1 holds the
    columns, so a weight split on out labels it Shard(1)."""
    x, w = torch.randn(1, 1, S, K), torch.randn(K, N)
    out = _replicated(torch.zeros(S, N // 2), mesh_device)

    ttnn.matmul(_replicated(x, mesh_device), _sharded(w, mesh_device, 1), optional_output_tensor=out)

    assert _placements(out) == [REPLICATE, "PlacementShard(1)"]
    assert_with_pcc((x @ w)[0, 0], _gathered(out), 0.999)


def test_preallocated_output_of_higher_rank(mesh_device):
    """A 3-D activation split on its batch dim, written to a 4-D preallocated output: the batch dim is the output's
    dim 1, counted in the output's own rank."""
    a, b = torch.randn(2, S, K), torch.randn(K, N)
    out = _replicated(torch.zeros(1, 1, S, N), mesh_device)

    ttnn.matmul(_sharded(a, mesh_device, 0), _replicated(b, mesh_device), optional_output_tensor=out)

    assert _placements(out) == [REPLICATE, "PlacementShard(1)"]
    assert_with_pcc((a @ b).unsqueeze(0), _gathered(out), 0.999)


def test_transposed_activation_split_on_rows(mesh_device):
    """An a stored [K, M] and split on M: with transpose_a each device computes its slice of the output's rows."""
    a, b = torch.randn(1, 1, K, S), torch.randn(1, 1, K, N)

    out = ttnn.matmul(_sharded(a, mesh_device, 3), _replicated(b, mesh_device), transpose_a=True)

    assert _placements(out) == [REPLICATE, "PlacementShard(2)"]
    assert_with_pcc(a.mT @ b, _gathered(out), 0.999)


@pytest.mark.parametrize("dim, expected", [(1, "PlacementShard(3)"), (-1, "PlacementShard(-1)")])
def test_lower_rank_weight_split_on_out(mesh_device, dim, expected):
    """A 2-D [in, out] weight split on out: its dim 1 is the 4-D output's dim 3. Spelled -1 the label already names it."""
    x, w = torch.randn(1, 1, S, K), torch.randn(K, N)

    out = ttnn.matmul(_replicated(x, mesh_device), _sharded(w, mesh_device, dim))

    assert _placements(out) == [REPLICATE, expected]
    assert_with_pcc(x @ w, _gathered(out), 0.999)


def test_lower_rank_activation_split_on_batch(mesh_device):
    """A 3-D activation split on its batch dim against a 4-D weight: its dim 0 is the output's dim 1."""
    a, b = torch.randn(2, S, K), torch.randn(1, 1, K, N)

    out = ttnn.matmul(_sharded(a, mesh_device, 0), _replicated(b, mesh_device))

    assert _placements(out) == [REPLICATE, "PlacementShard(1)"]
    assert_with_pcc(a @ b, _gathered(out), 0.999)


@pytest.mark.parametrize("dim", [3, -1])
def test_untransposed_weight_label_unchanged(mesh_device, dim):
    """A [in, out] weight split on out: the default label already names the output's columns."""
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

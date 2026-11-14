# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh-topology labels of ttnn.matmul and ttnn.linear outputs with a transposed operand.

a's M dim and b's N dim become the output's last two dims, but transpose_a / transpose_b store them on the other side
of K in the operand. A Shard on a transposed operand's M or N follows it to the output dim it becomes, and a Shard
counted from the front of an operand of lower rank than the output to the output dim it lines up with, so the output
gathered along the dim its label names is the product of the gathered operands. A Shard keeps its spelling (a negative
dim stays negative). Otherwise the output keeps the default label, the union of the operands' labels: without a
transpose flag, when an operand is split on the contracted dim (each device then holds a partial sum), and when a
preallocated output has another shape than the operands give.

The 1x2 mesh's first axis has one device: a Shard on it leaves every device the whole dim, but the label still names
it, which is how these tests put a label on each of two mesh axes.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)

S, K, N = 64, 64, 128  # rows, contracted dim, columns
REPLICATE = "PlacementReplicate()"


def _put(tensor, mesh_device, placement, nd=True, first_axis=None):
    """`tensor` on the 1x2 mesh, labelled [first_axis or Replicate, placement] (one placement per mesh axis) or, with
    nd=False, with the collapsed 1-D label ShardTensorToMesh / ReplicateTensorToMesh give."""
    if nd:
        first_axis = first_axis or ttnn.PlacementReplicate()
        mapper = ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig([first_axis, placement]))
    elif isinstance(placement, ttnn.PlacementShard):
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=placement.dim)
    else:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)


def _replicated(tensor, mesh_device, nd=True):
    return _put(tensor, mesh_device, ttnn.PlacementReplicate(), nd)


def _sharded(tensor, mesh_device, dim, nd=True, first_axis=None):
    return _put(tensor, mesh_device, ttnn.PlacementShard(dim), nd, first_axis)


def _placements(tensor):
    return [repr(placement) for placement in tensor.tensor_topology().placements()]


def _gathered(tensor):
    """The tensor a 1xN mesh tensor's label describes: its device tensors concatenated along the dim its last mesh
    axis, the one with N devices, is split on."""
    pieces = [ttnn.to_torch(device_tensor) for device_tensor in ttnn.get_device_tensors(tensor)]
    placement = tensor.tensor_topology().placements()[-1]
    if not isinstance(placement, ttnn.PlacementShard):
        return pieces[0]
    return torch.cat(pieces, dim=placement.dim)


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


def test_bias_label_of_higher_mesh_rank_split_on_out(mesh_device):
    """Collapsed 1-D labels on x and w and a per-axis label on the [1, out] bias: the bias's label, of the higher mesh
    rank, is the one the output takes, so its Shard(1) moves to the output's columns too."""
    x, w, bias = torch.randn(1, 1, S, K), torch.randn(1, 1, N, K), torch.randn(1, N)

    out = ttnn.linear(
        _replicated(x, mesh_device, nd=False),
        _sharded(w, mesh_device, 2, nd=False),
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


def test_lower_rank_operand_split_on_out(mesh_device):
    """A 2-D [in, out] b split on out (dim 1) against a 4-D a with transpose_a: out is the output's last dim."""
    a, b = torch.randn(1, 1, K, S), torch.randn(K, N)

    out = ttnn.matmul(_replicated(a, mesh_device), _sharded(b, mesh_device, 1), transpose_a=True)

    assert _placements(out) == [REPLICATE, "PlacementShard(3)"]
    assert_with_pcc(a.mT @ b, _gathered(out), 0.999)


def test_transposed_and_lower_rank_shards_on_two_axes(mesh_device):
    """a split on M on the first mesh axis and a 3-D b split on N on the second: each Shard lands on its own output dim,
    so neither axis is labelled Replicate."""
    a, b = torch.randn(1, 1, K, S), torch.randn(1, K, N)

    out = ttnn.matmul(
        _put(a, mesh_device, ttnn.PlacementReplicate(), first_axis=ttnn.PlacementShard(3)),
        _sharded(b, mesh_device, 2),
        transpose_a=True,
    )

    assert _placements(out) == ["PlacementShard(2)", "PlacementShard(3)"]
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


def test_contracted_dim_split_with_a_moved_shard_keeps_default_label(mesh_device):
    """a split on K on the first mesh axis, a transposed weight split on out on the second. The output keeps the default
    label: renumbering only the weight's Shard would claim dim 3 twice and label the second axis Replicate, although its
    devices hold different columns."""
    x, w = torch.randn(1, 1, S, K), torch.randn(1, 1, N, K)

    out = ttnn.linear(
        _put(x, mesh_device, ttnn.PlacementReplicate(), first_axis=ttnn.PlacementShard(3)),
        _sharded(w, mesh_device, 2),
        transpose_b=True,
    )

    assert _placements(out)[1] != REPLICATE


def test_preallocated_output_label_replaced(mesh_device):
    """A preallocated output of the product's shape whose label still describes earlier contents (a Shard on the first
    mesh axis): the output takes the operands' label, not the buffer's."""
    x, w = torch.randn(1, 1, S, K), torch.randn(1, 1, N, K)
    out = _sharded(torch.zeros(1, 1, S, N), mesh_device, 3, first_axis=ttnn.PlacementShard(0))

    ttnn.linear(_replicated(x, mesh_device), _sharded(w, mesh_device, 2), transpose_b=True, optional_output_tensor=out)

    assert _placements(out) == [REPLICATE, "PlacementShard(3)"]
    assert_with_pcc(x @ w.mT, _gathered(out), 0.999)


def test_preallocated_output_of_another_shape_keeps_default_label(mesh_device):
    """A preallocated output may fold leading dims into its rows: a 3-D out for a 4-D product. Its dims are not the
    operands', so it keeps the default label, which here names its columns."""
    x, w = torch.randn(1, 1, S, K), torch.randn(1, 1, N, K)
    out = _replicated(torch.zeros(1, S, N // 2), mesh_device)

    ttnn.linear(_replicated(x, mesh_device), _sharded(w, mesh_device, 2), transpose_b=True, optional_output_tensor=out)

    assert _placements(out) == [REPLICATE, "PlacementShard(2)"]
    assert_with_pcc((x @ w.mT)[0], _gathered(out), 0.999)

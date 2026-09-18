# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for testing the sequence-parallel linears (``ttml.ops.distributed.sp_column_parallel_linear`` /
``sp_row_parallel_linear``) against the collective + linear sequence they replace, on any mesh with a ``tp``
axis. Shared by test_sequence_parallel.py (1x2, line) and test_sp_linear_ops_1x4.py (1x4, ring and line)."""

from __future__ import annotations

import contextlib

import numpy as np

import ttnn
import ttml
from ttml.parallel import SEQUENCE_DIM
from bf16_ulp import bf16_ulp_error

SPLinearImpl = ttml.ops.distributed.SPLinearImpl


def normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return (rng.standard_normal(shape) * 0.1).astype(np.float32)


def per_rank(tensor) -> np.ndarray:
    """Every tp rank's copy of ``tensor`` stacked along dim 1, which is 1 on every tensor here, so tensors
    compare rank by rank whatever their placement. The layout is imposed by a composer rather than read off
    the tensor: activation and gradient topologies are stale after collectives."""
    mesh = ttml.mesh()
    dims = [0] * len(mesh.shape)
    dims[mesh.axis_index("tp")] = 1
    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttnn.create_mesh_composer(device, ttnn.MeshComposerConfig(dims))
    return tensor.to_numpy(ttnn.DataType.FLOAT32, composer=composer).astype(np.float64)


def mesh_tensor(data: np.ndarray, shard_dim: int | None, requires_grad: bool = False):
    """``data`` on the tp mesh, sharded along ``shard_dim`` across tp (replicated when None)."""
    kwargs = {} if shard_dim is None else {"mapper": ttml.mesh().axis_mapper("tp", shard_dim)}
    tensor = ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, **kwargs)
    tensor.set_requires_grad(requires_grad)
    return tensor


@contextlib.contextmanager
def sp_linear_impl(impl: str | SPLinearImpl):
    """Run the block under ``impl`` ("composed" | "fused"), then restore whatever was selected before."""
    previous = ttml.ops.distributed.get_sp_linear_impl()
    ttml.ops.distributed.set_sp_linear_impl(impl)
    try:
        yield
    finally:
        ttml.ops.distributed.set_sp_linear_impl(previous)


def column_linear_reference(x, weight, bias, cluster_axis):
    """What ColumnParallelLinear(sequence_parallel=True) issued before the ops existed."""
    gathered = ttml.ops.distributed.all_gather(
        x, SEQUENCE_DIM, cluster_axis, ttml.ops.distributed.GradOutputType.SHARDED
    )
    return ttml.ops.linear.linear(gathered, weight, bias)


def row_linear_reference(x, weight, cluster_axis):
    """What RowParallelLinear(sequence_parallel=True) issued before the ops existed (bias excluded)."""
    return ttml.ops.distributed.reduce_scatter(ttml.ops.linear.linear(x, weight, None), SEQUENCE_DIM, cluster_axis)


def column_operands(
    rng: np.random.Generator, batch: int, has_bias: bool, seq_len: int, in_features: int, out_features: int
):
    """Operands and upstream grad of a column-parallel linear as ``(data, shard_dim)`` pairs (None data: no bias)."""
    operands = {
        "x": (normal(rng, (batch, 1, seq_len, in_features)), SEQUENCE_DIM),  # per rank [B,1,S/T,K]
        "weight": (normal(rng, (1, 1, out_features, in_features)), 2),  # per rank [1,1,N/T,K]
        "bias": (normal(rng, (1, 1, 1, out_features)) if has_bias else None, 3),
    }
    grad_out = (normal(rng, (batch, 1, seq_len, out_features)), 3)  # per rank [B,1,S,N/T]
    return operands, grad_out


def row_operands(rng: np.random.Generator, batch: int, seq_len: int, in_features: int, out_features: int):
    """Operands and upstream grad of a row-parallel linear as ``(data, shard_dim)`` pairs."""
    operands = {
        "x": (normal(rng, (batch, 1, seq_len, in_features)), 3),  # per rank [B,1,S,K/T]
        "weight": (normal(rng, (1, 1, out_features, in_features)), 3),  # per rank [1,1,N,K/T]
    }
    grad_out = (normal(rng, (batch, 1, seq_len, out_features)), SEQUENCE_DIM)  # per rank [B,1,S/T,N]
    return operands, grad_out


def forward_backward(op, operands: dict, grad_out: tuple, cluster_axis: int) -> dict[str, np.ndarray]:
    """Forward and backward of ``op`` from ``grad_out``; each operand is ``(data, shard_dim)`` (``None``
    data for an absent bias) and gets a fresh device copy, so two runs never share a tensor.
    Returns the per-rank output and the per-rank gradient of every operand."""
    tensors = {
        name: None if data is None else mesh_tensor(data, shard_dim, requires_grad=True)
        for name, (data, shard_dim) in operands.items()
    }
    out = op(**tensors, cluster_axis=cluster_axis)
    out.set_grad(mesh_tensor(*grad_out).get_value())
    out.backward(False)
    result = {"out": per_rank(out)}
    result.update({name: per_rank(t.get_grad_tensor()) for name, t in tensors.items() if t is not None})
    ttml.autograd.AutoContext.get_instance().reset_graph()
    return result


def _check_reference(got: dict[str, np.ndarray], expected: dict[str, np.ndarray], label: str) -> None:
    assert got.keys() == expected.keys(), f"{label}: {sorted(got)} != {sorted(expected)}"
    for name, reference in expected.items():
        assert np.isfinite(reference).all(), f"{label}: {name} reference is not finite"
        assert reference.std() > 0, f"{label}: {name} reference is constant; agreement would prove nothing"
        assert got[name].shape == reference.shape, f"{label}: {name} shape {got[name].shape} != {reference.shape}"


def assert_bitwise_equal(got: dict[str, np.ndarray], expected: dict[str, np.ndarray], label: str) -> None:
    """For two runs of the very same ttnn ops (the Composed implementation against the sequence it replaces)."""
    _check_reference(got, expected, label)
    for name, reference in expected.items():
        if not np.array_equal(got[name], reference):
            differing = int((got[name] != reference).sum())
            raise AssertionError(
                f"{label}: {name} differs in {differing}/{reference.size} elements, "
                f"max |diff| = {np.abs(got[name] - reference).max():.3g}"
            )


def assert_within_ulp(got: dict[str, np.ndarray], expected: dict[str, np.ndarray], label: str, max_ulp: float) -> float:
    """For two implementations of the same math (Fused against Composed): every tensor within ``max_ulp`` bf16 ULP
    at the peak. Prints and returns the worst peak ULP so the run log records how close the two really are."""
    _check_reference(got, expected, label)
    worst = 0.0
    for name, reference in expected.items():
        peak, p99 = bf16_ulp_error(got[name], reference)
        print(f"ULP {label} {name}: peak={peak:.3f} p99={p99:.3f} (limit {max_ulp})")
        worst = max(worst, peak)
        assert peak <= max_ulp, f"{label}: {name} peak ulp={peak:.3f} > {max_ulp} (p99={p99:.3f})"
    return worst

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel LoRA gradients on a ``[1, 2]`` mesh.

``LoraColumnParallelLinear`` keeps ``lora_A`` replicated and ``lora_B`` sharded on its output rows. The grad that
reaches ``lora_A`` on device ``d`` only carries the output columns of ``d``'s ``lora_B`` shard, so without a sum
across TP each copy is updated with a different partial grad and the copies drift apart after the first step.
``LoraRowParallelLinear`` shards ``lora_A`` and keeps ``lora_B`` replicated; its ``all_reduce`` on ``h`` makes
``lora_B``'s grad identical on every device without a sum. These tests compare every grad against a numpy reference
and check that the column-parallel ``lora_A`` copies stay identical over optimizer steps.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttml
import ttnn
from ttml.modules import ColumnParallelLinear, LoraConfig, RowParallelLinear
from ttml.modules.lora import LoraColumnParallelLinear, LoraRowParallelLinear
from tp_block import assert_identical_across_devices, device_slices, make_adamw, train_step

pytestmark = pytest.mark.requires_device

DIM = 64  # 32 rows / cols per device after the 2-way split
SEQ = 32
RANK = 32
NATIVE = ttml.autograd.PreferredPrecision.NATIVE


def _assert_close(got: np.ndarray, ref: np.ndarray, what: str) -> None:
    np.testing.assert_allclose(got, ref, rtol=5e-2, atol=2e-2 * np.abs(ref).max(), err_msg=what)


def _tensor(array: np.ndarray, shard_dim: int | None = None) -> ttml.autograd.Tensor:
    """``array`` on the mesh, sharded on ``shard_dim`` across tp, or replicated."""
    if shard_dim is None:
        return ttml.autograd.Tensor.from_numpy(array, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)
    mapper = ttml.mesh().axis_mapper("tp", tdim=shard_dim)
    return ttml.autograd.Tensor.from_numpy(array, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)


def _lora_config() -> LoraConfig:
    """``alpha = 2 * rank`` so the scaling is 2: a missing scaling, or one applied twice, fails the reference."""
    return LoraConfig(rank=RANK, alpha=2.0 * RANK)


def _column_layer(gather_output: bool) -> LoraColumnParallelLinear:
    """A column-parallel LoRA layer with a non-zero ``lora_B`` (its zero init would make ``lora_A``'s grad zero)."""
    base = ColumnParallelLinear(DIM, DIM, has_bias=False, gather_output=gather_output)
    layer = LoraColumnParallelLinear(base, _lora_config())
    B = np.random.default_rng(0).uniform(-0.5, 0.5, (1, 1, DIM, RANK)).astype(np.float32)
    layer.lora_B.tensor.set_value(_tensor(B, shard_dim=2).get_value())
    return layer


def _gathered(value: ttnn.Tensor, dim: int) -> np.ndarray:
    """The per-device slices concatenated on ``dim`` on the host, as a 2D array."""
    return np.concatenate(device_slices(value), axis=dim)[0, 0]


def _replicated(value: ttnn.Tensor, what: str) -> np.ndarray:
    return assert_identical_across_devices(value, what)[0, 0]


def _weighted_mean(out: ttml.autograd.Tensor, shard_dim: int | None) -> tuple[ttml.autograd.Tensor, np.ndarray]:
    """``mean(out * G)`` for a fixed random ``G`` laid out like ``out``, and ``G`` as a 2D array of the full output.

    The weighting makes the upstream grad differ per output column, so the reference depends on which device holds
    which rows of ``lora_B`` / ``W`` and on the gather order."""
    G = _tensor(np.random.default_rng(3).standard_normal((1, 1, SEQ, DIM)).astype(np.float32), shard_dim)
    G_full = _replicated(G.get_value(NATIVE), "G") if shard_dim is None else _gathered(G.get_value(NATIVE), 3)
    return ttml.ops.unary.mean(out * G), G_full


@pytest.mark.parametrize("gather_output", [True, False], ids=["gathered", "sharded_output"])
def test_column_parallel_grads_match_reference(tp_mesh, gather_output):
    layer = _column_layer(gather_output)
    x = _tensor(np.random.default_rng(1).standard_normal((1, 1, SEQ, DIM)).astype(np.float32))
    x.set_requires_grad(True)

    loss, G = _weighted_mean(layer(x), shard_dim=None if gather_output else 3)
    loss.backward(False)

    # Reference in float32 from the bf16 values actually on device. A gathered output is replicated, so its mean
    # spans every column; a sharded output's mean only spans the local columns, and backward starts from 1 on each
    # device, so the loss being differentiated is the sum of the per-device means.
    x_ref = _replicated(x.get_value(NATIVE), "input")
    W = _gathered(layer.weight.tensor.get_value(NATIVE), 2)
    A = _replicated(layer.lora_A.tensor.get_value(NATIVE), "lora_A")
    B = _gathered(layer.lora_B.tensor.get_value(NATIVE), 2)
    out_per_mean = DIM if gather_output else DIM // tp_mesh.axis_size("tp")
    dY = G / (SEQ * out_per_mean)
    h = x_ref @ A.T
    dh = layer.scaling * dY @ B
    ref_dA = dh.T @ x_ref
    ref_dB = layer.scaling * dY.T @ h
    ref_dx = dY @ W + dh @ A

    _assert_close(_replicated(layer.lora_A.tensor.get_grad(), "lora_A grad"), ref_dA, "lora_A grad")
    _assert_close(_gathered(layer.lora_B.tensor.get_grad(), 2), ref_dB, "lora_B grad")
    _assert_close(_replicated(x.get_grad(), "input grad"), ref_dx, "input grad")

    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_row_parallel_grads_match_reference(tp_mesh):
    base = RowParallelLinear(DIM, DIM, has_bias=False, input_is_parallel=True)
    layer = LoraRowParallelLinear(base, _lora_config())
    B = np.random.default_rng(0).uniform(-0.5, 0.5, (1, 1, DIM, RANK)).astype(np.float32)
    layer.lora_B.tensor.set_value(_tensor(B).get_value())
    x = _tensor(np.random.default_rng(1).standard_normal((1, 1, SEQ, DIM)).astype(np.float32), shard_dim=3)
    x.set_requires_grad(True)

    # The output is all-reduced, so it is replicated and its mean spans every column.
    loss, G = _weighted_mean(layer(x), shard_dim=None)
    loss.backward(False)

    x_ref = _gathered(x.get_value(NATIVE), 3)
    W = _gathered(layer.weight.tensor.get_value(NATIVE), 3)
    A = _gathered(layer.lora_A.tensor.get_value(NATIVE), 3)
    B = _replicated(layer.lora_B.tensor.get_value(NATIVE), "lora_B")
    dY = G / (SEQ * DIM)
    h = x_ref @ A.T
    dh = layer.scaling * dY @ B
    ref_dA = dh.T @ x_ref
    ref_dB = layer.scaling * dY.T @ h
    ref_dx = dY @ W + dh @ A

    _assert_close(_gathered(layer.lora_A.tensor.get_grad(), 3), ref_dA, "lora_A grad")
    _assert_close(_replicated(layer.lora_B.tensor.get_grad(), "lora_B grad"), ref_dB, "lora_B grad")
    _assert_close(_gathered(x.get_grad(), 3), ref_dx, "input grad")

    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_column_parallel_lora_A_stays_replicated_over_steps(tp_mesh):
    layer = _column_layer(gather_output=False)
    opt = make_adamw(layer.parameters())
    A_init = _replicated(layer.lora_A.tensor.get_value(NATIVE), "lora_A before training")

    for step in range(3):
        train_step(layer, opt, DIM)
        A = _replicated(layer.lora_A.tensor.get_value(NATIVE), f"lora_A after step {step + 1}")

    assert not np.array_equal(A, A_init), "lora_A should have been updated"

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Numeric coverage of the fused Llama projections on one device."""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from bf16_ulp import assert_within_bf16_ulp
from ttml.models.llama.transformer import LlamaMLP

BATCH = 2
SEQ = 64
HEAD_DIM = 32


def to_device(array: np.ndarray, requires_grad: bool = False) -> ttml.autograd.Tensor:
    tensor = ttml.autograd.Tensor.from_numpy(array.astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)
    tensor.set_requires_grad(requires_grad)
    return tensor


def to_host(tensor: ttml.autograd.Tensor) -> np.ndarray:
    return tensor.to_numpy().astype(np.float32)


def grad(tensor: ttml.autograd.Tensor) -> np.ndarray:
    return tensor.get_grad_tensor().to_numpy().astype(np.float32)


def loss_with_upstream(tensors, upstream) -> ttml.autograd.Tensor:
    """A scalar loss whose backward sends ``upstream[i] / numel`` into ``tensors[i]``."""
    terms = [ttml.ops.unary.mean(ttml.ops.binary.mul(t, to_device(u))) for t, u in zip(tensors, upstream)]
    loss = terms[0]
    for term in terms[1:]:
        loss = loss + term
    return loss


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    """(B, 1, S, heads * D) -> (B, heads, S, D)."""
    b, _, s, _ = x.shape
    return np.ascontiguousarray(x.reshape(b, s, num_heads, -1).transpose(0, 2, 1, 3))


@pytest.mark.requires_device
@pytest.mark.parametrize("num_heads, num_groups", [(4, 4), (4, 2), (8, 1)], ids=["mha", "gqa", "mqa"])
def test_heads_creation_matches_grouped_heads_creation(num_heads, num_groups):
    rng = np.random.default_rng(0)
    q = rng.standard_normal((BATCH, 1, SEQ, num_heads * HEAD_DIM), dtype=np.float32)
    kv = rng.standard_normal((BATCH, 1, SEQ, 2 * num_groups * HEAD_DIM), dtype=np.float32)
    k, v = np.split(kv, 2, axis=3)

    qkv_t = to_device(np.concatenate([q, kv], axis=3), requires_grad=True)
    q_t, kv_t = to_device(q, requires_grad=True), to_device(kv, requires_grad=True)
    fused = ttml.ops.multi_head_utils.heads_creation(qkv_t, num_heads, num_groups)
    split = ttml.ops.multi_head_utils.grouped_heads_creation(q_t, kv_t, num_heads, num_groups)

    expected = [split_heads(q, num_heads), split_heads(k, num_groups), split_heads(v, num_groups)]
    for name, f, s, e in zip("qkv", fused, split, expected):
        np.testing.assert_array_equal(to_host(f), to_host(s), err_msg=f"{name} heads differ from grouped")
        # The inputs went through bf16 on the way in; the split itself must not change any value.
        np.testing.assert_array_equal(to_host(f), to_host(to_device(e)), err_msg=f"{name} heads mislabelled")

    upstream = [rng.standard_normal(e.shape, dtype=np.float32) for e in expected]
    loss_with_upstream(fused, upstream).backward(False)
    loss_with_upstream(split, upstream).backward(False)
    np.testing.assert_array_equal(grad(qkv_t), np.concatenate([grad(q_t), grad(kv_t)], axis=3))
    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.requires_device
def test_llama_mlp_matches_composite_swiglu_on_split_halves():
    embedding_size, intermediate_size = 64, 128
    rng = np.random.default_rng(0)
    mlp = LlamaMLP(embedding_size, intermediate_size)
    w_gate_up = to_host(mlp.w_gate_up.weight.tensor)
    w_gate, w_up = w_gate_up[:, :, :intermediate_size], w_gate_up[:, :, intermediate_size:]
    w_down = to_host(mlp.w2.weight.tensor)
    x = rng.standard_normal((BATCH, 1, SEQ, embedding_size), dtype=np.float32)

    x_fused = to_device(x, requires_grad=True)
    out_fused = mlp(x_fused)

    x_ref = to_device(x, requires_grad=True)
    w_gate_ref, w_up_ref, w_down_ref = (to_device(w, requires_grad=True) for w in (w_gate, w_up, w_down))
    gate_ref = ttml.ops.unary.silu(ttml.ops.linear.linear(x_ref, w_gate_ref, None))
    up_ref = ttml.ops.linear.linear(x_ref, w_up_ref, None)
    out_ref = ttml.ops.linear.linear(ttml.ops.binary.mul(gate_ref, up_ref), w_down_ref, None)

    assert_within_bf16_ulp(to_host(out_fused), to_host(out_ref), "forward", max_ulp=2.0)

    upstream = [rng.standard_normal(to_host(out_fused).shape, dtype=np.float32)]
    loss_with_upstream([out_fused], upstream).backward(False)
    loss_with_upstream([out_ref], upstream).backward(False)
    assert_within_bf16_ulp(grad(x_fused), grad(x_ref), "d x", max_ulp=2.0)
    assert_within_bf16_ulp(
        grad(mlp.w_gate_up.weight.tensor),
        np.concatenate([grad(w_gate_ref), grad(w_up_ref)], axis=2),
        "d w_gate_up",
        max_ulp=2.0,
    )
    assert_within_bf16_ulp(grad(mlp.w2.weight.tensor), grad(w_down_ref), "d w2", max_ulp=2.0)
    ttml.autograd.AutoContext.get_instance().reset_graph()

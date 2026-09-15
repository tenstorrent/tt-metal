# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the composed chunked gated delta rule.

Two things are pinned here:

1. :func:`wy_inverse` -- the repeated-squaring replacement for the reference's
   63-step sequential WY loop -- must equal that loop exactly (up to fp error).
   This is the only place the implementation deliberately departs from the
   reference, so it gets its own test against the loop itself.

2. The full :func:`chunk_gated_delta_rule` must match the in-repo FLA torch
   reference on both the output and the gradients w.r.t. every input.  There is
   no fused backward to fall back on, so a wrong gradient here would silently
   mistrain 48 of Qwen3.8's 64 layers.

Comparisons use PCC rather than elementwise tolerances: Blackhole's most
accurate matmul (HiFi4) still carries ~4e-3 relative error, and the delta rule
chains a triangular inverse with a 16-step recurrent scan, so absolute
elementwise agreement with float64-ish torch is not the right bar.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ttnn
import ttml

from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
    chunk_gated_delta_rule as torch_chunk_gated_delta_rule,
)
from ttml.models.qwen38.delta_rule import chunk_gated_delta_rule, wy_inverse


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if np.allclose(a, b):
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def _assert_pcc(got, ref, name, threshold=0.995):
    got = np.asarray(got)
    ref = np.asarray(ref)
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    pcc = _pcc(got, ref)
    print(f"  PCC {name}: {pcc:.6f}")
    assert pcc >= threshold, f"{name}: PCC {pcc:.6f} < {threshold}"
    return pcc


def _leaf(x_np):
    t = ttml.autograd.Tensor.from_numpy(
        np.ascontiguousarray(x_np, dtype=np.float32),
        layout=ttnn.Layout.TILE,
        new_type=ttnn.DataType.BFLOAT16,
    )
    t.set_requires_grad(True)
    return t


def _reference_wy_loop(a_np):
    """The reference's sequential WY transform, verbatim, plus the trailing +I."""
    attn = torch.tensor(a_np, dtype=torch.float64)
    chunk = attn.shape[-1]
    for i in range(1, chunk):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(
            -2
        )
    return (attn + torch.eye(chunk, dtype=torch.float64)).numpy()


@pytest.mark.requires_device
@pytest.mark.parametrize("chunk", [32, 64])
def test_wy_inverse_matches_sequential_loop(chunk):
    """Repeated squaring must reproduce the reference's in-place loop."""
    batch = (2, 3)
    rng = np.random.RandomState(0)
    # Strictly lower triangular, the only case the identity is valid for.
    a_np = rng.randn(*batch, chunk, chunk).astype(np.float32) * 0.3
    a_np = a_np * np.tril(np.ones((chunk, chunk), np.float32), k=-1)

    got = wy_inverse(_leaf(a_np), chunk, batch).to_numpy()
    ref = _reference_wy_loop(a_np)

    # Also confirm the identity being relied on: loop(+I) == (I - A)^-1.
    inv = np.linalg.inv(np.eye(chunk) - a_np.astype(np.float64))
    _assert_pcc(ref, inv, f"reference loop == (I-A)^-1 (chunk={chunk})", 0.9999)

    _assert_pcc(got, ref, f"wy_inverse (chunk={chunk})", 0.999)


def _to_bh(x_np):
    """[B, T, H, D] -> [B*H, 1, T, D] (head axis folded into batch)."""
    b, t, h, d = x_np.shape
    return np.ascontiguousarray(x_np.transpose(0, 2, 1, 3).reshape(b * h, 1, t, d))


def _from_bh(x_np, b, h):
    """[B*H, 1, T, V] -> [B, T, H, V]."""
    bh, _, t, v = x_np.shape
    return x_np.reshape(b, h, t, v).transpose(0, 2, 1, 3)


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "batch,heads,seq,chunk,key_dim,val_dim",
    [
        (1, 2, 128, 64, 128, 128),  # Qwen3.8 head dims, 2 chunks
        (1, 4, 256, 64, 128, 128),  # 4 chunks
        (2, 2, 128, 32, 64, 64),  # smaller dims, batch > 1, chunk 32
    ],
)
def test_chunk_gated_delta_rule_vs_torch(batch, heads, seq, chunk, key_dim, val_dim):
    rng = np.random.RandomState(1234)
    q_np = rng.randn(batch, seq, heads, key_dim).astype(np.float32)
    k_np = rng.randn(batch, seq, heads, key_dim).astype(np.float32)
    v_np = rng.randn(batch, seq, heads, val_dim).astype(np.float32)
    # beta in (0, 1) like a sigmoid output; g negative like -exp(A_log)*softplus(.).
    beta_np = rng.uniform(0.1, 0.9, size=(batch, seq, heads)).astype(np.float32)
    g_np = (-rng.uniform(0.01, 0.4, size=(batch, seq, heads))).astype(np.float32)

    # ---- ttml ----
    q = _leaf(_to_bh(q_np))
    k = _leaf(_to_bh(k_np))
    v = _leaf(_to_bh(v_np))
    beta = _leaf(_to_bh(beta_np[..., None]))
    g = _leaf(_to_bh(g_np[..., None]))

    out = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk)
    out_np = _from_bh(out.to_numpy(), batch, heads)

    # A random projection as the loss, so gradients are not all-ones upstream.
    w_np = np.random.RandomState(99).randn(*out.to_numpy().shape).astype(np.float32)
    w = ttml.autograd.Tensor.from_numpy(w_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
    ttml.ops.unary.mean(ttml.ops.binary.mul(out, w)).backward(False)

    # ---- torch reference ----
    qt = torch.tensor(q_np, requires_grad=True)
    kt = torch.tensor(k_np, requires_grad=True)
    vt = torch.tensor(v_np, requires_grad=True)
    bt = torch.tensor(beta_np, requires_grad=True)
    gt = torch.tensor(g_np, requires_grad=True)

    ref, _ = torch_chunk_gated_delta_rule(q=qt, k=kt, v=vt, g=gt, beta=bt, chunk_size=chunk, use_qk_l2norm=True)
    # Same loss: w is laid out in [BH, 1, T, V], so map it onto [B, T, H, V].
    w_ref = torch.tensor(_from_bh(w_np, batch, heads))
    (ref * w_ref).mean().backward()

    _assert_pcc(out_np, ref.detach().numpy(), "delta rule forward", 0.995)

    grads = [
        ("q", _from_bh(q.get_grad_tensor().to_numpy(), batch, heads), qt.grad.numpy()),
        ("k", _from_bh(k.get_grad_tensor().to_numpy(), batch, heads), kt.grad.numpy()),
        ("v", _from_bh(v.get_grad_tensor().to_numpy(), batch, heads), vt.grad.numpy()),
        ("beta", _from_bh(beta.get_grad_tensor().to_numpy(), batch, heads)[..., 0], bt.grad.numpy()),
        ("g", _from_bh(g.get_grad_tensor().to_numpy(), batch, heads)[..., 0], gt.grad.numpy()),
    ]
    for name, got, ref_grad in grads:
        _assert_pcc(got, ref_grad, f"delta rule grad {name}", 0.99)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Correctness of the fused-op delta rule and its wiring into the DeltaNet mixer.

1. :func:`fused_chunk_gated_delta_rule` against the in-repo FLA torch
   reference: output and the gradient w.r.t. every input.
2. The whole :class:`Qwen38GatedDeltaNet` mixer, fused path against the
   composite (ttml-op) path on the same weights.  This pins everything around
   the op that differs between the two -- the token-major layout, the L2 norm
   moved ahead of the GVA repeat, and the per-head output norm.
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
from ttml.models.qwen38 import Qwen38Config, Qwen38GatedDeltaNet
from ttml.models.qwen38.fused_delta_rule import (
    flat_l2_norm,
    fused_chunk_gated_delta_rule,
    fused_chunk_gated_delta_rule_flat,
)
from ttml.models.qwen38.gated_norm import gated_rmsnorm


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


def _assert_pcc(got, ref, name, threshold):
    got = np.asarray(got)
    ref = np.asarray(ref)
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    pcc = _pcc(got, ref)
    print(f"  PCC {name}: {pcc:.6f}")
    assert pcc >= threshold, f"{name}: PCC {pcc:.6f} < {threshold}"


def _leaf(x_np):
    t = ttml.autograd.Tensor.from_numpy(
        np.ascontiguousarray(x_np, dtype=np.float32),
        layout=ttnn.Layout.TILE,
        new_type=ttnn.DataType.BFLOAT16,
    )
    t.set_requires_grad(True)
    return t


def _projection_loss(out, seed=99):
    """``mean(out * w)`` for a fixed random ``w``, so the upstream gradient is not all ones."""
    w_np = np.random.RandomState(seed).randn(*out.to_numpy().shape).astype(np.float32)
    w = ttml.autograd.Tensor.from_numpy(w_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
    ttml.ops.unary.mean(ttml.ops.binary.mul(out, w)).backward(False)
    return w_np


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "batch,heads,seq,chunk,key_dim,val_dim",
    [
        (1, 12, 256, 64, 128, 128),  # Qwen3.8 per-chip heads at TP=4
        (2, 4, 128, 32, 64, 64),
    ],
)
def test_fused_delta_rule_vs_torch(batch, heads, seq, chunk, key_dim, val_dim):
    rng = np.random.RandomState(1234)

    def l2(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    q_np = l2(rng.randn(batch, seq, heads, key_dim)).astype(np.float32)
    k_np = l2(rng.randn(batch, seq, heads, key_dim)).astype(np.float32)
    v_np = rng.randn(batch, seq, heads, val_dim).astype(np.float32)
    beta_np = rng.uniform(0.1, 0.9, size=(batch, seq, heads)).astype(np.float32)
    g_np = (-rng.uniform(0.01, 0.4, size=(batch, seq, heads))).astype(np.float32)

    q, k, v = _leaf(q_np), _leaf(k_np), _leaf(v_np)
    beta = _leaf(beta_np[:, None])
    g = _leaf(g_np[:, None])

    out = fused_chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk)  # [B, 1, T*H, V]
    out_np = out.to_numpy().reshape(batch, seq, heads, val_dim)
    w_np = _projection_loss(out)

    tensors = [torch.tensor(x, requires_grad=True) for x in (q_np, k_np, v_np, beta_np, g_np)]
    qt, kt, vt, bt, gt = tensors
    ref, _ = torch_chunk_gated_delta_rule(q=qt, k=kt, v=vt, g=gt, beta=bt, chunk_size=chunk, use_qk_l2norm=False)
    (ref * torch.tensor(w_np.reshape(batch, seq, heads, val_dim))).mean().backward()

    _assert_pcc(out_np, ref.detach().numpy(), "forward", 0.999)
    _assert_pcc(q.get_grad_tensor().to_numpy(), qt.grad.numpy(), "grad q", 0.99)
    _assert_pcc(k.get_grad_tensor().to_numpy(), kt.grad.numpy(), "grad k", 0.99)
    _assert_pcc(v.get_grad_tensor().to_numpy(), vt.grad.numpy(), "grad v", 0.99)
    _assert_pcc(beta.get_grad_tensor().to_numpy()[:, 0], bt.grad.numpy(), "grad beta", 0.99)
    _assert_pcc(g.get_grad_tensor().to_numpy()[:, 0], gt.grad.numpy(), "grad g", 0.99)


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "batch,k_heads,repeats,seq,chunk,key_dim,val_dim",
    [
        (1, 4, 3, 256, 64, 128, 128),  # Qwen3.8 per-chip heads at TP=4
        (1, 4, 3, 256, 32, 128, 128),  # chunk 32: the forward re-normalizes q/k in-kernel
        (2, 2, 2, 128, 32, 64, 64),
    ],
)
def test_fused_delta_rule_flat_vs_torch(batch, k_heads, repeats, seq, chunk, key_dim, val_dim):
    """Flat layout: raw [B,1,T,H_k*K] q/k in, [B,1,T,H_v*V] out; L2 norm, GVA pairing and
    the GVA gradient sum all happen without materializing a head axis."""
    rng = np.random.RandomState(4321)
    v_heads = k_heads * repeats

    q_np = rng.randn(batch, 1, seq, k_heads * key_dim).astype(np.float32)
    k_np = rng.randn(batch, 1, seq, k_heads * key_dim).astype(np.float32)
    v_np = rng.randn(batch, 1, seq, v_heads * val_dim).astype(np.float32)
    beta_np = rng.uniform(0.1, 0.9, size=(batch, 1, seq, v_heads)).astype(np.float32)
    g_np = (-rng.uniform(0.01, 0.4, size=(batch, 1, seq, v_heads))).astype(np.float32)

    q, k, v, beta, g = (_leaf(x) for x in (q_np, k_np, v_np, beta_np, g_np))
    out = fused_chunk_gated_delta_rule_flat(
        flat_l2_norm(q, key_dim),
        flat_l2_norm(k, key_dim),
        v,
        g,
        beta,
        num_k_heads=k_heads,
        num_v_heads=v_heads,
        key_dim=key_dim,
        chunk_size=chunk,
    )  # [B, 1, T, H_v * V]
    out_np = out.to_numpy()
    w_np = _projection_loss(out)

    # Reference: per-head L2 norm -> GVA repeat -> delta rule, all in torch autograd.
    tensors = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (q_np, k_np, v_np, beta_np, g_np)]
    qt, kt, vt, bt, gt = tensors

    def heads(x, n, d):
        return x.reshape(batch, seq, n, d)

    qn = torch.nn.functional.normalize(heads(qt, k_heads, key_dim), dim=-1, eps=1e-6)
    kn = torch.nn.functional.normalize(heads(kt, k_heads, key_dim), dim=-1, eps=1e-6)
    qn = qn.repeat_interleave(repeats, dim=2)
    kn = kn.repeat_interleave(repeats, dim=2)
    ref, _ = torch_chunk_gated_delta_rule(
        q=qn,
        k=kn,
        v=heads(vt, v_heads, val_dim),
        g=gt.reshape(batch, seq, v_heads),
        beta=bt.reshape(batch, seq, v_heads),
        chunk_size=chunk,
        use_qk_l2norm=False,
    )
    ref = ref.reshape(batch, 1, seq, v_heads * val_dim)
    (ref * torch.tensor(w_np, dtype=torch.float64)).mean().backward()

    _assert_pcc(out_np, ref.detach().numpy(), "forward", 0.999)
    _assert_pcc(q.get_grad_tensor().to_numpy(), qt.grad.numpy(), "grad q", 0.99)
    _assert_pcc(k.get_grad_tensor().to_numpy(), kt.grad.numpy(), "grad k", 0.99)
    _assert_pcc(v.get_grad_tensor().to_numpy(), vt.grad.numpy(), "grad v", 0.99)
    _assert_pcc(beta.get_grad_tensor().to_numpy(), bt.grad.numpy(), "grad beta", 0.99)
    _assert_pcc(g.get_grad_tensor().to_numpy(), gt.grad.numpy(), "grad g", 0.99)


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "batch,seq,heads,head_dim,gamma_requires_grad",
    [
        (1, 256, 12, 128, True),  # Qwen3.8 per-chip value heads at TP=4
        (2, 96, 4, 64, True),
        (1, 64, 3, 128, False),  # frozen gamma: backward skips dgamma
    ],
)
def test_gated_rmsnorm_vs_torch(batch, seq, heads, head_dim, gamma_requires_grad):
    """Fused per-head ``rmsnorm(x) * gamma * silu(gate)`` on flat ``[B, 1, T, H*V]`` vs torch."""
    eps = 1e-6
    rng = np.random.RandomState(7)
    x_np = (rng.randn(batch, 1, seq, heads * head_dim) * 2.0).astype(np.float32)
    gate_np = rng.randn(batch, 1, seq, heads * head_dim).astype(np.float32)
    gamma_np = (1.0 + 0.3 * rng.randn(1, 1, 1, head_dim)).astype(np.float32)

    def bf16_f64(a):
        return torch.tensor(a).to(torch.bfloat16).to(torch.float64).requires_grad_(True)

    xt, gt, gam = (bf16_f64(a) for a in (x_np, gate_np, gamma_np))
    xh = xt.reshape(batch, 1, seq, heads, head_dim)
    normed = (xh * torch.rsqrt((xh * xh).mean(-1, keepdim=True) + eps)).reshape(batch, 1, seq, heads * head_dim)
    out_ref = normed * gam.repeat(1, 1, 1, heads) * torch.nn.functional.silu(gt)

    x, gate, gamma = _leaf(x_np), _leaf(gate_np), _leaf(gamma_np)
    gamma.set_requires_grad(gamma_requires_grad)
    out = gated_rmsnorm(x, gate, gamma, eps)
    w_np = _projection_loss(out)
    (out_ref * torch.tensor(w_np, dtype=torch.float64)).mean().backward()

    _assert_pcc(out.to_numpy(), out_ref.detach().numpy(), "gated rmsnorm out", 0.999)
    _assert_pcc(x.get_grad_tensor().to_numpy(), xt.grad.numpy(), "grad x", 0.999)
    _assert_pcc(gate.get_grad_tensor().to_numpy(), gt.grad.numpy(), "grad gate", 0.999)
    if gamma_requires_grad:
        _assert_pcc(gamma.get_grad_tensor().to_numpy(), gam.grad.numpy(), "grad gamma", 0.999)
    else:
        assert not gamma.is_grad_initialized()


# One chip, so the value-head count has to fit the fused ops' 32-head cap.
SMALL = dict(
    hidden_size=512,
    linear_num_key_heads=4,
    linear_num_value_heads=12,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    delta_chunk_size=64,
)


def _run_mixer(mixer, x_np, impl):
    mixer.config.delta_rule_impl = impl
    x = _leaf(x_np)
    out = mixer(x)
    out_np = out.to_numpy()
    _projection_loss(out)
    grads = {"input": x.get_grad_tensor().to_numpy()}
    for name, p in mixer.parameters().items():
        if p.is_grad_initialized():
            grads[name] = p.get_grad_tensor().to_numpy()
    ttml.autograd.AutoContext.get_instance().reset_graph()
    return out_np, grads


@pytest.mark.requires_device
def test_gated_deltanet_fused_matches_composite():
    config = Qwen38Config(**SMALL)
    mixer = Qwen38GatedDeltaNet(config, layer_idx=0)
    # The zero-initialized decay parameters would give every head the same
    # gate; spread them so the per-head plumbing is actually exercised.
    rng = np.random.RandomState(7)
    for param, values in (
        (mixer.A_log, rng.uniform(-1.0, 1.0, (1, 1, 1, config.linear_num_value_heads))),
        (mixer.dt_bias, rng.uniform(-1.0, 1.0, (1, 1, 1, config.linear_num_value_heads))),
    ):
        param.tensor.set_value(
            ttml.autograd.Tensor.from_numpy(
                values.astype(np.float32), layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
            ).get_value()
        )

    x_np = np.random.RandomState(0).randn(1, 1, 256, config.hidden_size).astype(np.float32)

    out_c, grads_c = _run_mixer(mixer, x_np, "composite")
    optimizer = ttml.optimizers.SGD(mixer.parameters(), ttml.optimizers.SGDConfig.make(0.0, 0.0, 0.0, 0.0, False))
    optimizer.zero_grad()
    out_f, grads_f = _run_mixer(mixer, x_np, "fused")

    _assert_pcc(out_f, out_c, "mixer forward", 0.995)
    assert grads_f.keys() == grads_c.keys()
    for name in grads_c:
        _assert_pcc(grads_f[name], grads_c[name], f"mixer grad {name}", 0.98)

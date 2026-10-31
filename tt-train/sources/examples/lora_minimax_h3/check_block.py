# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Backward check of a tiny H3 block and its parts on one chip, against fp64 numpy finite differences.

    python check_block.py

The numpy forward reimplements each module with the device's own (bf16) weights; its full gradient comes from
per-element central differences in fp64, so the reference carries no bf16 noise. The device gradient is
compared elementwise: |tt|/|ref| and pcc."""

from __future__ import annotations

import sys

import numpy as np
import ttnn
import ttml

from ttml.models.minimax_h3.packing import attention_mask, rope_tables, rope_trans_mat, selection_matrix
from ttml.models.minimax_h3.transformer import H3Attention, H3Block, H3Config, H3FeedForward, RMSNorm, upload

S, D, H, HD, F, TE = 32, 128, 2, 64, 128, 64
VALID = 24
EPS = 1e-3


def bf16(a):
    import ml_dtypes

    return a.astype(ml_dtypes.bfloat16).astype(np.float32)


def tensor(a, requires_grad=False):
    t = ttml.autograd.Tensor.from_numpy(
        np.ascontiguousarray(a, dtype=np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )
    t.set_requires_grad(requires_grad)
    return t


def host(t):
    return np.asarray(t.to_numpy()).astype(np.float64)


def randomize(module, rng):
    for name, p in module.parameters().items():
        shape = list(p.shape())
        if shape[-2] == 1 and "adaln" not in name:  # norm weights
            value = 1.0 + 0.1 * rng.standard_normal(shape, dtype=np.float32)
        else:
            value = rng.standard_normal(shape, dtype=np.float32) / np.sqrt(shape[-1])
        p.assign(tensor(value))
        p.set_requires_grad(False)


def w(module, name):
    return host(module.parameters()[name])[0, 0]


# --------------------------------------------------------------------------- numpy forwards (fp64)


def np_rmsnorm(x, weight, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * weight


def np_rope(x, cos, sin):  # x (H, S, hd) adjacent-pair layout
    rot = np.empty_like(x)
    rot[..., 0::2] = -x[..., 1::2]
    rot[..., 1::2] = x[..., 0::2]
    return x * cos + rot * sin


class NpAttention:
    def __init__(self, attn, config, cos=None, sin=None):
        root = next(iter(attn.parameters())).split("/")[0]
        g = lambda n: w(attn, f"{root}/{n}")
        self.wq, self.wk, self.wv, self.wo = g("to_q/weight"), g("to_k/weight"), g("to_v/weight"), g("to_out/weight")
        self.nq, self.nk = g("norm_q/weight").reshape(-1), g("norm_k/weight").reshape(-1)
        self.eps, self.cos, self.sin = config.qk_norm_eps, cos, sin

    def __call__(self, x, mask):  # x (S, D), mask (S, S) 1 = attend
        heads = lambda a: a.reshape(S, H, HD).transpose(1, 0, 2)
        q, k, v = heads(x @ self.wq.T), heads(x @ self.wk.T), heads(x @ self.wv.T)
        q, k = np_rmsnorm(q, self.nq, self.eps), np_rmsnorm(k, self.nk, self.eps)
        if self.cos is not None:
            q, k = np_rope(q, self.cos, self.sin), np_rope(k, self.cos, self.sin)
        sc = q @ k.transpose(0, 2, 1) / np.sqrt(HD)
        sc = np.where(mask[None] > 0, sc, -1e9)
        sc = sc - sc.max(-1, keepdims=True)
        p = np.exp(sc)
        p /= p.sum(-1, keepdims=True)
        o = (p @ v).transpose(1, 0, 2).reshape(S, H * HD)
        return o @ self.wo.T


class NpFeedForward:
    def __init__(self, ff):
        root = next(iter(ff.parameters())).split("/")[0]
        self.wp, self.wo = w(ff, f"{root}/proj/weight"), w(ff, f"{root}/out/weight")

    def __call__(self, x):
        p = x @ self.wp.T
        gate, value = p[:, :F], p[:, F:]
        return (value * (gate / (1.0 + np.exp(-gate)))) @ self.wo.T


def fd_gradient(f, x, g_weight):
    """d/dx mean(f(x) * g_weight) by central differences, fp64."""
    base = x.astype(np.float64)
    grad = np.zeros_like(base)
    flat = base.reshape(-1)
    for i in range(flat.size):
        xp, xm = flat.copy(), flat.copy()
        xp[i] += EPS
        xm[i] -= EPS
        lp = (f(xp.reshape(base.shape)) * g_weight).mean()
        lm = (f(xm.reshape(base.shape)) * g_weight).mean()
        grad.reshape(-1)[i] = (lp - lm) / (2 * EPS)
    return grad


def check(name, dev_fn, np_fn, x_np, rng, ctx):
    out_shape = dev_fn(tensor(x_np)).shape()
    g_np = bf16(rng.standard_normal(out_shape, dtype=np.float32))
    weight = tensor(g_np * g_np.size, False)
    x = tensor(x_np, True)
    ttml.ops.unary.mean(dev_fn(x) * weight).backward(False)
    got = host(ttml.autograd.create_tensor(x.get_grad(), False))[0, 0]
    ctx.reset_graph()
    ref = fd_gradient(np_fn, x_np[0, 0], g_np[0, 0].astype(np.float64) * g_np.size)
    ratio = np.linalg.norm(got) / np.linalg.norm(ref)
    pcc = np.corrcoef(got.ravel(), ref.ravel())[0, 1]
    flag = "" if abs(ratio - 1) < 0.1 and pcc > 0.98 else "   <-- MISMATCH"
    print(f"{name:28s} |tt|/|ref| {ratio:7.4f}   pcc {pcc:.5f}{flag}", flush=True)


def main() -> int:
    ctx = ttml.autograd.AutoContext.get_instance()
    try:
        ttml.open_device_mesh(ttml.Mesh((1, 1), ("dp", "tp")))
        close = ttml.close_device_mesh
    except Exception as e:  # noqa: BLE001
        print(f"mesh (1,1) unavailable ({e}); opening a plain device")
        ctx.open_device()
        close = ctx.close_device
    rng = np.random.default_rng(0)
    config = H3Config(
        hidden_size=D,
        num_attention_heads=H,
        attention_head_dim=HD,
        ffn_dim=F,
        num_layers=1,
        num_refiner_layers=0,
        time_embed_dim=TE,
        text_dim=64,
        rope_freq_dim=HD // 8,
        use_tp=False,
    )

    pos = rng.uniform(0, 40, size=(VALID, 3)).astype(np.float64)
    cos, sin = rope_tables(pos, S, HD, config.rope_freq_dim, config.rope_theta)
    cos, sin = bf16(cos), bf16(sin)
    params = ttml.ops.rope.RotaryEmbeddingParams()
    params.cos_cache = upload(cos[None, None]).get_value()
    params.sin_cache = upload(sin[None, None]).get_value()
    params.neg_cos_cache = upload(cos[None, None]).get_value()
    params.neg_sin_cache = upload(-sin[None, None]).get_value()
    params.trans_mat = upload(rope_trans_mat()).get_value()
    params.sequence_length, params.head_dim, params.theta = S, HD, config.rope_theta
    mask_np = attention_mask(VALID, S)
    mask, mask_all = upload(mask_np), upload(np.ones((1, 1, S, S), dtype=np.float32))
    x_np = bf16(rng.standard_normal((1, 1, S, D), dtype=np.float32))

    attn = H3Attention(config)
    randomize(attn, rng)
    ref = NpAttention(attn, config)
    ref_rope = NpAttention(attn, config, cos.astype(np.float64), sin.astype(np.float64))
    check("attention, full mask", lambda x: attn(x, mask_all), lambda x: ref(x, np.ones((S, S))), x_np, rng, ctx)
    check("attention, partial mask", lambda x: attn(x, mask), lambda x: ref(x, mask_np[0, 0]), x_np, rng, ctx)
    check(
        "attention, rope, partial",
        lambda x: attn(x, mask, params),
        lambda x: ref_rope(x, mask_np[0, 0]),
        x_np,
        rng,
        ctx,
    )

    ff = H3FeedForward(config)
    randomize(ff, rng)
    check("feed-forward", ff, NpFeedForward(ff), x_np, rng, ctx)

    block = H3Block(config)
    randomize(block, rng)
    root = next(iter(block.parameters())).split("/")[0]
    silu_temb = upload(np.pad(0.5 * rng.standard_normal((2, TE), dtype=np.float32), ((0, 30), (0, 0)))[None, None])
    sel = upload(selection_matrix(np.full(VALID, 1), S, 96, fill=1)[None, None]).get_value()
    mods = block.modulation(block.adaln_table(silu_temb), sel)
    m = [host(t)[0, 0] for t in mods]
    n1, n2 = w(block, f"{root}/norm1/weight").reshape(-1), w(block, f"{root}/norm2/weight").reshape(-1)
    b_attn = NpAttention(block.attn, config, cos.astype(np.float64), sin.astype(np.float64))
    b_ff = NpFeedForward(block.ff)

    def np_block(x):
        n = np_rmsnorm(x, n1, config.norm_eps) * m[1] + m[0]
        x = x + b_attn(n, mask_np[0, 0]) * m[2]
        n = np_rmsnorm(x, n2, config.norm_eps) * m[4] + m[3]
        return x + b_ff(n) * m[5]

    check("block", lambda x: block(x, mods, mask, params), np_block, x_np, rng, ctx)
    close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

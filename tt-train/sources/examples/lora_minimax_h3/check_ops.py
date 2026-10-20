# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Backward audit of the ttml ops on the loss path of train.py, on one chip, against analytic numpy gradients.

    python check_ops.py

Each case prints the ratio |grad_tt| / |grad_ref| and the pcc; a wrong scale shows as a ratio far from 1."""

from __future__ import annotations

import sys

import numpy as np
import ttnn
import ttml

from ttml.models.minimax_h3.transformer import RMSNorm, Split, const

S, D, C = 64, 128, 96  # rows, hidden, out channels
EPS = 1e-5


def tensor(a: np.ndarray, requires_grad: bool) -> ttml.autograd.Tensor:
    t = ttml.autograd.Tensor.from_numpy(
        np.ascontiguousarray(a, dtype=np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )
    t.set_requires_grad(requires_grad)
    return t


def host(t) -> np.ndarray:
    return np.asarray(t.to_numpy()).astype(np.float32)


def grad(t) -> np.ndarray:
    return host(ttml.autograd.create_tensor(t.get_grad(), False))


def bf16(a: np.ndarray) -> np.ndarray:
    import ml_dtypes

    return a.astype(ml_dtypes.bfloat16).astype(np.float32)


def report(name: str, got: np.ndarray, ref: np.ndarray) -> None:
    got, ref = got.reshape(-1).astype(np.float64), ref.reshape(-1).astype(np.float64)
    ratio = np.linalg.norm(got) / (np.linalg.norm(ref) + 1e-30)
    pcc = np.corrcoef(got, ref)[0, 1] if ref.std() > 0 else float("nan")
    flag = "" if 0.9 < ratio < 1.1 else "   <-- SCALE"
    print(f"{name:34s} |tt|/|ref| {ratio:8.4f}   pcc {pcc:.5f}{flag}")


def rmsnorm_backward(x, w, g, eps):
    r = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    gw = g * w
    return r * gw - x * r**3 * (x * gw).mean(-1, keepdims=True)


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
    x_np = bf16(rng.standard_normal((1, 1, S, D), dtype=np.float32))
    t_np = bf16(rng.standard_normal((1, 1, S, C), dtype=np.float32))
    w_np = bf16(rng.standard_normal((1, 1, 1, D), dtype=np.float32))
    mask_np = np.zeros((1, 1, S, C), dtype=np.float32)
    mask_np[0, 0, : S // 2] = 1.0

    # 1. mean backward
    x = tensor(x_np, True)
    ttml.ops.unary.mean(x).backward(False)
    report("mean", grad(x), np.full(x_np.shape, 1.0 / x_np.size))
    ctx.reset_graph()

    # 2. loss * float
    x = tensor(x_np, True)
    (ttml.ops.unary.mean(x) * 3.5).backward(False)
    report("mean * float", grad(x), np.full(x_np.shape, 3.5 / x_np.size))
    ctx.reset_graph()

    # 3. mse_loss MEAN
    p = tensor(t_np + 0.1 * x_np[..., :C], True)
    ttml.ops.loss.mse_loss(p, tensor(t_np, False), ttml.ops.ReduceType.MEAN).backward(False)
    report("mse_loss", grad(p), 2.0 * (host(p) - t_np) / t_np.size)
    ctx.reset_graph()

    # 4. masked mse as in masked_mse(): mse(pred * mask, target) * scale
    p = tensor(t_np + 0.1 * x_np[..., :C], True)
    m = tensor(mask_np, False)
    scale = float(S / (S // 2))
    (ttml.ops.loss.mse_loss(p * m, tensor(t_np * mask_np, False), ttml.ops.ReduceType.MEAN) * scale).backward(False)
    report("masked mse * S/V", grad(p), 2.0 * (host(p) - t_np) * mask_np / t_np.size * scale)
    ctx.reset_graph()

    # 5. x * const + const
    x = tensor(x_np, True)
    c = tensor(bf16(rng.standard_normal(x_np.shape, dtype=np.float32)), False)
    s = tensor(bf16(rng.standard_normal(x_np.shape, dtype=np.float32)), False)
    ttml.ops.unary.mean(x * c + s).backward(False)
    report("x * const + const", grad(x), host(c) / x_np.size)
    ctx.reset_graph()

    # 6. rmsnorm backward wrt x (weight is a parameter, frozen)
    norm = RMSNorm(D, EPS)
    norm.weight.tensor.set_value(tensor(w_np, False).get_value())
    norm.weight.tensor.set_requires_grad(False)
    x = tensor(x_np, True)
    g_np = bf16(rng.standard_normal(x_np.shape, dtype=np.float32))
    ttml.ops.unary.mean(norm(x) * tensor(g_np * x_np.size, False)).backward(False)
    report("rmsnorm (dx)", grad(x), rmsnorm_backward(x_np, w_np, g_np, EPS))
    ctx.reset_graph()

    # 7. linear backward wrt input and weight
    wl_np = bf16(rng.standard_normal((1, 1, C, D), dtype=np.float32) * 0.05)
    x = tensor(x_np, True)
    wl = tensor(wl_np, True)
    ttml.ops.unary.mean(ttml.ops.linear.linear(x, wl, None) * tensor(g_np[..., :C] * S * C, False)).backward(False)
    report("linear (dx)", grad(x), g_np[..., :C][0, 0] @ wl_np[0, 0])
    report("linear (dW)", grad(wl), g_np[..., :C][0, 0].T @ x_np[0, 0])
    ctx.reset_graph()

    # 8. Split backward
    x = tensor(x_np, True)
    a, b = Split.apply(x, (D // 2, D // 2), 3)
    ttml.ops.unary.mean(a * 2.0 + b * 3.0).backward(False)
    ref = np.concatenate([np.full((1, 1, S, D // 2), 2.0), np.full((1, 1, S, D // 2), 3.0)], -1) / (S * D // 2)
    report("Split", grad(x), ref)
    ctx.reset_graph()

    # 9. fused SDPA with an all-ones mask, 2 heads, head_dim 128: dq, dk, dv vs numpy
    H, hd = 2, 128
    q_np, k_np, v_np = (bf16(rng.standard_normal((1, H, S, hd), dtype=np.float32)) for _ in range(3))
    q, k, v = tensor(q_np, True), tensor(k_np, True), tensor(v_np, True)
    mask = tensor(np.ones((1, 1, S, S), dtype=np.float32), False)
    out = ttml.ops.attention.scaled_dot_product_attention(q, k, v, mask)
    go_np = bf16(rng.standard_normal((1, H, S, hd), dtype=np.float32))
    ttml.ops.unary.mean(out * tensor(go_np * go_np.size, False)).backward(False)
    sc = q_np @ k_np.transpose(0, 1, 3, 2) / np.sqrt(hd)
    sc = sc - sc.max(-1, keepdims=True)
    pr = np.exp(sc)
    pr /= pr.sum(-1, keepdims=True)
    o_ref = pr @ v_np
    report("sdpa forward", host(out), o_ref)
    dv = pr.transpose(0, 1, 3, 2) @ go_np
    dp = go_np @ v_np.transpose(0, 1, 3, 2)
    ds = pr * (dp - (dp * pr).sum(-1, keepdims=True))
    dq = ds @ k_np / np.sqrt(hd)
    dk = ds.transpose(0, 1, 3, 2) @ q_np / np.sqrt(hd)
    report("sdpa dq", grad(q), dq)
    report("sdpa dk", grad(k), dk)
    report("sdpa dv", grad(v), dv)
    ctx.reset_graph()

    # 10. composite SDPA, same check
    q, k, v = tensor(q_np, True), tensor(k_np, True), tensor(v_np, True)
    out = ttml.ops.attention.scaled_dot_product_attention_composite(q, k, v, None)
    ttml.ops.unary.mean(out * tensor(go_np * go_np.size, False)).backward(False)
    report("sdpa composite dq", grad(q), dq)
    report("sdpa composite dv", grad(v), dv)
    ctx.reset_graph()

    close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

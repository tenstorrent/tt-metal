# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""On-device checks for every primitive the DiT relies on, cheapest first.

Run on a machine with a Tenstorrent device:
    python test_device_primitives.py

Opens a [1,1] mesh explicitly (single chip; avoids multi-chip ethernet paths).
Backward checks route operands through a LinearLayer so a grad graph exists —
tensors created via from_numpy are leaves and do not require grad themselves.
Each check prints OK/FAIL and the script exits nonzero on any failure.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import ttnn

import ttml
from ttml.modules import LinearLayer

FAILURES = []


def check(name):
    def wrap(fn):
        def run():
            try:
                fn()
                print(f"OK   {name}", flush=True)
            except Exception:
                print(f"FAIL {name}", flush=True)
                traceback.print_exc()
                FAILURES.append(name)
            finally:
                ttml.autograd.AutoContext.get_instance().reset_graph()

        return run

    return wrap


def to_np(t):
    return t.to_numpy(ttnn.DataType.FLOAT32)


def from_np(a):
    return ttml.autograd.Tensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


B, T, D = 4, 64, 128


@check("roundtrip rank-4 from_numpy/to_numpy")
def t_roundtrip():
    a = np.random.randn(B, 1, T, D).astype(np.float32)
    out = to_np(from_np(a))
    assert out.shape == (B, 1, T, D), out.shape
    assert np.allclose(out, a, atol=2e-2), np.abs(out - a).max()  # bf16 tolerance


@check("add broadcast pos-emb [1,1,T,D] + [B,1,T,D]")
def t_add_posemb():
    x, p = np.random.randn(B, 1, T, D).astype(np.float32), np.random.randn(1, 1, T, D).astype(np.float32)
    out = to_np(ttml.ops.binary.add(from_np(x), from_np(p)))
    assert np.allclose(out, x + p, atol=5e-2)


@check("mul row-broadcast [B,1,T,D] * [B,1,1,D] fwd+bwd through linear")
def t_mul_rows():
    x_np = np.random.randn(B, 1, T, D).astype(np.float32)
    s_np = np.random.randn(B, 1, 1, D).astype(np.float32)
    out = ttml.ops.binary.mul(from_np(x_np), from_np(s_np))
    out_np = to_np(out)
    # bf16: |err| <= ~1% of |product|; use scaled tolerance
    tol = 0.02 * np.abs(x_np * s_np) + 5e-2
    assert (np.abs(out_np - x_np * s_np) <= tol).all(), np.abs(out_np - x_np * s_np).max()
    # backward: grads must flow through the broadcast into linear params
    lin = LinearLayer(D, D, True)
    out2 = ttml.ops.binary.mul(lin(from_np(x_np)), from_np(s_np))
    loss = ttml.ops.loss.mse_loss(out2, from_np(np.zeros_like(x_np)))
    loss.backward(False)


@check("mul row-broadcast with grad on the [B,1,1,D] side")
def t_mul_rows_cond_grad():
    # In adaLN the modulation (small side) is the parameter-dependent branch.
    x_np = np.random.randn(B, 1, T, D).astype(np.float32)
    c_np = np.random.randn(B, 1, 1, D).astype(np.float32)
    lin = LinearLayer(D, D, True)
    out = ttml.ops.binary.mul(from_np(x_np), lin(from_np(c_np)))
    loss = ttml.ops.loss.mse_loss(out, from_np(np.zeros_like(x_np)))
    loss.backward(False)


@check("add row-broadcast [B,1,T,D] + [B,1,1,D] fwd+bwd through linear")
def t_add_rows():
    x_np = np.random.randn(B, 1, T, D).astype(np.float32)
    s_np = np.random.randn(B, 1, 1, D).astype(np.float32)
    out = ttml.ops.binary.add(from_np(x_np), from_np(s_np))
    assert np.allclose(to_np(out), x_np + s_np, atol=5e-2)
    lin = LinearLayer(D, D, True)
    out2 = ttml.ops.binary.add(from_np(x_np), lin(from_np(s_np)))
    loss = ttml.ops.loss.mse_loss(out2, from_np(np.zeros_like(x_np)))
    loss.backward(False)


@check("one-hot label embed [B,1,1,C] @ linear -> [B,1,1,D] fwd+bwd")
def t_label_embed():
    from diffusion import one_hot

    lin = LinearLayer(11, D, False)
    oh = one_hot(np.array([1, 3, 10, 0]), 11)
    out = lin(from_np(oh))
    arr = to_np(out)
    assert arr.shape[0] == B and arr.shape[-1] == D, arr.shape
    loss = ttml.ops.loss.mse_loss(out, from_np(np.zeros(arr.shape, dtype=np.float32)))
    loss.backward(False)


@check("composite SDPA non-causal (mask=None) fwd+bwd")
def t_sdpa():
    H, hd = 4, D // 4
    q_np = np.random.randn(B, H, T, hd).astype(np.float32)
    k_np = np.random.randn(B, H, T, hd).astype(np.float32)
    v_np = np.random.randn(B, H, T, hd).astype(np.float32)
    q, k, v = from_np(q_np), from_np(k_np), from_np(v_np)
    out = ttml.ops.attention.scaled_dot_product_attention_composite(q, k, v, None)
    att = q_np @ k_np.transpose(0, 1, 3, 2) / np.sqrt(hd)
    att = np.exp(att - att.max(-1, keepdims=True))
    att = att / att.sum(-1, keepdims=True)
    golden = att @ v_np
    got = to_np(out)
    assert np.allclose(got, golden, atol=1.5e-1), np.abs(got - golden).max()
    lin = LinearLayer(hd, hd, True)
    out2 = ttml.ops.attention.scaled_dot_product_attention_composite(lin(q), k, v, None)
    loss = ttml.ops.loss.mse_loss(out2, from_np(np.zeros_like(golden)))
    loss.backward(False)


@check("mse_loss value + grad through linear")
def t_mse():
    lin = LinearLayer(D, D, True)
    x = from_np(np.random.randn(B, 1, T, D))
    tgt = from_np(np.random.randn(B, 1, T, D))
    loss = ttml.ops.loss.mse_loss(lin(x), tgt)
    loss.backward(False)
    val = float(to_np(loss).reshape(-1)[0])
    assert np.isfinite(val)


@check("tiny DiT fwd + bwd + 5 optimizer steps decrease loss")
def t_dit_end2end():
    from dit_ttml import dit_tiny
    from diffusion import DiffusionSchedule, make_training_batch

    rng = np.random.default_rng(0)
    model = dit_tiny(in_dim=48, num_tokens=64, num_classes=10)
    sched = DiffusionSchedule()
    images = (rng.random((B, 3, 32, 32), dtype=np.float32) * 2 - 1).astype(np.float32)
    labels = rng.integers(0, 10, size=B)

    opt = ttml.optimizers.create_optimizer({"type": "AdamW", "lr": 1e-3}, model.parameters())

    losses = []
    for step in range(5):
        rng_step = np.random.default_rng(1)  # fixed batch -> loss must fall
        tokens, t_feats, onehot, target = make_training_batch(
            images, labels, sched, patch=4, model_dim=model.dim, rng=rng_step,
            cfg_drop_prob=0.0, null_class=10,
        )
        opt.zero_grad()
        pred = model(from_np(tokens), from_np(t_feats), from_np(onehot))
        loss = ttml.ops.loss.mse_loss(pred, from_np(target))
        loss.backward(False)
        opt.step()
        losses.append(float(to_np(loss).reshape(-1)[0]))
        ttml.autograd.AutoContext.get_instance().reset_graph()
    print(f"     losses: {[f'{v:.4f}' for v in losses]}", flush=True)
    assert losses[-1] < losses[0], losses


if __name__ == "__main__":
    ttml.open_device_mesh((1, 1))
    try:
        for fn in [
            t_roundtrip,
            t_add_posemb,
            t_mul_rows,
            t_mul_rows_cond_grad,
            t_add_rows,
            t_label_embed,
            t_sdpa,
            t_mse,
            t_dit_end2end,
        ]:
            fn()
    finally:
        if FAILURES:
            print("FAILED:", FAILURES, flush=True)
        else:
            print("ALL DEVICE PRIMITIVES OK", flush=True)
    sys.exit(1 if FAILURES else 0)

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""On-device gate for every primitive the SongUNet (edm_unet.py) relies on.

Run on a machine with a Tenstorrent device:
    python test_edm_primitives.py

Style mirrors test_device_primitives.py: [1,1] mesh, cheapest check first,
prints OK/FAIL per check, exits nonzero on any failure. Goldens are numpy
(already validated against torch by test_reference_unet.py on CPU); the two
torch-dependent checks (UNet block parity) skip cleanly if torch is absent.

bf16 tolerances: fwd |err| <= atol + rtol*|golden|; grads looser (they
accumulate more bf16 roundoff through matmul chains).
"""

from __future__ import annotations

import math
import sys
import traceback

import numpy as np
import ttnn

import ttml

FAILURES = []
SKIPPED = []


def check(name):
    def wrap(fn):
        def run():
            try:
                fn()
                print(f"OK   {name}", flush=True)
            except _Skip as s:
                print(f"SKIP {name}: {s}", flush=True)
                SKIPPED.append(name)
            except Exception:
                print(f"FAIL {name}", flush=True)
                traceback.print_exc()
                FAILURES.append(name)
            finally:
                ttml.autograd.AutoContext.get_instance().reset_graph()

        return run

    return wrap


class _Skip(Exception):
    pass


def to_np(t):
    return t.to_numpy(ttnn.DataType.FLOAT32)


def from_np(a, requires_grad=False):
    t = ttml.autograd.Tensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    if requires_grad:
        t.set_requires_grad(True)
    return t


def grad_np(t):
    return t.get_grad_tensor().to_numpy(ttnn.DataType.FLOAT32)


def assert_close(got, ref, rtol, atol, tag=""):
    err = np.abs(got - ref)
    tol = atol + rtol * np.abs(ref)
    bad = err > tol
    assert not bad.any(), f"{tag}: {bad.sum()}/{bad.size} elems out of tol, max err {err.max():.5f}"


# --- numpy goldens (same algorithms validated vs torch in test_reference_unet.py) ---


def np_im2col(x_nhwc):
    b, h, w, c = x_nhwc.shape
    p = np.pad(x_nhwc, ((0, 0), (1, 1), (1, 1), (0, 0)))
    views = [p[:, kh : kh + h, kw : kw + w, :] for kh in range(3) for kw in range(3)]
    return np.concatenate(views, axis=-1).reshape(b * h * w, 9 * c)


def np_conv3x3(x_nhwc, w_flat, bias):
    b, h, w, _ = x_nhwc.shape
    return (np_im2col(x_nhwc) @ w_flat + bias).reshape(b, h, w, -1)


def np_conv3x3_bwd(x_nhwc, w_flat, g_nhwc):
    b, h, w, c = x_nhwc.shape
    cout = w_flat.shape[1]
    g2 = g_nhwc.reshape(-1, cout)
    cols = np_im2col(x_nhwc)
    dw, db = cols.T @ g2, g2.sum(axis=0)
    dcols = (g2 @ w_flat.T).reshape(b, h, w, 9 * c)
    dp = np.zeros((b, h + 2, w + 2, c), dtype=np.float32)
    for kh in range(3):
        for kw in range(3):
            k = kh * 3 + kw
            dp[:, kh : kh + h, kw : kw + w, :] += dcols[..., k * c : (k + 1) * c]
    return dp[:, 1 : h + 1, 1 : w + 1, :], dw, db


def np_group_norm(x_nchw, groups, gamma, beta, eps=1e-6):
    b, c, h, w = x_nchw.shape
    g = x_nchw.reshape(b, groups, -1)
    mean = g.mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(g.var(-1, keepdims=True) + eps)
    xhat = ((g - mean) * rstd).reshape(b, c, h, w)
    return xhat * gamma.reshape(1, c, 1, 1) + beta.reshape(1, c, 1, 1), xhat, rstd.reshape(b, groups)


def np_group_norm_bwd(x_nchw, groups, gamma, g_out, eps=1e-6):
    b, c, h, w = x_nchw.shape
    _, xhat, _ = np_group_norm(x_nchw, groups, gamma, np.zeros_like(gamma), eps)
    dgamma = (g_out * xhat).sum(axis=(0, 2, 3))
    dbeta = g_out.sum(axis=(0, 2, 3))
    dxhat = (g_out * gamma.reshape(1, c, 1, 1)).reshape(b, groups, -1)
    xh = xhat.reshape(b, groups, -1)
    gg = x_nchw.reshape(b, groups, -1)
    rstd = 1.0 / np.sqrt(gg.var(-1, keepdims=True) + eps)
    m1 = dxhat.mean(-1, keepdims=True)
    m2 = (dxhat * xh).mean(-1, keepdims=True)
    dx = (rstd * (dxhat - m1 - xh * m2)).reshape(b, c, h, w)
    return dx.astype(np.float32), dgamma.astype(np.float32), dbeta.astype(np.float32)


def tokens(x_nchw):  # [B,C,H,W] -> [B,1,HW,C]
    b, c, h, w = x_nchw.shape
    return np.ascontiguousarray(x_nchw.transpose(0, 2, 3, 1)).reshape(b, 1, h * w, c)


def untokens(t, h, w):  # [B,1,HW,C] -> [B,H,W,C]
    return t.reshape(t.shape[0], h, w, t.shape[-1])


def mse_grad(pred_np):
    """d/dpred of mse_loss(pred, 0) = 2*pred/numel."""
    return (2.0 / pred_np.size) * pred_np


# ---------------------------------------------------------------------------


@check("RM NHWC pad/slice/concat: raw ttnn im2col vs numpy")
def t_im2col_raw():
    from edm_ops import _im2col

    rng = np.random.default_rng(0)
    b, h, w, c = 2, 8, 8, 32
    x = rng.standard_normal((b, h, w, c)).astype(np.float32)
    v = from_np(x.reshape(b, 1, h * w, c)).get_value()
    cols = _im2col(v, b, h, w, c)
    got = ttml.autograd.create_tensor(cols, False).to_numpy(ttnn.DataType.FLOAT32).reshape(b * h * w, 9 * c)
    assert_close(got, np_im2col(x), rtol=0.02, atol=0.02, tag="im2col")


@check("Permute fwd+bwd (NHWC <-> NCHW)")
def t_permute():
    from edm_ops import Permute

    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((2, 8, 8, 32)).astype(np.float32)
    x = from_np(x_np, requires_grad=True)
    y = Permute.apply(x, (0, 3, 1, 2))
    assert_close(to_np(y), x_np.transpose(0, 3, 1, 2), rtol=0.02, atol=0.02, tag="fwd")
    loss = ttml.ops.loss.mse_loss(y, from_np(np.zeros_like(x_np).transpose(0, 3, 1, 2)))
    loss.backward(False)
    assert_close(grad_np(x), mse_grad(x_np), rtol=0.05, atol=1e-4, tag="bwd")


@check("moreh.group_norm raw fwd+bwd vs numpy (N=4,C=128,32x32,G=32)")
def t_moreh_group_norm_raw():
    rng = np.random.default_rng(2)
    b, c, h, w, groups = 4, 128, 32, 32, 32
    x = rng.standard_normal((b, c, h, w)).astype(np.float32)
    gamma = rng.standard_normal(c).astype(np.float32) * 0.5 + 1.0
    beta = rng.standard_normal(c).astype(np.float32) * 0.1
    g_out = rng.standard_normal((b, c, h, w)).astype(np.float32)

    xv = from_np(x).get_value()
    gv = from_np(gamma.reshape(1, 1, 1, c)).get_value()
    bv = from_np(beta.reshape(1, 1, 1, c)).get_value()
    out, mean, rstd = ttnn_moreh.group_norm(
        xv, groups, eps=1e-6, gamma=gv, beta=bv, are_required_outputs=[True, True, True]
    )
    ref, _, _ = np_group_norm(x, groups, gamma, beta)
    got = ttml.autograd.create_tensor(out, False).to_numpy(ttnn.DataType.FLOAT32)
    assert_close(got, ref, rtol=0.03, atol=0.05, tag="fwd")

    gov = from_np(g_out).get_value()
    dx, dgamma, dbeta = ttnn_moreh.group_norm_backward(
        gov, xv, mean, rstd, groups, are_required_outputs=[True, True, True], gamma=gv
    )
    rdx, rdgamma, rdbeta = np_group_norm_bwd(x, groups, gamma, g_out)
    assert_close(ttml.autograd.create_tensor(dx, False).to_numpy(ttnn.DataType.FLOAT32), rdx, 0.05, 0.05, "dx")
    assert_close(
        ttml.autograd.create_tensor(dgamma, False).to_numpy(ttnn.DataType.FLOAT32).reshape(-1), rdgamma,
        0.05, 0.05 * abs(rdgamma).max(), "dgamma",
    )
    assert_close(
        ttml.autograd.create_tensor(dbeta, False).to_numpy(ttnn.DataType.FLOAT32).reshape(-1), rdbeta,
        0.05, 0.05 * abs(rdbeta).max(), "dbeta",
    )


@check("AvgPool2x2 / UpsampleNearest2 fwd parity + adjoint pair")
def t_pool_upsample():
    from edm_ops import AvgPool2x2, UpsampleNearest2

    rng = np.random.default_rng(3)
    b, c, h, w = 2, 32, 16, 16
    x_np = rng.standard_normal((b, c, h, w)).astype(np.float32)

    # avgpool fwd
    x = from_np(tokens(x_np), requires_grad=True)
    y = AvgPool2x2.apply(x, h, w)
    ref_pool = x_np.reshape(b, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5))
    assert_close(to_np(y), tokens(ref_pool), 0.03, 0.03, "pool fwd")
    # avgpool bwd == 0.25 * nearest-upsample of dOut
    loss = ttml.ops.loss.mse_loss(y, from_np(np.zeros_like(tokens(ref_pool))))
    loss.backward(False)
    g_out = mse_grad(tokens(ref_pool))  # [B,1,HW/4,C]
    ref_dx = 0.25 * untokens(g_out, h // 2, w // 2).repeat(2, axis=1).repeat(2, axis=2)
    assert_close(untokens(grad_np(x), h, w), ref_dx, 0.05, 1e-5, "pool bwd")
    ttml.autograd.AutoContext.get_instance().reset_graph()

    # upsample fwd
    x2 = from_np(tokens(ref_pool), requires_grad=True)
    y2 = UpsampleNearest2.apply(x2, h // 2, w // 2)
    ref_up = untokens(tokens(ref_pool), h // 2, w // 2).repeat(2, axis=1).repeat(2, axis=2)
    assert_close(untokens(to_np(y2), h, w), ref_up, 0.03, 0.03, "up fwd")
    # upsample bwd == 2x2 SUM pool of dOut (adjoint of nearest-repeat)
    loss2 = ttml.ops.loss.mse_loss(y2, from_np(np.zeros((b, 1, h * w, c), dtype=np.float32)))
    loss2.backward(False)
    g2 = untokens(mse_grad(tokens(ref_up.transpose(0, 3, 1, 2))), h, w)
    ref_dx2 = g2.reshape(b, h // 2, 2, w // 2, 2, c).sum(axis=(2, 4))
    assert_close(untokens(grad_np(x2), h // 2, w // 2), ref_dx2, 0.05, 1e-5, "up bwd")


@check("Conv3x3Im2col fwd vs numpy golden (B=2,C=32,16x16)")
def t_conv_fwd():
    from edm_ops import Conv3x3Im2col

    rng = np.random.default_rng(4)
    b, h, w, cin, cout = 2, 16, 16, 32, 32
    x_np = rng.standard_normal((b, h, w, cin)).astype(np.float32)
    w_np = (rng.standard_normal((9 * cin, cout)) / math.sqrt(9 * cin)).astype(np.float32)
    b_np = rng.standard_normal(cout).astype(np.float32) * 0.1
    x = from_np(x_np.reshape(b, 1, h * w, cin))
    wt = from_np(w_np.reshape(1, 1, 9 * cin, cout))
    bt = from_np(b_np.reshape(1, 1, 1, cout))
    out = Conv3x3Im2col.apply(x, wt, bt, h, w)
    ref = np_conv3x3(x_np, w_np, b_np)
    assert_close(untokens(to_np(out), h, w), ref, 0.05, 0.08, "conv fwd")


@check("Conv3x3Im2col bwd grads (dX, dW, dB) vs numpy golden")
def t_conv_bwd():
    from edm_ops import Conv3x3Im2col

    rng = np.random.default_rng(5)
    b, h, w, cin, cout = 2, 16, 16, 32, 32
    x_np = rng.standard_normal((b, h, w, cin)).astype(np.float32)
    w_np = (rng.standard_normal((9 * cin, cout)) / math.sqrt(9 * cin)).astype(np.float32)
    b_np = np.zeros(cout, dtype=np.float32)
    x = from_np(x_np.reshape(b, 1, h * w, cin), requires_grad=True)
    wt = from_np(w_np.reshape(1, 1, 9 * cin, cout), requires_grad=True)
    bt = from_np(b_np.reshape(1, 1, 1, cout), requires_grad=True)
    out = Conv3x3Im2col.apply(x, wt, bt, h, w)
    loss = ttml.ops.loss.mse_loss(out, from_np(np.zeros((b, 1, h * w, cout), dtype=np.float32)))
    loss.backward(False)
    g = mse_grad(np_conv3x3(x_np, w_np, b_np))  # [B,H,W,Cout] fp32 golden upstream grad
    rdx, rdw, rdb = np_conv3x3_bwd(x_np, w_np, g)
    scale_w = max(abs(rdw).max(), 1e-8)
    assert_close(untokens(grad_np(x), h, w), rdx, 0.08, 0.05 * abs(rdx).max(), "dX")
    assert_close(grad_np(wt).reshape(9 * cin, cout), rdw, 0.08, 0.05 * scale_w, "dW")
    assert_close(grad_np(bt).reshape(-1), rdb, 0.08, 0.05 * abs(rdb).max(), "dB")


@check("GroupNormMoreh wrapper fwd+bwd on tokens (C=64,16x16,G=16)")
def t_groupnorm_wrapper():
    from edm_ops import GroupNormMoreh

    rng = np.random.default_rng(6)
    b, c, h, w, groups = 2, 64, 16, 16, 16
    x_np = rng.standard_normal((b, c, h, w)).astype(np.float32)
    gamma = (rng.standard_normal(c) * 0.3 + 1.0).astype(np.float32)
    beta = (rng.standard_normal(c) * 0.1).astype(np.float32)
    x = from_np(tokens(x_np), requires_grad=True)
    gt = from_np(gamma.reshape(1, 1, 1, c), requires_grad=True)
    bt = from_np(beta.reshape(1, 1, 1, c), requires_grad=True)
    y = GroupNormMoreh.apply(x, gt, bt, groups, h, w)
    ref, _, _ = np_group_norm(x_np, groups, gamma, beta)
    assert_close(untokens(to_np(y), h, w), tokens(ref).reshape(b, h, w, c), 0.03, 0.05, "fwd")
    loss = ttml.ops.loss.mse_loss(y, from_np(np.zeros((b, 1, h * w, c), dtype=np.float32)))
    loss.backward(False)
    g_nchw = untokens(mse_grad(tokens(ref)), h, w).transpose(0, 3, 1, 2)
    rdx, rdgamma, rdbeta = np_group_norm_bwd(x_np, groups, gamma, g_nchw)
    assert_close(untokens(grad_np(x), h, w).transpose(0, 3, 1, 2), rdx, 0.08, 0.05 * abs(rdx).max(), "dx")
    assert_close(grad_np(gt).reshape(-1), rdgamma, 0.08, 0.05 * abs(rdgamma).max(), "dgamma")
    assert_close(grad_np(bt).reshape(-1), rdbeta, 0.08, 0.05 * abs(rdbeta).max(), "dbeta")


@check("ConcatChannels fwd+bwd (skip concat / slice)")
def t_concat_channels():
    from edm_ops import ConcatChannels

    rng = np.random.default_rng(7)
    a_np = rng.standard_normal((2, 1, 64, 32)).astype(np.float32)
    b_np = rng.standard_normal((2, 1, 64, 64)).astype(np.float32)
    a = from_np(a_np, requires_grad=True)
    b = from_np(b_np, requires_grad=True)
    y = ConcatChannels.apply(a, b)
    assert_close(to_np(y), np.concatenate([a_np, b_np], axis=-1), 0.02, 0.02, "fwd")
    loss = ttml.ops.loss.mse_loss(y, from_np(np.zeros_like(to_np(y))))
    loss.backward(False)
    g = mse_grad(np.concatenate([a_np, b_np], axis=-1))
    assert_close(grad_np(a), g[..., :32], 0.05, 1e-5, "dA")
    assert_close(grad_np(b), g[..., 32:], 0.05, 1e-5, "dB")


def _torch_or_skip():
    try:
        import torch  # noqa: F401

        return torch
    except ImportError:
        raise _Skip("torch not installed on this machine")


def _copy_torch_params(tt_module, torch_module):
    """Copy a torch mirror module's params into the ttml module (same attr names)."""
    import ml_dtypes

    tsd = {k: v.detach().float().numpy() for k, v in torch_module.named_parameters()}
    for name, t in tt_module.named_parameters():
        key = name
        if key.endswith(".gamma") or key == "gamma":
            key = key[: -len("gamma")] + "weight"
        elif key.endswith(".beta") or key == "beta":
            key = key[: -len("beta")] + "bias"
        src = tsd[key]
        cur = t.to_numpy(ttnn.DataType.FLOAT32)
        t.set_value(
            ttml.autograd.Tensor.from_numpy(
                src.reshape(cur.shape).astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ).get_value()
        )


@check("full UNet block fwd+bwd parity vs reference_unet (attn block)")
def t_block_parity():
    torch = _torch_or_skip()
    from edm_unet import UNetBlock as TTBlock
    from reference_unet import UNetBlock as TorchBlock

    rng = np.random.default_rng(8)
    b, cin, cout, h, w, emb_dim = 2, 32, 64, 16, 16, 64
    torch.manual_seed(0)
    tblock = TorchBlock(cin, cout, emb_dim, attn=True, dropout=0.0)
    tblock.eval()
    ttblock = TTBlock(cin, cout, emb_dim, attn=True, dropout=0.0)
    _copy_torch_params(ttblock, tblock)

    x_np = rng.standard_normal((b, cin, h, w)).astype(np.float32)
    emb_np = rng.standard_normal((b, 1, 1, emb_dim)).astype(np.float32)

    ref = tblock(torch.from_numpy(x_np), torch.from_numpy(emb_np.reshape(b, emb_dim))).detach().numpy()
    x = from_np(tokens(x_np), requires_grad=True)
    emb = from_np(emb_np)
    out = ttblock(x, emb, h, w)
    got = untokens(to_np(out), h, w).transpose(0, 3, 1, 2)
    assert_close(got, ref, 0.05, 0.08, "fwd")

    # bwd parity on dX through an identical mse loss
    xt = torch.from_numpy(x_np).requires_grad_(True)
    loss_t = torch.nn.functional.mse_loss(
        tblock(xt, torch.from_numpy(emb_np.reshape(b, emb_dim))), torch.zeros(b, cout, h, w)
    )
    loss_t.backward()
    loss = ttml.ops.loss.mse_loss(out, from_np(np.zeros((b, 1, h * w, cout), dtype=np.float32)))
    loss.backward(False)
    rdx = xt.grad.numpy()
    assert_close(
        untokens(grad_np(x), h, w).transpose(0, 3, 1, 2), rdx, 0.10, 0.08 * max(abs(rdx).max(), 1e-8), "dX"
    )


@check("tiny full-UNet e2e: fwd + bwd + 5 optimizer steps decrease loss")
def t_unet_end2end():
    from edm import make_edm_batch_image, nchw_to_nhwc_tokens
    from edm_unet import song_unet_tiny

    rng = np.random.default_rng(0)
    batch = 2
    model = song_unet_tiny()
    images = (rng.random((batch, 3, 32, 32), dtype=np.float32) * 2 - 1).astype(np.float32)
    labels = rng.integers(0, 10, size=batch)
    opt = ttml.optimizers.create_optimizer({"type": "AdamW", "lr": 1e-3}, model.parameters())

    losses = []
    for _ in range(5):
        rng_step = np.random.default_rng(1)  # fixed batch -> loss must fall
        net_in, feats, onehot, target = make_edm_batch_image(
            images, labels, model.emb_dim, rng_step, null_class=10
        )
        opt.zero_grad()
        pred = model(from_np(nchw_to_nhwc_tokens(net_in)), from_np(feats), from_np(onehot))
        loss = ttml.ops.loss.mse_loss(pred, from_np(nchw_to_nhwc_tokens(target)))
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
            t_im2col_raw,
            t_permute,
            t_moreh_group_norm_raw,
            t_pool_upsample,
            t_conv_fwd,
            t_conv_bwd,
            t_groupnorm_wrapper,
            t_concat_channels,
            t_block_parity,
            t_unet_end2end,
        ]:
            fn()
    finally:
        if FAILURES:
            print("FAILED:", FAILURES, flush=True)
        else:
            print("ALL EDM UNET PRIMITIVES OK" + (f" (skipped: {SKIPPED})" if SKIPPED else ""), flush=True)
    sys.exit(1 if FAILURES else 0)

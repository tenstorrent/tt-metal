# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU validation for reference_unet.py (no device needed).

    python test_reference_unet.py [--full-overfit]

Checks, cheapest first:
  1. im2col/col2im numpy implementation (the EXACT algorithm edm_ops.py runs
     on device, same ROW_ORDER) matches torch F.conv2d forward AND autograd
     gradients — this pins the flattened-weight convention.
  2. Full EDM CIFAR-10 config parameter count is printed and must land in
     55M..62M.
  3. Shape walk: full config forward prints every block's output shape.
  4. Single-batch EDM overfit on random data: 200 Adam steps, loss must at
     least halve (reduced config by default for runtime; --full-overfit runs
     the 56M model).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from edm import make_edm_batch_image
from reference_unet import Conv3x3Flat, TorchSongUNet, param_count

FAILURES: list[str] = []


def check(name, fn):
    try:
        fn()
        print(f"OK   {name}", flush=True)
    except Exception as e:
        print(f"FAIL {name}: {e}", flush=True)
        import traceback

        traceback.print_exc()
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1. numpy im2col / col2im — the device algorithm, validated against torch
# ---------------------------------------------------------------------------


def numpy_im2col(x_nhwc: np.ndarray) -> np.ndarray:
    """[B,H,W,C] -> [B*H*W, 9C]; slice order (kh, kw) row-major == ROW_ORDER."""
    b, h, w, c = x_nhwc.shape
    p = np.pad(x_nhwc, ((0, 0), (1, 1), (1, 1), (0, 0)))
    views = [p[:, kh : kh + h, kw : kw + w, :] for kh in range(3) for kw in range(3)]
    return np.concatenate(views, axis=-1).reshape(b * h * w, 9 * c)


def numpy_conv3x3_fwd(x_nhwc, w_flat, bias):
    b, h, w, _ = x_nhwc.shape
    return (numpy_im2col(x_nhwc) @ w_flat + bias).reshape(b, h, w, -1)


def numpy_conv3x3_bwd(x_nhwc, w_flat, g_nhwc):
    """Returns (dx [B,H,W,C], dW [9C,Cout], db [Cout]) — mirrors edm_ops backward."""
    b, h, w, c = x_nhwc.shape
    cout = w_flat.shape[1]
    g2 = g_nhwc.reshape(b * h * w, cout)
    cols = numpy_im2col(x_nhwc)
    dw = cols.T @ g2
    db = g2.sum(axis=0)
    dcols = (g2 @ w_flat.T).reshape(b, h, w, 9 * c)
    dp = np.zeros((b, h + 2, w + 2, c), dtype=x_nhwc.dtype)
    for kh in range(3):
        for kw in range(3):
            k = kh * 3 + kw
            dp[:, kh : kh + h, kw : kw + w, :] += dcols[..., k * c : (k + 1) * c]
    return dp[:, 1 : h + 1, 1 : w + 1, :], dw, db


def t_im2col_row_order():
    rng = np.random.default_rng(0)
    b, h, w, cin, cout = 2, 8, 8, 5, 7  # deliberately non-tile shapes
    x = rng.standard_normal((b, cin, h, w)).astype(np.float32)
    conv = Conv3x3Flat(cin, cout)
    with torch.no_grad():
        golden = conv(torch.from_numpy(x)).numpy()
    got = numpy_conv3x3_fwd(
        np.ascontiguousarray(x.transpose(0, 2, 3, 1)),
        conv.weight.detach().numpy(),
        conv.bias.detach().numpy(),
    ).transpose(0, 3, 1, 2)
    err = np.abs(got - golden).max()
    assert err < 1e-4, f"im2col fwd mismatch, max err {err}"


def t_col2im_grads():
    rng = np.random.default_rng(1)
    b, h, w, cin, cout = 2, 6, 6, 4, 3
    x = torch.from_numpy(rng.standard_normal((b, cin, h, w)).astype(np.float32)).requires_grad_(True)
    conv = Conv3x3Flat(cin, cout)
    g = torch.from_numpy(rng.standard_normal((b, cout, h, w)).astype(np.float32))
    out = conv(x)
    out.backward(g)
    x_nhwc = x.detach().numpy().transpose(0, 2, 3, 1)
    g_nhwc = g.numpy().transpose(0, 2, 3, 1)
    dx, dw, db = numpy_conv3x3_bwd(x_nhwc, conv.weight.detach().numpy(), g_nhwc)
    e_dx = np.abs(dx.transpose(0, 3, 1, 2) - x.grad.numpy()).max()
    e_dw = np.abs(dw - conv.weight.grad.numpy()).max()
    e_db = np.abs(db - conv.bias.grad.numpy()).max()
    assert e_dx < 1e-4 and e_dw < 1e-3 and e_db < 1e-3, (e_dx, e_dw, e_db)


# ---------------------------------------------------------------------------
# 2 + 3. full-config param count and shape walk
# ---------------------------------------------------------------------------


def t_param_count_and_shapes():
    torch.manual_seed(0)
    model = TorchSongUNet()  # exact EDM CIFAR-10 config
    n = param_count(model)
    print(f"     full CIFAR-10 SongUNet parameters: {n:,} ({n/1e6:.2f}M)", flush=True)
    assert 55e6 < n < 62e6, f"param count {n} outside 55M..62M"
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    feats = torch.randn(2, 1, 1, model.emb_dim)
    onehot = F.one_hot(torch.tensor([3, 7]), 11).float().reshape(2, 1, 1, 11)
    with torch.no_grad():
        out = model(x, feats, onehot, verbose=True)
    assert out.shape == (2, 3, 32, 32), out.shape


# ---------------------------------------------------------------------------
# 4. single-batch EDM overfit
# ---------------------------------------------------------------------------


def t_overfit(full: bool):
    torch.manual_seed(0)
    if full:
        model, batch = TorchSongUNet(dropout=0.0), 4
    else:  # same structure, smaller widths, still every block kind
        model = TorchSongUNet(model_channels=32, num_blocks=2, dropout=0.0)
        batch = 4
    model.train()
    rng = np.random.default_rng(0)
    images = (rng.random((batch, 3, 32, 32), dtype=np.float32) * 2 - 1).astype(np.float32)
    labels = rng.integers(0, 10, size=batch)
    net_in, feats, onehot, target = make_edm_batch_image(
        images, labels, model.emb_dim, np.random.default_rng(1), null_class=10
    )
    xt = torch.from_numpy(net_in)
    ft = torch.from_numpy(feats)
    ot = torch.from_numpy(onehot)
    tt = torch.from_numpy(target)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    t0 = time.time()
    for step in range(200):
        opt.zero_grad()
        loss = F.mse_loss(model(xt, ft, ot), tt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 25 == 0:
            print(f"     step {step:4d} loss {loss.item():.5f}", flush=True)
    print(f"     step  199 loss {losses[-1]:.5f}  ({time.time()-t0:.1f}s)", flush=True)
    assert losses[-1] < 0.5 * losses[0], f"loss did not halve: {losses[0]:.5f} -> {losses[-1]:.5f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-overfit", action="store_true", help="overfit the full 56M model (slow)")
    args = ap.parse_args()

    check("im2col ROW_ORDER fwd vs F.conv2d", t_im2col_row_order)
    check("col2im backward vs torch autograd", t_col2im_grads)
    check("param count + shape walk (full config)", t_param_count_and_shapes)
    check(f"single-batch EDM overfit ({'full' if args.full_overfit else 'reduced'})", lambda: t_overfit(args.full_overfit))

    if FAILURES:
        print("FAILED:", FAILURES, flush=True)
    else:
        print("ALL REFERENCE UNET CHECKS OK", flush=True)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()

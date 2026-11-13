# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S1 (second stage): Hy4 iHC on the DSV4 mHC kernels (identity comb, hc_fn padded 8 -> 24 rows) on REAL layer-1 streams vs the fp32 oracle formulas.
Per T = 1, 4, 16 tokens: pre/post PCC, comb vs identity, collapse + norm (bf16) PCC, expand PCC; attention side (x4_in, y = h stand-in) and MLP side (x4_mid, y = real MoE output
-> x4_out of the oracle layer)."""
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import (
    ACTS,
    close_mesh,
    dev0,
    open_mesh,
    p,
    pcc,
    rep,
    traced_us,
)
from models.demos.blackhole.hy4_preview.tt.hy4_ihc import Hy4IHC

HID = 6144
NORM_EPS = float(os.environ.get("HY4_NORM_EPS", "1e-5"))
R = torch.load(ACTS + "ihc_l1.pt")
acts = torch.load(ACTS + "acts_l0_l1.pt")
cat = lambda k: torch.cat([t.reshape(-1, *t.shape[2:]) if t.dim() > 3 else t.reshape(-1, t.shape[-1]) for t in R[k]])
x4_in, pre_a, post_a, h_a = cat("x4_in"), cat("pre_a"), cat("post_a"), cat("h_a")
x4_mid, pre_m, post_m, h_m, x4_out = cat("x4_mid"), cat("pre_m"), cat("post_m"), cat("h_m"), cat("x4_out")
Ymoe = torch.cat([y.reshape(-1, HID) for h, y, r in acts["moe"]])
md = open_mesh()
try:
    for side, hp, nw, X4, PRE, POST, H in (
        ("attn", R["hc_attn"], R["norm_in"], x4_in, pre_a, post_a, h_a),
        ("mlp", R["hc_mlp"], R["norm_post"], x4_mid, pre_m, post_m, h_m),
    ):
        t0 = time.time()
        ihc = Hy4IHC(
            md, hp["hc_fn"], hp["hc_base"], hp["hc_scale"], dim=HID, hc_eps=R["cfg"]["hc_eps"], norm_eps=NORM_EPS
        )
        p(f"{side}: Hy4IHC built {time.time()-t0:.1f}s")
        w = rep(md, nw.float().reshape(1, 1, 1, HID), dtype=ttnn.float32)
        for T in (1, 4, 16):
            idx = torch.randperm(X4.shape[0], generator=torch.Generator().manual_seed(7 + T))[:T]
            x = rep(md, X4[idx].reshape(T, 1, 4, HID), dtype=ttnn.float32)
            pre, post, comb = ihc.mixes(x)
            ttnn.synchronize_device(md)
            g_pre, g_post, g_comb = (
                dev0(pre).reshape(T, 4).float(),
                dev0(post).reshape(T, 4).float(),
                dev0(comb).reshape(T, 4, 4).float(),
            )
            eye = torch.eye(4).expand(T, 4, 4)
            p(
                f"{side} T={T} pre PCC {pcc(g_pre, PRE[idx]):.6f} (maxabs {(g_pre-PRE[idx]).abs().max():.2e})  post PCC {pcc(g_post, POST[idx]):.6f} (maxabs {(g_post-POST[idx]).abs().max():.2e})  comb-I maxabs {(g_comb-eye).abs().max():.2e}"
            )
            h, h_tok = ihc.collapse_norm_rm(x, pre, w, NORM_EPS)
            ttnn.synchronize_device(md)
            g_h = dev0(h).reshape(T, HID).float()
            p(
                f"{side} T={T} collapse+norm PCC {pcc(g_h, H[idx].reshape(T, HID)):.6f} (maxabs {(g_h-H[idx].reshape(T,HID)).abs().max():.3e}; ref rms {H[idx].pow(2).mean().sqrt():.3f})"
            )
            y = (H[idx] if side == "attn" else Ymoe[idx]).reshape(1, 1, T, HID)
            ty = rep(md, y.to(torch.bfloat16))
            xn = ihc.expand(ty, x, post, comb)
            ttnn.synchronize_device(md)
            g_x = dev0(xn).reshape(T, 4, HID).float()
            ref = (
                X4[idx] + POST[idx].unsqueeze(-1) * y.reshape(T, 1, HID).to(torch.bfloat16).float()
                if side == "attn"
                else x4_out[idx]
            )
            p(
                f"{side} T={T} expand PCC {pcc(g_x, ref):.6f} (maxabs {(g_x-ref).abs().max():.3e}; ref rms {ref.pow(2).mean().sqrt():.3f})"
            )
            if side == "mlp":
                us_m = traced_us(md, lambda: ihc.mixes(x))
                us_c = traced_us(md, lambda: ihc.collapse_norm_rm(x, pre, w, NORM_EPS))
                us_e = traced_us(md, lambda: ihc.expand(ty, x, post, comb))
                p(f"iHC T={T} traced: mixes {us_m:.1f} us, collapse+norm(+rm) {us_c:.1f} us, expand {us_e:.1f} us")
        del ihc
finally:
    close_mesh(md)

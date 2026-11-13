# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused iHC with the expand folded into the next pre-kernel, on REAL layer-1 streams (one layer = two sub-blocks):
    pre_a(x_in) -> h_a, pp_a;  pre_m(x_in, y_a, pp_a) -> x_mid = x_in + post_a*y_a (checked vs oracle x4_mid), h_m, pp_m;  pre_a'(x_mid, y_m, pp_m) -> x_out (checked vs oracle x4_out)
(y_a, y_m = the sub-block outputs reconstructed from the oracle streams, bf16). Env: HY4_T=1,4,8  HY4_TIME=1"""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, p, pcc, rep, traced_us
from models.demos.blackhole.hy4_preview.tt.hy4_ihc_fused import Hy4IHCFused

HID = 6144
NORM_EPS = 1e-5
TS = [int(v) for v in os.environ.get("HY4_T", "1,4,8").split(",")]
TIME = os.environ.get("HY4_TIME", "1") == "1"
R = torch.load(ACTS + "ihc_l1.pt")
cat = lambda k: torch.cat([t.reshape(-1, *t.shape[2:]) if t.dim() > 3 else t.reshape(-1, t.shape[-1]) for t in R[k]])
x_in, pre_a, post_a, h_a = cat("x4_in"), cat("pre_a"), cat("post_a"), cat("h_a")
x_mid, pre_m, post_m, h_m, x_out = cat("x4_mid"), cat("pre_m"), cat("post_m"), cat("h_m"), cat("x4_out")
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=200_000_000)
d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])
hp_a, hp_m = R["hc_attn"], R["hc_mlp"]
try:
    fa = Hy4IHCFused(
        md, hp_a["hc_fn"], hp_a["hc_base"], hp_a["hc_scale"], dim=HID, hc_eps=R["cfg"]["hc_eps"], norm_eps=NORM_EPS
    )
    fm = Hy4IHCFused(
        md, hp_m["hc_fn"], hp_m["hc_base"], hp_m["hc_scale"], dim=HID, hc_eps=R["cfg"]["hc_eps"], norm_eps=NORM_EPS
    )
    for T in TS:
        idx = torch.randperm(x_in.shape[0], generator=torch.Generator().manual_seed(11 + T))[:T]
        # sub-block outputs reconstructed from the oracle streams (stream 0 carries y: x' = x + post_i * y)
        y_a = ((x_mid[idx] - x_in[idx])[:, 0, :] / post_a[idx][:, 0:1]).to(torch.bfloat16)
        y_m = ((x_out[idx] - x_mid[idx])[:, 0, :] / post_m[idx][:, 0:1]).to(torch.bfloat16)
        wa = fa.norm_weight(R["norm_in"], T)
        wm = fm.norm_weight(R["norm_post"], T)
        x0 = rep(md, x_in[idx].reshape(1, 1, 4 * T, HID), dtype=ttnn.float32)
        ya = rep(md, y_a.reshape(1, 1, T, HID))
        ym = rep(md, y_m.reshape(1, 1, T, HID))
        h1, _, pp1 = fa.pre(x0, wa)
        h2, _, pp2, x1 = fm.pre(x0, wm, y=ya, pp_prev=pp1)
        h3, _, pp3, x2 = fa.pre(x1, wa, y=ym, pp_prev=pp2)
        ttnn.synchronize_device(md)
        ref_x1 = x_in[idx] + post_a[idx].unsqueeze(-1) * y_a.float().unsqueeze(1)
        ref_x2 = x_mid[idx] + post_m[idx].unsqueeze(-1) * y_m.float().unsqueeze(1)
        g_x1 = d0(x1).reshape(T, 4, HID).float()
        g_x2 = d0(x2).reshape(T, 4, HID).float()
        g_h1 = d0(h1).reshape(T, HID).float()
        g_h2 = d0(h2).reshape(T, HID).float()
        g_pp2 = d0(pp2).reshape(32, 32).float()
        g_pp1 = d0(pp1).reshape(32, 32).float()
        ref_dev = x_in[idx] + g_pp1[:T, 4:8].unsqueeze(-1) * y_a.float().unsqueeze(
            1
        )  # same expand with the DEVICE post: isolates the expand arithmetic
        p(
            f"chain T={T}: expand arithmetic only (device post): x_mid rel L2 {float((g_x1-ref_dev).norm()/ref_dev.norm()):.2e} maxabs {(g_x1-ref_dev).abs().max():.2e}"
        )
        rel = lambda a, b: float((a - b).norm() / b.norm())
        p(
            f"chain T={T}: expand rel L2 err vs bf16-y reference: x_mid {rel(g_x1, ref_x1):.2e}  x_out {rel(g_x2, ref_x2):.2e}  (bf16 rounding would be 2e-3)"
        )
        p(f"chain T={T}: h_a vs oracle PCC {pcc(g_h1, h_a[idx].reshape(T, HID)):.6f}")
        p(
            f"chain T={T}: x_mid (expand fused into pre_m) vs oracle x4_mid PCC {pcc(g_x1, x_mid[idx]):.7f} maxabs {(g_x1-x_mid[idx]).abs().max():.2e} | vs bf16-y reference {pcc(g_x1, ref_x1):.8f} maxabs {(g_x1-ref_x1).abs().max():.2e}"
        )
        p(
            f"chain T={T}: pre_m {pcc(g_pp2[:T,0:4], pre_m[idx]):.6f} post_m {pcc(g_pp2[:T,4:8], post_m[idx]):.6f} h_m {pcc(g_h2, h_m[idx].reshape(T, HID)):.6f}"
        )
        p(
            f"chain T={T}: x_out (expand fused into next pre_a) vs oracle x4_out PCC {pcc(g_x2, x_out[idx]):.7f} maxabs {(g_x2-x_out[idx]).abs().max():.2e} | vs reference {pcc(g_x2, ref_x2):.8f}"
        )
        if TIME:
            xb = rep(md, x_in[idx].reshape(1, 1, 4 * T, HID), dtype=ttnn.float32)
            us_pre = traced_us(md, lambda: fa.pre(xb, wa))
            us_exp = traced_us(md, lambda: fm.pre(xb, wm, y=ya, pp_prev=pp1, x_out=xb)[:3])
            p(
                f"chain T={T} TRACED us: pre (no expand) {us_pre:.1f}   pre with fused expand (in place) {us_exp:.1f}   => per layer (2 calls) {2*us_exp:.1f} us"
            )
finally:
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

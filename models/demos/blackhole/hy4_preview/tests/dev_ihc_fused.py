# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused Hy4 iHC pre-kernel (Hy4IHCFused.pre) on REAL layer-1 streams: PCC vs the fp32 oracle values (pre / post / h) and vs the existing identity-comb DSV4-kernel path
(Hy4IHC), us per call traced (fused vs mixes + collapse_norm_rm of the old path). Env: HY4_T=1,2,4,8,16  HY4_SIDES=attn,mlp  HY4_TIME=1  HY4_RM=1"""
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, p, pcc, rep, traced_us
from models.demos.blackhole.hy4_preview.tt.hy4_ihc import Hy4IHC
from models.demos.blackhole.hy4_preview.tt.hy4_ihc_fused import Hy4IHCFused

HID = 6144
NORM_EPS = float(os.environ.get("HY4_NORM_EPS", "1e-5"))
TS = [int(v) for v in os.environ.get("HY4_T", "1,2,4,8,16").split(",")]
SIDES = os.environ.get("HY4_SIDES", "attn,mlp").split(",")
TIME = os.environ.get("HY4_TIME", "1") == "1"
RM = os.environ.get("HY4_RM", "1") == "1"
R = torch.load(ACTS + "ihc_l1.pt")
cat = lambda k: torch.cat([t.reshape(-1, *t.shape[2:]) if t.dim() > 3 else t.reshape(-1, t.shape[-1]) for t in R[k]])
x4_in, pre_a, post_a, h_a = cat("x4_in"), cat("pre_a"), cat("post_a"), cat("h_a")
x4_mid, pre_m, post_m, h_m = cat("x4_mid"), cat("pre_m"), cat("post_m"), cat("h_m")
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=200_000_000)
d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])
res = {}
try:
    for side, hp, nw, X4, PRE, POST, H in (
        ("attn", R["hc_attn"], R["norm_in"], x4_in, pre_a, post_a, h_a),
        ("mlp", R["hc_mlp"], R["norm_post"], x4_mid, pre_m, post_m, h_m),
    ):
        if side not in SIDES:
            continue
        t0 = time.time()
        fu = Hy4IHCFused(
            md, hp["hc_fn"], hp["hc_base"], hp["hc_scale"], dim=HID, hc_eps=R["cfg"]["hc_eps"], norm_eps=NORM_EPS
        )
        old = Hy4IHC(
            md, hp["hc_fn"], hp["hc_base"], hp["hc_scale"], dim=HID, hc_eps=R["cfg"]["hc_eps"], norm_eps=NORM_EPS
        )
        p(f"{side}: built {time.time()-t0:.1f}s")
        w = rep(md, nw.float().reshape(1, 1, 1, HID), dtype=ttnn.float32)
        for T in TS:
            idx = torch.randperm(X4.shape[0], generator=torch.Generator().manual_seed(7 + T))[:T]
            x = rep(md, X4[idx].reshape(1, 1, 4 * T, HID), dtype=ttnn.float32)  # packed streams (fused kernel)
            x_old = rep(md, X4[idx].reshape(T, 1, 4, HID), dtype=ttnn.float32)
            w4 = fu.norm_weight(nw, T)
            h, h_tok, pp = fu.pre(x, w4, emit_rm=RM)
            ttnn.synchronize_device(md)
            if os.environ.get("HY4_IHC_DBG", "0") == "1":
                dbg = (
                    d0(fu.dbg).reshape(32, 32, 32)[:, :2, :16].contiguous().view(torch.int32).to(torch.int64)
                    & 0xFFFFFFFF
                )
                rd, wr = dbg[:, 0, :11], dbg[:, 1, :5]
                pgs = d0(fu.dbg).reshape(32, 32, 32).contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
                comp = torch.cat([pgs[:, 2, :16], pgs[:, 3, :2]], dim=1)
                t0 = rd[:, 0].min()
                p("RAW core0 reader", rd[0].tolist(), "comp", comp[0].tolist(), "writer", wr[0].tolist())
                p(f"{side} T={T} DBG reader start skew (max-min) {int(rd[:,0].max()-rd[:,0].min())} cyc")
                if os.environ.get("HY4_DUMP", "0") == "1":
                    dd = d0(fu.dbg).reshape(32, 32, 32)
                    # page k: face 0 rows 16..31? words 256.. = byte 1024 = tile row 8..: reinterpret raw tile words
                    raw = d0(fu.dbg)
                    p("dbg raw shape", tuple(raw.shape))
                    full = raw.reshape(32, 1024) if raw.numel() == 32 * 1024 else raw.reshape(-1)
                    xs = X4[idx].reshape(T, 4, 6144).float()
                    fnh = hp["hc_fn"].float().reshape(8, 4, 6144)
                    for kk in (0, 5):
                        sl = slice(kk * 192, (kk + 1) * 192)
                        ref = torch.einsum("tic,mic->tm", xs[:, :, sl], fnh[:, :, sl])
                        mean_sq = xs[:, :, sl].square().sum((1, 2)) / (4 * 6144)
                        # tile layout: dbg tensor is a TILE tensor; to_torch un-tiles -> row-major rows. our data sits at tile words 256.. (face 0 of a re-tiled view) -> read from the untiled [32,32] page as rows 0..15 of face0 = rows 8.. ?
                        pg = raw.reshape(32, 32, 32)[kk]
                        p(
                            f"core {kk} dump (untiled rows 0..{T-1}, cols 0..9):",
                            pg[0:T, 0:9].tolist() if False else "see below",
                        )
                        p("   ref", ref[:, :].tolist(), mean_sq.tolist())
                        p("   got rows 16..16+T", pg[0:T, 16:25].tolist())
                names_r = [
                    "start",
                    "inputs landed",
                    "part ready(compute Z..P)",
                    "ex1 done",
                    "pp ready(compute..sigmoid)",
                    "A built",
                    "p2 ready(compute collapse+sq)",
                    "ex2 done",
                ]
                for i, nm in enumerate(names_r):
                    d = (rd[:, i] - rd[:, 0]).float()
                    p(f"   reader {nm:34s} median {d.median():8.0f} max {d.max():8.0f} cyc")
                names_c = [
                    "start",
                    "inputs",
                    "squares",
                    "Z",
                    "ZM",
                    "U",
                    "Q",
                    "PART",
                    "recv sum FL(+wait)",
                    "FR",
                    "MIX",
                    "pp",
                    "collapse HS",
                    "sqh",
                    "P2",
                    "tot wait",
                    "rs",
                    "tmp",
                    "out",
                ]
                prev = None
                for i in range(18):
                    d = (comp[:, i] - rd[:, 0]).float()
                    p(
                        f"   compute stamp {i:2d} (after: {names_c[i] if i < len(names_c) else ''}) median {d.median():8.0f} max {d.max():8.0f} cyc"
                    )
                names_w = ["start", "cst built", "ones built", "out ready", "out written"]
                for i, nm in enumerate(names_w):
                    d = (wr[:, i] - rd[:, 0]).float()
                    p(f"   writer {nm:34s} median {d.median():8.0f} max {d.max():8.0f} cyc (vs reader start)")
            g_pp = d0(pp).reshape(32, 32).float()
            g_pre, g_post = g_pp[:T, 0:4], g_pp[:T, 4:8]
            if os.environ.get("HY4_DUMP", "0") == "1":
                torch.set_printoptions(precision=4, linewidth=200)
                p("pp row0 [:16]", g_pp[0, :16].tolist())
                p("oracle pre/post row0", PRE[idx][0].tolist(), POST[idx][0].tolist())
                p(
                    "mix=0 pre/post",
                    (torch.sigmoid(hp["hc_base"][:4]) + 1e-6).tolist(),
                    (2 * torch.sigmoid(hp["hc_base"][4:])).tolist(),
                )
            g_h = d0(h).reshape(T, HID).float()
            ref_h = H[idx].reshape(T, HID)
            pre_o, post_o, comb_o = old.mixes(x_old)
            h_o = old.collapse_norm(
                x_old, pre_o, rep(md, nw.float().reshape(1, 1, 1, HID), dtype=ttnn.float32), NORM_EPS
            )
            ttnn.synchronize_device(md)
            o_pre, o_post = d0(pre_o).reshape(T, 4).float(), d0(post_o).reshape(T, 4).float()
            o_h = d0(h_o).reshape(T, HID).float()
            p(
                f"{side} T={T} vs oracle: pre {pcc(g_pre, PRE[idx]):.6f} (maxabs {(g_pre-PRE[idx]).abs().max():.2e})  post {pcc(g_post, POST[idx]):.6f} (maxabs {(g_post-POST[idx]).abs().max():.2e})  h {pcc(g_h, ref_h):.6f} (maxabs {(g_h-ref_h).abs().max():.3e}, ref rms {ref_h.pow(2).mean().sqrt():.3f})"
            )
            p(
                f"{side} T={T} vs old path: pre {pcc(g_pre, o_pre):.6f}  post {pcc(g_post, o_post):.6f}  h {pcc(g_h, o_h):.6f}   [old vs oracle: pre {pcc(o_pre, PRE[idx]):.6f} post {pcc(o_post, POST[idx]):.6f} h {pcc(o_h, ref_h):.6f}]"
            )
            if RM:
                g_rm = ttnn.to_torch(ttnn.get_device_tensors(h_tok)[0]).reshape(T, HID).float()
                p(f"{side} T={T} rm copy vs h tile: maxabs {(g_rm-g_h).abs().max():.3e}")
            if TIME:
                us_f = traced_us(md, lambda: fu.pre(x, w4, emit_rm=RM))
                us_o = traced_us(
                    md,
                    lambda: old.collapse_norm_rm(x_old, old.mixes(x_old)[0], w, NORM_EPS)
                    if RM
                    else old.collapse_norm(x_old, old.mixes(x_old)[0], w, NORM_EPS),
                )
                p(
                    f"{side} T={T} TRACED us: fused pre {us_f:.1f}   old (mixes + collapse_norm{'_rm' if RM else ''}) {us_o:.1f}"
                )
                res[f"{side}_T{T}"] = (us_f, us_o)
        del fu, old
finally:
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
p("RES", res)

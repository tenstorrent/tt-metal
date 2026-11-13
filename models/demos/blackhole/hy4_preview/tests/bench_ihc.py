# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""iHC cost per layer-pair path (4 calls: attn pre, attn post/expand, mlp pre, mlp post/expand), traced us, before/after, T = 1,2,4,8,16 token rows per chip.
BEFORE: Hy4IHC (DSV4 mHC kernels, identity comb): per sub-block mixes + collapse_norm_rm + expand, streams [T,1,4,D] fp32.
AFTER : Hy4IHCFused: per sub-block ONE program pre(x, y_prev, pp_prev) with the previous sub-block's expand folded in (in place), packed streams [1,1,4T,D].
Env: HY4_T=1,2,4,8,16"""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, p, rep, traced_us
from models.demos.blackhole.hy4_preview.tt.hy4_ihc import Hy4IHC
from models.demos.blackhole.hy4_preview.tt.hy4_ihc_fused import Hy4IHCFused

HID = 6144
TS = [int(v) for v in os.environ.get("HY4_T", "1,2,4,8,16").split(",")]
R = torch.load(ACTS + "ihc_l1.pt")
x_in = torch.cat([t.reshape(-1, *t.shape[2:]) for t in R["x4_in"]])
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=300_000_000)
res = {}
try:
    ha, hm = R["hc_attn"], R["hc_mlp"]
    kw = dict(dim=HID, hc_eps=R["cfg"]["hc_eps"], norm_eps=1e-5)
    olds = [Hy4IHC(md, h["hc_fn"], h["hc_base"], h["hc_scale"], **kw) for h in (ha, hm)]
    news = [Hy4IHCFused(md, h["hc_fn"], h["hc_base"], h["hc_scale"], **kw) for h in (ha, hm)]
    wn = [rep(md, w.float().reshape(1, 1, 1, HID), dtype=ttnn.float32) for w in (R["norm_in"], R["norm_post"])]
    for T in TS:
        idx = torch.randperm(x_in.shape[0], generator=torch.Generator().manual_seed(3 + T))[:T]
        xo = rep(md, x_in[idx].reshape(T, 1, 4, HID), dtype=ttnn.float32)
        xn = rep(md, x_in[idx].reshape(1, 1, 4 * T, HID), dtype=ttnn.float32)
        y = rep(md, (0.1 * torch.randn(1, 1, T, HID)).to(torch.bfloat16))
        w4 = [n.norm_weight(w, T) for n, w in zip(news, (R["norm_in"], R["norm_post"]))]
        # ---- BEFORE: per sub-block mixes -> collapse_norm_rm -> (attention / moe stand-in: y) -> expand
        pre, post, comb = olds[0].mixes(xo)
        olds[0].collapse_norm_rm(xo, pre, wn[0], 1e-5)  # warm-up (first call per T class outside the trace)
        olds[1].mixes(xo)
        olds[1].collapse_norm_rm(xo, pre, wn[1], 1e-5)
        olds[0].expand(y, xo, post, comb)

        def sub_old(i):
            pr, po, co = olds[i].mixes(xo)
            h, ht = olds[i].collapse_norm_rm(xo, pr, wn[i], 1e-5)
            return [h, ht, olds[i].expand(y, xo, po, co)]

        t_mix = traced_us(md, lambda: list(olds[0].mixes(xo)))
        pr, po, co = olds[0].mixes(xo)
        t_cn = traced_us(md, lambda: list(olds[0].collapse_norm_rm(xo, pr, wn[0], 1e-5)))
        t_ex = traced_us(md, lambda: olds[0].expand(y, xo, po, co))
        t_old_layer = traced_us(md, lambda: sub_old(0) + sub_old(1))
        # ---- AFTER: two fused programs per layer, each also doing the expand of the previous sub-block
        _, _, pp0 = news[0].pre(xn, w4[0])
        news[1].pre(xn, w4[1], y=y, pp_prev=pp0, x_out=xn)  # warm-up / compile

        def layer_new():
            _, _, ppa, _ = news[0].pre(xn, w4[0], y=y, pp_prev=pp0, x_out=xn)
            _, _, ppm, _ = news[1].pre(xn, w4[1], y=y, pp_prev=ppa, x_out=xn)
            return [ppa, ppm]

        t_pre = traced_us(md, lambda: list(news[0].pre(xn, w4[0])[:3]))
        t_pre_e = traced_us(md, lambda: list(news[0].pre(xn, w4[0], y=y, pp_prev=pp0, x_out=xn)[:3]))
        t_new_layer = traced_us(md, layer_new)
        p(
            f"T={T}: BEFORE per sub-block mixes {t_mix:.1f} + collapse_norm_rm {t_cn:.1f} + expand {t_ex:.1f} = {t_mix+t_cn+t_ex:.1f} us;  per layer (4 calls, one trace) {t_old_layer:.1f} us"
        )
        p(
            f"T={T}: AFTER  fused pre {t_pre:.1f} us, fused pre+expand {t_pre_e:.1f} us;  per layer (2 programs, one trace) {t_new_layer:.1f} us  => speedup {t_old_layer/t_new_layer:.2f}x"
        )
        res[T] = dict(
            mixes=t_mix, cn=t_cn, expand=t_ex, old_layer=t_old_layer, pre=t_pre, pre_e=t_pre_e, new_layer=t_new_layer
        )
finally:
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
p("RES", res)

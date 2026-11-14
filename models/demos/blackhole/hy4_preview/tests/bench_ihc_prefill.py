# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""iHC for a PREFILL chunk of 512 token rows per chip (per sub-block: pre = mixes + collapse + norm, and expand), traced us:
  old   : Hy4IHC (DSV4 mHC kernels, identity comb) in 32 chunks of T = 16 (its limit at D = 6144: Hy4ProjPlan max_cores)
  fused : Hy4IHCFused decode kernel in 32 chunks of T = 16 (pre + folded expand)
  comp  : Hy4IHCPrefill composite ttnn ops on [1,1,512,4D] (also checked vs the oracle on the 160 real dump_v1 tokens of layer 1)"""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import p, pcc, rep, traced_us
from models.demos.blackhole.hy4_preview.tt.hy4_ihc import Hy4IHC
from models.demos.blackhole.hy4_preview.tt.hy4_ihc_fused import Hy4IHCFused
from models.demos.blackhole.hy4_preview.tt.hy4_ihc_prefill import Hy4IHCPrefill

HID = 6144
NROWS = int(os.environ.get("HY4_ROWS", "512"))
R = torch.load("/mnt/tt-data/ssinghal/hy4-logs/program/ihc-fuse/ref_dump_v1_ihc.pt")
cfg = R["cfg"]
e = R["layers"][1]["attn"]
X4 = R["layers"][1]["x4"]  # [N, 4, D]
N = X4.shape[0]
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=600_000_000)
d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])
try:
    kw = dict(dim=HID, hc_eps=cfg["hc_eps"], norm_eps=cfg["norm_eps"])
    comp = Hy4IHCPrefill(md, e["hc_fn"], e["hc_base"], e["hc_scale"], **kw)
    w_t = rep(md, e["norm_w"].reshape(1, 1, 1, HID), dtype=ttnn.float32)
    # ---- correctness on the 160 real tokens
    x = rep(md, X4.reshape(1, 1, N, 4 * HID), dtype=ttnn.float32)
    h, pp = comp.pre(x, w_t)
    ttnn.synchronize_device(md)
    g_pp = d0(pp).reshape(N, 32).float()
    p(
        f"composite T={N} vs oracle (layer 1 attn): pre {pcc(g_pp[:, :4], e['pre']):.6f} post {pcc(g_pp[:, 4:8], e['post']):.6f} h {pcc(d0(h).reshape(N, HID).float(), e['h']):.6f}"
    )
    y = rep(md, (0.1 * torch.randn(1, 1, N, HID)).to(torch.bfloat16))
    xn = comp.expand(x, y, pp)
    ttnn.synchronize_device(md)
    # ---- timing at NROWS rows
    xs = rep(md, torch.randn(1, 1, NROWS, 4 * HID), dtype=ttnn.float32)
    ys = rep(md, (0.1 * torch.randn(1, 1, NROWS, HID)).to(torch.bfloat16))
    hh, ppp = comp.pre(xs, w_t)
    comp.expand(xs, ys, ppp)
    t_pre = traced_us(md, lambda: list(comp.pre(xs, w_t)), n=7, rep_=3)
    t_exp = traced_us(md, lambda: comp.expand(xs, ys, ppp), n=7, rep_=3)
    p(f"COMPOSITE {NROWS} rows: pre {t_pre:.0f} us + expand {t_exp:.0f} us = {t_pre+t_exp:.0f} us per sub-block")
    # ---- chunked decode-kernel paths (T = 16 chunks)
    C = NROWS // 16
    old = Hy4IHC(md, e["hc_fn"], e["hc_base"], e["hc_scale"], **kw)
    fu = Hy4IHCFused(md, e["hc_fn"], e["hc_base"], e["hc_scale"], **kw)
    xo = rep(md, torch.randn(16, 1, 4, HID), dtype=ttnn.float32)
    xn16 = rep(md, torch.randn(1, 1, 64, HID), dtype=ttnn.float32)
    y16 = rep(md, (0.1 * torch.randn(1, 1, 16, HID)).to(torch.bfloat16))
    w4 = fu.norm_weight(e["norm_w"], 16)
    pr, po, co = old.mixes(xo)
    old.collapse_norm_rm(xo, pr, w_t, cfg["norm_eps"])
    old.expand(y16, xo, po, co)
    _, _, pp16 = fu.pre(xn16, w4)
    fu.pre(xn16, w4, y=y16, pp_prev=pp16, x_out=xn16)

    def old_block():
        pr_, po_, co_ = old.mixes(xo)
        h_, ht_ = old.collapse_norm_rm(xo, pr_, w_t, cfg["norm_eps"])
        return [h_, ht_, old.expand(y16, xo, po_, co_)]

    t_old16 = traced_us(md, old_block, n=7, rep_=4)
    t_fu16 = traced_us(md, lambda: list(fu.pre(xn16, w4, y=y16, pp_prev=pp16, x_out=xn16)[:3]), n=7, rep_=4)
    p(
        f"CHUNKED T=16: old per chunk (mixes x2 + collapse_norm + expand) {t_old16:.0f} us -> x{C} = {t_old16*C:.0f} us;  fused pre+expand per chunk {t_fu16:.0f} us -> x{C} = {t_fu16*C:.0f} us"
    )
finally:
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

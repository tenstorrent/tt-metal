# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused iHC pre-kernel vs the fp32 oracle on the dump_v1 streams of layers 0..3 (real hc parameters of both sides, 8 prompts x 20 tokens; reference file from
tools/dump_ihc_v1.py): pre / post / h (collapse + rms_norm * w) PCC, T = 1,2,4,8,16, plus the fused expand (x_new vs x + post_dev * y, y random bf16) and the existing
identity-comb path (Hy4IHC) on the same inputs. Env: HY4_T=1,2,4,8,16 HY4_LAYERS=0,1,2,3"""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import p, pcc, rep
from models.demos.blackhole.hy4_preview.tt.hy4_ihc import Hy4IHC
from models.demos.blackhole.hy4_preview.tt.hy4_ihc_fused import Hy4IHCFused

HID = 6144
REF = "/mnt/tt-data/ssinghal/hy4-logs/program/ihc-fuse/ref_dump_v1_ihc.pt"
TS = [int(v) for v in os.environ.get("HY4_T", "1,2,4,8,16").split(",")]
LAYERS = [int(v) for v in os.environ.get("HY4_LAYERS", "0,1,2,3").split(",")]
R = torch.load(REF)
cfg = R["cfg"]
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384)
d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])
worst = {"pre": 1.0, "post": 1.0, "h": 1.0, "old_pre": 1.0, "old_post": 1.0, "old_h": 1.0}
try:
    for layer in LAYERS:
        ent = R["layers"][layer]
        X4 = ent["x4"]
        for side in ("attn", "mlp"):
            e = ent[side]
            fu = Hy4IHCFused(
                md, e["hc_fn"], e["hc_base"], e["hc_scale"], dim=HID, hc_eps=cfg["hc_eps"], norm_eps=cfg["norm_eps"]
            )
            old = Hy4IHC(
                md, e["hc_fn"], e["hc_base"], e["hc_scale"], dim=HID, hc_eps=cfg["hc_eps"], norm_eps=cfg["norm_eps"]
            )
            w_old = rep(md, e["norm_w"].reshape(1, 1, 1, HID), dtype=ttnn.float32)
            for T in TS:
                idx = torch.randperm(X4.shape[0], generator=torch.Generator().manual_seed(100 * layer + T))[:T]
                x = rep(md, X4[idx].reshape(1, 1, 4 * T, HID), dtype=ttnn.float32)
                x_old = rep(md, X4[idx].reshape(T, 1, 4, HID), dtype=ttnn.float32)
                w4 = fu.norm_weight(e["norm_w"], T)
                h, _, pp = fu.pre(x, w4)
                y = (0.1 * torch.randn(1, 1, T, HID, generator=torch.Generator().manual_seed(5))).to(torch.bfloat16)
                h2, _, pp2, xn = fu.pre(x, w4, y=rep(md, y), pp_prev=pp)
                po, qo, co = old.mixes(x_old)
                ho = old.collapse_norm(x_old, po, w_old, cfg["norm_eps"])
                ttnn.synchronize_device(md)
                g_pp = d0(pp).reshape(32, 32).float()
                pre_g, post_g = g_pp[:T, 0:4], g_pp[:T, 4:8]
                h_g = d0(h).reshape(T, HID).float()
                c = dict(pre=pcc(pre_g, e["pre"][idx]), post=pcc(post_g, e["post"][idx]), h=pcc(h_g, e["h"][idx]))
                o = dict(
                    old_pre=pcc(d0(po).reshape(T, 4).float(), e["pre"][idx]),
                    old_post=pcc(d0(qo).reshape(T, 4).float(), e["post"][idx]),
                    old_h=pcc(d0(ho).reshape(T, HID).float(), e["h"][idx]),
                )
                x_ref = X4[idx] + post_g.unsqueeze(-1) * y.float().reshape(T, 1, HID)
                x_g = d0(xn).reshape(T, 4, HID).float()
                h2_ok = pcc(
                    d0(h2).reshape(T, HID).float(), h_g
                )  # h of the second call (different input stream) only as a sanity value
                for k_, v_ in {**c, **o}.items():
                    worst[k_] = min(worst[k_], v_)
                p(
                    f"L{layer} {side} T={T:2d}: pre {c['pre']:.6f} (maxabs {(pre_g-e['pre'][idx]).abs().max():.1e}, std {e['pre'][idx].std():.1e}) post {c['post']:.6f} (maxabs {(post_g-e['post'][idx]).abs().max():.1e}) h {c['h']:.6f} | old path pre {o['old_pre']:.6f} post {o['old_post']:.6f} h {o['old_h']:.6f} | "
                    f"expand x_new vs x+post_dev*y: PCC {pcc(x_g, x_ref):.8f} maxabs {(x_g-x_ref).abs().max():.2e} rel L2 {float((x_g-x_ref).norm()/x_ref.norm()):.1e}"
                )
            del fu, old
finally:
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
p("WORST over layers/sides/T:", {k: round(v, 7) for k, v in worst.items()})

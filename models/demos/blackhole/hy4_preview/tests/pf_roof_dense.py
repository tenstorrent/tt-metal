# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""prefill-roof phase 2: dense layer-0 MLP chunk (T=512, TP8 bfp8) on a 1x8 slot, traced, variants fid (HiFi4 default | HiFi2 | LoFi), program configs (auto), PCC vs the oracle. Env: VARIANTS e.g. HiFi4,HiFi2,LoFi, CFG (0|1)."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, dev0, pcc, rep, traced_us
from models.demos.blackhole.hy4_preview.tt import hy4_dense as D

T = 512
acts = torch.load(ACTS + "acts_l0_l1.pt")
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=200_000_000)
try:
    w = torch.load(ACTS + "l0_dense.pt")
    t = lambda m: m.T.contiguous().to(torch.bfloat16)[None, None]
    w0, w1, w2 = t(w["mlp.gate_proj.weight"]), t(w["mlp.up_proj.weight"]), t(w["mlp.down_proj.weight"])
    hl, yl = max(acts["dense"], key=lambda hy: hy[0].shape[1])
    hl, yl = hl.reshape(-1, 6144)[:T], yl.reshape(-1, 6144)[:T]
    tx = rep(md, hl.to(torch.bfloat16).reshape(1, 1, T, 6144))
    for fid in [x for x in os.environ.get("VARIANTS", "HiFi4,HiFi2,LoFi").split(",") if x]:
        dense = D.make_dense(md, w0, w1, w2, decode=False, fid=fid)
        full = D.allreduce_cols(dense.forward(tx))
        ttnn.synchronize_device(md)
        got = dev0(full).reshape(T, 6144).float()
        r = {"fid": fid, "pcc": round(pcc(got, yl), 6)}
        r["partial_us"] = round(traced_us(md, lambda: dense.forward(tx), n=7, rep_=3), 1)
        r["with_ar_us"] = round(traced_us(md, lambda: D.allreduce_cols(dense.forward(tx)), n=7, rep_=3), 1)
        print("PF " + json.dumps(r), flush=True)
        if os.environ.get("TUNE") == "1":
            mk = lambda bw, h, w_, pm, pn: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(12, 8),
                in0_block_w=bw,
                out_subblock_h=h,
                out_subblock_w=w_,
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )
            for bw1, bw2, sh, sw in ((8, 8, 1, 4), (16, 12, 1, 4), (24, 24, 2, 2), (8, 8, 2, 2)):
                try:
                    dense.pc1 = mk(bw1, sh, sw, 2, 12)
                    dense.pc2 = mk(bw2, sh, sw, 2, 16)
                    full = D.allreduce_cols(dense.forward(tx))
                    ttnn.synchronize_device(md)
                    got = dev0(full).reshape(T, 6144).float()
                    r = {"fid": fid, "tuned": f"bw{bw1}/{bw2} sub{sh}x{sw}", "pcc": round(pcc(got, yl), 6)}
                    r["partial_us"] = round(traced_us(md, lambda: dense.forward(tx), n=7, rep_=3), 1)
                    print("PF " + json.dumps(r), flush=True)
                except Exception as e:
                    print(
                        "PF "
                        + json.dumps(
                            {"fid": fid, "tuned": f"bw{bw1}/{bw2} sub{sh}x{sw}", "err": str(e).split(chr(10))[0][:200]}
                        ),
                        flush=True,
                    )
        del dense
    if os.environ.get("DENSE_PF") == "1":  # integration check: make_dense(pf_opt=...) stock vs HY4_PF_OPT=all
        from models.demos.blackhole.hy4_preview.tt.hy4_pf_opt import PfOpt

        for v in ("off", "all"):
            dense = D.make_dense(md, w0, w1, w2, decode=False, pf_opt=PfOpt.parse("" if v == "off" else v))
            full = D.allreduce_cols(dense.forward(tx))
            ttnn.synchronize_device(md)
            got = dev0(full).reshape(T, 6144).float()
            print("PF " + json.dumps({"dense_pf": v, "pcc": round(pcc(got, yl), 6), "partial_us": round(traced_us(md, lambda: dense.forward(tx), n=7, rep_=3), 1),
                                      "with_ar_us": round(traced_us(md, lambda: D.allreduce_cols(dense.forward(tx)), n=7, rep_=3), 1)}), flush=True)
            del dense
    if os.environ.get("SHARED") == "1":  # layer-1 shared expert prefill chunk, stock vs HY4_PF_OPT (mm2d, mm2d+hifi2)
        from models.demos.blackhole.hy4_preview.tt.hy4_pf_opt import PfOpt

        g = torch.load(ACTS + "l1_gate_shared.pt")
        s0, s1, s2 = (t(g["mlp.shared_experts.gate_proj.weight"]), t(g["mlp.shared_experts.up_proj.weight"]), t(g["mlp.shared_experts.down_proj.weight"]))
        X = torch.cat([hh.reshape(-1, 6144) for hh, y, r_ in acts["moe"]])[:T]
        xs = rep(md, X.to(torch.bfloat16).reshape(1, 1, T, 6144))
        gg = (X.float() @ g["mlp.shared_experts.gate_proj.weight"].T).clamp(max=10.0)
        uu = (X.float() @ g["mlp.shared_experts.up_proj.weight"].T).clamp(-10, 10)
        ref = (torch.nn.functional.silu(gg) * uu) @ g["mlp.shared_experts.down_proj.weight"].T
        for v in ("off", "mm2d", "mm2d,hifi2", "hifi2"):
            sh = D.make_shared(md, s0, s1, s2, decode=False, pf_opt=PfOpt.parse("" if v == "off" else v))
            full = D.allreduce_cols(sh.forward(xs))
            ttnn.synchronize_device(md)
            got = dev0(full).reshape(T, 6144).float()
            print("PF " + json.dumps({"shared": v, "pcc": round(pcc(got, ref), 6), "partial_us": round(traced_us(md, lambda: sh.forward(xs), n=7, rep_=3), 1),
                                      "with_ar_us": round(traced_us(md, lambda: D.allreduce_cols(sh.forward(xs)), n=7, rep_=3), 1)}), flush=True)
            del sh
finally:
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

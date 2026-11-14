# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""skinny-kernels X25: Hy4 router, bank-local logits+sigmoid+bias kernel (tt/hy4_skinny.SkinnyRouter) + router_select vs the stock Hy4Gate (fused), 1x8 slot.
Real layer-1 activations; top-8 set-match vs the CPU bf16 emulation over all real tokens (gate >= 99.3% = the stock HiFi4 value), PCC of score / rank vs stock, traced us.
Env: SK_T=1,4 (tokens per call), SK_NTOK (default all), SK_VARIANTS (name:k=v;...), OUT."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, p, pcc, traced_us
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer
from models.demos.blackhole.hy4_preview.tt.hy4_router import Hy4Gate, route_cpu, set_match
from models.demos.blackhole.hy4_preview.tt.hy4_skinny import SkinnyRouter

HID = 6144
OUT = os.environ.get("OUT", "/tmp/skinny_router.json")
TS = [int(t) for t in os.environ.get("SK_T", "1,4").split(",")]
VARS = os.environ.get("SK_VARIANTS", "base").split("|")


def parse(v):
    name, _, kv = v.partition(":")
    d = {}
    for item in filter(None, kv.split(";")):
        k, val = item.split("=")
        d[k] = val if k == "remap" else int(val)
    return name, d


acts = torch.load(ACTS + "acts_l0_l1.pt")
gs = torch.load(ACTS + "l1_gate_shared.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]]).to(torch.bfloat16)
NTOK = int(os.environ.get("SK_NTOK", X.shape[0]))
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
res = {}
try:
    W = load_moe_layer(1, load_experts=False)
    stock = Hy4Gate(md, W["gate_weight"], W["gate_bias"], mode="fused")
    sks = {}
    for v in VARS:
        name, kw = parse(v)
        l1out = kw.pop("l1out", 0)
        sks[name] = SkinnyRouter(md, W["gate_weight"], W["gate_bias"], **kw)
        if l1out:  # dispatch-like input formats: weights height-sharded L1 [1, k] shards, indices L1 interleaved
            from models.demos.blackhole.deepseek_v41_flash.tt.router_mm import scores_shard_config

            sks[name].out_mcs = (scores_shard_config(max(TS), 8), ttnn.L1_MEMORY_CONFIG)
    _, ids_ref, _ = route_cpu(X[:NTOK], gs["mlp.gate.weight"], gs["mlp.gate.e_score_correction_bias"], emulate_device=True)
    if os.environ.get("SK_DBG"):
        T = TS[0]
        xs = X[:T]
        t0 = ttnn.from_torch(xs.reshape(1, 1, T, HID), device=md, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(md))
        ref_s = torch.sigmoid(xs.float() @ gs["mlp.gate.weight"].to(torch.bfloat16).float().T)
        for name, sk in sks.items():
            rank, score = sk.scores(t0, T)
            ttnn.synchronize_device(md)
            sc = ttnn.to_torch(ttnn.get_device_tensors(score)[0]).float().reshape(-1, 256)[:T]
            rk = ttnn.to_torch(ttnn.get_device_tensors(rank)[0]).float().reshape(-1, 256)[:T]
            p("DBG score row0 first8", sc[0, :8].tolist(), "ref", ref_s[0, :8].tolist())
            p("DBG rank row0 first8", rk[0, :8].tolist())
            for b in range(8):
                p(f"DBG tile {b}: PCC {pcc(sc[:, 32*b:32*b+32], ref_s[:, 32*b:32*b+32]):.5f} maxabs {(sc[:, 32*b:32*b+32]-ref_s[:, 32*b:32*b+32]).abs().max():.4f}")
            ref_r = ref_s + gs["mlp.gate.e_score_correction_bias"].float()
            for b in range(8):
                d = (rk[:, 32*b:32*b+32] - ref_r[:, 32*b:32*b+32]).abs()
                p(f"DBG rank tile {b}: maxabs {d.max():.6f} {[int(v) for v in d.argmax().reshape(1)]}")
            bb = gs["mlp.gate.e_score_correction_bias"].float()
            p("DBG bias[:8]", bb[:8].tolist(), "rank-score row0[:8]", (rk[0, :8] - sc[0, :8]).tolist(), "row3", (rk[3, :8] - sc[3, :8]).tolist())
            p("DBG rank-score vs bias: maxabs", (rk - sc - bb).abs().max().item(), "vs bias permuted?", [(b, (rk[0, b*32:(b+1)*32] - sc[0, b*32:(b+1)*32] - bb[b*32:(b+1)*32]).abs().max().item()) for b in range(8)])
            ti = rk.topk(8, dim=-1)[1].sort(-1)[0]; tr = ref_r.topk(8, dim=-1)[1].sort(-1)[0]
            p("DBG topk set-match from device rank tile (host topk)", (ti == tr).all(-1).float().mean().item())
            for r in range(T):
                p(f"DBG row {r}: PCC {pcc(sc[r], ref_s[r]):.5f}")
        if os.environ.get("SK_DBG") == "2":
            Wg = gs["mlp.gate.weight"].to(torch.bfloat16).float()  # [256, 6144]
            for kt in (0, 1, 7, 8, 47, 48, 95, 96, 100, 143, 144, 191):
                xs = torch.zeros(T, HID)
                for r in range(T):
                    xs[r, 32 * kt : 32 * kt + 32] = 0.01 * (r + 1)
                xs = xs.to(torch.bfloat16)
                t1 = ttnn.from_torch(xs.reshape(1, 1, T, HID), device=md, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(md))
                for name, sk in sks.items():
                    rank, score = sk.scores(t1, T)
                    ttnn.synchronize_device(md)
                    sc = ttnn.to_torch(ttnn.get_device_tensors(score)[0]).float().reshape(-1, 256)[:T].clamp(1e-6, 1 - 1e-6)
                    lg = torch.log(sc / (1 - sc))
                    ref = xs.float() @ Wg.T
                    p(f"DBG1 kt={kt}: PCC {pcc(lg, ref):.4f}  got[0,:4] {lg[0,:4].tolist()} ref {ref[0,:4].tolist()}")
        raise SystemExit
    for T in TS:
        n = (NTOK // T) * T

        def run(f):
            got_i, got_w = [], []
            for s in range(0, n, T):
                t = ttnn.from_torch(
                    X[s : s + T].reshape(1, 1, T, HID), device=md, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(md),
                )
                w_, i_ = f(t)
                got_i.append(ttnn.to_torch(ttnn.get_device_tensors(i_)[0]).reshape(-1, 8).long())
                got_w.append(ttnn.to_torch(ttnn.get_device_tensors(w_)[0]).reshape(-1, 8).float())
                for q in (w_, i_, t):
                    ttnn.deallocate(q)
            return torch.cat(got_i), torch.cat(got_w)

        si, sw = run(lambda t: stock.forward(t))
        t0 = ttnn.from_torch(
            X[:T].reshape(1, 1, T, HID), device=md, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        us_s = traced_us(md, lambda: stock.forward(t0))
        sm_s = set_match(ids_ref[:n], si)
        p(f"T={T} stock   {us_s:6.1f} us  set-match vs bf16 emulation ({n} tok) {sm_s:.4f}")
        res[f"T{T}_stock"] = {"us": us_s, "set_match": sm_s}
        for name, sk in sks.items():
            try:
                ki, kw_ = run(lambda t: sk.forward(t, T))
                sm = set_match(ids_ref[:n], ki)
                sm_vs = set_match(si, ki)
                wp = pcc(kw_, sw)
                us = traced_us(md, lambda: sk.forward(t0, T))
                us_sc = traced_us(md, lambda: sk.scores(t0, T))
                r = {"us": us, "us_scores_only": us_sc, "set_match": sm, "set_match_vs_stock": sm_vs, "w_pcc_vs_stock": wp}
                res[f"T{T}_{name}"] = r
                p(f"T={T} skinny {name:6s} {us:6.1f} us (scores kernel {us_sc:.1f})  set-match {sm:.4f} (vs stock {sm_vs:.4f})  weights PCC vs stock {wp:.6f}")
            except Exception as e:
                res[f"T{T}_{name}"] = {"err": str(e)[:300]}
                p(f"T={T} skinny {name} ERROR {str(e)[:300]}")
finally:
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    json.dump(res, open(OUT, "w"), indent=1)

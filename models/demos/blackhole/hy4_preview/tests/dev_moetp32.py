# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""moe-tp32 X40: TP32 routed-expert decode kernel on a 1x8 slot (chips = ranks chip0 .. chip0 + 7 of the 32; the kernel is per chip, no CCL).
Real layer-1 experts from the bf16 checkpoint (HY4_WEIGHT_CACHE=0), real layer-1 activations and routes.  Reference = torch emulation of the kernel dataflow on the SAME bfp4-rounded tiles
(chip-summed over the 8 chips): PCC must be ~0.9999+.  Env: TP_T=1,2,4,8, TP_REPS=5, TP_TIME=1, TP_REF=1, OUT=json, TP_CHIP0=0."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, p, pcc, traced_us
from models.demos.blackhole.hy4_preview.tt import hy4_moe_tp32_layout as L
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32 import Tp32Experts, route_tensor
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer

HID = 6144
TS = [int(t) for t in os.environ.get("TP_T", "1").split(",")]
REPS = int(os.environ.get("TP_REPS", "5"))
OUT = os.environ.get("OUT", "/tmp/moetp32.json")
CHIP0 = int(os.environ.get("TP_CHIP0", "0"))

acts = torch.load(ACTS + "acts_l0_l1.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]])
R = torch.cat([r.reshape(-1, 8) for h, y, r in acts["moe"]]).long()


def q_roundtrip(t):
    """bfp4 round trip of a tile stream [N, 32, 32] (host, bit-exact with the device tiles)"""
    n = t.shape[0]
    h = ttnn.from_torch(t.reshape(1, 1, n * 32, 32).to(torch.bfloat16), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(h).reshape(n, 32, 32).float()


md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
res = {"v": {}}
try:
    W = load_moe_layer(1, load_experts=True)
    w0, w1, w2 = [W[k].reshape(W[k].shape[-3], W[k].shape[-2], W[k].shape[-1]) for k in ("w0", "w1", "w2")]
    p("weights", tuple(w0.shape), tuple(w2.shape))
    tp = Tp32Experts(md, w0, w1, w2, chip0=CHIP0)
    p(
        "cores",
        [(c.x, c.y) for c in tp.cores],
        "cb bytes T=4",
        Tp32Experts.cb_bytes(4),
        "sum",
        sum(Tp32Experts.cb_bytes(4).values()),
    )
    ref_ok = os.environ.get("TP_REF", "1") == "1"
    if ref_ok:
        q01 = [torch.stack([q_roundtrip(tp.t01[k, b]) for b in range(8)]) for k in range(8)]
        q2 = [torch.stack([q_roundtrip(tp.t2[k, b]) for b in range(8)]) for k in range(8)]
        p("reference tiles quantised")
    comp = ttnn.ConcatMeshToTensor(md, dim=0)
    for T in TS:
        for rep in range(REPS):
            g = torch.Generator().manual_seed(100 * T + rep)
            sel = torch.randperm(X.shape[0], generator=g)[:T]
            x = X[sel].to(torch.bfloat16)
            ids = R[sel]
            sc = (torch.rand(T, 8, generator=g) * 0.4 + 0.05).to(torch.bfloat16)
            tt_x = ttnn.from_torch(
                x.reshape(1, 1, T, HID),
                device=md,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            )
            rt = route_tensor(md, ids, sc)
            o = tp(tt_x, rt, T)
            ttnn.synchronize_device(md)
            out = ttnn.to_torch(o, mesh_composer=comp).float()  # [8, 1, T, HID]
            dev = out[:, 0, :T].sum(0)
            line = {"distinct": len(set(ids.flatten().tolist()))}
            if ref_ok:
                refv = sum(L.emulate_chip(x, ids, sc, q01[k], q2[k]) for k in range(8))
                line["pcc_emul"] = pcc(dev, refv)
                line["per_chip_pcc_min"] = min(
                    pcc(out[k, 0, :T], L.emulate_chip(x, ids, sc, q01[k], q2[k])) for k in range(2)
                )
            dbgm = int(os.environ.get("HY4_TP32_DBG", "0"))
            if dbgm and ref_ok:
                uniq = sorted(set(ids.flatten().tolist()))
                xt = x.float().reshape(T, 192, 32)[0]
                k = 0
                parts = torch.zeros(8, len(uniq), 4, 32)
                for b in range(8):
                    for ai, e in enumerate(uniq):
                        for kl in range(24):
                            for j in range(4):
                                parts[b, ai, j] += xt[24 * b + kl] @ q01[k][b][(e * 24 + kl) * 4 + j]
                if dbgm == 1:
                    b0, a0 = 0, 0
                    e0 = uniq[a0]
                    cont = torch.stack(
                        [
                            torch.stack([xt[24 * b0 + kl] @ q01[k][b0][(e0 * 24 + kl) * 4 + j] for j in range(4)])
                            for kl in range(24)
                        ]
                    )  # [24, 4, 32]
                    dv0 = torch.stack(
                        [out[k, 0, 0, 32 * (24 * b0 + 4 * a0 + j) : 32 * (24 * b0 + 4 * a0 + j) + 32] for j in range(4)]
                    )
                    coef = torch.linalg.lstsq(cont.reshape(24, -1).T, dv0.reshape(-1, 1)).solution.flatten()
                    p("per-K-tile coef (bank0 expert0)", [round(float(c), 2) for c in coef])
                    # j-permutation test: which dev tile matches which ref tile
                    p(
                        "tile corr matrix dev j x ref j",
                        [[round(pcc(dv0[a], cont.sum(0)[b]), 2) for b in range(4)] for a in range(4)],
                    )
                    for b in (0, 3):
                        for ai in range(min(6, len(uniq))):
                            dv = torch.stack(
                                [
                                    out[k, 0, 0, 32 * (24 * b + 4 * ai + j) : 32 * (24 * b + 4 * ai + j) + 32]
                                    for j in range(4)
                                ]
                            )
                            p(
                                f"P bank{b} expert#{ai}",
                                "pcc",
                                round(pcc(dv, parts[b, ai]), 4),
                                "ratio",
                                round(float(dv.norm() / parts[b, ai].norm()), 3),
                            )
                else:
                    gu = parts.sum(0)
                    for ai in range(min(12, len(uniq))):
                        e = uniq[ai]
                        s_e = float(sc[ids == e].sum())
                        for c in range(2):
                            g, u = gu[ai, c], gu[ai, 2 + c]
                            ref_a = torch.nn.functional.silu(g.clamp(max=10.0)) * u.clamp(-10, 10) * s_e
                            for b in (0, 5):
                                pg = 24 * b + 2 * ai + c
                                dv = out[k, 0, 0, 32 * pg : 32 * pg + 32]
                                p(
                                    f"ACT bank{b} expert#{ai} c{c}",
                                    "pcc",
                                    round(pcc(dv, ref_a), 4),
                                    "ratio",
                                    round(float(dv.norm() / ref_a.norm()), 3),
                                )
                break
            if os.environ.get("TP_DBG") and ref_ok:
                r0 = L.emulate_chip(x, ids, sc, q01[0], q2[0])
                d0 = out[0, 0, :T]
                p("chip0 norm ratio dev/ref", float(d0.norm() / r0.norm()))
                p(
                    "per-bank pcc",
                    [round(pcc(d0[:, 768 * b : 768 * b + 768], r0[:, 768 * b : 768 * b + 768]), 3) for b in range(8)],
                )
                p(
                    "per-tile pcc bank0",
                    [round(pcc(d0[:, 32 * i : 32 * i + 32], r0[:, 32 * i : 32 * i + 32]), 2) for i in range(24)],
                )
                uniq = sorted(set(ids.flatten().tolist()))
                cols = []
                for e in uniq:
                    sc_e = sc.clone()
                    sc_e[ids != e] = 0
                    cols.append(L.emulate_chip(x, ids, sc_e, q01[0], q2[0])[0])
                A = torch.stack(cols, 1)
                coef = torch.linalg.lstsq(A, d0[0].unsqueeze(1)).solution.flatten()
                p(
                    "lstsq coef per active expert (ascending id)",
                    uniq,
                    [round(float(c), 3) for c in coef],
                    "scores",
                    [float(sc[ids == e].sum()) for e in uniq],
                )
                p("dev[0,:8]", d0[0, :8].tolist(), "ref", r0[0, :8].tolist())
                p("dev[0,32:40]", d0[0, 32:40].tolist(), "ref", r0[0, 32:40].tolist())
            if dbgm:
                continue
            p(f"T={T} rep={rep}", json.dumps(line))
            res["v"][f"T{T}_r{rep}"] = line
            ttnn.deallocate(o)
        if os.environ.get("TP_TIME", "1") == "1":
            us = traced_us(md, lambda: tp(tt_x, rt, T))
            p(f"T={T} traced {us:.1f} us (distinct {line['distinct']})")
            res["v"][f"T{T}_us"] = us
finally:
    json.dump(res, open(OUT, "w"), indent=1)
    ttnn.close_mesh_device(md)

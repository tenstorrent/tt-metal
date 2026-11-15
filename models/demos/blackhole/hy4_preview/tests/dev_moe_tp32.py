# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""kernel-track: TP32 MoE block (tt/hy4_moe_tp32_block) vs the fp32 oracle and vs the EP ring block, on the 4x8 mesh with REAL layer weights.
Per T (tokens per mesh row, B = 4 T <= 8): NREP repeated calls with different tokens; router set-match vs the fp32 oracle, routed+shared PCC vs the oracle MoE
output (acts of layers 0/1: tools/dump_lmoe_acts), PCC vs the EP block on the same input, vs a bfp4-expert CPU reference, traced us (TP32 block vs EP block).
Env: HY4_T=1,2  HY4_LAYER=1  NREP=5  EP=1 (also build the EP block for the A/B)  OUT=json path."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, close_mesh, open_mesh, p, pcc
from models.demos.blackhole.hy4_preview.tt import hy4_dense as D
from models.demos.blackhole.hy4_preview.tt.hy4_moe import Hy4MoEBlock
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32 import Tp32Experts
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32_block import Hy4MoeTp32Block
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer
from models.demos.blackhole.hy4_preview.tt.hy4_router import Hy4Gate, route_cpu, set_match
from models.demos.gpt_oss.tt.ccl import CCLManager

LAYER = int(os.environ.get("HY4_LAYER", "1"))
TS = [int(t) for t in os.environ.get("HY4_T", "1,2").split(",")]
NREP = int(os.environ.get("NREP", "5"))
HID = 6144


def traced_us(md, fn, n=15, rep_=10):
    def body():
        for _ in range(rep_):
            o = fn()
            if isinstance(o, ttnn.Tensor):
                ttnn.deallocate(o)

    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    body()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / rep_ * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts)


def quant(m):  # [in,out] bf16 -> bfp4 round trip
    return (
        ttnn.to_torch(ttnn.from_torch(m.reshape(1, 1, *m.shape), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT))
        .reshape(m.shape)
        .float()
    )


def main():
    acts = torch.load(ACTS + "acts_l0_l1.pt")
    gs = torch.load(ACTS + "l1_gate_shared.pt")
    X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]])
    Y = torch.cat([y.reshape(-1, HID) for h, y, r in acts["moe"]])
    md = open_mesh(trace=300_000_000)
    rows, cols = tuple(md.shape)
    res = {}
    try:
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        t0 = time.time()
        Wc = load_moe_layer(LAYER)  # warm cache: w0 None
        p(f"load_moe_layer({LAYER}) {time.time()-t0:.1f}s warm_cache={Wc['w0'] is None}")
        t0 = time.time()
        Wf = load_moe_layer(LAYER, experts=list(range(256)))
        p(f"full expert weights {time.time()-t0:.1f}s")
        t0 = time.time()
        tpe = Tp32Experts(md, Wf["w0"][0], Wf["w1"][0], Wf["w2"][0])
        res["tp32_build_s"] = round(time.time() - t0, 1)
        p(f"Tp32Experts pack+upload (32 chips, layer {LAYER}) {res['tp32_build_s']}s")
        shared_w = [Wc[k][256] for k in ("shared_w0", "shared_w1", "shared_w2")]
        gate = Hy4Gate(md, Wc["gate_weight"], Wc["gate_bias"])
        ep = None
        for T in TS:
            B = T * rows
            blk = Hy4MoeTp32Block(md, ccl, gate, tpe, T)
            if os.environ.get("EP", "1") == "1":
                ep = Hy4MoEBlock(md, Wc, batch_per_device=T) if ep is None else ep.for_batch(T)
                if T == TS[0]:
                    ep.warmup()
            shared = D.make_shared(md, *shared_w, decode=True)
            comp = ttnn.ConcatMesh2dToTensor(md, dims=(2, 3), mesh_shape=(rows, cols))
            r = res.setdefault(f"T{T}", dict(pcc_oracle=[], pcc_ep=[], pcc_bfp4=[], setmatch=[]))
            for rep in range(NREP):
                idx = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(100 * T + rep))[:B]
                x = X[idx].to(torch.bfloat16)
                tt_tok = ttnn.from_torch(
                    x.reshape(B, 1, 1, HID),
                    device=md,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
                )
                tt_gate = ttnn.from_torch(
                    x.reshape(1, 1, B, HID),
                    device=md,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols)),
                )
                part = shared.forward(tt_gate)
                out = blk.forward(tt_gate, tt_tok, fold=part)
                ttnn.synchronize_device(md)
                total = (
                    ttnn.to_torch(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG), mesh_composer=comp)
                    .reshape(B, HID)
                    .float()
                )
                w_dev, i_dev = blk.last_routing
                ix = torch.cat(
                    [
                        ttnn.to_torch(ttnn.get_device_tensors(i_dev)[r_ * cols]).reshape(-1, 8).long()
                        for r_ in range(rows)
                    ]
                )
                sc = torch.cat(
                    [
                        ttnn.to_torch(ttnn.get_device_tensors(w_dev)[r_ * cols]).reshape(-1, 8).float()
                        for r_ in range(rows)
                    ]
                )
                _, i_o, _ = route_cpu(X[idx], gs["mlp.gate.weight"], gs["mlp.gate.e_score_correction_bias"])
                sm = set_match(i_o, ix)
                ref_b = torch.zeros(B, HID)
                for t in range(B):
                    for j in range(8):
                        e = int(ix[t, j])
                        w0, w1, w2 = (quant(Wf[k][0, e]) for k in ("w0", "w1", "w2"))
                        g = (x[t].float() @ w0).clamp(max=10.0)
                        u = (x[t].float() @ w1).clamp(-10.0, 10.0)
                        ref_b[t] += sc[t, j] * ((torch.nn.functional.silu(g) * u) @ w2)
                sh_ref = (
                    torch.nn.functional.silu((x.float() @ gs["mlp.shared_experts.gate_proj.weight"].T).clamp(max=10.0))
                    * (x.float() @ gs["mlp.shared_experts.up_proj.weight"].T).clamp(-10, 10)
                ) @ gs["mlp.shared_experts.down_proj.weight"].T
                pc_o, pc_b = pcc(total, Y[idx]), pcc(total, ref_b + sh_ref)
                pc_e = None
                if ep is not None:
                    o2 = ep.forward(tt_gate, tt_tok, fold=part)
                    ttnn.synchronize_device(md)
                    t2 = (
                        ttnn.to_torch(ttnn.to_memory_config(o2, ttnn.DRAM_MEMORY_CONFIG), mesh_composer=comp)
                        .reshape(B, HID)
                        .float()
                    )
                    pc_e = pcc(total, t2)
                    pc_eo = pcc(t2, Y[idx])
                    ttnn.deallocate(o2)
                    r.setdefault("pcc_ep_vs_oracle", []).append(pc_eo)
                p(
                    f"T={T} rep {rep}: set-match {sm:.3f}  PCC vs oracle {pc_o:.5f}  vs bfp4-ref {pc_b:.5f}  vs EP block {pc_e}"
                )
                r["setmatch"].append(sm), r["pcc_oracle"].append(pc_o), r["pcc_bfp4"].append(pc_b), r["pcc_ep"].append(
                    pc_e
                )
                ttnn.deallocate(out)
                ttnn.deallocate(part)
            r["min"] = dict(setmatch=min(r["setmatch"]), pcc_oracle=min(r["pcc_oracle"]), pcc_bfp4=min(r["pcc_bfp4"]))
            if os.environ.get("TIME", "1") == "1":
                fold = shared.forward(tt_gate)
                r["us_tp32_block_with_shared_fold"] = round(
                    traced_us(md, lambda: blk.forward(tt_gate, tt_tok, fold=fold)), 1
                )
                r["us_tp32_block_routed_only"] = round(traced_us(md, lambda: blk.forward(tt_gate, tt_tok)), 1)
                if ep is not None:
                    r["us_ep_block_with_shared_fold"] = round(
                        traced_us(md, lambda: ep.forward(tt_gate, tt_tok, fold=fold)), 1
                    )
                    r["us_ep_block_routed_only"] = round(traced_us(md, lambda: ep.forward(tt_gate, tt_tok)), 1)
                r["us_router"] = round(traced_us(md, lambda: gate.forward(tt_gate)[0]), 1)
                w_, i_ = gate.forward(tt_gate)
                route = blk._pack_route(w_, i_)
                route_all = blk._ag_rows(route, 3)
                x_all = ttnn.to_layout(ttnn.reshape(blk._ag_rows(tt_tok, 0), [1, 1, B, HID]), ttnn.TILE_LAYOUT)
                part_k = tpe(x_all, route_all, B)
                y1 = blk._rs(part_k, 3, 1, ttnn.Topology.Ring)
                y2 = blk._rs(y1, 3, 0, ttnn.Topology.Linear)
                y3 = blk._ag_rows(y2, 3)
                bd = {}
                bd["pack_route"] = traced_us(md, lambda: blk._pack_route(w_, i_))
                bd["ag_route"] = traced_us(md, lambda: blk._ag_rows(route, 3))
                bd["ag_x_rm+tile"] = traced_us(
                    md, lambda: ttnn.to_layout(ttnn.reshape(blk._ag_rows(tt_tok, 0), [1, 1, B, HID]), ttnn.TILE_LAYOUT)
                )
                bd["tp32_kernel"] = traced_us(md, lambda: tpe(x_all, route_all, B))
                bd["fold_matmul_add"] = traced_us(md, lambda: ttnn.add(part_k, ttnn.matmul(blk.mask, fold)))
                bd["rs_cols"] = traced_us(md, lambda: blk._rs(part_k, 3, 1, ttnn.Topology.Ring))
                bd["rs_rows_tile"] = traced_us(md, lambda: blk._rs(y1, 3, 0, ttnn.Topology.Linear))
                bd["ag_rows_tile"] = traced_us(md, lambda: blk._ag_rows(y2, 3))
                bd["select_matmul"] = traced_us(md, lambda: ttnn.matmul(blk.sel, y3, compute_kernel_config=blk.ckc))
                r["breakdown_us"] = {k: round(v, 1) for k, v in bd.items()}
                p(f"T={T} breakdown us: {json.dumps(r['breakdown_us'])}")
                p(f"T={T} traced us: {json.dumps({k: v for k, v in r.items() if k.startswith('us_')})}")
            del shared
        p("SUMMARY " + json.dumps(res))
        with open(os.environ.get("OUT_JSON", "/out/res_dev_moe_tp32.json"), "w") as f:
            json.dump(res, f)
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()

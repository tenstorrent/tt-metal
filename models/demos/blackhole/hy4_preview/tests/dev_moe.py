# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S1: layer-1 MoE decode on REAL activations vs the fp32 oracle (acts dumped by tools/dump_lmoe_acts.py).
Per tokens-per-device T (1 = B=4, 2 = B=8, 4 = 16 tokens): router set-match vs the fp32 oracle ids AND vs the CPU emulation on the same bf16 input,
routed output PCC, routed + shared (TP fold) PCC vs the oracle MoE output, PCC vs a CPU reference with bfp4-quantised experts and the DEVICE routing
(separates quantisation from bugs), traced latency. Env: HY4_T=1,2,4 HY4_LAYER=1 HY4_ROUTER=fused|exact."""
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, close_mesh, open_mesh, p, pcc
from models.demos.blackhole.hy4_preview.tt import hy4_dense as D
from models.demos.blackhole.hy4_preview.tt.hy4_moe import Hy4MoEBlock
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer
from models.demos.blackhole.hy4_preview.tt.hy4_router import route_cpu, set_match

LAYER = int(os.environ.get("HY4_LAYER", "1"))
TS = [int(t) for t in os.environ.get("HY4_T", "1,2,4").split(",")]
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


def expert_ref(x, W, e, quantised):
    w0, w1, w2 = (W["w0"][0, e], W["w1"][0, e], W["w2"][0, e])
    if quantised:
        w0, w1, w2 = quant(w0), quant(w1), quant(w2)
    g = (x.float() @ w0.float()).clamp(max=10.0)
    u = (x.float() @ w1.float()).clamp(-10.0, 10.0)
    return (torch.nn.functional.silu(g) * u) @ w2.float()


acts = torch.load(ACTS + "acts_l0_l1.pt")
gs = torch.load(ACTS + "l1_gate_shared.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]])
Y = torch.cat([y.reshape(-1, HID) for h, y, r in acts["moe"]])
R = torch.cat([r.reshape(-1, 8) for h, y, r in acts["moe"]])
md = open_mesh()
rows, cols = tuple(md.shape)
try:
    t0 = time.time()
    W = load_moe_layer(LAYER)
    p(f"load_moe_layer({LAYER}) {time.time()-t0:.1f}s warm_cache={W['w0'] is None}")
    shared_w = [W[k][256] for k in ("shared_w0", "shared_w1", "shared_w2")]
    blk = None
    for T in TS:
        t0 = time.time()
        blk = Hy4MoEBlock(md, W, batch_per_device=T) if blk is None else blk.for_batch(T)
        if T == TS[0]:
            blk.warmup()
        p(f"block T={T} ready {time.time()-t0:.1f}s")
        shared = D.make_shared(md, *shared_w, decode=True)
        B = T * rows
        idx = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(100 + T))[:B]
        x = X[idx].to(torch.bfloat16)  # row r gets tokens [r*T, (r+1)*T)
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
        comp = ttnn.ConcatMesh2dToTensor(md, dims=(2, 3), mesh_shape=(rows, cols))

        def run(fold):
            part = shared.forward(tt_gate) if fold else None
            out = blk.forward(tt_gate, tt_tok, fold=part)
            if part is not None:
                ttnn.deallocate(part)
            return out

        # routed only
        out_r = run(False)
        ttnn.synchronize_device(md)
        routed = (
            ttnn.to_torch(ttnn.to_memory_config(out_r, ttnn.DRAM_MEMORY_CONFIG), mesh_composer=comp)
            .reshape(B, HID)
            .float()
        )
        sc, ix = blk.last_routing
        devs_ix, devs_sc = ttnn.get_device_tensors(ix), ttnn.get_device_tensors(sc)
        ix_dev = torch.cat([ttnn.to_torch(devs_ix[r * cols]).reshape(-1, 8).long() for r in range(rows)])
        sc_dev = torch.cat([ttnn.to_torch(devs_sc[r * cols]).reshape(-1, 8).float() for r in range(rows)])
        # routed + shared (fold)
        out_f = run(True)
        ttnn.synchronize_device(md)
        total = (
            ttnn.to_torch(ttnn.to_memory_config(out_f, ttnn.DRAM_MEMORY_CONFIG), mesh_composer=comp)
            .reshape(B, HID)
            .float()
        )
        # references
        w_o, i_o, _ = route_cpu(X[idx], gs["mlp.gate.weight"], gs["mlp.gate.e_score_correction_bias"])
        w_e, i_e, _ = route_cpu(x, gs["mlp.gate.weight"], gs["mlp.gate.e_score_correction_bias"], emulate_device=True)
        sm_or, sm_em = set_match(i_o, ix_dev), set_match(i_e, ix_dev)
        wmax = float(((sc_dev.gather(1, ix_dev.argsort(-1)) if False else 0) or 0))
        used = sorted(set(ix_dev.flatten().tolist()) | set(R[idx].flatten().tolist()))
        Wh = load_moe_layer(LAYER, experts=used, load_experts=True) if W["w0"] is None else W

        def moe_ref(ids, wts, quantised):
            o = torch.zeros(B, HID)
            for t in range(B):
                for j in range(8):
                    o[t] += wts[t, j] * expert_ref(x[t], Wh, int(ids[t, j]), quantised)
            return o

        sh_ref = (
            torch.nn.functional.silu((x.float() @ gs["mlp.shared_experts.gate_proj.weight"].T).clamp(max=10.0))
            * (x.float() @ gs["mlp.shared_experts.up_proj.weight"].T).clamp(-10, 10)
        ) @ gs["mlp.shared_experts.down_proj.weight"].T
        refB_r = moe_ref(ix_dev, sc_dev, True)  # bfp4 experts, device routing
        refC_r = moe_ref(ix_dev, sc_dev, False)  # bf16 experts, device routing
        refA = Y[idx]
        p(f"T={T} (B={B} tokens) router set-match vs fp32 oracle {sm_or:.3f}  vs bf16-input CPU emulation {sm_em:.3f}")
        p(
            f"T={T} routed-only PCC: vs bfp4-expert CPU ref (device routing) {pcc(routed, refB_r):.5f}  vs bf16-expert ref {pcc(routed, refC_r):.5f}  [bfp4 vs bf16 ref: {pcc(refB_r, refC_r):.5f}]"
        )
        p(
            f"T={T} routed+shared PCC vs ORACLE MoE output (fp32): {pcc(total, refA):.5f}   vs bfp4-expert CPU ref + shared: {pcc(total, refB_r + sh_ref):.5f}   vs bf16-expert ref + shared: {pcc(total, refC_r + sh_ref):.5f}"
        )
        per_tok = [pcc(total[t], refA[t]) for t in range(B)]
        p(f"T={T} per-token PCC vs oracle min {min(per_tok):.4f} median {statistics.median(per_tok):.4f}")
        us = traced_us(md, lambda: run(True))
        us_r = traced_us(md, lambda: run(False))
        p(f"T={T} traced latency per layer: routed+shared {us:.1f} us, routed only {us_r:.1f} us")
        us_g = traced_us(md, lambda: blk.gate.forward(tt_gate)[0])
        us_s = traced_us(md, lambda: shared.forward(tt_gate))
        from models.demos.blackhole.hy4_preview.tt.hy4_router import Hy4Gate

        g2 = Hy4Gate(md, W["gate_weight"], W["gate_bias"], mode="exact" if blk.gate.mode == "fused" else "fused")
        us_g2 = traced_us(md, lambda: g2.forward(tt_gate)[0])
        ix2 = ttnn.to_torch(ttnn.get_device_tensors(g2.forward(tt_gate)[1])[0]).reshape(-1, 8).long()
        p(
            f"T={T} router alone: {blk.gate.mode} {us_g:.1f} us, {g2.mode} {us_g2:.1f} us (set-match of the two modes on row 0: {set_match(ix2, ix_dev[:T]):.3f}); shared expert partial alone {us_s:.1f} us"
        )
        ttnn.deallocate(out_r)
        ttnn.deallocate(out_f)
        del shared
finally:
    close_mesh(md)

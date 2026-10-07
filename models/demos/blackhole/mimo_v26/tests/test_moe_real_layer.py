# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real-weight MoE layer on 4x8 vs the track-0 dump ffn_out: all 256 REAL experts of one layer (fp4 -> bf16 -> bfp8), real gate + bias, real ffn_in.
Expert ops compared in ONE process (MIMO_UM_OPS, default "loop,unified"): loop = bf16-activation path (extract -> routed_expert_ffn -> insert, DRAM-interleaved
weights), unified = bfp8-activation fused op (ND-sharded weights). For each op: (A) dump routing -> PCC per mesh row vs bf16-dump ffn_out (and vs fp32-chain dump),
(B) device `exact` router end to end + 5-run determinism, capacity check from the device counts; then paired A/B timing (alternating, host-timed hint).
Env: MIMO_LAYER (1), MIMO_UM_CF (4), MIMO_REPEAT (1: tile the 512 tokens/row k times -> 512k per row, real tokens repeated), MIMO_DECODE (0/1: 64-token rows,
32 slots/row with ragged live counts MIMO_LIVE=32,20,7,1, dead slots zero), MIMO_UM_FID (hifi2), MIMO_UM_FP32ACC (1), MIMO_TIME_ITERS (5).
"""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_decode_4x8 import load_gate
from models.demos.blackhole.mimo_v26.tests.test_moe_prefill_4x8 import pcc
from models.demos.blackhole.mimo_v26.tt.moe_prefill import (
    COLS,
    EPC,
    HIDDEN,
    NUM_ROUTED,
    ROWS,
    TOPK,
    MiMoPrefillMoE,
    build_expert_weights,
    global_expert,
    make_ckc,
)
from models.demos.blackhole.mimo_v26.tt.moe_weights import load_layer_experts
from models.demos.blackhole.mimo_v26.tt.router import MiMoGate

DUMP = os.environ.get("MIMO_DUMP", "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16")
DUMP32 = os.environ.get("MIMO_DUMP_F32", "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16-f32")
LAYER = int(os.environ.get("MIMO_LAYER", "1"))
OPS = os.environ.get("MIMO_UM_OPS", "loop,unified").split(",")
REPEAT = int(os.environ.get("MIMO_REPEAT", "1"))
DECODE = os.environ.get("MIMO_DECODE", "0") == "1"
LIVE = [int(v) for v in os.environ.get("MIMO_LIVE", "32,20,7,1").split(",")]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_moe_real_layer(mesh_device):
    md = mesh_device
    a = torch.load(f"{DUMP}/layer_{LAYER}.pt", mmap=True)["prefill"]
    f = torch.load(f"{DUMP32}/layer_{LAYER}.pt", mmap=True)["prefill"]
    n0 = a["ffn_in"].shape[0] * a["ffn_in"].shape[1] // ROWS  # 512 tokens per mesh row in the dump
    x = a["ffn_in"].reshape(ROWS, n0, HIDDEN)
    ref_b = a["ffn_out"].reshape(ROWS, n0, HIDDEN).float()
    ref_f = f["ffn_out"].reshape(ROWS, n0, HIDDEN).float()
    idx_d = a["routing_idx"].reshape(ROWS, n0, TOPK)
    w_d = a["routing_w"].reshape(ROWS, n0, TOPK)
    live = torch.ones(ROWS, n0, dtype=torch.bool)
    if DECODE:  # 64-slot rows: 32 slots (ragged live) + 32 dead; live tokens strided over the row's 4 users
        sel = torch.arange(0, n0, n0 // 32)[:32]
        x, ref_b, ref_f, idx_d, w_d = x[:, sel], ref_b[:, sel], ref_f[:, sel], idx_d[:, sel], w_d[:, sel]
        pad = 32
        x = torch.cat([x, torch.zeros(ROWS, pad, HIDDEN, dtype=x.dtype)], 1)
        ref_b, ref_f = (torch.cat([t, torch.zeros(ROWS, pad, HIDDEN)], 1) for t in (ref_b, ref_f))
        idx_d = torch.cat(
            [idx_d, idx_d[:, :pad] * 0 + torch.arange(TOPK)], 1
        )  # dead slots routed to experts 0..7 (padding rows must still route somewhere)
        w_d = torch.cat([w_d, torch.full((ROWS, pad, TOPK), 1.0 / TOPK)], 1)
        live = torch.zeros(ROWS, 64, dtype=torch.bool)
        for r in range(ROWS):
            live[r, : LIVE[r]] = True
        x = torch.where(live.unsqueeze(-1), x, torch.zeros_like(x))
    elif REPEAT > 1:
        rp = lambda t: t.repeat(1, REPEAT, *([1] * (t.dim() - 2)))
        x, ref_b, ref_f, idx_d, w_d = rp(x), rp(ref_b), rp(ref_f), rp(idx_d), rp(w_d)
        live = torch.ones(ROWS, x.shape[1], dtype=torch.bool)
    N = x.shape[1]
    idx_all = idx_d.reshape(-1, TOPK)
    print(
        f"RESULT layer {LAYER}: tokens per mesh row N={N} decode={DECODE} repeat={REPEAT} live/row={live.sum(1).tolist()}",
        flush=True,
    )

    t0 = time.time()
    W = load_layer_experts(LAYER)
    print(f"RESULT loaded+dequantised 256 real experts in {time.time() - t0:.0f}s", flush=True)
    cf = int(os.environ.get("MIMO_UM_CF", "4"))
    fid = os.environ.get("MIMO_UM_FID", "hifi2")
    f32 = os.environ.get("MIMO_UM_FP32ACC", "1") == "1"
    ums = {}
    for op in OPS:
        t0 = time.time()
        wts = build_expert_weights(md, lambda e: W[e], nd_sharded=(op not in ("loop", "mm")))
        ums[op] = MiMoPrefillMoE(
            md,
            wts,
            ckc=make_ckc(fid, fp32acc=(f32 and op != "fused"), packer_l1_acc=(op != "fused")),
            expert_op=op,
            cf=cf,
        )
        print(
            f"RESULT op={op}: bfp8 weights on device in {time.time() - t0:.0f}s (fidelity={fid} fp32acc={f32})",
            flush=True,
        )

    # real skew / capacity from routing of the live tokens
    cnt = torch.bincount(idx_all[live.flatten()].flatten(), minlength=NUM_ROUTED)
    load = {
        (r, c): sum(-(-int(cnt[global_expert(r, c, l)]) // 32) * 32 for l in range(EPC))
        for r in range(ROWS)
        for c in range(COLS)
    }
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants

    wl = max(load.values())
    print(
        f"RESULT skew: per-expert count mean {cnt.float().mean():.1f} max {int(cnt.max())} (expert {int(cnt.argmax())}); worst chip load {wl} rows",
        flush=True,
    )
    for cfx in (1, 2, 3, 4):
        mb = compute_constants(N, NUM_ROUTED, TOPK, ROWS * COLS, ROWS, cfx)
        print(
            f"RESULT capacity cf={cfx}: max_buf {mb[2]} rows ({mb[2] * HIDDEN * 2 / 2**20:.0f} MB bf16 dispatch buffer/chip), max_per_expert {mb[3]}: worst load {wl} -> {'OK' if wl <= mb[2] else 'OVERFLOW'}",
            flush=True,
        )

    m2 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    xt = ttnn.from_torch(
        x.reshape(ROWS, 1, N, HIDDEN),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=m2,
    )

    def run(um, wt, ix, keep=False):
        x_rm = ttnn.reshape(ttnn.to_layout(xt, ttnn.ROW_MAJOR_LAYOUT), [1, N, HIDDEN])
        part = um.forward(x_rm, wt, ix, keep_counts=keep)
        ttnn.synchronize_device(md)
        devs = ttnn.get_device_tensors(part)
        out = torch.stack(
            [
                torch.stack([ttnn.to_torch(devs[r * COLS + c]).reshape(N, HIDDEN).float() for c in range(COLS)]).sum(0)
                for r in range(ROWS)
            ]
        )
        ttnn.deallocate(part)
        return out

    def pr(out, ref):
        return [round(pcc(out[r][live[r]], ref[r][live[r]]), 5) for r in range(ROWS)]

    print("RESULT floor: bf16-chain ffn_out vs fp32-chain ffn_out PCC per mesh row %s" % pr(ref_b, ref_f), flush=True)
    ixd = ttnn.from_torch(
        idx_d.reshape(ROWS, N, 1, 1, TOPK).to(torch.int32),
        device=md,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=m2,
    )
    wtd = ttnn.from_torch(
        w_d.reshape(ROWS, N, 1, 1, TOPK),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=m2,
    )
    ixd, wtd = ttnn.reshape(ixd, [N, 1, 1, TOPK]), ttnn.reshape(wtd, [N, 1, 1, TOPK])
    gw, gb = load_gate(LAYER)
    gate = MiMoGate(md, gw.to(torch.bfloat16), gb, bias_shift=0.0, batch_per_device=None)
    failures = []
    for op, um in ums.items():
        outA = run(um, wtd, ixd, keep=True)
        sh = um.shared(N)
        worst = 0
        for r in range(ROWS):
            for c in range(COLS):
                cn = (
                    ttnn.to_torch(ttnn.get_device_tensors(um.last_counts)[r * COLS + c]).reshape(-1)[:NUM_ROUTED].long()
                )
                worst = max(worst, sum(-(-int(cn[global_expert(r, c, l)]) // 32) * 32 for l in range(EPC)))
        ttnn.deallocate(um.last_counts)
        print(
            f"RESULT op={op} device counts: worst chip load {worst} of max_buf {sh.max_buf} at cf={cf}: {'ok, no dropped tokens' if worst <= sh.max_buf else 'OVERFLOW'}",
            flush=True,
        )
        pa = pr(outA, ref_b)
        print(
            f"RESULT layer {LAYER} op={op} (A) dump routing: PCC vs bf16 dump ffn_out per row {pa} ; vs fp32 dump {pr(outA, ref_f)}",
            flush=True,
        )
        wtB, ixB = gate.route(xt, "exact")
        ixh = torch.stack(
            [ttnn.to_torch(ttnn.get_device_tensors(ixB)[r * COLS]).reshape(N, TOPK).long() for r in range(ROWS)]
        )
        eq = (ixh.sort(-1).values == idx_d.sort(-1).values).all(-1)[live]
        outs = [run(um, wtB, ixB) for _ in range(5)]
        det = all(bool(torch.equal(outs[0], o)) for o in outs[1:])
        pb = pr(outs[0], ref_b)
        print(
            f"RESULT layer {LAYER} op={op} (B) exact router (set match vs dump {float(eq.float().mean()):.4f}): PCC vs bf16 dump per row {pb} ; vs fp32 dump {pr(outs[0], ref_f)} ; 5 runs bit-identical: {det}",
            flush=True,
        )
        if min(pa) < 0.999 or min(pb) < 0.999 or not det:
            failures.append((op, pa, pb, det))
    if len(ums) > 1:  # paired A/B timing, alternating in one process (hint only)
        iters = int(os.environ.get("MIMO_TIME_ITERS", "5"))
        ts = {op: [] for op in ums}
        for it in range(iters + 1):
            for op, um in ums.items():
                t0 = time.time()
                x_rm = ttnn.reshape(ttnn.to_layout(xt, ttnn.ROW_MAJOR_LAYOUT), [1, N, HIDDEN])
                p_ = um.forward(x_rm, wtd, ixd)
                ttnn.synchronize_device(md)
                ts[op].append((time.time() - t0) * 1e3)
                ttnn.deallocate(p_)
        print(
            "RESULT time hint (paired, alternating, one process, host-timed, experts+dispatch+combine, N=%d/row): %s"
            % (
                N,
                {
                    op: f"step0 {v[0]:.1f} ms, median steps1-{iters} {sorted(v[1:])[len(v[1:]) // 2]:.1f} ms"
                    for op, v in ts.items()
                },
            ),
            flush=True,
        )
    assert not failures, failures

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""STAGE-INSTRUMENTED variant (race track): per-stage on-device bitwise compare of clones of every MoE stage output against replay 0.

Original doc: Race hunt for the rare decode-MoE glitch (LOG E261007-0818-033): chain MIMO_NL real layers (default 6) of MiMoMoEDecode (the decode agent's path: live mask, spread, fast/exact router)
in ONE trace, replay MIMO_REPS times on identical inputs and compare every layer's output bitwise with the first replay, per mesh row (columns identical after the all-reduce).
Inputs: real dump hidden states, distinct per mesh row, MIMO_LIVE live slots per row. Records every glitch (replay, layer, rows, max abs diff). Knobs: MIMO_ROUTER (exact|fast),
MIMO_STAGE (99 = full layer | 5 = per-column partial sums before the all-reduce | 4 = combine output etc: see MiMoMoEDecode.forward upto), env MOE_DISPATCH_SPARSE_MCAST, MIMO_DISP_LINKS/WORKERS.
"""
import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_decode_4x8 import load_gate
from models.demos.blackhole.mimo_v26.tt.moe_decode import SLOTS, MiMoMoEDecode
from models.demos.blackhole.mimo_v26.tt.moe_prefill import (
    COLS,
    EPC,
    HIDDEN,
    NUM_ROUTED,
    ROWS,
    TOPK,
    MiMoPrefillMoE,
    build_expert_weights,
    make_ckc,
)
from models.demos.blackhole.mimo_v26.tt.moe_weights import load_layer_experts
from models.demos.blackhole.mimo_v26.tt.router import MiMoGate

DUMP = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16"
DDUMP = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-device-s128b16"
NL = int(os.environ.get("MIMO_NL", "6"))
REPS = int(os.environ.get("MIMO_REPS", "3000"))
LIVE = int(os.environ.get("MIMO_LIVE", "8"))
ROUTER = os.environ.get("MIMO_ROUTER", "exact")
STAGE = int(os.environ.get("MIMO_STAGE", "99"))
NSLOT = 32


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 400_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_moe_race_stages(mesh_device):
    md = mesh_device
    m2 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    dec, xdev, layers = {}, {}, list(range(1, 1 + NL))
    sel = torch.arange(0, 512, 512 // NSLOT)[:NSLOT]
    for L in layers:
        src = DUMP if L <= 5 else DDUMP
        a = torch.load(f"{src}/layer_{L}.pt", mmap=True)["prefill"]
        x = a["ffn_in"].reshape(ROWS, 512, HIDDEN)[:, sel].reshape(ROWS, 1, NSLOT, HIDDEN).clone()
        x[:, :, LIVE:] = 0
        W = load_layer_experts(L)
        wts = build_expert_weights(md, lambda e: W[e], nd_sharded=True)
        del W
        gw, gb = load_gate(L)
        gate = MiMoGate(md, gw.to(torch.bfloat16), gb, bias_shift=0.0, batch_per_device=None)
        for n in ("tt_bias", "tt_input_indices", "tt_output", "tt_output_indices"):
            ttnn.deallocate(getattr(gate, n))
        moe = MiMoPrefillMoE(
            md, wts, ckc=make_ckc("hifi2", fp32acc=True, packer_l1_acc=True), expert_op="unified", cf=4
        )
        dec[L] = MiMoMoEDecode(md, gate, moe, router=ROUTER, dead=os.environ.get("MIMO_DEAD", "sentinel"))
        xdev[L] = ttnn.from_torch(
            x,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=m2,
        )
    d0 = dec[layers[0]]
    mask_dev = d0.alloc_mask(d0.make_mask_host([LIVE] * ROWS))
    main_body(md, dec, xdev, layers, mask_dev)


def tf(t):
    """any tensor -> float32 TILE (for comparison only)"""
    if t.layout != ttnn.TILE_LAYOUT:
        t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
    if t.dtype != ttnn.float32:
        t = ttnn.typecast(t, ttnn.float32)
    return t


def staged(d, x, mask, taps):
    """MiMoMoEDecode.forward (exact router) + MiMoPrefillMoE.forward, cloning every stage output into taps[name]."""
    dp = ttnn.experimental.deepseek_prefill
    moe = d.moe
    mc = ttnn.DRAM_MEMORY_CONFIG

    def tap(name, t):
        taps[name] = ttnn.clone(t, memory_config=mc)

    B = x.shape[2]
    xp = ttnn.pad(x, [(0, 0), (0, 0), (0, SLOTS - B), (0, 0)], 0.0)
    wts, idx = d.gate.route(xp, "exact")
    wts2, idx2 = d._spread(wts, idx, mask)
    tap("sw", wts2)
    tap("si", idx2)
    x_rm = ttnn.reshape(ttnn.to_layout(xp, ttnn.ROW_MAJOR_LAYOUT), [1, SLOTS, HIDDEN])
    N = SLOTS
    sh = moe.shared(N)
    idx_t = ttnn.to_layout(ttnn.reshape(idx2, [N, TOPK]), ttnn.TILE_LAYOUT)
    hist = dp.masked_bincount(idx_t, sh.dispatch_table, NUM_ROUTED, TOPK)
    ttnn.deallocate(idx_t)
    tap("hist", hist)
    offsets, counts, region_offsets, _ = dp.offset_cumsum(
        hist,
        cluster_axis=0,
        num_links=sh.num_links,
        experts_per_chip=EPC,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        use_l1_small_for_semaphores=True,
    )
    ttnn.deallocate(hist)
    tap("offsets", offsets)
    tap("counts", counts)
    tap("roff", region_offsets)
    disp, meta = dp.dispatch(
        input_tensor=x_rm,
        indices_tensor=ttnn.reshape(idx2, [1, N, TOPK]),
        expert_offsets_tensor=offsets,
        expert_dispatch_table_tensor=sh.dispatch_table,
        padding_config=None,
        scales_tensor=None,
        fp8_scaled_input=False,
        dispatch_group_size=ROWS,
        experts_per_chip=EPC,
        num_routed_experts=NUM_ROUTED,
        num_experts_per_tok=TOPK,
        metadata_len=sh.meta_len,
        max_dispatch_buffer_token_size=sh.max_buf,
        cluster_axis=0,
        num_links=sh.num_links,
        topology=sh.topology,
        fp8_output=False,
        subdevice_id=None,
        num_workers_per_sender=sh.workers,
        use_l1_small_for_semaphores=True,
    )
    ttnn.deallocate(offsets)
    tap("disp", disp)
    tap("meta", meta)
    disp2 = ttnn.reshape(disp, [sh.max_buf, x_rm.shape[2]])
    out = dp.unified_routed_expert_moe(
        disp2,
        region_offsets,
        counts,
        sh.gidx,
        moe.gate_projs,
        moe.up_projs,
        moe.down_projs,
        max_dispatched_tokens_per_expert=sh.max_per_expert,
        compute_kernel_config=moe.ckc,
        activation=ttnn.RoutedExpertActivation.Silu,
    )
    ttnn.deallocate(disp)
    tap("exp", out)
    out5 = ttnn.reshape(out, [1, 1, sh.max_buf, out.shape[-1]])
    comb = dp.combine(
        out5,
        meta,
        counts,
        region_offsets,
        dispatch_group_size=ROWS,
        experts_per_chip=EPC,
        num_experts_per_tok=TOPK,
        seq_len_per_chip=N,
        cluster_axis=0,
        num_links=sh.num_links,
        topology=sh.topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        init_zeros=False,
        use_fp8_combine=False,
        use_l1_small_for_semaphores=True,
    )
    ttnn.deallocate(out)
    ttnn.deallocate(meta)
    tap("comb", comb)
    ttnn.deallocate(counts)
    ttnn.deallocate(region_offsets)
    w5 = ttnn.reshape(wts2, [1, 1, N, TOPK, 1])
    i5 = ttnn.reshape(idx2, [1, 1, N, TOPK])
    summed = dp.post_combine_reduce(
        comb, w5, i5, sh.dispatch_table, expert_dim=3, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.deallocate(comb)
    tap("part", summed)
    ar = os.environ.get("MIMO_AR", "default")
    if ar == "ag_sum":
        g = ttnn.all_gather(summed, dim=0, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.sum(g, dim=0, keepdim=True)
        ttnn.deallocate(g)
    elif ar == "links1":
        o = ttnn.all_reduce(summed, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, num_links=1)
    elif ar == "linear":
        o = ttnn.all_reduce(
            summed, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=ttnn.Topology.Linear
        )
    elif ar == "rs_ag":
        r_ = ttnn.reduce_scatter(summed, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.all_gather(r_, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(r_)
    else:
        o = ttnn.all_reduce(summed, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(summed)
    o2 = ttnn.slice(o, [0, 0, 0, 0], [1, 1, B, HIDDEN])
    ttnn.deallocate(o)
    tap("final", o2)
    return o2


def main_body(md, dec, xdev, layers, mask_dev):
    # eager pass 1: shapes -> persistent refs allocated BEFORE trace capture
    def run_all(taps_l):
        for L in layers:
            t = {}
            staged(dec[L], xdev[L], mask_dev, t)
            taps_l.append(t)

    t_e = []
    run_all(t_e)
    ttnn.synchronize_device(md)
    keep = os.environ.get("MIMO_TAPS", "sw,si,hist,offsets,counts,roff,part,final").split(",")
    names = [(i, k) for i, t in enumerate(t_e) for k in t if k in keep]
    ref_f = {(i, k): ttnn.clone(tf(t_e[i][k])) for (i, k) in names}
    ttnn.synchronize_device(md)

    def cmp(tl):
        flags = []
        for i, k in names:
            a = tf(tl[i][k])
            ne = ttnn.ne(a, ref_f[(i, k)])
            s_ = ttnn.sum(ne)
            flags.append(ttnn.to_layout(ttnn.reshape(s_, [1, 1, 1, 1]), ttnn.ROW_MAJOR_LAYOUT))
        return ttnn.concat(flags, dim=3)

    cmp(t_e)  # compile the compare ops before the capture
    t_w = []
    run_all(t_w)  # second eager pass: warm program cache of every stage op in the exact capture sequence
    ttnn.synchronize_device(md)
    t_tr = []
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    run_all(t_tr)
    fl = cmp(t_tr)
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    comp = ttnn.ConcatMesh2dToTensor(md, dims=(0, 1), mesh_shape=tuple(md.shape))
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    for n in names:  # replay 0 defines the reference (in place: the trace reads the ref_f buffers)
        ttnn.copy(tf(t_tr[n[0]][n[1]]), ref_f[n])
    ttnn.synchronize_device(md)
    events = []
    t0 = time.time()
    for rep in range(REPS):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        f = ttnn.to_torch(fl, mesh_composer=comp).reshape(ROWS, COLS, -1)
        bad = f.nonzero()
        if len(bad):
            by_name = {}
            for r, c, j in bad.tolist():
                i, k = names[j]
                by_name.setdefault(f"L{layers[i]}.{k}", []).append((r, c, float(f[r, c, j])))
            events.append({"rep": rep, "taps": by_name})
            for r, c, j in bad.tolist():
                i, k = names[j]
                if k == "final" and "part" in t_tr[i]:
                    di = r * COLS + c
                    fin = ttnn.to_torch(ttnn.get_device_tensors(t_tr[i]["final"])[di]).float().reshape(-1, HIDDEN)
                    rf = ttnn.to_torch(ttnn.get_device_tensors(ref_f[(i, "final")])[di]).float().reshape(-1, HIDDEN)
                    parts = [
                        ttnn.to_torch(ttnn.get_device_tensors(t_tr[i]["part"])[r * COLS + kk])
                        .float()
                        .reshape(-1, HIDDEN)
                        for kk in range(COLS)
                    ]
                    sm = sum(parts)[: fin.shape[0]]
                    nz = (fin != rf).nonzero()
                    rows_ = sorted({int(a) for a, _ in nz.tolist()})
                    cols_ = [int(b) for _, b in nz.tolist()]
                    det = {
                        "layer": layers[i],
                        "dev": (r, c),
                        "rows": rows_,
                        "col_min": min(cols_),
                        "col_max": max(cols_),
                        "n": len(cols_),
                        "glitch_first8": fin[nz[0][0], cols_[0] : cols_[0] + 8].tolist(),
                        "ref_first8": rf[nz[0][0], cols_[0] : cols_[0] + 8].tolist(),
                        "sumparts_first8": sm[nz[0][0], cols_[0] : cols_[0] + 8].tolist(),
                        "part_cols_first": [p_[nz[0][0], cols_[0]].item() for p_ in parts],
                        "ref_eq_sumparts": bool(torch.allclose(rf, sm, atol=0.1, rtol=0.02)),
                    }
                    print("RESULT DETAIL", json.dumps(det), flush=True)
            print("RESULT EVENT", json.dumps(events[-1]), flush=True)
    print(
        f"RESULT staged race stress: layers={layers} live={LIVE} replays={REPS} ({REPS * len(layers)} layer-calls) in {time.time() - t0:.0f}s: "
        f"events {len(events)} links={os.environ.get('MIMO_DISP_LINKS', 2)} workers={os.environ.get('MIMO_DISP_WORKERS', 2)}",
        flush=True,
    )
    assert not events


# (test body below)

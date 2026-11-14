# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""moe-prefill: per-op table of the Hy4 MoE PREFILL block (whole 4x8 mesh, N tokens per mesh row = one user per row) by cumulative-prefix trace diff
(no profiler in the image) AND eager per-stage sync timing, on REAL layer-1 activations.  Stages = hy4_prefill_moe.moe_cols pieces + shared expert partial/AR.
Env: HY4_NS=512 (comma list of rows/user), HY4_LAYER=1, PM_REP=3, PM_N=9, PM_MODE=trace|eager|both, HY4_UNI_* levers (hy4_prefill_moe), OUT=json path,
PM_PCC=1 (routed+shared PCC vs oracle at the first N, per row)."""
import json
import os
import statistics
import time

import torch

import models.demos.blackhole.hy4_preview.tt.hy4_prefill_moe as PM
import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, close_mesh, open_mesh, p, pcc
from models.demos.blackhole.hy4_preview.tt import hy4_dense as D
from models.demos.blackhole.hy4_preview.tt.hy4_moe import Hy4MoEBlock
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer
from models.demos.blackhole.hy4_preview.tt.hy4_pf_opt import PfOpt
from models.demos.blackhole.hy4_preview.tt.hy4_prefill_moe import (
    COLS,
    NUM_ROUTED,
    ROWS,
    TOPK,
    Hy4UnifiedMoE,
    get_shared,
    moe_cols,
    reduce_scatter_tokens,
    row_topology,
)
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

LAYER = int(os.environ.get("HY4_LAYER", "1"))
NS = [int(v) for v in os.environ.get("HY4_NS", "512").split(",")]
REP = int(os.environ.get("PM_REP", "3"))
NREP = int(os.environ.get("PM_N", "9"))
MODE = os.environ.get("PM_MODE", "both")
HID = 6144
OUT = os.environ.get("OUT", "/tmp/pfmoe.json")
ttnn_dealloc = ttnn.deallocate


def stages_for(um, blk, mc, cc, shared, xod, xd, N):
    sh = get_shared(um.md, N)
    gate = blk.gate
    n = xod.shape[2]
    E = ttnn.experimental.deepseek_prefill

    def s_ag(st):
        st["h"] = mc.allgather(xod, cc, axis=1, dim=2)

    def s_router(st):
        gy = int(os.environ.get("HY4_PF_RT_GY", "0")) or min(4, max(1, n // 128))
        st["sc"], st["ix"] = gate.forward(xod, grid=(gy, 8))

    def s_pack(st):
        f32 = lambda a: ttnn.typecast(ttnn.to_layout(ttnn.reshape(a, [1, 1, n, TOPK]), ttnn.TILE_LAYOUT), ttnn.float32)
        ixf, scf = f32(st["ix"]), f32(st["sc"])
        pk_t = ttnn.concat([ixf, scf], dim=3)
        st["_ixf"], st["_scf"], st["_pkt"] = ixf, scf, pk_t
        st["pk"] = ttnn.reshape(ttnn.to_layout(pk_t, ttnn.ROW_MAJOR_LAYOUT), [1, 1, n // 32, 32 * 2 * TOPK])

    def s_ag_route(st):
        pg_w = mc.allgather(st["pk"], cc, axis=1, dim=2)
        st["_pgw"] = pg_w
        pg = ttnn.reshape(pg_w, [1, 1, COLS * n, 2 * TOPK])
        st["ix_all"] = ttnn.reshape(
            ttnn.typecast(ttnn.slice(pg, [0, 0, 0, 0], [1, 1, COLS * n, TOPK]), ttnn.uint16), [COLS * n, 1, 1, TOPK]
        )
        st["sc_all"] = ttnn.reshape(
            ttnn.typecast(ttnn.slice(pg, [0, 0, 0, TOPK], [1, 1, COLS * n, 2 * TOPK]), ttnn.bfloat16),
            [COLS * n, 1, 1, TOPK],
        )

    def s_rm(st):
        st["x_rm"] = ttnn.reshape(ttnn.to_layout(st["h"], ttnn.ROW_MAJOR_LAYOUT), [1, st["h"].shape[2], HID])

    def s_bincount(st):
        idx2 = ttnn.reshape(st["ix_all"], [N, TOPK])
        idx_t = ttnn.to_layout(idx2, ttnn.TILE_LAYOUT)
        st["_idxt"] = idx_t
        st["hist"] = E.masked_bincount(idx_t, sh.dispatch_table, NUM_ROUTED, TOPK)

    def s_cumsum(st):
        st["offsets"], st["counts"], st["region"], _ = E.offset_cumsum(
            st["hist"],
            cluster_axis=0,
            num_links=sh.num_links,
            experts_per_chip=8,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            use_l1_small_for_semaphores=sh.l1_small,
        )

    def s_dispatch(st):
        idx3 = ttnn.reshape(st["ix_all"], [1, N, TOPK])
        st["disp"], st["meta"] = E.dispatch(
            input_tensor=st["x_rm"],
            indices_tensor=idx3,
            expert_offsets_tensor=st["offsets"],
            expert_dispatch_table_tensor=sh.dispatch_table,
            padding_config=None,
            scales_tensor=None,
            fp8_scaled_input=False,
            dispatch_group_size=ROWS,
            experts_per_chip=8,
            num_routed_experts=NUM_ROUTED,
            num_experts_per_tok=TOPK,
            metadata_len=sh.meta_len,
            max_dispatch_buffer_token_size=sh.max_buf,
            cluster_axis=0,
            num_links=sh.num_links,
            topology=row_topology(),
            fp8_output=False,
            subdevice_id=None,
            num_workers_per_sender=sh.workers,
            use_l1_small_for_semaphores=sh.l1_small,
        )

    def s_experts(st):
        d2 = ttnn.reshape(st["disp"], [sh.max_buf, HID])
        st["out"] = E.unified_routed_expert_moe(
            d2,
            st["region"],
            st["counts"],
            sh.gidx,
            um.gate_projs,
            um.up_projs,
            um.down_projs,
            max_dispatched_tokens_per_expert=sh.max_per_expert,
            compute_kernel_config=sh.ckc,
            activation=ttnn.RoutedExpertActivation.ClampedSiluGlu,
        )

    def s_combine(st):
        out5 = ttnn.reshape(st["out"], [1, 1, sh.max_buf, st["out"].shape[-1]])
        st["comb"] = E.combine(
            out5,
            st["meta"],
            st["counts"],
            st["region"],
            dispatch_group_size=ROWS,
            experts_per_chip=8,
            num_experts_per_tok=TOPK,
            seq_len_per_chip=N,
            cluster_axis=0,
            num_links=sh.num_links,
            topology=row_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            init_zeros=False,
            use_fp8_combine=False,
            use_l1_small_for_semaphores=sh.l1_small,
        )

    def s_reduce(st):
        w5 = ttnn.reshape(st["sc_all"], [1, 1, N, TOPK, 1])
        i5 = ttnn.reshape(st["ix_all"], [1, 1, N, TOPK])
        st["summed"] = E.post_combine_reduce(
            st["comb"], w5, i5, sh.dispatch_table, expert_dim=3, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def s_rs(st):
        st["final"] = reduce_scatter_tokens(st.pop("summed"), cc)

    def s_shared(st):
        st["shp"] = shared.forward(xd)

    def s_shared_ar(st):
        st["shp2"] = D.allreduce_cols(st.pop("shp"))

    return [
        ("allgather hidden over cols", s_ag),
        ("router (own rows)", s_router),
        ("route pack (typecast/concat/RM)", s_pack),
        ("allgather routing", s_ag_route),
        ("hidden to ROW_MAJOR", s_rm),
        ("masked_bincount", s_bincount),
        ("offset_cumsum", s_cumsum),
        ("dispatch", s_dispatch),
        ("unified_routed_expert_moe", s_experts),
        ("combine", s_combine),
        ("post_combine_reduce", s_reduce),
        ("reduce_scatter+a2a (cols)", s_rs),
        ("shared expert partial", s_shared),
        ("shared expert all-reduce", s_shared_ar),
    ]


protect = set()


def free_all(st):
    seen = set()

    def rec(v):
        if isinstance(v, ttnn.Tensor):
            if id(v) not in seen and id(v) not in protect:
                seen.add(id(v))
                try:
                    ttnn_dealloc(v)
                except Exception:
                    pass
        elif isinstance(v, (tuple, list)):
            for e in v:
                rec(e)

    for v in list(st.values()):
        rec(v)
    st.clear()
    PM.flush_trash()


def run_prefix(md, stages, k):
    st = {}
    for name, fn in stages[:k]:
        fn(st)
    free_all(st)


def trace_us(md, stages, k):
    def body():
        for _ in range(REP):
            run_prefix(md, stages, k)

    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    body()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(NREP):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / REP * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts), min(ts)


def trace_block(md, block):
    def body():
        for _ in range(REP):
            free_all({"o": block()})

    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    body()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(NREP):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / REP * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts)


def eager_us(md, stages):
    """per-stage eager time with a sync after each stage, median of 5 (includes host dispatch; compare only)."""
    acc = [[] for _ in stages]
    for it in range(6):
        st = {}
        for i, (name, fn) in enumerate(stages):
            ttnn.synchronize_device(md)
            t0 = time.perf_counter()
            fn(st)
            ttnn.synchronize_device(md)
            if it:
                acc[i].append((time.perf_counter() - t0) * 1e6)
        free_all(st)
    return [statistics.median(a) for a in acc]


acts = torch.load(ACTS + "acts_l0_l1.pt")
gs = torch.load(ACTS + "l1_gate_shared.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]])
Y = torch.cat([y.reshape(-1, HID) for h, y, r in acts["moe"]])
md = open_mesh(600_000_000)
res = {
    "layer": LAYER,
    "env": {k: v for k, v in os.environ.items() if k.startswith(("MOE_", "HY4_", "PM_"))},
    "runs": {},
}
try:
    t0 = time.time()
    W = load_moe_layer(LAYER)
    p(f"load_moe_layer({LAYER}) {time.time()-t0:.1f}s")
    shared_w = [W[k][256] for k in ("shared_w0", "shared_w1", "shared_w2")]
    blk = Hy4MoEBlock(md, W, batch_per_device=1)
    blk.warmup()
    um = Hy4UnifiedMoE(md, LAYER, ring=(blk.decode.expert_state.tt_w0_w1, blk.decode.expert_state.tt_w2))
    mc, cc = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    shared = D.make_shared(md, *shared_w, decode=False)
    for N in NS:
        sh = get_shared(md, N)
        p(f"N={N}/row: max_buf={sh.max_buf} max_per_expert={sh.max_per_expert} cf={sh.cf}")
        idx = torch.stack([(torch.arange(N) + 80 * r) % X.shape[0] for r in range(ROWS)])
        x = X[idx.flatten()].reshape(ROWS, N, HID).to(torch.bfloat16)
        xd = ttnn.from_torch(
            x.reshape(ROWS, 1, N, HID),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS)),
        )
        n_own = N // COLS
        xo = x.reshape(ROWS, N // 256, COLS, 32, HID).permute(0, 2, 1, 3, 4).reshape(ROWS, COLS, n_own, HID)
        xod = ttnn.from_torch(
            xo,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(ROWS, COLS)),
        )
        protect.update([id(xd), id(xod)])
        stages = stages_for(um, blk, mc, cc, shared, xod, xd, N)
        # warm + correctness once eagerly (routed only chain = stages[:12])
        run_prefix(md, stages, len(stages))
        ttnn.synchronize_device(md)
        r = {"N": N}
        if os.environ.get("PM_PCC", "0") == "1" and N == NS[0]:
            st = {}
            for name, fn in stages:
                fn(st)
            ttnn.synchronize_device(md)
            dev = lambda t, rr=0, c=0: ttnn.to_torch(ttnn.get_device_tensors(t)[rr * COLS + c])
            full = torch.zeros(ROWS, N, HID)
            for rr in range(ROWS):
                for c in range(COLS):
                    o = dev(st["final"], rr, c).float().reshape(N // 256, 32, HID)
                    for g in range(N // 256):
                        full[rr, 256 * g + 32 * c : 256 * g + 32 * c + 32] = o[g]
            sh_full = torch.stack([dev(st["shp2"], rr, 0).float().reshape(N, HID) for rr in range(ROWS)])
            ref = Y[idx.flatten()].reshape(ROWS, N, HID)
            r["pcc_rows"] = [pcc(full[rr] + sh_full[rr], ref[rr]) for rr in range(ROWS)]
            p(
                "diag: routed norm",
                [float(full[rr].norm()) for rr in range(ROWS)],
                "shared norm",
                [float(sh_full[rr].norm()) for rr in range(ROWS)],
                "ref norm",
                [float(ref[rr].norm()) for rr in range(ROWS)],
            )
            p(
                "diag: pcc routed-only vs ref",
                [pcc(full[rr], ref[rr]) for rr in range(ROWS)],
                "shared-only vs ref",
                [pcc(sh_full[rr], ref[rr]) for rr in range(ROWS)],
            )
            r["pcc_routed_only_vs_baseline_file"] = None
            if os.environ.get("PM_SAVE"):
                torch.save(full, os.environ["PM_SAVE"])
            if os.environ.get("PM_CMP"):
                base = torch.load(os.environ["PM_CMP"])
                r["pcc_vs_base_rows"] = [pcc(full[rr], base[rr]) for rr in range(ROWS)]
            p(f"N={N} PCC rows routed+shared vs oracle {r['pcc_rows']} vs_base {r.get('pcc_vs_base_rows')}")
            free_all(st)
        if MODE in ("eager", "both"):
            e = eager_us(md, stages)
            r["eager_us"] = {s[0]: v for s, v in zip(stages, e)}
            p(f"N={N} eager per-stage us: " + "; ".join(f"{s[0]} {v:.0f}" for s, v in zip(stages, e)))
        if MODE in ("trace", "both"):
            prev, tab = 0.0, []
            for k in range(1, len(stages) + 1):
                med, mn = trace_us(md, stages, k)
                tab.append({"op": stages[k - 1][0], "cum_us": med, "cum_min_us": mn, "step_us": med - prev})
                p(f"N={N} prefix {k:2d} {stages[k-1][0]:34s} cum {med:9.1f} us (min {mn:9.1f})  step {med-prev:8.1f}")
                prev = med
                if k == 12:
                    r["routed_block_us"] = med
                r["table"] = tab
        if os.environ.get("PM_VARIANTS"):
            r["variants"] = {}
            base_env = dict(os.environ)
            for spec in os.environ["PM_VARIANTS"].split(";"):
                vname, _, kv = spec.partition("|")
                for kk in [k for k in os.environ if k.startswith(("HY4_UNI_", "HY4_PF_"))]:
                    del os.environ[kk]
                for k, v in base_env.items():
                    if k.startswith(("HY4_UNI_", "HY4_PF_")) and k not in ("HY4_PF_VAR",):
                        pass  # global env of the run is NOT inherited by variants: variants are self-contained
                for item in [i for i in kv.split(",") if i]:
                    k, _, v = item.partition("=")
                    os.environ[k] = v
                PM._SHARED.clear()
                sh_v = get_shared(md, N)
                pf_v = PfOpt.from_env()

                def block():
                    o = moe_cols(um, blk.gate, xod, mc, cc, shared=shared, pf=pf_v)
                    if pf_v.shfold:
                        return [o]
                    shp = D.allreduce_cols(shared.forward(xd))
                    return [o, shp]

                rv = {}
                outs = None
                try:
                    pccs = []
                    for rep_i in range(int(os.environ.get("PM_PCC_REPS", "3"))):
                        outs = block()
                        ttnn.synchronize_device(md)
                        dev = lambda t, rr=0, c=0: ttnn.to_torch(ttnn.get_device_tensors(t)[rr * COLS + c])
                        full = torch.zeros(ROWS, N, HID)
                        for rr in range(ROWS):
                            for c in range(COLS):
                                o = dev(outs[0], rr, c).float().reshape(N // 256, 32, HID)
                                for g in range(N // 256):
                                    full[rr, 256 * g + 32 * c : 256 * g + 32 * c + 32] = o[g]
                        if len(outs) > 1:
                            full = full + torch.stack(
                                [dev(outs[1], rr, 0).float().reshape(N, HID) for rr in range(ROWS)]
                            )
                        ref = Y[idx.flatten()].reshape(ROWS, N, HID)
                        pccs.append([pcc(full[rr], ref[rr]) for rr in range(ROWS)])
                        free_all({"o": outs})
                    rv["pcc_reps"] = pccs
                    if os.environ.get("PM_TRACE_PCC", "1") == "1":
                        # production mode: ONE traced block (outputs kept), replayed, PCC read after each replay
                        ttnn.synchronize_device(md)
                        tid = ttnn.begin_trace_capture(md, cq_id=0)
                        touts = block()
                        ttnn.end_trace_capture(md, tid, cq_id=0)
                        PM.flush_trash()
                        ttnn.synchronize_device(md)
                        tp = []
                        for rep_i in range(int(os.environ.get("PM_TRACE_REPS", "4"))):
                            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                            ttnn.synchronize_device(md)
                            full = torch.zeros(ROWS, N, HID)
                            for rr in range(ROWS):
                                for c in range(COLS):
                                    o = dev(touts[0], rr, c).float().reshape(N // 256, 32, HID)
                                    for g in range(N // 256):
                                        full[rr, 256 * g + 32 * c : 256 * g + 32 * c + 32] = o[g]
                            if len(touts) > 1:
                                full = full + torch.stack(
                                    [dev(touts[1], rr, 0).float().reshape(N, HID) for rr in range(ROWS)]
                                )
                            tp.append([pcc(full[rr], ref[rr]) for rr in range(ROWS)])
                        ttnn.release_trace(md, tid)
                        free_all({"o": touts})
                        rv["pcc_trace_reps"] = tp
                    t_us = []
                    for _ in range(3):
                        t_us.append(trace_block(md, block))
                    rv["traced_us"] = t_us
                    rv["traced_med_us"] = statistics.median(t_us)
                except Exception as ex:  # record, do not continue with a possibly-broken device state
                    rv["error"] = repr(ex)
                    res["runs"].setdefault(N, r)["variants"][vname] = rv
                    json.dump(res, open(OUT, "w"), indent=1)
                    raise
                r["variants"][vname] = rv
                p(
                    f"N={N} VARIANT {vname} [{kv}] traced {rv['traced_med_us']:.1f} us {rv['traced_us']} PCC eager reps(min) {[round(min(x),5) for x in pccs]} TRACE reps(min) {[round(min(x),5) for x in rv.get('pcc_trace_reps', [])]}"
                )
                json.dump(res, open(OUT, "w"), indent=1)
        res["runs"][N] = r
        json.dump(res, open(OUT, "w"), indent=1)
finally:
    close_mesh(md)
    json.dump(res, open(OUT, "w"), indent=1)

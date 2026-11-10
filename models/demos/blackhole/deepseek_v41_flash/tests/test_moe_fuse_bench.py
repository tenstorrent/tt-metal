# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""T15: one-layer prefill MoE bench on the 4x8 mesh for the unified routed-expert op (production config: ring weights read in place,
MOE_COMPUTE_RING=40, bfp4; or MF_WMODE=copy bfp8/bfp4 own copy).  New file, nothing else touched.

Env: MF_N (1024,4096) tokens per mesh row, MF_KINDS (real,uni_rand), MF_LAYER (3), MF_ROUTES, MF_OUT (jsonl dir), MF_PCC (1),
MF_SYN (comma list of tokens per expert for a synthetic uniform-count sweep of the op alone, default "" = off), MF_TIME_STAGES (1),
MF_LEVER (name tag stored in records), DSV41_CKPT.  Per-chip padding statistics are computed on the host from the route ids."""

import json
import os
import time

import numpy as np
import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tt import moe_weights as mw
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_unified_moe import (
    COLS,
    EPC,
    HIDDEN,
    ROWS,
    TOPK,
    DSV41UnifiedMoE,
    _wmc,
    get_shared,
    global_expert,
)

OUT = os.environ.get("MF_OUT", "/tmp/moe_fuse_out")
NS = [int(v) for v in os.environ.get("MF_N", "1024,4096").split(",")]
KINDS = os.environ.get("MF_KINDS", "real,uni_rand").split(",")
LAYER = int(os.environ.get("MF_LAYER", "3"))
ROUTES = os.environ.get("MF_ROUTES", "/mnt/tt-data/ssinghal/dsv4-logs/ceil/routes_cq.pt")
DO_PCC = os.environ.get("MF_PCC", "1") == "1"
STAGES = os.environ.get("MF_STAGES", "1") == "1"
SYN = [int(v) for v in os.environ.get("MF_SYN", "").split(",") if v]
LEVER = os.environ.get("MF_LEVER", "base")
WMODE = os.environ.get(
    "MF_WMODE", "ring"
)  # ring (production: decode ring weights in place) | copy4 | copy8 (random weights, own copy, non-ring layout)
OPK = os.environ.get("MF_OP", "unified")  # unified | fused (moe_fused_swiglu) | hybrid
THR = int(os.environ.get("MF_THR", "64"))  # hybrid: experts with <= THR tokens run on the fused half
NE = 384
D = HIDDEN


def rec(d):
    os.makedirs(OUT, exist_ok=True)
    d = dict(d, lever=LEVER)
    with open(os.path.join(OUT, "moe_fuse.jsonl"), "a") as f:
        f.write(json.dumps(d) + "\n")
    print("MF " + json.dumps(d), flush=True)


def free(o):
    if isinstance(o, (list, tuple)):
        for t in o:
            free(t)
    elif o is not None and hasattr(o, "deallocate"):
        ttnn.deallocate(o)


def chain_us(md, fn, ks=(1, 3), n=10):
    def run(k):
        def f():
            for _ in range(k):
                free(fn())

        f()
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        f()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        t = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        r = (time.perf_counter() - t) / n * 1e6
        ttnn.release_trace(md, tid)
        return r

    return (run(ks[1]) - run(ks[0])) / (ks[1] - ks[0])


def safe(name, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:  # noqa
        print(f"MF FAIL {name}: {type(e).__name__}: {str(e)[:400]}", flush=True)
        rec(dict(kind="fail", what=name, err=str(e)[:300]))
        return None


def route_pool():
    r = torch.load(ROUTES)
    arr = np.stack([np.stack([t.numpy() for t in st]) for st in r]).astype(np.int64)  # [S, L, B, 6]
    return arr[:, LAYER].reshape(-1, TOPK)


def make_ids(kind, T, g, pool):
    if kind == "uni_rand":
        return torch.rand(T, NE, generator=g).topk(TOPK, dim=-1)[1]
    sel = torch.randint(0, pool.shape[0], (T,), generator=g)
    return torch.from_numpy(pool[sel.numpy()])


def shard(md, t, dtype, layout, dims):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=dims, mesh_shape=(ROWS, COLS)),
    )


def pad_stats(ids, N, chunk_max=32, grid_y=8):
    """host padding statistics of layout A for route ids [4N, 6] (rows r*N..): per global expert counts over ALL rows (dispatch gathers the 4
    rows of a column into the chip that owns the expert); token rows, 32-row tile padded rows, op-computed rows (chunk granular: full chunks
    of chunk_max tiles, the tail chunk rounded up to a multiple of grid_y tile rows), per chip."""
    cnt = torch.bincount(ids.reshape(-1), minlength=NE).long()
    tiles = (cnt + 31) // 32
    full = tiles // chunk_max
    tail = tiles - full * chunk_max
    comp_tiles = full * chunk_max + ((tail + grid_y - 1) // grid_y) * grid_y
    per_chip = {}
    for r in range(ROWS):
        for c in range(COLS):
            gs = [global_expert(r, c, l) for l in range(EPC)]
            per_chip[(r, c)] = (
                int(cnt[gs].sum()),
                int(tiles[gs].sum()),
                int(comp_tiles[gs].sum()),
                int((cnt[gs] > 0).sum()),
                int(tiles[gs].max()),
            )
    tok = np.array([v[0] for v in per_chip.values()])
    til = np.array([v[1] for v in per_chip.values()])
    cmp_ = np.array([v[2] for v in per_chip.values()])
    nz = np.array([v[3] for v in per_chip.values()])
    c1 = int(((cnt > 0) & (cnt <= 32)).sum())
    return dict(
        mean_expert=round(float(cnt.float().mean()), 1),
        max_expert=int(cnt.max()),
        median_expert=int(cnt.median()),
        p90_expert=int(cnt.float().quantile(0.9)),
        zero_experts=int((cnt == 0).sum()),
        le32_experts=c1,
        le64_experts=int(((cnt > 0) & (cnt <= 64)).sum()),
        tile_pad_frac=round(1 - float(cnt.sum()) / float(32 * tiles.sum()), 4),
        op_pad_frac=round(1 - float(cnt.sum()) / float(32 * comp_tiles.sum()), 4),
        single_tile_slots=int((tiles == 1).sum()),
        tiles_total=int(tiles.sum()),
        chip_tok_max_over_mean=round(float(tok.max() / tok.mean()), 3),
        chip_tiles_max_over_mean=round(float(til.max() / til.mean()), 3),
        chip_comp_tiles_max_over_mean=round(float(cmp_.max() / cmp_.mean()), 3),
        chip_tiles_mean=round(float(til.mean()), 1),
        chip_tiles_max=int(til.max()),
        chip_active_slots_mean=round(float(nz.mean()), 2),
        chip_active_slots_min=int(nz.min()),
        chip_maxexp_tiles_max=int(max(v[4] for v in per_chip.values())),
    )


def build_copy_weights(md, dtype):
    """random weights, 12 distinct buffers per projection (same on every chip), non-ring layout, timing / op-vs-op comparison only"""
    from models.demos.blackhole.deepseek_v41_flash.tt.prefill_unified_moe import INTER

    rep = ttnn.ReplicateTensorToMesh(md)
    wt = {}
    for proj, (K, N_) in (("gate", (HIDDEN, INTER)), ("up", (HIDDEN, INTER)), ("down", (INTER, HIDDEN))):
        wt[proj] = [
            ttnn.from_torch(
                (torch.randn(K, N_) * 0.02).to(torch.bfloat16),
                device=md,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_wmc(md, N_, K),
                mesh_mapper=rep,
            )
            for _ in range(EPC)
        ]
    return wt["gate"], wt["up"], wt["down"]


def torch_ref_tokens(x, ids, wts, toks, layer, limit=10.0):
    sh = mw._Shards()
    cache = {}

    def W(e):
        if e not in cache:
            p = f"layers.{layer}.ffn.experts.{e}."
            cache[e] = tuple(mw._fp4_expert(sh, p + k, torch.float32) for k in ("w1", "w3", "w2"))
        return cache[e]

    out = torch.zeros(len(toks), D)
    for i, t in enumerate(toks):
        for j in range(TOPK):
            g, u, dn = W(int(ids[t, j]))
            gg = (x[t].float() @ g).clamp(max=limit)
            uu = (x[t].float() @ u).clamp(-limit, limit)
            out[i] += float(wts[t, j]) * ((torch.nn.functional.silu(gg) * uu) @ dn)
    return out


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
def test_moe_fuse_bench(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    pool = route_pool()
    if WMODE == "ring":
        w = mw.load_moe_layer(LAYER)
        blk = DSV41MoEBlock(md, w, batch_per_device=32, gate_bias_shift=0.0)
        um = DSV41UnifiedMoE(md, LAYER, ring=(blk.decode.expert_state.tt_w0_w1, blk.decode.expert_state.tt_w2))
    else:
        um = DSV41UnifiedMoE(
            md, LAYER, weights=build_copy_weights(md, ttnn.bfloat4_b if WMODE == "copy4" else ttnn.bfloat8_b)
        )
    rec(
        dict(
            kind="start",
            env={k: v for k, v in os.environ.items() if k.startswith(("DSV41_", "MOE_COMPUTE", "DS_", "MF_"))},
            ns=NS,
            kinds=KINDS,
        )
    )
    for N in NS:
        sh = get_shared(md, N)
        T = ROWS * N
        g = torch.Generator().manual_seed(N)
        x = torch.randn(T, D, generator=g).to(torch.bfloat16)
        xr = shard(md, x.view(ROWS, N, D), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (0, None))
        for kind in KINDS:
            ids = make_ids(kind, T, g, pool)
            wts = torch.rand(T, TOPK, generator=g).to(torch.bfloat16).float()
            ps = pad_stats(ids, N)
            rec(dict(kind="padstats", N=N, routing=kind, max_buf=sh.max_buf, max_per_expert=sh.max_per_expert, **ps))
            ix = shard(md, ids.view(ROWS, N, 1, 1, TOPK).to(torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, (0, None))
            ix = ttnn.reshape(ix, [N, 1, 1, TOPK])
            sc = shard(
                md, wts.view(ROWS, N, 1, 1, TOPK).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (0, None)
            )
            sc = ttnn.reshape(sc, [N, 1, 1, TOPK])
            if ps["max_expert"] > sh.max_per_expert:
                rec(dict(kind="skip", N=N, routing=kind, why="capacity"))
                continue
            row = dict(kind="timing", N=N, routing=kind)
            # correctness: full forward output summed over the 8 columns vs torch clamp-10 reference of a few tokens
            if DO_PCC:
                part = safe("fwd", um.forward, xr, sc, ix)
                if part is not None:
                    ttnn.synchronize_device(md)
                    toks = [r * N + i for r in range(ROWS) for i in (0, 1, N // 2, N - 1)]
                    got = torch.zeros(len(toks), D)
                    for k_, t_ in enumerate(toks):
                        r, i = divmod(t_, N)
                        for c in range(COLS):
                            got[k_] += (
                                ttnn.to_torch(ttnn.get_device_tensors(part)[r * COLS + c]).float().reshape(N, D)[i]
                            )
                    ref = torch_ref_tokens(x, ids, wts, toks, LAYER)
                    row["pcc_vs_torch"] = round(float(pcc(got, ref)), 6)
                    row["pcc_min_tok"] = round(
                        min(float(pcc(got[i : i + 1], ref[i : i + 1])) for i in range(len(toks))), 6
                    )
                    row["finite"] = bool(torch.isfinite(got).all())
                    # whole-output fingerprint for lever-vs-baseline comparison (row 0 col 0..7 summed, all tokens)
                    full0 = sum(
                        ttnn.to_torch(ttnn.get_device_tensors(part)[c]).float().reshape(N, D) for c in range(COLS)
                    )
                    torch.save(full0[:: max(1, N // 256)].clone(), os.path.join(OUT, f"out_{LEVER}_{N}_{kind}.pt"))
                    free(part)
            # stages and the expert op alone
            st = {}
            for name, upto in (("route", 1), ("disp", 2), ("exp", 3), ("comb", 4), ("full", 99)) if STAGES else ():

                def f(upto=upto):
                    r_ = um.forward(xr, sc, ix, upto=upto)
                    return list(r_) if isinstance(r_, tuple) else r_

                st[name] = safe("stage_" + name, chain_us, md, f)
            prev, d = 0.0, {}
            for name in ("route", "disp", "exp", "comb", "full"):
                if st.get(name) is None:
                    continue
                d[name] = round(st[name] - prev, 1)
                prev = st[name]
            row["cum_us"] = {k: (None if v is None else round(v, 1)) for k, v in st.items()}
            row["stage_us"] = d
            # the op alone on the real dispatch buffer
            disp, meta, counts, roff = um.forward(xr, sc, ix, upto=2)
            disp2 = ttnn.reshape(disp, [sh.max_buf, D])
            gp, up_, dp = um.gate_projs, um.up_projs, um.down_projs

            ckc_f = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

            def op(counts=counts, roff=roff, kind=OPK):
                dp_ = ttnn.experimental.deepseek_prefill
                if kind == "unified":
                    return dp_.unified_routed_expert_moe(
                        disp2,
                        roff,
                        counts,
                        sh.gidx,
                        gp,
                        up_,
                        dp,
                        max_dispatched_tokens_per_expert=sh.max_per_expert,
                        compute_kernel_config=sh.ckc,
                        activation=ttnn.RoutedExpertActivation.ClampedSiluGlu,
                    )
                if kind == "fused":
                    o_ = ttnn.empty(
                        disp2.shape,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=md,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    dp_.moe_fused_swiglu(
                        disp2,
                        gp,
                        up_,
                        dp,
                        counts,
                        sh.gidx,
                        input_m_tiles=sh.max_per_expert // ttnn.TILE_SIZE,
                        core_grid=ttnn.UNIFIED_ROUTED_EXPERT_CORE_GRID,
                        compute_kernel_config=ckc_f,
                        activation=ttnn.RoutedExpertActivation.Silu,
                        output=o_,
                        expert_region_offsets=roff,
                        read_x_at_offset=True,
                    )
                    return o_
                if (
                    kind == "hyb2"
                ):  # two launches over the same counts: unified takes experts > THR tokens, moe_fused_swiglu the rest (as tt_routed_expert's hybrid path)
                    o_ = dp_.unified_routed_expert_moe(
                        disp2,
                        roff,
                        counts,
                        sh.gidx,
                        gp,
                        up_,
                        dp,
                        max_dispatched_tokens_per_expert=sh.max_per_expert,
                        compute_kernel_config=sh.ckc,
                        activation=ttnn.RoutedExpertActivation.ClampedSiluGlu,
                        min_active_tokens=THR + 1,
                    )
                    dp_.moe_fused_swiglu(
                        disp2,
                        gp,
                        up_,
                        dp,
                        counts,
                        sh.gidx,
                        input_m_tiles=sh.max_per_expert // ttnn.TILE_SIZE,
                        core_grid=ttnn.UNIFIED_ROUTED_EXPERT_CORE_GRID,
                        compute_kernel_config=ckc_f,
                        activation=ttnn.RoutedExpertActivation.Silu,
                        output=o_,
                        expert_region_offsets=roff,
                        read_x_at_offset=True,
                        max_active_tokens=THR,
                    )
                    return o_
                if kind == "hybrid":
                    return dp_.hybrid_routed_expert_moe(
                        disp2,
                        roff,
                        counts,
                        sh.gidx,
                        gp,
                        up_,
                        dp,
                        sh.max_per_expert,
                        hybrid_token_threshold=THR,
                        compute_kernel_config=ckc_f,
                        activation=ttnn.RoutedExpertActivation.Silu,
                    )
                raise ValueError(kind)

            print(
                "MF probe ops:",
                [n for n in dir(ttnn.experimental.deepseek_prefill) if "hybrid" in n or "fused" in n or "unified" in n],
                flush=True,
            )
            row["op_alone_us"] = None if os.environ.get("MF_NOTIME") == "1" else safe("op_alone", chain_us, md, op)
            row["op"] = OPK
            row["wmode"] = WMODE
            if os.environ.get("MF_PROF") == "1":
                # per-chip device kernel durations of the op (device profiler, eager, 4 launches): mean / slowest / fastest chip
                free(op())
                ttnn.synchronize_device(md)
                ttnn.ReadDeviceProfiler(md)
                for _ in range(4):
                    free(op())
                ttnn.synchronize_device(md)
                ttnn.ReadDeviceProfiler(md)
                data = ttnn.get_latest_programs_perf_data()
                per_dev = {}
                for dev_id, progs in data.items():
                    for pr in progs:
                        nm = str(getattr(pr, "program_analyses_results", pr))[:0]
                        res = pr.program_analyses_results
                        dur = None
                        for k_, v_ in res.items():
                            if "DEVICE KERNEL DURATION" in k_.upper() and "PER CORE" not in k_.upper():
                                dur = v_.duration if hasattr(v_, "duration") else v_
                                break
                        per_dev.setdefault(dev_id, []).append(dur)
                row["prof_keys"] = (
                    list(next(iter(data.values()))[-1].program_analyses_results.keys())[:12] if data else None
                )
                row["prof_ns_last_per_dev"] = {int(k_): v_[-1] for k_, v_ in per_dev.items()}
                import statistics as _st

                lasts = [v_[-1] for v_ in per_dev.values() if v_ and v_[-1] is not None]
                if lasts:
                    row["prof_us_mean_chip"] = round(_st.mean(lasts) / 1000.0, 1)
                    row["prof_us_max_chip"] = round(max(lasts) / 1000.0, 1)
                    row["prof_us_min_chip"] = round(min(lasts) / 1000.0, 1)
            # op output vs the unified op on the same inputs (valid rows only), for fused / hybrid
            if OPK != "unified" and os.environ.get("MF_CMP", "1") == "1":
                oa = safe("cmp_ref", op, kind="unified")
                ob = safe("cmp_x", op)
                if oa is not None and ob is not None:
                    ttnn.synchronize_device(md)
                    worst = 1.0
                    for dev_i in (0, 9, 18, 27):
                        c_ = ttnn.to_torch(ttnn.get_device_tensors(counts)[dev_i]).reshape(-1).long()
                        r_ = ttnn.to_torch(ttnn.get_device_tensors(roff)[dev_i]).reshape(-1).long()
                        A_ = ttnn.to_torch(ttnn.get_device_tensors(oa)[dev_i]).float().reshape(-1, D)
                        B_ = ttnn.to_torch(ttnn.get_device_tensors(ob)[dev_i]).float().reshape(-1, D)
                        rr, cc_ = divmod(dev_i, COLS)
                        for l in range(EPC):
                            g_ = global_expert(rr, cc_, l)
                            n_ = int(c_[g_])
                            if n_ > 0:
                                worst = min(
                                    worst,
                                    float(pcc(A_[int(r_[g_]) : int(r_[g_]) + n_], B_[int(r_[g_]) : int(r_[g_]) + n_])),
                                )
                    row["pcc_op_vs_unified"] = round(worst, 6)
                free([oa, ob])
            rec(row)
            if SYN and kind == KINDS[0] and os.environ.get("MF_NOTIME") != "1":
                # synthetic uniform counts: every global expert gets c tokens (offsets = slot * pad32(c) inside the chip buffer)
                c0 = ttnn.to_torch(ttnn.get_device_tensors(counts)[0])
                r0 = ttnn.to_torch(ttnn.get_device_tensors(roff)[0])
                rec(
                    dict(
                        kind="fmt",
                        counts_shape=list(c0.shape),
                        counts_dtype=str(c0.dtype),
                        roff_shape=list(r0.shape),
                        counts_head=c0.reshape(-1)[:20].tolist(),
                        roff_head=r0.reshape(-1)[:20].tolist(),
                        roff_dtype=str(r0.dtype),
                    )
                )
                for c in SYN:
                    cs, os_ = [], []
                    for r in range(ROWS):
                        for cc in range(COLS):
                            cv = torch.zeros_like(c0.reshape(-1)).long()
                            ov = torch.zeros_like(r0.reshape(-1)).long()
                            for l in range(EPC):
                                gid = global_expert(r, cc, l)
                                cv[gid] = c
                                ov[gid] = l * ((c + 31) // 32) * 32
                            cs.append(cv.reshape(c0.shape))
                            os_.append(ov.reshape(r0.shape))
                    mp = ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(ROWS, COLS))
                    mk = lambda lst, ref: ttnn.from_torch(
                        torch.stack(lst).reshape(ROWS, COLS, *lst[0].shape).to(ref.dtype),
                        device=md,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=mp,
                    )
                    sc_t, so_t = mk(cs, c0), mk(os_, r0)
                    sc_t = ttnn.reshape(sc_t, list(c0.shape))
                    so_t = ttnn.reshape(so_t, list(r0.shape))
                    t_ = safe(f"syn{c}", chain_us, md, lambda: op(sc_t, so_t))
                    rec(dict(kind="syn", N=N, tokens_per_expert=c, tiles=(c + 31) // 32, op_us=t_))
                    free([sc_t, so_t])
            free([disp, meta, counts, roff])
            free([ix, sc])
        free(xr)

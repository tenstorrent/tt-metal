# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The MoE layer around the lane stream (ws_layer.py): head, shared expert (ws_shared.py), routed stream (row-major
outputs into the result buffer) and end pass, every chip on its own tokens. Checks against torch on the device's
routing: the head's counts, token lists and result-row lists exactly, the logits / gate weights / top-K choice, the
result buffer's routed and shared rows, and the layer output (every token, or WSY_TOKENS sampled tokens on the
WSY_CHIPS chips). Timing: WS_WARM warm-up + WS_MEAS measured layers, per phase wall time (synchronize to
synchronize; programs built before their phase starts) and the custom programs' device spans (compute start to end
stamps)."""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert import bench_util
from models.demos.ws_expert.bench_util import aiclk_mhz, pcc, quantize, zero_pending
from models.demos.ws_expert.tests.test_ws_fab_bench import TORUS
from models.demos.ws_expert.tests.test_ws_lane import first_visible
from models.demos.ws_expert.ws_end import build_end, end_span_us
from models.demos.ws_expert.ws_fab import slice_layout
from models.demos.ws_expert.ws_group import GCfg
from models.demos.ws_expert.ws_lane import LCfg, build_lane, make_tables, tables_tensor
from models.demos.ws_expert.ws_lane5 import L5Cfg, build_lane5, slice_layout5
from models.demos.ws_expert.ws_layer import (
    LayerCfg,
    gcfg,
    glx_lcfg,
    head,
    head_consts,
    proxy_lcfg,
    result_buffer,
    router_weights,
)
from models.demos.ws_expert.ws_sf import torus_cycles
from models.demos.ws_expert.ws_shared import SCfg, build_shared, shared_span_us, shared_weights

# WSY_HEAD: "router": one program, the MoE block router's structure (ws_router); "custom": router matmul + the routing
# program (ws_head); "ttnn": ttnn ops
HEAD = os.environ.get("WSY_HEAD", "ttnn")
from models.demos.ws_expert.ws_head import (
    build_route,
    clock_programs,
    clock_stamps,
    dram_clock_ticks,
    dram_clocks,
    head_outputs,
    route_span_us,
    router_logits,
    top_from_lists,
)
from models.demos.ws_expert.ws_router import build_router, logits_debug_tensor, router_inputs, router_span_us
from models.demos.ws_expert.ws_fold import build_fold, fold_span_us

LOGITS = os.environ.get("WSY_LOGITS", "0") == "1"  # (router head: its logits written for the checks)
FOLD = os.environ.get("WSY_FOLD", "0") == "1"  # (the shared expert folded into the end pass: ws_fold)
KS = os.environ.get("WSY_KS", "0") == "1"  # (the routed stream as the K-split lane: ws_layer.KS_LANE)
# WSY_L5=1: the routed stream as the 5 x 5 K-split lane with the corner-hub h multicast (ws_lane5: hmc 2, muxes on
# NOC0, row-major result rows); WSY_L5CFG "k=v;..." overrides L5Cfg fields
L5 = os.environ.get("WSY_L5", "0") == "1"
# WSY_SIDE=1 (with WSY_L5): the head places each routed row in a DRAM bank on the side of its expert's lane group
# (ws_router side=G), so the 5 x 5 groups' row writes stay on their half of the chip
SIDE = os.environ.get("WSY_SIDE", "0") == "1"


def router_side(lc):
    return lc.G if (L5 and SIDE) else 0


def l5_cfgs(lc):
    """(GCfg, L5Cfg) of the 5 x 5 lane in the layer (tests/test_ws_lane5.py's configs, row-major outputs)."""
    rcfg = GCfg(nexp=lc.nexp, G=lc.G, T=lc.T, MMAX=lc.T, GC=25, h_dt=ttnn.bfloat8_b, XRM=2, NS=5, dn_lag=3)
    kw = dict(hmc=2, mux_noc=0, yrm=1)
    for item in filter(None, os.environ.get("WSY_L5CFG", "").split(";")):
        k, v = item.split("=")
        kw[k] = int(v)
    return rcfg, L5Cfg(**kw)


def l5_lcfg(cycles):
    """The 5 x 5 lane's LCfg on the torus (tests/test_ws_lane5.py's LANE_DEFAULT)."""
    return LCfg(cycles=cycles, CHT=20, slots="d0,d1,d0d1,any", keep=2, sched_c=7, wpoll=1)


def routed_glx(mesh):
    cyc = torus_cycles(*tuple(mesh.shape))
    return l5_lcfg(cyc) if L5 else glx_lcfg(cyc, ks=KS)


def stream_program(mesh, lc, st, hd):
    """The routed stream's program (mesh program, io, stamps) on the head's outputs hd, into the result buffer."""
    if st.get("rl5") is not None:
        return build_lane5(
            mesh,
            st["rcfg"],
            st["rlc"],
            st["rl5"],
            hd["x_rm"],
            hd["idx_t"],
            hd["counts_t"],
            st["ws_t"],
            st["res_t"],
            st["tab_t"],
            st["rtb"],
            0,
            dest_t=hd["dest_t"],
            fence_row=lc.T * lc.S,
        )
    return build_lane(
        mesh,
        st["rcfg"],
        st["rlc"],
        hd["x_rm"],
        hd["idx_t"],
        hd["counts_t"],
        st["ws_t"],
        st["res_t"],
        st["tab_t"],
        st["rtb"],
        0,
        dest_t=hd["dest_t"],
        fence_row=lc.T * lc.S,
    )

WARM = int(os.environ.get("WS_WARM", "20"))
MEAS = int(os.environ.get("WS_MEAS", "5"))


def lane_span_us(st, ci, clk):
    """A lane program's device time on chip ci: earliest compute start to latest compute end over the group cores."""
    s = ttnn.to_torch(ttnn.get_device_tensors(st)[ci]).to(torch.int64) & 0xFFFFFFFF
    s = s.reshape(-1, 8 * 64)
    rel = lambda v: ((v - int(s[0, 0]) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    return (int(rel(s[:, 63]).max()) - int(rel(s[:, 0]).min())) / clk


def make_layer(mesh, lc: LayerCfg, routed_lc: LCfg, seed=0):
    """Weights (bf16 draws) and per-chip tokens; device state of every phase."""
    N = mesh.get_num_devices()
    shape = tuple(mesh.shape)
    torch.manual_seed(seed)
    H, I, E = lc.H, lc.I, lc.nexp
    wr = (torch.randn(H, E) / H**0.5).to(torch.bfloat16)
    wg = torch.randn(E, H, I, dtype=torch.bfloat16) / H**0.5
    wu = torch.randn(E, H, I, dtype=torch.bfloat16) / H**0.5
    wd = torch.randn(E, I, H, dtype=torch.bfloat16) / I**0.5
    w1 = (torch.randn(H, lc.IS) / H**0.5).to(torch.bfloat16)
    w3 = (torch.randn(H, lc.IS) / H**0.5).to(torch.bfloat16)
    w2 = (torch.randn(lc.IS, H) / lc.IS**0.5).to(torch.bfloat16)
    scfg = SCfg(T=lc.T, H=H, IS=lc.IS, S=lc.S, KSH=lc.K)
    x = torch.zeros(N, lc.TX, H, dtype=torch.bfloat16)
    for c in range(N):
        x[c, : lc.T] = torch.randn(lc.T, H).to(torch.bfloat16)
    # (WSY_LOGIT_SHIFT: every logit shifted by it - x's column 0 one, Wr's row 0 the shift; WSY_SKEW: the first n
    # experts' logits + 50 on top, so every token routes to them)
    shift, skew = float(os.environ.get("WSY_LOGIT_SHIFT", "0")), int(os.environ.get("WSY_SKEW", "0"))
    if shift or skew:
        x[:, : lc.T, 0] = 1.0
        wr[0, :] = shift
        wr[0, :skew] = shift + 50.0
    rl5 = None
    if L5:
        rcfg, rl5 = l5_cfgs(lc)
    else:
        rcfg = gcfg(E, lc.G, lc.T, ks=routed_lc.ks)
    rtb = make_tables(shape, rcfg, routed_lc)
    if L5:  # (each chip's slices in bf16 before the concatenation: host memory)
        per_chip = [
            slice_layout5(rcfg, routed_lc.CHT, wg[o].float(), wu[o].float(), wd[o].float()).to(torch.bfloat16)
            for o in rtb["own"]
        ]
    else:
        per_chip = [
            slice_layout(rcfg, routed_lc.CHT, wg[o].float(), wu[o].float(), wd[o].float(), ks=routed_lc.ks)
            for o in rtb["own"]
        ]
    shard = ttnn.ShardTensorToMesh(mesh, dim=0)
    dram = ttnn.DRAM_MEMORY_CONFIG
    st = dict(
        x_t=ttnn.from_torch(
            x.reshape(N * lc.TX, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard
        ),
        wr_t=router_weights(mesh, wr),
        hc=head_consts(mesh, lc),
        rcfg=rcfg,
        rlc=routed_lc,
        rl5=rl5,
        rtb=rtb,
        tab_t=tables_tensor(mesh, rtb["tabs"]),
        ws_t=ttnn.from_torch(
            torch.cat(per_chip, 0),
            dtype=rcfg.w_dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=dram,
            mesh_mapper=shard,
        ),
        res_t=result_buffer(mesh, lc),
        y_t=ttnn.empty([lc.T, H], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=dram),
    )
    del per_chip
    st["scfg"] = scfg
    st["w13_t"], st["w2s_t"] = shared_weights(mesh, scfg, w1, w3, w2)
    if HEAD == "custom":
        st["hout"] = head_outputs(mesh, lc)
        st["logits_t"] = ttnn.empty(
            [lc.TX, E], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram
        )
    if N > 1:  # (layer_programs' "barrier": a tiny tensor a chip, gathered along each torus axis)
        st["bar_t"] = ttnn.from_torch(
            torch.zeros(N, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard
        )
        device_barrier(st)  # (its semaphores in place before any stream is built)
    if HEAD == "router":  # (x arrives row-major, as the previous layer's end pass writes it)
        st["hout"] = head_outputs(mesh, lc)
        st["x_rm"] = ttnn.from_torch(
            x.reshape(N * lc.TX, H), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=shard
        )
        st["wrT_t"], st["ids_t"] = router_inputs(mesh, wr)
        st["logd_t"] = logits_debug_tensor(mesh, lc) if LOGITS else None
    return st, dict(x=x, wr=wr, wg=wg, wu=wu, wd=wd, w1=w1, w3=w3, w2=w2)


def layer_step(mesh, lc: LayerCfg, st, clk, span_chips=(), detail=False, trace=False):
    """One layer: (head outputs, {phase: wall ms}, {phase: [device ms of span_chips]}); detail: the head's steps apart
    (separate keys; the ttnn head synchronized apart, the custom head by wall-clock stamps between its programs);
    trace: log each phase's launch and end."""
    sync = lambda: ttnn.synchronize_device(mesh)
    ph, dev = {}, {}
    sync()
    marks = [] if detail else None
    if HEAD == "custom":
        rmpd, rio, rst = build_route(mesh, lc, st["logits_t"], st["hout"])
        if detail:  # (wall-clock stamps between the head's programs: their device times, no syncs in between)
            cps, cst = clock_programs(mesh, 4)
        sync()
    if HEAD == "router":
        rmpd, rio, rst = build_router(
            mesh, lc, st["x_rm"], st["wrT_t"], st["ids_t"], st["hout"], st["logd_t"], side=router_side(lc)
        )
        sync()
    t0 = time.perf_counter()
    if trace:
        logger.info("WSY phase head: launch")
    if HEAD == "custom":
        tick = (lambda i: ttnn.generic_op([cst, cst], cps[i])) if detail else (lambda i: None)
        tick(0)
        router_logits(mesh, st["x_t"], st["wr_t"], out=st["logits_t"])
        tick(1)
        x_rm = ttnn.to_layout(st["x_t"], ttnn.ROW_MAJOR_LAYOUT)
        tick(2)
        ttnn.generic_op(rio, rmpd)
        tick(3)
        hd = dict(x_rm=x_rm, logits=st["logits_t"], **{k: st["hout"][k] for k in ("idx_t", "dest_t", "counts_t", "w")})
    elif HEAD == "router":
        ttnn.generic_op(rio, rmpd)
        hd = dict(x_rm=st["x_rm"], **{k: st["hout"][k] for k in ("idx_t", "dest_t", "counts_t", "w")})
        if st["logd_t"] is not None:
            hd["logits_T"] = st["logd_t"]
    else:
        hd = head(mesh, lc, st["x_t"], st["wr_t"], st["hc"], marks)
    sync()
    ph["head"] = (time.perf_counter() - t0) * 1e3
    if HEAD == "router":
        for ci in span_chips:
            r = router_span_us(rst, ci, clk)
            dev.setdefault("head", []).append(r["end"] / 1e3)
            if detail:
                for k, v in r.items():
                    if k != "end":
                        dev.setdefault(f"head_{k}", []).append(v / 1e3)
        for t in rio:
            if t.memory_config().buffer_type == ttnn.BufferType.L1:
                ttnn.deallocate(t)
        del rmpd, rio, rst
        for d in ttnn.get_device_tensors(hd["counts_t"]):
            if bool((ttnn.to_torch(d).to(torch.int64) & 0xFFFFFFFF == 0xFFFFFFFF).any()):
                raise RuntimeError("routing program: an owner core's records overflowed its capacity")
    if HEAD == "custom":
        if detail:
            rel = lambda v, base: (((v - base + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)) / clk / 1e3
            for ci in span_chips:
                a, b, c, d = clock_stamps(cst, ci, 4)
                r, r0 = route_span_us(rst, ci, clk)
                steps = dict(
                    head_matmul=rel(b, a),
                    head_x_rm=rel(c, b),
                    head_gap_in=rel(r0, c),
                    head_routing=r["end"] / 1e3,
                    head_gap_out=rel(d, r0) - r["end"] / 1e3,
                    head_device=rel(d, a),
                    **{f"route_{k}": v / 1e3 for k, v in r.items() if k != "end"},
                )
                for k, v in steps.items():
                    dev.setdefault(k, []).append(v)
            ttnn.deallocate(cst)
            del cps, cst
        bench_util.l1_log("ran head (eager)")
        for t in rio:
            if t.memory_config().buffer_type == ttnn.BufferType.L1:
                bench_util.l1_log(f"free {t.buffer_address()}")
                ttnn.deallocate(t)
        del rmpd, rio, rst
        # (outside the timed head: an owner core with more than its record capacity writes counts 0xFFFFFFFF; the
        # stream must not run on them)
        for d in ttnn.get_device_tensors(hd["counts_t"]):
            if bool((ttnn.to_torch(d).to(torch.int64) & 0xFFFFFFFF == 0xFFFFFFFF).any()):
                raise RuntimeError("routing program: an owner core's records overflowed its capacity")
    if trace:
        logger.info("WSY phase head: done")
    if detail:
        for (_, a), (n, b) in zip(marks, marks[1:]):
            dev[f"head_{n}"] = [(b - a) * 1e3]
    progs = (
        (
            "shared",
            lambda: build_shared(mesh, st["scfg"], hd["x_rm"], st["w13_t"], st["w2s_t"], st["res_t"])[:3],
            shared_span_us,
        ),
        (
            "stream",
            lambda: stream_program(mesh, lc, st, hd),
            lane_span_us,
        ),
        ("end", lambda: build_end(mesh, lc.T, lc.K, lc.H, st["res_t"], hd["w"], st["y_t"]), end_span_us),
    )
    if lc.fold:  # (the shared expert folded into the end pass: after the stream)
        progs = (progs[1], ("fold", lambda: fold_program(mesh, lc, st, hd), fold_span_us))
    for name, build, span in progs:
        if name in os.environ.get("WSY_SKIP", "").split(","):
            continue
        mpd, io, stamps = build()
        sync()
        if trace:
            logger.info(f"WSY phase {name}: launch")
        t0 = time.perf_counter()
        ttnn.generic_op(io, mpd)
        sync()
        if trace:
            logger.info(f"WSY phase {name}: done")
        ph[name] = (time.perf_counter() - t0) * 1e3
        if span_chips:
            dev[name] = [span(stamps, ci, clk) / 1e3 for ci in span_chips]
        bench_util.l1_log(f"ran {name} (eager)")
        for t in io:  # (the program's L1 tensors freed now: the next phase's ops place their CBs below them)
            if t.memory_config().buffer_type == ttnn.BufferType.L1:
                bench_util.l1_log(f"free {t.buffer_address()}")
                ttnn.deallocate(t)
        del mpd, io, stamps
    return hd, ph, dev


class Ref:
    """torch reference on the device's routing, quantized as the device sees the operands (bfp4 weights, bfp8 x rows,
    the routed experts' bf16 h and the shared expert's h format)."""

    def __init__(self, lc: LayerCfg, host, cfg, scfg):
        self.lc, self.h, self.cfg, self.scfg, self.cache = lc, host, cfg, scfg, {}
        q = lambda w: quantize(w.float(), scfg.w_dt)
        self.shw = (q(host["w1"]), q(host["w3"]), q(host["w2"]))

    def shared(self, c, toks):
        """The shared expert's rows [n, H] of chip c's tokens toks."""
        X = quantize(self.h["x"][c, toks].float(), self.scfg.x_dt)
        a, b, d = self.shw
        return quantize(torch.nn.functional.silu(X @ a) * (X @ b), self.scfg.h_dt) @ d

    def expert(self, e):
        if e not in self.cache:
            h, dt = self.h, self.cfg.w_dt
            self.cache[e] = (
                quantize(h["wg"][e].float(), dt),
                quantize(h["wu"][e].float(), dt),
                quantize(h["wd"][e].float(), dt),
            )
        return self.cache[e]

    def swiglu(self, X, a, b, d):
        return quantize(torch.nn.functional.silu(X @ a) * (X @ b), self.cfg.h_dt) @ d

    def routed(self, c, toks, top):
        """Expert outputs [n, K, H] of chip c's tokens toks with the device's top-K of those tokens ([n, K])."""
        X = quantize(self.h["x"][c, toks].float(), self.cfg.x_dt)
        out = torch.zeros(len(toks), self.lc.K, self.lc.H)
        for e in top.unique().tolist():
            j, k = torch.nonzero(top == e, as_tuple=True)
            out[j, k] = self.swiglu(X[j], *self.expert(e))
        return out


def check_chip(lc: LayerCfg, ref: Ref, c, hd, st, ntok, tag):
    """Head exactness (counts, token lists, result-row lists), routing vs torch, the result buffer's routed rows and
    the layer output on chip c."""
    T, E, K, S = lc.T, lc.nexp, lc.K, lc.S
    get = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[c])
    w = get(hd["w"]).float().reshape(-1, 16)[:T, :K]
    counts = get(hd["counts_t"]).long().reshape(-1)
    idx = get(hd["idx_t"]).long().reshape(E, lc.RS)
    dest = get(hd["dest_t"]).long().reshape(E, lc.RS)
    if "top" in hd:
        top = get(hd["top"]).reshape(-1, K)[:T].long()
    else:  # (the routing program's choice, recovered from its destination lists)
        top = top_from_lists(lc, idx, dest, counts.clamp(max=lc.RS))
        assert top is not None, "the destination lists do not hold every (token, slot) exactly once"
    cnt_ref = torch.bincount(top.flatten(), minlength=E)
    ok_cnt = bool((counts == cnt_ref).all())
    tk = torch.arange(T).repeat_interleave(K)
    order = torch.argsort(top.flatten() * T + tk, stable=True)
    off = torch.cumsum(cnt_ref, 0) - cnt_ref
    ok_idx = ok_dest = True
    for e in range(E):
        # (an expert's entries as a set: the router head's sources hold interleaved row tiles; each entry's token
        # matches its result row)
        n = int(cnt_ref[e])
        sel = order[off[e] : off[e] + n]
        mp = -(-n // 32) * 32  # (the entries the stream reads: whole row tiles, padded with T)
        d = dest[e, :n]
        ok_idx &= bool((idx[e, :n] == d // S).all()) and bool((idx[e, n:mp] == T).all())
        ok_dest &= bool((torch.sort(d).values == torch.sort(tk[sel] * S + sel % K).values).all())
    toks = torch.arange(T) if ntok <= 0 or ntok >= T else torch.linspace(0, T - 1, ntok).round().long()
    if "logits" in hd:
        lg_dev = get(hd["logits"]).float().reshape(-1, E)[toks]
    elif "logits_T" in hd:
        lg_dev = get(hd["logits_T"]).float().reshape(E, -1).t()[toks].contiguous()
    else:
        lg_dev = None
    lg_ref = ref.h["x"][c, toks].float() @ ref.h["wr"].float()
    sc_ref = torch.softmax(lg_ref, -1) if lc.score == "softmax" else torch.sigmoid(lg_ref)
    kth = sc_ref.topk(K, -1).values[:, -1:]
    picked = sc_ref.gather(1, top[toks])
    same = int((torch.sort(top[toks], 1).values == torch.sort(sc_ref.topk(K, -1).indices, 1).values).all(1).sum())
    worst_pick = float((picked - kth).min() / kth.mean())  # < 0: a pick below torch's K-th score (near ties)
    # the device's picks against a top-K of its own fp32 logits: exact compares (custom head), or of its keys (router
    # head: the logit with its low mantissa bits replaced by the expert id, compared sign-magnitude)
    same_dev = None
    if lg_dev is not None:
        if HEAD == "router":
            idm = 0xFF if E <= 256 else 0x1FF if E <= 512 else 0x3FF
            key = (lg_dev.view(torch.int32) & ~idm) | torch.arange(E, dtype=torch.int32)
            dev_top = (key ^ ((key >> 31) & 0x7FFFFFFF)).long().topk(K, -1).indices
        else:
            dev_top = lg_dev.topk(K, -1).indices
        same_dev = int((torch.sort(top[toks], 1).values == torch.sort(dev_top, 1).values).all(1).sum())
        if HEAD in ("custom", "router"):
            assert same_dev == len(toks), f"the head's picks differ from a top-{K} of its logits"
    w_ref = picked / picked.sum(1, keepdim=True)
    eo = ref.routed(c, toks, top[toks])  # [n, K, H]
    res = get(st["res_t"]).float()[: T * S].reshape(T, S, lc.H)[toks]  # (fold: S = K, no shared rows)
    yd = get(st["y_t"]).float().reshape(T, lc.H)[toks]
    sh = ref.shared(c, toks)
    yr = (w[toks].unsqueeze(2) * eo).sum(1) + sh
    full = {}
    if os.environ.get("WSY_FULLCHECK", "0") == "1" and not lc.fold:
        # every token: the result buffer read twice around a device sync (stable), and y against the end pass's sum
        # over the rows as read (bf16 gate weights, fp32 sums)
        ttnn.synchronize_device(st["res_t"].device())
        r2 = get(st["res_t"])[: T * S].reshape(T, S, lc.H)
        allr = get(st["res_t"])[: T * S].reshape(T, S, lc.H)
        yall = get(st["y_t"]).float().reshape(T, lc.H)
        reread = bool(torch.equal(allr, r2))
        wq = w.to(torch.bfloat16).float()
        worst, bad = 0.0, 0
        for a in range(0, T, 2048):
            b = min(T, a + 2048)
            ys = torch.einsum("tk,tkh->th", wq[a:b], allr[a:b, :K].float()) + allr[a:b, K].float()
            e = (yall[a:b] - ys).abs().amax(1) / ys.abs().amax(1).clamp_min(1e-6)
            worst = max(worst, float(e.max()))
            bad += int((e > 2e-2).sum())
        full = dict(reread_equal=reread, y_vs_rows_worst_rel=round(worst, 5), y_vs_rows_bad_tokens=bad)
        del r2, allr, yall
    out = dict(
        counts=ok_cnt,
        lists=ok_idx,
        dests=ok_dest,
        logits_pcc=pcc(lg_dev, lg_ref) if lg_dev is not None else None,
        w_pcc=pcc(w[toks], w_ref),
        same_topk=f"{same}/{len(toks)}",
        topk_of_device_logits=f"{same_dev}/{len(toks)}" if same_dev is not None else None,
        worst_pick=round(worst_pick, 5),
        rows_pcc=pcc(res[:, :K], eo),
        shared_rows_pcc=pcc(res[:, K], sh) if not lc.fold else None,
        y_pcc=pcc(yd, yr),
        y_max_rel=float((yd - yr).abs().max() / yr.abs().max()),
        **full,
    )
    logger.info(f"WSY {tag} chip {c} check ({len(toks)} tokens): {out}")
    return out, counts


def run_layer(mesh, lc: LayerCfg, routed_lc: LCfg, tag, chips, ntok, warm, meas):
    N = mesh.get_num_devices()
    st, host = make_layer(mesh, lc, routed_lc)
    clk_dev = first_visible()
    clk = aiclk_mhz(clk_dev)
    logger.info(f"WSY {tag}: {N} chips, {lc}, routed {routed_lc}, aiclk {clk} MHz")
    span_chips = sorted({0, N - 1})
    rows, drows = [], []
    nd = int(os.environ.get("WSY_HEAD_DETAIL", "0"))  # extra layers after the measured ones: head steps apart
    for i in range(warm + meas + nd):
        hd, ph, dev = layer_step(mesh, lc, st, clk, span_chips if i >= warm else (), i >= warm + meas, trace=i < 2)
        if warm <= i < warm + meas:
            rows.append((ph, dev))
            logger.info(f"WSY {tag} layer {i - warm + 1}/{meas}: wall ms {ph}, device ms {dev}")
        elif i >= warm + meas:
            drows.append(dict(dev, head_wall=[ph["head"]]))
        if i < warm + meas + nd - 1:
            del hd
    if drows:
        hs = {k: round(statistics.median(max(r[k]) for r in drows), 3) for k in drows[0]}
        logger.info(f"WSY {tag} SUMMARY head steps ms (median of {nd} layers, max over chips {span_chips}): {hs}")
    ref = Ref(lc, host, st["rcfg"], st["scfg"])
    res = [check_chip(lc, ref, c, hd, st, ntok, tag)[0] for c in chips]
    M = torch.stack([ttnn.to_torch(ttnn.get_device_tensors(hd["counts_t"])[c]).long().reshape(-1) for c in range(N)])
    logger.info(
        f"WSY {tag} routed M over {N} chip(s) x {lc.nexp} experts: min {int(M.min())}, median "
        f"{float(M.float().median()):.0f}, max {int(M.max())}, mean {float(M.float().mean()):.1f}"
    )
    if meas:
        med = {k: statistics.median(r[0][k] for r in rows) for k in rows[0][0]}
        devm = {k: statistics.median(max(r[1][k]) for r in rows) for k in rows[0][1]}
        out_gb = lc.T * lc.K * lc.H * 2 / 1e9
        logger.info(
            f"WSY {tag} SUMMARY: {warm} warm-up + {meas} measured layers, aiclk {clk} MHz (chip {clk_dev}, "
            f"{aiclk_mhz(clk_dev)} at the end); wall ms median {', '.join(f'{k} {v:.2f}' for k, v in med.items())}, "
            f"total {sum(med.values()):.2f}; device ms (max over chips {span_chips}, median) "
            f"{', '.join(f'{k} {v:.2f}' for k, v in devm.items())}; stream output writes {out_gb:.2f} GB a chip, "
            f"{out_gb / devm['stream'] * 1e3:.0f} GB/s"
        )
    ok = all(r["counts"] and r["lists"] and r["dests"] for r in res)
    logger.info(
        f"WSY {tag} RESULT: min y pcc {min(r['y_pcc'] for r in res):.5f}, min rows pcc "
        f"{min(r['rows_pcc'] for r in res):.5f}, head exact {ok}"
    )
    return res


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize(
    "nexp,T",
    [(64, 2048), (640, 2048), (640, 31250), (640, 8192)],
    ids=["e64t2048", "e640t2048", "e640t31250", "e640t8192"],
)
def test_ws_layer_chip(mesh_device, nexp, T):
    """One chip (TT_VISIBLE_DEVICES): the routed stream as the one-chip proxy (every expert the chip's own)."""
    lc = LayerCfg(T=T, nexp=nexp, fold=FOLD)
    small = T <= 4096
    warm = int(os.environ.get("WSY_WARM", 1 if small else WARM))
    meas = int(os.environ.get("WSY_MEAS", 1 if small else MEAS))
    ntok = int(os.environ.get("WSY_TOKENS", 0 if small else 256))
    rlc = proxy_lcfg(
        lc.G,
        ks=KS,
        y_nowrite=os.environ.get("WSY_NOWRITE", "0") == "1",
        y_posted=os.environ.get("WSY_POSTED", "1") == "1",
    )
    res = run_layer(mesh_device, lc, rlc, f"chip e{nexp} T {T}", [0], ntok, warm, meas)
    assert all(r["counts"] and r["lists"] and r["dests"] for r in res)
    assert min(r["y_pcc"] for r in res) > 0.999 and min(r["rows_pcc"] for r in res) > 0.999
    if not lc.fold:
        assert min(r["shared_rows_pcc"] for r in res) > 0.999


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TORUS], indirect=True)
def test_ws_layer_glx(mesh_device):
    """The whole Galaxy torus: T 31250 tokens a chip, 640 experts streamed over 2 cycles x 2 links (G 4)."""
    lc = LayerCfg(T=31250, nexp=640, fold=FOLD)
    rlc = glx_lcfg(torus_cycles(*tuple(mesh_device.shape)), ks=KS)
    chips = [int(c) for c in os.environ.get("WSY_CHIPS", "0,31").split(",")]
    res = run_layer(
        mesh_device,
        lc,
        rlc,
        "glx e640 T 31250",
        chips,
        int(os.environ.get("WSY_TOKENS", 64)),
        int(os.environ.get("WSY_WARM", WARM)),
        int(os.environ.get("WSY_MEAS", MEAS)),
    )
    assert all(r["counts"] and r["lists"] and r["dests"] for r in res)
    assert min(r["y_pcc"] for r in res) > 0.999 and min(r["rows_pcc"] for r in res) > 0.999


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_ws_head_overflow(mesh_device, monkeypatch):
    """The custom head with every token routed to owner 0's experts (6 x 8192 records against its 16K capacity):
    the overflow shows in the counts and the layer stops before the stream."""
    if HEAD not in ("custom", "router"):
        pytest.skip("WSY_HEAD=custom or router only")
    monkeypatch.setenv("WSY_SKEW", "6")
    lc = LayerCfg(T=8192, nexp=640)
    st, _ = make_layer(mesh_device, lc, proxy_lcfg(lc.G, ks=KS))
    with pytest.raises(RuntimeError, match="overflowed"):
        layer_step(mesh_device, lc, st, aiclk_mhz(first_visible()))
    logger.info("WSY head overflow: caught on the host before the stream")


TRACE_REGION = 64 << 20


PHASES = ("head", "shared", "stream", "end")
FOLD_PHASES = ("head", "stream", "fold")


def device_barrier(st):
    """Every chip past this point: two all-gathers of a tiny DRAM tensor, one a torus axis, the second on the first's
    output. Their first build allocates their global semaphores in L1 (kept with the cached programs): make_layer runs
    it once, so no stream is built before and after that allocation."""
    g0 = ttnn.all_gather(st["bar_t"], 3, cluster_axis=0, num_links=1, topology=ttnn.Topology.Ring)
    g1 = ttnn.all_gather(g0, 3, cluster_axis=1, num_links=1, topology=ttnn.Topology.Ring)
    ttnn.deallocate(g0)
    ttnn.deallocate(g1)


def layer_programs(mesh, lc: LayerCfg, st, phases=None, diag=None, clk=None, clocks=None):
    """One layer's programs (`phases` of them, in layer order) back to back, with no syncs and no host reads
    (capturable in a trace; bench_util.DEVICE_ZERO set, so the L1 tensors' zeros come from a device program): each
    phase builds its program, zeroes its new L1 tensors, runs and frees them, so the phases' L1 tensors sit where the
    eager layer's do (the routed stream's leave it little slack: nothing may stay in L1 across phases). Returns the
    head's outputs. diag (a dict, untraced only): each zero program and phase synchronized apart - wall ms, the bytes a
    core zeroes, the phase's device ms (chip 0) - into it. clocks ((programs, io), ws_head.dram_clocks): a clock
    program before the first phase and after each (the phases' times, dispatch included)."""
    phases = phases or (FOLD_PHASES if lc.fold else PHASES)
    ticks = iter(clocks[0]) if clocks else None
    tick = (lambda: ttnn.generic_op(clocks[1], next(ticks))) if clocks else (lambda: None)
    tick()
    spans = dict(head=lambda t: router_span_us(t, 0, clk)["end"], shared=lambda t: shared_span_us(t, 0, clk))
    spans.update(stream=lambda t: lane_span_us(t, 0, clk), end=lambda t: end_span_us(t, 0, clk))
    spans.update(fold=lambda t: fold_span_us(t, 0, clk))

    held = []  # (the stream's state tensors, kept across the barrier)
    zeroed = []  # (their addresses: the rebuilt stream's state words must be exactly these)
    is_state = lambda t: t.memory_config().buffer_type == ttnn.BufferType.L1 and t.dtype == ttnn.uint32

    def run(name, mpd, io, stamps, zero=True, launch=True):
        if name == "stream" and not zero and zeroed:  # ("stream:run" after "stream:zero")
            addrs = [t.buffer_address() for t in io if is_state(t)]
            assert addrs == zeroed, f"the rebuilt stream's state words moved: zeroed at {zeroed}, built at {addrs}"
        bench_util.l1_log(f"run {name} (traced layer)")
        if diag is not None:
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
        nz = zero_pending() if zero else bench_util.drop_pending()
        if diag is not None:
            ttnn.synchronize_device(mesh)
            t1 = time.perf_counter()
            diag[f"zero_{name}"] = dict(ms=round((t1 - t0) * 1e3, 3), bytes=nz)
        if launch:
            ttnn.generic_op(io, mpd)
        if diag is not None and launch:
            ttnn.synchronize_device(mesh)
            diag[name] = dict(wall=round((time.perf_counter() - t1) * 1e3, 3), dev=round(spans[name](stamps) / 1e3, 3))
        if launch:
            tick()
        for t in io:
            if not launch and t.memory_config().buffer_type == ttnn.BufferType.L1 and t.dtype == ttnn.uint32:
                held.append(t)  # (the zeroed state words stay allocated until after the barrier)
                zeroed.append(t.buffer_address())
                continue
            if t.memory_config().buffer_type == ttnn.BufferType.L1:
                bench_util.l1_log(f"free {t.buffer_address()}")
                ttnn.deallocate(t)

    hd = dict(x_rm=st["x_rm"], **{k: st["hout"][k] for k in ("idx_t", "dest_t", "counts_t", "w")})
    # (the head, shared expert and end pass clear their own state words; the routed stream's are zeroed before it)
    if "head" in phases:
        with bench_util.self_zeroing():
            built = build_router(
                mesh, lc, st["x_rm"], st["wrT_t"], st["ids_t"], st["hout"], st["logd_t"], side=router_side(lc)
            )
        run("head", *built)
    if "shared" in phases:
        with bench_util.self_zeroing():
            built = build_shared(mesh, st["scfg"], hd["x_rm"], st["w13_t"], st["w2s_t"], st["res_t"])[:3]
        run("shared", *built)
    # ("stream:zero": the stream built and its state words zeroed, not run - those few KB stay allocated, so nothing
    # the barrier places in L1 lands on them; "barrier": every chip past that point (two all-gathers of a tiny DRAM
    # tensor, one a torus axis, the second on the first's output) - no stream's fabric writes land before every
    # chip's zeros; then the state words freed; "stream:run": built again - the same addresses - and run without the
    # zeros)
    for sp in [p for p in phases if p.startswith("stream") or p == "barrier"]:
        if sp == "barrier":
            device_barrier(st)
            for t in held:
                ttnn.deallocate(t)
            held.clear()
            continue
        run(
            "stream",
            *stream_program(mesh, lc, st, hd),
            zero=sp != "stream:run",
            launch=sp != "stream:zero",
        )
    if "end" in phases:
        with bench_util.self_zeroing():
            built = build_end(mesh, lc.T, lc.K, lc.H, st["res_t"], hd["w"], st["y_t"])
        run("end", *built)
    if "fold" in phases:
        with bench_util.self_zeroing():
            built = fold_program(mesh, lc, st, hd)
        run("fold", *built)
    for t in held:  # (no barrier followed: freed at the end)
        ttnn.deallocate(t)
    return hd


def fold_program(mesh, lc: LayerCfg, st, hd):
    """The shared expert folded into the end pass (ws_fold): y from the routed rows (K a token) and the shared rows
    through L1."""
    return build_fold(
        mesh,
        st["scfg"],
        lc.K,
        hd["x_rm"],
        st["w13_t"],
        st["w2s_t"],
        st["res_t"],
        hd["w"],
        st["y_t"],
        st["hout"]["fence_t"],
        slices=int(os.environ.get("WSY_FOLD_SLICES", "0")),  # (a diagnostic: the routed rows read in that many pieces)
    )


def run_trace(mesh, lc: LayerCfg, routed_lc: LCfg, tag, chips, ntok):
    """The layer eager (the reference: checks, per-phase wall and device ms), then WS_WARM layers of its programs
    back to back, one captured in a trace and replayed WS_WARM + WS_MEAS times: wall per layer (replay to synchronize,
    and replays back to back) against the eager device sum, the head's outputs and y against the eager layer's
    (bit-exact), the checks again. (The phases' stamps are freed with their phase, as in the eager layer: no device
    spans from the replays.)"""
    assert HEAD == "router", "the traced layer runs the router head (WSY_HEAD=router)"
    N = mesh.get_num_devices()
    st, host = make_layer(mesh, lc, routed_lc)
    allp = FOLD_PHASES if lc.fold else PHASES
    cprogs, cio, cdram = dram_clocks(mesh, len(allp) + 1)  # (first: its L1 scratch stays at the top)
    clk_dev = first_visible()
    clk = aiclk_mhz(clk_dev)
    span_chips = sorted({0, N - 1})
    logger.info(f"WSY {tag}: {N} chips, {lc}, routed {routed_lc}, aiclk {clk} MHz")
    for _ in range(2):
        hd, ph, dev = layer_step(mesh, lc, st, clk, span_chips)
    eager_dev = {k: max(v) for k, v in dev.items()}
    ref = Ref(lc, host, st["rcfg"], st["scfg"])
    for c in chips:
        check_chip(lc, ref, c, hd, st, ntok, f"{tag} eager")
    names = ("y", "counts", "idx", "dest", "w")
    snap = lambda h: {
        k: [ttnn.to_torch(ttnn.get_device_tensors(t)[c]) for c in chips]
        for k, t in zip(names, (st["y_t"], h["counts_t"], h["idx_t"], h["dest_t"], h["w"]))
    }
    s0 = snap(hd)
    bench_util.DEVICE_ZERO = True
    try:
        for _ in range(2):
            layer_programs(mesh, lc, st)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(WARM):
            layer_programs(mesh, lc, st)
        ttnn.synchronize_device(mesh)
        untraced = (time.perf_counter() - t0) * 1e3 / max(WARM, 1)  # (builds included: host-side)
        walls_u = []
        for _ in range(MEAS):
            layer_programs(mesh, lc, st)
            ttnn.synchronize_device(mesh)
        for _ in range(MEAS):
            t0 = time.perf_counter()
            layer_programs(mesh, lc, st)
            ttnn.synchronize_device(mesh)
            walls_u.append((time.perf_counter() - t0) * 1e3)
        diag = {}
        layer_programs(mesh, lc, st, diag=diag, clk=clk)
        logger.info(f"WSY {tag} SUMMARY device-zero layer, phases synchronized apart: {diag}")
        # (the program cache over device-zero layers; then the same layers with the zeros from the host)
        n0 = mesh.num_program_cache_entries()
        for _ in range(3):
            layer_programs(mesh, lc, st)
        ttnn.synchronize_device(mesh)
        n1 = mesh.num_program_cache_entries()
        bench_util.DEVICE_ZERO = False
        for _ in range(2):
            layer_programs(mesh, lc, st)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(MEAS):
            layer_programs(mesh, lc, st)
        ttnn.synchronize_device(mesh)
        host_zero_b2b = (time.perf_counter() - t0) * 1e3 / max(MEAS, 1)
        bench_util.DEVICE_ZERO = True
        logger.info(
            f"WSY {tag} SUMMARY program cache entries {n0} -> {n1} over 3 device-zero layers; host-zero layers back to "
            f"back {host_zero_b2b:.2f} ms"
        )
        # (WSY_TRACE_STREAM=eager: head and shared in one trace, the end pass in another, the routed stream launched
        # eagerly between them)
        tmode = os.environ.get("WSY_TRACE_STREAM", "trace")
        split = tmode == "eager"
        synced = tmode == "sync"  # (two traces, a host synchronize after each: the stream's zeros done on every chip)
        if tmode == "barrier":  # (one trace, a device barrier between the stream's zeros and its run)
            allp = (
                ("head", "stream:zero", "barrier", "stream:run", "fold")
                if lc.fold
                else ("head", "shared", "stream:zero", "barrier", "stream:run", "end")
            )
        if synced:
            groups = (
                [("head", "stream:zero"), ("stream:run", "fold")]
                if lc.fold
                else [("head", "shared", "stream:zero"), ("stream:run", "end")]
            )
        else:
            groups = ([("head",), ("fold",)] if lc.fold else [("head", "shared"), ("end",)]) if split else [allp]
        # (the whole-layer trace carries the DRAM clocks: each phase's time inside a replay)
        clocks = (cprogs, cio) if not split and not synced else None
        if clocks:
            for _ in range(2):
                layer_programs(mesh, lc, st, allp, clocks=clocks)
            ttnn.synchronize_device(mesh)
        tids = []
        for g in groups:
            tids.append(ttnn.begin_trace_capture(mesh, cq_id=0))
            hd = layer_programs(mesh, lc, st, g, clocks=clocks)
            ttnn.end_trace_capture(mesh, tids[-1], cq_id=0)
            try:  # (the trace region's bytes so far: the captured command streams)
                mv = ttnn._ttnn.device.GetMemoryView(mesh, ttnn.BufferType.TRACE)
                logger.info(
                    f"WSY {tag} SUMMARY trace {len(tids)} {g} captured: trace region "
                    f"{mv.total_bytes_allocated_per_bank * mv.num_banks} B allocated ({mv.num_banks} banks)"
                )
            except Exception as ex:  # (a report only)
                logger.info(f"WSY {tag} trace region size unavailable: {ex}")

        def layer():
            ttnn.execute_trace(mesh, tids[0], cq_id=0, blocking=False)
            if split:
                layer_programs(mesh, lc, st, ("stream",))
                ttnn.execute_trace(mesh, tids[1], cq_id=0, blocking=False)
            if synced:
                ttnn.synchronize_device(mesh)
                ttnn.execute_trace(mesh, tids[1], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)

        walls = []
        for _ in range(WARM + MEAS):
            t0 = time.perf_counter()
            layer()
            ttnn.synchronize_device(mesh)
            walls.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        for _ in range(MEAS):
            layer()
        ttnn.synchronize_device(mesh)
        b2b = (time.perf_counter() - t0) * 1e3 / max(MEAS, 1)
        if clocks:
            named = [p for p in allp if p not in ("stream:zero", "barrier")]  # (the phases that tick)
            ticks = dram_clock_ticks(cdram, 0, len(named) + 1)
            seg = {p: round((ticks[i + 1] - ticks[i]) / clk / 1e3, 3) for i, p in enumerate(named)}
            logger.info(
                f"WSY {tag} SUMMARY inside a replay (chip 0, the last; DRAM clocks between the programs): ms {seg}, "
                f"first clock to last {(ticks[-1] - ticks[0]) / clk / 1e3:.2f}"
            )
        s1 = snap(hd)
        same = {k: all(torch.equal(a, b) for a, b in zip(s0[k], s1[k])) for k in names}
        for d in ttnn.get_device_tensors(hd["counts_t"]):
            if bool((ttnn.to_torch(d).to(torch.int64) & 0xFFFFFFFF == 0xFFFFFFFF).any()):
                raise RuntimeError("routing program: an owner core's records overflowed its capacity")
        res = [check_chip(lc, ref, c, hd, st, ntok, f"{tag} trace")[0] for c in chips]
        for tid in tids:
            ttnn.release_trace(mesh, tid)
    finally:
        bench_util.DEVICE_ZERO = False
    med = statistics.median(walls[WARM:])
    logger.info(
        f"WSY {tag} SUMMARY trace{' (head + shared, end; the stream eager)' if split else ''}"
        f"{' (two traces, a synchronize after each)' if synced else ''}: {WARM} warm-up + "
        f"{MEAS} measured replays; wall ms a layer (replay to synchronize, "
        f"median) {med:.2f}, back to back {b2b:.2f}; the same programs untraced (device zeros, builds included) "
        f"{untraced:.2f} back to back, {statistics.median(walls_u):.2f} synchronized; eager wall ms {', '.join(f'{k} {v:.2f}' for k, v in ph.items())}, "
        f"total {sum(ph.values()):.2f}; device ms eager {', '.join(f'{k} {v:.2f}' for k, v in eager_dev.items())}, "
        f"sum {sum(eager_dev.values()):.2f}; equal to the eager layer's: {same}"
    )
    assert all(same.values()), f"the traced layer's outputs differ from the eager layer's: {same}"
    assert all(r["counts"] and r["lists"] and r["dests"] for r in res)
    assert min(r["y_pcc"] for r in res) > 0.999
    return res


BRACKET = int(os.environ.get("WSY_BRACKET", "20"))
_SAMPLER = r"""
import glob, json, os, statistics, sys, time
devs, stop = sys.argv[1].split(","), sys.argv[2]
scale = dict(aiclk=1, w=1e-6, c=1e-3, a=1e-3)
files = {}
for d in devs:
    base = f"/sys/class/tenstorrent/tenstorrent!{d}"
    hw = (glob.glob(base + "/device/hwmon/hwmon*") or [""])[0]
    files[d] = dict(aiclk=base + "/tt_aiclk", w=hw + "/power1_input", c=hw + "/temp1_input", a=hw + "/curr1_input")
    if len(devs) > 1:  # (many chips: AICLK and power only)
        files[d] = {k: files[d][k] for k in ("aiclk", "w")}
s = {d: {k: [] for k in f} for d, f in files.items()}
t_end = time.time() + 60
while not os.path.exists(stop) and time.time() < t_end:
    for d, f in files.items():
        for k, fn in f.items():
            try:
                with open(fn) as fh:
                    s[d][k].append(int(fh.read().strip()) * scale[k])
            except (OSError, ValueError):
                pass
    time.sleep(0.002)
st = lambda v: dict(min=round(min(v), 1), med=round(statistics.median(v), 1), max=round(max(v), 1))
d0 = devs[0]
r = {k: st(v) for k, v in s[d0].items() if v}
r["n"] = len(s[d0]["aiclk"])
r["aiclk_hist"] = {str(k): s[d0]["aiclk"].count(k) for k in sorted(set(s[d0]["aiclk"]))}
if len(devs) > 1:  # (per chip: AICLK median / min, power median / max)
    r["chips"] = {
        d: [statistics.median(v["aiclk"]), min(v["aiclk"]), round(statistics.median(v["w"]), 1), round(max(v["w"]), 1)]
        for d, v in s.items() if v["aiclk"] and v["w"]
    }
print(json.dumps(r))
"""


def telemetry_sampler(dev):
    """Samples chip dev's AICLK (MHz), power (W), temperature (C) and current (A) from sysfs every ~2 ms in a child
    process from now until the returned function is called; that function returns min / median / max of each. dev may
    be a list of chips (PCI indices): the first as above, then every chip's AICLK median / min and power median / max."""
    import subprocess
    import sys
    import tempfile

    stop = tempfile.mktemp(prefix="wsy_tel_")
    devs = ",".join(str(d) for d in (dev if isinstance(dev, (list, tuple)) else [dev]))
    p = subprocess.Popen([sys.executable, "-c", _SAMPLER, devs, stop], stdout=subprocess.PIPE, text=True)
    time.sleep(0.3)

    def finish():
        open(stop, "w").close()
        out, _ = p.communicate(timeout=30)
        os.remove(stop)
        try:
            import json

            return json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {}

    return finish


def run_trace_bisect(mesh, lc: LayerCfg, routed_lc: LCfg, tag):
    """Where a traced layer's time between replays goes: traces of the DRAM clocks alone, of each phase alone, of the
    layer, and of the layer twice, each replayed WS_WARM times synchronized and WS_MEAS times back to back: wall per
    replay against the clocks' span inside it (the last replay), and the host's time in execute_trace. WSY_BISECT
    picks the traces (comma separated: clocks, head, stream, fold, layer, layer2). WSY_PROF_READ=1 (the device
    profiler on): its buffers read before and after the back-to-back replays, so they hold those replays only."""
    N = mesh.get_num_devices()
    allp = FOLD_PHASES if lc.fold else PHASES
    if N > 1:  # (a mesh: the stream's zeros, the device barrier, then the stream - test_ws_layer's "barrier" layer)
        allp = ("head", "stream:zero", "barrier", "stream:run") + (("fold",) if lc.fold else ("shared", "end"))
    tel_chips = list(range(N)) if N > 1 else first_visible()
    st, host = make_layer(mesh, lc, routed_lc)
    nclk = 2 * len(allp) + 2
    cprogs, cio, cdram = dram_clocks(mesh, nclk + 2)  # (first: its L1 scratch stays at the top; the last 2 bracket)
    clk = aiclk_mhz(first_visible())
    prof = os.environ.get("WSY_PROF_READ", "0") == "1"
    layer_step(mesh, lc, st, clk, [0])
    logger.info(f"WSY {tag}: {lc}, routed {routed_lc}, aiclk {clk} MHz")
    sel = os.environ.get("WSY_BISECT", "clocks,head,stream,fold,layer,layer2").split(",")
    out = {}
    bench_util.DEVICE_ZERO = True
    try:
        for _ in range(2):
            layer_programs(mesh, lc, st, allp, clocks=(cprogs, cio))
        ttnn.synchronize_device(mesh)
        # (the wall-clock counter against host time with the chip idle: two clocks 1 s apart)
        sampler = telemetry_sampler(tel_chips)
        t0 = time.perf_counter()
        ttnn.generic_op(cio, cprogs[nclk])
        ttnn.synchronize_device(mesh)
        time.sleep(1.0)
        ttnn.generic_op(cio, cprogs[nclk + 1])
        ttnn.synchronize_device(mesh)
        host_us = (time.perf_counter() - t0) * 1e6
        bt = dram_clock_ticks(cdram, 0, nclk + 2)
        out["idle"] = dict(counter_mhz=round((bt[nclk + 1] - bt[nclk]) / host_us, 1), tel=sampler())
        logger.info(f"WSY {tag} SUMMARY bisect idle: {out['idle']}")

        def body(name):
            """The trace's programs; returns the number of clocks they write."""
            if name == "clocks":
                for i in range(2):
                    ttnn.generic_op(cio, cprogs[i])
                return 2
            if name == "layer2":
                assert N == 1, "layer2: one chip (its clocks' pages assume every phase ticks)"
                layer_programs(mesh, lc, st, allp, clocks=(cprogs[: len(allp) + 1], cio))
                layer_programs(mesh, lc, st, allp, clocks=(cprogs[len(allp) + 1 :], cio))
                return nclk
            ph = allp if name == "layer" else (name,)
            layer_programs(mesh, lc, st, ph, clocks=(cprogs, cio))
            return 1 + len([p for p in ph if p not in ("stream:zero", "barrier")])  # (the phases that launch tick)

        for name in sel:
            for _ in range(2):  # (every op of this trace run once before its capture)
                body(name)
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            n = body(name)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            walls = []
            for _ in range(WARM):
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                walls.append((time.perf_counter() - t0) * 1e3)
            if prof:
                ttnn.ReadDeviceProfiler(mesh)
            enq = 0.0
            t0 = time.perf_counter()
            for _ in range(MEAS):
                t1 = time.perf_counter()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                enq += time.perf_counter() - t1
            ttnn.synchronize_device(mesh)
            b2b = (time.perf_counter() - t0) * 1e3 / MEAS
            if prof:
                ttnn.ReadDeviceProfiler(mesh)
            ticks = dram_clock_ticks(cdram, 0, n)
            seg = [round((ticks[i + 1] - ticks[i]) / clk / 1e3, 3) for i in range(n - 1)]
            inside = (ticks[-1] - ticks[0]) / clk / 1e3
            # (BRACKET back-to-back replays between two clocks outside the trace: the wall-clock counter's cycles
            # against host time, under load, with the chip's AICLK / power / temperature sampled meanwhile)
            sampler = telemetry_sampler(tel_chips)
            t0 = time.perf_counter()
            ttnn.generic_op(cio, cprogs[nclk])
            for _ in range(BRACKET):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.generic_op(cio, cprogs[nclk + 1])
            ttnn.synchronize_device(mesh)
            host_us = (time.perf_counter() - t0) * 1e6
            tel = sampler()
            bt = dram_clock_ticks(cdram, 0, nclk + 2)
            out[name] = dict(
                sync=round(statistics.median(walls[WARM // 2 :]), 3),
                b2b=round(b2b, 3),
                enqueue=round(enq * 1e3 / MEAS, 3),
                inside=round(inside, 3),
                outside=round(b2b - inside, 3),
                ticks=seg,
                bracket_host=round(host_us / 1e3 / BRACKET, 3),
                bracket_counter=round((bt[nclk + 1] - bt[nclk]) / clk / 1e3 / BRACKET, 3),
                counter_mhz=round((bt[nclk + 1] - bt[nclk]) / host_us, 1),
                tel=tel,
            )
            logger.info(f"WSY {tag} SUMMARY bisect {name}: {out[name]}")
            ttnn.release_trace(mesh, tid)
    finally:
        bench_util.DEVICE_ZERO = False
    logger.info(f"WSY {tag} SUMMARY bisect (ms a replay; outside = back to back minus the clocks' span): {out}")
    return out


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
def test_ws_layer_trace_bisect_chip(mesh_device):
    """run_trace_bisect on one chip (the routed stream as the one-chip proxy), T 31250, 640 experts."""
    lc = LayerCfg(T=31250, nexp=640, fold=FOLD)
    run_trace_bisect(mesh_device, lc, proxy_lcfg(lc.G, ks=KS), "trace bisect chip e640 T 31250")


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dict(TORUS, trace_region_size=TRACE_REGION)], indirect=True)
def test_ws_layer_trace_bisect_glx(mesh_device):
    """run_trace_bisect on the Galaxy torus (the layer with the device barrier; WSY_BISECT clocks, layer)."""
    lc = LayerCfg(T=31250, nexp=640, fold=FOLD)
    rlc = routed_glx(mesh_device)
    run_trace_bisect(mesh_device, lc, rlc, "trace bisect glx e640 T 31250")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
def test_ws_layer_trace_chip(mesh_device):
    """The traced layer on one chip (the routed stream as the one-chip proxy), T 31250, 640 experts."""
    lc = LayerCfg(T=31250, nexp=640, fold=FOLD)
    rlc = proxy_lcfg(lc.G, ks=KS)
    run_trace(mesh_device, lc, rlc, "trace chip e640 T 31250", [0], int(os.environ.get("WSY_TOKENS", 256)))


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dict(TORUS, trace_region_size=TRACE_REGION)], indirect=True)
def test_ws_layer_trace_glx(mesh_device):
    """The traced layer on the Galaxy torus, as test_ws_layer_glx."""
    lc = LayerCfg(T=31250, nexp=640, fold=FOLD)
    rlc = routed_glx(mesh_device)
    chips = [int(c) for c in os.environ.get("WSY_CHIPS", "0,31").split(",")]
    run_trace(mesh_device, lc, rlc, "trace glx e640 T 31250", chips, int(os.environ.get("WSY_TOKENS", 64)))


def soak_aiclks():
    """Every chip's AICLK from sysfs (MHz)."""
    import glob

    out = []
    for f in sorted(glob.glob("/sys/class/tenstorrent/tenstorrent!*/tt_aiclk")):
        try:
            out.append(int(open(f).read().strip()))
        except (OSError, ValueError):
            pass
    return out


def run_soak(mesh, lc: LayerCfg, routed_lc: LCfg, tag, chips):
    """WSY_SOAK_ITERS layers (default 10000) in WSY_SOAK_MODE: "trace" (the whole layer one trace), "barrier" (one
    trace, a device barrier between the stream's zeros and the stream), "sync" (two traces split there, a host
    synchronize after each), "split" (head + shared and the end pass traced, the routed stream eager between them) or
    "eager" (the layer's programs back to back, untraced). Every WSY_SOAK_CHECK layers (default 500) y and the head's outputs on `chips` against the first
    eager layer's: bit-exact, else the y PCC; the wall per layer over the stretch."""
    assert HEAD == "router"
    iters = int(os.environ.get("WSY_SOAK_ITERS", "10000"))
    every = int(os.environ.get("WSY_SOAK_CHECK", "500"))
    mode = os.environ.get("WSY_SOAK_MODE", "trace")
    st, _ = make_layer(mesh, lc, routed_lc)
    clk = aiclk_mhz(first_visible())
    hd, _, _ = layer_step(mesh, lc, st, clk)
    names = ("y", "counts", "idx", "dest", "w")
    snap = lambda: {
        k: [ttnn.to_torch(ttnn.get_device_tensors(t)[c]) for c in chips]
        for k, t in zip(names, (st["y_t"], hd["counts_t"], hd["idx_t"], hd["dest_t"], hd["w"]))
    }
    s0 = snap()
    bench_util.DEVICE_ZERO = True
    tids, bad, done = [], [], 0
    try:
        allp = FOLD_PHASES if lc.fold else PHASES
        split_groups = [("head",), ("fold",)] if lc.fold else [("head", "shared"), ("end",)]
        tail = ("stream:run", "fold") if lc.fold else ("stream:run", "end")
        head_p = ("head", "stream:zero") if lc.fold else ("head", "shared", "stream:zero")
        groups = {
            "trace": [allp],
            "barrier": [head_p + ("barrier",) + tail],  # (one trace, a device barrier before the stream)
            "sync": [head_p, tail],  # (two traces, a host synchronize after each)
            "split": split_groups,
            "eager": [],
        }[mode]
        # (warm-ups with the very phases the traces capture: every op - the barrier's all-gathers too - built and run
        # once before a capture, which takes no host writes)
        for _ in range(WARM):
            for g in groups or [None]:
                layer_programs(mesh, lc, st, g)
                if mode == "sync":
                    ttnn.synchronize_device(mesh)
        ttnn.synchronize_device(mesh)
        for g in groups:
            tids.append(ttnn.begin_trace_capture(mesh, cq_id=0))
            layer_programs(mesh, lc, st, g)
            ttnn.end_trace_capture(mesh, tids[-1], cq_id=0)

        def layer():
            if mode == "eager":
                layer_programs(mesh, lc, st)
                return
            ttnn.execute_trace(mesh, tids[0], cq_id=0, blocking=False)
            if mode == "split":
                layer_programs(mesh, lc, st, ("stream",))
                ttnn.execute_trace(mesh, tids[1], cq_id=0, blocking=False)
            if mode == "sync":
                ttnn.synchronize_device(mesh)
                ttnn.execute_trace(mesh, tids[1], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)

        t0 = time.perf_counter()
        while done < iters:
            n = min(every, iters - done)
            for _ in range(n):
                layer()
            ttnn.synchronize_device(mesh)
            done += n
            t1 = time.perf_counter()
            s1 = snap()
            same = {k: all(torch.equal(a, b) for a, b in zip(s0[k], s1[k])) for k in names}
            ypcc = min(pcc(b.float(), a.float()) for a, b in zip(s0["y"], s1["y"]))
            if not all(same.values()):
                bad.append((done, same, ypcc))
            ck = soak_aiclks()
            logger.info(
                f"WSY {tag} soak {mode}: {done}/{iters} layers, wall ms a layer {(t1 - t0) * 1e3 / n:.2f} over the last "
                f"{n}; equal to the first layer's {same}, y pcc {ypcc:.6f}; AICLK {len(ck)} chips median "
                f"{statistics.median(ck) if ck else -1:.0f} min {min(ck) if ck else -1} MHz"
            )
            t0 = time.perf_counter()
    finally:
        for tid in tids:
            ttnn.release_trace(mesh, tid)
        bench_util.DEVICE_ZERO = False
    logger.info(f"WSY {tag} SUMMARY soak {mode}: {done} layers, {len(bad)} checks off the first layer's: {bad[:5]}")
    assert not bad, f"{len(bad)} soak checks differ from the first layer's"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
def test_ws_layer_soak_chip(mesh_device):
    lc = LayerCfg(T=31250, nexp=640, fold=FOLD)
    run_soak(mesh_device, lc, proxy_lcfg(lc.G, ks=KS), "soak chip", [0])


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [dict(TORUS, trace_region_size=TRACE_REGION)], indirect=True)
def test_ws_layer_soak_glx(mesh_device):
    lc = LayerCfg(T=31250, nexp=640, fold=FOLD)
    rlc = routed_glx(mesh_device)
    run_soak(mesh_device, lc, rlc, "soak glx", [0, 31])

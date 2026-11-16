# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The MoE layer around the lane stream (ws_layer.py): head, shared expert, routed stream and combine, every chip on
its own tokens. Checks against torch on the device's routing: the head's counts, token lists and output rows exactly,
the logits / gate weights / top-K choice, and the layer output (every token, or WSY_TOKENS sampled tokens on the
WSY_CHIPS chips). Timing: WS_WARM warm-up + WS_MEAS measured layers, per phase wall time (synchronize to
synchronize; the lane programs are built before their phase starts) and the lane programs' device spans (the group
cores' compute start to end stamps)."""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, pcc, quantize
from models.demos.ws_expert.tests.test_ws_fab_bench import TORUS
from models.demos.ws_expert.tests.test_ws_lane import first_visible
from models.demos.ws_expert.ws_fab import slice_layout
from models.demos.ws_expert.ws_lane import LCfg, build_lane, make_tables, tables_tensor
from models.demos.ws_expert.ws_layer import (
    LayerCfg,
    combine,
    combine_prep,
    gcfg,
    head,
    head_consts,
    proxy_lcfg,
    router_weights,
    shared_setup,
    shared_sum,
)
from models.demos.ws_expert.ws_sf import torus_cycles

WARM = int(os.environ.get("WS_WARM", "20"))
MEAS = int(os.environ.get("WS_MEAS", "5"))
SOL_CORES = 110  # the roofline's per-chip speed of light: 110 cores at 4096 LoFi FLOP a cycle


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
    w1 = (torch.randn(H, lc.IS) / H**0.5).to(torch.bfloat16)
    w3 = (torch.randn(H, lc.IS) / H**0.5).to(torch.bfloat16)
    w2 = (torch.randn(lc.IS, H) / lc.IS**0.5).to(torch.bfloat16)
    wg = torch.randn(E, H, I, dtype=torch.bfloat16) / H**0.5
    wu = torch.randn(E, H, I, dtype=torch.bfloat16) / H**0.5
    wd = torch.randn(E, I, H, dtype=torch.bfloat16) / I**0.5
    x = torch.zeros(N, lc.TX, H, dtype=torch.bfloat16)
    for c in range(N):
        x[c, : lc.T] = torch.randn(lc.T, H).to(torch.bfloat16)
    rcfg = gcfg(E, lc.G, lc.T)
    rtb = make_tables(shape, rcfg, routed_lc)
    per_chip = [slice_layout(rcfg, routed_lc.CHT, wg[o].float(), wu[o].float(), wd[o].float()) for o in rtb["own"]]
    shard = ttnn.ShardTensorToMesh(mesh, dim=0)
    st = dict(
        x_t=ttnn.from_torch(
            x.reshape(N * lc.TX, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard
        ),
        wr_t=router_weights(mesh, wr),
        hc=head_consts(mesh, lc),
        sh=shared_setup(mesh, lc, w1, w3, w2),
        rcfg=rcfg,
        rlc=routed_lc,
        rtb=rtb,
        tab_t=tables_tensor(mesh, rtb["tabs"]),
        ws_t=ttnn.from_torch(
            torch.cat(per_chip, 0),
            dtype=rcfg.w_dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        ),
        out_r=ttnn.empty(
            [lc.ROUT * 32, H],
            dtype=rcfg.out_dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )
    del per_chip
    host = dict(x=x, wr=wr, w1=w1, w3=w3, w2=w2, wg=wg, wu=wu, wd=wd)
    return st, host


def layer_step(mesh, lc: LayerCfg, st, clk=None, span_chips=(), detail=False):
    """One layer: (y, head outputs, {phase: ms}, {lane phase: [device ms of span_chips]}); detail: the head's steps
    synchronized apart (separate keys, not in the head's time)."""
    sync = lambda: ttnn.synchronize_device(mesh)
    ph, dev = {}, {}
    sync()
    marks = [] if detail else None
    t0 = time.perf_counter()
    hd = head(mesh, lc, st["x_t"], st["wr_t"], st["hc"], marks)
    Rk, wk = combine_prep(lc, hd, st["hc"])
    sync()
    ph["head"] = (time.perf_counter() - t0) * 1e3
    if detail:
        marks.append(("combine_prep", time.perf_counter()))
        for (_, a), (n, b) in zip(marks, marks[1:]):
            dev[f"head_{n}"] = [(b - a) * 1e3]
    sh = st["sh"]
    for name, args in (
        (
            "shared",
            (
                sh["cfg"],
                sh["lc"],
                hd["x_rm"],
                sh["idx_t"],
                sh["counts_t"],
                sh["ws_t"],
                sh["out_t"],
                sh["tab_t"],
                sh["tb"],
            ),
        ),
        (
            "stream",
            (
                st["rcfg"],
                st["rlc"],
                hd["x_rm"],
                hd["idx_t"],
                hd["counts_t"],
                st["ws_t"],
                st["out_r"],
                st["tab_t"],
                st["rtb"],
            ),
        ),
    ):
        mpd, io, stamps = build_lane(mesh, *args, 0)
        sync()
        t0 = time.perf_counter()
        ttnn.generic_op(io, mpd)
        sync()
        ph[name] = (time.perf_counter() - t0) * 1e3
        if span_chips:
            dev[name] = [lane_span_us(stamps, ci, clk) / 1e3 for ci in span_chips]
        for t in io:  # (the program's L1 tensors freed now: the next phase's ops place their CBs below them)
            if t.memory_config().buffer_type == ttnn.BufferType.L1:
                ttnn.deallocate(t)
        del mpd, io, stamps
    t0 = time.perf_counter()
    ysh = shared_sum(lc, sh["out_t"])
    if detail:
        sync()
        dev["combine_shared_sum"] = [(time.perf_counter() - t0) * 1e3]
    cm = [] if detail else None
    y = combine(mesh, lc, st["out_r"], Rk, wk, ysh, cm)
    sync()
    ph["combine"] = (time.perf_counter() - t0) * 1e3
    if detail:
        for (_, a), (n, b) in zip(cm, cm[1:]):
            dev[f"combine_{n}"] = [dev.get(f"combine_{n}", [0.0])[0] + (b - a) * 1e3]
    return y, hd, ph, dev


class Ref:
    """torch reference on the device's routing, quantized as the device sees the operands (bfp4 weights, bfp8 x rows,
    bf16 h)."""

    def __init__(self, lc: LayerCfg, host, cfg):
        self.lc, self.h, self.cfg, self.cache = lc, host, cfg, {}
        pad = lc.NSL * lc.I - lc.IS
        self.s1 = quantize(torch.nn.functional.pad(host["w1"].float(), (0, pad)), cfg.w_dt)
        self.s3 = quantize(torch.nn.functional.pad(host["w3"].float(), (0, pad)), cfg.w_dt)
        self.s2 = quantize(torch.nn.functional.pad(host["w2"].float(), (0, 0, 0, pad)), cfg.w_dt)

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

    def layer(self, c, toks, top, w):
        """y rows of chip c's tokens toks, with the device's top-K / gate weights of those tokens ([n, K])."""
        X = quantize(self.h["x"][c, toks].float(), self.cfg.x_dt)
        y = self.swiglu(X, self.s1, self.s3, self.s2)
        for e in top.unique().tolist():
            j, k = torch.nonzero(top == e, as_tuple=True)
            y[j] += w[j, k].unsqueeze(1) * self.swiglu(X[j], *self.expert(e))
        return y


def check_chip(lc: LayerCfg, ref: Ref, c, hd, y, ntok, tag):
    """Head exactness (counts, token lists, output rows), routing vs torch, and the layer PCC on chip c."""
    T, E, K = lc.T, lc.nexp, lc.K
    get = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[c])
    top = get(hd["top"]).reshape(-1, K)[:T].long()
    w = get(hd["w"]).float().reshape(-1, K)[:T]
    R = get(hd["R"]).float().reshape(-1, K)[:T].long()
    counts = get(hd["counts_t"]).long().reshape(-1)
    idx = get(hd["idx_t"]).long().reshape(E, lc.RS)
    # exact: counts, the token lists (ascending tokens, then T), each routed row's output row
    cnt_ref = torch.bincount(top.flatten(), minlength=E)
    ok_cnt = bool((counts == cnt_ref).all())
    order = torch.argsort(top.flatten() * T + torch.arange(T).repeat_interleave(K), stable=True)
    tok_sorted = torch.arange(T).repeat_interleave(K)[order]
    idx_ref = torch.full((E, lc.RS), T, dtype=torch.long)
    off = torch.cumsum(cnt_ref, 0) - cnt_ref
    pos_flat = torch.empty(T * K, dtype=torch.long)
    pos_flat[order] = torch.arange(T * K) - off[top.flatten()[order]]
    for e in range(E):
        idx_ref[e, : cnt_ref[e]] = tok_sorted[off[e] : off[e] + cnt_ref[e]]
    ok_idx = bool((idx == idx_ref).all())
    base = torch.cumsum((cnt_ref + 31) // 32, 0) - (cnt_ref + 31) // 32
    R_ref = base[top] * 32 + pos_flat.reshape(T, K)
    ok_R = bool((R == R_ref).all())
    # routing vs torch on the checked tokens
    toks = torch.arange(T) if ntok <= 0 or ntok >= T else torch.linspace(0, T - 1, ntok).round().long()
    lg_dev = get(hd["logits"]).float().reshape(-1, E)[toks]
    lg_ref = ref.h["x"][c, toks].float() @ ref.h["wr"].float()
    sc_ref = torch.softmax(lg_ref, -1) if lc.score == "softmax" else torch.sigmoid(lg_ref)
    kth = sc_ref.topk(K, -1).values[:, -1:]
    picked = sc_ref.gather(1, top[toks])
    same = int((torch.sort(top[toks], 1).values == torch.sort(sc_ref.topk(K, -1).indices, 1).values).all(1).sum())
    worst_pick = float((picked - kth).min() / kth.mean())  # < 0: a pick below torch's K-th score (near ties)
    w_ref = picked / picked.sum(1, keepdim=True)
    yd = ttnn.to_torch(ttnn.get_device_tensors(y)[c]).float().reshape(-1, lc.H)[toks]
    yr = ref.layer(c, toks, top[toks], w[toks])
    res = dict(
        counts=ok_cnt,
        lists=ok_idx,
        rows=ok_R,
        logits_pcc=pcc(lg_dev, lg_ref),
        w_pcc=pcc(w[toks], w_ref),
        same_topk=f"{same}/{len(toks)}",
        worst_pick=round(worst_pick, 5),
        y_pcc=pcc(yd, yr),
        y_max_rel=float((yd - yr).abs().max() / yr.abs().max()),
    )
    logger.info(f"WSY {tag} chip {c} check ({len(toks)} tokens): {res}")
    return res, counts


def run_layer(mesh, lc: LayerCfg, routed_lc: LCfg, tag, chips, ntok, warm, meas):
    N = mesh.get_num_devices()
    st, host = make_layer(mesh, lc, routed_lc)
    clk_dev = first_visible()
    clk = aiclk_mhz(clk_dev)
    logger.info(f"WSY {tag}: {N} chips, {lc}, routed {routed_lc}, aiclk {clk} MHz")
    span_chips = sorted({0, N - 1})
    rows = []
    nd = int(os.environ.get("WSY_HEAD_DETAIL", "0"))  # extra layers after the measured ones: head steps apart
    drows = []
    for i in range(warm + meas + nd):
        y, hd, ph, dev = layer_step(mesh, lc, st, clk, span_chips if warm <= i < warm + meas else (), i >= warm + meas)
        if warm <= i < warm + meas:
            rows.append((ph, dev))
            logger.info(f"WSY {tag} layer {i - warm + 1}/{meas}: wall ms {ph}, lane device ms {dev}")
        elif i >= warm + meas:
            drows.append(dev)
        if i < warm + meas + nd - 1:
            del y, hd
    if drows:
        hs = {k: round(statistics.median(r[k][0] for r in drows), 3) for k in drows[0]}
        logger.info(f"WSY {tag} SUMMARY head / combine steps ms (synchronized apart, median of {nd}): {hs}")
    ref = Ref(lc, host, st["rcfg"])
    res, all_counts = [], []
    for c in chips:
        r, _ = check_chip(lc, ref, c, hd, y, ntok, tag)
        res.append(r)
    for c in range(N):
        all_counts.append(ttnn.to_torch(ttnn.get_device_tensors(hd["counts_t"])[c]).long().reshape(-1))
    M = torch.stack(all_counts)
    logger.info(
        f"WSY {tag} routed M over {N} chip(s) x {lc.nexp} experts: min {int(M.min())}, median "
        f"{float(M.float().median()):.0f}, max {int(M.max())}, mean {float(M.float().mean()):.1f}; per chip rows "
        f"{M.sum(1).tolist()[:4]}.."
    )
    if meas:
        med = {k: statistics.median(r[0][k] for r in rows) for k in rows[0][0]}
        devm = {k: statistics.median(max(r[1][k]) for r in rows) for k in rows[0][1]}
        tot = sum(med.values())
        sh_fl = 2 * lc.T * lc.H * lc.IS * 3
        sh_ex = 2 * lc.G * lc.TC * lc.H * lc.NSL * lc.I * 3
        sol = SOL_CORES * 4096 * clk * 1e6
        sh_ms = devm.get("shared", med["shared"])
        logger.info(
            f"WSY {tag} SUMMARY: {warm} warm-up + {meas} measured layers, aiclk {clk} MHz (chip {clk_dev}, "
            f"{aiclk_mhz(clk_dev)} at the end); wall ms median head {med['head']:.2f}, shared {med['shared']:.2f}, "
            f"stream {med['stream']:.2f}, combine {med['combine']:.2f}, total {tot:.2f}; lane device ms (max over chips "
            f"{span_chips}, median) shared {devm.get('shared', float('nan')):.2f}, stream "
            f"{devm.get('stream', float('nan')):.2f}; shared floor {sh_fl / sol * 1e3:.2f} ms at {SOL_CORES} cores "
            f"({sh_fl / (80 * 4096 * clk * 1e6) * 1e3:.2f} ms at 80), util {100 * sh_fl / sol / (sh_ms / 1e3):.1f}% "
            f"of {SOL_CORES}-core SOL ({100 * sh_ex / (80 * 4096 * clk * 1e6) / (sh_ms / 1e3):.1f}% of the 80 group "
            f"cores incl. the zero padding)"
        )
    logger.info(
        f"WSY {tag} RESULT: min y pcc {min(r['y_pcc'] for r in res):.5f}, head exact "
        f"{all(r['counts'] and r['lists'] and r['rows'] for r in res)}"
    )
    return res


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("nexp,T", [(64, 2048), (640, 31250)], ids=["e64t2048", "e640t31250"])
def test_ws_layer_chip(mesh_device, nexp, T):
    """One chip (TT_VISIBLE_DEVICES): the routed stream as the one-chip proxy (every expert the chip's own)."""
    lc = LayerCfg(T=T, nexp=nexp)
    small = T <= 4096
    warm = int(os.environ.get("WSY_WARM", 1 if small else WARM))
    meas = int(os.environ.get("WSY_MEAS", 1 if small else MEAS))
    res = run_layer(
        mesh_device,
        lc,
        proxy_lcfg(lc.G),
        f"chip e{nexp} T {T}",
        [0],
        int(os.environ.get("WSY_TOKENS", 0 if small else 256)),
        warm,
        meas,
    )
    assert all(r["counts"] and r["lists"] and r["rows"] for r in res)
    assert min(r["y_pcc"] for r in res) > 0.999


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TORUS], indirect=True)
def test_ws_layer_glx(mesh_device):
    """The whole Galaxy torus: T 31250 tokens a chip, 640 experts streamed over 2 cycles x 2 links (G 4)."""
    lc = LayerCfg(T=31250, nexp=640)
    rlc = LCfg(
        cycles=torus_cycles(*tuple(mesh_device.shape)),
        slots="d0,d1,d0d1,any",
        keep=2,
        bcast=True,
        noc_split=True,
        out_eord=True,
    )
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
    assert all(r["counts"] and r["lists"] and r["rows"] for r in res)
    assert min(r["y_pcc"] for r in res) > 0.999

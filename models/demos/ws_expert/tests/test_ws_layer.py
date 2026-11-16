# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The MoE layer around the lane stream (ws_layer.py): head, routed stream (row-major outputs into the result buffer)
and end pass, every chip on its own tokens. Checks against torch on the device's routing: the head's counts, token
lists and result-row lists exactly, the logits / gate weights / top-K choice, the result buffer's routed rows and the
layer output (every token, or WSY_TOKENS sampled tokens on the WSY_CHIPS chips). The shared expert's rows are not
written yet (zero), so the layer reference is the routed sum. Timing: WS_WARM warm-up + WS_MEAS measured layers, per
phase wall time (synchronize to synchronize; programs built before their phase starts) and the custom programs'
device spans (compute start to end stamps)."""

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
from models.demos.ws_expert.ws_end import build_end, end_span_us
from models.demos.ws_expert.ws_fab import slice_layout
from models.demos.ws_expert.ws_lane import LCfg, build_lane, make_tables, tables_tensor
from models.demos.ws_expert.ws_layer import LayerCfg, gcfg, head, head_consts, proxy_lcfg, result_buffer, router_weights
from models.demos.ws_expert.ws_sf import torus_cycles

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
    x = torch.zeros(N, lc.TX, H, dtype=torch.bfloat16)
    for c in range(N):
        x[c, : lc.T] = torch.randn(lc.T, H).to(torch.bfloat16)
    rcfg = gcfg(E, lc.G, lc.T)
    rtb = make_tables(shape, rcfg, routed_lc)
    per_chip = [slice_layout(rcfg, routed_lc.CHT, wg[o].float(), wu[o].float(), wd[o].float()) for o in rtb["own"]]
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
    return st, dict(x=x, wr=wr, wg=wg, wu=wu, wd=wd)


def layer_step(mesh, lc: LayerCfg, st, clk, span_chips=(), detail=False, trace=False):
    """One layer: (head outputs, {phase: wall ms}, {phase: [device ms of span_chips]}); detail: the head's steps
    synchronized apart (separate keys, not in the head's time); trace: log each phase's launch and end."""
    sync = lambda: ttnn.synchronize_device(mesh)
    ph, dev = {}, {}
    sync()
    marks = [] if detail else None
    t0 = time.perf_counter()
    if trace:
        logger.info("WSY phase head: launch")
    hd = head(mesh, lc, st["x_t"], st["wr_t"], st["hc"], marks)
    sync()
    if trace:
        logger.info("WSY phase head: done")
    ph["head"] = (time.perf_counter() - t0) * 1e3
    if detail:
        for (_, a), (n, b) in zip(marks, marks[1:]):
            dev[f"head_{n}"] = [(b - a) * 1e3]
    progs = (
        (
            "stream",
            lambda: build_lane(
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
            ),
            lane_span_us,
        ),
        ("end", lambda: build_end(mesh, lc.T, lc.K, lc.H, st["res_t"], hd["w"], st["y_t"]), end_span_us),
    )
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
        for t in io:  # (the program's L1 tensors freed now: the next phase's ops place their CBs below them)
            if t.memory_config().buffer_type == ttnn.BufferType.L1:
                ttnn.deallocate(t)
        del mpd, io, stamps
    return hd, ph, dev


class Ref:
    """torch reference on the device's routing, quantized as the device sees the operands (bfp4 weights, bfp8 x rows,
    bf16 h)."""

    def __init__(self, lc: LayerCfg, host, cfg):
        self.lc, self.h, self.cfg, self.cache = lc, host, cfg, {}

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
    top = get(hd["top"]).reshape(-1, K)[:T].long()
    w = get(hd["w"]).float().reshape(-1, 16)[:T, :K]
    counts = get(hd["counts_t"]).long().reshape(-1)
    idx = get(hd["idx_t"]).long().reshape(E, lc.RS)
    dest = get(hd["dest_t"]).long().reshape(E, lc.RS)
    cnt_ref = torch.bincount(top.flatten(), minlength=E)
    ok_cnt = bool((counts == cnt_ref).all())
    tk = torch.arange(T).repeat_interleave(K)
    order = torch.argsort(top.flatten() * T + tk, stable=True)
    off = torch.cumsum(cnt_ref, 0) - cnt_ref
    ok_idx = ok_dest = True
    for e in range(E):
        sel = order[off[e] : off[e] + cnt_ref[e]]
        ok_idx &= bool((idx[e, : cnt_ref[e]] == tk[sel]).all()) and bool((idx[e, cnt_ref[e] :] == T).all())
        ok_dest &= bool((dest[e, : cnt_ref[e]] == tk[sel] * S + sel % K).all())
    toks = torch.arange(T) if ntok <= 0 or ntok >= T else torch.linspace(0, T - 1, ntok).round().long()
    lg_dev = get(hd["logits"]).float().reshape(-1, E)[toks]
    lg_ref = ref.h["x"][c, toks].float() @ ref.h["wr"].float()
    sc_ref = torch.softmax(lg_ref, -1) if lc.score == "softmax" else torch.sigmoid(lg_ref)
    kth = sc_ref.topk(K, -1).values[:, -1:]
    picked = sc_ref.gather(1, top[toks])
    same = int((torch.sort(top[toks], 1).values == torch.sort(sc_ref.topk(K, -1).indices, 1).values).all(1).sum())
    worst_pick = float((picked - kth).min() / kth.mean())  # < 0: a pick below torch's K-th score (near ties)
    w_ref = picked / picked.sum(1, keepdim=True)
    eo = ref.routed(c, toks, top[toks])  # [n, K, H]
    res = get(st["res_t"]).float()[: T * S].reshape(T, S, lc.H)[toks]
    yd = get(st["y_t"]).float().reshape(T, lc.H)[toks]
    yr = (w[toks].unsqueeze(2) * eo).sum(1) + res[:, K]
    full = {}
    if os.environ.get("WSY_FULLCHECK", "0") == "1":
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
        logits_pcc=pcc(lg_dev, lg_ref),
        w_pcc=pcc(w[toks], w_ref),
        same_topk=f"{same}/{len(toks)}",
        worst_pick=round(worst_pick, 5),
        rows_pcc=pcc(res[:, :K], eo),
        shared_rows_absmax=float(res[:, K].abs().max()),
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
        hd, ph, dev = layer_step(
            mesh, lc, st, clk, span_chips if warm <= i < warm + meas else (), i >= warm + meas, trace=i < 2
        )
        if warm <= i < warm + meas:
            rows.append((ph, dev))
            logger.info(f"WSY {tag} layer {i - warm + 1}/{meas}: wall ms {ph}, device ms {dev}")
        elif i >= warm + meas:
            drows.append(dev)
        if i < warm + meas + nd - 1:
            del hd
    if drows:
        hs = {k: round(statistics.median(r[k][0] for r in drows), 3) for k in drows[0]}
        logger.info(f"WSY {tag} SUMMARY head steps ms (synchronized apart, median of {nd}): {hs}")
    ref = Ref(lc, host, st["rcfg"])
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
@pytest.mark.parametrize("nexp,T", [(64, 2048), (640, 31250), (640, 8192)], ids=["e64t2048", "e640t31250", "e640t8192"])
def test_ws_layer_chip(mesh_device, nexp, T):
    """One chip (TT_VISIBLE_DEVICES): the routed stream as the one-chip proxy (every expert the chip's own)."""
    lc = LayerCfg(T=T, nexp=nexp)
    small = T <= 4096
    warm = int(os.environ.get("WSY_WARM", 1 if small else WARM))
    meas = int(os.environ.get("WSY_MEAS", 1 if small else MEAS))
    ntok = int(os.environ.get("WSY_TOKENS", 0 if small else 256))
    rlc = proxy_lcfg(
        lc.G,
        y_nowrite=os.environ.get("WSY_NOWRITE", "0") == "1",
        y_posted=os.environ.get("WSY_POSTED", "1") == "1",
    )
    res = run_layer(mesh_device, lc, rlc, f"chip e{nexp} T {T}", [0], ntok, warm, meas)
    assert all(r["counts"] and r["lists"] and r["dests"] for r in res)
    assert min(r["y_pcc"] for r in res) > 0.999 and min(r["rows_pcc"] for r in res) > 0.999


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
        y_rm=True,
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
    assert all(r["counts"] and r["lists"] and r["dests"] for r in res)
    assert min(r["y_pcc"] for r in res) > 0.999 and min(r["rows_pcc"] for r in res) > 0.999

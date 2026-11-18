# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""EP-4 with weights on Y on the LoudBox (2 x 4, FABRIC_2D): the layer's phases against the host reference (ep4_ref.py)
at the Galaxy-equivalent load (EP4_LEN: 302K / 588K / 874K / 1M -> n tokens a chip). Timing: traced replays (WS_WARM +
WS_MEAS), AICLK of the first chip.

- test_ep4_head (stage 2): the head (ws_router, EP-4 mode) and the group exchange (all-gather of the topk rows on X):
  route words and grp bit-exact against the reference built from the device's own top-K; the top-K against torch's
  (set agreement, near ties only), the gate weights against torch's.
- test_ep4_dispatch (stage 3): the head's route words, then sparse_dispatch_2d with masks inside the mesh row: every
  routed row bit-exact at its destinations, every other slot as it was (a pass-through chip may keep the home's
  row); traced dispatch time against the line's speed of light.
- test_ep4_lists (stage 4): the receive-side lists from the device's grp: counts, token lists, O rows, run records,
  ready lists and the O rows used, exact against ep4_ref.chip_layout on the same routing.
- test_ep4_experts (stage 5): the 5 x 5 lane on the Y columns (2-chip cycles on the LoudBox, one mux a group) over
  xbuf and the lists: weights only (zero counts: the per-lane weight delivery), then the real rows; O against torch
  (sampled experts, every chip), device time per chip (end to end, the lane's stamps).
- test_ep4_layer (stage 6): the whole layer in sequence - head, exchange, dispatch, lists, experts, shared expert
  (into P slot C), pre-sum + combine (ports, counters), home pass: y against torch on sampled tokens of every chip,
  two layers bit-identical, per-phase times."""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz
from models.demos.ws_expert.ep4_layer import (
    S5,
    EP4Cfg,
    build_combine,
    build_home,
    build_lists,
    counters_tensor,
    dispatch,
    dispatch_resources,
    exchange,
    grp_tensor,
    head_outputs,
    head_program,
    lane_cfgs,
    lane_program,
    lane_tables,
    lists_outputs,
    make_inputs,
    o_rows,
    o_tensor,
    own_slices,
    p_tensor,
    pos_tensor,
    route_pieces,
    unpack_topk,
    x_rows,
    xbuf_tensor,
    xrow,
)
from models.demos.ws_expert.ep4_ref import Geo, chip_layout, lane_orders, n_per_chip, route_words
from models.demos.ws_expert.tests.test_ws_fab_bench import FAB, TORUS
from models.demos.ws_expert.ws_router import router_span_us

LEN = os.environ.get("EP4_LEN", "302K")
# EP4_MESH glx: the 8 x 4 Galaxy (FABRIC_2D_TORUS_XY: X a 4-ring, Y 8-chip rings) instead of the LoudBox 2 x 4
GLX = os.environ.get("EP4_MESH", "lb") == "glx"
WARM = int(os.environ.get("WS_WARM", "20"))
MEAS = int(os.environ.get("WS_MEAS", "5"))
TRACE_REGION = 64 << 20
# (EP4_PAYLOAD: the fabric's max packet payload; the lane's 15232 by default, >= 15424 lets the dispatch pack 2 rows)
FABT = dict(TORUS if GLX else FAB, trace_region_size=TRACE_REGION)
MESH = (8, 4) if GLX else (2, 4)
if os.environ.get("EP4_PAYLOAD"):
    from models.demos.ws_expert.tests.test_ws_fab_bench import router_cfg

    FABT["fabric_router_config"] = router_cfg(int(os.environ["EP4_PAYLOAD"]))


def traced(mesh, body, warm=WARM, meas=MEAS):
    """body() captured once in a trace; replayed warm + meas times; returns the median ms a replay (replay to
    synchronize) over the measured ones."""
    body()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    walls = []
    for i in range(warm + meas):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        if i >= warm:
            walls.append((time.perf_counter() - t0) * 1e3)
    ttnn.release_trace(mesh, tid)
    return statistics.median(walls)


def per_chip(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def free_l1(io, keep=()):
    """Free a program's L1 tensors (its state; the next phase's program reuses the space), but those in keep."""
    ka = {t.buffer_address() for t in keep}
    for t in io:
        if t.memory_config().buffer_type == ttnn.BufferType.L1 and t.buffer_address() not in ka:
            ttnn.deallocate(t)


def run_free(built, keep=()):
    """Run a built program (mesh program, io, ...) and free its L1 tensors."""
    ttnn.generic_op(built[1], built[0])
    free_l1(built[1], keep)


def rows32(x):
    """A topk tensor's chip copy as [tokens, 32] rows."""
    return x.reshape(-1, 32)


@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("device_params", [FABT], indirect=True)
@pytest.mark.parametrize("routing", ["uniform", "zipf"])
def test_ep4_head(mesh_device, routing):
    mesh = mesh_device
    R, C = tuple(mesh.shape)
    geo = Geo(R, C)
    n = n_per_chip(LEN)
    lc = EP4Cfg(T=n)
    host, dev = make_inputs(mesh, lc, routing)
    outs = head_outputs(mesh, lc)
    grp = grp_tensor(mesh, lc)
    mpd, io, st = head_program(mesh, lc, dev, outs)
    clk = aiclk_mhz(0)

    ttnn.generic_op(io, mpd)
    exchange(outs["w"], grp)
    ttnn.synchronize_device(mesh)
    span = router_span_us(st, 0, clk)
    tk = [rows32(v) for v in per_chip(outs["w"])]
    rt = [torch.cat([v.reshape(-1) for v in parts]) for parts in zip(*[per_chip(t) for t in outs["route_pieces"]])]
    gp = [rows32(v) for v in per_chip(grp)]
    ids = torch.empty(R, C, n, lc.K, dtype=torch.int64)
    for r in range(R):
        for c in range(C):
            i, w, rw = unpack_topk(tk[r * C + c][:n], lc.K)
            ids[r, c] = i
    # route words: the reference from the device's own top-K, link = t & 1
    RW = 1 if R * C <= 16 else 2
    ref_route = route_words(geo, ids, link=lambda t: t & 1, words=RW)
    bad_route = 0
    for r in range(R):
        for c in range(C):
            ch = r * C + c
            dev_rw = rt[ch].reshape(-1).to(torch.int64) & 0xFFFFFFFF
            tkw = lambda j: tk[ch][:n, j].to(torch.int64) & 0xFFFFFFFF
            if RW == 1:
                bad_route += int((dev_rw[:n] != ref_route[r, c]).sum()) + int((dev_rw[n:] != 0).sum())
                bad_route += int((tkw(28) != ref_route[r, c]).sum())
            else:
                dv = dev_rw.view(-1, 2)
                for j, want in enumerate((ref_route[0][r, c], ref_route[1][r, c])):
                    bad_route += int((dv[:n, j] != want).sum()) + int((tkw(28 + j) != want).sum())
                bad_route += int((dv[n:] != 0).sum())
    # grp: chip (r, c) holds the topk rows of (r, 0..C-1), home-major
    bad_grp = 0
    for r in range(R):
        want = torch.cat([tk[r * C + c] for c in range(C)], 0)
        for c in range(C):
            bad_grp += int((gp[r * C + c] != want).any(-1).sum())
    # the top-K and gate weights against torch (fp32 logits of the same bf16 operands)
    from models.demos.ws_expert.bench_util import pcc

    same, worst_gap, werr, wp = 0, 0.0, 0.0, 1.0
    for ch in range(R * C):
        lg = host["x"][ch, :n].float() @ host["wr"].float()
        v, ti = torch.softmax(lg, -1).topk(lc.K, -1)
        di = ids[ch // C, ch % C]
        eq = (torch.sort(ti, -1).values == torch.sort(di, -1).values).all(-1)
        same += int(eq.sum())
        if (~eq).any():
            s = torch.softmax(lg[~eq], -1)
            dev_min = torch.gather(s, 1, di[~eq]).min(-1).values
            worst_gap = max(worst_gap, float(((v[~eq, -1] - dev_min) / v[~eq, -1]).max()))
        dw = unpack_topk(tk[ch][:n], lc.K)[1]
        rw = torch.gather(torch.softmax(lg, -1), 1, di)
        rw = rw / rw.sum(-1, keepdim=True)
        werr = max(werr, float(((dw[eq] - rw[eq]).abs() / rw[eq]).max()))
        wp = min(wp, pcc(dw[eq], rw[eq]))
    total = R * C * n
    logger.info(
        f"EP4 head {routing} n {n}: route words bad {bad_route}, grp rows bad {bad_grp}; top-K set equal to torch on "
        f"{same}/{total} tokens (worst miss {worst_gap * 100:.3f}% below torch's K-th score), gate weights PCC {wp:.7f}, "
        f"max rel err {werr:.2e}; eager head program {span['end']:.1f} us (aiclk {clk} MHz)"
    )
    assert bad_route == 0 and bad_grp == 0
    # (the head's top-K and exp as the WS layer's: near ties only, the weights to the soft-float exp's precision)
    assert same >= 0.98 * total and worst_gap < 5e-3 and wp > 0.99999 and werr < 1e-2

    def body_head():
        ttnn.generic_op(io, mpd)

    def body_all():
        ttnn.generic_op(io, mpd)
        exchange(outs["w"], grp)

    def body_x():
        exchange(outs["w"], grp)

    t_head = traced(mesh, body_head)
    t_x = traced(mesh, body_x)
    t_all = traced(mesh, body_all)
    clk2 = aiclk_mhz(0)
    logger.info(
        f"EP4 head SUMMARY {routing} {LEN} (n {n}): traced head {t_head:.3f} ms, exchange {t_x:.3f} ms, head + exchange "
        f"{t_all:.3f} ms; aiclk {clk} -> {clk2} MHz"
    )


@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("device_params", [FABT], indirect=True)
@pytest.mark.parametrize("routing", ["uniform", "zipf"])
def test_ep4_dispatch(mesh_device, routing):
    mesh = mesh_device
    R, C = tuple(mesh.shape)
    N = R * C
    n = n_per_chip(LEN)
    lc = EP4Cfg(T=n)
    TX = lc.TX
    host, dev = make_inputs(mesh, lc, routing)
    outs = head_outputs(mesh, lc)
    run_free(head_program(mesh, lc, dev, outs))
    res = dispatch_resources(mesh, lc)
    xbuf = xbuf_tensor(mesh, lc, host["x"])
    clk = aiclk_mhz(0)
    routes = route_pieces(lc, outs)
    out = dispatch(lc, xbuf, routes, res, N)
    ttnn.synchronize_device(mesh)
    rt = [
        torch.cat([v.reshape(-1) for v in parts]).to(torch.int64) & 0xFFFFFFFF
        for parts in zip(*[per_chip(t) for t in outs["route_pieces"]])
    ]
    masks = rt if N <= 16 else [v.view(-1, 2)[:, 0] for v in rt]  # (2 route words a token past 16 chips: mask, flags)
    devs = ttnn.get_device_tensors(out)  # (read back a chip at a time: every chip's xbuf at once is N^2 KP SP rows)
    bad, routed, held = 0, 0, 0
    for ch in range(N):
        r, c = divmod(ch, C)
        xb = ttnn.to_torch(devs[ch]).reshape(x_rows(lc, N), lc.H)
        for home in range(N):
            rows = xb[xrow(lc, N, home, torch.arange(TX))]
            if home == ch:
                bad += int((rows != host["x"][ch]).any(-1).sum())
                continue
            to_me = ((masks[home][:TX] >> ch) & 1).bool()
            eq_home = (rows == host["x"][home]).all(-1)
            zero = (rows == 0).all(-1)
            if home // C != r:
                assert not to_me.any(), "a route word names a chip outside the mesh row"
            bad += int((to_me & ~eq_home).sum()) + int((~to_me & ~(zero | eq_home)).sum())
            if (to_me & ~eq_home).any() and ch in (0, 1):
                bt = torch.nonzero(to_me & ~eq_home).flatten()
                rr = rows[bt[:4]]
                which = [
                    [
                        (h2, int(t2))
                        for h2 in range(N)
                        for t2 in torch.nonzero((host["x"][h2] == rr[q]).all(-1)).flatten()[:2]
                    ]
                    for q in range(len(rr))
                ]
                logger.info(
                    f"EP4 dispatch diag chip {ch} home {home}: bad routed {int((to_me & ~eq_home).sum())} (zero "
                    f"{int((to_me & zero).sum())}), first t {bt[:4].tolist()}, rows found at (home, t) {which}; "
                    f"stray {int((~to_me & ~(zero | eq_home)).sum())}"
                )
            routed += int(to_me.sum())
            held += int((~to_me & eq_home & ~zero).sum())
        del xb
    logger.info(
        f"EP4 dispatch {routing} n {n}: rows routed {routed} ({routed / (N * n):.3f} a token), bad {bad}, pass-through "
        f"copies kept {held}"
    )
    assert bad == 0

    t_disp = traced(mesh, lambda: dispatch(lc, xbuf, routes, res, N))
    clk2 = aiclk_mhz(0)
    logger.info(
        f"EP4 dispatch SUMMARY {routing} {LEN} (n {n}): traced dispatch {t_disp:.3f} ms ({lc.KP} pieces of {lc.SP}); rows sent a chip "
        f"{routed / N / n:.2f} n; aiclk {clk} -> {clk2} MHz"
    )
    res.release()


def lane_pos(geo: Geo, G=2):
    """Integer stream-position key a local expert: its index in its group's lane order x G + g (chip position 0's
    order; equal-length groups)."""
    per = lane_orders(geo.R, geo.EPC, G, "lane")[0]
    pos = [0] * geo.EPC
    for g, seq in enumerate(per):
        for k, le in enumerate(seq):
            pos[le] = k * G + g
    return pos


@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("device_params", [FABT], indirect=True)
@pytest.mark.parametrize("routing", ["uniform", "zipf"])
def test_ep4_lists(mesh_device, routing):
    mesh = mesh_device
    R, C = tuple(mesh.shape)
    geo = Geo(R, C)
    n = n_per_chip(LEN)
    lc = EP4Cfg(T=n)
    TX, EPC, K = lc.TX, lc.EPC, lc.K
    host, dev = make_inputs(mesh, lc, routing)
    hout = head_outputs(mesh, lc)
    grp = grp_tensor(mesh, lc)
    hp = head_program(mesh, lc, dev, hout)
    run_free(hp)
    exchange(hout["w"], grp)
    pos = lane_pos(geo)
    pos_t = pos_tensor(mesh, pos)
    louts = lists_outputs(mesh, lc)
    mpd, io, st = build_lists(mesh, lc, grp, pos_t, louts)
    clk = aiclk_mhz(0)
    ttnn.generic_op(io, mpd)
    ttnn.synchronize_device(mesh)
    tk = [rows32(v) for v in per_chip(hout["w"])]
    ids = torch.empty(R, C, n, K, dtype=torch.int64)
    w = torch.empty(R, C, n, K)
    for ch in range(R * C):
        i, ww, _ = unpack_topk(tk[ch][:n], K)
        ids[ch // C, ch % C], w[ch // C, ch % C] = i, ww
    key = torch.tensor(pos, dtype=torch.float64)
    got = {k: per_chip(louts[k]) for k in ("runs", "idx", "dest", "rdy", "counts", "rcounts")}
    u = lambda t: t.to(torch.int64) & 0xFFFFFFFF
    bad = {}
    rows_used = []
    for ch in range(R * C):
        r, c = divmod(ch, C)
        ref = chip_layout(geo, ids, w, r, c, S=TX)
        cnt = u(got["counts"][ch]).reshape(-1)
        rc = u(got["rcounts"][ch]).reshape(-1)
        rows_used.append(int(rc[EPC]))
        b = dict(counts=int((cnt != ref.counts).sum()), rows=int(rc[EPC] != ref.R_rows))
        b["expected"] = int((rc[EPC + 1 : EPC + 1 + C] != ref.cnt).sum())
        idx, dest, rdy = (u(got[k][ch]) for k in ("idx", "dest", "rdy"))
        bl = bd = 0
        for le in range(EPC):
            m = int(ref.counts[le])
            lt = ref.lists_tok[le]
            bl += int((idx[le, :m] != xrow(lc, R * C, r * C + lt // TX, lt % TX)).sum())
            bd += int((dest[le, :m] != ref.lists_row[le]).sum())
            pad = -(-m // 32) * 32
            bl += int((idx[le, m:pad] != xrow(lc, R * C, ch, lc.T)).sum())
            bd += int((dest[le, m:pad] != o_rows(lc) - 1).sum())
        b.update(idx=bl, dest=bd)
        # run records of the tokens with local pairs
        g = ref.run_home * TX + ref.run_t
        rr = u(got["runs"][ch])[g]
        b["run_base"] = int((rr[:, 0] != ref.run_base).sum())
        b["run_cnt"] = int((rr[:, 1] != ref.run_cnt).sum())
        b["run_g"] = int((rr[:, 3] != g).sum())
        vm = ref.run_exp >= 0
        b["run_le"] = int((rr[:, 16:28][vm] != ref.run_exp[vm]).sum())
        b["run_w"] = int((rr[:, 4:16][vm].to(torch.int32).view(torch.float32) != ref.run_w[vm]).sum())
        allg = u(got["runs"][ch])
        live = torch.zeros(C * TX, dtype=torch.bool)
        live[g] = True
        valid = (torch.arange(C * TX) % TX) < lc.T
        b["run_empty"] = int((allg[~live & valid, 1] != 0).sum())
        # ready lists: runs (as group tokens g) by their last expert in stream order, g order
        kx = torch.where(vm, key[ref.run_exp.clamp(min=0)], torch.full(vm.shape, -1.0, dtype=key.dtype))
        last = torch.gather(ref.run_exp, 1, kx.argmax(1, keepdim=True)).squeeze(1)
        b["run_last"] = int((rr[:, 2] != last).sum())
        br = 0
        for le in range(EPC):
            want = g[last == le]
            q = int(rc[le])
            br += int(q != want.numel()) + (int((rdy[le, :q] != want).sum()) if q == want.numel() else 0)
        b["ready"] = br
        bad[ch] = {k: v for k, v in b.items() if v}
    logger.info(
        f"EP4 lists {routing} n {n}: mismatches per chip {bad}; O rows used / 12n {[round(x / (12 * n), 3) for x in rows_used]}"
    )
    assert all(not v for v in bad.values())
    assert max(rows_used) <= o_rows(lc) - 1
    # determinism of the whole chain head -> exchange -> lists
    snap = {k: per_chip(louts[k]) for k in ("runs", "idx", "dest", "rdy", "counts", "rcounts")}
    free_l1(io)
    run_free(head_program(mesh, lc, dev, hout))
    exchange(hout["w"], grp)
    mpd, io, st = build_lists(mesh, lc, grp, pos_t, louts)
    ttnn.generic_op(io, mpd)
    ttnn.synchronize_device(mesh)
    again = {k: per_chip(louts[k]) for k in snap}
    assert all(torch.equal(a, b) for k in snap for a, b in zip(snap[k], again[k])), "lists differ between two runs"
    t_l = traced(mesh, lambda: ttnn.generic_op(io, mpd))
    clk2 = aiclk_mhz(0)
    logger.info(
        f"EP4 lists SUMMARY {routing} {LEN} (n {n}): traced lists {t_l:.3f} ms; aiclk {clk} -> {clk2} MHz; bit-identical"
        " over two head -> exchange -> lists runs"
    )


def expert_weights(lc: EP4Cfg, seed=1):
    g = torch.Generator().manual_seed(seed)
    E, H, I = lc.nexp, lc.H, lc.I
    wg = (torch.randn(E, H, I, generator=g) / H**0.5).to(torch.bfloat16)
    wu = (torch.randn(E, H, I, generator=g) / H**0.5).to(torch.bfloat16)
    wd = (torch.randn(E, I, H, generator=g) / I**0.5).to(torch.bfloat16)
    return wg, wu, wd


def front(mesh, lc: EP4Cfg, routing, pos=None):
    """head -> exchange -> dispatch -> lists on device; returns the state (host inputs, device tensors)."""
    host, dev = make_inputs(mesh, lc, routing)
    hout = head_outputs(mesh, lc)
    grp = grp_tensor(mesh, lc)
    run_free(head_program(mesh, lc, dev, hout))
    exchange(hout["w"], grp)
    res = dispatch_resources(mesh, lc)
    xbuf = xbuf_tensor(mesh, lc, host["x"])
    xbuf = dispatch(lc, xbuf, route_pieces(lc, hout), res, mesh.get_num_devices())
    pos_t = pos_tensor(mesh, pos)
    louts = lists_outputs(mesh, lc)
    run_free(build_lists(mesh, lc, grp, pos_t, louts))
    ttnn.synchronize_device(mesh)
    return dict(host=host, dev=dev, hout=hout, grp=grp, res=res, xbuf=xbuf, pos_t=pos_t, louts=louts)


def lane_times(cfg, T, counts, st, chips, clk):
    """Per chip (end to end us, NCRISC end per group us) from the lane's stamps."""
    from models.demos.ws_expert.tests.test_ws_lane import chip_time

    sts = ttnn.get_device_tensors(st)
    out = {}
    for ci in chips:
        Ms = [int(v) for v in counts[ci]]
        _, _, _, e2e, ncend = chip_time(cfg, Ms, T, ci, sts[ci], x0=512)
        out[ci] = (e2e / clk, [v / clk for v in ncend])
    return out


@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("device_params", [FABT], indirect=True)
@pytest.mark.parametrize("routing", ["uniform", "zipf"])
def test_ep4_experts(mesh_device, routing):
    from models.demos.ws_expert.bench_util import quantize
    from models.demos.ws_expert.ws_expert import interleave_gu
    from models.demos.ws_expert.ws_lane import tables_tensor

    mesh = mesh_device
    R, C = tuple(mesh.shape)
    N = R * C
    n = n_per_chip(LEN)
    lc = EP4Cfg(T=n)
    TX, EPC = lc.TX, lc.EPC
    G = int(os.environ.get("EP4_G", "2"))
    cfg, lane, l5 = lane_cfgs(mesh, lc, G)
    T, pos = lane_tables(mesh, cfg, lane)
    tab_t = tables_tensor(mesh, T["tabs"])
    s = front(mesh, lc, routing, pos)
    wg, wu, wd = expert_weights(lc)
    ws_t = ttnn.from_torch(
        torch.cat(own_slices(cfg, lane, T, lc, wg, wu, wd, C), 0),
        dtype=cfg.w_dt,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    louts = s["louts"]
    counts = [(c.reshape(-1).to(torch.int64) & 0xFFFFFFFF) for c in per_chip(louts["counts"])]
    assert all(int(c.max()) < 0xFFFFFFFF for c in counts), "an owner over CAP"
    clk = aiclk_mhz(0)
    chips = list(range(N))
    exp_bytes = 4.147e6  # (an expert's bfp4 slices)
    # weights only: the same lane with zero counts
    zc = ttnn.from_torch(
        torch.zeros(1, EPC, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    lz = dict(louts, counts=zc)
    O = o_tensor(mesh, lc)
    wo = []
    for i in range(3):
        mpd, io, st = lane_program(mesh, lc, cfg, lane, l5, T, tab_t, s["xbuf"], lz, ws_t, O)
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh)
        wo.append(lane_times(cfg, T, [[0] * EPC] * N, st, chips, clk))
        del mpd, io, st
    sent = EPC // G // R  # (experts a chip sends a lane: its own of the group)
    wend = [max(max(r[ci][1][g] for ci in chips) for r in wo[1:]) for g in range(G)]
    logger.info(
        f"EP4 experts {routing}: weights only (zero counts), NCRISC end per group (us, max over chips, last 2 launches) "
        f"{[round(v, 1) for v in wend]} = {[round(sent * exp_bytes / (v * 1e3), 1) for v in wend]} GB/s a link "
        f"direction ({sent} experts a direction a lane)"
    )
    # real rows
    e2es = []
    for i in range(WARM + MEAS if WARM < 5 else 5):
        mpd, io, st = lane_program(mesh, lc, cfg, lane, l5, T, tab_t, s["xbuf"], louts, ws_t, O)
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh)
        e2es.append(lane_times(cfg, T, counts, st, chips, clk))
        del mpd, io, st
    clk2 = aiclk_mhz(0)
    # O against torch: sampled local experts on every chip
    host = s["host"]
    Ox = per_chip(O)
    idx = [u.to(torch.int64) & 0xFFFFFFFF for u in per_chip(louts["idx"])]
    dest = [u.to(torch.int64) & 0xFFFFFFFF for u in per_chip(louts["dest"])]

    def host_rows(rows):  # (xbuf rows -> chip, token)
        p, rem = rows // (N * lc.SP), rows % (N * lc.SP)
        return host["x"][rem // lc.SP, p * lc.SP + rem % lc.SP]

    pccs = {}
    nsamp = int(os.environ.get("EP4_PCC_EXPERTS", "6"))
    for ch in chips:
        c = ch % C
        worst = 1.0
        for le in torch.linspace(0, EPC - 1, nsamp).round().long().tolist():
            m = int(counts[ch][le])
            if m == 0:
                continue
            e = EPC * c + le
            rows = idx[ch][le, :m]
            xr = quantize(host_rows(rows).float(), cfg.x_dt)
            wguq = quantize(interleave_gu(wg[e].float(), wu[e].float()), cfg.w_dt)
            wgq = wguq.reshape(lc.H, -1, 2, 32)[:, :, 0].reshape(lc.H, lc.I)
            wuq = wguq.reshape(lc.H, -1, 2, 32)[:, :, 1].reshape(lc.H, lc.I)
            h = quantize(torch.nn.functional.silu(xr @ wgq) * (xr @ wuq), cfg.h_dt)
            ref = h @ quantize(wd[e].float(), cfg.w_dt)
            got = Ox[ch][dest[ch][le, :m]].float()
            from models.demos.ws_expert.bench_util import pcc

            worst = min(worst, pcc(got, ref))
        pccs[ch] = worst
    per = {ci: round(statistics.median(r[ci][0] for r in e2es) / 1e3, 3) for ci in chips}
    rows = {ci: int(counts[ci].sum()) for ci in chips}
    logger.info(
        f"EP4 experts SUMMARY {routing} {LEN} (n {n}) G {G}: expert phase per chip (ms, median end to end) {per}, "
        f"max {max(per.values()):.3f}; rows a chip / 12n {[round(rows[ci] / (12 * n), 3) for ci in chips]}; min PCC "
        f"per chip {[round(v, 5) for v in pccs.values()]}; aiclk {clk} -> {clk2} MHz"
    )
    assert min(pccs.values()) > 0.999


def span_us(sts, ci, clk, start=0, end=63):
    """Device span on chip ci over stamps tensors sts (lists of 8 x 64-word stamps a core, one wall clock a chip):
    latest end - earliest start, us."""
    vals = []
    for st in sts:
        s = (ttnn.to_torch(ttnn.get_device_tensors(st)[ci]).to(torch.int64) & 0xFFFFFFFF).reshape(-1, 8 * 64)
        live = (s[:, start] != 0) | (s[:, end] != 0)  # (cores that ran nothing on this chip keep zeros)
        if live.any():
            vals.append((s[live, start], s[live, end]))
    ref = int(vals[0][0][0])
    rel = lambda v: ((v - ref + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    return (max(int(rel(e).max()) for _, e in vals) - min(int(rel(b).min()) for b, _ in vals)) / clk


def run_phase(mesh, fn, reps=1):
    """fn() reps times, each synchronized: median wall ms."""
    ws = []
    for _ in range(reps):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh)
        ws.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ws)


@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("device_params", [FABT], indirect=True)
@pytest.mark.parametrize("routing", ["uniform", "zipf"])
def test_ep4_layer(mesh_device, routing):
    from models.demos.ws_expert.bench_util import pcc
    from models.demos.ws_expert.tests.test_ws_layer import Ref
    from models.demos.ws_expert.ws_lane import tables_tensor
    from models.demos.ws_expert.ws_shared import SCfg, build_shared, shared_weights

    mesh = mesh_device
    R, C = tuple(mesh.shape)
    N = R * C
    n = n_per_chip(LEN)
    lc = EP4Cfg(T=n)
    TX, EPC, K, H = lc.TX, lc.EPC, lc.K, lc.H
    G = int(os.environ.get("EP4_G", "2"))
    reps = int(os.environ.get("EP4_REPS", "3"))
    cfg, lane, l5 = lane_cfgs(mesh, lc, G)
    T, pos = lane_tables(mesh, cfg, lane)
    tab_t = tables_tensor(mesh, T["tabs"])
    clk = aiclk_mhz(0)
    # inputs and every phase's state
    host, dev = make_inputs(mesh, lc, routing)
    g = torch.Generator().manual_seed(7)
    w1 = (torch.randn(H, lc.IS, generator=g) / H**0.5).to(torch.bfloat16)
    w3 = (torch.randn(H, lc.IS, generator=g) / H**0.5).to(torch.bfloat16)
    w2 = (torch.randn(lc.IS, H, generator=g) / lc.IS**0.5).to(torch.bfloat16)
    wg, wu, wd = expert_weights(lc)
    ws_t = ttnn.from_torch(
        torch.cat(own_slices(cfg, lane, T, lc, wg, wu, wd, C), 0),
        dtype=cfg.w_dt,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    scfg = SCfg(T=n, H=H, IS=lc.IS, S=S5, KSH=C)
    w13_t, w2s_t = shared_weights(mesh, scfg, w1, w3, w2)
    hout = head_outputs(mesh, lc)
    grp = grp_tensor(mesh, lc)
    res = dispatch_resources(mesh, lc)
    xbuf0 = xbuf_tensor(mesh, lc, host["x"])
    pos_t = pos_tensor(mesh, pos)
    louts = lists_outputs(mesh, lc)
    O = o_tensor(mesh, lc)
    P = p_tensor(mesh, lc)
    cnt_t = counters_tensor(mesh)
    fence = ttnn.empty(
        [16, 16], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    y_t = ttnn.empty(
        [TX, H], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    state = dict(xbuf=xbuf0)

    # (each phase builds its program, runs it and frees its L1 tensors: the phases' L1 does not fit at once; the
    # home pass's counters stay)
    def ph_head():
        run_free(head_program(mesh, lc, dev, hout))

    def ph_exch():
        exchange(hout["w"], grp)

    def ph_disp():
        state["xbuf"] = dispatch(lc, xbuf0, route_pieces(lc, hout), res, N)

    # (each custom phase keeps its stamps until read: a function of chip -> device us)
    def ph_lists():
        b = build_lists(mesh, lc, grp, pos_t, louts)
        run_free(b, keep=[b[2]])
        return [b[2]], lambda ci: span_us([b[2]], ci, clk)

    def ph_experts():
        from models.demos.ws_expert.tests.test_ws_lane import chip_time

        b = lane_program(mesh, lc, cfg, lane, l5, T, tab_t, state["xbuf"], louts, ws_t, O)
        run_free(b, keep=[b[2]])
        if "counts" not in state:
            ttnn.synchronize_device(mesh)
            state["counts"] = [(c_.reshape(-1).to(torch.int64) & 0xFFFFFFFF) for c_ in per_chip(louts["counts"])]
        Ms = lambda ci: [int(v) for v in state["counts"][ci]]
        return [b[2]], lambda ci: chip_time(cfg, Ms(ci), T, ci, ttnn.get_device_tensors(b[2])[ci], x0=512)[3] / clk

    def ph_shared():
        from models.demos.ws_expert.ws_shared import shared_span_us

        b = build_shared(mesh, scfg, dev["x_rm"], w13_t, w2s_t, P)
        sts = list(b[2]) if isinstance(b[2], (list, tuple)) else [b[2]]
        run_free(b[:3], keep=sts)
        return sts, lambda ci: shared_span_us(sts, ci, clk)

    def ph_comb():
        b = build_combine(mesh, lc, res, louts, O, P, cnt_t, fence)
        run_free(b, keep=[cnt_t, b[2], b[3]])

        def span(ci):
            a_, p_ = span_us([b[2]], ci, clk), span_us([b[3]], ci, clk)
            if ci in (0, 1) and "comb_diag" not in state:
                g = lambda t_: (ttnn.to_torch(ttnn.get_device_tensors(t_)[ci]).to(torch.int64) & 0xFFFFFFFF).reshape(
                    -1, 8 * 64
                )
                sa, sp = g(b[2]), g(b[3])
                live = sp[:, 63] != 0
                k = lambda v: [round(float(x) / clk, 1) for x in v]
                logger.info(
                    f"EP4 combine diag chip {ci}: ports rows sent {sp[live, 10].tolist()}, router-space wait us "
                    f"{k(sp[live, 11])}, loop us {k(sp[live, 12])}; adders mean / max us: credit wait "
                    f"{k([sa[:, 7].double().mean(), sa[:, 7].max()])}, row wait {k([sa[:, 8].double().mean(), sa[:, 8].max()])}, "
                    f"O reads wait {k([sa[:, 5].double().mean(), sa[:, 5].max()])}, CB space {k([sa[:, 6].double().mean(), sa[:, 6].max()])}, "
                    f"compute in {k([sa[:, 1].double().mean(), sa[:, 1].max()])}, out {k([sa[:, 2].double().mean(), sa[:, 2].max()])}"
                )
                if ci == 1:
                    state["comb_diag"] = True
            dev_ms.setdefault("combine:adders", []).append(a_ / 1e3)
            dev_ms.setdefault("combine:ports", []).append(p_ / 1e3)
            return span_us([b[2], b[3]], ci, clk)

        return [b[2], b[3]], span

    def ph_home():
        b = build_home(mesh, lc, P, hout["w"], louts["rcounts"], cnt_t, y_t)
        run_free(b, keep=[cnt_t, b[2]])
        return [b[2]], lambda ci: span_us([b[2]], ci, clk)

    def ph_head_t():
        b = head_program(mesh, lc, dev, hout)
        run_free(b, keep=[b[2]])
        return [b[2]], lambda ci: router_span_us(b[2], ci, clk)["end"]

    phases = [("head", ph_head_t), ("exchange", ph_exch), ("dispatch", ph_disp), ("lists", ph_lists)]
    phases += [("experts", ph_experts), ("shared", ph_shared), ("combine", ph_comb), ("home", ph_home)]
    dev_ms = {}
    ys = []
    for it in range(max(2, reps)):
        for name, fn in phases:
            ttnn.synchronize_device(mesh)
            out = fn()
            ttnn.synchronize_device(mesh)
            if out is not None:
                sts, span = out
                dev_ms.setdefault(name, []).append(max(span(ci) for ci in range(N)) / 1e3)
                for t_ in sts:
                    ttnn.deallocate(t_)
        if it < 2:
            ys.append(per_chip(y_t))
    same = all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(ys[0], ys[1]))
    # the ttnn ops alone, traced (stateless)
    dev_ms["exchange"] = [traced(mesh, ph_exch, warm=5, meas=5)]
    dev_ms["dispatch"] = [traced(mesh, ph_disp, warm=5, meas=5)]
    clk2 = aiclk_mhz(0)
    # y against torch on sampled tokens of every chip (the device's routing and gate weights)
    ntok = int(os.environ.get("EP4_TOKENS", "32"))
    ref = Ref(lc, dict(host, wg=wg, wu=wu, wd=wd, w1=w1, w3=w3, w2=w2), cfg, scfg)
    tk = [rows32(v) for v in per_chip(hout["w"])]
    pccs = {}
    for ch in range(N):
        toks = torch.linspace(0, n - 1, ntok).round().long()
        ids, w, _ = unpack_topk(tk[ch][toks], K)
        rout = ref.routed(ch, toks, ids).to(torch.bfloat16).float()  # [n, K, H]: O rows (bf16)
        wb = w.to(torch.bfloat16).float()
        acc = torch.zeros(len(toks), H)
        for cc in range(C):  # slots in order; a partial is a bf16 row
            m = (ids // EPC) == cc
            part = (wb.unsqueeze(-1) * rout * m.unsqueeze(-1)).sum(1)
            acc += torch.where(m.any(-1, keepdim=True), part.to(torch.bfloat16).float(), torch.zeros_like(part))
        yref = (acc + ref.shared(ch, toks).to(torch.bfloat16).float()).to(torch.bfloat16)
        pccs[ch] = pcc(ys[1][ch][toks].float(), yref.float())
    order = ["head", "exchange", "dispatch", "lists", "experts", "shared", "combine", "home"]
    med = {k: round(statistics.median(dev_ms[k]), 3) for k in order}
    logger.info(
        f"EP4 layer SUMMARY {routing} {LEN} (n {n}) G {G}: y min PCC per chip {[round(v, 5) for v in pccs.values()]}; "
        f"two layers bit-identical {same}; per-phase device ms (max over chips, median; exchange / dispatch traced) "
        f"{med}, sum {sum(med.values()):.2f}; combine adders / ports (max over chips and runs) "
        f"{max(dev_ms['combine:adders']):.3f} / {max(dev_ms['combine:ports']):.3f}; aiclk {clk} -> {clk2} MHz"
    )
    assert same and min(pccs.values()) > 0.999

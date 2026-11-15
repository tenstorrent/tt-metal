# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Lane design (ws_lane.py). Host: the schedule completes under random timing (ws_lane_sched.replay), and the tables
drive a model of the device's queues (QO / QW / QF / QR and compute, with fabric delays) to completion without ever
landing a packet in a slot that still holds another expert. Device: the 2 x 4 tray (row snake, 2 lanes, G 2) and the
whole Galaxy torus (2 cycles x 2 links, G 4); every launch builds the program again (fresh arrival / credit words),
WS_WARM (default 20) warm-up + WS_MEAS (default 5) measured launches; the steady-state time per expert comes from the
group cores' expert-done stamps, PCC from the last launch (each group's last expert, or every expert with out_all)."""

import os
import random
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, pcc, quantize, to_dev
from models.demos.ws_expert.tests.test_ws_fab_bench import FAB, TORUS
from models.demos.ws_expert.ws_expert import interleave_gu
from models.demos.ws_expert.ws_fab import slice_layout
from models.demos.ws_expert.ws_group import GCfg
from models.demos.ws_expert.ws_lane import LCfg, build_lane, make_tables, slot_packets, tables_tensor
from models.demos.ws_expert.ws_lane_sched import replay, slot_types
from models.demos.ws_expert.ws_sf import torus_cycles

WARM = int(os.environ.get("WS_WARM", "20"))
MEAS = int(os.environ.get("WS_MEAS", "5"))


def table_replay(T, N, GU, P, seed=0, prob=0.5, delay=6):
    """One lane on N chips driven by the tables exactly as the group kernels read them, with random step costs and
    in-order fabric delays. True if every chip computes every expert; asserts that a packet only lands in a slot whose
    current occupant is its expert and that own reads only start once their slot is free."""
    rng = random.Random(seed)
    E, K = T["E"], T["K"]
    tabs = [T["tabs"][X, 0].tolist() for X in range(N)]
    pos_of = {T["meta"][(X, 0)]["pos"]: X for X in range(N)}
    chip_next = lambda X, d: pos_of[(T["meta"][(X, 0)]["pos"] + (1 if d == 0 else -1)) % N]
    ent = lambda X, s, k: tabs[X][tabs[X][1 + K + s] + 4 * k : tabs[X][1 + K + s] + 4 * k + 4]
    L = lambda X, s: tabs[X][1 + s]
    order = lambda X, n: tabs[X][16 + n]
    st = [
        dict(occ=[0] * K, sent=[[0, 0] for _ in range(K)], done=[False] * K, warr=[0] * K, cred=[0] * (2 * K),
             freew=[0] * K, w_n=0, w_dn=0, freed=0, comp=0, cstep=0, ok=[0] * K, obusy=None, olanded=0)
        for _ in range(N)
    ]
    flight = {(X, d): [] for X in range(N) for d in range(2)}  # in-order fabric queues: [ready step, fn]
    t = 0

    def land(Y, s2, k2, e):
        y = st[Y]
        assert y["occ"][s2] == k2 and ent(Y, s2, k2)[0] & 0xFFFF == e, f"chip {Y} slot {s2}: packet of {e} early"
        y["warr"][s2] += 1

    def setcred(Y, idx, v):
        st[Y]["cred"][idx] = max(st[Y]["cred"][idx], v)

    def go():
        return rng.random() < prob

    def step(X):
        u = st[X]
        moved = False
        # QO (BRISC): one chunk of the next own occupant
        if u["obusy"] is None and go():
            for s in range(K):
                while u["ok"][s] < L(X, s) and (ent(X, s, u["ok"][s])[0] >> 16) & 3 != 1:
                    u["ok"][s] += 1
                if u["ok"][s] < L(X, s) and u["freew"][s] >= u["ok"][s]:
                    assert u["occ"][s] == u["ok"][s], f"chip {X} slot {s}: own read into an occupied slot"
                    u["obusy"], u["olanded"] = s, 0
                    break
        if u["obusy"] is not None and go():
            s = u["obusy"]
            u["olanded"] += 1
            u["warr"][s] = P * u["ok"][s] + u["olanded"]
            if u["olanded"] == P:
                u["ok"][s] += 1
                u["obusy"] = None
            moved = True
        # QW
        cur = lambda n: u["occ"][order(X, n) >> 16] < L(X, order(X, n) >> 16) and (
            ent(X, order(X, n) >> 16, u["occ"][order(X, n) >> 16])[0] & 0xFFFF
        ) == order(X, n) & 0xFFFF
        if u["w_n"] < E and cur(u["w_n"]) and go():
            s = order(X, u["w_n"]) >> 16
            if u["warr"][s] >= P * u["occ"][s] + GU and u["w_n"] - u["freed"] < K:
                u["w_n"] += 1
                moved = True
        if u["w_dn"] < u["w_n"] and cur(u["w_dn"]) and go():
            s = order(X, u["w_dn"]) >> 16
            if u["warr"][s] >= P * (u["occ"][s] + 1):
                u["w_dn"] += 1
                moved = True
        # compute: gate / up of entry comp once pushed, then its down once pushed
        c = u["comp"]
        if c < E and go():
            if u["cstep"] == 0 and u["w_n"] > c:
                u["cstep"] = 1
                moved = True
            elif 0 < u["cstep"] < 4:
                u["cstep"] += 1
                moved = True
            elif u["cstep"] == 4 and u["w_dn"] > c:
                u["done"][order(X, c) >> 16] = True
                u["freed"] += 1
                u["comp"] += 1
                u["cstep"] = 0
                moved = True
        # QF
        for s in range(K):
            for d in range(2):
                if u["occ"][s] >= L(X, s) or not go():
                    continue
                w = ent(X, s, u["occ"][s])
                if not (w[0] >> 18) & (1 << d) or u["sent"][s][d] >= P:
                    continue
                k = u["sent"][s][d]
                if u["warr"][s] < P * u["occ"][s] + k + 1:
                    continue
                s2, k2 = w[1 + d] & 3, w[1 + d] >> 2
                if k == 0 and u["cred"][d * K + s2] < k2:
                    continue
                Y, e = chip_next(X, d), w[0] & 0xFFFF
                flight[(X, d)].append([t + rng.randint(1, delay), lambda Y=Y, s2=s2, k2=k2, e=e: land(Y, s2, k2, e)])
                u["sent"][s][d] += 1
                moved = True
        # QR
        for s in range(K):
            if u["occ"][s] >= L(X, s) or not u["done"][s]:
                continue
            w = ent(X, s, u["occ"][s])
            fwd = (w[0] >> 18) & 3
            if any((fwd >> d) & 1 and u["sent"][s][d] < P for d in range(2)):
                continue
            nk, nxt = (w[0] >> 20) & 3, u["occ"][s] + 1
            if nk == 1:
                u["freew"][s] = nxt
            elif nk >= 2:
                d = nk - 2
                Xp = chip_next(X, 1 - d)  # the chip it comes from (one hop back on direction d)
                assert chip_next(Xp, d) == X
                flight[(X, 1 - d)].append(
                    [t + rng.randint(1, delay), lambda Xp=Xp, i=d * K + s, v=nxt: setcred(Xp, i, v)]
                )
            u["occ"][s] = nxt
            u["sent"][s] = [0, 0]
            u["done"][s] = False
            moved = True
        return moved

    idle = 0
    while not all(st[X]["comp"] >= E for X in range(N)):
        moved = False
        for X in rng.sample(range(N), N):
            moved |= step(X)
        for q in flight.values():
            while q and q[0][0] <= t:
                q.pop(0)[1]()
                moved = True
        t += 1
        idle = 0 if moved or any(flight.values()) else idle + 1
        if idle > 2000:
            return False
    assert all(st[X]["occ"][s] == L(X, s) for X in range(N) for s in range(K))
    return True


@pytest.mark.parametrize("slots,keep", [("d0,d1,any", 1), ("d0,d1,d0d1,any", 2)], ids=["k3", "k4"])
@pytest.mark.parametrize("shape,nexp,G", [((2, 4), 32, 2), ((2, 4), 64, 2), ((8, 4), 640, 4)])
def test_lane_tables(shape, nexp, G, slots, keep):
    """The schedule completes under random timing, and the tables drive the device-queue model to completion."""
    cfg = GCfg(nexp=nexp, G=G)
    kw = dict(slots=slots, keep=keep)
    lc = LCfg(cycles=torus_cycles(*shape), **kw) if shape[0] > 2 else LCfg(**kw)
    GU, P = slot_packets(cfg, lc)
    T = make_tables(shape, cfg, lc)
    N = shape[0] * shape[1]
    ok, u = replay(N, T["E"], T["sch"], seed=0, P=P, GU=GU, keep=keep, types=slot_types(slots))
    assert ok
    logger.info(
        f"lane {shape} E {T['E']} slots {slots} keep {keep}: model util {T['sch']['util']:.3f}, random-timing {u:.3f}"
    )
    for seed in range(3):
        assert table_replay(T, N, GU, P, seed=seed), f"tables stuck (seed {seed})"
    own = [len(o) for o in T["own"]]
    assert len(set(own)) == 1, f"own experts a chip differ: {own}"


def make_inputs(mesh, cfg: GCfg, lc: LCfg, T, Ms, out_rows, seed=0):
    """Inputs (every chip the same tokens and counts): each chip's slices tensor holds its own experts (T["own"], at
    their local index), weights drawn in bf16; references per chip and group: its last expert, or (out_all) every
    expert in the chip's order."""
    N = mesh.get_num_devices()
    torch.manual_seed(seed)
    x = torch.randn(cfg.T + 1, cfg.H)
    x[cfg.T] = 0
    idx = torch.full((cfg.nexp, cfg.RMAX * 32), cfg.T, dtype=torch.int32)
    for e, M in enumerate(Ms):
        idx[e, :M] = torch.randperm(cfg.T)[:M].to(torch.int32)
    counts = torch.tensor(Ms, dtype=torch.int32).reshape(1, -1)
    wg = torch.randn(cfg.nexp, cfg.H, cfg.I, dtype=torch.bfloat16) / cfg.H**0.5
    wu = torch.randn(cfg.nexp, cfg.H, cfg.I, dtype=torch.bfloat16) / cfg.H**0.5
    wd = torch.randn(cfg.nexp, cfg.I, cfg.H, dtype=torch.bfloat16) / cfg.I**0.5
    per_chip = []
    for X in range(N):
        o = T["own"][X]
        per_chip.append(slice_layout(cfg, lc.CHT, wg[o].float(), wu[o].float(), wd[o].float()))
    rm = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    dev = dict(
        x_t=rm(x, ttnn.bfloat16),
        idx_t=rm(idx, ttnn.uint32),
        counts_t=rm(counts, ttnn.uint32),
        ws_t=ttnn.from_torch(
            torch.cat(per_chip, 0),
            dtype=cfg.w_dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        ),
        out_t=to_dev(torch.zeros(cfg.G * out_rows * 32, cfg.H), cfg.out_dt, mesh),
    )
    del per_chip
    cache = {}

    def ref(e):
        if e not in cache:
            wguq = quantize(interleave_gu(wg[e].float(), wu[e].float()), cfg.w_dt)
            wdq = quantize(wd[e].float(), cfg.w_dt)
            Xq = quantize(x[idx[e].long()].to(torch.bfloat16).float(), cfg.x_dt)
            wgq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 0].reshape(cfg.H, cfg.I)
            wuq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 1].reshape(cfg.H, cfg.I)
            h = quantize(torch.nn.functional.silu(Xq @ wgq) * (Xq @ wuq), cfg.h_dt)
            cache[e] = (h @ wdq)[: Ms[e]]
        return cache[e]

    refs = {}  # (chip, group) -> [(out tile row, expert, reference)]
    for X in range(N):
        for g in range(cfg.G):
            order = T["meta"][(X, g)]["order"]
            if lc.out_all:
                lst, row = [], g * out_rows
                for e in order:
                    if Ms[e]:
                        lst.append((row, e, ref(e)))
                    row += (Ms[e] + 31) // 32
                refs[(X, g)] = lst
            else:
                e = order[-1]
                refs[(X, g)] = [(g * out_rows, e, ref(e))] if Ms[e] else []
    return dev, refs


def chip_pcc(refs, X, G, out_chip):
    out = ttnn.to_torch(out_chip).float()
    vals = [pcc(out[row * 32 : row * 32 + r.shape[0]], r) for g in range(G) for row, e, r in refs[(X, g)]]
    return min(vals) if vals else 1.0


def chip_time(cfg, Ms, T, X, st_chip):
    """(cycles per expert in the steady state, tile utilization, mean UNPACK waits in K cycles) of one chip: from the
    latest end of any core's first expert to the latest end of its last."""
    NC, E = cfg.G * cfg.GC, T["E"]
    s = (ttnn.to_torch(st_chip).to(torch.int64) & 0xFFFFFFFF).reshape(NC, -1)
    rel = ((s - int(s[0, 0]) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    first, last = int(rel[:, 256].max()), int(rel[:, 256 + E - 1].max())
    span = max(last - first, 1)
    ops = 0
    for g in range(cfg.G):
        for e in T["meta"][(X, g)]["order"][1:]:
            ops += (Ms[e] + 31) // 32 * (2 * cfg.PAIRS * cfg.KT + cfg.HT * cfg.PAIRS)
    done = cfg.G * (E - 1)
    return span / done, ops * 16 / (NC * span), (s[:, 40:44].double().mean(dim=0) / 1000).round().tolist()


def risc_report(cfg, st_chip):
    """Per chip, means over the group cores (K cycles): NCRISC loop time in weights (QW / QF / QR) / x multicast /
    x ring / h ring, BRISC loop time in x multicast / x gather / h out / h leader / outputs; packets forwarded, credits
    sent, own experts read a core."""
    s = (ttnn.to_torch(st_chip).to(torch.int64) & 0xFFFFFFFF).reshape(cfg.G * cfg.GC, -1).double()
    k = lambda a, b: (s[:, a:b].mean(dim=0) / 1000).round().tolist()
    return dict(nc=k(470, 474), br=k(50, 55), fwd=float(s[:, 58].mean()), cred=float(s[:, 59].mean()),
                own=float(s[:, 56].mean()))


def run(mesh, cfg: GCfg, lc: LCfg, Ms, tag, warm=WARM, meas=MEAS):
    shape = tuple(mesh.shape)
    N = mesh.get_num_devices()
    T = make_tables(shape, cfg, lc)
    R = [(m + 31) // 32 for m in Ms]
    if lc.out_all:
        out_rows = max(sum(R[e] for e in T["meta"][(X, g)]["order"]) for X in range(N) for g in range(cfg.G))
    else:
        out_rows = max(R[T["meta"][(X, g)]["order"][-1]] for X in range(N) for g in range(cfg.G))
    out_rows = max(out_rows, 1)
    dev, refs = make_inputs(mesh, cfg, lc, T, Ms, out_rows)
    tab_t = tables_tensor(mesh, T["tabs"])
    clk = aiclk_mhz(0)
    logger.info(
        f"WSL {tag}: grid {mesh.compute_with_storage_grid_size()}, {T['E']} experts a lane, table {T['TLW']} words, "
        f"schedule model util {T['sch']['util']:.3f}, {len(T['own'][0])} own experts a chip, {lc}"
    )
    chips = list(range(N))
    rows, pcs = [], None
    for i in range(warm + meas):
        logger.debug(f"WSL {tag} launch {i + 1} of {warm + meas}")
        mpd, io, st = build_lane(
            mesh, cfg, lc, dev["x_t"], dev["idx_t"], dev["counts_t"], dev["ws_t"], dev["out_t"], tab_t, T, out_rows
        )
        ttnn.synchronize_device(mesh)  # (every chip's fresh arrival / credit words are written before any starts)
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh)
        if i >= warm:
            sts = ttnn.get_device_tensors(st)
            row = {ci: chip_time(cfg, Ms, T, ci, sts[ci]) for ci in chips}
            rows.append(row)
            del sts
            logger.info(
                f"WSL {tag} launch {i - warm + 1}/{meas}: util tile per chip "
                f"{[round(100 * row[ci][1], 1) for ci in chips]}"
            )
        if i == warm + meas - 1:
            outs = ttnn.get_device_tensors(dev["out_t"])
            pcs = {ci: chip_pcc(refs, ci, cfg.G, outs[ci]) for ci in chips}
            sts = ttnn.get_device_tensors(st)
            for ci in (0, N - 1):
                logger.info(f"WSL {tag} chip {ci} riscs: {risc_report(cfg, sts[ci])}")
            del outs, sts
        del mpd, io, st
    we = 3 * cfg.H * cfg.I * 576 / 1024
    for ci in chips:
        cyc = statistics.median(r[ci][0] for r in rows)
        ut = [100 * r[ci][1] for r in rows]
        logger.info(
            f"WSL {tag} chip {ci}: median {cyc:.0f} cyc/expert ({cyc / clk:.2f} us), util tile median "
            f"{statistics.median(ut):.1f}% (min {min(ut):.1f}, max {max(ut):.1f}), weights in "
            f"{we / (cyc / clk * 1e3):.0f} GB/s; waits (K) tilize/x/weight/h {rows[-1][ci][2]}; pcc {pcs[ci]:.5f}"
        )
    allu = [100 * r[ci][1] for r in rows for ci in chips]
    logger.info(
        f"WSL {tag} SUMMARY: {warm} warm-up + {meas} measured launches, aiclk {clk} MHz, util tile over chips and "
        f"launches median {statistics.median(allu):.1f}% (min {min(allu):.1f}, max {max(allu):.1f}), "
        f"min pcc {min(pcs.values()):.5f}"
    )
    return rows, pcs


def lcfg_from_env(**kw):
    """LCfg with overrides from WS_LANE (e.g. "nbuf=3;D=4;out_all=1;slots=d0,d1,d0d1,any")."""
    for item in filter(None, os.environ.get("WS_LANE", "").split(";")):
        k, v = item.split("=")
        kw[k] = bool(int(v)) if k == "out_all" else (v if k == "slots" else int(v))
    return LCfg(**kw)


def gcfg(nexp, G, T):
    # (the token-row ring shrunk so that the third weight slot fits: BK 20, NX 9, LOOK 18; MMAX = T: no multi-pass)
    return GCfg(nexp=nexp, G=G, T=T, MMAX=T, BK=20, NX=9, LOOK=18)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
@pytest.mark.parametrize("M", [586, 2048, 0, 5], ids=["m586", "m2048", "mzero", "mfive"])
def test_ws_lane_tray(mesh_device, M):
    """One 2 x 4 tray: the row-snake cycle x 2 links = 2 lanes, G 2, 32 experts (16 a lane)."""
    cfg, lc = gcfg(32, 2, 8192), lcfg_from_env()
    Ms = [M] * cfg.nexp
    warm = int(os.environ.get("WS_LANE_WARM", WARM))
    rows, pcs = run(mesh_device, cfg, lc, Ms, f"tray M {M}", warm=warm)
    assert min(pcs.values()) > 0.999


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TORUS], indirect=True)
@pytest.mark.parametrize("nexp,T,M", [(128, 8192, 586), (640, 31250, 586)], ids=["e128m586", "e640m586"])
def test_ws_lane_glx(mesh_device, nexp, T, M):
    """The whole Galaxy (8 x 4, FABRIC_2D_TORUS_XY): two edge-disjoint Hamiltonian cycles x 2 links = 4 lanes, G 4."""
    cfg = gcfg(nexp, 4, T)
    lc = lcfg_from_env(cycles=torus_cycles(*tuple(mesh_device.shape)))
    Ms = [M] * cfg.nexp
    warm = int(os.environ.get("WS_LANE_WARM", WARM))
    rows, pcs = run(mesh_device, cfg, lc, Ms, f"glx {nexp} experts M {M}", warm=warm)
    assert min(pcs.values()) > 0.999


def dyn_counts(nexp, nchip, T, last, seed=3):
    """Per-chip, per-expert row counts (test_ws_fab_bench.dyn_counts on the lane design's orders): zeros, tiny, tile
    edges, above the old MMAX (2048), every token of a chip on one expert; the extremes on each chip's last expert of
    each group (last[c][g]), where the last-expert check sees them."""
    G = len(last[0])
    g_ = torch.Generator().manual_seed(seed)
    Ms = torch.zeros(nchip, nexp, dtype=torch.int64)
    pick = [0, 0, 1, 31, 32, 33, 300, 586, 586, 700, 1500, 2049, 4000, 7000]
    for c in range(nchip):
        for e in range(nexp):
            Ms[c, e] = pick[int(torch.randint(len(pick), (1,), generator=g_))]
    huge_g = int(os.environ.get("WSD_HUGE_GROUP", "0"))  # which group's last expert takes chip 1's huge M
    huge_m = int(os.environ.get("WSD_HUGE_M", str(T)))
    others = [k for k in range(G) if k != huge_g]
    for c in range(nchip):
        Ms[c, last[c][huge_g]] = huge_m if c == 1 else (0 if c % 2 else 1)  # chip 1: one huge expert; zeros elsewhere
        extra = [2049 + 7 * c, 5000 + 113 * c, int(torch.randint(0, 600, (1,), generator=g_))]
        for k, m in zip(others, extra):
            Ms[c, last[c][k]] = m
    return Ms


def run_dynm(mesh, cfg: GCfg, lc: LCfg, tag, check_chips):
    """Correctness for arbitrary per-expert, per-chip M: every chip its own counts and token lists (WSD_SEED,
    WSD_HUGE_GROUP, WSD_HUGE_M, WSD_LAUNCHES launches); WSD_ALL=1 checks every expert of chips 1 and 0 (WL_OUT_ALL),
    otherwise each group's last expert on check_chips."""
    shape = tuple(mesh.shape)
    nchip, Tn = mesh.get_num_devices(), cfg.T
    Tb = make_tables(shape, cfg, lc)
    last = [[Tb["meta"][(c, g)]["order"][-1] for g in range(cfg.G)] for c in range(nchip)]
    Ms = dyn_counts(cfg.nexp, nchip, Tn, last, seed=int(os.environ.get("WSD_SEED", "3")))
    Rs = (Ms + 31) // 32
    if lc.out_all:
        out_rows = max(
            int(sum(Rs[c, e] for e in Tb["meta"][(c, g)]["order"])) for c in range(nchip) for g in range(cfg.G)
        )
        check_chips = [c for c in (1, 0) if c < nchip]
    else:
        out_rows = max(int(Rs[c, last[c][g]]) for c in range(nchip) for g in range(cfg.G))
    out_rows = max(out_rows, 1)
    torch.manual_seed(0)
    x = torch.randn(Tn + 1, cfg.H)
    x[Tn] = 0
    idx = torch.full((nchip, cfg.nexp, cfg.RMAX * 32), Tn, dtype=torch.int32)
    for c in range(nchip):
        for e in range(cfg.nexp):
            M = int(Ms[c, e])
            if M:
                idx[c, e, :M] = torch.randperm(Tn)[:M].to(torch.int32)
    wg = torch.randn(cfg.nexp, cfg.H, cfg.I, dtype=torch.bfloat16) / cfg.H**0.5
    wu = torch.randn(cfg.nexp, cfg.H, cfg.I, dtype=torch.bfloat16) / cfg.H**0.5
    wd = torch.randn(cfg.nexp, cfg.I, cfg.H, dtype=torch.bfloat16) / cfg.I**0.5
    per_chip = []
    for c in range(nchip):
        o = Tb["own"][c]
        per_chip.append(slice_layout(cfg, lc.CHT, wg[o].float(), wu[o].float(), wd[o].float()))
    shard = lambda t, dt, lay, **kw: ttnn.from_torch(
        t, dtype=dt, layout=lay, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0), **kw
    )
    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    idx_t = shard(idx.reshape(nchip * cfg.nexp, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    counts_t = shard(Ms.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    ws_t = shard(torch.cat(per_chip, 0), cfg.w_dt, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    del per_chip
    out_t = ttnn.empty(
        [cfg.G * out_rows * 32, cfg.H],
        dtype=cfg.out_dt,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tab_t = tables_tensor(mesh, Tb["tabs"])
    logger.info(
        f"WSD {tag}: counts zeros {int((Ms == 0).sum())}, max {int(Ms.max())}, rows a chip {Ms.sum(1).tolist()[:4]}.., "
        f"output block {out_rows} tile rows, {lc}"
    )
    for launch in range(int(os.environ.get("WSD_LAUNCHES", "2"))):
        mpd, io, st = build_lane(mesh, cfg, lc, x_t, idx_t, counts_t, ws_t, out_t, tab_t, Tb, out_rows)
        ttnn.synchronize_device(mesh)
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh)
        del mpd, io, st
        logger.info(f"WSD {tag} launch {launch} done")

    def ref_of(c, e, M):
        Xq = quantize(x[idx[c, e, :M].long()].to(torch.bfloat16).float(), cfg.x_dt)
        wguq = quantize(interleave_gu(wg[e].float(), wu[e].float()), cfg.w_dt)
        wgq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 0].reshape(cfg.H, cfg.I)
        wuq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 1].reshape(cfg.H, cfg.I)
        h = quantize(torch.nn.functional.silu(Xq @ wgq) * (Xq @ wuq), cfg.h_dt)
        return h @ quantize(wd[e].float(), cfg.w_dt)

    outs = ttnn.get_device_tensors(out_t)
    worst = 1.0
    for c in check_chips:
        out = ttnn.to_torch(outs[c]).float()
        for g in range(cfg.G):
            order = Tb["meta"][(c, g)]["order"]
            if lc.out_all:
                logger.info(f"WSD {tag} chip {c} group {g} M sequence {[int(Ms[c, e]) for e in order]}")
                row, bad = g * out_rows, []
                for k, e in enumerate(order):
                    M = int(Ms[c, e])
                    if M:
                        p = pcc(out[row * 32 : row * 32 + M], ref_of(c, e, M))
                        worst = min(worst, p)
                        if p < 0.999:
                            bad.append((k, e, M, round(p, 5)))
                    row += int(Rs[c, e])
                logger.info(f"WSD {tag} chip {c} group {g}: experts below 0.999 (index, expert, M, pcc) {bad}")
            else:
                e = order[-1]
                M = int(Ms[c, e])
                if M == 0:
                    logger.info(f"WSD {tag} chip {c} group {g} expert {e}: M 0 (nothing to check)")
                    continue
                p = pcc(out[g * out_rows * 32 : g * out_rows * 32 + M], ref_of(c, e, M))
                worst = min(worst, p)
                logger.info(f"WSD {tag} chip {c} group {g} expert {e}: M {M} pcc {p:.5f}")
    logger.info(f"WSD {tag} SUMMARY: dynamic M on {nchip} chips, checked {list(check_chips)}, min pcc {worst:.5f}")
    assert worst > 0.999


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
def test_ws_lane_tray_dynm(mesh_device):
    """Dynamic M on the tray: 32 experts, T 8192, G 2."""
    cfg = gcfg(32, 2, 8192)
    run_dynm(mesh_device, cfg, lcfg_from_env(), "tray", check_chips=range(8))


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TORUS], indirect=True)
def test_ws_lane_glx_dynm(mesh_device):
    """Dynamic M on the whole Galaxy (torus, 4 lanes): 128 experts, T 31250 (MMAX = T)."""
    cfg = gcfg(128, 4, 31250)
    lc = lcfg_from_env(cycles=torus_cycles(*tuple(mesh_device.shape)))
    run_dynm(mesh_device, cfg, lc, "glx", check_chips=(0, 1, 2, 13, 31))

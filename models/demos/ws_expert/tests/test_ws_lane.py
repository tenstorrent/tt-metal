# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Lane design (ws_lane.py). Host: the schedule completes under random timing (ws_lane_sched.replay), and the tables
drive a model of the device's queues (QO / QW / QF / QR and compute, with fabric delays) to completion without ever
landing a packet in a slot that still holds another expert. Device: the 2 x 4 tray (row snake, 2 lanes, G 2) and the
whole Galaxy torus (2 cycles x 2 links, G 4); every launch builds the program again (fresh arrival / credit words),
WS_WARM (default 20) warm-up + WS_MEAS (default 5) measured launches; the steady-state time per expert comes from the
group cores' expert-done stamps, PCC from the last launch (every expert's rows are written by default (out_all); the
checked experts and chips follow WS_PCC_EXPERTS / WS_PCC_CHIPS)."""

import os
import random
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, pcc, quantize
from models.demos.ws_expert.tests.test_ws_fab_bench import FAB, TORUS
from models.demos.ws_expert.ws_expert import interleave_gu
from models.demos.ws_expert.ws_fab import slice_layout
from models.demos.ws_expert.ws_group import GCfg
from models.demos.ws_expert.ws_lane import LCfg, build_lane, make_tables, slot_packets, tables_tensor
from models.demos.ws_expert.ws_lane_sched import replay, slot_types
from models.demos.ws_expert.ws_sf import torus_cycles

WARM = int(os.environ.get("WS_WARM", "20"))
MEAS = int(os.environ.get("WS_MEAS", "5"))
# PCC sample: WS_PCC_EXPERTS = k checks each group's last expert plus k evenly spaced ones (default every expert),
# WS_PCC_CHIPS = comma-separated chips checked (default every chip); only the checked rows are read back
PCC_EXPERTS = int(os.environ.get("WS_PCC_EXPERTS", "-1"))
PCC_CHIPS = os.environ.get("WS_PCC_CHIPS", "")


def pcc_chips(N):
    return [int(c) for c in PCC_CHIPS.split(",") if c != "" and int(c) < N] if PCC_CHIPS else list(range(N))


def pcc_pick(entries):
    """The checked subset of a group's (row, expert) entries in compute order: all, or the last plus k evenly
    spaced (WS_PCC_EXPERTS)."""
    if PCC_EXPERTS < 0 or len(entries) <= PCC_EXPERTS + 1:
        return entries
    n = len(entries) - 1
    keep = {round(i * (n - 1) / max(PCC_EXPERTS - 1, 1)) for i in range(PCC_EXPERTS)} if PCC_EXPERTS else set()
    return [entries[i] for i in sorted(keep)] + [entries[-1]]


def out_rows_of(cfg: GCfg, lc: LCfg, T, R, N):
    """Tile rows of a group's output block: every expert's rows (out_all) or the last expert's."""
    if lc.out_all:
        rows = max(sum(R[e] for e in T["meta"][(X, g)]["order"]) for X in range(N) for g in range(cfg.G))
    else:
        rows = max(R[T["meta"][(X, g)]["order"][-1]] for X in range(N) for g in range(cfg.G))
    return max(rows, 1)


def out_tensor(mesh, cfg: GCfg, out_rows):
    return ttnn.empty(
        [cfg.G * out_rows * 32, cfg.H],
        dtype=cfg.out_dt,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


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
        dict(
            occ=[0] * K,
            sent=[[0, 0] for _ in range(K)],
            done=[False] * K,
            warr=[0] * K,
            cred=[0] * (2 * K),
            freew=[0] * K,
            w_n=0,
            w_dn=0,
            freed=0,
            comp=0,
            cstep=0,
            ok=[0] * K,
            obusy=None,
            olanded=0,
        )
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
        cur = (
            lambda n: u["occ"][order(X, n) >> 16] < L(X, order(X, n) >> 16)
            and (ent(X, order(X, n) >> 16, u["occ"][order(X, n) >> 16])[0] & 0xFFFF) == order(X, n) & 0xFFFF
        )
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
                s2, k2 = w[1 + d] & 7, w[1 + d] >> 3
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


@pytest.mark.parametrize(
    "slots,keep", [("d0,d1,any", 1), ("d0,d1,d0d1,any", 2), ("d0,d1,d0d1,any,any", 2)], ids=["k3", "k4", "k5"]
)
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


def make_inputs(mesh, cfg: GCfg, lc: LCfg, T, Ms, out_rows, seed=0, ref_last_only=False):
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
        per_chip.append(slice_layout(cfg, lc.CHT, wg[o].float(), wu[o].float(), wd[o].float(), ks=lc.ks))
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
        out_t=out_tensor(mesh, cfg, out_rows),
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

    refs = {}  # (chip, group) -> [(out tile row, expert, reference)], checked chips only
    for X in pcc_chips(N):
        for g in range(cfg.G):
            order = T["meta"][(X, g)]["order"]
            if lc.out_all:  # (ref_last_only: every expert is written, only the group's last one is checked)
                ent, row = [], g * out_rows
                for k, e in enumerate(order):
                    if Ms[e] and (not ref_last_only or k == len(order) - 1):
                        ent.append((row, e))
                    row += (Ms[e] + 31) // 32
                refs[(X, g)] = [(row, e, ref(e)) for row, e in pcc_pick(ent)]
            else:
                e = order[-1]
                refs[(X, g)] = [(g * out_rows, e, ref(e))] if Ms[e] else []
    return dev, refs


def out_rows_back(out_chip, row, n):
    """Tile rows [row, row + n) of one chip's output, read back alone."""
    W = out_chip.shape[-1]
    return ttnn.to_torch(ttnn.slice(out_chip, [row * 32, 0], [(row + n) * 32, W])).float()


def chip_pcc(refs, X, G, out_chip):
    """Min PCC over the chip's checked experts (1.0 if none is checked on this chip); experts noticeably below the
    format's floor are logged with their bad row tiles and core column blocks."""
    vals = []
    for g in range(G):
        for row, e, r in refs.get((X, g), []):
            o = out_rows_back(out_chip, row, (r.shape[0] + 31) // 32)[: r.shape[0]]
            p = pcc(o, r)
            vals.append(p)
            if p < 0.9998:
                err = (o - r).abs()
                bad_rows = sorted({int(i) // 32 for i in (err.amax(1) > 0.2 * r.abs().max()).nonzero().flatten()})
                bad_cols = sorted({int(i) // 192 for i in (err.amax(0) > 0.2 * r.abs().max()).nonzero().flatten()})
                logger.warning(
                    f"PCC chip {X} group {g} expert {e}: {p:.5f}, row tiles {bad_rows} core column blocks {bad_cols}"
                )
    return min(vals) if vals else 1.0


def chip_time(cfg, Ms, T, X, st_chip, x0=256):
    """(cycles per expert in the steady state, tile utilization, mean UNPACK waits in K cycles, end-to-end cycles) of
    one chip: steady state from the latest end of any core's first expert to the latest end of its last; end to end
    from the earliest loop start of any group core's RISC (compute, BRISC, NCRISC) to the latest compute end."""
    NC, E = cfg.G * cfg.GC, T["E"]
    s = (ttnn.to_torch(st_chip).to(torch.int64) & 0xFFFFFFFF).reshape(NC, -1)
    rel = ((s - int(s[0, 0]) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    first, last = int(rel[:, x0].max()), int(rel[:, x0 + E - 1].max())
    span = max(last - first, 1)
    ops = 0
    for g in range(cfg.G):
        for e in T["meta"][(X, g)]["order"][1:]:
            ops += (Ms[e] + 31) // 32 * (2 * cfg.PAIRS * cfg.KT + cfg.HT * cfg.PAIRS)
    done = cfg.G * (E - 1)
    t0 = int(min(rel[:, 0].min(), rel[:, 485].min(), rel[:, 496].min()))
    e2e = int(rel[:, 63].max()) - t0
    # per group: the end of its last NCRISC loop (every weight packet landed, forwarded, every slot released); the
    # weights-only time when the experts have no rows
    nc_end = [
        int((rel[g * cfg.GC : (g + 1) * cfg.GC, 485] + s[g * cfg.GC : (g + 1) * cfg.GC, 484]).max()) - t0
        for g in range(cfg.G)
    ]
    return (
        span / done,
        ops * 16 / (NC * span),
        (s[:, 40:45].double().mean(dim=0) / 1000).round().tolist(),
        e2e,
        nc_end,
    )


def timeline(cfg, Ms, T, X, st_chip):
    """Chip X's start-up and per-expert timeline (K cycles from the earliest compute start over the group cores):
    BRISC / NCRISC loop starts, compute start, then per expert of the compute order (group 0's): its rows and the
    median / max over all cores of its end."""
    NC, E = cfg.G * cfg.GC, T["E"]
    s = (ttnn.to_torch(st_chip).to(torch.int64) & 0xFFFFFFFF).reshape(NC, -1)
    t0 = int(s[:, 0].min())
    rel = lambda c: ((s[:, c] - t0 + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    k = lambda v: round(float(v) / 1000, 1)
    order = T["meta"][(X, 0)]["order"]
    ends = [
        (order[n], (Ms[order[n]] + 31) // 32, k(rel(256 + n).double().median()), k(rel(256 + n).max()))
        for n in range(min(E, 200))
    ]
    return dict(
        br_start=k(rel(496).median()),
        nc_start=k(rel(485).median()),
        compute_start_max=k(rel(0).max()),
        experts_e_rows_endmed_endmax=ends,
    )


def risc_report(cfg, st_chip):
    """Per chip, means over the group cores (K cycles): NCRISC loop time in weights (QW / QF / QR) / x multicast /
    x ring / h ring, BRISC loop time in x multicast / x gather / h out / h leader / outputs; packets forwarded, credits
    sent, own experts read a core."""
    s = (ttnn.to_torch(st_chip).to(torch.int64) & 0xFFFFFFFF).reshape(cfg.G * cfg.GC, -1).double()
    k = lambda a, b: (s[:, a:b].mean(dim=0) / 1000).round().tolist()
    return dict(
        nc=k(470, 474),
        nc_fu=k(478, 479),
        br=k(50, 55),
        br_fd_own=k(60, 62),
        busy_br_x_g_h_l_y=k(490, 495),
        br_span=k(495, 496),
        busy_nc_wyl_a_b_h=k(480, 484),
        nc_span=k(484, 485),
        fwd=float(s[:, 58].mean()),
        cred=float(s[:, 59].mean()),
        own=float(s[:, 56].mean()),
    )


def first_visible():
    """PCI index (/dev/tenstorrent/N, sysfs tenstorrent!N) of the first opened chip: TT_VISIBLE_DEVICES' first entry, or
    0 when every chip is open."""
    try:
        return int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])
    except ValueError:
        return 0


def run(mesh, cfg: GCfg, lc: LCfg, Ms, tag, warm=WARM, meas=MEAS):
    shape = tuple(mesh.shape)
    N = mesh.get_num_devices()
    T = make_tables(shape, cfg, lc)
    R = [(m + 31) // 32 for m in Ms]
    out_rows = out_rows_of(cfg, lc, T, R, N)
    # (WS_REF_LAST=1: every expert still written, only each group's last checked)
    dev, refs = make_inputs(mesh, cfg, lc, T, Ms, out_rows, ref_last_only=os.environ.get("WS_REF_LAST", "0") == "1")
    tab_t = tables_tensor(mesh, T["tabs"])
    clk_dev = first_visible()
    clk = aiclk_mhz(clk_dev)
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
                f"{[round(100 * row[ci][1], 1) for ci in chips]}, end-to-end us per chip "
                f"{[round(row[ci][3] / clk, 1) for ci in chips]}"
            )
        if i == warm + meas - 1:
            clk_end = aiclk_mhz(clk_dev)
            outs = ttnn.get_device_tensors(dev["out_t"])
            pcs = {ci: chip_pcc(refs, ci, cfg.G, outs[ci]) for ci in pcc_chips(N)}
            sts = ttnn.get_device_tensors(st)
            for ci in (0, N - 1):
                logger.info(f"WSL {tag} chip {ci} riscs: {risc_report(cfg, sts[ci])}")
                if os.environ.get("WS_PERCORE"):
                    sc = (ttnn.to_torch(sts[ci]).to(torch.int64) & 0xFFFFFFFF).reshape(cfg.G * cfg.GC, -1)
                    for col, name in (
                        (41, "x"),
                        (42, "weight"),
                        (43, "h"),
                        (44, "landing"),
                        (45, "mm gu"),
                        (46, "fold"),
                        (47, "mm down"),
                        (64, "pack p wait"),
                        (65, "pack h wait"),
                        (66, "pack y wait"),
                        (67, "math dest wait"),
                    ):
                        logger.info(
                            f"WSL chip {ci} per core (group 0 col 0 rows 0-9 | col 1 | group 1 ...) {name} K: "
                            f"{(sc[:, col] // 1000).tolist()}"
                        )
                    logger.info(f"WSL chip {ci} per core BRISC ks busy K: {(sc[:, 497] // 1000).tolist()}")
                logger.info(f"WSL {tag} chip {ci} timeline: {timeline(cfg, Ms, T, ci, sts[ci])}")
            del outs, sts
        del mpd, io, st
    we = 3 * cfg.H * cfg.I * 576 / 1024
    # bytes written a chip per expert: every expert's rows (out_all) or only each group's last
    ob = sum(R) * 32 * cfg.H * 2 / len(R) if lc.out_all else 0.0
    for ci in chips:
        cyc = statistics.median(r[ci][0] for r in rows)
        ut = [100 * r[ci][1] for r in rows]
        logger.info(
            f"WSL {tag} chip {ci}: median {cyc:.0f} cyc/expert ({cyc / clk:.2f} us), util tile median "
            f"{statistics.median(ut):.1f}% (min {min(ut):.1f}, max {max(ut):.1f}), end to end "
            f"{statistics.median(r[ci][3] for r in rows) / clk:.1f} us, weights in "
            f"{we / (cyc / clk * 1e3):.0f} GB/s, outputs {ob / (cyc / clk * 1e3):.0f} GB/s; waits (K) "
            f"tilize/x/weight/h/h-drain {rows[-1][ci][2]}; pcc {pcs.get(ci, float('nan')):.5f}"
        )
    allu = [100 * r[ci][1] for r in rows for ci in chips]
    logger.info(
        f"WSL {tag} SUMMARY: {warm} warm-up + {meas} measured launches, aiclk {clk} MHz (chip {clk_dev}, {clk_end} at "
        f"the end), util tile over chips and "
        f"launches median {statistics.median(allu):.1f}% (min {min(allu):.1f}, max {max(allu):.1f}), "
        f"min pcc {min(pcs.values()):.5f} (chips {pcc_chips(N) if PCC_CHIPS else 'all'}, "
        f"{'every' if PCC_EXPERTS < 0 else f'last + {PCC_EXPERTS}'} expert(s) a group, "
        f"out_all {lc.out_all}), end to end (slowest chip) median "
        f"{statistics.median(max(r[ci][3] for ci in chips) for r in rows) / clk:.1f} us, NCRISC end per group "
        f"(slowest chip) median "
        f"{[round(statistics.median(max(r[ci][4][g] for ci in chips) for r in rows) / clk, 1) for g in range(cfg.G)]} us"
    )
    return rows, pcs


def lcfg_from_env(**kw):
    """LCfg with overrides from WS_LANE (e.g. "nbuf=3;D=4;out_all=1;slots=d0,d1,d0d1,any")."""
    for item in filter(None, os.environ.get("WS_LANE", "").split(";")):
        k, v = item.split("=")
        flags = ("out_all", "mcast_linked", "arr_write", "mc_noc1", "bcast", "own_read", "gather", "mux_noc1")
        flags += ("noc_split", "nofab", "ucast", "xchain", "hchain", "xhop", "jump", "fprio", "fnb", "harrw", "fasync")
        flags += ("ks", "ks_pall", "wtwice", "arr_nf", "ks_ff", "ks_fz", "ybarrier", "y_rm", "y_nowrite", "y_posted")
        kw[k] = bool(int(v)) if k in flags else (v if k == "slots" else int(v))
    return LCfg(**kw)


def gcfg(nexp, G, T):
    # (the token-row ring shrunk so that the third weight slot fits: BK 20, NX 9, LOOK 18; MMAX = T: no multi-pass)
    # WS_XORDER: x sends in stream order (block s waits for s - xorder to land; 0 off), default GCfg's 2
    xo = int(os.environ.get("WS_XORDER", "2"))
    # the chains (ucast) spread a block's delivery over ~10 K cycles, so they need a deeper x ring than the multicasts:
    # 16 slots (2.7 rows of x), paid for by bfp8 h (half the h ring), 2 row-major gather blocks instead of 3 and 2
    # staging blocks of slack instead of 4 (LCfg.xst); the multicast build (ucast=0) keeps 9 slots, bf16 h, XRM 3
    lc0 = lcfg_from_env()
    chains = lc0.ucast
    # WS_XRING "BK,NX,LOOK": the token-row ring (x block K-tiles, ring slots, tilize look-ahead)
    bk, nx, look = (int(v) for v in os.environ.get("WS_XRING", "20,16,18" if chains else "20,9,18").split(","))
    # WS_GCFG "h_dt=bfp8;XRM=2;NS=6": other GCfg overrides (h_dt bfp8 / bf16, ints)
    kw = dict(h_dt=ttnn.bfloat8_b, XRM=2) if chains else {}
    if lc0.ks:  # (K split: XRM in whole 12-tile rows; the down lags 2 rows - single-chip prototype's best)
        kw.update(XRM=2, dn_lag=2)
    for item in filter(None, os.environ.get("WS_GCFG", "").split(";")):
        k, v = item.split("=")
        kw[k] = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[v] if k.endswith("_dt") else int(v)
    return GCfg(nexp=nexp, G=G, T=T, MMAX=T, BK=bk, NX=nx, LOOK=look, xorder=xo, **kw)


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


MSWEEP = [32, 64, 128, 256, 384, 512, 586, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
@pytest.mark.parametrize("M", MSWEEP, ids=[f"m{m}" for m in MSWEEP])
def test_ws_lane_tray_msweep(mesh_device, M):
    """test_ws_lane_tray over constant M per expert from 1 row tile to T (32 experts, T 8192)."""
    test_ws_lane_tray(mesh_device, M)


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TORUS], indirect=True)
@pytest.mark.parametrize(
    "nexp,T,M", [(128, 8192, 586), (640, 31250, 586), (640, 31250, 2048)], ids=["e128m586", "e640m586", "e640m2048"]
)
def test_ws_lane_glx(mesh_device, nexp, T, M):
    """The whole Galaxy (8 x 4, FABRIC_2D_TORUS_XY): two edge-disjoint Hamiltonian cycles x 2 links = 4 lanes, G 4."""
    cfg = gcfg(nexp, 4, T)
    lc = lcfg_from_env(cycles=torus_cycles(*tuple(mesh_device.shape)))
    Ms = [M] * cfg.nexp
    warm = int(os.environ.get("WS_LANE_WARM", WARM))
    rows, pcs = run(mesh_device, cfg, lc, Ms, f"glx {nexp} experts M {M}", warm=warm)
    assert min(pcs.values()) > 0.999


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize(
    "G,nexp,T,M",
    [
        (4, 64, 8192, 586),
        (4, 64, 8192, 0),
        (4, 64, 8192, 5),
        (2, 320, 31250, 586),
        (4, 640, 31250, 586),
        (2, 320, 31250, 2048),
        (4, 640, 31250, 2048),
    ],
    ids=["g4e64m586", "g4e64mzero", "g4e64mfive", "g2e320m586", "g4e640m586", "g2e320m2048", "g4e640m2048"],
)
def test_ws_lane_chip(mesh_device, G, nexp, T, M):
    """Single-chip proxy of the lane design (LCfg.nofab): one chip (TT_VISIBLE_DEVICES), no fabric, every expert the
    chip's own and read from DRAM into its slots (nothing forwarded, no credits), the G groups on the chip's tokens as
    on the Galaxy (160 experts a group for e320 / e640, T 31250); slots of any kind (keep 0), the rest from WS_LANE."""
    cfg = gcfg(nexp, G, T)
    lc = lcfg_from_env(cycles=[[(0, 0)]] * ((G + 1) // 2), nofab=True, slots="any,any,any,any", keep=0)
    Ms = [M] * cfg.nexp
    warm = int(os.environ.get("WS_LANE_WARM", WARM))
    rows, pcs = run(mesh_device, cfg, lc, Ms, f"chip G {G} {nexp} experts M {M}", warm=warm)
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
        per_chip.append(slice_layout(cfg, lc.CHT, wg[o].float(), wu[o].float(), wd[o].float(), ks=lc.ks))
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


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
def test_ws_lane_tray_soak(mesh_device):
    """Soak on the tray at the 640-expert torus load of a lane (160 experts a lane, T 31250, M 586; G 2): launches
    until WS_SOAK_SECS (default 900) or WS_SOAK_MAX launches, the util of every 50th launch logged, PCC (the checked
    experts, see WS_PCC_EXPERTS) on the last."""
    import time

    cfg, lc = gcfg(320, 2, 31250), lcfg_from_env()
    Ms = [586] * cfg.nexp
    mesh = mesh_device
    N = mesh.get_num_devices()
    T = make_tables(tuple(mesh.shape), cfg, lc)
    out_rows = out_rows_of(cfg, lc, T, [(m + 31) // 32 for m in Ms], N)
    dev, refs = make_inputs(mesh, cfg, lc, T, Ms, out_rows, ref_last_only=True)
    tab_t = tables_tensor(mesh, T["tabs"])
    secs, mx = float(os.environ.get("WS_SOAK_SECS", "900")), int(os.environ.get("WS_SOAK_MAX", "100000"))
    logger.info(f"WSK soak: {T['E']} experts a lane, {secs:.0f} s or {mx} launches, {lc}")
    t0, i, utils = time.time(), 0, []
    while i < mx and time.time() - t0 < secs:
        mpd, io, st = build_lane(
            mesh, cfg, lc, dev["x_t"], dev["idx_t"], dev["counts_t"], dev["ws_t"], dev["out_t"], tab_t, T, out_rows
        )
        ttnn.synchronize_device(mesh)
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh)
        i += 1
        if i % 50 == 0:
            sts = ttnn.get_device_tensors(st)
            u = [100 * chip_time(cfg, Ms, T, ci, sts[ci])[1] for ci in range(N)]
            utils += u
            logger.info(f"WSK soak launch {i}, {time.time() - t0:.0f} s: util tile per chip {[round(x, 1) for x in u]}")
            del sts
        del mpd, io, st
    outs = ttnn.get_device_tensors(dev["out_t"])
    pcs = {ci: chip_pcc(refs, ci, cfg.G, outs[ci]) for ci in pcc_chips(N)}
    logger.info(
        f"WSK soak SUMMARY: {i} launches in {time.time() - t0:.0f} s without a hang, util tile median "
        f"{statistics.median(utils) if utils else 0:.1f}% over the logged launches, min pcc {min(pcs.values()):.5f}"
    )
    assert min(pcs.values()) > 0.999


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
def test_ws_lane_tray_trace(mesh_device):
    """Debug: x block timeline on chip 0, group 0 (needs WL_DEFS=WG_TRACE0=<s0>): for the 40 blocks from stream index
    s0 (an expert's start), the origin's send time and the push time (block handed to compute) on every core, relative
    to the first send; after WS_TRACE_LAUNCHES launches (the last one traced)."""
    s0 = int(dict(d.partition("=")[::2] for d in os.environ.get("WL_DEFS", "").split(",") if d)["WG_TRACE0"])
    cfg, lc = gcfg(32, 2, 8192), lcfg_from_env()
    Ms = [586] * cfg.nexp
    T = make_tables(tuple(mesh_device.shape), cfg, lc)
    out_rows = (586 + 31) // 32
    dev, refs = make_inputs(mesh_device, cfg, lc, T, Ms, out_rows)
    tab_t = tables_tensor(mesh_device, T["tabs"])
    n_l = int(os.environ.get("WS_TRACE_LAUNCHES", "4"))
    for i in range(n_l):
        if i:
            del mpd, io, st
        mpd, io, st = build_lane(
            mesh_device,
            cfg,
            lc,
            dev["x_t"],
            dev["idx_t"],
            dev["counts_t"],
            dev["ws_t"],
            dev["out_t"],
            tab_t,
            T,
            out_rows,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh_device)
    s = (ttnn.to_torch(ttnn.get_device_tensors(st)[0]).to(torch.int64) & 0xFFFFFFFF).reshape(cfg.G * cfg.GC, -1)
    sent = [int(s[(c % 2) * 10 + (c // 2) % 10, 128 + c]) for c in range(40)]
    base = min(v for v in sent if v)
    rel = lambda v: ((v - base + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    # chain order of a block from origin row r: the push times on its own column by row, then the partner column
    hops = []
    for c in range(40):
        oc, orow = c % 2, (c // 2) % 10
        own = [rel(int(s[oc * 10 + y, 192 + c])) for y in range(10)]
        par = [rel(int(s[(1 - oc) * 10 + y, 192 + c])) for y in range(10)]
        snd = rel(sent[c])
        o = oc * 10 + orow
        seen = rel(int(s[o, 320 + c])) if oc == 0 and int(s[o, 320 + c]) else None
        last = max(own + par)
        if seen is not None:
            logger.info(f"TRACE block {s0 + c}: origin saw it ready (tilized) at {seen}, sent at {snd} (+{snd - seen})")
        hops.append(last - snd)
        logger.info(
            f"TRACE block {s0 + c} origin (col {oc}, row {orow}): sent {snd}, landed-in-compute own column by row "
            f"{[v - snd for v in own]}, partner column {[v - snd for v in par]}, last {last - snd}"
        )
    gaps = [rel(sent[c + 1]) - rel(sent[c]) for c in range(39)]
    logger.info(
        f"TRACE summary: send-to-last-core median {statistics.median(hops):.0f} cycles, sends every "
        f"{statistics.median(gaps):.0f} cycles (median gap between consecutive blocks' sends)"
    )

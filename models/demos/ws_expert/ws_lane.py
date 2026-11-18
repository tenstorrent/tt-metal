# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert lane design: every chip runs the G groups on its own tokens; the weights of group g's experts travel lane
g (one Hamiltonian cycle and link, both directions), forwarded by the group cores themselves. Core j of a group sends
its slice of an expert one hop, packet by packet, into the same core of the next chip through the group's mux core of
that direction (fabric_mux: one full-size channel a core), straight into a weight slot. Every group core has K weight
slots at the same address on every chip, each typed (LCfg.slots): by default slot 0 takes experts arriving on
direction 0, slot 1 on direction 1, slot 2 any, including the chip's own experts, which the core's BRISC reads from
DRAM. An expert's down projection is drained
before the next expert starts, so a slot frees once its expert is computed and forwarded on.

The order of everything comes from a static schedule (ws_lane_sched.lane_sched, the same on all lanes): per chip and
slot the experts it holds in turn, per chip the compute order. The group cores follow it through a table per chip and
group (make_tables, layout in kernels/wl_reader.cpp); a slot's next occupant is only sent (or read) once the slot freed
the previous one: a credit from the receiving core to the sending one, one hop back, through the other direction's
mux. Every wait is on an event earlier in the schedule's model time, so the lanes cannot deadlock (ws_lane_sched).

Core layout (group_layout): group g in a pair of logical columns (contiguous in NoC x), its two mux cores in a column
west of it (rows MUX_ROWS): the cores reach their muxes on NOC1 (north, then west) without entering another
group's columns, whose linked multicasts run on both NoCs.
"""

import os
from dataclasses import dataclass

import torch

import ttnn
from models.demos.ws_expert.bench_util import TILE_BYTES, cb, crs, kpath, stamps_tensor
from models.demos.ws_expert.ws_expert import noc, noc_rects, sharded_l1
from models.demos.ws_expert.ws_fab import FAB_DEFS, node
from models.demos.ws_expert.ws_group import GCfg
from models.demos.ws_expert.ws_lane_sched import lane_sched, slot_types
from models.demos.ws_expert.ws_sf import snake

LANE_PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11)]
MUX_COL = [0, 3, 6, 9]
MUX_ROWS = tuple(int(v) for v in os.environ.get("WS_MUXROWS", "4,5").split(","))  # direction 0, direction 1
KIND_OWN, KIND_D0, KIND_D1 = 1, 2, 3
fabric_mux = ttnn.experimental.fabric_mux


@dataclass
class LCfg:
    CHT: int = 24  # tiles a packet (24 x 576 B bfp4 = 13.8 KB): 15 packets a slice, the first 10 gate / up
    NL: int = 2  # links a cycle edge (lanes = cycles x links)
    cycles: tuple = None  # Hamiltonian cycles (mesh coordinates in order); default: the row snake
    nbuf: int = 2  # mux buffers a channel
    D: int = 4  # own-expert chunk reads in flight
    slots: str = "d0,d1,any"  # weight slots a core and the experts each takes (ws_lane_sched.slot_types)
    keep: int = 1  # schedule: slots an owner leaves free when it injects
    out_all: bool = True  # every expert's down rows to DRAM (WL_OUT_ALL); False: only each group's last (old default)
    wpoll: int = 2  # the reader serves its weight queues every wpoll turns of its loop (WL_WPOLL)
    mcast_linked: bool = False  # group multicast payloads linked to their flags (WL_MCAST_LINKED)
    arr_write: bool = False  # arrival counts as plain writes after the payloads, no router atomics (WL_ARR_WRITE)
    mc_noc1: bool = False  # column 0 multicasts from its NCRISC on NOC1 too, one x multicast at a time (WL_MC_NOC1)
    # one multicaster a group a NoC, at the NoC's multicast start corner (WL_BCAST): x blocks funneled into core 0 and
    # multicast from its BRISC on NOC0, h rows gathered on core 19 and multicast from its NCRISC on NOC1, one at a time
    bcast: bool = False
    xf: int = 2  # funnel slots of x blocks on core 0 (WL_XF)
    own_read: bool = True  # own experts read from DRAM into their slots; False (WL_NO_OWN_READ): marked arrived unread
    gather: bool = True  # token rows gathered from DRAM; False (WG_NO_GATHER, a diagnostic): x staged unread
    mux_noc1: bool = False  # the fabric mux kernels on NOC1 (BRISC, writer NoC) instead of NOC0
    # NOC0 for the group multicasts only (all from the BRISCs) and NOC1 for every DRAM read (moved to the NCRISCs) and
    # the muxes, so nothing enters a group rectangle from outside on its multicasts' NoC (WL_NOC_SPLIT); with bcast,
    # core 0's BRISC (NOC0's start corner) multicasts both the x blocks and the h rows, fed on NOC1
    noc_split: bool = False
    # one chip, no fabric (WL_NO_FABRIC): no mux kernels, every expert the chip's own and read from DRAM, nothing
    # forwarded; with cycles of that one chip, the per-chip load of the lane design without its fabric traffic. On a
    # mesh every chip runs the same one-chip program on its own inputs (the tables of the one-chip cycles on each)
    nofab: bool = False
    # row-major outputs (WL_Y_RM): each down row of each expert to its own row of a row-major [rows, H] output, the
    # row given by the expert's destination list (build_lane's dest_t, the token lists' layout); needs noc_split or ucast
    y_rm: bool = False
    y_nowrite: bool = False  # (a diagnostic, WL_Y_NOWRITE) y_rm without the row writes: destinations read, rows dropped
    y_posted: bool = True  # y_rm's row writes posted (no acks; WL_Y_POSTED)
    # no NoC multicasts (WL_UCAST): every x block and h row goes to the group's 20 cores as unicast writes, payloads then
    # flags (the group multicasts wedged the NoC under the lane design's other traffic); False: multicasts as before
    ucast: bool = True
    # with ucast: x blocks as a unicast chain (origin -> partner column, then row by row down on NOC0 / up on NOC1, each
    # link once; WL_XCHAIN) instead of the 20-way fan-out; the h rows stay a fan-out
    xchain: bool = True
    # with xchain: slot credits hop by hop (each writer checks only its target core; WL_XHOP). Off: no gain on the
    # tray (37.9% vs 38.1% at M 586) and it deadlocked within 50 soak launches (cores of a chain drift apart)
    xhop: bool = False
    xst: int = -1  # x tilize staging blocks a core beyond the look-ahead's share (CB 8); -1: 2 with ucast, else 4
    jump: bool = False  # chains: the origin also sends to row (r + 5) % 10, at most 4 hops, not 9 (WL_JUMP)
    hchain: bool = True  # with ucast: the h row parts as the same kind of chain (WL_HCHAIN)
    # chains: pieces a hop (cut-through): an x block / h row part moves in xp / hp pieces, each with its own flag word
    # (word p of the slot's 16 B), so a forwarder passes piece p on while piece p + 1 is still arriving (WL_XP / WL_HP).
    # Tray, with fnb: xp 2 48.7% / 58.2% at M 586 / 2048 vs 46.1% / 54.2% for xp 1 without fnb; xp 4 and hp 2 lose
    # (x waits fall, h waits rise more)
    xp: int = 2
    hp: int = 1
    # chains: the forwarders polled between every other queue of their RISC's loop (WL_FPRIO). Off: 35-38% at M 586
    # (h waits double)
    fprio: bool = False
    # chains: forwarders never block on a flush; what they walked is published once their RISC's writes have left (WL_FNB)
    fnb: bool = True
    # with ucast: an h tile's arrival at its row leader as a 4-byte write after it (a word a producer, same command
    # buffer) instead of a write barrier and a semaphore increment (WL_HARRW)
    harrw: bool = False  # (neutral on the tray)
    # chains: forwarders issue only on a free write command buffer, never stall in it (WL_FASYNC). Off: 40.3% / 43.0% at
    # M 586 / 2048 (a hop that yields waits a whole loop turn to go on)
    fasync: bool = False
    # h rows as a column all-to-all (WL_HA2A, kernels/wl_ha2a.hpp): each core's tile to its column and partner, the
    # partner's on to its column - at most 2 hops, no leaders / chains (replaces hchain; needs hcred)
    ha2a: bool = False
    hcred: bool = True  # h ring slot credits (WL_HCRED; without them the ring depth had to cover the cores' skew)
    # K-split gate/up (WL_KS; ws_ks.py's layout, single-chip prototype 75.5% at M 586 with the lane's L1): core (c, r)
    # holds W_gu[K tiles KSL r .., pairs 10 c ..] and computes partials of its column's 10 pairs; x rows go partner to
    # partner (whole rows alternately, one hop, no chain), partials are reduce-scattered down the column (bf16; rows below
    # on NOC0, above on NOC1) to the pair's owner, which folds them and produces h tile j as before (the h chain, down,
    # weights unchanged). Replaces the x chain (xchain does not apply).
    ks: bool = False
    ks_nxr: int = 8  # x ring rows
    ks_d: int = 3  # landing slots (rows of 10 contributions)
    ks_pd: int = 2  # partial ring rows
    ks_fl: int = 1  # fold lag (rows)
    ks_tl: int = 3  # tilize look-ahead (rows)
    ydep: int = 2  # down output staging (CB 17), rows
    # K split: an expert's last folds / downs overlap the next expert's first rows (drained only when the next one reuses
    # its weight slot) instead of a full drain at every expert's end (KS_OVL)
    ks_ovl: bool = False
    # K split: compute tilizes its x rows straight into its ring slot (no CB 8 staging, no local copy; CB 8 becomes a
    # 1-page-a-row signal), so the tilize look-ahead can reach ks_nxr - 1 rows (KS_XDIRECT / WL_KS_XDIRECT)
    ks_xdirect: bool = False
    idle: int = 0  # >0: the BRISC / NCRISC loops back off this many cycles after a turn that did nothing (WL_IDLE)
    ks_fz: bool = False  # K split: a row's fold after its down (KS_FOLD_LAST)
    ks_ff: bool = (
        False  # K split: a row's fold before its gate/up (the SwiGLU on PACK overlaps the gate/up) (KS_FOLD_FIRST)
    )
    sched_c: int = 9  # the schedule model's compute time per expert (packet times) the static order is planned with
    ks_bal: int = 0  # >0: partials by distance - the BRISC to itself + the next ks_bal - 1 rows down (wrapping), the NCRISC to the rest up (WL_KS_BAL)
    ks_pall: bool = False  # the BRISC sends every partial contribution (rows above wrap round on NOC0) (WL_KS_PALL)
    wtwice: bool = False  # the reader's weight queues twice a loop turn (WL_WTWICE)
    ybarrier: bool = False  # output rows: a write barrier each (old) instead of a flush (WL_YBARRIER)
    # weight packets' fused arrival increment without the router's write-ack wait (WL_ARR_NF; ordering rests on the
    # router's write and increment reaching the same core in order; mux bench: 43.7 vs 26.2 GB/s a link)
    arr_nf: bool = False
    # cycles are disjoint sub-cycles that together cover every chip once (EP-4: one a mesh column, of any length >= 2),
    # instead of Hamiltonian cycles: each lane is the set of them on its link, every chip in the one sub-cycle it is in
    subcycles: bool = False
    # one-way lanes: each (cycle, link) gives two lanes, one a direction (the second on the reversed cycle), every expert
    # all the way round its lane's direction; a lane's credits go back on the other direction's connection, which is its
    # partner lane's (g ^ 1): header-only channels on the partner's mux (ws_lane5)
    oneway: bool = False


# G <= 2 (the tray): columns (4, 5) and (9, 10), muxes in 3 and 8. Measured on the tray (K split, M 586): 62.5% against
# 59.5% in (1, 2) / (4, 5); (1, 2) / (7, 8) 58.7-59.1%, (4, 5) / (7, 8) 58.7%, (3, 4) or (5, 6) with (9, 10) 61.2%. The
# group cores' compute is position dependent (gate/up MATH time a row ~10% longer in columns 1-2 / 7-8 than in 4-5).
LAYOUT_G2 = [(4, 5, 3), (9, 10, 8)]


def group_layout(G):
    """(column a, column b, mux column) of each group; WS_GLAYOUT "a,b,mux;..." overrides (experiments)."""
    if os.environ.get("WS_GLAYOUT"):
        return [tuple(int(v) for v in it.split(",")) for it in os.environ["WS_GLAYOUT"].split(";")]
    if G <= 2:
        return LAYOUT_G2
    return [(a, b, m) for (a, b), m in zip(LANE_PAIRS, MUX_COL)]


def lane_cores(g, G=4):
    a, b, _ = group_layout(G)[g]
    return [(a, y) for y in range(10)] + [(b, y) for y in range(10)]  # core j = c * 10 + y


def mux_core(g, d, G=4):
    return (group_layout(G)[g][2], MUX_ROWS[d])


def lanes(shape, lc: LCfg):
    """Lane g: (cycle as a list of mesh coordinates, link index); direction 0 follows the list, 1 goes back. With
    subcycles: (the list of sub-cycles, link index) - lane_cycle() picks a chip's."""
    R, Cn = shape
    every = sorted((r, c) for r in range(R) for c in range(Cn))
    rev = lambda c: [c[0]] + c[1:][::-1]  # (the same cycle the other way round)
    links = list(range(lc.NL))
    if lc.subcycles:
        subs = [list(c) for c in lc.cycles]
        assert sorted(x for c in subs for x in c) == every, "the sub-cycles cover every chip once"
        assert len({len(c) for c in subs}) == 1, "sub-cycles of one length"
        if lc.oneway:
            return [(s, link) for link in links for s in (subs, [rev(c) for c in subs])]
        return [(subs, link) for link in links]
    cycles = [list(c) for c in (lc.cycles or [snake(shape)])]
    out = []
    for cyc in cycles:
        assert lc.nofab or sorted(cyc) == every, "a cycle visits every chip once"
        for link in links:
            out += [(cyc, link), (rev(cyc), link)] if lc.oneway else [(cyc, link)]
    return out


def lane_cycles(lane):
    """The cycles of a lane (a list: one, or its sub-cycles)."""
    cyc = lane[0]
    return cyc if cyc and isinstance(cyc[0], list) else [cyc]


def lane_cycle(lane, me):
    """The cycle of a lane that chip me (r, c) is on."""
    for cyc in lane_cycles(lane):
        if tuple(me) in [tuple(x) for x in cyc]:
            return cyc
    raise AssertionError(f"chip {me} on no cycle of the lane")


def slot_packets(cfg: GCfg, lc: LCfg):
    WGU, WDN = 2 * cfg.KT, cfg.PAIRS * cfg.DCOLS
    assert WGU % lc.CHT == 0 and WDN % lc.CHT == 0
    return WGU // lc.CHT, (WGU + WDN) // lc.CHT  # (gate / up packets, packets) a slice


def schedule(N, E, cfg: GCfg, lc: LCfg):
    GU, P = slot_packets(cfg, lc)
    return lane_sched(N, E, P=P, GU=GU, C=lc.sched_c, keep=lc.keep, types=slot_types(lc.slots), oneway=lc.oneway)


def make_tables(shape, cfg: GCfg, lc: LCfg):
    """Per chip and group the table of kernels/wl_reader.cpp, and per chip its own experts (global ids, in the order
    of their local index in the chip's slices tensor). Lane g holds the experts e = i G + g (lane-local i)."""
    R, Cn = shape
    G = cfg.G
    ln = lanes(shape, lc)
    assert len(ln) >= G, f"{len(ln)} lanes for {G} groups"
    assert cfg.nexp % G == 0
    E = cfg.nexp // G
    N = len(lane_cycles(ln[0])[0])  # (positions on a cycle; R Cn, or a sub-cycle's length)
    sch = schedule(N, E, cfg, lc)
    K = len(slot_types(lc.slots))
    assert K <= 7 and 1 + 2 * K <= 16, "3-bit slot fields, header below word 16"
    owners, kf = sch["owners"], sch["kf"]
    TLW = -(-(16 + 5 * E) // 16) * 16  # words a table (64 B multiple)
    NX = R * Cn
    tabs = torch.zeros(NX, G, TLW, dtype=torch.int64)
    own = [[] for _ in range(NX)]  # chip -> [(g, i)]
    meta = {}
    for g in range(G):
        for cyc in lane_cycles(ln[g]):
            chip_at = [r * Cn + c for r, c in cyc]
            for pos in range(N):
                for i in range(E):
                    if owners[i] == pos:
                        own[chip_at[pos]].append((g, i))
    own = [sorted(o) for o in own]
    own_idx = {(X, g, i): k for X in range(NX) for k, (g, i) in enumerate(own[X])}

    def arrives(i, pos):
        """(kind, directions it goes on from pos) of lane-local expert i at position pos."""
        o = owners[i]
        if o == pos:
            return KIND_OWN, [d for d, n in ((0, kf[i]), (1, N - 1 - kf[i])) if n > 0]
        h0 = (pos - o) % N  # hops on direction 0
        if 1 <= h0 <= kf[i]:
            return KIND_D0, [0] if h0 < kf[i] else []
        h1 = (o - pos) % N
        assert 1 <= h1 <= N - 1 - kf[i]
        return KIND_D1, [1] if h1 < N - 1 - kf[i] else []

    where = {}  # (pos, i) -> (slot, occupant index)
    for (pos, s), lst in sch["held"].items():
        for k, i in enumerate(lst):
            assert (pos, i) not in where, "an expert is held once a chip"
            where[(pos, i)] = (s, k)
    for g, cyc in [(g, cyc) for g in range(G) for cyc in lane_cycles(ln[g])]:
        chip_at = [r * Cn + c for r, c in cyc]
        for pos in range(N):
            X = chip_at[pos]
            t = tabs[X, g]
            t[0] = E
            off = 16 + E
            for s in range(K):
                lst = sch["held"][(pos, s)]
                t[1 + s] = len(lst)
                t[1 + K + s] = off
                for k, i in enumerate(lst):
                    kind, dirs = arrives(i, pos)
                    nxt = 0
                    if k + 1 < len(lst):
                        nkind = arrives(lst[k + 1], pos)[0]
                        nxt = 1 if nkind == KIND_OWN else (2 if nkind == KIND_D0 else 3)
                    w0 = (i * G + g) | kind << 16 | sum(1 << d for d in dirs) << 18 | nxt << 20
                    w = [w0, 0, 0, own_idx.get((X, g, i), 0) if kind == KIND_OWN else 0]
                    for d in dirs:
                        y = (pos + 1) % N if d == 0 else (pos - 1) % N
                        s2, k2 = where[(y, i)]
                        assert arrives(i, y)[0] == (KIND_D0 if d == 0 else KIND_D1)
                        w[1 + d] = s2 | k2 << 3
                    t[off + 4 * k : off + 4 * k + 4] = torch.tensor(w)
                off += 4 * len(lst)
            assert off <= TLW
            order = sch["order"][pos]
            assert sorted(order) == list(range(E)), "every expert computed once a chip"
            for n, i in enumerate(order):
                t[16 + n] = (i * G + g) | where[(pos, i)][0] << 16
            meta[(X, g)] = dict(pos=pos, order=[i * G + g for i in order])
    return dict(tabs=tabs, TLW=TLW, own=[[i * G + g for g, i in o] for o in own], meta=meta, sch=sch, E=E, K=K)


def tables_tensor(mesh, tabs):
    N, G, TLW = tabs.shape
    return ttnn.from_torch(
        tabs.reshape(N * G, TLW).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def build_lane(
    mesh, cfg: GCfg, lc: LCfg, x_t, idx_t, counts_t, ws_t, out_t, tab_t, T, out_rows, dest_t=None, fence_row=None
):
    """generic_op mesh program of the lane design; returns (mesh program, io tensors, stamps). T: make_tables();
    out_rows: tile rows of an output block (a group's last expert, or with out_all a group's experts); with y_rm, out_t
    is row-major [rows, H] and dest_t the destination lists [nexp, RMAX 32] uint32 (out_rows unused); fence_row: the
    first of out_t's spare rows (NUM_DRAM_BANKS of them) the posted row writes' fence rewrites."""
    R, Cn = tuple(mesh.shape)
    N, G, GC = R * Cn, cfg.G, cfg.GC
    assert GC == 20 and cfg.PAIRS == GC and cfg.DCOLS * GC == cfg.HT
    ln = lanes((R, Cn), lc)
    groups = [lane_cores(g, G) for g in range(G)]
    C = [c for gr in groups for c in gr]
    muxes = [] if lc.nofab else [mux_core(g, d, G) for g in range(G) for d in range(2)]
    assert not set(muxes) & set(C)
    gx, gy = mesh.compute_with_storage_grid_size().x, mesh.compute_with_storage_grid_size().y
    assert all(x < gx and y < gy for x, y in C + muxes), f"core layout outside the {gx} x {gy} grid"
    cset = crs(C)
    KT, KTD, DC, BK, NX, NS = cfg.KT, cfg.PAIRS, cfg.DCOLS, cfg.BK, cfg.NX, cfg.NS
    DB = 3
    WGU, WDN = KT * 2, KTD * DC
    wtile, htile, xtile = TILE_BYTES[cfg.w_dt], TILE_BYTES[cfg.h_dt], TILE_BYTES[cfg.x_dt]
    E, TLW, K = T["E"], T["TLW"], T["K"]
    # L1 tensors (same address on every core and chip); the weight slots first, at the top of L1: the BRISC reads
    # whole chunks from DRAM straight into them (64 B aligned)
    wgu_ring = sharded_l1(mesh, C, [K * WGU * 32, 32], cfg.w_dt, ttnn.TILE_LAYOUT)
    wd_ring = sharded_l1(mesh, C, [K * WDN * 32, 32], cfg.w_dt, ttnn.TILE_LAYOUT)
    assert wgu_ring.buffer_address() % 64 == 0 and wd_ring.buffer_address() % 64 == 0, "weight slots 64 B aligned"
    w16 = lambda n: sharded_l1(mesh, C, [n, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    warr, cred, freew, arr_stage = w16(K), w16(2 * K), w16(K), w16(2)
    hring = sharded_l1(mesh, C, [NS * KTD * 32, 32], cfg.h_dt, ttnn.TILE_LAYOUT)
    hflags = w16(NS * 2)
    # (each only with its feature: the NoC-split lane's L1 tensors sit close above its circular buffers)
    hchain_on = lc.ucast and lc.hchain and not lc.ha2a
    # h arrivals at a row leader, a word a producer row of its column (WL_HARRW)
    harrw = w16(GC // 2) if lc.ucast and lc.harrw else None
    # h ring credits: consumed h rows of the column's cores, then the partner (WL_HCRED)
    hcred = w16(GC // 2 + 1) if lc.hcred and (hchain_on or lc.ha2a) else None
    # (WL_HA2A: BRISC rows forwarded + flag scratch; a flag a tile of a slot)
    ha2a_w, ha2a_f = (w16(2), w16(NS * KTD)) if lc.ha2a else (None, None)
    if lc.ks:
        assert lc.ucast and lc.hchain and not lc.bcast, "the K split runs with the unicast h chain"
        KSL = KT // 10
        BK, NX = KSL, lc.ks_nxr  # (the x ring: NXR rows of KSL tiles; xflags 2 words a row: the two partners)
        xring = sharded_l1(mesh, C, [NX * BK * 32, 32], cfg.x_dt, ttnn.TILE_LAYOUT)
        xflags = w16(2 * NX)
        ptile = TILE_BYTES[ttnn.bfloat16]
        xcred, lflags, lcred, kscr = w16(2), w16(lc.ks_d * 10), w16(10), w16(8)
        land = sharded_l1(mesh, C, [lc.ks_d * 20 * 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT)
        pring = sharded_l1(mesh, C, [lc.ks_pd * 20 * 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    else:
        xring = sharded_l1(mesh, C, [NX * BK * 32, 32], cfg.x_dt, ttnn.TILE_LAYOUT)
        xflags = w16(NX)
    xprog = w16(GC)
    assert not (lc.bcast and lc.mc_noc1), "bcast and mc_noc1 place the multicasts differently"
    assert not (lc.noc_split and lc.mc_noc1), "noc_split places every multicast on NOC0"
    hcnt = w16(NS) if lc.bcast else None  # core 19's h tiles arrived a ring slot (WL_BCAST)
    # core 0's x funnel slots and their filled flags (WL_XIN, WL_XINF)
    xin = sharded_l1(mesh, C, [lc.xf * BK * 32, 32], cfg.x_dt, ttnn.TILE_LAYOUT) if lc.bcast else None
    xinf = w16(lc.xf) if lc.bcast else None
    # DRAM -> L1 reads need the L1 address congruent to the DRAM one mod 64: 64-aligned landing spots inside these
    counts_l1 = sharded_l1(mesh, C, [1, 32 + max(4, cfg.nexp)], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    idx_l1 = sharded_l1(mesh, C, [12 if lc.ks else 4, 32], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    table_l1 = sharded_l1(mesh, C, [1, TLW + 16], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    a64 = lambda a: (a + 63) // 64 * 64
    cnt_base, idx_addr, tab_base = (
        a64(counts_l1.buffer_address()),
        a64(idx_l1.buffer_address()),
        a64(table_l1.buffer_address()),
    )
    st = stamps_tensor(mesh, C)
    # (y_rm: QY on the NCRISC's NOC1 reads the destination lists and posts the rows; no group multicast may use NOC1
    # meanwhile: the NoC split, or the unicast chains)
    assert not lc.y_rm or ((lc.noc_split or lc.ucast) and dest_t is not None), "y_rm: noc_split or ucast, and dest_t"
    # (y_rm: the row writes need CB 17 64 B aligned)
    ysh = sharded_l1(mesh, C, [lc.ydep * cfg.DCOLS * 32, 32], cfg.out_dt, ttnn.TILE_LAYOUT) if lc.y_rm else None
    dest_t = dest_t if lc.y_rm else idx_t
    if lc.nofab:
        mcfg, ctype, client_ct, mesh_id = None, None, [0] * 5, 0  # (the reader's mux CT args are placeholders)
    else:
        mcfg = fabric_mux.Config(
            num_full_size_channels=GC,
            num_header_only_channels=0,
            num_buffers_per_full_size_channel=lc.nbuf,
            num_buffers_per_header_only_channel=0,
            full_size_channel_buffer_size_bytes=fabric_mux.channel_buffer_size_bytes(mesh),
            base_l1_address=ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1),
        )
        ctype = fabric_mux.ChannelType.FULL_SIZE
        client_ct = list(fabric_mux.client_compile_time_args(num_clients=GC, channel_type=ctype, config=mcfg))
        mesh_id = int(node(mesh, (0, 0)).mesh_id)
    dm = lambda risc: ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_0_default if risc == 0 else ttnn.NOC.RISCV_1_default,
    )
    utd = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode([ttnn.UnpackToDestMode.Default] * 64)
    ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True)
    ccfg.unpack_to_dest_mode = utd
    act_defs = [
        ("EXPERT_ACTIVATION_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LINEAR_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LIMIT", "7.0f"),
        ("EXPERT_ACTIVATION_ALPHA", "1.702f"),
    ]
    # (the x stream-order gate fixed starving concurrent multicasts; with the x chain it only costs: off)
    xorder = 1 if lc.mc_noc1 or lc.bcast or lc.noc_split else (0 if lc.ucast and lc.xchain else cfg.xorder)
    ydefs = [("WG_XORDER", str(xorder)), ("WG_Y_NCRISC", "1"), ("WL_K", str(K))]
    ydefs += [("WL_MC_NOC1", "1")] if lc.mc_noc1 else []
    ydefs += [("WL_NOC_SPLIT", "1")] if lc.noc_split else []
    if lc.bcast:
        ydefs += [("WL_BCAST", str(hcnt.buffer_address())), ("WL_XIN", str(xin.buffer_address()))]
        ydefs += [("WL_XINF", str(xinf.buffer_address())), ("WL_XF", str(lc.xf))]
    ydefs += [("WL_OUT_ALL", "1")] if lc.out_all else []
    ydefs += [("WL_Y_RM", "1")] if lc.y_rm else []
    ydefs += [("WL_Y_NOWRITE", "1")] if lc.y_nowrite else []
    ydefs += [("WL_Y_POSTED", "1")] if lc.y_rm and lc.y_posted else []
    ydefs += [("WL_Y_FENCE", str(fence_row))] if lc.y_rm and lc.y_posted and fence_row is not None else []
    ydefs += [("WL_MCAST_LINKED", "1")] if lc.mcast_linked else []
    ydefs += [] if lc.own_read else [("WL_NO_OWN_READ", "1")]
    ydefs += [] if lc.gather else [("WG_NO_GATHER", "1")]
    assert not (lc.ucast and lc.bcast), "WL_BCAST still multicasts its h rows: run it with ucast=0"
    ydefs += [("WL_UCAST", "1")] if lc.ucast else []
    # (the chains replace the unicast fan-out: without ucast, i.e. with multicasts, they do not apply)
    assert not lc.ha2a or (lc.ucast and lc.hcred and not lc.bcast), "ha2a: unicast, with h credits"
    xchain, hchain = lc.ucast and lc.xchain and not lc.ks, lc.ucast and lc.hchain and not lc.ha2a
    ydefs += [("WL_XCHAIN", "1")] if xchain else []
    ydefs += [("WL_XHOP", "1")] if xchain and lc.xhop else []
    ydefs += [("WL_JUMP", "1")] if lc.jump and (xchain or hchain) else []
    ydefs += [("WL_HCHAIN", "1")] if hchain else []
    assert 1 <= lc.xp <= 4 and 1 <= lc.hp <= 4, "a piece's flag is a word of its slot's 16 B"
    ydefs += [("WL_XP", str(lc.xp if xchain else 1)), ("WL_HP", str(lc.hp if hchain else 1))]
    ydefs += [("WL_FPRIO", "1")] if lc.fprio and (xchain or hchain) else []
    ydefs += [("WL_FNB", "1")] if lc.fnb and (xchain or hchain) else []
    ydefs += [("WL_FASYNC", "1")] if lc.fasync and (xchain or hchain) else []
    ydefs += [("WL_HARRW", str(harrw.buffer_address()))] if lc.ucast and lc.harrw else []
    ydefs += [("WL_HCRED", str(hcred.buffer_address()))] if lc.hcred and (hchain or lc.ha2a) else []
    if lc.ha2a:
        ydefs += [("WL_HA2A", str(ha2a_w.buffer_address())), ("WL_HA2A_F", str(ha2a_f.buffer_address()))]
    if lc.ks:
        assert not lc.ks_xdirect or lc.ks_tl < lc.ks_nxr, "direct tilize: the look-ahead must stay inside the x ring"
        # (the down of row q runs once row q + DN_LAG is folded; the h ring must hold the rows in between - NS 2 with
        # DN_LAG 3 hung on the tray, NS 3 with DN_LAG 3 runs: folded h tiles can wait in CB 16 meanwhile)
        assert NS >= cfg.dn_lag, f"h ring NS {NS} below DN_LAG {cfg.dn_lag} (deadlock)"
        ydefs += [("WL_KS_XDIRECT", "1")] if lc.ks_xdirect else []
        ydefs += [
            ("WL_KS_XRM", str(cfg.XRM)),
            ("WL_KS", str(KSL)),
            ("WL_KS_XCRED", str(xcred.buffer_address())),
            ("WL_KS_D", str(lc.ks_d)),
        ]
        ydefs += [("WL_KS_LAND", str(land.buffer_address())), ("WL_KS_LFLAGS", str(lflags.buffer_address()))]
        ydefs += [("WL_KS_LCRED", str(lcred.buffer_address())), ("WL_KS_PTILE", str(ptile))]
        ydefs += [("WL_KS_SCR", str(kscr.buffer_address())), ("WL_KS_PD", str(lc.ks_pd))]
        ydefs += [("WL_KS_PRING", str(pring.buffer_address()))] + ([("WL_KS_PALL", "1")] if lc.ks_pall else [])
        ydefs += [("WL_KS_BAL", str(lc.ks_bal))] if lc.ks_bal else []
    ydefs += [("WL_WTWICE", "1")] if lc.wtwice else []
    ydefs += [("WL_YBARRIER", "1")] if lc.ybarrier else []
    ydefs += [("WL_IDLE", str(lc.idle))] if lc.idle else []
    # extra data-movement defines for experiments, e.g. WL_DEFS="WG_NO_GATHER,WL_READ_NOC=1"
    for d in filter(None, os.environ.get("WL_DEFS", "").split(",")):
        k, _, v = d.partition("=")
        ydefs += [(k, v or "1")]
    # x staging blocks a core; with the funnel (bcast), core 0's lc.xf funnel slots take two of them on every core
    XST = -(-cfg.LOOK // GC) + (2 if lc.bcast else lc.xst if lc.xst >= 0 else 2 if lc.ucast else 4)
    if lc.ks:
        XST = lc.ks_tl + 2  # (rows of KSL tiles)
    mpd = ttnn.MeshProgramDescriptor()
    for r in range(R):
        for c in range(Cn):
            me_rc, me = (r, c), r * Cn + c
            kernels, rd_idx = [], []
            nxt = {}
            for g, gr in enumerate(groups):
                cyc, link = ln[g]
                if lc.nofab:
                    nxt[g] = (me_rc, me_rc)
                else:
                    i = cyc.index(me_rc)
                    nxt[g] = (cyc[(i + 1) % N], cyc[(i - 1) % N])
                nid = [0, 0] if lc.nofab else [int(node(mesh, nxt[g][d]).chip_id) for d in range(2)]
                rs = noc_rects(mesh, gr)
                assert len(rs) == 1, rs
                r0 = list(rs[0][:4])
                r_noc1 = [r0[2], r0[3], r0[0], r0[1]]
                cx0, ny0 = noc(mesh, *gr[0])
                cx1, _ = noc(mesh, *gr[10])
                cp_rt, rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for j, (x, y) in enumerate(gr):
                    assert noc(mesh, x, y) == ((cx0, cx1)[j // 10], ny0 + j % 10)
                    xy = noc(mesh, x, y)
                    cp_rt[x][y] = [st.buffer_address(), j, g, E]
                    wr_rt[x][y] = [j, g, E, cx0, cx1, ny0, out_t.buffer_address(), x_t.buffer_address()] + [
                        idx_t.buffer_address(),
                        counts_t.buffer_address(),
                        st.buffer_address(),
                        tab_t.buffer_address(),
                        ws_t.buffer_address(),
                    ]
                    rd_rt[x][y] = [j, g, E, cx0, cx1, ny0, st.buffer_address(), out_t.buffer_address(), mesh_id] + [
                        nid[0],
                        nid[1],
                        xy[0] | xy[1] << 8,
                        x_t.buffer_address(),
                        idx_t.buffer_address(),
                        counts_t.buffer_address(),
                        tab_t.buffer_address(),
                        ws_t.buffer_address(),
                        dest_t.buffer_address(),
                    ]
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=kpath("wl_writer.cpp"),
                        core_ranges=crs(gr),
                        compile_time_args=[KT, BK, KTD, DC, GC, G, NS, hring.buffer_address(), hflags.buffer_address()]
                        + r0
                        + [htile, cfg.HT, cfg.RMAX, cfg.H * 2, cnt_base, idx_addr, cfg.nexp]
                        + [NX, xring.buffer_address(), xflags.buffer_address(), xprog.buffer_address(), xtile]
                        + [tab_base, TLW * 4, wgu_ring.buffer_address(), wd_ring.buffer_address()]
                        + [warr.buffer_address(), freew.buffer_address(), lc.CHT, wtile, lc.D]
                        + list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(x_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(idx_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(counts_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(tab_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(ws_t).get_compile_time_args()),
                        runtime_args=wr_rt,
                        defines=ydefs,
                        config=dm(0),
                    )
                )
                rd_idx.append(len(kernels))
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=kpath("wl_reader.cpp"),
                        core_ranges=crs(gr),
                        compile_time_args=[KT, BK, KTD, DC, GC, G, NS, NX, hflags.buffer_address()]
                        + [xring.buffer_address(), xflags.buffer_address(), xprog.buffer_address()]
                        + r_noc1
                        + [xtile, cnt_base, tab_base, hring.buffer_address(), htile, warr.buffer_address()]
                        + [out_rows, cfg.HT, wgu_ring.buffer_address(), wd_ring.buffer_address()]
                        + [cred.buffer_address(), freew.buffer_address(), lc.CHT, wtile]
                        + client_ct
                        + list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(x_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(idx_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(counts_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(tab_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(ws_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(dest_t).get_compile_time_args())
                        + [cfg.RMAX, cfg.H * 2, cfg.nexp, TLW * 4, idx_addr, lc.D],
                        runtime_args=rd_rt,
                        defines=ydefs
                        + ([("WL_NO_FABRIC", "1")] if lc.nofab else FAB_DEFS)
                        + [("WL_WPOLL", str(lc.wpoll))]
                        + ([("WL_ARR_WRITE", str(arr_stage.buffer_address()))] if lc.arr_write else [])
                        + ([("WL_ARR_NF", "1")] if lc.arr_nf else []),
                        config=dm(1),
                    )
                )
                if lc.ks:
                    kernels.append(
                        ttnn.KernelDescriptor(
                            kernel_source=kpath("kl_compute.cpp"),
                            core_ranges=crs(gr),
                            compile_time_args=[KSL, KTD, DC, DB, cnt_base, G, GC, lc.ks_fl, lc.ks_tl, tab_base + 64],
                            named_compile_time_args=[("expert_activation", 0)],
                            defines=act_defs
                            + [("DN_LAG", str(cfg.dn_lag)), ("KS_XALT", "1")]
                            + ([("KS_OVL", "1")] if lc.ks_ovl else [])
                            + ([("KS_XDIRECT", str(lc.ks_nxr))] if lc.ks_xdirect else [])
                            + ([("KS_FOLD_FIRST", "1")] if lc.ks_ff else [])
                            + ([("KS_FOLD_LAST", "1")] if lc.ks_fz else [])
                            + ([("WL_Y_RM", "1")] if lc.y_rm else []),
                            runtime_args=cp_rt,
                            config=ccfg,
                        )
                    )
                else:
                    kernels.append(
                        ttnn.KernelDescriptor(
                            kernel_source=kpath("wl_compute.cpp"),
                            core_ranges=crs(gr),
                            compile_time_args=[KT, BK, KTD, DC, DB, NX, cfg.LOOK, cnt_base, G, GC, tab_base + 64],
                            named_compile_time_args=[("expert_activation", 0)],
                            defines=act_defs
                            + [("DN_LAG", str(cfg.dn_lag)), ("WC_GLU", str(cfg.glu))]
                            + ([("WL_Y_RM", "1")] if lc.y_rm else [])
                            + ([("WX_X_FMT_DIFFERS", "1")] if cfg.x_dt != cfg.h_dt else [])
                            + ([("WX_H_PACK_DIFFERS", "1")] if cfg.h_dt != cfg.out_dt else []),
                            runtime_args=cp_rt,
                            config=ccfg,
                        )
                    )
            mux_idx = len(kernels)
            mrt = ttnn.RuntimeArgs()
            for m in muxes:
                mrt[m[0]][m[1]] = []
            if not lc.nofab:
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source="tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                        core_ranges=crs(muxes),
                        compile_time_args=list(mcfg.kernel_compile_time_args()),
                        runtime_args=mrt,
                        defines=FAB_DEFS,
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_0,
                            noc=(ttnn.NOC.RISCV_1_default if lc.mux_noc1 or lc.noc_split else ttnn.NOC.RISCV_0_default),
                        ),
                        opt_level=fabric_mux.KernelBuildOptLevel.O3,
                    )
                )
            cbs = [
                ttnn.cb_descriptor_from_sharded_tensor(0, xring),
                ttnn.cb_descriptor_from_sharded_tensor(1, wgu_ring),
                ttnn.cb_descriptor_from_sharded_tensor(2, hring),
                ttnn.cb_descriptor_from_sharded_tensor(3, wd_ring),
                cb(7, ttnn.bfloat16, cfg.XRM * BK, cset),
                cb(8, ttnn.uint32, lc.ks_nxr, cset, page=32)
                if lc.ks and lc.ks_xdirect
                else cb(8, cfg.x_dt, XST * BK, cset),
                cb(16, cfg.h_dt, 2, cset),
                ttnn.cb_descriptor_from_sharded_tensor(17, ysh) if lc.y_rm else cb(17, cfg.out_dt, lc.ydep * DC, cset),
            ]
            if lc.ks:
                cbs += [
                    ttnn.cb_descriptor_from_sharded_tensor(4, land),
                    ttnn.cb_descriptor_from_sharded_tensor(18, pring),
                ]
            # (0: the chains' down-forwarder progress, BRISC -> NCRISC)
            sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=cset, initial_value=0) for s in range(6)]
            p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
            if not lc.nofab:
                for g, gr in enumerate(groups):
                    cyc, link = ln[g]
                    for d in range(2):
                        m = mux_core(g, d, G)
                        p.kernels[mux_idx].runtime_args[m[0]][m[1]].extend(
                            mcfg.kernel_runtime_args(
                                source_node_id=node(mesh, me_rc),
                                destination_node_id=node(mesh, nxt[g][d]),
                                link_index=link,
                                program_descriptor=p,
                                mux_logical_core=ttnn.CoreCoord(*m),
                            )
                        )
                    for d in range(2):
                        m = mux_core(g, d, G)
                        mux_vc = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*m))
                        master_vc = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*gr[0]))
                        term_sem = None
                        for j, (x, y) in enumerate(gr):
                            args = fabric_mux.client_runtime_args(
                                connection_valid=True,
                                is_termination_master=j == 0,
                                channel_type=ctype,
                                mux_virtual_core=mux_vc,
                                client_index=j,
                                client_logical_core=ttnn.CoreCoord(x, y),
                                config=mcfg,
                                program_descriptor=p,
                                termination_master_virtual_core=master_vc,
                                termination_master_semaphore_id=term_sem,
                            )
                            if j == 0:
                                term_sem = args[10]
                            p.kernels[rd_idx[g]].runtime_args[x][y].extend(args)
            mc = ttnn.MeshCoordinate(r, c)
            mpd[ttnn.MeshCoordinateRange(mc, mc)] = p
    io = [x_t, idx_t, counts_t, ws_t, tab_t, wgu_ring, wd_ring, hring, hflags, xring, xflags, xprog, counts_l1, harrw]
    io += [hcred, ha2a_w, ha2a_f]
    io += [idx_l1, table_l1, warr, cred, freew, arr_stage] + ([hcnt, xin, xinf] if lc.bcast else [])
    io += [xcred, lflags, lcred, kscr, land, pring] if lc.ks else []
    io += ([ysh, dest_t] if lc.y_rm else []) + [st, out_t]
    io = [t for t in io if t is not None]
    return mpd, io, st

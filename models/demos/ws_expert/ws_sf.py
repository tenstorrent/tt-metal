# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert groups with the weights forwarded store-and-forward around Hamiltonian cycles (every chip runs experts;
expert e is owned by chip e % NCHIP, as in ws_fab.build_all).

The fabric's own multicast trees send everything over an N/S trunk with E/W teeth and the shortest way round, which
loads some links about twice their share. Here every packet goes one hop only: a cycle through all chips, in each
direction, on each of its links, is a directed ring with one forwarder core per chip. A lane is a (cycle, link) pair;
the group cores of every expert are dealt to the lanes (j % NL), and each slice goes both ways round its lane from the
owner and stops half way (reach: the two directions add up to N - 1 hops; on an even ring the half-way chip alternates
slice by slice).

Order on the rings. A forwarder handles its packets in a fixed order known to both ends, so packets carry no
metadata. The order comes from a synchronous model of each ring as a conveyor (conveyor()): every hop takes D packet
times, a chip forwards what arrives unless it ends there, and puts its own next packet into every free slot. Expert-
major order (all of an expert's packets before the next expert's) serializes the owners around the ring (about 55% of
the link in a model run); the conveyor keeps every link busy (99% in the same model). Owners inject their experts in
order, and expert e only once every earlier expert of its group up to e - 2G has reached all chips: the group cores'
2-slot rings then never hold back a packet that an older expert is waiting behind (see wr_fwd.cpp for why this, with
all forwarders following the model's order, cannot deadlock).

Per directed ring and chip one forwarder core:
- BRISC (wr_land.cpp): loads this ring's order (a DRAM table), reads the chip's own slices from DRAM into CB 0 ahead of
  the sender, and lands passing packets from its staging slots (CB 1) into the group cores' 2-slot rings, fusing the
  arrival increment after a slice's last packet;
- NCRISC (wr_fwd.cpp): sends injected and passing packets one hop into the next chip's forwarder (fused write +
  arrival increment), and relays the opposite ring's per-hop staging credits to the next chip.
A local lander (wf_sreader + wf_bsend WF_LOCAL) writes the chip's own slices into its own group cores, and a relay core
(wr_relay.cpp) fans the group cores' freed-slot counts out to the chip's landers: no credit travels more than one hop,
and none goes back to the owners.
"""

from collections import deque
from dataclasses import dataclass

import torch

import ttnn
from models.demos.ws_expert.bench_util import cb, crs, kpath
from models.demos.ws_expert.ws_expert import noc, sharded_l1
from models.demos.ws_expert.ws_fab import FAB_DEFS, node
from models.demos.ws_expert.ws_group import GCfg, build, group_cores

INJ, FWD, TERM = 0, 1, 2


@dataclass
class RCfg:
    CHT: int = 24  # tiles a packet (24 x 576 B bfp4 = 13.8 KB)
    S: int = 12  # staging slots a forwarder (packets)
    SCB: int = 6  # owner read-ahead a forwarder / local lander staging (chunks)
    RD: int = 4  # DRAM reads in flight (chunks)
    CRB: int = 2  # while packets flow, relay staging credits CRB at a time (at once when the forwarder waits)
    NL: int = 2  # links a cycle edge
    D: int = 3  # hop time of the conveyor model (packet times)
    relay: tuple = (10, 9)
    local: tuple = (9, 9)
    # forwarder cores, index 2 lane + dir (outside the groups; lanes = cycles x links)
    fwd: tuple = ((9, 0), (10, 0), (9, 1), (10, 1), (9, 2), (10, 2), (9, 3), (10, 3))
    cycles: tuple = None  # Hamiltonian cycles (mesh coordinates in ring order); default: the row snake
    own_slices: bool = False  # each chip's slices tensor holds only its own experts (WS_OWN_SLICES)
    owners: tuple = None  # chip of expert class k = e % NCHIP (default: chip k)


def snake(shape):
    """Row snake (0, 0)..(0, C-1), (1, C-1)..(1, 0), ...: a cycle over direct links for 2 rows (more rows need the Y
    wrap)."""
    R, C = shape
    return [(r, c) for r in range(R) for c in (range(C) if r % 2 == 0 else range(C - 1, -1, -1))]


def torus_cycles(rows, cols, seed=1):
    """Two edge-disjoint Hamiltonian cycles covering every edge of a rows x cols torus (both >= 3): randomized
    Warnsdorff search for cycle A, accepted when the leftover edges form one cycle B (the host construction of
    high_bw_all_gather's fabric factory)."""
    import random
    import sys

    n = rows * cols
    nbrs = lambda v: [
        ((v // cols + 1) % rows) * cols + v % cols,
        ((v // cols - 1) % rows) * cols + v % cols,
        (v // cols) * cols + (v % cols + 1) % cols,
        (v // cols) * cols + (v % cols - 1) % cols,
    ]
    key = lambda a, b: (min(a, b), max(a, b))
    rng = random.Random(seed)
    sys.setrecursionlimit(max(10000, 4 * n))
    for _ in range(20000):
        cyc, seen, budget = [0], {0}, [200000]

        def extend():
            if budget[0] == 0:
                return False
            budget[0] -= 1
            if len(cyc) == n:
                return cyc[0] in nbrs(cyc[-1])
            cand = [v for v in nbrs(cyc[-1]) if v not in seen]
            rng.shuffle(cand)
            cand.sort(key=lambda v: sum(w not in seen for w in nbrs(v)))
            for v in cand:
                cyc.append(v)
                seen.add(v)
                if extend():
                    return True
                cyc.pop()
                seen.discard(v)
            return False

        if not extend():
            continue
        ea = {key(cyc[i], cyc[(i + 1) % n]) for i in range(n)}
        left = {v: [w for w in nbrs(v) if key(v, w) not in ea] for v in range(n)}
        if any(len(x) != 2 for x in left.values()):
            continue
        cb_, prev, cur = [0], None, 0
        while True:
            nxt = left[cur][0] if left[cur][0] != prev else left[cur][1]
            if nxt == 0:
                break
            cb_.append(nxt)
            prev, cur = cur, nxt
            if len(cb_) > n:
                break
        if len(cb_) != n:
            continue
        coords = lambda vs: [(v // cols, v % cols) for v in vs]
        return coords(cyc), coords(cb_)
    raise RuntimeError(f"no Hamiltonian decomposition of a {rows} x {cols} torus")


def reach(e, s, d, N):
    """Hops slice s (lane-local) of expert e travels from its owner in direction d."""
    kf = N // 2 if (e + s) & 1 else (N - 1) // 2
    return kf if d == 0 else N - 1 - kf


def owner_of(rc: RCfg, NCHIP, e):
    return rc.owners[e % NCHIP] if rc.owners else e % NCHIP


def rings(mesh_shape, rc: RCfg):
    """Directed rings [(lane, dir, cycle, link)], index 2 lane + dir."""
    R, Cn = mesh_shape
    cycles = [list(c) for c in (rc.cycles or [snake(mesh_shape)])]
    out = []
    for k, cyc in enumerate(cycles):
        assert sorted(cyc) == sorted((r, c) for r in range(R) for c in range(Cn)), "a cycle visits every chip once"
        for l in range(rc.NL):
            for d in range(2):
                out.append((k * rc.NL + l, d, cyc, l))
    return out


def flow(cyc, d, Cn):
    """Chip indices of a cycle in flow order (direction d: 0 list order, 1 reversed), starting at cyc[0]."""
    chips = [r * Cn + c for r, c in cyc]
    return chips if d == 0 else [chips[0]] + chips[1:][::-1]


def conveyor(shape, rc: RCfg, cfg: GCfg):
    """The rings' packet order from a synchronous conveyor model (module doc). Every hop takes D steps; a packet never
    waits once injected, so its arrival times are known when it is injected. An owner may inject a packet of expert e
    (the n-th of group g) once all packets of the group's experts n' <= n - 2 are injected and the packet reaches
    every chip on its path after their last packet there. Returns {(chip, ring): [(kind, e, c, u)]} (each chip's in-
    and out-stream merged in time order, TERM packets included) and the model's steps."""
    R, Cn = shape
    NCHIP = R * Cn
    G, GC, D = cfg.G, cfg.GC, rc.D
    NCH = (2 * cfg.KT + cfg.PAIRS * cfg.DCOLS) // rc.CHT
    rl = rings(shape, rc)
    NL = len(rl) // 2
    order = [flow(cyc, d, Cn) for _, d, cyc, _ in rl]
    N = [len(o) for o in order]
    queue, uninj = [], [0] * cfg.nexp
    for f, (lane, d, cyc, _) in enumerate(rl):
        pos = {chip: i for i, chip in enumerate(order[f])}
        qf = [deque() for _ in range(N[f])]
        for e in range(cfg.nexp):
            for s in range(GC // NL):
                r = reach(e, s, d, N[f])
                if r == 0:
                    continue
                c = (e % G) * GC + NL * s + lane
                for u in range(NCH):
                    qf[pos[owner_of(rc, NCHIP, e)]].append((e, c, u, r))
                uninj[e] += NCH
        queue.append(qf)
    last = [dict() for _ in range(cfg.nexp)]  # chip -> last arrival time of the expert's packets
    ready = [0] * G  # leading experts n of each group with every packet injected
    cum = [[dict()] for _ in range(G)]  # cum[g][m]: chip -> last arrival over the group's experts n < m

    def advance(g):
        while ready[g] * G + g < cfg.nexp and uninj[ready[g] * G + g] == 0:
            m = dict(cum[g][-1])
            for q, v in last[ready[g] * G + g].items():
                m[q] = max(m.get(q, -1), v)
            cum[g].append(m)
            ready[g] += 1

    for g in range(G):
        advance(g)
    seqs = {(order[f][p], f): [] for f in range(len(rl)) for p in range(N[f])}
    pipe = [[None] * (N[f] * D) for f in range(len(rl))]
    left, t = sum(uninj), 0
    while left:
        for f in range(len(rl)):
            old, new, Nf, of, qf = pipe[f], [None] * (N[f] * D), N[f], order[f], queue[f]
            for p in range(Nf):
                base = p * D
                new[base + 1 : base + D] = old[base : base + D - 1]
                inc = old[((p - 1) % Nf) * D + D - 1]
                out = None
                sq = seqs[(of[p], f)]
                if inc is not None:
                    e, c, u, r, h = inc
                    if h == r:
                        sq.append((TERM, e, c, u))
                        left -= 1
                    else:
                        sq.append((FWD, e, c, u))
                        out = (e, c, u, r, h + 1)
                if out is None and qf[p]:
                    e, c, u, r = qf[p][0]
                    g, n = e % G, e // G
                    ok = n < 2 or (
                        ready[g] >= n - 1
                        and all(t + D * h > cum[g][n - 1].get(of[(p + h) % Nf], -1) for h in range(1, r + 1))
                    )
                    if ok:
                        qf[p].popleft()
                        sq.append((INJ, e, c, u))
                        out = (e, c, u, r, 1)
                        for h in range(1, r + 1):
                            q = of[(p + h) % Nf]
                            last[e][q] = max(last[e].get(q, -1), t + D * h)
                        uninj[e] -= 1
                        if uninj[e] == 0:
                            advance(g)
                new[base] = out
            pipe[f] = new
        t += 1
        assert t < 100 * sum(len(s) for s in seqs.values()) + 1000, "conveyor model stuck"
    return seqs, t


def ring_tables(shape, rc: RCfg, cfg: GCfg):
    """Per chip: [nring, TL] uint32 order tables (entry kind | u << 2 | c << 6 | e << 14) and per ring (length,
    packets in, packets out); TL rounded up to 16 entries (64 B)."""
    R, Cn = shape
    NCHIP = R * Cn
    seqs, _ = conveyor(shape, rc, cfg)
    nr = len(rings(shape, rc))
    TL = max(len(s) for s in seqs.values())
    TL = -(-TL // 16) * 16
    tab = torch.zeros(NCHIP, nr, TL, dtype=torch.int64)
    meta = {}
    for (chip, f), sq in seqs.items():
        assert all(e < (1 << 18) and c < 256 and u < 16 for _, e, c, u in sq)
        tab[chip, f, : len(sq)] = torch.tensor([k | u << 2 | c << 6 | e << 14 for k, e, c, u in sq], dtype=torch.int64)
        meta[(chip, f)] = (len(sq), sum(k != INJ for k, *_ in sq), sum(k != TERM for k, *_ in sq))
    return tab.to(torch.int32), meta, seqs


def make_tables(mesh, cfg: GCfg, rc: RCfg):
    """Device copy of the order tables (each chip its own [nring, TL], DRAM, one page a ring) and their metadata."""
    tab, meta, seqs = ring_tables(tuple(mesh.shape), rc, cfg)
    NCHIP, nr, TL = tab.shape
    tab_t = ttnn.from_torch(
        tab.reshape(NCHIP * nr, TL),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    return dict(tab_t=tab_t, TL=TL, meta=meta, seqs=seqs)


def build_sf(mesh, cfg: GCfg, rc: RCfg, x_t, idx_t, counts_t, ws_t, out_t, tabs):
    R, Cn = tuple(mesh.shape)
    NCHIP = R * Cn
    NC = cfg.G * cfg.GC
    KT, KTD, DC = cfg.KT, cfg.PAIRS, cfg.DCOLS
    rl = rings((R, Cn), rc)
    NLANE = len(rl) // 2
    assert cfg.GC % NLANE == 0, "lanes deal every expert's group cores evenly"
    fcores = list(rc.fwd[: 2 * NLANE])
    fcore = lambda lane, d: fcores[2 * lane + d]
    land = [rc.local] + fcores
    gcores = [c for g in range(cfg.G) for c in group_cores(cfg, g)]
    assert not set([rc.relay] + land) & set(gcores) and len(set([rc.relay] + land)) == 2 + len(fcores)
    # group slot credits: word c on the relay (written by the group cores) and on every lander (by the relay)
    icred = sharded_l1(mesh, [rc.relay] + land, [NC, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    words = sharded_l1(mesh, fcores, [16, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    sst = sharded_l1(mesh, land, [2, 16], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    gprog, io, st, ex = build(mesh, cfg, x_t, idx_t, counts_t, ws_t, ws_t, out_t, fab=dict(relay=rc.relay, icred=icred))
    tword = lambda c: c | noc(mesh, *gcores[c])[0] << 8 | noc(mesh, *gcores[c])[1] << 16
    tl_all = [tword(c) for c in range(NC)]
    xy = lambda core: noc(mesh, *core)[0] | noc(mesh, *core)[1] << 8
    mesh_id = int(node(mesh, (0, 0)).mesh_id)
    dm = lambda risc: ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_0_default if risc == 0 else ttnn.NOC.RISCV_1_default,
    )
    tab_t, TL, meta = tabs["tab_t"], tabs["TL"], tabs["meta"]
    sdefs = [("SST_ADDR", str(sst.buffer_address()))]
    odefs = [("WS_OWN_SLICES", "1")] if rc.own_slices else []
    ct_ring = [cfg.nexp, KT, KTD, DC, rc.CHT, cfg.G, cfg.GC, NCHIP, rc.S, words.buffer_address()]
    ct_send = [cfg.nexp, KT, KTD, DC, rc.CHT, cfg.G, cfg.GC]
    ct_send += [ex["wgu_ring"].buffer_address(), ex["wd_ring"].buffer_address(), ex["warr"].buffer_address()]
    # (wf_bsend reads its credit word at icred + (me NC + c) 16: the base is shifted so that this is word c)
    ct_send_icred = lambda me: [icred.buffer_address() - me * NC * 16, NC, NCHIP, me]
    mpd = ttnn.MeshProgramDescriptor()
    for r in range(R):
        for c in range(Cn):
            me_rc, me = (r, c), r * Cn + c
            rrt = ttnn.RuntimeArgs()
            rrt[rc.relay[0]][rc.relay[1]] = [xy(x) for x in land]
            relay = ttnn.KernelDescriptor(
                kernel_source=kpath("wr_relay.cpp"),
                core_ranges=crs([rc.relay]),
                compile_time_args=[NC, cfg.GC, cfg.G, cfg.nexp, icred.buffer_address(), len(land)],
                runtime_args=rrt,
                config=dm(0),
            )
            lx, ly = rc.local
            lrd, lsd, frd, fsd = (ttnn.RuntimeArgs() for _ in range(4))
            lrd[lx][ly] = [ws_t.buffer_address(), me, NC] + tl_all
            lsd[lx][ly] = [mesh_id, me, 0, 0, 0, 0, 1 << me, NC] + tl_all
            nxt = {}
            for lane, d, cyc, l in rl:
                f = 2 * lane + d
                x, y = fcore(lane, d)
                N = len(cyc)
                i = cyc.index(me_rc)
                nxt[f] = cyc[(i + (1 if d == 0 else -1)) % N]
                ln, tin, tout = meta[(me, f)]
                pin = meta[(me, 2 * lane + 1 - d)][1]
                frd[x][y] = [f, ln, tab_t.buffer_address(), ws_t.buffer_address(), xy((x, y))] + tl_all
                pair = fcore(lane, 1 - d)
                # (every chip has the same forwarder cores: the next chip's forwarder and pair sit at the same xy)
                nid = int(node(mesh, nxt[f]).chip_id)
                fsd[x][y] = [ln, tin, tout, pin, mesh_id, nid, xy((x, y)), xy(pair), xy(pair)]
            kernels = list(gprog.kernels) + [
                relay,
                ttnn.KernelDescriptor(
                    kernel_source=kpath("wf_sreader.cpp"),
                    core_ranges=crs([rc.local]),
                    compile_time_args=[cfg.nexp, KT, KTD, DC, cfg.PAIRS, cfg.HT, rc.CHT, cfg.G, cfg.GC]
                    + list(ttnn.TensorAccessorArgs(ws_t).get_compile_time_args()),
                    runtime_args=lrd,
                    defines=[("RD_DEPTH", str(rc.RD)), ("SCB_CHUNKS", str(rc.SCB)), ("NOWN", str(NCHIP))]
                    + sdefs
                    + odefs,
                    config=dm(0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=kpath("wf_bsend.cpp"),
                    core_ranges=crs([rc.local]),
                    compile_time_args=ct_send + ct_send_icred(me),
                    runtime_args=lsd,
                    defines=[("WF_LOCAL", "1")] + sdefs,
                    config=dm(1),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=kpath("wr_land.cpp"),
                    core_ranges=crs(fcores),
                    compile_time_args=ct_ring
                    + [ex["wgu_ring"].buffer_address(), ex["wd_ring"].buffer_address(), ex["warr"].buffer_address()]
                    + [icred.buffer_address(), TL]
                    + list(ttnn.TensorAccessorArgs(ws_t).get_compile_time_args())
                    + list(ttnn.TensorAccessorArgs(tab_t).get_compile_time_args()),
                    runtime_args=frd,
                    defines=[("RD_DEPTH", str(rc.RD)), ("SCB_CHUNKS", str(rc.SCB))] + sdefs + odefs,
                    config=dm(0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=kpath("wr_fwd.cpp"),
                    core_ranges=crs(fcores),
                    compile_time_args=ct_ring + [rc.CRB],
                    runtime_args=fsd,
                    defines=FAB_DEFS + sdefs,
                    config=dm(1),
                ),
            ]
            kf = len(kernels) - 1
            cbs = list(gprog.cbs) + [
                cb(0, cfg.w_dt, rc.SCB * rc.CHT, crs(land)),
                cb(1, cfg.w_dt, rc.S * rc.CHT, crs(fcores)),
                cb(2, ttnn.uint32, TL, crs(fcores), page=4),
            ]
            p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=list(gprog.semaphores), cbs=cbs)
            for lane, d, cyc, l in rl:
                x, y = fcore(lane, d)
                p.kernels[kf].runtime_args[x][y].extend(
                    ttnn.setup_fabric_connection(
                        mesh_device=mesh,
                        src_fabric_node_id=node(mesh, me_rc),
                        dst_fabric_node_id=node(mesh, nxt[2 * lane + d]),
                        link_idx=l,
                        program_descriptor=p,
                        worker_core=ttnn.CoreCoord(x, y),
                    )
                )
            mc = ttnn.MeshCoordinate(r, c)
            mpd[ttnn.MeshCoordinateRange(mc, mc)] = p
    return mpd, io + [icred, words, sst, tab_t], st, sst

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert groups with the weights forwarded store-and-forward around Hamiltonian cycles (every chip runs experts;
expert e is owned by chip e % NCHIP, as in ws_fab.build_all).

The fabric's own multicast trees send everything over an N/S trunk with E/W teeth and the shortest way round, which
loads some links about twice their share. Here every packet goes one hop only: a cycle through all chips, both
directions, each of its links, is a directed ring with one forwarder core per chip. A lane is a (cycle, link) pair;
the group cores of every expert are dealt to the lanes (j % NL), and each slice goes both ways round its lane from the
owner, half way each (wr_seq.hpp). Per directed ring and chip one forwarder core:
- BRISC (wr_land.cpp): reads the chip's own slices from DRAM into CB 0 (owner), lands passing packets from its
  staging slots (CB 1) into the group cores' 2-slot rings, fusing the arrival increment after a slice's last packet;
- NCRISC (wr_fwd.cpp): sends injected and passing packets one hop into the next chip's forwarder (fused write +
  arrival increment), and relays the opposite ring's per-hop staging credits to the next chip.
A local lander (wf_sreader + wf_bsend WF_LOCAL) writes the chip's own slices into its own group cores, and a relay
core (wr_relay.cpp) fans the group cores' freed-slot counts out to all landers of the chip: no credit crosses more
than one hop, and none goes to the owners.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.ws_expert.bench_util import cb, crs, kpath
from models.demos.ws_expert.ws_expert import noc, sharded_l1
from models.demos.ws_expert.ws_fab import FAB_DEFS, node
from models.demos.ws_expert.ws_group import GCfg, build, group_cores


@dataclass
class RCfg:
    CHT: int = 24  # tiles a packet (24 x 576 B bfp4 = 13.8 KB)
    S: int = 8  # staging slots a forwarder (packets)
    SCB: int = 8  # owner read-ahead a forwarder / local lander staging (chunks)
    RD: int = 4  # DRAM reads in flight (chunks)
    CRB: int = 2  # while packets flow, relay staging credits CRB at a time (at once when the forwarder waits)
    NL: int = 2  # links a cycle edge
    relay: tuple = (10, 9)
    local: tuple = (9, 9)
    # forwarder cores, index 2 lane + dir (outside the groups; lanes = cycles x links)
    fwd: tuple = ((9, 0), (10, 0), (9, 1), (10, 1), (9, 2), (10, 2), (9, 3), (10, 3))
    cycles: tuple = None  # Hamiltonian cycles (mesh coordinates in ring order); default: the row snake
    own_slices: bool = False  # each chip's slices tensor holds only its own experts (WS_OWN_SLICES)


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
    nb = lambda v: [((v // cols + 1) % rows) * cols + v % cols, ((v // cols - 1) % rows) * cols + v % cols]
    nbrs = lambda v: nb(v) + [(v // cols) * cols + (v % cols + 1) % cols, (v // cols) * cols + (v % cols - 1) % cols]
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
    kf = N // 2 if (e + s) & 1 else (N - 1) // 2
    return kf if d == 0 else N - 1 - kf


def ring_schedule(nexp, NCHIP, G, GC, NL, NCH, hop, d, lane, N):
    """Host mirror of wr::walk: [(kind, e, n, c, u)] (0 INJ, 1 FWD, 2 TERM)."""
    out = []
    for e in range(nexp):
        h = hop[e % NCHIP]
        if h > N // 2:
            continue
        g, n = e % G, e // G
        for s in range(GC // NL):
            r = reach(e, s, d, N)
            if r == 0 or h > r:
                continue
            kind = 0 if h == 0 else (1 if h < r else 2)
            for u in range(NCH):
                out.append((kind, e, n, g * GC + NL * s + lane, u))
    return out


def rings(mesh_shape, rc: RCfg):
    """Directed rings: [(lane, dir, cycle, link)] and, per chip index and ring, (next chip coord, hop[owner])."""
    R, Cn = mesh_shape
    cycles = [list(c) for c in (rc.cycles or [snake(mesh_shape)])]
    out = []
    for k, cyc in enumerate(cycles):
        assert sorted(cyc) == sorted((r, c) for r in range(R) for c in range(Cn)), "a cycle visits every chip once"
        for l in range(rc.NL):
            for d in range(2):
                out.append((k * rc.NL + l, d, cyc, l))
    return out


def hops(cyc, me_rc, d, Cn):
    """hop[o]: hops from chip o (index r * Cn + c) to me along the cycle in direction d (0: list order)."""
    N = len(cyc)
    pos = {rc: i for i, rc in enumerate(cyc)}
    i = pos[me_rc]
    step = 1 if d == 0 else -1
    out = [0] * N
    for rc, p in pos.items():
        out[rc[0] * Cn + rc[1]] = ((i - p) * step) % N
    return out


def build_sf(mesh, cfg: GCfg, rc: RCfg, x_t, idx_t, counts_t, ws_t, out_t):
    R, Cn = tuple(mesh.shape)
    NCHIP = R * Cn
    NC = cfg.G * cfg.GC
    KT, KTD, DC = cfg.KT, cfg.PAIRS, cfg.DCOLS
    NCH = (2 * KT + KTD * DC) // rc.CHT
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
    sdefs = [("SST_ADDR", str(sst.buffer_address()))]
    odefs = [("WS_OWN_SLICES", "1")] if rc.own_slices else []
    ct_ring = [cfg.nexp, KT, KTD, DC, rc.CHT, cfg.G, cfg.GC, NCHIP, NLANE, rc.S]
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
                x, y = fcore(lane, d)
                N = len(cyc)
                i = cyc.index(me_rc)
                nxt[(lane, d)] = cyc[(i + (1 if d == 0 else -1)) % N]
                hop, hop_pair = hops(cyc, me_rc, d, Cn), hops(cyc, me_rc, 1 - d, Cn)
                frd[x][y] = [d, lane, N, me, ws_t.buffer_address(), xy((x, y))] + hop + tl_all
                pair = fcore(lane, 1 - d)
                fsd[x][y] = [d, lane, N, mesh_id, int(node(mesh, nxt[(lane, d)]).chip_id), xy((x, y)), xy(pair)]
                fsd[x][y] += [xy(pair)] + hop + hop_pair
            kernels = list(gprog.kernels) + [
                relay,
                ttnn.KernelDescriptor(
                    kernel_source=kpath("wf_sreader.cpp"),
                    core_ranges=crs([rc.local]),
                    compile_time_args=[cfg.nexp, KT, KTD, DC, cfg.PAIRS, cfg.HT, rc.CHT, cfg.G, cfg.GC]
                    + list(ttnn.TensorAccessorArgs(ws_t).get_compile_time_args()),
                    runtime_args=lrd,
                    defines=[("RD_DEPTH", str(rc.RD)), ("SCB_CHUNKS", str(rc.SCB)), ("NOWN", str(NCHIP))] + sdefs + odefs,
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
                    + [words.buffer_address(), icred.buffer_address()]
                    + list(ttnn.TensorAccessorArgs(ws_t).get_compile_time_args()),
                    runtime_args=frd,
                    defines=[("RD_DEPTH", str(rc.RD)), ("SCB_CHUNKS", str(rc.SCB))] + sdefs + odefs,
                    config=dm(0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=kpath("wr_fwd.cpp"),
                    core_ranges=crs(fcores),
                    compile_time_args=ct_ring + [words.buffer_address(), rc.CRB],
                    runtime_args=fsd,
                    defines=FAB_DEFS + sdefs,
                    config=dm(1),
                ),
            ]
            kf = len(kernels) - 1
            cbs = list(gprog.cbs) + [
                cb(0, cfg.w_dt, rc.SCB * rc.CHT, crs(land)),
                cb(1, cfg.w_dt, rc.S * rc.CHT, crs(fcores)),
            ]
            p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=list(gprog.semaphores), cbs=cbs)
            for lane, d, cyc, l in rl:
                x, y = fcore(lane, d)
                p.kernels[kf].runtime_args[x][y].extend(
                    ttnn.setup_fabric_connection(
                        mesh_device=mesh,
                        src_fabric_node_id=node(mesh, me_rc),
                        dst_fabric_node_id=node(mesh, nxt[(lane, d)]),
                        link_idx=l,
                        program_descriptor=p,
                        worker_core=ttnn.CoreCoord(x, y),
                    )
                )
            mc = ttnn.MeshCoordinate(r, c)
            mpd[ttnn.MeshCoordinateRange(mc, mc)] = p
    return mpd, io + [icred, words, sst], st, sst

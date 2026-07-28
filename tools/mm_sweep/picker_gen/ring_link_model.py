#!/usr/bin/env python3
"""Whole-op NoC link-load model for Regime-A, used to choose the in0 ring order offline.

Extends ring_topology_probe.py (in0-ring-only) with the three traffic classes the ring competes with:
in1 DRAM reads, the split-K reduction chain, output DRAM writes, and the in0 own-shard DRAM read. Those are
FIXED given the placement, so they form a static background load map; only the ring order is choosable, and
the right objective is to route the ring through the valleys of that background rather than to minimise the
ring's own hops (which is what production does) or the ring's own peak (bit9, which regressed where in1 is
heavy).

Also sweeps a HOP BUDGET: minimise peak combined load subject to ring hops <= (1+eps) * hop-optimal.
eps=0 is production-like, eps=inf is bit9.

Coordinate frame (all validated against the device, see notes below):
  logical (x,y) -> relative (px[x], py[y]); physical = (px+1, py+2); torus 17x12.
  DRAM: banks 0-3 at physical x=0 -> relative x=16 (wrap); banks 4-7 at physical x=9 -> relative x=8.
  Each bank's endpoint row == its bank-adjacent worker's row (verified against blackhole_140_arch.yaml).
  NOC_0 travels +x/+y, NOC_1 -x/-y, dimension-ordered; both orders charged since x-first vs y-first is not
  observable from hop counts.

usage: ring_link_model.py Mt Kt Nt Ns Pk Sm kb nsb
"""
import sys, itertools
import ttnn
from ttnn._ttnn.multi_device import experimental as dexp

GX, GY = 11, 10
TB = 2048  # bf16 tile bytes


def cdiv(a, b):
    return -(-a // b)


class Geo:
    def __init__(self, Mt, Kt, Nt, Ns, Pk, Sm, kb, nsb):
        self.Mt, self.Kt, self.Nt = Mt, Kt, Nt
        self.Ns, self.Pk, self.Sm, self.kb = Ns, Pk, Sm, kb
        self.K_slice_cap = cdiv(cdiv(Kt, Pk), kb * 8) * kb * 8
        self.M_block = cdiv(Mt, Sm)
        self.N_band = cdiv(Nt, 8)
        self.N_own = cdiv(self.N_band, Ns)
        self.N_sub = nsb if nsb else self.N_own
        self.N_bpc = cdiv(self.N_own, self.N_sub)
        self.W = (self.K_slice_cap // kb) // 8
        self.preaders = Pk * Ns * Sm
        self.ncores = 8 * self.preaders
        self.shard_bytes = self.W * self.M_block * kb * TB
        self.in1_bytes = self.K_slice_cap * self.N_sub * self.N_bpc * TB  # per core, whole kernel
        self.red_bytes = self.N_bpc * self.M_block * self.N_sub * TB  # per non-top core
        self.out_bytes = self.M_block * self.N_sub * self.N_bpc * TB  # per root core


class Frame:
    """Relative torus frame + routing, reconstructed from device hop distances."""

    def __init__(self, md):
        self.md = md
        self.px = [0] * GX
        for i in range(GX - 1):
            self.px[i + 1] = self.px[i] + self._h((i, 0), (i + 1, 0), 0)
        self.py = [0] * GY
        for j in range(GY - 1):
            self.py[j + 1] = self.py[j] + self._h((0, j), (0, j + 1), 0)
        self.WX = self.px[GX - 1] + self._h((GX - 1, 0), (0, 0), 0)
        self.WY = self.py[GY - 1] + self._h((0, GY - 1), (0, 0), 0)

    def _h(self, a, b, noc):
        return dexp.get_worker_noc_hop_distance(
            self.md, ttnn.CoreCoord(*a), ttnn.CoreCoord(*b), {0: ttnn.NOC.NOC_0, 1: ttnn.NOC.NOC_1}[noc])

    def hop(self, a, b, noc):
        return 0 if a == b else self._h(a, b, noc)

    def rel(self, c):
        return (self.px[c[0]], self.py[c[1]])

    def links(self, s, d, noc):
        """Distinct links on both dimension orders, in relative coords (s, d already relative)."""
        out = set()
        for xfirst in (True, False):
            x, y = s
            tx, ty = d
            sx = 1 if noc == 0 else self.WX - 1
            sy = 1 if noc == 0 else self.WY - 1
            if xfirst:
                while x != tx:
                    out.add((noc, 0, x, y))
                    x = (x + sx) % self.WX
                while y != ty:
                    out.add((noc, 1, x, y))
                    y = (y + sy) % self.WY
            else:
                while y != ty:
                    out.add((noc, 1, x, y))
                    y = (y + sy) % self.WY
                while x != tx:
                    out.add((noc, 0, x, y))
                    x = (x + sx) % self.WX
        return out


def place(g, fr, opt):
    """Replicate build_plan's find_near, plus place_m_split_workers (IN1_NEAR) when Sm>1."""
    coords = [None] * g.ncores
    nocs = [0] * g.ncores
    mm = [0] * g.ncores
    used = set()

    def find_near(t):
        for d in range(GX + GY):
            for dx in range(-d, d + 1):
                rem = d - abs(dx)
                for sgn in (0, 1):
                    dy = -rem if sgn else rem
                    x, y = t[0] + dx, t[1] + dy
                    if 0 <= x < GX and 0 <= y < GY and (x, y) not in used:
                        used.add((x, y))
                        return (x, y)
        return t

    for b in range(8):
        for p in range(g.preaders):
            i = b * g.preaders + p
            nocs[i] = (p // g.Sm) & 1 if g.Sm > 1 else p & 1
            mm[i] = p % g.Sm
    if g.Sm == 1:
        for b in range(8):
            for p in range(g.preaders):
                i = b * g.preaders + p
                coords[i] = find_near(opt[b])
    else:
        for b in range(8):  # pass 1: mm==0 readers first
            for p in range(g.preaders):
                i = b * g.preaders + p
                if mm[i] == 0:
                    coords[i] = find_near(opt[b])
        for b in range(8):  # pass 2: slaves at the free core minimising reader->slave hop (reader NoC)
            for p in range(g.preaders):
                i = b * g.preaders + p
                if mm[i] == 0:
                    continue
                rc = coords[i - mm[i]]
                rnoc = nocs[i]
                best, bd = None, None
                for y in range(GY):
                    for x in range(GX):
                        if (x, y) in used:
                            continue
                        d = fr.hop(rc, (x, y), rnoc)
                        if bd is None or d < bd:
                            bd, best = d, (x, y)
                used.add(best)
                coords[i] = best
    return coords, nocs, mm


def main():
    Mt, Kt, Nt, Ns, Pk, Sm, kb, nsb = (int(x) for x in sys.argv[1:9])
    g = Geo(Mt, Kt, Nt, Ns, Pk, Sm, kb, nsb)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        fr = Frame(md)
        opt = [(c.x, c.y) for c in md.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]
        coords, nocs, mm = place(g, fr, opt)
        wnoc = [1 - nocs[i] for i in range(g.ncores)]  # writer NoC = opposite the reader's
        # DRAM nodes in relative coords: banks 0-3 one column west of logical x=0; banks 4-7 in the middle gap.
        dram = [((fr.px[0] - 1) % fr.WX if b < 4 else fr.px[5] + 3, fr.py[opt[b][1]]) for b in range(8)]

        # ---- static background: in1 reads, in0 own-shard read, reduction chain, output writes ----
        bg = {}

        def add(s, d, noc, nbytes, into):
            ls = fr.links(s, d, noc)
            if not ls:
                return
            for l in ls:
                into[l] = into.get(l, 0) + nbytes

        for i in range(g.ncores):
            c = fr.rel(coords[i])
            b = i // g.preaders
            kk = (i % g.preaders) // (g.Ns * g.Sm)
            is_top = kk == g.Pk - 1
            # in1: DRAM->core on the READER NoC, from this core's own bank (M-split slaves get it from the
            # mm==0 reader instead, over the reader NoC).
            if g.Sm == 1 or mm[i] == 0:
                add(dram[b], c, nocs[i], g.in1_bytes, bg)
            else:
                add(fr.rel(coords[i - mm[i]]), c, nocs[i], g.in1_bytes, bg)
            # in0 own shard: interleaved DRAM -> core on the WRITER NoC, spread over all 8 banks.
            for bb in range(8):
                add(dram[bb], c, wnoc[i], g.shard_bytes // 8, bg)
            # split-K reduction: core -> next k-band on the WRITER NoC.
            if g.Pk > 1 and not is_top:
                add(c, fr.rel(coords[i + g.Ns * g.Sm]), wnoc[i], g.red_bytes, bg)
            # output: root core -> interleaved DRAM on the WRITER NoC.
            if g.Pk == 1 or is_top:
                for bb in range(8):
                    add(c, dram[bb], wnoc[i], g.out_bytes // 8, bg)

        bg_peak = max(bg.values())
        print(f"geo: cores={g.ncores} shard={g.shard_bytes/1024:.0f}KB W={g.W} M_block={g.M_block} "
              f"N_sub={g.N_sub} N_bpc={g.N_bpc}")
        print(f"background peak link load = {bg_peak/1e6:.2f} MB   (links loaded: {len(bg)})")

        # ---- ring candidates: per (kk,nn) group, one permutation shared by its Sm mm-rings ----
        ngroups = g.preaders // g.Sm
        groups = []
        for gi in range(ngroups):
            base = gi * g.Sm
            per_mm = []
            for m in range(g.Sm):
                members = [b * g.preaders + base + m for b in range(8)]
                w = wnoc[members[0]]
                per_mm.append((members, w))
            groups.append(per_mm)

        lkcache = {}
        def edge_links(a, b, noc):
            k = (coords[a], coords[b], noc)
            if k not in lkcache:
                lkcache[k] = fr.links(fr.rel(coords[a]), fr.rel(coords[b]), noc)
            return lkcache[k]

        def cyc(order, members, w):
            """(max edge, total hops) of one cyclic order on one mm-ring."""
            es = [fr.hop(coords[members[order[p]]], coords[members[order[(p + 1) % 8]]], w) for p in range(8)]
            return max(es), sum(es)

        def cyc_hops(order, members, w):
            return cyc(order, members, w)[1]

        def agg(order, gi):
            """(aggmax, aggtot) over the group's Sm mm-rings — the production PARETO metrics."""
            mx = tot = 0
            for mem, w in groups[gi]:
                m, t = cyc(order, mem, w)
                mx = max(mx, m)
                tot += t
            return mx, tot

        perms = [(0,) + p for p in itertools.permutations(range(1, 8))]

        prod_order = [min(perms, key=lambda o: agg(o, gi)) for gi in range(ngroups)]

        def evaluate(eps, cap_edge=True):
            """Sequential greedy (2 passes): minimise peak COMBINED load subject to a hop budget and (when
            cap_edge) to never worsening the worst directed edge, which sets the 7-step ring LATENCY and
            dominates when the shard is small."""
            load = dict(bg)
            chosen = [None] * ngroups
            # Budgets are anchored on the order PRODUCTION would pick (PARETO: min aggmax then aggtot), so the
            # feasible set always contains it => the search can never do worse than production by construction.
            hopt = [agg(prod_order[gi], gi)[1] for gi in range(ngroups)]
            mxopt = [agg(prod_order[gi], gi)[0] for gi in range(ngroups)]
            for _ in range(2):
                for gi in range(ngroups):
                    if chosen[gi] is not None:
                        for m, (mem, w) in enumerate(groups[gi]):
                            o = chosen[gi]
                            for p in range(8):
                                for l in edge_links(mem[o[p]], mem[o[(p + 1) % 8]], w):
                                    load[l] -= 7 * g.shard_bytes
                    budget = hopt[gi] * (1 + eps)
                    best, bscore = None, None
                    for o in perms:
                        amx, hops = agg(o, gi)
                        if hops > budget or (cap_edge and amx > mxopt[gi]):
                            continue
                        peak = 0
                        for mem, w in groups[gi]:
                            for p in range(8):
                                for l in edge_links(mem[o[p]], mem[o[(p + 1) % 8]], w):
                                    peak = max(peak, load.get(l, 0) + 7 * g.shard_bytes)
                        score = (peak, hops)
                        if bscore is None or score < bscore:
                            bscore, best = score, o
                    chosen[gi] = best
                    for mem, w in groups[gi]:
                        for p in range(8):
                            for l in edge_links(mem[best[p]], mem[best[(p + 1) % 8]], w):
                                load[l] = load.get(l, 0) + 7 * g.shard_bytes
            return max(load.values()), sum(sum(cyc_hops(chosen[gi], mem, w) for mem, w in groups[gi])
                                           for gi in range(ngroups))

        # production reference: per-group PARETO-optimal (min aggmax, then aggtot), chosen independently.
        load = dict(bg)
        prod_hops = 0
        for gi in range(ngroups):
            best = prod_order[gi]
            prod_hops += agg(best, gi)[1]
            for mem, w in groups[gi]:
                for p in range(8):
                    for l in edge_links(mem[best[p]], mem[best[(p + 1) % 8]], w):
                        load[l] = load.get(l, 0) + 7 * g.shard_bytes
        prod_peak = max(load.values())
        print(f"\n{'strategy':<40} {'peak MB':>9} {'vs prod':>8} {'ring hops':>10}")
        print(f"{'production (PARETO, independent)':<40} {prod_peak/1e6:9.2f} {1.0:8.2f} {prod_hops:10d}")
        for cap in (True, False):
            for eps in (0.1, 0.5, 99.0):
                pk, hp = evaluate(eps, cap_edge=cap)
                tag = ("edge-capped, " if cap else "edge-free,   ") + (f"eps={eps}" if eps < 99 else "no hop budget")
                print(f"{tag:<40} {pk/1e6:9.2f} {pk/prod_peak:8.2f} {hp:10d}")
    finally:
        ttnn.close_mesh_device(md)


main()

#!/usr/bin/env python3
"""Offline search for a better in1-read core placement (no matmul runs).

The in1 read response path is DRAM endpoint -> core on the reader's NoC, dimension-ordered and strictly
unidirectional with torus wrap. So each (bank, noc) pair has a "downstream" region that is cheap and a
wrap-around region that costs a full lap. Placement therefore decides in1 read distance almost entirely.

KEY OBSERVATION about how to search: with the reader NoC fixed, "minimise total in1 read hop-bytes" is a
LINEAR ASSIGNMENT problem (slots = (bank, kk, nn) readers, cells = logical cores, cost = response hops), so
it has an EXACT polynomial solution (Hungarian) - no heuristic search needed. Only the peak-link-load
objective is non-linear (min-max), and that is handled by iterative reweighting on top of the exact solution.

Variants compared:
  current   : build_plan find_near spiral + noc = (p/Sm)&1 (production)
  cross     : a priori heuristic - place each (bank, noc) group in its endpoint's row, walking downstream
  hung/alt  : exact assignment, production's alternating NoC rule
  hung/block: exact assignment, NoC assigned by CONTIGUOUS kk blocks (so each split-K reduction chain
              crosses the chip once instead of Pk-1 times)
  hung/pk   : hung/block plus peak-link reweighting iterations

Reported per variant: in1 read hops/hop-bytes/peak link, and the induced reduction-chain and in0-ring cost,
because moving cores for in1 charges those (the "second nail").

usage: in1_place_search.py Mt Kt Nt Ns Pk Sm kb nsb
"""
import sys, itertools, yaml
import numpy as np
from scipy.optimize import linear_sum_assignment
import ttnn
from ttnn._ttnn.multi_device import experimental as dexp

GX, GY = 11, 10
TB = 2048
SOC = "tt_metal/soc_descriptors/blackhole_140_arch.yaml"


def cdiv(a, b):
    return -(-a // b)


def dram_endpoints():
    d = yaml.safe_load(open(SOC))
    banks = [[tuple(int(v) for v in c.split("-")) for c in grp] for grp in d["dram"]]
    out = {}
    for v in d["dram_views"]:
        ch, we = v["channel"], v["worker_endpoint"]
        for noc in (0, 1):
            out[(ch, noc)] = banks[ch][we[noc]]
    return out


class Frame:
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
        self.XOFF, self.YOFF = 1, 2

    def _h(self, a, b, noc):
        return dexp.get_worker_noc_hop_distance(
            self.md, ttnn.CoreCoord(*a), ttnn.CoreCoord(*b), {0: ttnn.NOC.NOC_0, 1: ttnn.NOC.NOC_1}[noc])

    def hop(self, a, b, noc):
        return 0 if a == b else self._h(a, b, noc)

    def rl(self, c):
        return (self.px[c[0]], self.py[c[1]])

    def rp(self, p):
        return ((p[0] - self.XOFF) % self.WX, (p[1] - self.YOFF) % self.WY)

    def route(self, s, d, noc):
        x, y = s
        tx, ty = d
        sx = 1 if noc == 0 else self.WX - 1
        sy = 1 if noc == 0 else self.WY - 1
        ls = []
        while x != tx:
            ls.append((noc, 0, x, y))
            x = (x + sx) % self.WX
        while y != ty:
            ls.append((noc, 1, x, y))
            y = (y + sy) % self.WY
        return ls


def main():
    Mt, Kt, Nt, Ns, Pk, Sm, kb, nsb = (int(x) for x in sys.argv[1:9])
    K_slice = cdiv(cdiv(Kt, Pk), kb * 8) * kb * 8
    M_block = cdiv(Mt, Sm)
    N_band = cdiv(Nt, 8)
    N_own = cdiv(N_band, Ns)
    N_sub = nsb if nsb else N_own
    N_bpc = cdiv(N_own, N_sub)
    W = (K_slice // kb) // 8
    preaders = Pk * Ns * Sm
    ncores = 8 * preaders
    shard_b = W * M_block * kb * TB
    in1_b = K_slice * N_sub * N_bpc * TB
    red_b = N_bpc * M_block * N_sub * TB
    ep = dram_endpoints()

    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        fr = Frame(md)
        opt = [(c.x, c.y) for c in md.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]
        cells = [(x, y) for y in range(GY) for x in range(GX)]
        print(f"cfg Ns={Ns} Pk={Pk} Sm={Sm} kb={kb} nsb={nsb}: {ncores} cores, {preaders}/bank, "
              f"{8*Pk*Ns} DRAM readers, in1/reader {in1_b/1024:.0f} KB, shard {shard_b/1024:.0f} KB")

        # ---- reader slots: (bank, kk, nn); mm==0 only reads DRAM ----
        slots = [(b, kk, nn) for b in range(8) for kk in range(Pk) for nn in range(Ns)]

        def noc_alt(b, kk, nn):      # production rule: noc = (p/Sm)&1, p = kk*Ns*Sm + nn*Sm
            return (kk * Ns + nn) & 1

        def noc_block(b, kk, nn):    # contiguous kk blocks => each reduction chain crosses once
            return 0 if kk < (Pk + 1) // 2 else 1

        def resp_hops(slot, cell, nocf):
            b, kk, nn = slot
            noc = nocf(b, kk, nn)
            return len(fr.route(fr.rp(ep[(b, noc)]), fr.rl(cell), noc))

        # ---- variant builders: each returns {slot: cell} plus a noc function ----
        def v_current():
            used, place = set(), {}
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
                for p in range(preaders):
                    kk = p // (Ns * Sm)
                    nn = (p % (Ns * Sm)) // Sm
                    mm = p % Sm
                    c = find_near(opt[b])
                    if mm == 0:
                        place[(b, kk, nn)] = c
            return place, noc_alt

        def v_cross(nocf):
            """A priori: put each (bank,noc) group in its endpoint's row, walking downstream in x."""
            used, place = set(), {}
            groups = {}
            for s in slots:
                groups.setdefault((s[0], nocf(*s)), []).append(s)
            # order groups by how constrained they are (smallest cheap region first): banks 4-7 NOC_1 etc.
            for (b, noc), ss in sorted(groups.items()):
                e = fr.rp(ep[(b, noc)])
                cand = sorted(cells, key=lambda c: (len(fr.route(e, fr.rl(c), noc)), c[1], c[0]))
                for s in ss:
                    for c in cand:
                        if c not in used:
                            used.add(c)
                            place[s] = c
                            break
            return place, nocf

        def v_hungarian(nocf, extra_cost=None):
            C = np.zeros((len(slots), len(cells)))
            for i, s in enumerate(slots):
                for j, c in enumerate(cells):
                    C[i, j] = resp_hops(s, c, nocf) * in1_b
                    if extra_cost is not None:
                        C[i, j] += extra_cost(s, c, nocf)
            r, cc = linear_sum_assignment(C)
            return {slots[i]: cells[j] for i, j in zip(r, cc)}, nocf

        # ---- metrics ----
        def evaluate(name, place, nocf):
            # slaves (Sm>1) go IN1_NEAR to their reader on the reader NoC; readers keep their cells
            used = set(place.values())
            full = {}
            for (b, kk, nn), c in place.items():
                full[(b, kk, nn, 0)] = c
            for (b, kk, nn), c in sorted(place.items()):
                noc = nocf(b, kk, nn)
                for mm in range(1, Sm):
                    best = min((len(fr.route(fr.rl(c), fr.rl(d), noc)), d) for d in cells if d not in used)[1]
                    used.add(best)
                    full[(b, kk, nn, mm)] = best
            load, in1_hops = {}, 0
            def charge(s, d, noc, nbytes):
                for l in fr.route(s, d, noc):
                    load[l] = load.get(l, 0) + nbytes
            # in1 reads (readers only) + slave forwards
            for (b, kk, nn), c in place.items():
                noc = nocf(b, kk, nn)
                h = len(fr.route(fr.rp(ep[(b, noc)]), fr.rl(c), noc))
                in1_hops += h
                charge(fr.rp(ep[(b, noc)]), fr.rl(c), noc, in1_b)
                for mm in range(1, Sm):
                    charge(fr.rl(c), fr.rl(full[(b, kk, nn, mm)]), noc, in1_b)
            in1_peak = max(load.values()) if load else 0
            # reduction chain: (b,nn,mm) fixed, kk -> kk+1, on the WRITER NoC (opposite the reader's)
            red_hops = 0
            for (b, kk, nn, mm), c in full.items():
                if kk + 1 < Pk:
                    w = 1 - nocf(b, kk, nn)
                    d = full[(b, kk + 1, nn, mm)]
                    red_hops += len(fr.route(fr.rl(c), fr.rl(d), w))
                    charge(fr.rl(c), fr.rl(d), w, red_b)
            # in0 ring: per slice (kk,nn,mm), the 8 banks, production PARETO order on the writer NoC
            ring_hops = 0
            for kk in range(Pk):
                for nn in range(Ns):
                    for mm in range(Sm):
                        w = 1 - nocf(0, kk, nn)
                        mem = [full[(b, kk, nn, mm)] for b in range(8)]
                        dm = [[len(fr.route(fr.rl(mem[i]), fr.rl(mem[j]), w)) if i != j else 0
                               for j in range(8)] for i in range(8)]
                        best = None
                        for perm in itertools.permutations(range(1, 8)):
                            o = (0,) + perm
                            es = [dm[o[p]][o[(p + 1) % 8]] for p in range(8)]
                            m = (max(es), sum(es))
                            if best is None or m < best[0]:
                                best = (m, o)
                        ring_hops += best[0][1]
                        o = best[1]
                        for p in range(8):
                            charge(fr.rl(mem[o[p]]), fr.rl(mem[o[(p + 1) % 8]]), w, 7 * shard_b)
            allpeak = max(load.values())
            nr = len(place)
            print(f"  {name:<12} in1: {in1_hops:5d} hops ({in1_hops/nr:4.1f} mean) "
                  f"{in1_hops*in1_b/1e6:6.0f} MB-hops  peak_in1 {in1_peak/1e6:5.2f} MB | "
                  f"red {red_hops:4d} hops | ring {ring_hops:4d} hops | ALL-peak {allpeak/1e6:5.2f} MB")
            return in1_hops, in1_peak, red_hops, ring_hops, allpeak

        # per-endpoint egress floor: every reader of an endpoint shares its first link
        per_ep = {}
        for s in slots:
            per_ep[(s[0], noc_alt(*s))] = per_ep.get((s[0], noc_alt(*s)), 0) + 1
        floor = max(per_ep.values()) * in1_b
        print(f"  endpoint-egress floor for peak in1 link load: {floor/1e6:.2f} MB "
              f"({max(per_ep.values())} readers share one endpoint's first link)\n")

        base = evaluate("current", *v_current())
        evaluate("cross/alt", *v_cross(noc_alt))
        evaluate("cross/block", *v_cross(noc_block))
        h_alt = evaluate("hung/alt", *v_hungarian(noc_alt))
        h_blk = evaluate("hung/block", *v_hungarian(noc_block))

        # peak-aware reweighting: penalise cells whose path uses links already loaded by the previous solve
        place, nocf = v_hungarian(noc_block)
        for it in range(3):
            load = {}
            for s, c in place.items():
                noc = nocf(*s)
                for l in fr.route(fr.rp(ep[(s[0], noc)]), fr.rl(c), noc):
                    load[l] = load.get(l, 0) + in1_b
            lam = 0.5
            def extra(s, c, nf, load=load, lam=lam):
                noc = nf(*s)
                return lam * sum(load.get(l, 0) for l in fr.route(fr.rp(ep[(s[0], noc)]), fr.rl(c), noc))
            place, nocf = v_hungarian(noc_block, extra_cost=extra)
        evaluate("hung/pk", place, nocf)

        print(f"\n  (current in1 = {base[0]} hops / {base[1]/1e6:.2f} MB peak; lower is better everywhere)")
    finally:
        ttnn.close_mesh_device(md)


main()

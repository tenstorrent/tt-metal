#!/usr/bin/env python3
"""Offline search for a placement that minimises in0 RING traffic (no matmul runs).

Redirected here by the measured result that in1 reads are already DRAM-bound (76-98% of peak in isolation), so
placement cannot pay through in1 - but the one large full-op win we saw (+15.2% on 256x15360x768) came from
the in0 ring getting shorter as a side effect. So the ring is the nail placement can actually hit.

THE STRUCTURAL TENSION. Two traffic classes want opposite groupings of the same 8 x preaders core array:
  - the in0 RING connects the 8 cores of one SLICE (one per bank)  -> wants slice-compact clusters
  - the split-K REDUCTION chain connects the Pk cores of one BANK  -> wants bank-compact clusters
These are orthogonal partitions, so no layout makes both compact... except a 2D EMBEDDING: lay banks along one
grid axis and slices along the other, and then BOTH a ring step (bank -> bank) and a reduction step
(kk -> kk+1) become one hop. Production makes bank-compact blobs (short reduction, long ring).

Candidates evaluated:
  current    production find_near spiral around opt[bank] (now per-NoC correct after the API cache fix)
  ring_block partition the grid into preaders blocks of 8 adjacent cells; slice p -> block p
  mesh       2D embedding: bank along x, slice along y (folded when preaders > grid.y)
  mesh_t     transposed: slice along x, bank along y
Reports ring / reduction / in1 hop-bytes and the whole-op peak link load, so the trade is explicit.

usage: in0_ring_place_search.py Mt Kt Nt Ns Pk Sm kb nsb
"""
import sys, itertools, yaml
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

    def _h(self, a, b, noc):
        return dexp.get_worker_noc_hop_distance(
            self.md, ttnn.CoreCoord(*a), ttnn.CoreCoord(*b), {0: ttnn.NOC.NOC_0, 1: ttnn.NOC.NOC_1}[noc])

    def hop(self, a, b, noc):
        return 0 if a == b else self._h(a, b, noc)

    def rl(self, c):
        return (self.px[c[0]], self.py[c[1]])

    def rp(self, p):
        return ((p[0] - 1) % self.WX, (p[1] - 2) % self.WY)

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
    out_b = M_block * N_sub * N_bpc * TB
    ep = dram_endpoints()

    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        fr = Frame(md)
        optn = [[(c.x, c.y) for c in md.get_optimal_dram_bank_to_logical_worker_assignment(n)]
                for n in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)]
        cells = [(x, y) for y in range(GY) for x in range(GX)]
        print(f"cfg Ns={Ns} Pk={Pk} Sm={Sm} kb={kb} nsb={nsb}: {ncores} cores, {preaders} slices/bank")
        print(f"  weights per edge: ring 7x{shard_b/1024:.0f}KB={7*shard_b/1024:.0f}KB, red {red_b/1024:.0f}KB, "
              f"in1 {in1_b/1024:.0f}KB, out {out_b/1024:.0f}KB")

        def noc_of(p):
            return (p // Sm) & 1 if Sm > 1 else p & 1

        def slice_of(p):
            return (p // (Ns * Sm), (p % (Ns * Sm)) // Sm, p % Sm)  # (kk, nn, mm)

        # ---------- layouts: return coord[(bank, p)] ----------
        def lay_current():
            used, co = set(), {}
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
                    co[(b, p)] = find_near(optn[noc_of(p)][b])
            return co

        def lay_ring_block():
            """preaders blocks of 8 adjacent cells (2x4 where possible); slice p -> block p, bank -> cell."""
            co = {}
            blocks = []
            # tile the grid with 2x4 blocks in row-major order, then whatever fits
            for y0 in range(0, GY - 1, 2):
                for x0 in range(0, GX - 3, 4):
                    blocks.append([(x0 + i, y0 + j) for j in range(2) for i in range(4)])
            free = [c for c in cells if not any(c in b for b in blocks)]
            while len(blocks) < preaders and len(free) >= 8:
                blocks.append(free[:8])
                free = free[8:]
            for p in range(min(preaders, len(blocks))):
                for b in range(8):
                    co[(b, p)] = blocks[p][b]
            return co

        def lay_mesh(transpose=False):
            """2D embedding: one axis = bank, other = slice, folded to fit the grid."""
            co = {}
            for b in range(8):
                for p in range(preaders):
                    if not transpose:
                        x, y = b, p
                        if y >= GY:                    # fold: wrap extra slices into the free columns
                            x, y = 8 + (p - GY) % (GX - 8), (p - GY) // (GX - 8) * 4 + b % GY
                    else:
                        x, y = p, b
                        if x >= GX:
                            x, y = (p - GX) % GX, 8 + (b % 2)
                    co[(b, p)] = (min(x, GX - 1), min(y, GY - 1))
            # repair collisions by nearest-free
            used, out = set(), {}
            for key in sorted(co, key=lambda k: (k[1], k[0])):
                c = co[key]
                if c in used:
                    c = min((abs(d[0] - c[0]) + abs(d[1] - c[1]), d) for d in cells if d not in used)[1]
                used.add(c)
                out[key] = c
            return out

        # ---------- evaluation ----------
        def ev(name, co):
            load = {}
            def charge(s, d, noc, nb):
                for l in fr.route(s, d, noc):
                    load[l] = load.get(l, 0) + nb
            ring_h = red_h = in1_h = 0
            # in1 read (mm==0 only) + slave forwards
            for b in range(8):
                for p in range(preaders):
                    kk, nn, mm = slice_of(p)
                    noc = noc_of(p)
                    c = co[(b, p)]
                    if mm == 0:
                        h = len(fr.route(fr.rp(ep[(b, noc)]), fr.rl(c), noc))
                        in1_h += h
                        charge(fr.rp(ep[(b, noc)]), fr.rl(c), noc, in1_b)
                    else:
                        r = co[(b, p - mm)]
                        charge(fr.rl(r), fr.rl(c), noc, in1_b)
                    # in0 own-shard read: interleaved DRAM, spread over all banks, on the WRITER NoC
                    for bb in range(8):
                        charge(fr.rp(ep[(bb, 1 - noc)]), fr.rl(c), 1 - noc, shard_b // 8)
                    # reduction: (b, kk+1) same nn/mm, on the WRITER NoC
                    if kk + 1 < Pk:
                        d = co[(b, p + Ns * Sm)]
                        red_h += len(fr.route(fr.rl(c), fr.rl(d), 1 - noc))
                        charge(fr.rl(c), fr.rl(d), 1 - noc, red_b)
                    else:  # top band writes the output to interleaved DRAM
                        for bb in range(8):
                            charge(fr.rl(c), fr.rp(ep[(bb, 1 - noc)]), 1 - noc, out_b // 8)
            # in0 ring: per slice, PARETO-optimal cycle over the 8 banks on the writer NoC
            for p in range(preaders):
                w = 1 - noc_of(p)
                mem = [co[(b, p)] for b in range(8)]
                dm = [[len(fr.route(fr.rl(mem[i]), fr.rl(mem[j]), w)) if i != j else 0 for j in range(8)]
                      for i in range(8)]
                best = None
                for perm in itertools.permutations(range(1, 8)):
                    o = (0,) + perm
                    es = [dm[o[q]][o[(q + 1) % 8]] for q in range(8)]
                    m = (max(es), sum(es))
                    if best is None or m < best[0]:
                        best = (m, o)
                ring_h += best[0][1]
                o = best[1]
                for q in range(8):
                    charge(fr.rl(mem[o[q]]), fr.rl(mem[o[(q + 1) % 8]]), w, 7 * shard_b)
            print(f"  {name:<11} ring {ring_h:5d} hops ({ring_h*7*shard_b/1e6:6.0f} MB-hops) | "
                  f"red {red_h:5d} ({red_h*red_b/1e6:5.0f} MB-h) | in1 {in1_h:5d} ({in1_h*in1_b/1e6:6.0f} MB-h) | "
                  f"ALL-peak {max(load.values())/1e6:6.2f} MB | sum {sum(load.values())/1e6:7.0f} MB-h")

        ev("current", lay_current())
        ev("ring_block", lay_ring_block())
        ev("mesh", lay_mesh(False))
        ev("mesh_t", lay_mesh(True))
    finally:
        ttnn.close_mesh_device(md)


main()

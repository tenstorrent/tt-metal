#!/usr/bin/env python3
"""Analysis of Regime-A core placement from the in1-READ point of view (no matmul run).

Answers: where does the current (Sm, Pk, Ns) placement put each bank's readers, how far is each from the DRAM
endpoint it actually reads through, and how much in1 read traffic is spent on avoidable NoC distance.

Key hardware facts this uses (all verified, see the report):
  - Each BH DRAM channel has THREE NoC endpoints ("subchannels"); `dram_views.worker_endpoint = [a, b]`
    selects subchannel a for NOC_0 and b for NOC_1, so the endpoint a core reads through depends on its NoC.
  - Each NoC is strictly unidirectional per dimension with torus wrap: NOC_0 = +x/+y, NOC_1 = -x/-y
    (device-verified: logical (3,0)->(3,1) is 1 hop on NOC_0 and 11 on NOC_1).
  - A read's DATA travels from the DRAM endpoint to the worker on the SAME NoC the request used, so the
    response path length depends on which side of the DRAM column the worker sits.

usage: in1_placement_probe.py Mt Kt Nt Ns Pk Sm kb nsb
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
    """(bank, noc) -> physical (x, y) of the DRAM NoC endpoint actually used."""
    d = yaml.safe_load(open(SOC))
    banks = [[tuple(int(v) for v in c.split("-")) for c in grp] for grp in d["dram"]]
    out = {}
    for v in d["dram_views"]:
        ch, we = v["channel"], v["worker_endpoint"]
        for noc in (0, 1):
            out[(ch, noc)] = banks[ch][we[noc]]
    return out, banks


class Frame:
    """Relative torus frame reconstructed from device hop distances; physical = (rel_x + 1, rel_y + 2)."""

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
        self.XOFF, self.YOFF = 1, 2  # physical = rel + off (validated: torus 17x12 matches the SOC grid)

    def _h(self, a, b, noc):
        return dexp.get_worker_noc_hop_distance(
            self.md, ttnn.CoreCoord(*a), ttnn.CoreCoord(*b), {0: ttnn.NOC.NOC_0, 1: ttnn.NOC.NOC_1}[noc])

    def hop(self, a, b, noc):
        return 0 if a == b else self._h(a, b, noc)

    def rel_of_logical(self, c):
        return (self.px[c[0]], self.py[c[1]])

    def rel_of_phys(self, p):
        return ((p[0] - self.XOFF) % self.WX, (p[1] - self.YOFF) % self.WY)

    def route(self, s, d, noc):
        """Links traversed, dimension-ordered (x then y), in the relative frame."""
        x, y = s
        tx, ty = d
        sx = 1 if noc == 0 else self.WX - 1
        sy = 1 if noc == 0 else self.WY - 1
        links = []
        while x != tx:
            links.append((noc, 0, x, y))
            x = (x + sx) % self.WX
        while y != ty:
            links.append((noc, 1, x, y))
            y = (y + sy) % self.WY
        return links


def place(g, fr, opt):
    """build_plan find_near (bank-major, slice-minor, first-fit spiral) + place_m_split_workers for Sm>1."""
    n = g["ncores"]
    coords, nocs, mm = [None] * n, [0] * n, [0] * n
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

    P, Sm = g["preaders"], g["Sm"]
    for b in range(8):
        for p in range(P):
            i = b * P + p
            nocs[i] = (p // Sm) & 1 if Sm > 1 else p & 1
            mm[i] = p % Sm
    if Sm == 1:
        for b in range(8):
            for p in range(P):
                coords[b * P + p] = find_near(opt[b])
    else:
        for b in range(8):
            for p in range(P):
                i = b * P + p
                if mm[i] == 0:
                    coords[i] = find_near(opt[b])
        for b in range(8):
            for p in range(P):
                i = b * P + p
                if mm[i] == 0:
                    continue
                rc = coords[i - mm[i]]
                best, bd = None, None
                for y in range(GY):
                    for x in range(GX):
                        if (x, y) in used:
                            continue
                        d = fr.hop(rc, (x, y), nocs[i])
                        if bd is None or d < bd:
                            bd, best = d, (x, y)
                used.add(best)
                coords[i] = best
    return coords, nocs, mm


def main():
    Mt, Kt, Nt, Ns, Pk, Sm, kb, nsb = (int(x) for x in sys.argv[1:9])
    g = {
        "Ns": Ns, "Pk": Pk, "Sm": Sm, "kb": kb,
        "K_slice_cap": cdiv(cdiv(Kt, Pk), kb * 8) * kb * 8,
        "M_block": cdiv(Mt, Sm), "N_band": cdiv(Nt, 8),
    }
    g["N_own"] = cdiv(g["N_band"], Ns)
    g["N_sub"] = nsb if nsb else g["N_own"]
    g["N_bpc"] = cdiv(g["N_own"], g["N_sub"])
    g["W"] = (g["K_slice_cap"] // kb) // 8
    g["preaders"] = Pk * Ns * Sm
    g["ncores"] = 8 * g["preaders"]
    g["shard_bytes"] = g["W"] * g["M_block"] * kb * TB
    g["in1_bytes"] = g["K_slice_cap"] * g["N_sub"] * g["N_bpc"] * TB

    ep, banks = dram_endpoints()
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        fr = Frame(md)
        opt = [(c.x, c.y) for c in md.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]
        opt1 = [(c.x, c.y) for c in md.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_1)]
        coords, nocs, mm = place(g, fr, opt)
        P = g["preaders"]

        print(f"config Ns={Ns} Pk={Pk} Sm={Sm} kb={kb} nsb={nsb} -> {g['ncores']} cores, {P} per bank; "
              f"in1 per reader {g['in1_bytes']/1024:.0f} KB, shard {g['shard_bytes']/1024:.0f} KB")
        print(f"torus {fr.WX}x{fr.WY}; bank-adjacent workers NOC_0={opt}")
        print(f"                              NOC_1={opt1}   (identical => placement ignores the NoC)")
        print("\nDRAM endpoints actually read through (physical coords; 3 subchannels per bank, 2 used):")
        for b in range(8):
            used_sub = {ep[(b, 0)], ep[(b, 1)]}
            unused = [c for c in banks[b] if c not in used_sub]
            print(f"  bank {b}: NOC_0 endpoint {ep[(b,0)]}   NOC_1 endpoint {ep[(b,1)]}   unused subchannel {unused}")

        # ---- per-core in1 read response path: DRAM endpoint -> core, on the core's reader NoC ----
        rows = []
        for i in range(g["ncores"]):
            if Sm > 1 and mm[i] != 0:
                continue  # slaves receive in1 from their mm==0 reader, they do not read DRAM
            b, noc = i // P, nocs[i]
            src = fr.rel_of_phys(ep[(b, noc)])
            dst = fr.rel_of_logical(coords[i])
            hops = len(fr.route(src, dst, noc))
            rows.append({"i": i, "bank": b, "noc": noc, "coord": coords[i], "hops": hops})

        print(f"\nin1 read response path length (DRAM endpoint -> reader, on the reader's NoC), "
              f"{len(rows)} DRAM readers:")
        print(f"  {'bank':>4} {'NOC_0 readers: hops':>34}   {'NOC_1 readers: hops':>34}")
        tot0 = tot1 = 0
        for b in range(8):
            h0 = sorted(r["hops"] for r in rows if r["bank"] == b and r["noc"] == 0)
            h1 = sorted(r["hops"] for r in rows if r["bank"] == b and r["noc"] == 1)
            tot0 += sum(h0)
            tot1 += sum(h1)
            print(f"  {b:>4} {str(h0):>34}   {str(h1):>34}")
        n0 = sum(1 for r in rows if r["noc"] == 0)
        n1 = len(rows) - n0
        print(f"  NOC_0: {n0} readers, mean {tot0/max(n0,1):.1f} hops   "
              f"NOC_1: {n1} readers, mean {tot1/max(n1,1):.1f} hops")
        print(f"  in1 read hop-bytes: NOC_0 {tot0*g['in1_bytes']/1e6:.0f} MB-hops, "
              f"NOC_1 {tot1*g['in1_bytes']/1e6:.0f} MB-hops "
              f"(NOC_1 share {100*tot1/max(tot0+tot1,1):.0f}%)")

        # ---- counterfactuals ----
        def cost(noc_of, coord_of):
            t = 0
            for r in rows:
                b = r["bank"]
                noc = noc_of(r)
                src = fr.rel_of_phys(ep[(b, noc)])
                t += len(fr.route(src, fr.rel_of_logical(coord_of(r)), noc))
            return t

        cur = cost(lambda r: r["noc"], lambda r: r["coord"])
        # (A) side-aware NoC: pick the NoC whose response direction reaches this core without wrapping
        def side_aware(r):
            b = r["bank"]
            dst = fr.rel_of_logical(r["coord"])
            best, bn = None, 0
            for noc in (0, 1):
                h = len(fr.route(fr.rel_of_phys(ep[(b, noc)]), dst, noc))
                if best is None or h < best:
                    best, bn = h, noc
            return bn
        alt_a = cost(side_aware, lambda r: r["coord"])
        # (B) free choice of BOTH: for each reader, the best (noc, any free core) — a lower bound on what
        #     placement+NoC could achieve if in1 were the only consideration (ignores collisions).
        lb = 0
        for r in rows:
            b = r["bank"]
            best = None
            for noc in (0, 1):
                src = fr.rel_of_phys(ep[(b, noc)])
                for y in range(GY):
                    for x in range(GX):
                        h = len(fr.route(src, fr.rel_of_logical((x, y)), noc))
                        if best is None or h < best:
                            best = h
            lb += best
        # (C)/(D) ACHIEVABLE greedy placements honouring collisions: walk the readers in the existing
        # bank-major order and give each the closest still-free core in its response direction. (C) keeps the
        # existing noc=p&1 assignment, (D) also lets each reader pick its NoC. Slaves (Sm>1) are then placed
        # IN1_NEAR as today, so they still consume cells; model that by reserving Sm-1 neighbours per reader.
        def greedy(free_noc):
            used = set()
            tot = 0
            for r in rows:
                b = r["bank"]
                cands = []
                for noc in ((0, 1) if free_noc else (r["noc"],)):
                    src = fr.rel_of_phys(ep[(b, noc)])
                    for y in range(GY):
                        for x in range(GX):
                            if (x, y) in used:
                                continue
                            cands.append((len(fr.route(src, fr.rel_of_logical((x, y)), noc)), x, y))
                if not cands:
                    continue
                h, x, y = min(cands)
                used.add((x, y))
                tot += h
                for _ in range(Sm - 1):  # reserve the IN1_NEAR slave cells this reader will pull in
                    nb = min(((abs(xx - x) + abs(yy - y), xx, yy) for yy in range(GY) for xx in range(GX)
                              if (xx, yy) not in used), default=None)
                    if nb:
                        used.add((nb[1], nb[2]))
            return tot

        alt_c = greedy(False)
        alt_d = greedy(True)
        print(f"\nin1 read distance, total hops over all DRAM readers:")
        print(f"  current placement + noc=p&1 rule      : {cur:5d} hops  ({cur*g['in1_bytes']/1e6:6.0f} MB-hops)")
        print(f"  (A) same placement, side-aware NoC    : {alt_a:5d} hops  "
              f"({alt_a*g['in1_bytes']/1e6:6.0f} MB-hops)  {100*(alt_a-cur)/cur:+.0f}%")
        print(f"  (C) direction-aware placement, noc=p&1: {alt_c:5d} hops  "
              f"({alt_c*g['in1_bytes']/1e6:6.0f} MB-hops)  {100*(alt_c-cur)/cur:+.0f}%   ACHIEVABLE")
        print(f"  (D) direction-aware placement + NoC   : {alt_d:5d} hops  "
              f"({alt_d*g['in1_bytes']/1e6:6.0f} MB-hops)  {100*(alt_d-cur)/cur:+.0f}%   ACHIEVABLE")
        print(f"  (B) unconstrained lower bound         : {lb:5d} hops  ({lb*g['in1_bytes']/1e6:6.0f} MB-hops)  "
              f"{100*(lb-cur)/cur:+.0f}%")
        # in0 ring traffic for scale: every ring edge carries 7 shards, 8 edges per ring, Pk*Ns*Sm rings.
        ring_bytes = g["preaders"] * 8 * 7 * g["shard_bytes"]
        print(f"\nscale check: in0 ring = {ring_bytes/1e6:.1f} MB of payload (~x3.6 hops = "
              f"{ring_bytes*3.6/1e6:.0f} MB-hops) vs in1 read {cur*g['in1_bytes']/1e6:.0f} MB-hops "
              f"=> in1 read link traffic is {cur*g['in1_bytes']/(ring_bytes*3.6):.1f}x the in0 ring's")

        # ---- link hotspots from in1 read traffic alone ----
        load = {}
        for r in rows:
            src = fr.rel_of_phys(ep[(r["bank"], r["noc"])])
            for l in fr.route(src, fr.rel_of_logical(r["coord"]), r["noc"]):
                load[l] = load.get(l, 0) + g["in1_bytes"]
        top = sorted(load.items(), key=lambda kv: -kv[1])[:8]
        print(f"\nin1-read-only link load: {len(load)} links, busiest {top[0][1]/1e6:.1f} MB")
        for (noc, ax, x, y), v in top:
            print(f"  NOC_{noc} {'x' if ax==0 else 'y'}-link at rel({x},{y}) phys({x+fr.XOFF},{y+fr.YOFF}): "
                  f"{v/1e6:.2f} MB")
    finally:
        ttnn.close_mesh_device(md)


main()

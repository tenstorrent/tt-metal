#!/usr/bin/env python3
"""Offline geometry comparison of the two in0 ring topologies (no matmul run).

Replicates build_plan's find_near placement + both ring strategies (production per-slice bank ring with the
two-pass PARETO order, and diag bit8's region-local regrouping), then reports for each:
  - total / max directed hop cost per ring (real device API)
  - modelled PER-LINK load (dimension-ordered torus routing, weight = 7 shards per edge), i.e. the
    busiest-link metric that actually gates a congested NoC.
Routing assumption (NOC_0 = +x/+y, NOC_1 = -x/-y, wrap) is validated against the device hop counts.

usage: ring_topology_probe.py Mt Kt Nt Ns Pk Sm   (Sm=1 only: Sm>1 also gets IN1_NEAR placement)
"""
import sys, itertools
import ttnn
from ttnn._ttnn.multi_device import experimental as dexp

GX, GY = 11, 10
Mt, Kt, Nt, Ns, Pk, Sm = (int(x) for x in sys.argv[1:7])
assert Sm == 1, "probe models the Sm==1 placement path only"
PREADERS = Pk * Ns * Sm
NCORES = 8 * PREADERS


def find_near_all(opt):
    """build_plan placement: bank-major, slice-minor, Manhattan spiral, first-fit."""
    used, coords, nocs = set(), [None] * NCORES, [0] * NCORES
    for b in range(8):
        for p in range(PREADERS):
            i = b * PREADERS + p
            noc = (p // Sm) & 1 if Sm > 1 else p & 1
            nocs[i] = noc
            tx, ty = opt[b]
            placed = None
            for d in range(GX + GY):
                for dx in range(-d, d + 1):
                    rem = d - abs(dx)
                    for sgn in (0, 1):
                        dy = -rem if sgn else rem
                        x, y = tx + dx, ty + dy
                        if not (0 <= x < GX and 0 <= y < GY) or (x, y) in used:
                            continue
                        used.add((x, y))
                        placed = (x, y)
                        break
                    if placed:
                        break
                if placed:
                    break
            coords[i] = placed
    return coords, nocs


class Phys:
    """Logical->physical torus coords, reconstructed empirically from device hop distances.

    Logical worker columns/rows are NOT contiguous in physical NoC space (the DRAM/PCIe columns sit between
    them), so logical Manhattan != hops. Walk single logical steps on NOC_0 (which routes +x/+y) to recover
    the physical spacing, then derive the torus extent from the wrap-around distance.
    """

    def __init__(self, md, dexp):
        self.md, self.dexp = md, dexp
        self.px = [0] * GX
        for i in range(GX - 1):
            self.px[i + 1] = self.px[i] + self._h((i, 0), (i + 1, 0), 0)
        self.py = [0] * GY
        for j in range(GY - 1):
            self.py[j + 1] = self.py[j] + self._h((0, j), (0, j + 1), 0)
        self.WX = self.px[GX - 1] + self._h((GX - 1, 0), (0, 0), 0)
        self.WY = self.py[GY - 1] + self._h((0, GY - 1), (0, 0), 0)

    def _h(self, a, b, noc):
        return self.dexp.get_worker_noc_hop_distance(
            self.md, ttnn.CoreCoord(*a), ttnn.CoreCoord(*b), {0: ttnn.NOC.NOC_0, 1: ttnn.NOC.NOC_1}[noc])

    def manhattan(self, a, b):
        """Direction-agnostic torus proximity (the right metric for clustering a CYCLE)."""
        dx = abs(self.px[a[0]] - self.px[b[0]])
        dy = abs(self.py[a[1]] - self.py[b[1]])
        return min(dx, self.WX - dx) + min(dy, self.WY - dy)

    def route_links(self, src, dst, noc, xfirst=True):
        """Physical links traversed under dimension-ordered torus routing (NOC_0 +x/+y, NOC_1 -x/-y)."""
        x, y = self.px[src[0]], self.py[src[1]]
        tx, ty = self.px[dst[0]], self.py[dst[1]]
        step = 1 if noc == 0 else -1
        links = []

        def walk_x(x, y):
            while x != tx:
                links.append((noc, "x", x, y))
                x = (x + step) % self.WX
            return x

        def walk_y(x, y):
            while y != ty:
                links.append((noc, "y", x, y))
                y = (y + step) % self.WY
            return y

        if xfirst:
            x = walk_x(x, y)
            y = walk_y(x, y)
        else:
            y = walk_y(x, y)
            x = walk_x(x, y)
        return links


def best_cycle(items, dm, Smn=1):
    """Two-pass PARETO order used by both production and the regrouping (Sm=1 => single ring)."""
    def metrics(order):
        mx = tot = 0
        for p in range(8):
            e = dm[order[p]][order[(p + 1) % 8]]
            tot += e
            mx = max(mx, e)
        return mx, tot
    head, tail = items[0], sorted(items[1:])
    best, bm = None, None
    for perm in itertools.permutations(tail):
        cand = (head,) + perm
        m = metrics(cand)
        if bm is None or m < bm:
            bm, best = m, cand
    return list(best), bm


def main():
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        opt = [(c.x, c.y) for c in md.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]
        coords, nocs = find_near_all(opt)
        wnoc = [1 if nocs[i] == 0 else 0 for i in range(NCORES)]  # writer NoC = opposite the reader's
        NOCE = {0: ttnn.NOC.NOC_0, 1: ttnn.NOC.NOC_1}

        hop_cache = {}
        def hop(a, b, noc):
            k = (coords[a], coords[b], noc)
            if k not in hop_cache:
                hop_cache[k] = 0 if a == b else dexp.get_worker_noc_hop_distance(
                    md, ttnn.CoreCoord(*coords[a]), ttnn.CoreCoord(*coords[b]), NOCE[noc])
            return hop_cache[k]

        ph = Phys(md, dexp)
        print(f"physical torus {ph.WX}x{ph.WY}; logical col->phys {ph.px}; row->phys {ph.py}")
        # validate the routing model against the device hop counts
        bad = tot = 0
        for a in range(0, NCORES, 7):
            for b in range(0, NCORES, 5):
                if a == b:
                    continue
                tot += 1
                if len(ph.route_links(coords[a], coords[b], wnoc[a])) != hop(a, b, wnoc[a]):
                    bad += 1
        print(f"routing-model mismatches vs device hop counts: {bad} of {tot}")

        def report(name, rings):
            load, tot_all, mx_all, cross = {}, 0, 0, 0
            for order in rings:
                for p in range(8):
                    a, b = order[p], order[(p + 1) % 8]
                    e = hop(a, b, wnoc[a])
                    tot_all += e
                    mx_all = max(mx_all, e)
                    if (coords[a][0] <= 3) != (coords[b][0] <= 3):
                        cross += 1  # edge straddles the left/right bank-column split
                    for lk in ph.route_links(coords[a], coords[b], wnoc[a]):
                        load[lk] = load.get(lk, 0) + 7  # 7 shards cross every ring edge
            srt = sorted(load.values(), reverse=True)
            print(f"\n{name}")
            print(f"  rings={len(rings)}  total hops={tot_all}  worst edge={mx_all}  bisection-crossing edges={cross}")
            print(f"  links used={len(srt)}  busiest={srt[0]} shards  top8={srt[:8]}  sum shard-hops={sum(srt)}")
            return srt[0], tot_all, sum(srt), cross

        # --- production: one ring per slice = the 8 banks, PARETO order on the slice's shared writer NoC
        prod = []
        for j in range(PREADERS):
            items = [b * PREADERS + j for b in range(8)]
            dm = {a: {b: hop(a, b, wnoc[items[0]]) for b in items} for a in items}
            order, _ = best_cycle(items, dm)
            prod.append(order)
        base = report("PRODUCTION (per-slice bank ring)", prod)

        def partitioned(name, part_fn):
            rings = []
            for kk in range(Pk):
                items = [b * PREADERS + (kk * Ns + nn) for b in range(8) for nn in range(Ns)]
                for grp in part_fn(kk, items):
                    dm = {a: {b: hop(a, b, wnoc[a]) for b in grp} for a in grp}
                    order, _ = best_cycle(grp, dm)
                    rings.append(order)
            r = report(name, rings)
            print(f"  vs production: hops {r[1]/base[1]:.2f}x  shard-hops {r[2]/base[2]:.2f}x  "
                  f"BUSIEST LINK {r[0]/base[0]:.2f}x  crossings {base[3]}->{r[3]}")
            for order in rings[:2]:
                print("    ring: " + " -> ".join(str(coords[i]) for i in order))
            return r

        # (a) as shipped in diag bit8: greedy on summed DIRECTED hop distance
        def part_directed(kk, items):
            sym = {a: {b: (hop(a, b, wnoc[a]) + hop(b, a, wnoc[b])) for b in items} for a in items}
            un, out = list(items), []
            for _ in range(Ns):
                seed = max(un, key=lambda a: sum(sym[a][b] for b in un if b != a))
                grp = [seed]
                un.remove(seed)
                while len(grp) < 8:
                    nxt = min(un, key=lambda a: min(sym[a][m] for m in grp))
                    grp.append(nxt)
                    un.remove(nxt)
                out.append(grp)
            return out

        # (b) same greedy but on DIRECTION-AGNOSTIC physical torus Manhattan distance
        def part_manhattan(kk, items):
            un, out = list(items), []
            for _ in range(Ns):
                seed = max(un, key=lambda a: sum(ph.manhattan(coords[a], coords[b]) for b in un if b != a))
                grp = [seed]
                un.remove(seed)
                while len(grp) < 8:
                    nxt = min(un, key=lambda a: min(ph.manhattan(coords[a], coords[m]) for m in grp))
                    grp.append(nxt)
                    un.remove(nxt)
                out.append(grp)
            return out

        # (c) deterministic bank-half split: banks 0-3 target logical col 0, banks 4-7 col 6 => guaranteed
        #     region-local for Ns==2 (8+8). Only exact when 4*Ns == 8.
        def part_bankhalf(kk, items):
            lo = [i for i in items if (i // PREADERS) < 4]
            hi = [i for i in items if (i // PREADERS) >= 4]
            return [lo[i:i + 8] for i in range(0, len(lo), 8)] + [hi[i:i + 8] for i in range(0, len(hi), 8)]

        partitioned("(a) REGIONAL as shipped (directed-hop clustering)", part_directed)
        partitioned("(b) REGIONAL manhattan clustering", part_manhattan)
        if (4 * Ns) % 8 == 0:
            partitioned("(c) REGIONAL bank-half split (deterministic)", part_bankhalf)
        for j, order in enumerate(prod[:2]):
            print(f"  production ring {j}: " + " -> ".join(str(coords[i]) for i in order))

        # ---- (d) LINK-BALANCED ordering: same partition, but choose each ring's cycle to minimise the
        # GLOBAL max link load (tie-break total hops) given the rings already placed. Sequential greedy +
        # one re-optimisation pass. This is idea 2 and needs no re-partitioning.
        linkcache = {}
        def lk(a, b):
            if (a, b) not in linkcache:
                linkcache[(a, b)] = ph.route_links(coords[a], coords[b], wnoc[a])
            return linkcache[(a, b)]

        def cycles(items):
            head, tail = items[0], sorted(items[1:])
            return [(head,) + p for p in itertools.permutations(tail)]

        def balance(part_fn, label, passes=2):
            groups = [g for kk in range(Pk)
                      for g in part_fn(kk, [b * PREADERS + (kk * Ns + nn) for b in range(8) for nn in range(Ns)])]
            chosen = [None] * len(groups)
            load = {}
            def add(order, sign):
                for p in range(8):
                    a, b = order[p], order[(p + 1) % 8]
                    for l in lk(a, b):
                        load[l] = load.get(l, 0) + sign
            for _ in range(passes):
                for gi, grp in enumerate(groups):
                    if chosen[gi] is not None:
                        add(chosen[gi], -1)
                    best, bscore = None, None
                    for cand in cycles(grp):
                        mx = hops = 0
                        seen = {}
                        for p in range(8):
                            a, b = cand[p], cand[(p + 1) % 8]
                            hops += hop(a, b, wnoc[a])
                            for l in lk(a, b):
                                seen[l] = seen.get(l, 0) + 1
                        for l, c in seen.items():
                            mx = max(mx, load.get(l, 0) + c)
                        for l, c in load.items():
                            if l not in seen:
                                mx = max(mx, c)
                        score = (mx, hops)
                        if bscore is None or score < bscore:
                            bscore, best = score, cand
                    chosen[gi] = best
                    add(best, +1)
            r = report(label, chosen)
            print(f"  vs production: hops {r[1]/base[1]:.2f}x  shard-hops {r[2]/base[2]:.2f}x  "
                  f"BUSIEST LINK {r[0]/base[0]:.2f}x  crossings {base[3]}->{r[3]}")
            return r

        def part_prod(kk, items):
            return [[b * PREADERS + (kk * Ns + nn) for b in range(8)] for nn in range(Ns)]

        balance(part_prod, "(d) PRODUCTION partition + LINK-BALANCED order")
        if (4 * Ns) % 8 == 0:
            balance(part_bankhalf, "(e) BANK-HALF partition + LINK-BALANCED order")
    finally:
        ttnn.close_mesh_device(md)


main()

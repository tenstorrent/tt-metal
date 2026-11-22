# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""K2a (M5 option B) host replica of the unit cores' schedule (kernels/k2a_common.hpp Builder), the packed Q layout,
and the bounds the kernels' records and CBs are sized to. Pure python: no device. (m5_14 F1) the tail chunk is the
block's last tile's; a straddling block's first-row tile goes to its own chunk's unit, a piece (tail = piece = True).

  python -m models.demos.gemma4_31b_qb2.resident_m5_14.k2a_sched --bounds   (worst cases over many position mixes)
"""
import argparse
import random

R = 32
DTS, DTF = 8, 16
WINDOW = 1024
CHUNK_S, CHUNK_F = 8, 64
NVS, NVF = 8, 8  # (m5_14 F1) NVS 9 -> 8: no straddling tail tile
MAX_CHUNKS_F = 8  # (K2b step 1) the flat merge of <= 8 partials
MAX_UNITS, MAX_ROOTS, MAX_Q = 40, 4, 8  # (K2b step 1) MAX_Q: the per-kind key list (Q streams one key at a time)
MAX_CHUNKS_T, GROUP, MAX_NODES, NP = 128, 16, 8, 150  # (K2b step 3) the tree; cb_parts pages
ROOT_CAP_S, ROOT_CAP_F = 3, 1


def pack(full, M, rows=None):
    """(QH, Tu, Ht, k, tiles) of the packed Q layout (k2a_common.hpp Pack); rows: the launch's R (D16: 8 or 32)"""
    rows = R if rows is None else rows
    QH = 8 if full else 2
    Ru = QH * M
    Tu = (Ru + 31) // 32
    Ht = QH // Tu
    k = 32 // (Ht * M)
    U = rows // M
    return QH, Tu, Ht, k, -(-U // k) * Tu


def pack_tile_row(full, M, u, h, m):
    QH, Tu, Ht, k, _ = pack(full, M)
    return (u // k) * Tu + h // Ht, ((h % Ht) * k + u % k) * M + m


def pt(full, M):
    """partial pages of a unit (one Q tile: a full user spanning two packed tiles, M = 8, has units per half)"""
    return (DTF if full else DTS) + 2


def window(full, p, M):
    tlo = 0 if full else ((p - (WINDOW - 1)) // 32 if p >= WINDOW - 1 else 0)
    wl = (p + M - 1 - (WINDOW - 1)) // 32 - tlo + 1 if (not full and p + M - 1 >= WINDOW - 1) else 0
    return tlo, wl


class Schedule:
    """the Builder's global assignment: per core its unit / root / Q-slot records (both kinds)"""

    def __init__(self, positions, M, num_units=60, per_head=15, rows=None, live=None):
        self.M, self.U = M, (R if rows is None else rows) // M
        self.positions = positions  # per user: p, or None (inert / flagged)
        # (m5_14 F1, M131) per user the block rows to its last live one (default M): a block capped by verify_cap
        self.live = [M] * len(positions) if live is None else live
        self.cores = [dict(units=[], roots=[], nodes=[], q={0: [], 1: []}, parts=[0, 0]) for _ in range(num_units)]
        self.ok = True
        load = [0] * num_units
        nunits, ntask = [0] * num_units, [0] * num_units  # (K2b step 3) the global fit: every core's counts
        for kind in (0, 1):
            full = kind == 1
            nroot, parts, ntailc = [0] * num_units, [0] * num_units, [0] * num_units
            tails = {c: [] for c in range(num_units)}
            nonroot = {c: [] for c in range(num_units)}
            heads, ch = (pack(True, M)[1], CHUNK_F) if full else (4, CHUNK_S)  # (full: g = the half)
            PT, cap = pt(full, M), (ROOT_CAP_F if full else ROOT_CAP_S)
            for g in range(heads):
                c0, cn = (0, num_units) if full else (g * per_head, per_head)
                for u in range(self.U):
                    p = positions[u]
                    if p is None:
                        continue
                    ml = self.live[u]
                    tt0, tend = p // 32, (p + ml - 1) // 32
                    tlo, wl = window(full, p, ml)
                    # (m5_14 F1) ctl: the last tile's chunk; cfirst: the first tile's (ctl - 1 when straddling)
                    clo, ctl, cfirst = tlo // ch, tend // ch, tt0 // ch
                    k = ctl - clo + 1
                    lng = full and k > MAX_CHUNKS_F  # (K2b step 3) a long user: the group-node tree
                    if full and k > MAX_CHUNKS_T:
                        self.ok = False
                        continue
                    ng = -(-k // GROUP) if lng else k
                    # (m5_14 F1) and room for the root's partials (binds only for a 6-chunk sliding user)
                    cands = [c for c in range(c0, c0 + cn) if nroot[c] < cap and parts[c] + ng * PT <= NP]
                    if not cands:
                        self.ok = False
                        continue
                    root = min(cands, key=lambda c: (load[c], c))
                    pbase = parts[root]
                    parts[root] += ng * PT
                    nroot[root] += 1
                    a0 = max(tlo, ctl * ch)
                    load[root] += tend - a0 + 1 + 4
                    nodes = []
                    for j in range(ng if lng else 0):
                        b = min(range(num_units), key=lambda c: (ntask[c], load[c], c))
                        n_j = min(GROUP, k - GROUP * j)
                        if ntask[b] >= MAX_NODES:
                            self.ok = False
                        nodes.append((b, ntask[b] % MAX_NODES))
                        ntask[b] += 1
                        load[b] += n_j
                        self.cores[b]["nodes"].append(dict(user=u, g=g, n=n_j, lane=nodes[-1][1], c0=GROUP * j,
                                                           root=root, pslot=j, pbase=pbase))

                    def dest(cc):
                        if lng:
                            return dict(dram=True, root=nodes[cc // GROUP][0], pslot=cc, lane=nodes[cc // GROUP][1])
                        return dict(dram=False, root=root, pslot=cc - clo, pbase=pbase)

                    nunits[root] += 1
                    ntailc[root] += 1
                    if nunits[root] > MAX_UNITS or ntailc[root] > 8:
                        self.ok = False
                    wend = tlo + wl
                    n = tend - a0 + 1
                    nw = min(wend - a0, n) if a0 < wend else 0
                    rec = dict(user=u, g=g, full=full, tail=True, piece=False, nW=nw, nT=tend - max(tt0, a0) + 1,
                               t0=a0, n=n, k=ng, tt0=tt0, **dest(ctl))
                    self.cores[root]["roots"].append(dict(user=u, g=g, full=full, k=ng, pbase=pbase))
                    tails[root].append(rec)
                    for cc in range(clo, ctl):
                        b = min(range(c0, c0 + cn), key=lambda c: (load[c], c))
                        s0 = max(tlo, cc * ch)
                        n = (cc + 1) * ch - s0
                        load[b] += n
                        nunits[b] += 1
                        if nunits[b] > MAX_UNITS:
                            self.ok = False
                        nw = min(wend - s0, n) if s0 < wend else 0
                        # (m5_14 F1) cc == cfirst < ctl: the straddling block's piece (splices [tt0, (cc + 1) ch))
                        pc = cc == cfirst
                        nonroot[b].append(dict(user=u, g=g, full=full, tail=pc, piece=pc, nW=nw,
                                               nT=(cc + 1) * ch - tt0 if pc else 0, t0=s0, n=n, tt0=tt0 if pc else 0,
                                               **dest(cc)))
            for c in range(num_units):
                units = nonroot[c] + tails[c]
                kq = pack(full, M)[3]
                for w in units:
                    Tu = pack(True, M)[1]
                    key = (w["user"] // kq) * Tu + w["g"] if full else (w["user"] // kq, w["g"])
                    if key not in self.cores[c]["q"][kind]:
                        self.cores[c]["q"][kind].append(key)
                    w["qslot"] = self.cores[c]["q"][kind].index(key)
                # (K2b) the kind's units stably sorted by (tail, key index) (the Builder's insertion sort)
                units = sorted(units, key=lambda w: (w["tail"], w["qslot"]))
                self.cores[c]["units"] += units
                self.cores[c]["parts"][kind] = parts[c]
                if parts[c] > NP:  # (K2b step 3) the global parts check
                    self.ok = False

    def bounds(self):
        return dict(
            units=max(len(c["units"]) for c in self.cores),
            roots=max(len(c["roots"]) for c in self.cores),
            nodes=max(len(c["nodes"]) for c in self.cores),
            q_s=max(len(c["q"][0]) for c in self.cores),
            q_f=max(len(c["q"][1]) for c in self.cores),
            parts=max(max(c["parts"]) for c in self.cores),
            tiles_s=max(sum(w["n"] for w in c["units"] if not w["full"]) for c in self.cores),
            tiles_f=max(sum(w["n"] for w in c["units"] if w["full"]) for c in self.cores),
            pieces=max(sum(1 for w in c["units"] if w.get("piece")) for c in self.cores),  # (m5_14 F1)
            k_s=max([r["k"] for c in self.cores for r in c["roots"] if not r["full"]] or [0]),
            k_f=max([r["k"] for c in self.cores for r in c["roots"] if r["full"]] or [0]),
            ok=self.ok,
        )


def scenarios(M, capacity, rng, n=300):
    U = R // M
    out = []
    for p in (0, 31, 32 - M, 1022, 1023, 1024, 1150, 2047, 4096, capacity - M):
        out.append([p] * U)  # everyone at one position (the most skewed schedule)
    # (m5_14 F1) everyone straddling one chunk boundary (sliding 256 c, full 2,048 c): every user has a piece
    for b in (256, 1280, 1536, 2048, 2304, 16384, capacity - 2048):
        for o in range(1, M):
            if 0 <= b - o and b - o + M <= capacity:
                out.append([b - o] * U)
    for _ in range(n):
        mode = rng.random()
        if mode < 0.3:
            ps = [rng.randrange(0, capacity - M + 1) for _ in range(U)]
        elif mode < 0.6:
            base = rng.randrange(0, capacity - M + 1)
            ps = [min(capacity - M, max(0, base + rng.randrange(-64, 64))) for _ in range(U)]
        else:
            ps = [rng.choice((rng.randrange(0, 64), rng.randrange(capacity - 600, capacity - M + 1))) for _ in range(U)]
        dead = rng.random() < 0.2
        out.append([None if dead and rng.random() < 0.3 else p for p in ps])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bounds", action="store_true")
    ap.add_argument("--capacity", type=int, default=MAX_CHUNKS_F * CHUNK_F * 32)
    a = ap.parse_args()
    rng = random.Random(7)
    for M in (1, 2, 4, 8):
        worst = {}
        for ps in scenarios(M, a.capacity, rng):
            b = Schedule(ps, M).bounds()
            for key, v in b.items():
                worst[key] = (worst.get(key, True) and v) if key == "ok" else max(worst.get(key, 0), v)
        print(f"M={M} pt_s={pt(False, M)} pt_f={pt(True, M)} pack_s={pack(False, M)} pack_f={pack(True, M)} worst={worst}")


if __name__ == "__main__":
    main()

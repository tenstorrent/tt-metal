# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""M6b integration GI0: host replica of the d_k2a unit cores' schedule in drafter mode (kernels/d_k2a_common.hpp
Builder with kDraftW > 0), copied from resident_m5_3/k2a_sched.py (frozen) with the drafter deltas:
  - window: every row of a user block at p .. p + M - 1 sees positions [p - W, p + M - 1] (W = the drafter's
    sliding_window, 2048; tt/dflash2.py:117 + tt/dflash.py _mask), non-causal in the block;
  - sliding layers only; roots per core per kv head <= 2; partials per root k <= ceil(W / 256) + 1;
  - K / V tiles from the slot-major ring (72 position tiles per (slot, layer, kv head)), no page tables.
Checks (pure python, no device):
  bounds   units / roots / Q slots / parts / k / nW / nT within the kernels' records and CBs, over every
           U = 32 / M (M in 2, 4, 8) x position mix, at capacity 10240 and 262144;
  keys     per (user, kv head): the union of the user's units' tiles minus build_mask's hidden positions equals
           ttnn _mask's valid set (the ring emulated with tt/dflash.py ring_slot), every tile read exactly once, and
           no two read tiles alias one ring tile (review finding 1: p < W, p - W off the chunk grid, the ring wrap).

  python -m models.demos.gemma4_31b_qb2.resident_m7.d_k2a_sched [--json out.json]
"""
import argparse
import ast
import json
import random
from pathlib import Path

R = 32
DTS = 8
CHUNK_S = 8
NVS = 9
PTS = DTS + 2  # partial pages: O Dt, m, l
MAX_UNITS, MAX_ROOTS, MAX_Q, MAX_TAILS = 20, 4, 4, 8
RING_TILES = 72  # drafter ring position tiles per (slot, layer, kv head) (drafter_common.RING_PAGES)
TT = Path(__file__).resolve().parents[1] / "tt"


def kmax(W):
    return -(-W // (32 * CHUNK_S)) + 1


def root_cap(W):
    return 2 if W else 3


def pack_s(M):
    """(QH, Tu, Ht, k, tiles) of the sliding packed Q layout (d_k2a_common.hpp Pack<false>)"""
    QH = 2
    Tu = (QH * M + 31) // 32
    Ht = QH // Tu
    k = 32 // (Ht * M)
    return QH, Tu, Ht, k, ((R // M) // k) * Tu


def window(p, M, W):
    """(tlo, wl): Builder::window, sliding; W = 0 is K2a's per-row 1024 window"""
    if W:
        tlo = (p - W) // 32 if p > W else 0
        wl = 1 if (p > W and (p - W) % 32) else 0
        return tlo, wl
    tlo = (p - 1023) // 32 if p >= 1023 else 0
    wl = (p + M - 1 - 1023) // 32 - tlo + 1 if p + M - 1 >= 1023 else 0
    return tlo, wl


def hidden(pos, p, m, M, W):
    """build_mask<false>'s rule for one of the user's own rows (m: the row in the block)"""
    if W:
        return pos >= p + M or pos + W < p
    return pos > p + m or pos + 1023 < p + m


class Schedule:
    """the Builder's assignment (sliding only): per core its unit / root / Q-slot records"""

    def __init__(self, positions, M, W, num_units=60, per_head=15):
        self.M, self.U, self.W = M, R // M, W
        self.positions = positions
        self.cores = [dict(units=[], roots=[], q=[], parts=0, tails=0) for _ in range(num_units)]
        self.ok, self.why = True, []
        load = [0] * num_units
        nroot, parts = [0] * num_units, [0] * num_units
        tails = {c: [] for c in range(num_units)}
        nonroot = {c: [] for c in range(num_units)}
        cap, K = root_cap(W), kmax(W) if W else 5
        for g in range(4):
            c0, cn = g * per_head, per_head
            for u in range(self.U):
                p = positions[u]
                if p is None:
                    continue
                tt0, tend = p // 32, (p + M - 1) // 32
                tlo, wl = window(p, M, W)
                clo, ctl = tlo // CHUNK_S, tt0 // CHUNK_S
                k = ctl - clo + 1
                if k > K:
                    self.ok = False
                    self.why.append(("k", u, p, k))
                cands = [c for c in range(c0, c0 + cn) if nroot[c] < cap]
                if not cands:
                    self.ok = False
                    self.why.append(("root", u, g))
                    continue
                root = min(cands, key=lambda c: (load[c], c))
                pbase = parts[root]
                parts[root] += k * PTS
                nroot[root] += 1
                a0 = max(tlo, ctl * CHUNK_S)
                load[root] += tend - a0 + 1 + 4
                wend = tlo + wl
                n = tend - a0 + 1
                nw = min(wend - a0, n) if a0 < wend else 0
                rec = dict(user=u, g=g, tail=True, nW=nw, nT=tend - tt0 + 1, t0=a0, n=n, root=root, pslot=ctl - clo,
                           pbase=pbase, k=k, psem=nroot[root] - 1)
                self.cores[root]["roots"].append(dict(user=u, g=g, k=k, pbase=pbase))
                tails[root].append(rec)
                for cc in range(clo, ctl):
                    b = min(range(c0, c0 + cn), key=lambda c: (load[c], c))
                    s0 = max(tlo, cc * CHUNK_S)
                    n = (cc + 1) * CHUNK_S - s0
                    load[b] += n
                    nw = min(wend - s0, n) if s0 < wend else 0
                    nonroot[b].append(dict(user=u, g=g, tail=False, nW=nw, nT=0, t0=s0, n=n, root=root,
                                           pslot=cc - clo, pbase=pbase))
        kq = pack_s(M)[3]
        for c in range(num_units):
            units = nonroot[c] + tails[c]
            for w in units:
                key = (w["user"] // kq, w["g"])
                if key not in self.cores[c]["q"]:
                    self.cores[c]["q"].append(key)
                w["qslot"] = self.cores[c]["q"].index(key)
            self.cores[c]["units"] = units
            self.cores[c]["parts"] = parts[c]
            self.cores[c]["tails"] = len(tails[c])

    def bounds(self):
        us = [w for c in self.cores for w in c["units"]]
        return dict(
            units=max(len(c["units"]) for c in self.cores),
            roots=max(len(c["roots"]) for c in self.cores),
            tails=max(c["tails"] for c in self.cores),
            q=max(len(c["q"]) for c in self.cores),
            parts=max(c["parts"] for c in self.cores),
            k=max([r["k"] for c in self.cores for r in c["roots"]] or [0]),
            unit_tiles=max([w["n"] for w in us] or [0]),
            nW=max([w["nW"] for w in us] or [0]),
            nT=max([w["nT"] for w in us] or [0]),
            pslot=max([w["pslot"] for w in us] or [0]),
            t0=max([w["t0"] for w in us] or [0]),
            ok=self.ok,
        )

    def visible(self, u, g):
        """per position tile: the 32-bit set of positions user u's own rows see through its kv head g units, and the
        number of units reading the tile"""
        p, M, W = self.positions[u], self.M, self.W
        if not hasattr(self, "_by"):
            self._by = {}
            for c in self.cores:
                for w in c["units"]:
                    self._by.setdefault((w["user"], w["g"]), []).append(w)
        bits, reads = {}, {}
        for w in self._by.get((u, g), []):
            for i in range(w["n"]):
                t = w["t0"] + i
                reads[t] = reads.get(t, 0) + 1
                masked = i < w["nW"] or i + w["nT"] >= w["n"]
                if not masked:
                    bits[t] = bits.get(t, 0) | 0xFFFFFFFF
                    continue
                b = 0
                for e in range(32):
                    pos = 32 * t + e
                    # the kernel's mask is per row; drafter rows must all agree (row-invariant window)
                    hid = [hidden(pos, p, m, M, W) for m in range(M)]
                    assert len(set(hid)) == 1, ("row-dependent mask", u, p, pos)
                    b |= (0 if hid[0] else 1) << e
                bits[t] = bits.get(t, 0) | b
        return bits, reads


def ring_slot_fn():
    """tt/dflash.py ring_slot, taken from the source (no ttnn import)"""
    src = (TT / "dflash.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "ring_slot":
            ns = {}
            exec(compile(ast.Module([node], []), "dflash.py", "exec"), ns)
            return ns["ring_slot"]
    raise AssertionError("ring_slot not found")


def drafter_window():
    """DFlash2's anchor-relative window: tl["sliding_window"] (dflash2.py), the window _mask applies (dflash.py)"""
    d2, d1 = (TT / "dflash2.py").read_text(), (TT / "dflash.py").read_text()
    assert 'self.window = tl["sliding_window"]' in d2, "dflash2.py window source changed"
    assert "valid &= self.slot_pos >= position - self.window" in d1, "dflash.py _mask source changed"
    from models.demos.gemma4_31b_qb2.resident_m7.drafter_common import drafter_config

    cfg = drafter_config()
    tl = cfg.get("transformer_layer_config", cfg)
    return int(tl["sliding_window"])


class TtnnRing:
    """tt/dflash.py's ring: slot_pos after every position < p + M was written (committed context + the block)"""

    def __init__(self, ring_slot, ring=2304):
        self.ring_slot, self.ring = ring_slot, ring
        self.slot_of = {}

    def valid_bits(self, p, M, W):
        """per position tile: _mask's valid set, as 32-bit sets"""
        top = p + M - 1
        slot_pos = {}
        # only the last `ring` positions can still be held; older ones are overwritten before them in write order
        for q in range(max(0, top - 2 * self.ring), top + 1):
            slot_pos[self.ring_slot(q, self.ring)] = q
        bits = {}
        for q in slot_pos.values():
            if q >= p - W:
                bits[q // 32] = bits.get(q // 32, 0) | 1 << (q % 32)
        return bits


def scenarios(M, capacity, W, rng, n=200):
    U = R // M
    edge = [0, 1, 31, 32 - M, 33 - M, 255, 256, 257, W - M - 1, W - M, W - 1, W, W + 1, W + 31, W + 32, W + 33,
            W + 255, W + 256, 2303, 2304, 2305, 2304 + W, 4607, 4608, 4609, 9000, capacity - M]
    edge = sorted({min(max(0, e), capacity - M) for e in edge})
    out = [[e] * U for e in edge]  # everyone at one position (the most skewed schedule)
    out.append([edge[i % len(edge)] for i in range(U)])
    for i in range(len(edge)):
        out.append([edge[(i + j) % len(edge)] for j in range(U)])
    for _ in range(n):
        mode = rng.random()
        if mode < 0.3:
            ps = [rng.randrange(0, capacity - M + 1) for _ in range(U)]
        elif mode < 0.6:
            base = rng.randrange(0, capacity - M + 1)
            ps = [min(capacity - M, max(0, base + rng.randrange(-64, 64))) for _ in range(U)]
        else:
            ps = [rng.choice((rng.randrange(0, 64), rng.randrange(capacity - 600, capacity - M + 1)))
                  for _ in range(U)]
        dead = rng.random() < 0.2
        out.append([None if dead and rng.random() < 0.3 else p for p in ps])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    W = drafter_window()
    ring_slot = ring_slot_fn()
    ttnn_ring = TtnnRing(ring_slot)
    rng = random.Random(7)
    result = dict(W=W, kmax=kmax(W), root_cap=root_cap(W), runs={})
    all_ok = True
    key_checks = 0
    for capacity in (10240, 262144):
        for M in (2, 4, 8):
            worst, fails = {}, []
            for ps in scenarios(M, capacity, W, rng, a.n):
                s = Schedule(ps, M, W)
                b = s.bounds()
                for key, v in b.items():
                    worst[key] = (worst.get(key, True) and v) if key == "ok" else max(worst.get(key, 0), v)
                if not b["ok"]:
                    fails.append(("schedule", ps[:4], s.why[:3]))
                for u, p in enumerate(ps):
                    if p is None:
                        continue
                    want = ttnn_ring.valid_bits(p, M, W)
                    for g in range(4):
                        got, reads = s.visible(u, g)
                        key_checks += 1
                        tiles = sorted(reads)
                        if any(r != 1 for r in reads.values()):
                            fails.append(("double read", M, p, g))
                        if tiles and tiles[-1] - tiles[0] >= RING_TILES:
                            fails.append(("ring alias", M, p, tiles[0], tiles[-1]))
                        if {t: b for t, b in got.items() if b} != want:
                            fails.append(("keys", M, p, g, len(got), len(want)))
            lim = dict(units=MAX_UNITS, roots=MAX_ROOTS, tails=MAX_TAILS, q=MAX_Q, parts=root_cap(W) * kmax(W) * PTS,
                       k=kmax(W), unit_tiles=NVS, nW=15, nT=15, pslot=255, t0=65535)
            within = all(worst[k] <= v for k, v in lim.items()) and worst["ok"]
            ok = within and not fails
            all_ok &= ok
            result["runs"][f"cap{capacity}_M{M}"] = dict(worst=worst, limits=lim, fails=fails[:10], nfails=len(fails),
                                                         ok=ok)
            print(f"cap={capacity} M={M} U={R // M} worst={worst} fails={len(fails)} {'OK' if ok else 'FAIL'}")
    result["key_checks"] = key_checks
    result["np"] = root_cap(W) * kmax(W) * PTS
    result["ok"] = all_ok
    print(f"GI0 W={W} kmax={kmax(W)} root_cap={root_cap(W)} NP={result['np']} key_checks={key_checks} "
          f"{'PASS' if all_ok else 'FAIL'}")
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU model of the MoE exchange with the dispatch inside the experts' program (MoEOptions.experts_during_dispatch;
sparse_all_gather_2d.dispatch_phase + sparse_combine_rf's dispatch hook + the flat op's x_ready).

Each port core runs, on one fabric connection a link, the sparse 2D dispatch (the plan-free protocol of
sparse_all_gather_2d: dimension-order trees, END(g) markers, rings that hold a whole call - no credits), then the
combine. A port's reader is done with the dispatch once END(GMAX) is out and every input delivered END(GMAX); it then
counts in on its chip's x relays (+1; the chip's first port also the pad to XREADY_N). The x relays read x only once
the count reaches XREADY_N. The sender phase sends the combine's rows after the dispatch's, on the same link.

Link model: everything a port sends to its neighbour - rows (landing in the neighbour's DRAM slots), stream entries
for the neighbour's ports, then combine rows - goes through one FIFO a directed link, delivered in order at random
times (a row lands before the entries behind it, as a fabric link keeps a sender's order).

Checks under random interleavings, line and ring (torus) meshes, several routings: no deadlock, every needed row lands,
every stream is consumed, each chip's count reaches exactly XREADY_N, and when a chip's relays pass the gate every row
that chip needs has landed (the x-ready gate is sound); every combine row on a link follows that link's dispatch."""

import random

import pytest

DIRS = ("X+", "X-", "Y+", "Y-")
OPP = {"X+": "X-", "X-": "X+", "Y+": "Y-", "Y-": "Y+"}
XREADY_N = 8  # (sparse_all_gather_2d.XREADY_N)
GATE = XREADY_N  # (the relays' threshold; the early-gate test lowers it)


def hbit(t, k=31):
    return ((t * 2654435761) & 0xFFFFFFFF) >> k & 1


class Mesh:
    """R x C, each axis a ring (wrap) or a line; L links (the stateless route of sparse_all_gather_2d)."""

    def __init__(self, R, C, wrap_x, wrap_y, L):
        self.R, self.C, self.L = R, C, L
        self.wx = wrap_x and C > 2
        self.wy = wrap_y and R > 2
        self.GMAX = (C - 1) + (R - 1) - 1

    def nb(self, chip, d):
        r, c = chip
        if d[0] == "X":
            c2 = c + (1 if d[1] == "+" else -1)
            if self.wx:
                c2 %= self.C
            return (r, c2) if 0 <= c2 < self.C else None
        r2 = r + (1 if d[1] == "+" else -1)
        if self.wy:
            r2 %= self.R
        return (r2, c) if 0 <= r2 < self.R else None

    def span(self, n, wrap, pos, targets, t):
        offs = {(x - pos) % n if wrap else x - pos for x in targets if x != pos}
        if not offs:
            return 0, 0
        if not wrap:
            return max([o for o in offs if o > 0], default=0), max([-o for o in offs if o < 0], default=0)
        cands = [(a, b) for a in range(n) for b in range(n - a) if all(o <= a or n - o <= b for o in offs)]
        m = min(a + b for a, b in cands)
        cands = [c for c in cands if c[0] + c[1] == m]
        mm = min(max(c) for c in cands)
        cands = sorted(c for c in cands if max(c) == mm)
        return cands[hbit(t, 30) if len(cands) > 1 else 0]

    def tree(self, origin, need, t, order):
        """{chip: [(dir, hops left, hops travelled before leaving)]}."""
        r0, c0 = origin
        out = {}
        if order == "XY":
            a, b = self.span(self.C, self.wx, c0, {c for _, c in need}, t)
            first = [(0, origin, None)]
            first += [(i, (r0, (c0 + i) % self.C), "X+") for i in range(1, a + 1)]
            first += [(i, (r0, (c0 - i) % self.C), "X-") for i in range(1, b + 1)]
            for i, chip, d in first:
                sends = []
                if i == 0:
                    sends += [("X+", a, 0)] if a else []
                    sends += [("X-", b, 0)] if b else []
                elif d == "X+" and i < a:
                    sends.append(("X+", a - i, i))
                elif d == "X-" and i < b:
                    sends.append(("X-", b - i, i))
                a2, b2 = self.span(self.R, self.wy, r0, {r for r, c in need if c == chip[1]}, t)
                sends += [("Y+", a2, i)] if a2 else []
                sends += [("Y-", b2, i)] if b2 else []
                out.setdefault(chip, []).extend(sends)
                for j in range(1, a2):
                    out.setdefault(((r0 + j) % self.R, chip[1]), []).append(("Y+", a2 - j, i + j))
                for j in range(1, b2):
                    out.setdefault(((r0 - j) % self.R, chip[1]), []).append(("Y-", b2 - j, i + j))
            return out
        tm = Mesh(self.C, self.R, self.wy, self.wx, self.L)
        sw = lambda ch: (ch[1], ch[0])  # noqa: E731
        flip = {"X+": "Y+", "X-": "Y-", "Y+": "X+", "Y-": "X-"}
        res = tm.tree(sw(origin), {sw(x) for x in need}, t, "XY")
        return {sw(ch): [(flip[d], h, g) for d, h, g in s] for ch, s in res.items()}

    def route(self, origin, need, t):
        need = set(need) - {origin}
        if not need:
            return {}
        trees = {o: self.tree(origin, need, t, o) for o in ("XY", "YX")}
        hops = {o: sum(len(s) for s in tr.values()) for o, tr in trees.items()}
        if hops["XY"] != hops["YX"]:
            return trees[min(hops, key=hops.get)]
        return trees["XY" if hbit(t) else "YX"]


def _run(mesh, S, E, K, seed):
    rnd = random.Random(seed)
    chips = [(r, c) for r in range(mesh.R) for c in range(mesh.C)]
    N = len(chips)
    epc = E // N
    needs = {(o, t): {chips[e // epc] for e in rnd.sample(range(E), K)} for o in chips for t in range(S)}
    routes = {(o, t): mesh.route(o, needs[(o, t)], t) for o in chips for t in range(S)}
    link = {(o, t): hbit(t, 29) if mesh.L == 2 else 0 for o in chips for t in range(S)}
    ports = [(ch, d, l) for ch in chips for d in DIRS for l in range(mesh.L) if mesh.nb(ch, d) is not None]

    def inputs_of(p):
        ch, d, l = p
        ins = []
        for dd in DIRS:
            up = mesh.nb(ch, OPP[dd])
            if up is None or up == ch or dd == OPP[d]:
                continue
            ins.append((up, dd, l))
        return ins

    inputs = {p: inputs_of(p) for p in ports}
    fifo = {p: [] for p in ports}  # a port's link: ("ROW", row) | ("ENT", target port, entry) | ("CMB", n)
    vis = {p: 0 for p in ports}  # entries delivered (in order)
    ring = {(u, p): [] for p in ports for u in inputs[p]}  # delivered stream entries at port p from u
    rd = {k: 0 for k in ring}
    own = {p: [] for p in ports}
    for o in chips:
        for t in range(S):
            for d, h, g in routes[(o, t)].get(o, []):
                own[(o, d, link[(o, t)])].append((o, t, h))
    st = {p: dict(own_i=0, end_out=-1, end_in={u: -1 for u in inputs[p]}, rdone=False, sdone=False) for p in ports}
    landed = {ch: set() for ch in chips}
    counts = {ch: 0 for ch in chips}
    first_port = {ch: min(p for p in ports if p[0] == ch) for ch in chips}
    relay_ok = {ch: False for ch in chips}
    cmb = {p: 0 for p in ports}  # combine rows sent

    def send(p, row, g):
        ch, d, l = p
        nxt = mesh.nb(ch, d)
        fifo[p].append(("ROW", row, nxt))
        o, t = row
        for d2, h2, g2 in routes[(o, t)].get(nxt, []):
            tgt = (nxt, d2, l)
            assert p in inputs[tgt] and g2 == g + 1, (row, p, tgt)
            fifo[p].append(("ENT", tgt, ("ROW", row, g2)))

    def deliver(p, n):
        for e in fifo[p][vis[p] : vis[p] + n]:
            if e[0] == "ROW":
                landed[e[2]].add(e[1])
            elif e[0] == "ENT":
                ring[(p, e[1])].append(e[2])
        vis[p] = min(len(fifo[p]), vis[p] + n)

    def step(p):
        s = st[p]
        ch = p[0]
        if not s["rdone"]:  # the dispatch (reader + sender of the port, as one actor: the sender sends what it reads)
            for u in inputs[p]:
                k = (u, p)
                if rd[k] < len(ring[k]):
                    e = ring[k][rd[k]]
                    rd[k] += 1
                    if e[0] == "END":
                        assert e[1] == s["end_in"][u] + 1
                        s["end_in"][u] = e[1]
                    else:
                        assert e[2] > s["end_out"], "a row behind its END"
                        send(p, e[1], e[2])
                    return True
            if s["own_i"] < len(own[p]):
                o, t, _ = own[p][s["own_i"]]
                s["own_i"] += 1
                send(p, (o, t), 0)
                return True
            g = s["end_out"] + 1
            if g <= mesh.GMAX and (g == 0 or all(v >= g - 1 for v in s["end_in"].values())):
                nxt = mesh.nb(ch, p[1])
                tg = [q for q in ports if q[0] == nxt and p in inputs[q]]
                for q in tg:
                    fifo[p].append(("ENT", q, ("END", g)))
                s["end_out"] = g
                return True
            if g > mesh.GMAX and all(v >= mesh.GMAX for v in s["end_in"].values()):
                s["rdone"] = s["sdone"] = True
                counts[ch] += 1 + (XREADY_N - sum(1 for q in ports if q[0] == ch) if p == first_port[ch] else 0)
                return True
            return False
        if cmb[p] < 3:  # the combine's rows, after the dispatch on this link
            fifo[p].append(("CMB", cmb[p]))
            cmb[p] += 1
            return True
        return False

    def relays(ch):
        if relay_ok[ch] or counts[ch] < GATE:
            return False
        miss = [(o, t) for o in chips for t in range(S) if ch in needs[(o, t)] and o != ch and (o, t) not in landed[ch]]
        assert not miss, f"x-ready on {ch} before rows landed: {miss[:3]}"
        relay_ok[ch] = True
        return True

    while True:
        progress = False
        order = ports[:]
        rnd.shuffle(order)
        for p in order:
            for _ in range(rnd.randint(0, 3)):
                progress |= step(p)
        for p in ports:
            if vis[p] < len(fifo[p]) and rnd.random() < 0.6:
                deliver(p, rnd.randint(1, 4))
                progress = True
        for ch in chips:
            progress |= relays(ch)
        if all(s["sdone"] and cmb[p] == 3 for p, s in st.items()) and all(vis[p] == len(fifo[p]) for p in ports):
            break
        if not progress:  # (a random sweep may do nothing: deliver everything and step every port)
            pend = any(vis[p] < len(fifo[p]) for p in ports)
            for p in ports:
                deliver(p, len(fifo[p]))
            progress = pend or any([step(p) for p in ports]) or any([relays(ch) for ch in chips])
        assert progress, f"deadlock: {[p for p, s in st.items() if not s['sdone']][:3]}"
    for ch in chips:
        relays(ch)
        assert relay_ok[ch] and counts[ch] == XREADY_N
    for k in ring:
        assert rd[k] == len(ring[k]), f"unread entries {k}"
    for (o, t), need in needs.items():
        for ch in need - {o}:
            assert (o, t) in landed[ch]
    for p in ports:  # (one sender a link: the combine's rows after every dispatch item)
        kinds = [e[0] for e in fifo[p]]
        assert "CMB" not in kinds or all(k == "CMB" for k in kinds[kinds.index("CMB") :])


CASES = [
    # (R, C, wrap X, wrap Y, links, S, E, K)
    (2, 4, False, False, 2, 24, 64, 8),  # the LoudBox (lines)
    (2, 4, False, False, 2, 24, 224, 16),
    (4, 4, True, False, 2, 16, 128, 8),  # TP ring, SP line
    (8, 4, True, True, 2, 8, 256, 8),  # the Galaxy torus
    (8, 4, True, True, 1, 8, 896, 16),
]


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}x{c[1]}-w{int(c[2])}{int(c[3])}-L{c[4]}-K{c[7]}" for c in CASES])
@pytest.mark.parametrize("seed", range(3))
def test_dispatch_then_combine_phases(case, seed):
    R, C, wx, wy, L, S, E, K = case
    _run(Mesh(R, C, wx, wy, L), S, E, K, seed)


def test_model_catches_an_early_gate(expect_error):
    """The model is not vacuous: counting in before the inputs ended must trip the x-ready check."""
    mesh = Mesh(2, 4, False, False, 2)
    global GATE
    GATE = 1  # (the first port's count opens the gate)
    try:
        with expect_error(AssertionError, "x-ready"):
            for seed in range(5):
                _run(mesh, 24, 64, 8, seed)
    finally:
        GATE = XREADY_N

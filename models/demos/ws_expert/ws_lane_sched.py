# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Static schedules for the lane design (ws_lane): one group's experts travel its own lane (one Hamiltonian cycle and
link, both directions), owned whole, forwarded cut-through by the group cores themselves into a weight slot of the next
chip's same core. Every group core has K typed weight slots (the same index on all 20 cores of a group), by default 3:
slot 0 takes experts arriving on direction 0, slot 1 those arriving on direction 1, slot 2 any (and the chip's own
experts). Each expert's down projection is drained before the next expert starts, so a slot frees once its expert is
computed and forwarded.

lane_sched() runs a lock-step model of one lane (packet steps: the 20 cores move one packet each over the link per
step; P packets a slice, compute C steps, starting after the first GU packets) with a greedy policy - passing experts
take any allowed free slot of the next chip, an owner injects its next expert only while `keep` slots stay free after
it and no passing expert waits for its slots - and records the schedule: per chip and slot the experts it holds in
order, per chip the compute order. The device follows only these orders: a core sends occupant k of a slot into the
next chip's slot s' once that slot has freed its previous occupant (credit), computes in the recorded order, and frees
a slot after its compute and forwarding. Every such wait is on an event earlier in the model's time, so no cycle of
waits can form (replay() checks it with random step costs)."""

import random
from collections import deque

ANY = frozenset({"d0", "d1", "own"})
TYPES = (frozenset({"d0"}), frozenset({"d1"}), ANY)


def slot_types(spec):
    """Slot types from a spec such as "d0,d1,d0d1,any": per slot the experts it takes (arriving on direction 0 /
    direction 1 / either, or any including the chip's own)."""
    names = {"d0": frozenset({"d0"}), "d1": frozenset({"d1"}), "d0d1": frozenset({"d0", "d1"}), "any": ANY}
    return tuple(names[x] for x in spec.split(","))


def lane_plan(N, E, spread=7, oneway=False):
    """Owner position on the cycle and hops on direction 0 (kf) for lane-local expert i; direction 1 covers the rest.
    oneway: every expert all N - 1 hops on direction 0 (direction 1 carries only credits)."""
    owners = [(i * spread + i // N) % N for i in range(E)]
    kf = [N - 1] * E if oneway else [N // 2 if i % 2 else (N - 1) // 2 for i in range(E)]
    return owners, kf


class Lane:
    def __init__(self, N, E, P=15, C=9, GU=10, keep=1, sched=None, prob=1.0, seed=0, types=TYPES, oneway=False):
        self.N, self.E, self.P, self.C, self.GU, self.keep = N, E, P, C, GU, keep
        self.K = len(types)
        self.types = types
        self.owners, self.kf = lane_plan(N, E, oneway=oneway)
        self.sched, self.prob, self.rng = sched, prob, random.Random(seed)
        self.nxt = lambda x, d: (x + 1) % N if d == 0 else (x - 1) % N
        self.route = {}  # (i, position) -> (direction it arrives on, hops left after it)
        for i in range(E):
            o = self.owners[i]
            for d, n in ((0, self.kf[i]), (1, N - 1 - self.kf[i])):
                for h in range(1, n + 1):
                    self.route[(i, (o + (h if d == 0 else -h)) % N)] = (d, n - h)
        self.own_q = [deque(i for i in range(E) if self.owners[i] == x) for x in range(N)]
        self.slot = {(x, s): None for x in range(N) for s in range(self.K)}
        self.held = {(x, s): [] for x in range(N) for s in range(self.K)}
        self.order = {x: [] for x in range(N)}
        self.comp = {x: dict(end=-1, cur=None) for x in range(N)}
        self.total, self.done, self.t, self.busy = N * E, 0, 0, 0

    def go(self):
        return self.prob >= 1.0 or self.rng.random() < self.prob

    def nexp_for(self, x, s):
        q = self.sched["held"][(x, s)]
        k = len(self.held[(x, s)])
        return q[k] if k < len(q) else None

    def fwd_dirs(self, i, x):
        if self.owners[i] == x:
            return [d for d, n in ((0, self.kf[i]), (1, self.N - 1 - self.kf[i])) if n > 0]
        d, rest = self.route[(i, x)]
        return [d] if rest > 0 else []

    def new(self, x, s, i):
        self.held[(x, s)].append(i)
        sl = dict(i=i, arr=0, fwd={d: 0 for d in self.fwd_dirs(i, x)}, computed=False, started=False, since=self.t)
        self.slot[(x, s)] = sl
        return sl

    def find(self, x, i):
        for s in range(self.K):
            sl = self.slot[(x, s)]
            if sl is not None and sl["i"] == i:
                return sl
        return None

    def step(self):
        N, K, P = self.N, self.K, self.P
        moved, wanting = False, set()
        for x in range(N):
            for d in (0, 1):
                y = self.nxt(x, d)
                cand = [
                    self.slot[(x, s)]
                    for s in range(K)
                    if self.slot[(x, s)] and d in self.slot[(x, s)]["fwd"] and self.slot[(x, s)]["fwd"][d] < P
                ]
                if not cand:
                    continue
                if self.sched is not None:
                    exp = {self.nexp_for(y, s) for s in range(K) if self.slot[(y, s)] is None}
                    exp |= {
                        self.slot[(y, s)]["i"] for s in range(K) if self.slot[(y, s)] and self.slot[(y, s)]["arr"] < P
                    }
                    cand = [c for c in cand if c["i"] in exp]
                    if not cand:
                        continue
                sl = min(cand, key=lambda c: c["since"])
                i = sl["i"]
                tgt = self.find(y, i)
                if tgt is None:
                    if self.sched is not None:
                        st = [s for s in range(K) if self.slot[(y, s)] is None and self.nexp_for(y, s) == i]
                    else:
                        st = [s for s in range(K) if self.slot[(y, s)] is None and f"d{d}" in self.types[s]]
                        wanting.add(y)
                    if not st or not self.go():
                        continue
                    tgt = self.new(y, st[0], i)
                u = sl["fwd"][d]
                if u < sl["arr"] and tgt["arr"] == u and self.go():
                    sl["fwd"][d] += 1
                    tgt["arr"] += 1
                    moved = True
                    self.busy += 1
        for x in range(N):
            if not self.own_q[x]:
                continue
            i = self.own_q[x][0]
            if self.sched is not None:
                st = [s for s in range(K) if self.slot[(x, s)] is None and self.nexp_for(x, s) == i]
            else:
                fr = [s for s in range(K) if self.slot[(x, s)] is None]
                st = [s for s in fr if "own" in self.types[s]] if (len(fr) > self.keep and x not in wanting) else []
            if not st or not self.go():
                continue
            self.own_q[x].popleft()
            self.new(x, st[0], i)["arr"] = P
            moved = True
        for x in range(N):
            c = self.comp[x]
            if self.t < c["end"]:
                continue
            if c["cur"] is not None:
                c["cur"]["computed"] = True
                c["cur"] = None
                self.done += 1
                moved = True
            cands = [self.slot[(x, s)] for s in range(K) if self.slot[(x, s)]]
            if self.sched is not None:
                k = len(self.order[x])
                if k >= len(self.sched["order"][x]):
                    continue
                cands = [c_ for c_ in cands if c_["i"] == self.sched["order"][x][k]]
            cands = [c_ for c_ in cands if not c_["started"] and c_["arr"] >= self.GU]
            if not cands or not self.go():
                continue
            if self.sched is None:
                key = lambda c_: (any(v < P for v in c_["fwd"].values()), c_["since"])
            else:
                key = lambda c_: c_["since"]
            sl = min(cands, key=key)
            sl["started"] = True
            self.order[x].append(sl["i"])
            c["cur"], c["end"] = sl, self.t + self.C
            moved = True
        for key_, sl in self.slot.items():
            if sl and sl["computed"] and sl["arr"] == P and all(v == P for v in sl["fwd"].values()):
                self.slot[key_] = None
                moved = True
        self.t += 1
        return moved

    def run(self, limit=10**7, stall=50000):
        idle = 0
        while self.done < self.total and self.t < limit:
            if self.step():
                idle = 0
            else:
                idle += 1
                if idle > stall:
                    return False
        return self.done >= self.total


def lane_sched(N, E, **kw):
    """The static schedule of one lane: dict(held={(position, slot): [i, ...]}, order={position: [i, ...]}, owners,
    kf, steps, util (compute share of the model's time))."""
    m = Lane(N, E, **kw)
    assert m.run(stall=5000), "lane model stuck"
    return dict(
        held={k: list(v) for k, v in m.held.items()},
        order={k: list(v) for k, v in m.order.items()},
        owners=m.owners,
        kf=m.kf,
        steps=m.t,
        util=E * 7 / m.t,
    )


def replay(N, E, sch, seed=0, prob=0.6, **kw):
    """Run a schedule with random step costs; True if it completes (no deadlock)."""
    r = Lane(N, E, sched=sch, prob=prob, seed=seed, **kw)
    return r.run(stall=200000), E * 7 / (r.t * prob)

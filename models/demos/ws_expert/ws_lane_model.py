"""Expert-granularity model of one group's lane (one Hamiltonian cycle, one link, both directions) on N chips.

Time unit: one packet step = all 20 group cores move one 13.8 KB packet over the link (20 x 13.8 KB at 25 GB/s =
11 us). An expert slice is P packets (15): a whole expert crosses a link in P steps. Compute of one expert at full
speed takes C steps (78 us ~ 7 steps); it can start once the first G packets (gate/up part) are in.

Each group core has 2 weight slots. Slot 0 takes the experts arriving on direction 0 and the chip's own experts,
slot 1 those arriving on direction 1. The chip computes its slots alternately 0, 1, 0, 1 (the cores' 2-slot FIFO), so
a direction with fewer experts is padded with dummies (no rows, no data). An expert finishes only when the next one
starts computing (down-projection lag). A slot frees once its expert is computed and forwarded. Forwarding is
cut-through, one packet a step a link: packet u leaves once it has arrived; the downstream slot must be free (or
already taking this expert). Experts are owned whole; expert e goes kf(e) hops on direction 0 and N - 1 - kf(e) on
direction 1, both sent from the owner's slot 0.

sync(...) runs the policy in lock step and records, per chip and slot, the order in which experts entered it; those
orders are the static schedule. replay(...) executes a schedule with random step costs (each action happens in a
step with probability p) and reports whether it completes."""
import random
from collections import deque


def plan(N, E, spread=7):
    owners = [(e * spread + e // N) % N for e in range(E)]
    kf = [N // 2 if e % 2 else (N - 1) // 2 for e in range(E)]
    return owners, kf


class Lane:
    def __init__(self, N, E, P=15, C=7, G=10, seqs=None, prob=1.0, rng=None, inject="passing_first", lag=True,
                 corder=None, bubble=False):
        self.N, self.E, self.P, self.C, self.G = N, E, P, C, G
        self.owners, self.kf = plan(N, E)
        self.seqs = seqs  # None: free (sync); else {(x, s): [e, ...]} fixed orders
        self.prob, self.rng, self.inject = prob, rng or random.Random(0), inject
        self.lag, self.corder, self.bubble = lag, corder, bubble
        self.order = {x: [] for x in range(N)}  # compute order recorded: (s, nth)
        self.nxt = lambda x, d: (x + 1) % N if d == 0 else (x - 1) % N
        # hops left after chip x, per (e, x): direction, remaining hops
        self.route = {}
        for e in range(E):
            o = self.owners[e]
            for d, n in ((0, self.kf[e]), (1, N - 1 - self.kf[e])):
                for h in range(1, n + 1):
                    self.route[(e, (o + (h if d == 0 else -h)) % N)] = (d, n - h)
        self.own_q = [deque(e for e in range(E) if self.owners[e] == x) for x in range(N)]
        self.cnt = {(x, s): 0 for x in range(N) for s in (0, 1)}  # experts that will enter slot s at chip x
        for (e, x), (d, _) in self.route.items():
            self.cnt[(x, d)] += 1
        for x in range(N):
            self.cnt[(x, 0)] += len(self.own_q[x])
        self.slot = {}  # (x, s) -> dict or None
        for x in range(N):
            for s in (0, 1):
                self.slot[(x, s)] = None
        self.entered = {(x, s): [] for x in range(N) for s in (0, 1)}
        self.pos = {(x, s): 0 for x in range(N) for s in (0, 1)}  # next index in a fixed sequence
        self.comp = {x: dict(k=0, end=-1, started=[], cur=None) for x in range(N)}  # k-th compute entry
        self.done = 0
        self.t = 0

    def want(self, x, s):
        """Next expert allowed into slot s of chip x under a fixed schedule (or None = any)."""
        if self.seqs is None:
            return None
        q = self.seqs[(x, s)]
        return q[self.pos[(x, s)]] if self.pos[(x, s)] < len(q) else -1

    def step(self):
        N, P, C, G = self.N, self.P, self.C, self.G
        rnd = lambda: self.prob >= 1.0 or self.rng.random() < self.prob
        moved = False
        # forwards: for each chip and direction d, the slot expert that continues on d moves one packet
        for x in range(N):
            for d in (0, 1):
                y = self.nxt(x, d)
                cand = []
                for s in (0, 1):
                    sl = self.slot[(x, s)]
                    if sl and d in sl["fwd"] and sl["fwd"][d] < P:
                        cand.append(sl)
                if not cand:
                    continue
                tgt = self.slot[(y, d)]
                if self.seqs is not None:
                    # the link carries experts in the downstream slot's order
                    w = tgt["e"] if tgt is not None else self.want(y, d)
                    cand = [c_ for c_ in cand if c_["e"] == w]
                    if not cand:
                        continue
                sl = min(cand, key=lambda c: c["since"])
                e = sl["e"]
                if tgt is None:
                    w = self.want(y, d)
                    if w is not None and w != e:
                        continue
                    if self.seqs is None and self.inject == "passing_first":
                        pass
                    if not rnd():
                        continue
                    self.slot[(y, d)] = self.new_slot(y, d, e)
                    tgt = self.slot[(y, d)]
                elif tgt["e"] != e:
                    continue
                u = sl["fwd"][d]
                if u < sl["arr"] and tgt["arr"] == u and rnd():
                    sl["fwd"][d] += 1
                    tgt["arr"] += 1
                    moved = True
        # owners inject their next expert into a free slot 0 (passing experts first: none waiting upstream)
        for x in range(N):
            if self.slot[(x, 0)] is None and self.own_q[x]:
                e = self.own_q[x][0]
                w = self.want(x, 0)
                if w is not None and w != e:
                    continue
                if self.seqs is None and self.inject == "passing_first":
                    up = self.nxt(x, 1)  # upstream on direction 0
                    if any(self.slot[(up, s)] and 0 in self.slot[(up, s)]["fwd"] and self.slot[(up, s)]["fwd"][0] < P
                           for s in (0, 1)):
                        continue
                if self.seqs is None and self.bubble:
                    busy = [sum(1 for y in range(N) if self.slot[(y, dd)] and self.slot[(y, dd)]["fwd"]) for dd in (0, 1)]
                    if max(busy) >= N - 2:
                        continue
                if not rnd():
                    continue
                self.own_q[x].popleft()
                sl = self.new_slot(x, 0, e)
                sl["arr"] = P
                self.slot[(x, 0)] = sl
                moved = True
        if self.lag:
            # compute: entry k uses slot k & 1
            for x in range(N):
                c = self.comp[x]
                total = 2 * max(self.cnt[(x, 0)], self.cnt[(x, 1)])
                if c["k"] >= total:
                    continue
                s = c["k"] & 1
                if self.t < c["end"]:
                    continue
                nth = c["k"] // 2  # this slot's nth expert
                if nth >= self.cnt[(x, s)]:  # dummy
                    c["started"].append(None)
                    self.finish_prev(x)
                    c["k"] += 1
                    c["end"] = self.t
                    moved = True
                    continue
                sl = self.slot[(x, s)]
                if sl is None or sl["nth"] != nth or sl["arr"] < G or sl["started"] or not rnd():
                    continue
                sl["started"] = True
                c["started"].append(sl)
                self.finish_prev(x)
                c["k"] += 1
                c["end"] = self.t + C
                moved = True
            # end: the last entry finishes when its compute ends
            for x in range(N):
                c = self.comp[x]
                total = 2 * max(self.cnt[(x, 0)], self.cnt[(x, 1)])
                if c["k"] >= total and c["started"] and c["started"][-1] is not None and self.t >= c["end"]:
                    sl = c["started"][-1]
                    if not sl["computed"] and sl["arr"] == P:
                        sl["computed"] = True
                        moved = True

        else:
            # drained down projection: compute any slot whose expert is complete enough (or the next one of a fixed
            # order); the expert is computed C steps later
            for x in range(N):
                c = self.comp[x]
                if self.t < c["end"]:
                    continue
                if c["cur"] is not None:
                    c["cur"]["computed"] = True
                    c["cur"] = None
                    moved = True
                cand = []
                for s_ in (0, 1):
                    sl = self.slot[(x, s_)]
                    if sl and not sl["started"] and sl["arr"] >= G:
                        cand.append((sl["since"], s_, sl))
                if self.corder is not None:
                    k = len(self.order[x])
                    if k >= len(self.corder[x]):
                        continue
                    ws, wn = self.corder[x][k]
                    cand = [c_ for c_ in cand if c_[1] == ws and c_[2]["nth"] == wn]
                if not cand or not rnd():
                    continue
                _, s_, sl = min(cand)
                sl["started"] = True
                self.order[x].append((s_, sl["nth"]))
                c["cur"] = sl
                c["end"] = self.t + C
                moved = True
        # free slots
        for (x, s), sl in self.slot.items():
            if sl and sl["computed"] and all(v == self.P for v in sl["fwd"].values()) and sl["arr"] == self.P:
                self.slot[(x, s)] = None
                self.done += 1
                moved = True
        self.t += 1
        return moved

    def finish_prev(self, x):
        st = self.comp[x]["started"]
        if len(st) >= 2 and st[-2] is not None:
            st[-2]["computed_pending"] = True
            st[-2]["computed"] = True

    def new_slot(self, x, s, e):
        if self.seqs is not None:
            self.pos[(x, s)] += 1
        nth = len(self.entered[(x, s)])
        self.entered[(x, s)].append(e)
        fwd = {}
        if self.owners[e] == x:
            if self.kf[e] > 0:
                fwd[0] = 0
            if self.N - 1 - self.kf[e] > 0:
                fwd[1] = 0
        else:
            d, rest = self.route[(e, x)]
            if rest > 0:
                fwd[d] = 0
        return dict(e=e, arr=0, fwd=fwd, computed=False, started=False, nth=nth, since=self.t)

    def run(self, limit=200000, stall=2000):
        idle = 0
        total = sum(self.cnt.values())
        while self.done < total and self.t < limit:
            if self.step():
                idle = 0
            else:
                idle += 1
                if idle > stall:
                    return False
        return self.done >= total


def sync(N, E, **kw):
    m = Lane(N, E, **kw)
    ok = m.run()
    return ok, m


def main_b(N, E, bubble):
    m = Lane(N, E, lag=False, bubble=bubble)
    ok = m.run()
    print(f"N {N} E {E} drained lag, bubble {bubble}: sync {'done' if ok else 'STUCK'} at t {m.t}; util {E * 7 / m.t:.3f}")
    if ok:
        seqs = {k: list(v) for k, v in m.entered.items()}
        for seed in range(3):
            r = Lane(N, E, seqs=seqs, prob=0.6, rng=random.Random(seed), lag=False, corder=m.order)
            ok2 = r.run(limit=10 ** 6, stall=20000)
            print(f"  replay p 0.6 seed {seed}: {'done' if ok2 else 'DEADLOCK'} at t {r.t} (util {E * 7 / r.t / 0.6:.3f} scaled)")
    return m


if __name__ == "__main__" and len(__import__("sys").argv) > 3:
    import sys
    main_b(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1")
    sys.exit(0)

if __name__ == "__main__":
    import sys
    N, E = int(sys.argv[1]), int(sys.argv[2])
    ok, m = sync(N, E)
    busy = E * 7 / m.t
    print(f"N {N} E {E}: sync {'done' if ok else 'STUCK'} at t {m.t}; util vs compute {busy:.3f}")
    if ok:
        seqs = {k: list(v) for k, v in m.entered.items()}
        for seed in range(3):
            r = Lane(N, E, seqs=seqs, prob=0.6, rng=random.Random(seed))
            ok2 = r.run(limit=10 ** 6, stall=20000)
            print(f"  replay p 0.6 seed {seed}: {'done' if ok2 else 'DEADLOCK'} at t {r.t} (util {E * 7 / r.t / 0.6:.3f} scaled)")

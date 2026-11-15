"""K-slot model of one group's lane (one Hamiltonian cycle, one link, both directions) on N chips: design B.

Time unit: one packet step = the group's 20 cores move one 13.8 KB packet each over the link (11 us at 25 GB/s); an
expert slice is P packets (15), so a whole expert crosses a link in P steps. Compute takes C steps (78 us ~ 7) and
can start once the first G packets (gate/up) are in; each expert's down projection is finished before the next
expert starts (no lag), so a slot frees once its expert is computed and forwarded.

Each group core has K weight slots (same index on all 20 cores). Experts are owned whole and travel kf(e) hops on
direction 0 and N - 1 - kf(e) on direction 1 from the owner, cut-through, one packet a step a link, into a slot of
the next chip. A link carries one expert at a time.

sync() runs a greedy policy in lock step and records the static schedule: per chip and slot the order of experts it
holds, per chip the compute order. Policy: passing experts take any free slot of the next chip; an owner injects its
next expert only when at least `keep` slots stay free after it and no passing expert waits for its slots. replay()
executes a schedule with random step costs: each slot takes only its next expert, each link only the expert its
downstream slot expects, compute follows the chip's order. Every wait of the replay is on an event earlier in the
sync model's time, so a completed sync model gives a deadlock-free schedule; the replay checks it."""
import random
from collections import deque


def plan(N, E, spread=7):
    owners = [(e * spread + e // N) % N for e in range(E)]
    kf = [N // 2 if e % 2 else (N - 1) // 2 for e in range(E)]
    return owners, kf


class Lane:
    def __init__(self, N, E, K=3, P=15, C=7, G=10, keep=1, sched=None, prob=1.0, seed=0, types=None, bubble=None):
        self.N, self.E, self.K, self.P, self.C, self.G, self.keep = N, E, K, P, C, G, keep
        # types[s]: set of kinds slot s takes ("d0", "d1": arrivals on that direction, "own"); bubble: an owner injects
        # only while fewer than `bubble` slots of the whole lane hold experts still to forward (sync model only)
        self.types = types or [{"d0", "d1", "own"}] * K
        self.bubble = bubble
        self.cprio = "oldest"
        self.owners, self.kf = plan(N, E)
        self.sched, self.prob, self.rng = sched, prob, random.Random(seed)
        self.nxt = lambda x, d: (x + 1) % N if d == 0 else (x - 1) % N
        self.route = {}  # (e, x) -> (direction it arrives on, hops left after x)
        for e in range(E):
            o = self.owners[e]
            for d, n in ((0, self.kf[e]), (1, N - 1 - self.kf[e])):
                for h in range(1, n + 1):
                    self.route[(e, (o + (h if d == 0 else -h)) % N)] = (d, n - h)
        self.own_q = [deque(e for e in range(E) if self.owners[e] == x) for x in range(N)]
        self.slot = {(x, s): None for x in range(N) for s in range(K)}
        self.held = {(x, s): [] for x in range(N) for s in range(K)}  # schedule: experts each slot held, in order
        self.order = {x: [] for x in range(N)}  # schedule: compute order (expert ids)
        self.comp = {x: dict(end=-1, cur=None) for x in range(N)}
        self.total = N * E  # (chip, expert) pairs to compute
        self.done = 0
        self.t = 0
        self.busy_links = 0

    def go(self):
        return self.prob >= 1.0 or self.rng.random() < self.prob

    def nexp_for(self, x, s):
        """Replay: the next expert slot s of chip x takes (or None when its schedule is done)."""
        q = self.sched["held"][(x, s)]
        k = len(self.held[(x, s)])
        return q[k] if k < len(q) else None

    def new(self, x, s, e):
        fwd = {}
        if self.owners[e] == x:
            for d, n in ((0, self.kf[e]), (1, self.N - 1 - self.kf[e])):
                if n > 0:
                    fwd[d] = 0
        else:
            d, rest = self.route[(e, x)]
            if rest > 0:
                fwd[d] = 0
        self.held[(x, s)].append(e)
        sl = dict(e=e, arr=0, fwd=fwd, computed=False, started=False, since=self.t)
        self.slot[(x, s)] = sl
        return sl

    def free_slots(self, x):
        return [s for s in range(self.K) if self.slot[(x, s)] is None]

    def find(self, x, e):
        for s in range(self.K):
            sl = self.slot[(x, s)]
            if sl is not None and sl["e"] == e:
                return sl
        return None

    def step(self):
        N, K, P = self.N, self.K, self.P
        moved = False
        wanting = set()  # chips a passing expert wants to enter this step (owners hold back for them)
        for x in range(N):
            for d in (0, 1):
                y = self.nxt(x, d)
                cand = [self.slot[(x, s)] for s in range(K)
                        if self.slot[(x, s)] and d in self.slot[(x, s)]["fwd"] and self.slot[(x, s)]["fwd"][d] < P]
                if not cand:
                    continue
                if self.sched is not None:
                    # the link carries experts in the order the downstream slots expect them
                    exp = {self.nexp_for(y, s): s for s in range(K) if self.slot[(y, s)] is None}
                    for s in range(K):
                        sl = self.slot[(y, s)]
                        if sl is not None and sl["arr"] < P:
                            exp[sl["e"]] = s
                    cand = [c for c in cand if c["e"] in exp]
                    if not cand:
                        continue
                sl = min(cand, key=lambda c: c["since"])
                e = sl["e"]
                tgt = self.find(y, e)
                if tgt is None:
                    if self.sched is not None:
                        s_t = [s for s in range(K) if self.slot[(y, s)] is None and self.nexp_for(y, s) == e]
                    else:
                        s_t = [s for s in self.free_slots(y) if f"d{d}" in self.types[s]]
                        wanting.add(y)
                    if not s_t or not self.go():
                        continue
                    tgt = self.new(y, s_t[0], e)
                u = sl["fwd"][d]
                if u < sl["arr"] and tgt["arr"] == u and self.go():
                    sl["fwd"][d] += 1
                    tgt["arr"] += 1
                    moved = True
                    self.busy_links += 1
        for x in range(N):
            if not self.own_q[x]:
                continue
            e = self.own_q[x][0]
            if self.sched is not None:
                s_t = [s for s in range(K) if self.slot[(x, s)] is None and self.nexp_for(x, s) == e]
            else:
                fr = self.free_slots(x)
                s_t = [s for s in fr if "own" in self.types[s]] if (len(fr) > self.keep and x not in wanting) else []
                if s_t and self.bubble is not None:
                    pending = sum(1 for sl in self.slot.values() if sl and any(v < P for v in sl["fwd"].values()))
                    if pending >= self.bubble:
                        s_t = []
            if not s_t or not self.go():
                continue
            self.own_q[x].popleft()
            sl = self.new(x, s_t[0], e)
            sl["arr"] = P
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
            if self.sched is not None:
                k = len(self.order[x])
                if k >= len(self.sched["order"][x]):
                    continue
                want = self.sched["order"][x][k]
                cands = [self.slot[(x, s)] for s in range(K) if self.slot[(x, s)] and self.slot[(x, s)]["e"] == want]
            else:
                cands = [self.slot[(x, s)] for s in range(K) if self.slot[(x, s)]]
            cands = [sl for sl in cands if not sl["started"] and sl["arr"] >= self.G]
            if not cands or not self.go():
                continue
            if self.sched is None and self.cprio == "fwd_done":
                key = lambda c_: (any(v < self.P for v in c_["fwd"].values()), c_["since"])
            else:
                key = lambda c_: c_["since"]
            sl = min(cands, key=key)
            sl["started"] = True
            self.order[x].append(sl["e"])
            c["cur"] = sl
            c["end"] = self.t + self.C
            moved = True
        for (x, s), sl in self.slot.items():
            if sl and sl["computed"] and sl["arr"] == P and all(v == P for v in sl["fwd"].values()):
                self.slot[(x, s)] = None
                moved = True
        self.t += 1
        return moved

    def run(self, limit=10 ** 6, stall=3000):
        idle = 0
        while self.done < self.total and self.t < limit:
            if self.step():
                idle = 0
            else:
                idle += 1
                if idle > stall:
                    return False
        return self.done >= self.total

    def schedule(self):
        return dict(held={k: list(v) for k, v in self.held.items()}, order={k: list(v) for k, v in self.order.items()})


def evaluate(N=32, E=160, K=3, keep=1, C=7, G=10, seeds=3, prob=0.6, types=None, bubble=None, cprio="oldest"):
    m = Lane(N, E, K=K, keep=keep, C=C, G=G, types=types, bubble=bubble)
    m.cprio = cprio
    ok = m.run()
    res = dict(sync_ok=ok, sync_t=m.t, sync_util=E * 7 / m.t, link_busy=m.busy_links / (2 * N * m.t))
    if ok:
        sch = m.schedule()
        reps = []
        for s in range(seeds):
            r = Lane(N, E, K=K, keep=keep, C=C, G=G, sched=sch, prob=prob, seed=s, types=types)
            ok2 = r.run(limit=10 ** 7, stall=50000)
            reps.append((ok2, E * C / r.t / prob))
        res["replay"] = [(a, b * 7 / C) for a, b in reps]  # (util against ideal compute, C = 7)
    return res


if __name__ == "__main__":
    import sys
    N, E = int(sys.argv[1]), int(sys.argv[2])
    for K in [int(k) for k in sys.argv[3].split(",")]:
        for keep in [int(k) for k in sys.argv[4].split(",")]:
            r = evaluate(N, E, K=K, keep=keep)
            rep = ", ".join(f"{'ok' if a else 'DEADLOCK'} {b:.3f}" for a, b in r.get("replay", []))
            print(f"N {N} E {E} K {K} keep {keep}: sync {'done' if r['sync_ok'] else 'STUCK'} util {r['sync_util']:.3f} "
                  f"(links {r['link_busy']:.2f} busy); replay p 0.6: {rep}", flush=True)

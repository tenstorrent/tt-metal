# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU model of the combine's landing protocol (sparse_combine_rf): every row a source chip sends to an accumulator on a
home chip lands in a slot of that accumulator's landing pool, and only in a slot reserved for it - so the fabric always
drains (a row in flight never waits for a slot).

landing "rings": the pool carved per call into one ring per source (depth by demand); a source's room on a ring is
depth + credits - sent, credits = the cumulative rows the accumulator consumed from it, returned by the home port's
snapshots.
landing "shared": no rings - the accumulator keeps a free list and grants a source specific slot ids (cumulative grant
count, each id tagged with its grant's sequence number); a grant goes accumulator -> home port (the id and the count as
two separate NoC writes: they may land in either order) -> the source port in the home port's snapshot (the port
forwards a count only up to its last id whose tag matches - validated); the source writes its n-th row into grant n's
slot, only once that id's tag says n; a consumed slot goes back to the free list and to a source with rows still to
come (at most G grants outstanding a (source, accumulator): the grant ring's size).

Links: each directed (chip, chip) pair a FIFO delivered at random times (a fabric link keeps a sender's order); the
NoC writes inside a chip (accumulator -> its home port) delivered in random order. Checks under random interleavings:
no deadlock, no slot written while it holds an unconsumed row, every row consumed exactly once, no row in a slot not
granted / reserved to its source. The negative tests drop one rule and must be caught."""

import collections
import random

import pytest


class Model:
    def __init__(self, N, A, pool, want, landing, G=8, seed=0, break_rule=None):
        """want[(s, h, o)]: rows source s sends to accumulator o of home h."""
        self.N, self.A, self.pool, self.want, self.mode, self.G = N, A, pool, dict(want), landing, G
        self.rng = random.Random(seed)
        self.brk = break_rule
        self.links = collections.defaultdict(collections.deque)  # (src chip, dst chip) -> messages in order
        self.noc = []  # (inside a chip: accumulator -> home port writes, any order)
        self.slot_row = {}  # (h, o, slot) -> row id held, unconsumed
        self.consumed = collections.Counter()
        self.sent = collections.Counter()  # (s, h, o) -> rows sent
        self.left = dict(want)  # rows still to send
        self.acc = {}
        for h in range(N):
            for o in range(A):
                self.acc[(h, o)] = self._acc_init(h, o)
        # source port state: per (s, h, o) the counts / ids it has from the home's snapshots
        self.src_cnt = collections.Counter()
        self.src_ids = collections.defaultdict(dict)  # (s, h, o) -> {ring index: (seq, slot)}
        # home port state: per (h, s, o) the grants written by the accumulator (shared) / consumed counts (rings)
        self.hp_cnt = collections.Counter()
        self.hp_ids = collections.defaultdict(dict)
        self.hp_fwd = collections.Counter()  # the count last forwarded

    def _acc_init(self, h, o):
        srcs = [s for s in range(self.N) if s != h and self.want.get((s, h, o), 0)]
        st = dict(srcs=srcs, taken=collections.Counter(), out=collections.Counter(), granted=collections.Counter())
        if self.mode == "rings":  # (a ring per source: depth by demand, at least 1; the carve's shape is not the point)
            deps, n = {}, len(srcs)
            assert self.pool >= n
            for i, s in enumerate(srcs):
                deps[s] = max(1, min(self.want[(s, h, o)], self.pool // n + (1 if i < self.pool % n else 0)))
            base = 0
            st["ring"] = {}
            for s in srcs:
                st["ring"][s] = (base, deps[s])
                base += deps[s]
        else:
            st["free"] = list(range(self.pool))
            st["owner"] = {}
            for s in srcs:  # (one each first, then round robin while slots and rows remain)
                self._grant(h, o, st, s)
            more = True
            while st["free"] and more:
                more = False
                for s in srcs:
                    if st["free"] and self._need(h, o, st, s) > 0 and self._grant(h, o, st, s):
                        more = True
        return st

    def _need(self, h, o, st, s):
        return self.want[(s, h, o)] - st["granted"][s]

    def _grant(self, h, o, st, s):
        if self._need(h, o, st, s) <= 0 or not st["free"]:
            return False
        if self.brk != "unbounded_grants" and st["out"][s] >= self.G:
            return False
        slot = st["free"].pop(0)
        seq = st["granted"][s]
        st["granted"][s] += 1
        st["out"][s] += 1
        st["owner"][slot] = s
        # (two NoC writes to the home port: the id (tagged) and the count - either may land first)
        self.noc.append(("id", h, s, o, seq, slot))
        self.noc.append(("cnt", h, s, o, seq + 1))
        return True

    # ---- steps ----
    def step(self):
        acts = []
        for k, q in self.links.items():
            if q:
                acts.append(("link", k))
        if self.noc:
            acts.append(("noc", None))
        acts += [("src", s) for s in range(self.N)]
        acts += [("port", h) for h in range(self.N)]
        acts += [("acc", ho) for ho in self.acc]
        self.rng.shuffle(acts)
        for a, k in acts:
            if getattr(self, "_" + a)(k):
                return True
        return False

    def _link(self, k):
        msg = self.links[k].popleft()
        if msg[0] == "row":
            _, s, h, o, slot, rid = msg
            key = (h, o, slot)
            assert key not in self.slot_row, f"slot {key} overwritten (holds {self.slot_row[key]}, row {rid})"
            st = self.acc[(h, o)]
            if self.mode == "shared":
                assert st["owner"].get(slot) == s, f"row of source {s} in slot {key} not granted to it"
            else:
                b, d = st["ring"][s]
                assert b <= slot < b + d, f"row of source {s} outside its ring at {key}"
            self.slot_row[key] = (s, rid)
        else:  # snapshot: (counts / ids) of home h for source s
            _, h, s, cnts, ids = msg
            for o, c in cnts.items():
                self.src_cnt[(s, h, o)] = max(self.src_cnt[(s, h, o)], c)
            for o, d in ids.items():
                self.src_ids[(s, h, o)].update(d)
        return True

    def _noc(self, _):
        i = self.rng.randrange(len(self.noc))
        m = self.noc.pop(i)
        if m[0] == "id":
            _, h, s, o, seq, slot = m
            self.hp_ids[(h, s, o)][seq % self.G] = (seq, slot)
        else:
            _, h, s, o, c = m
            self.hp_cnt[(h, s, o)] = max(self.hp_cnt[(h, s, o)], c)
        return True

    def _src(self, s):
        cands = [(h, o) for (s2, h, o), n in self.left.items() if s2 == s and n > 0]
        self.rng.shuffle(cands)
        for h, o in cands:
            key = (s, h, o)
            n = self.sent[key]
            if self.mode == "rings":
                b, d = self.acc[(h, o)]["ring"][s]
                if d + self.src_cnt[key] - n < 1:
                    continue
                slot = b + n % d
            else:
                if self.src_cnt[key] <= n:
                    continue
                e = self.src_ids[key].get(n % self.G)
                if self.brk != "no_tag_check" and (e is None or e[0] != n):
                    continue
                if e is None:
                    continue
                slot = e[1]
            self.links[(s, h)].append(("row", s, h, o, slot, n))
            self.sent[key] += 1
            self.left[key] -= 1
            return True
        return False

    def _port(self, h):
        did = False
        for s in range(self.N):
            if s == h:
                continue
            cnts, ids = {}, {}
            for o in range(self.A):
                k = (h, s, o)
                c = self.hp_cnt[k]
                if self.mode == "shared":
                    v = self.hp_fwd[k]
                    if self.brk == "no_validate":
                        v = c
                    else:
                        while v < c and self.hp_ids[k].get(v % self.G, (-1,))[0] == v:
                            v += 1
                    c = v
                    ids[o] = dict(self.hp_ids[k])
                if c > self.hp_fwd[k]:
                    cnts[o] = c
                    self.hp_fwd[k] = c
            if cnts:  # (a snapshot's counts and ids are one fabric packet, but its NoC write may land in any order)
                parts = [("snap", h, s, cnts, {}), ("snap", h, s, {}, ids)]
                self.rng.shuffle(parts)
                self.links[(h, s)].extend(parts)
                did = True
        return did

    def _acc(self, ho):
        h, o = ho
        st = self.acc[ho]
        full = [(slot, v) for (h2, o2, slot), v in self.slot_row.items() if (h2, o2) == ho]
        if self.mode == "rings":  # (a ring is read in order: its head only)
            full = [(slot, v) for slot, v in full if v[1] == st["taken"][v[0]]]
        if not full:
            return False
        slot, (s, rid) = self.rng.choice(full)
        del self.slot_row[(h, o, slot)]
        self.consumed[(s, h, o, rid)] += 1
        st["taken"][s] += 1
        if self.mode == "rings":  # (the credit: one consumed row, through the home port)
            self.hp_cnt[(h, s, o)] += 1
        else:
            st["out"][s] -= 1
            del st["owner"][slot]
            st["free"].append(slot)
            # regrant: the source with the most rows still to grant, fewest outstanding (at least 1 each kept)
            cands = [x for x in st["srcs"] if self._need(h, o, st, x) > 0]
            cands.sort(key=lambda x: (st["out"][x], -self._need(h, o, st, x)))
            for x in cands:
                if not st["free"]:
                    break
                self._grant(h, o, st, x)
        return True

    def run(self, max_steps=2_000_000):
        for _ in range(max_steps):
            if not self.step():
                break
        total = sum(self.want.values())
        assert all(v == 1 for v in self.consumed.values()), "a row consumed twice"
        assert sum(self.consumed.values()) == total, (
            f"deadlock: {sum(self.consumed.values())} of {total} rows consumed, "
            f"{sum(len(q) for q in self.links.values())} messages in flight"
        )


def _want(N, A, rows, kind, rng):
    w = {}
    for s in range(N):
        for h in range(N):
            if s == h:
                continue
            for o in range(A):
                if kind == "uniform":
                    n = rows
                elif kind == "skewed":  # (one hot source a home)
                    n = rows * 4 if s == (h + 1) % N else rng.randint(0, rows)
                else:  # sparse: many sources with nothing
                    n = rows if rng.random() < 0.3 else 0
                if n:
                    w[(s, h, o)] = n
    return w


CASES = [  # (N, A, pool, rows, routing)
    (4, 2, 3, 6, "uniform"),  # (the pool = one slot a source: D 1)
    (4, 2, 6, 6, "skewed"),
    (8, 2, 7, 4, "uniform"),
    (8, 3, 11, 5, "skewed"),
    (8, 2, 9, 5, "sparse"),
]


@pytest.mark.parametrize("landing", ["rings", "shared"])
@pytest.mark.parametrize("case", CASES, ids=[f"N{c[0]}-A{c[1]}-pool{c[2]}-{c[4]}" for c in CASES])
@pytest.mark.parametrize("seed", range(4))
def test_combine_landing(landing, case, seed):
    N, A, pool, rows, kind = case
    want = _want(N, A, rows, kind, random.Random(seed))
    Model(N, A, pool, want, landing, G=4, seed=seed).run()


@pytest.mark.parametrize("rule", ["no_tag_check", "no_validate", "unbounded_grants"])
def test_combine_landing_model_catches(rule):
    """Each shared-pool rule dropped: some interleaving writes a row where it must not (or uses a stale id)."""
    caught = 0
    for seed in range(60):
        N, A, pool, rows = 4, 2, 6, 8
        want = _want(N, A, rows, "skewed", random.Random(seed))
        try:
            Model(N, A, pool, want, "shared", G=2, seed=seed, break_rule=rule).run()
        except AssertionError:
            caught += 1
    assert caught, f"{rule}: the model never caught it"

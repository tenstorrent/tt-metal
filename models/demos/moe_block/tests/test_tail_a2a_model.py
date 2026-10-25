# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU model of tail_a2a's protocol (tt/tail_a2a.py: TailSchedule and the kernels' rules), several calls back to back
on a TP group (a line or a ring), random interleavings: workers (their row tiles in order, then +1 on every carrying
port's z-ready word), port readers (a row tile once z-ready says every worker wrote it; re-arm at the end), port
senders (readies first, every destination's ready awaited, the runs in the reader's order, a fused arrival increment
every INC_EVERY runs of a destination and its last; at the end every expected arrival, re-armed), the fabric (a
packet lands any time later, in order per (sender, destination)), a chip's next call only after its last one ended.
Checks: no deadlock; every run of every (source, destination) lands exactly once a call, with that call's data, never
before the destination started that call (the fence) and always before it ended (the arrival wait); every word at 0
after a call. Negative tests: without the fence, or with one arrival too few expected, the model must object."""

import random

import pytest

from models.demos.moe_block.tt.tail_a2a import INC_EVERY, TailSchedule


def _run(sc: TailSchedule, calls, seed, NW=3, fence=True, expect_delta=0, one_word=False):
    rnd = random.Random(seed)
    G, L, RT = sc.G, sc.L, sc.RT
    ports = [(p, d, l) for p in range(G) for d in ("fwd", "bwd") for l in range(L)]
    active = {k: bool(sc.dests(k[0], k[1])) for k in ports}
    st = dict(
        call={p: 0 for p in range(G)},  # (the call a chip runs; it may start call c + 1 once call c ended there)
        zr={(k, r): 0 for k in ports for r in range(RT)},  # (a port's z-ready word a row tile)
        rdy={k: 0 for k in ports},
        arr={k: 0 for k in ports},
    )
    landed = {}  # (dest p, call, source q, part, link, row, run) -> count
    fabric = {}  # (sender port, dest) -> [packets]
    prog = {}

    def start(p, c):
        prog[p] = dict(
            c=c,
            workers=[0] * NW,  # (rows done)
            rd={k: dict(r=0, fifo=0, done=False) for k in ports if k[0] == p},
            sd={k: dict(phase=0, i=0, sent={}, done=False) for k in ports if k[0] == p},
        )
        st["call"][p] = c

    def runs_of(p, d, l):  # (the sender's walk: per row tile, per destination, its runs)
        return [
            (r, q, part, i) for r in range(RT) for q, part in sc.dests(p, d) for i in range(len(sc.run_list(part, l)))
        ]

    for p in range(G):
        start(p, 0)

    def step_worker(p, w):
        pr = prog[p]
        r = pr["workers"][w]
        if r >= RT:
            return False
        pr["workers"][w] += 1  # (row tile r of my columns written to z)
        for k in ports:
            if k[0] == p and active[k]:
                st["zr"][(k, 0 if one_word else r)] += 1
        return True

    def step_reader(k):
        pr = prog[k[0]]
        rd = pr["rd"][k]
        if rd["done"] or not active[k]:
            rd["done"] = True
            return False
        if rd["r"] >= RT:
            if one_word:
                assert st["zr"][(k, 0)] == RT * NW, ("z-ready count off", k)
                st["zr"][(k, 0)] -= RT * NW
            else:
                for r in range(RT):
                    assert st["zr"][(k, r)] == NW, ("z-ready count off", k, r)
                    st["zr"][(k, r)] -= NW
            rd["done"] = True
            return True
        r = rd["r"]
        if (st["zr"][(k, 0)] < (r + 1) * NW) if one_word else (st["zr"][(k, r)] < NW):
            return False
        assert all(w > r for w in pr["workers"]), ("a row tile read from z before every worker wrote it", k, r)
        rd["fifo"] += sum(len(sc.run_list(part, k[2])) for _, part in sc.dests(k[0], k[1]))  # (row r's runs)
        rd["r"] += 1
        return True

    def step_sender(k):
        p, d, l = k
        pr = prog[p]
        sd = pr["sd"][k]
        if sd["done"]:
            return False
        if sd["phase"] == 0:  # (readies to everyone sending into my output through me)
            for xp, xd, xl in sorted(sc.readies.get(k, set())):
                fabric.setdefault((k, xp), []).append(("ready", (xp, xd, xl)))
            sd["phase"] = 1
            return True
        if sd["phase"] == 1:
            need = len({q for q, _ in sc.dests(p, d)})
            if fence and need and st["rdy"][k] < need:
                return False
            if fence:
                st["rdy"][k] -= need
            sd["phase"] = 2
            sd["walk"] = runs_of(p, d, l)
            return True
        if sd["phase"] == 2:
            if sd["i"] < len(sd["walk"]):
                if pr["rd"][k]["fifo"] == 0:
                    return False
                pr["rd"][k]["fifo"] -= 1
                r, q, part, i = sd["walk"][sd["i"]]
                n = sd["sent"].get(q, 0) + 1
                sd["sent"][q] = n
                total = RT * sc.runs(part, l)
                inc = n % INC_EVERY == 0 or n == total
                fabric.setdefault((k, q), []).append(
                    ("data", (q, pr["c"], p, part, l, r, i, inc, (q, sc.dests(p, d) and _opp(d), l)))
                )
                sd["i"] += 1
                return True
            sd["phase"] = 3
            return True
        expect = sc.expect.get(k, 0) + expect_delta * (1 if sc.expect.get(k, 0) else 0)
        if st["arr"][k] < expect:
            return False
        st["arr"][k] -= expect
        sd["done"] = True
        return True

    def deliver(key):
        q_ = fabric.get(key)
        if not q_:
            return False
        kind, v = q_.pop(0)
        if kind == "ready":
            st["rdy"][v] += 1
            return True
        q, c, src, part, l, r, i, inc, aport = v
        assert st["call"][q] == c and q in prog and prog[q]["c"] == c, (
            "a write before its destination started the call",
            v,
        )
        kk = (q, c, src, part, l, r, i)
        landed[kk] = landed.get(kk, 0) + 1
        assert landed[kk] == 1, ("a run landed twice", kk)
        if inc:
            st["arr"][aport] += 1
        return True

    def chip_done(p):
        pr = prog[p]
        return (
            all(w >= RT for w in pr["workers"])
            and all(v["done"] for v in pr["rd"].values())
            and all(v["done"] for v in pr["sd"].values())
        )

    def _opp(d):
        return "bwd" if d == "fwd" else "fwd"

    finished = {p: 0 for p in range(G)}
    while True:
        acts = []
        for p in range(G):
            if p in prog:
                acts += [("w", p, w) for w in range(NW)]
                acts += [("r", k) for k in ports if k[0] == p] + [("s", k) for k in ports if k[0] == p]
                acts.append(("end", p))
        acts += [("f", key) for key in list(fabric)]
        rnd.shuffle(acts)
        moved = False
        for a in acts:
            if a[0] == "w":
                moved |= a[1] in prog and step_worker(a[1], a[2])
            elif a[0] == "r":
                moved |= a[1][0] in prog and step_reader(a[1])
            elif a[0] == "s":
                moved |= a[1][0] in prog and step_sender(a[1])
            elif a[0] == "f":
                moved |= deliver(a[1])
            elif a[1] in prog and chip_done(a[1]):
                p = a[1]
                c = prog[p]["c"]
                # (the program ended: everything sent into my output this call has landed)
                for src in range(G):
                    if src == p:
                        continue
                    for d in ("fwd", "bwd"):
                        for l in range(L):
                            for q, part in sc.dests(src, d):
                                if q != p:
                                    continue
                                for r in range(RT):
                                    for i in range(len(sc.run_list(part, l))):
                                        assert landed.get((p, c, src, part, l, r, i)) == 1, (
                                            "the call ended before a write landed",
                                            (p, c, src, part, l, r, i),
                                        )
                for k in ports:
                    if k[0] == p:
                        assert st["arr"][k] == 0 and all(st["zr"][(k, r)] == 0 for r in range(RT)), ("not re-armed", k)
                del prog[p]
                finished[p] = c + 1
                if c + 1 < calls:
                    start(p, c + 1)
                moved = True
        if not moved:
            break
    assert all(v == calls for v in finished.values()), f"deadlock: calls finished {finished}"
    assert not any(fabric.values()), "packets left in the fabric"


CASES = [
    dict(G=4, ring=False, L=2, NTB=56, RT=2, NBK=8, RP=2),  # (K2.7's block on the 2x4 line)
    dict(G=4, ring=True, L=2, NTB=56, RT=2, NBK=8, RP=2),  # (a ring of 4)
    dict(G=4, ring=False, L=2, NTB=32, RT=3, NBK=8, RP=4),
    dict(G=4, ring=True, L=1, NTB=12, RT=2, NBK=8, RP=2),
    dict(G=2, ring=False, L=2, NTB=24, RT=2, NBK=8, RP=2),
    dict(G=8, ring=True, L=2, NTB=16, RT=1, NBK=8, RP=2),
]


@pytest.mark.parametrize("case", CASES, ids=[f"G{c['G']}-{'ring' if c['ring'] else 'line'}-L{c['L']}" for c in CASES])
@pytest.mark.parametrize("seed", range(6))
def test_tail_a2a_protocol(case, seed):
    _run(TailSchedule(**case), calls=3, seed=seed)


@pytest.mark.parametrize("what", ["fence", "expect", "one_word"])
def test_tail_a2a_model_catches(what):
    """Without the ready fence a write lands before its destination started the call (a fast chip runs ahead); with
    one arrival too few expected a call ends before its last write landed; with one z-ready count over the call (not
    a word a row tile) a port reads a row tile some slow worker has not written (the device: K2.7 640 rows, a row
    tile of garbage): the model must object (not vacuous)."""
    sc = TailSchedule(**{**CASES[0], "RT": 4})
    for seed in range(80):
        try:
            _run(
                sc,
                calls=3,
                seed=seed,
                NW=4,
                fence=what != "fence",
                expect_delta=-1 if what == "expect" else 0,
                one_word=what == "one_word",
            )
        except AssertionError:
            return
    raise AssertionError(f"the model accepted the protocol without the {what}")

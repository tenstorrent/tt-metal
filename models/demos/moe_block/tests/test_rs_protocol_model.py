# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU model of the fused head's down + reduce-scatter protocol (tt/se_rs.py RsPlan, the kernels' rules): per chip
the cores' compute programs (passes, hand-offs, finishes behind the next entry), the BRISC hand-offs to the ports'
mailboxes, the port readers / senders walking their units in entry order, the fused arrival increments, the NCRISC's
arrival credits by threshold. Data is symbolic (the set of chips summed into a tile). Checks: everything completes
(no deadlock under random interleavings), every fin tile is the sum over all G chips exactly once, every zone slot is
written once, the thresholds count exactly the increments."""

import random

import pytest

from models.demos.moe_block.tt.se_rs import DIRS, LogitsPlan, RsPlan


def _run(plan, seed):
    rnd = random.Random(seed)
    G, GX, GY = plan.G, plan.GX, plan.GY
    cores = [(cx, cy) for cy in range(GY) for cx in range(GX)]
    # state
    part = {}  # (p, cx, cy, e) -> frozenset (the sub-block's sum; tiles share it)
    zone = {}  # (p, cx, cy, d, slot) -> frozenset
    ctr = {(p, c, d): 0 for p in range(G) for c in cores for d in DIRS}
    fin = {}
    mbox = {(p, d, l, c): 0 for p in range(G) for d in DIRS for l in range(plan.nv) for c in cores}
    prog = {(p, c): plan.ops(p, *c) for p in range(G) for c in cores}
    pc = {k: 0 for k in prog}
    hand_q = {(p, c): [] for p in range(G) for c in cores}  # finished hand-offs not yet signalled (BRISC)
    arr_lists = {(p, c): plan.arrivals(p, *c) for p in range(G) for c in cores}
    credited = {(p, c, d): 0 for p in range(G) for c in cores for d in DIRS}  # NCRISC: arrivals pushed
    # ports: per (p, d, l): units walk state
    ports = {}
    for p in range(G):
        for d in DIRS:
            units = plan.port_units(p, d) if plan.down(p, d) is not None else []
            for l in range(plan.nv):
                ports[(p, d, l)] = dict(units=units, u=0, i=0, seen={c: 0 for c in cores})

    def compute_step(p, c):
        ops = prog[(p, c)]
        if pc[(p, c)] >= len(ops):
            return False
        kind, i = ops[pc[(p, c)]]
        cx, cy = c
        k = plan.k(cx, cy)
        h = plan.half(k)
        ents = plan.entries(p, h)
        if kind == "pass":
            for e, b in enumerate(ents):  # (symbolic: a pass contributes to its entries; all of an entry by its last)
                part.setdefault((p, cx, cy, e), frozenset([p]))
            pc[(p, c)] += 1
            return True
        e = i
        b = ents[e]
        d, start = plan.role(p, b, h)
        if kind == "hand":
            hand_q[(p, c)].append(e)
            pc[(p, c)] += 1
            return True
        # fin: every arrival of this entry credited (the NCRISC pushed its slot)
        need = [(dd, slot) for dd in DIRS for (ee, slot, _) in arr_lists[(p, c)][dd] if ee == e]
        for dd, slot in need:
            idx = [s for (_, s, _) in arr_lists[(p, c)][dd]].index(slot)
            if credited[(p, c, dd)] <= idx:
                return False
        s = part[(p, cx, cy, e)]
        for dd, slot in need:
            z = zone[(p, cx, cy, dd, slot)]
            assert not (s & z), ("double count", p, c, e, s, z)
            s = s | z
        if d == "fin":
            assert (p, cx, cy) not in fin
            fin[(p, cx, cy)] = s
        else:
            part[(p, cx, cy, e)] = s
            hand_q[(p, c)].append(e)
        pc[(p, c)] += 1
        return True

    def brisc_step(p, c):
        if not hand_q[(p, c)]:
            return False
        e = hand_q[(p, c)].pop(0)
        k = plan.k(*c)
        h = plan.half(k)
        b = plan.entries(p, h)[e]
        d, _ = plan.role(p, b, h)
        mbox[(p, d, plan.link(k), c)] += 1
        return True

    def ncrisc_step(p, c):
        moved = False
        for d in DIRS:
            lst = arr_lists[(p, c)][d]
            n = credited[(p, c, d)]
            if n < len(lst):
                _, slot, thr = lst[n]
                if thr == 0 or ctr[(p, c, d)] >= thr:
                    credited[(p, c, d)] += 1
                    moved = True
        return moved

    def port_step(p, d, l):
        st = ports[(p, d, l)]
        units = st["units"]
        pcs = plan.port_cores(l)
        while st["u"] < len(units):
            e, b, mask, s0, s1 = units[st["u"]]
            while st["i"] < len(pcs):
                c = pcs[st["i"]]
                k = plan.k(*c)
                h = plan.half(k)
                if not (mask >> h) & 1 or plan.count(*c, b) == 0:
                    st["i"] += 1
                    continue
                if mbox[(p, d, l, c)] < st["seen"][c] + 1:
                    return False  # blocked on this core's hand-off
                st["seen"][c] += 1
                q = plan.down(p, d)
                slot = s1 if h else s0
                key = (q, c[0], c[1], d, slot)
                assert key not in zone, ("zone slot written twice", key)
                zone[key] = part[(p, c[0], c[1], e)]
                ctr[(q, c, d)] += 1
                st["i"] += 1
                return True
            st["u"] += 1
            st["i"] = 0
        return False

    actors = (
        [("c", p, c) for p in range(G) for c in cores]
        + [("b", p, c) for p in range(G) for c in cores]
        + [("n", p, c) for p in range(G) for c in cores]
        + [("port",) + k for k in ports]
    )
    step = {
        "c": lambda a: compute_step(a[1], a[2]),
        "b": lambda a: brisc_step(a[1], a[2]),
        "n": lambda a: ncrisc_step(a[1], a[2]),
        "port": lambda a: port_step(a[1], a[2], a[3]),
    }
    while True:
        rnd.shuffle(actors)
        if not any(step[a[0]](a) for a in actors):
            break
    stuck = [k for k in prog if pc[k] < len(prog[k])]
    assert not stuck, f"deadlock: {len(stuck)} cores stuck, e.g. {stuck[:3]} at {[prog[k][pc[k]] for k in stuck[:3]]}"
    full = frozenset(range(G))
    for p in range(G):
        for c in cores:
            if plan.count(*c, p) > 0:
                assert fin[(p,) + c] == full, (p, c, fin.get((p,) + c))
    # thresholds = the increments: every counter ends at its core's last threshold
    for p in range(G):
        for c in cores:
            for d in DIRS:
                thr = [t for _, _, t in arr_lists[(p, c)][d] if t]
                assert ctr[(p, c, d)] == (max(thr) if thr else 0), (p, c, d, ctr[(p, c, d)], thr)
    for st in ports.values():
        assert st["u"] == len(st["units"])


CASES = [
    # (G, ring, GX, GY, MT, NTH, scatter, nv)
    (4, False, 11, 10, 2, 224, -1, 2),  # K3 on the LoudBox line
    (4, False, 11, 10, 2, 128, -2, 2),  # GLM-Flash on the line
    (4, True, 11, 10, 2, 224, -1, 2),  # Galaxy TP ring
    (4, True, 11, 10, 2, 128, -2, 2),
    (2, False, 11, 10, 2, 128, -2, 2),
    (4, False, 12, 10, 2, 128, -2, 1),
]


@pytest.mark.parametrize("case", CASES, ids=[f"G{c[0]}-{'ring' if c[1] else 'line'}-s{c[6]}-nv{c[7]}" for c in CASES])
@pytest.mark.parametrize("seed", range(4))
def test_rs_protocol(case, seed):
    G, ring, GX, GY, MT, NTH, sc, nv = case
    _run(RsPlan(G=G, ring=ring, GX=GX, GY=GY, MT=MT, NTH=NTH, scatter=sc, nv=nv, pwmax=8 // MT), seed)


def test_rs_model_catches_bad_threshold():
    """An off-by-one threshold must break it (the model is not vacuous)."""
    plan = RsPlan(G=4, ring=False, GX=11, GY=10, MT=2, NTH=224, scatter=-1, nv=2, pwmax=4)
    orig = plan.arrivals

    def bad(p, cx, cy):
        a = orig(p, cx, cy)
        if (p, cx, cy) == (1, 0, 0):
            a["fwd"] = [(e, s, t - 1 if t > 1 else t) for e, s, t in a["fwd"]]
        return a

    plan.arrivals = bad
    try:
        _run(plan, 0)
    except (AssertionError, KeyError):  # (a slot read before it landed, or the counts off)
        return
    raise AssertionError("the model accepted a wrong threshold")


@pytest.mark.parametrize("ring", [False, True], ids=["line", "ring"])
@pytest.mark.parametrize("seed", range(4))
def test_logits_protocol(ring, seed):
    """The router logits' reduce-scatter (5 router cores, one owner row tile each, fp32) on the same protocol."""
    _run(LogitsPlan(G=4, ring=ring, RB=5, ET=14, nv=2), seed)

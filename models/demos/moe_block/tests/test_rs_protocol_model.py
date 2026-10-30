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


def _zone_write(plan, q, c, d, h, slot, b, live):
    """The port's write into core c's direction-d zone on chip q, addressed as x_gather3's rs_send does (P-tile slots,
    or compact: the sum of the core's tiles of the unit's prefix-mask blocks, from its counts word): must be the
    core's own slot offset (zone_slots), inside the zone, and overlap no other slot written this chunk (live)."""
    n = plan.count(*c, b)
    if not plan.compact:
        off = slot * plan.P
    else:
        cnts, pm = plan.counts_word(*c), plan.prefix_mask(q, h, d, slot)
        off = sum((cnts >> (8 * j)) & 0xFF for j in range(4) if (pm >> j) & 1)
    zo, zn = plan.zone_slots(q, *c, d)[slot]
    assert off == zo and n <= zn, ("the port's zone offset is not the core's slot", q, c, d, slot, off, zo, n, zn)
    assert off + n <= plan.zone_tiles(d), ("a zone write past the zone", q, c, d, slot, off, n)
    for o2, n2 in live:
        assert off + n <= o2 or o2 + n2 <= off, ("zone writes overlap", q, c, d, slot, (off, n), (o2, n2))
    live.append((off, n))


def _run(plan, seed):
    rnd = random.Random(seed)
    G, GX, GY = plan.G, plan.GX, plan.GY
    cores = [(cx, cy) for cy in range(GY) for cx in range(GX)]
    # state
    part = {}  # (p, cx, cy, e) -> frozenset (the sub-block's sum; tiles share it)
    zone = {}  # (p, cx, cy, d, slot) -> frozenset
    zlive = {}  # (p, c, d) -> [(tile offset, tiles)] written into the zone
    ctr = {(p, c, d): 0 for p in range(G) for c in cores for d in DIRS}
    fin = {}
    mbox = {(p, d, l, c): 0 for p in range(G) for d in DIRS for l in range(plan.nv) for c in cores}
    mlog = {k: [] for k in mbox}  # (the entries handed, in order: a port reading a count must expect that entry)
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
                ports[(p, d, l)] = dict(units=units, u=0, ip=0, i=0, seen={c: 0 for c in cores})

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
        mlog[(p, d, plan.link(k), c)].append(e)
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
            e, b, mask, s0, s1, pcw = units[st["u"]]
            npc = plan.npass if pcw else 1  # (pieces: pass-major over the cores, a hand-off each)
            while st["i"] < len(pcs):
                c = pcs[st["i"]]
                k = plan.k(*c)
                h = plan.half(k)
                if not (mask >> h) & 1 or plan.count(*c, b) == 0:
                    st["i"] += 1
                    continue
                if mbox[(p, d, l, c)] < st["seen"][c] + 1:
                    return False  # blocked on this core's hand-off
                assert mlog[(p, d, l, c)][st["seen"][c]] == e, ("a hand-off count of another entry", p, d, l, c, e)
                st["seen"][c] += 1
                if st["ip"] + 1 == npc:  # (the core's last piece: the sub-block and its increment)
                    q = plan.down(p, d)
                    slot = s1 if h else s0
                    key = (q, c[0], c[1], d, slot)
                    assert key not in zone, ("zone slot written twice", key)
                    _zone_write(plan, q, c, d, h, slot, b, zlive.setdefault((q, c, d), []))
                    zone[key] = part[(p, c[0], c[1], e)]
                    ctr[(q, c, d)] += 1
                st["i"] += 1
                return True
            if st["ip"] + 1 < npc:
                st["ip"] += 1
                st["i"] = 0
                continue
            st["u"] += 1
            st["i"] = 0
            st["ip"] = 0
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


# ---- row chunks (NCH > 1): the chunk loop around the same protocol (se_engine3 / x_gather3) ----
def _run_chunks(plan, seed, nch, zone_credit=True, finb_credit=True, router=False, router_credit=True):
    """The chunked RS: every chunk the same programs; counters count up (thresholds m x a chunk's count + the chunk's);
    a core's arrival zones reused behind zone credits (its NCRISC, at chunk m + 1's first slot, once the compute read
    chunk m - +1 on the port facing that upstream; the port forwards one credit a chunk to the upstream's port, which
    sends chunk m once it has m), its FINB behind the ports' read credits (a port +1s each core it read, after a chunk;
    the compute starts chunk m's RS at m x its readers). Values carry their chunk: a stale or overwritten one fails.
    router: the router's logits on the matmul cores, their buffers over the zones (se_engine3 rtr_overlay) - a core's
    router phase comes before its compute and no zone write may land during it: every core counts its phase over on
    its chip's ports, which then send the upstream chunk 0's credit (the sender waits credits m + 1)."""
    rnd = random.Random(seed)
    G, GX, GY = plan.G, plan.GX, plan.GY
    cores = [(cx, cy) for cy in range(GY) for cx in range(GX)]
    opp = {"fwd": "bwd", "bwd": "fwd"}
    prog = {(p, c): plan.ops(p, *c) for p in range(G) for c in cores}
    arr = {(p, c): plan.arrivals(p, *c) for p in range(G) for c in cores}
    amax = {(p, c, d): max([t for _, _, t in arr[(p, c)][d]] or [0]) for p in range(G) for c in cores for d in DIRS}
    readers = {(p, c): len(plan.finb_readers(p, *c)) for p in range(G) for c in cores}
    finb = {}  # (p, c, e) -> (chunk, set): the exchanged sub-block (start / relay)
    zone = {}  # (p, c, d, slot) -> [chunk, set, read]
    zlive = {}  # (p, c, d, chunk) -> [(tile offset, tiles)] written into the zone
    ctr = {(p, c, d): 0 for p in range(G) for c in cores for d in DIRS}
    fin = {}
    mbox = {(p, d, l, c): 0 for p in range(G) for d in DIRS for l in range(plan.nv) for c in cores}
    mlog = {k: [] for k in mbox}
    ch = {k: 0 for k in prog}  # (the compute's chunk)
    pc = {k: 0 for k in prog}
    hand_q = {k: [] for k in prog}
    cred = {(p, c, d): 0 for p in range(G) for c in cores for d in DIRS}  # (NCRISC: slots pushed, all chunks)
    zsig = {(p, c, d): 0 for p in range(G) for c in cores for d in DIRS}
    ffree = {k: 0 for k in prog}
    zfree = {(p, d, l): 0 for p in range(G) for d in DIRS for l in range(plan.nv)}
    zcred = {(p, d, l): 0 for p in range(G) for d in DIRS for l in range(plan.nv)}
    zfw = {(p, d, l): 0 for p in range(G) for d in DIRS for l in range(plan.nv)}
    rtr = {(p, c): not router for p in range(G) for c in cores}  # (a core's router phase over)
    rdone = {p: 0 for p in range(G)}  # (a chip's "router done" count - every port's word, one model counter)
    rsent = {(p, d, l): False for p in range(G) for d in DIRS for l in range(plan.nv)}
    z0 = 1 if router and router_credit else 0
    zneed = {
        (p, d, l): sum(1 for c in plan.port_cores(l) if amax[(p, c, opp[d])])
        for p in range(G)
        for d in DIRS
        for l in range(plan.nv)
    }
    ports = {}
    for p in range(G):
        for d in DIRS:
            units = plan.port_units(p, d) if plan.down(p, d) is not None else []
            for l in range(plan.nv):
                ports[(p, d, l)] = dict(units=units, m=0, u=0, ip=0, i=0, seen={c: 0 for c in cores}, read=set())

    def compute_step(p, c):
        k_ = (p, c)
        m = ch[k_]
        if m >= nch or not rtr[k_]:  # (the router phase first: gate / up after it)
            return False
        ops = prog[k_]
        if pc[k_] == 0 and m > 0 and finb_credit and ffree[k_] < m * readers[k_]:
            return False  # (the ports still read the last chunk's FINB)
        if pc[k_] >= len(ops):  # (chunk end: the zones popped)
            ch[k_] += 1
            pc[k_] = 0
            return True
        kind, i = ops[pc[k_]]
        h = plan.half(plan.k(*c))
        ents = plan.entries(p, h)
        if kind == "pass":
            for e, b in enumerate(ents):
                d, start = plan.role(p, b, h)
                if d != "fin" and start:
                    finb[(p, c, e)] = (m, frozenset([p]))
            pc[k_] += 1
            return True
        e = i
        b = ents[e]
        d, start = plan.role(p, b, h)
        if kind == "hand":
            hand_q[k_].append((m, e))
            pc[k_] += 1
            return True
        need = [(dd, slot) for dd in DIRS for (ee, slot, _) in arr[k_][dd] if ee == e]
        for dd, slot in need:
            idx = [s_ for (_, s_, _) in arr[k_][dd]].index(slot)
            if cred[(p, c, dd)] <= m * len(arr[k_][dd]) + idx:
                return False
        s = frozenset([p])
        for dd, slot in need:
            z = zone[(p, c, dd, slot)]
            assert z[0] == m, ("a zone slot of another chunk", p, c, dd, slot, z[0], m)
            assert not (s & z[1]), ("double count", p, c, e)
            s = s | z[1]
            z[2] = True
        if d == "fin":
            assert (m, p, c) not in fin
            fin[(m, p, c)] = s
        else:
            finb[(p, c, e)] = (m, s)
            hand_q[k_].append((m, e))
        pc[k_] += 1
        return True

    def brisc_step(p, c):
        if not hand_q[(p, c)]:
            return False
        m, e = hand_q[(p, c)].pop(0)
        k = plan.k(*c)
        b = plan.entries(p, plan.half(k))[e]
        d, _ = plan.role(p, b, plan.half(k))
        mbox[(p, d, plan.link(k), c)] += 1
        mlog[(p, d, plan.link(k), c)].append((m, e))
        return True

    def ncrisc_step(p, c):
        moved = False
        for d in DIRS:
            lst = arr[(p, c)][d]
            n = cred[(p, c, d)]
            if not lst or n >= nch * len(lst):
                continue
            mc, sl = divmod(n, len(lst))
            if sl == 0 and mc > zsig[(p, c, d)]:  # (chunk mc's first slot: chunk mc - 1's zone read first)
                if ch[(p, c)] >= mc:
                    if amax[(p, c, d)]:  # (only a core something lands on signals)
                        zfree[(p, opp[d], plan.link(plan.k(*c)))] += 1
                    zsig[(p, c, d)] += 1
                    moved = True
                continue
            _, slot, thr = lst[sl]
            if thr == 0 or ctr[(p, c, d)] >= mc * amax[(p, c, d)] + thr:
                cred[(p, c, d)] += 1
                moved = True
        return moved

    def port_step(p, d, l):
        st = ports[(p, d, l)]
        moved = False
        q = plan.down(p, d)
        if z0 and not rsent[(p, d, l)] and rdone[p] == len(cores):  # (every core's router phase over: chunk 0's credit)
            if q is not None and zneed[(p, d, l)]:
                zcred[(q, opp[d], l)] += 1
            rsent[(p, d, l)] = True
            moved = True
        # (forward my cores' zone-free signals - behind chunk 0's credit)
        if (rsent[(p, d, l)] or not z0) and q is not None and zfw[(p, d, l)] < nch - 1 and zneed[(p, d, l)]:
            if zfree[(p, d, l)] >= (zfw[(p, d, l)] + 1) * zneed[(p, d, l)]:
                zcred[(q, opp[d], l)] += 1
                zfw[(p, d, l)] += 1
                moved = True
        units = st["units"]
        pcs = plan.port_cores(l)
        while st["m"] < nch:
            m = st["m"]
            if st["u"] >= len(units):  # (chunk m sent: my reads of it done - FINB-free to each core read)
                for c in st["read"]:
                    ffree[(p, c)] += 1 if m + 1 < nch else 0
                st.update(m=m + 1, u=0, ip=0, i=0, read=set())
                return True
            if m + z0 > 0 and zone_credit and zcred[(p, d, l)] < m + z0:
                return moved
            e, b, mask, s0, s1, pcw = units[st["u"]]
            npc = plan.npass if pcw else 1
            while st["i"] < len(pcs):
                c = pcs[st["i"]]
                h = plan.half(plan.k(*c))
                if not (mask >> h) & 1 or plan.count(*c, b) == 0:
                    st["i"] += 1
                    continue
                if mbox[(p, d, l, c)] < st["seen"][c] + 1:
                    return moved
                assert mlog[(p, d, l, c)][st["seen"][c]] == (m, e), ("a hand-off count of another entry", p, d, l, c)
                st["seen"][c] += 1
                st["read"].add(c)
                if st["ip"] + 1 < npc:  # (a piece before the core's last: its tiles land, no increment yet)
                    assert finb[(p, c, e)][0] == m, ("FINB of another chunk read", p, c, e)
                    assert rtr[(q, c)], (
                        "a zone write into a core's router buffers (its router phase not over)",
                        q,
                        c,
                        d,
                    )
                    st["i"] += 1
                    return True
                v = finb[(p, c, e)]
                assert v[0] == m, ("FINB of another chunk read", p, c, e, v[0], m)
                key = (q, c, d, s1 if h else s0)
                assert rtr[(q, c)], ("a zone write into a core's router buffers (its router phase not over)", q, c, d)
                _zone_write(plan, q, c, d, h, s1 if h else s0, b, zlive.setdefault((q, c, d, m), []))
                old = zone.get(key)
                assert old is None or (old[2] and old[0] < m), ("a zone slot overwritten before it was read", key)
                zone[key] = [m, v[1], False]
                ctr[(q, c, d)] += 1
                st["i"] += 1
                return True
            if st["ip"] + 1 < npc:
                st["ip"] += 1
                st["i"] = 0
                continue
            st["u"] += 1
            st["i"] = 0
            st["ip"] = 0
        return moved

    def router_step(p, c):
        if rtr[(p, c)]:
            return False
        rtr[(p, c)] = True
        rdone[p] += 1
        return True

    actors = (
        [("r", p, c) for p in range(G) for c in cores]
        + [("c", p, c) for p in range(G) for c in cores]
        + [("b", p, c) for p in range(G) for c in cores]
        + [("n", p, c) for p in range(G) for c in cores]
        + [("port",) + k for k in ports]
    )
    step = {
        "r": lambda a: router_step(a[1], a[2]),
        "c": lambda a: compute_step(a[1], a[2]),
        "b": lambda a: brisc_step(a[1], a[2]),
        "n": lambda a: ncrisc_step(a[1], a[2]),
        "port": lambda a: port_step(a[1], a[2], a[3]),
    }
    while True:
        rnd.shuffle(actors)
        if not any(step[a[0]](a) for a in actors):
            break
    stuck = [k for k in prog if ch[k] < nch]
    assert not stuck, f"deadlock: {len(stuck)} cores stuck, e.g. {stuck[:3]} (chunk {[ch[k] for k in stuck[:3]]})"
    full = frozenset(range(G))
    for m in range(nch):
        for p in range(G):
            for c in cores:
                if plan.count(*c, p) > 0:
                    assert fin[(m, p) + (c,)] == full, (m, p, c)
    for st in ports.values():
        assert st["m"] == nch
    for k, v in zfw.items():  # (every credit forwarded; the counters at a call's count: re-armable)
        assert v == (nch - 1 if zneed[k] and plan.down(k[0], k[1]) is not None else 0), (k, v)
    for (p, c, d), n in ctr.items():
        assert n == nch * amax[(p, c, d)], (p, c, d, n)


CHUNK_CASES = [
    # (G, ring, GX, GY, MT, NTH, scatter, nv): 1024 rows = 2 chunks of GY 8 x MT 2 (the LoudBox line, the Galaxy ring)
    (4, False, 11, 8, 2, 224, -1, 2),
    (4, False, 11, 8, 2, 128, -2, 2),
    (4, True, 11, 8, 2, 224, -1, 2),
    (4, True, 11, 8, 2, 128, -2, 2),
    (2, False, 11, 8, 2, 128, -2, 2),
    (4, False, 11, 6, 2, 224, -1, 2),  # (768 rows = 2 chunks of GY 6 x MT 2: K3, the device's PCC loss on engine row 3)
    (4, False, 11, 6, 2, 128, -2, 2),
    (8, True, 6, 4, 2, 64, -1, 2),  # (a ring of 8: small grid, the schedule's shape)
    (8, False, 6, 4, 2, 64, -2, 2),
]


@pytest.mark.parametrize("router", [False, True], ids=["", "rtr"])
@pytest.mark.parametrize("nch", [1, 2, 3])
@pytest.mark.parametrize(
    "case", CHUNK_CASES, ids=[f"G{c[0]}-{'ring' if c[1] else 'line'}-s{c[6]}" for c in CHUNK_CASES]
)
@pytest.mark.parametrize("seed", range(3))
def test_rs_protocol_chunks(case, nch, seed, router):
    G, ring, GX, GY, MT, NTH, sc, nv = case
    plan = RsPlan(G=G, ring=ring, GX=GX, GY=GY, MT=MT, NTH=NTH, scatter=sc, nv=nv, pwmax=8 // MT)
    _run_chunks(plan, seed, nch, router=router)


@pytest.mark.parametrize("what", ["zone", "finb"])
def test_rs_chunks_model_catches_missing_credit(what):
    """Without the zone credit (or the FINB credit) some interleaving overwrites a zone slot before it is read (or
    reads the next chunk's FINB): the model must catch it (not vacuous)."""
    # (a small ring of 4, rows scattered: cores with nothing landing run ahead of the slower ones - quick to expose)
    plan = RsPlan(G=4, ring=True, GX=6, GY=4, MT=2, NTH=64, scatter=-2, nv=2, pwmax=4)
    for seed in range(40):
        try:
            _run_chunks(plan, seed, 3, zone_credit=what != "zone", finb_credit=what != "finb")
        except AssertionError:
            return
    raise AssertionError(f"the model accepted the chunks without the {what} credit")


@pytest.mark.parametrize("nch", [1, 2])
def test_rs_chunks_model_catches_router_overlay(nch):
    """The router's buffers over the zones without the router-done credit (only the "started" fence): an upstream's
    chunk 0 lands in a core still in its router phase - the model must catch it (the device's first-call wrong rows)."""
    plan = RsPlan(G=4, ring=True, GX=6, GY=4, MT=2, NTH=64, scatter=-2, nv=2, pwmax=4)
    for seed in range(40):
        try:
            _run_chunks(plan, seed, nch, router=True, router_credit=False)
        except AssertionError as e:
            assert "router buffers" in str(e), e
            return
    raise AssertionError("the model accepted a zone write during the router phase")


@pytest.mark.parametrize("compact", [True, False], ids=["compact", "slots"])
@pytest.mark.parametrize("ring", [False, True], ids=["line", "ring"])
@pytest.mark.parametrize("seed", range(3))
def test_rs_protocol_zones(compact, ring, seed):
    """Compact arrival zones (the rows scatter's default) vs P-tile slots, one call and chunked: the ports' offsets are
    the cores' slots, inside the zone, never overlapping (K2.7's 640-row shape: H 7168, GY 10 x MT 2; GY 4 x MT 1)."""
    for GY, MT, nch in ((10, 2, 1), (4, 1, 3)):
        plan = RsPlan(G=4, ring=ring, GX=11, GY=GY, MT=MT, NTH=224, scatter=-2, nv=2, pwmax=8 // MT, compact=compact)
        if nch == 1:
            _run(plan, seed)
        else:
            _run_chunks(plan, seed, nch)


def test_rs_model_catches_bad_zone_offset():
    """A port whose prefix mask drops a block writes over another slot: the model must catch it."""
    plan = RsPlan(G=4, ring=False, GX=11, GY=10, MT=2, NTH=224, scatter=-2, nv=2, pwmax=4)
    orig = plan.prefix_mask
    plan.prefix_mask = lambda q, h, d, slot: orig(q, h, d, slot) & ~(1 << 3)
    try:
        _run(plan, 0)
    except AssertionError as err:
        assert "zone" in str(err)
        return
    raise AssertionError("the model accepted a wrong zone offset")

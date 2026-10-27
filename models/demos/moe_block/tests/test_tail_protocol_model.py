# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU model of the fused K3 tail's latent all-gather into the engine's L1 (tt/tail_engine.py): ports, fence, landed
words, several calls back to back, line and ring (the balanced ring of fabric_all_gather's _schedule).

Per chip: an engine (its owners' landed words: a word a (chunk position, K half, row tile, column) share - only ever
incremented; it bumps its call count at call start and takes chunk position c once every share of it is >= that
count - position 0 its own rows, tilized at call start) and a port a (direction, link) - link l carries K half l. A
port at call start sends a ready to the peer port that writes into its chip (when that one has anything to send),
waits for its own ready (if it sends), then sends its units in order (own block first, then relays): a share once it
landed here (its owner's word >= the port's call count), as packets into one FIFO a directed link (delivered in order
at random times), the share's last packet incrementing the receiver's landed word. A chip starts its next call once
its engine and every port finished the current one.

Checks under random interleavings: no deadlock; no packet lands in a chip before that chip started the packet's call
(the fence: its landing regions may still be in use by the previous call); every landing lands the block the
receiver's chunk position expects; every share lands exactly once a call."""

import random

import pytest
from ttnn.bringup.combine_rf.fabric_all_gather_py import _schedule

from models.demos.moe_block.tt.tail_engine import chunk_order, share_in_part, step_of

OPP = {"fwd": "bwd", "bwd": "fwd"}


def run_model(G, ring, calls, GX=3, RT=2, L=2, seed=0, max_steps=2_000_000, PAIRS=None):
    rnd = random.Random(seed)
    order = {p: chunk_order(p, G, ring) for p in range(G)}
    peer = {}
    sends = {}
    for p in range(G):
        fwd, bwd = _schedule(p, G, ring, balance=True)
        nx = (p + 1) % G if (ring or p < G - 1) else None
        pv = (p - 1) % G if (ring or p > 0) else None
        peer[(p, "fwd")], peer[(p, "bwd")] = nx, pv
        sends[(p, "fwd")] = fwd if nx is not None else []
        sends[(p, "bwd")] = bwd if pv is not None else []

    def needs_ready(p, d):  # (fabric_all_gather._needs_ready: the peer's opposite port sends to p)
        q = peer[(p, d)]
        return q is not None and bool(sends[(q, OPP[d])])

    PAIRS = PAIRS or RT  # (the engine's row pairs: row tile t = c RT + R lands at pair t % PAIRS, step t / PAIRS)
    shares = [(r, cx) for r in range(RT) for cx in range(GX)]
    slot = lambda c, r: step_of(c * RT + r, PAIRS)  # noqa: E731
    landed = {(p, slot(c, r), h, cx): 0 for p in range(G) for c in range(G) for h in range(L) for r, cx in shares}
    ready = {(p, d, l): 0 for p in range(G) for d in OPP for l in range(L)}
    fifo = {(p, d, l): [] for p in range(G) for d in OPP for l in range(L)}  # (p -> peer[p, d], link l)
    call = {p: 0 for p in range(G)}  # (the call a chip runs: 1, 2, ...; 0 before its first)
    procs = {}

    def engine(p):
        ep = call[p]
        for r, cx in shares:  # (my own rows tilized: chunk position 0)
            for h in range(L):
                landed[(p, slot(0, r), h, cx)] += 1
        for c in range(G):
            for h in range(L):
                for r, cx in shares:
                    while landed[(p, slot(c, r), h, cx)] < ep:
                        yield "wait"
        yield "done"

    def port(p, d, l):
        ep = call[p]
        q_ = peer[(p, d)]
        if needs_ready(p, d):
            fifo[(p, d, l)].append(("ready",))
        if sends[(p, d)]:
            while ready[(p, d, l)] < 1:
                yield "wait"
            ready[(p, d, l)] -= 1
        for q, part in sends[(p, d)]:
            here, there = order[p].index(q), order[q_].index(q)
            for r, cx in shares:
                if not share_in_part(cx, GX, part):
                    continue
                while landed[(p, slot(here, r), l, cx)] < ep:
                    yield "wait"
                npk = rnd.choice((1, 2))  # (a share: one or two packets; the last increments)
                for i in range(npk):
                    fifo[(p, d, l)].append(("data", ep, q, there, l, r, cx, i == npk - 1))
                    yield "sent"
        yield "done"

    def start(p):
        call[p] += 1
        procs[p] = [engine(p)] + [port(p, d, l) for d in OPP for l in range(L)]

    for p in range(G):
        start(p)
    done = {p: False for p in range(G)}
    steps = 0
    while not all(done.values()):
        steps += 1
        assert steps < max_steps, "model: no progress (livelock)"
        acts = []
        for p in range(G):
            if not done[p]:
                acts += [("proc", p, i) for i, g in enumerate(procs[p]) if g is not None]
        acts += [("deliver", k) for k, f in fifo.items() if f]
        assert acts, "deadlock: nothing runnable, nothing in flight"
        a = rnd.choice(acts)
        if a[0] == "deliver":
            p, d, l = a[1]
            msg = fifo[a[1]].pop(0)
            q_ = peer[(p, d)]
            if msg[0] == "ready":
                ready[(q_, OPP[d], l)] += 1
                continue
            _, ep, q, there, h, r, cx, last = msg
            assert call[q_] == ep, f"fence: chip {q_} in call {call[q_]} got a packet of call {ep}"
            assert order[q_][there] == q, "a landing at the wrong chunk position"
            if last:
                landed[(q_, slot(there, r), h, cx)] += 1
            continue
        _, p, i = a
        try:
            v = next(procs[p][i])
        except StopIteration:
            v = "done"
        if v == "done":
            procs[p][i] = None
            if all(g is None for g in procs[p]):
                if call[p] == calls:
                    done[p] = True
                else:
                    start(p)
    assert all(not f for f in fifo.values()), "packets left in flight"
    for k, v in landed.items():
        assert v == calls, f"share {k} landed {v} times over {calls} calls"
    assert all(v == 0 for v in ready.values()), "readies left over"


@pytest.mark.parametrize("ring", [False, True], ids=["line", "ring"])
@pytest.mark.parametrize("G", [2, 3, 4, 8])
@pytest.mark.parametrize("rt", [(2, 2), (5, 5), (8, 5)], ids=["rt2", "640", "1024"])  # (RT, PAIRS)
@pytest.mark.parametrize("seed", range(4))
def test_tail_gather_model(G, ring, rt, seed):
    if ring and G < 3:
        pytest.skip("a 2-chip ring is a line")
    run_model(G, ring, calls=3, GX=2, RT=rt[0], PAIRS=rt[1], seed=seed)


def test_model_catches_a_missing_fence():
    """Without the ready fence a fast chip writes into a neighbour still in its previous call: the check trips."""
    hits = 0
    for seed in range(40):
        try:
            run_model_nofence(4, False, 3, seed)
        except AssertionError as e:
            hits += "fence" in str(e)
    assert hits > 0, "the model never caught an unfenced write"


def run_model_nofence(G, ring, calls, seed):
    """run_model with the ports' ready wait removed (the fence check must trip)."""
    import inspect
    import textwrap

    import models.demos.moe_block.tests.test_tail_protocol_model as m

    src = textwrap.dedent(inspect.getsource(m.run_model))
    src = src.replace(
        'while ready[(p, d, l)] < 1:\n                yield "wait"\n            ready[(p, d, l)] -= 1', "pass"
    )
    src = src.replace('assert all(v == 0 for v in ready.values()), "readies left over"', "pass")
    ns = dict(m.__dict__)
    exec(src.replace("def run_model(", "def _rm("), ns)
    ns["_rm"](G, ring, calls, seed=seed)


def run_engine_row(nst, nslot, GX=3, calls=3, seed=0, max_steps=3_000_000, credits=True):
    """One engine row pair of a chip (tt/tail_engine.py): GX cores a K half, partners (cx, h0) / (cx, h1), nst steps,
    NSLOT ring / E slots. Per core an NCRISC (shares: landed + the slot freed on the row (FREE) + its turn (the step's
    share count = cx) -> the row's step count +1; then per step: the row's shares in -> A to the compute; its E tile
    once that E slot is free on the row -> the row's E count +1; the row's E tiles in -> PART; its exchange to the
    partner; the partner's in -> XRECV; the previous step's output written -> FREE +1 on the row) and a compute (A ->
    E tile; PART -> XSEND; XRECV -> output; CB room as the kernels'). Shares land at random times. Checks: no deadlock,
    a ring / E slot never overwritten before every core of the row freed it, every count exact."""
    rnd = random.Random(seed)
    H = 2
    cores = [(cx, h) for h in range(H) for cx in range(GX)]
    sa = {(h, s): 0 for h in range(H) for s in range(nst)}  # (the row's shares of step s in, per core: equal)
    sp = {(h, s): 0 for h in range(H) for s in range(nst)}
    free = {h: 0 for h in range(H)}  # (FREE: steps done, the row's cores)
    sx = {c: 0 for c in cores}
    slot_owner = {(h, k): None for h in range(H) for k in range(nslot)}  # (the ring slot's live step)
    eslot_owner = {(h, k): None for h in range(H) for k in range(nslot)}
    cb = {c: dict(a=0, pmy=0, part=0, xsend=0, xrecv=0, out=0) for c in cores}
    landed_at = {}

    def ncrisc(c):
        cx, h = c
        partner = (cx, 1 - h)
        cs = 0

        def pump():
            nonlocal cs
            if (
                cs < nst
                and landed_at.get((c, cs), 0)
                and (not credits or cs < nslot or free[h] >= GX * (cs - nslot + 1))
            ):
                if sa[(h, cs)] == cx:  # (my turn: the step's shares so far)
                    k = cs % nslot
                    assert slot_owner[(h, k)] in (None, cs - nslot, cs), "ring slot overwritten while in use"
                    slot_owner[(h, k)] = cs
                    sa[(h, cs)] += 1
                    cs += 1

        for s in range(nst):
            while sa[(h, s)] < GX:
                pump()
                yield "wait"
            cb[c]["a"] += 1
            while cb[c]["pmy"] < 1:
                pump()
                yield "wait"
            while credits and s >= nslot and free[h] < GX * (s - nslot + 1):
                pump()
                yield "wait"
            k = s % nslot
            assert eslot_owner[(h, k)] in (None, s - nslot, s), "E slot overwritten while in use"
            eslot_owner[(h, k)] = s
            cb[c]["pmy"] -= 1
            sp[(h, s)] += 1
            while sp[(h, s)] < GX:
                pump()
                yield "wait"
            cb[c]["part"] += 1
            while cb[c]["xsend"] < 1:
                pump()
                yield "wait"
            cb[c]["xsend"] -= 1
            sx[partner] += 1
            while sx[c] < s + 1:
                pump()
                yield "wait"
            cb[c]["xrecv"] += 1
            if s > 0:
                while cb[c]["out"] < 1:
                    pump()
                    yield "wait"
                cb[c]["out"] -= 1
                free[h] += 1
        if nst:
            while cb[c]["out"] < 1:
                pump()
                yield "wait"
            cb[c]["out"] -= 1
            free[h] += 1
        while cs < nst:
            pump()
            yield "wait"

    def compute(c):
        for s in range(nst):
            while cb[c]["a"] < 1:
                yield "wait"
            while cb[c]["pmy"] >= 1:  # (PMY: one page)
                yield "wait"
            cb[c]["pmy"] += 1
            cb[c]["a"] -= 1  # (popped after the matmul)
            while cb[c]["part"] < 1 or cb[c]["xsend"] >= 2:
                yield "wait"
            cb[c]["part"] -= 1
            cb[c]["xsend"] += 1
            while cb[c]["xrecv"] < 1 or cb[c]["out"] >= 2:
                yield "wait"
            cb[c]["xrecv"] -= 1
            cb[c]["out"] += 1

    for call in range(calls):
        for k in list(sa):
            sa[k] = sp[k] = 0
        for h in range(H):
            free[h] = 0
        for c in cores:
            sx[c] = 0
            cb[c] = dict(a=0, pmy=0, part=0, xsend=0, xrecv=0, out=0)
        for k in slot_owner:
            slot_owner[k] = eslot_owner[k] = None
        landed_at.clear()
        pending = [(c, s) for c in cores for s in range(nst)]
        rnd.shuffle(pending)
        procs = [ncrisc(c) for c in cores] + [compute(c) for c in cores]
        live = list(range(len(procs)))
        steps = 0
        while live:
            steps += 1
            assert steps < max_steps, "engine row model: no progress (deadlock)"
            if pending and rnd.random() < 0.3:
                landed_at[pending.pop()] = 1
                continue
            i = rnd.choice(live)
            try:
                next(procs[i])
            except StopIteration:
                live.remove(i)
            if not live and pending:
                raise AssertionError("deadlock: the cores finished with shares still to land")
        assert all(v == GX for v in sa.values()) and all(v == GX for v in sp.values()), (sa, sp)
        assert all(v == GX * nst for v in free.values()), free


@pytest.mark.parametrize("shape", [(4, 4), (7, 4), (6, 4), (7, 3), (7, 2)], ids=lambda s: f"steps{s[0]}-slots{s[1]}")
@pytest.mark.parametrize("seed", range(5))
def test_tail_engine_row_model(shape, seed):
    """640 rows: 4 steps, 4 slots; 1024: 7 / 6 steps, 4 slots (and fewer: the credits carry it)."""
    run_engine_row(*shape, seed=seed)


def test_engine_row_model_catches_missing_credits():
    """Without the FREE waits a fast core overwrites a ring / E slot a slower one still reads: the check trips."""
    hits = 0
    for seed in range(40):
        try:
            run_engine_row(7, 2, seed=seed, credits=False)
        except AssertionError as e:
            hits += "overwritten" in str(e)
    assert hits > 0, "the model never caught a slot overwritten in use"

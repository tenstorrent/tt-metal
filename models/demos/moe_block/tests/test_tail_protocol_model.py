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

from models.demos.moe_block.tt.tail_engine import chunk_order, share_in_part

OPP = {"fwd": "bwd", "bwd": "fwd"}


def run_model(G, ring, calls, GX=3, RT=2, L=2, seed=0, max_steps=2_000_000):
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

    shares = [(r, cx) for r in range(RT) for cx in range(GX)]
    landed = {(p, c, h, r, cx): 0 for p in range(G) for c in range(G) for h in range(L) for r, cx in shares}
    ready = {(p, d, l): 0 for p in range(G) for d in OPP for l in range(L)}
    fifo = {(p, d, l): [] for p in range(G) for d in OPP for l in range(L)}  # (p -> peer[p, d], link l)
    call = {p: 0 for p in range(G)}  # (the call a chip runs: 1, 2, ...; 0 before its first)
    procs = {}

    def engine(p):
        ep = call[p]
        for r, cx in shares:  # (my own rows tilized: chunk position 0)
            for h in range(L):
                landed[(p, 0, h, r, cx)] += 1
        for c in range(G):
            for h in range(L):
                for r, cx in shares:
                    while landed[(p, c, h, r, cx)] < ep:
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
                while landed[(p, here, l, r, cx)] < ep:
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
                landed[(q_, there, h, r, cx)] += 1
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
@pytest.mark.parametrize("seed", range(6))
def test_tail_gather_model(G, ring, seed):
    if ring and G < 3:
        pytest.skip("a 2-chip ring is a line")
    run_model(G, ring, calls=3, seed=seed)


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

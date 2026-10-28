# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU protocol model of the fused head's x all-gather (tt/x_gather3.py), line and ring, 4 and 8 chips (host only).

Uses the real schedule (fabric_all_gather_py._schedule, balanced on a ring) and the real per-port bookkeeping
(x_gather3.port_plan: packet counts, upstream increment bases, expected arrivals) and replays the kernels' protocol:
  - reader (a port): chunk-major entries; its own shard from DRAM; a relay packet waits for the upstream's arrival
    count (m up_ipc + up_base + idx / INC + 1) and must find the data landed here; batches of GROUP into the CB
    (a chunk's entry end closes a batch);
  - sender: the call fence (ready to the neighbour's port facing it; wait its own), then the packets in order over a
    bounded in-order channel (fused write + arrival increment every INC-th packet of an entry and at its end), a
    watcher increment per entry, finally waits for every arrival from upstream and re-arms;
  - watcher: in the engine's order (x_gather3.shard_order), chunk m's shard of rank k is complete once each incoming
    stream carrying it counted it (a stream counts every entry it sends): asserts all of it landed, then XREADY + 1;
    re-arms;
  - calls: two calls back to back, a chip starts call 2 once its call 1 is done (chips drift apart); a packet of
    call 2 landing on a chip still in call 1 is a fence violation.
A random scheduler runs any enabled action; nothing enabled before everything finished = a deadlock. Checks:
delivery (every packet exactly once), the watcher's completeness, the fence, termination with every counter re-armed."""

import random
from collections import deque

import pytest
from ttnn.bringup.combine_rf.fabric_all_gather_py import _OPP, _schedule

from models.demos.moe_block.tt.x_gather3 import INC_EVERY, port_plan, shard_order


def model_chips(G, ring):
    """The planner's per-chip ring dicts for one mesh row of G chips (ring order = row order)."""
    chips = {}
    for p in range(G):
        fwd, bwd = _schedule(p, G, ring, balance=ring)
        chips[p] = dict(
            rings=[
                dict(
                    p=p,
                    next=(p + 1) % G if (ring or p < G - 1) else None,
                    prev=(p - 1) % G if (ring or p > 0) else None,
                    fwd=fwd,
                    bwd=bwd,
                )
            ]
        )
    return chips


def walk(m, q, part, v, *, CR, nv, sps, kb, rp):
    """The kernels' walk: (row tile R, step, first tile, idx in the link's full walk) of chunk m's shard q."""
    pps = -(-kb // rp)
    out, cnt = [], 0
    for lr in range(CR):
        if (lr // 2) % nv != v:
            continue
        if part == 0 or (lr & 1) == part - 1:
            for st in range(sps):
                for i, kk0 in enumerate(range(0, kb, rp)):
                    out.append((m * CR + lr, q * sps + st, kk0, cnt * sps * pps + st * pps + i))
        cnt += 1
    return out


class Sim:
    def __init__(self, G, ring, nv, CR, nch, sps, kb, rp, group, slots, seed, calls=2):
        self.G, self.nv, self.calls = G, nv, calls
        self.rng = random.Random(seed)
        self.chips = model_chips(G, ring)
        self.geo = dict(CR=CR, nv=nv, sps=sps, kb=kb, rp=rp)
        self.nch, self.group, self.slots = nch, group, slots
        self.plans = {
            (c, d, v): port_plan(self.chips, c, d, v, nch=nch, **self.geo)
            for c in range(G)
            for d in ("fwd", "bwd")
            for v in range(nv)
        }
        self.call = {c: 0 for c in range(G)}  # (the call each chip is in)
        self.landed = {c: set() for c in range(G)}  # (call, m, shard, R, step, kk0) in this chip's L1
        self.arrival = {k: 0 for k in self.plans}
        self.ready = {k: 0 for k in self.plans}
        self.watch = {k: 0 for k in self.plans}  # (on the chip receiving stream (d, v))
        self.xready = {c: 0 for c in range(G)}
        self.chan = {k: deque() for k in self.plans}  # (packets from port k toward its peer)
        self.violations = []
        for c in range(G):
            self._start_call(c)

    # ---- per-chip call state
    def _start_call(self, c):
        self.call[c] += 1
        call = self.call[c]
        for d in ("fwd", "bwd"):
            for v in range(self.nv):
                k = (c, d, v)
                pl = self.plans[k]
                pkts = []
                for m in range(self.nch):
                    for e, (q, part, ub, npk) in enumerate(pl["mine"]):
                        w = walk(m, q, part, v, **self.geo)
                        assert len(w) == npk, (k, m, e, len(w), npk)
                        for i, (R, st, kk0, idx) in enumerate(w):
                            pkts.append(
                                dict(
                                    m=m,
                                    e=e,
                                    q=q,
                                    R=R,
                                    st=st,
                                    kk0=kk0,
                                    idx=idx,
                                    ub=ub,
                                    i=i,
                                    npk=npk,
                                )
                            )
                        if npk == 0:  # (an entry with nothing on this link still counts at the watcher)
                            pkts.append(dict(m=m, e=e, q=q, empty=True, i=0, npk=0))
                peer = pl["peer"]
                sends_to_me = peer is not None and bool(self.chips[peer]["rings"][0][_OPP[d]])
                self.port = getattr(self, "port", {})
                self.port[k] = dict(
                    call=call,
                    rd=0,
                    cb=deque(),
                    batch=[],
                    pkts=pkts,
                    sd=0,
                    sent_in_entry=0,
                    fence_sent=not sends_to_me,
                    fence_ok=not pl["mine"],
                    done=False,
                    rearmed=False,
                )
        self.wm = getattr(self, "wm", {})
        self.wm[c] = 0  # (the watcher's next (chunk, rank) step)
        self.order = getattr(self, "order", {})
        self.order[c] = shard_order(self.chips, c, **self.geo)

    def chip_done(self, c):
        return all(self.port[(c, d, v)]["done"] for d in ("fwd", "bwd") for v in range(self.nv)) and self.wm[
            c
        ] == self.nch * (len(self.order[c][0]) - 1)

    # ---- actions
    def actions(self):
        acts = []
        for k, st in self.port.items():
            c, d, v = k
            pl = self.plans[k]
            # reader: next packet into the batch
            if st["rd"] < len(st["pkts"]) and len(st["cb"]) + len(st["batch"]) < 2 * self.group:
                p = st["pkts"][st["rd"]]
                if p["e"] == 0 or p.get("empty"):
                    acts.append(("read", k))
                else:
                    need = p["m"] * pl["up_ipc"] + p["ub"] + p["idx"] // INC_EVERY + 1
                    if self.arrival[k] >= need:
                        acts.append(("read", k))
            if not st["fence_sent"]:
                acts.append(("fence", k))
            if not st["fence_ok"] and self.ready[k] >= 1:
                acts.append(("fence_wait", k))
            if st["fence_ok"] and st["cb"] and len(self.chan[k]) < self.slots:
                acts.append(("send", k))
            if st["sd"] == len(st["pkts"]) and st["fence_sent"] and st["fence_ok"] and not st["done"]:
                if self.arrival[k] >= pl["expect_in"]:
                    acts.append(("finish", k))
            if self.chan[k]:
                acts.append(("deliver", k))
        for c in range(self.G):
            shards, carry = self.order[c]
            nk = len(shards) - 1
            if self.wm[c] < self.nch * nk:
                m, kk = divmod(self.wm[c], nk)
                if all(self.watch[(c, d, v)] >= m * n + j + 1 for d, v, j, n in carry[shards[1 + kk]]):
                    acts.append(("watch", c))
            if self.chip_done(c) and self.call[c] < self.calls:
                acts.append(("next_call", c))
        return acts

    def do(self, a):
        kind, k = a
        if kind == "read":
            st = self.port[k]
            c, d, v = k
            p = st["pkts"][st["rd"]]
            if p.get("empty"):  # (nothing to read: the entry's end only)
                st["cb"].extend(st["batch"] + [p])
                st["batch"] = []
                st["rd"] += 1
                return
            if p["e"] > 0:  # relay: the data must have landed here (this call)
                key = (st["call"], p["m"], p["q"], p["R"], p["st"], p["kk0"])
                if key not in self.landed[c]:
                    self.violations.append(f"relay read before landing {k} {key}")
            st["batch"].append(p)
            st["rd"] += 1
            if len(st["batch"]) == self.group or p["i"] == p["npk"] - 1:
                st["cb"].extend(st["batch"])
                st["batch"] = []
        elif kind == "fence":
            st = self.port[k]
            self.chan[k].append(dict(kind="ready", call=st["call"]))
            st["fence_sent"] = True
        elif kind == "fence_wait":
            self.ready[k] -= 1
            self.port[k]["fence_ok"] = True
        elif kind == "send":
            st = self.port[k]
            p = st["cb"].popleft()
            if p.get("empty"):
                self.chan[k].append(dict(kind="watch", call=st["call"]))
                st["sd"] += 1
                return
            st["sent_in_entry"] = 1 if p["i"] == 0 else st["sent_in_entry"] + 1
            inc = st["sent_in_entry"] % INC_EVERY == 0 or p["i"] == p["npk"] - 1
            self.chan[k].append(dict(kind="data", call=st["call"], p=p, inc=inc))
            if p["i"] == p["npk"] - 1:  # (the entry's last packet: its watcher increment follows it)
                self.chan[k].append(dict(kind="watch", call=st["call"]))
            st["sd"] += 1
        elif kind == "finish":
            st = self.port[k]
            self.arrival[k] -= self.plans[k]["expect_in"]
            st["done"] = True
        elif kind == "deliver":
            c, d, v = k
            peer = self.plans[k]["peer"]
            pk = self.chan[k].popleft()
            if pk["kind"] == "ready":  # (to the peer's port facing back: it sends to c)
                self.ready[(peer, _OPP[d], v)] += 1
            else:
                if self.call[peer] != pk["call"]:
                    self.violations.append(
                        f"{pk['kind']} of call {pk['call']} landed on chip {peer} in call {self.call[peer]}"
                    )
                if pk["kind"] == "data":
                    p = pk["p"]
                    key = (pk["call"], p["m"], p["q"], p["R"], p["st"], p["kk0"])
                    if key in self.landed[peer]:
                        self.violations.append(f"duplicate {key} on {peer}")
                    self.landed[peer].add(key)
                    if pk["inc"]:
                        self.arrival[(peer, d, v)] += 1
                else:
                    self.watch[(peer, d, v)] += 1
        elif kind == "watch":
            c = k
            shards, carry = self.order[c]
            nk = len(shards) - 1
            m, kk = divmod(self.wm[c], nk)
            q = shards[1 + kk]  # (shard q of chunk m: every row, step and piece must be here)
            for lr in range(self.geo["CR"]):
                for st in range(self.geo["sps"]):
                    for kk0 in range(0, self.geo["kb"], self.geo["rp"]):
                        key = (self.call[c], m, q, m * self.geo["CR"] + lr, q * self.geo["sps"] + st, kk0)
                        if key not in self.landed[c]:
                            self.violations.append(f"chip {c} chunk {m} shard {q} incomplete: missing {key}")
                            return
            self.wm[c] += 1
            self.xready[c] = self.wm[c]
            if self.wm[c] == self.nch * nk:  # (re-arm: each stream counted all its entries)
                for d in ("fwd", "bwd"):
                    for v in range(self.nv):
                        pl = self.plans[(c, d, v)]
                        if pl["has_up"]:
                            self.watch[(c, d, v)] -= self.nch * len(self.chips[pl["up"]]["rings"][0][d])
        elif kind == "next_call":
            self._start_call(k)

    def run(self, max_steps=10_000_000):
        for _ in range(max_steps):
            acts = self.actions()
            if not acts:
                break
            self.do(self.rng.choice(acts))
        done = all(self.chip_done(c) and self.call[c] == self.calls for c in range(self.G))
        return done


CASES = [
    # (G, ring, links, chunk row tiles, chunks, steps a shard, KB, RP)
    (4, False, 2, 10, 2, 2, 8, 4),
    (4, True, 2, 10, 2, 2, 8, 4),
    (4, True, 2, 20, 1, 2, 8, 4),
    (4, True, 1, 10, 2, 1, 4, 4),
    (8, False, 2, 10, 2, 1, 8, 4),
    (8, True, 2, 10, 2, 1, 8, 4),
    (8, True, 1, 6, 3, 1, 4, 4),
]


@pytest.mark.parametrize("G, ring, nv, CR, nch, sps, kb, rp", CASES)
@pytest.mark.parametrize("slots", [1, 4])
def test_head_gather_protocol(G, ring, nv, CR, nch, sps, kb, rp, slots):
    for seed in range(6):
        sim = Sim(G, ring, nv, CR, nch, sps, kb, rp, group=4, slots=slots, seed=seed)
        done = sim.run()
        assert not sim.violations, sim.violations[:5]
        assert done, f"deadlock (seed {seed}): " + str(
            {
                k: (s["rd"], s["sd"], len(s["pkts"]), s["fence_ok"], s["done"])
                for k, s in sim.port.items()
                if not s["done"]
            }
        )
        # termination: every counter re-armed, every channel drained
        assert all(v == 0 for v in sim.arrival.values()), sim.arrival
        assert all(v == 0 for v in sim.watch.values()), sim.watch
        assert all(v == 0 for v in sim.ready.values()), sim.ready
        assert all(not q for q in sim.chan.values())

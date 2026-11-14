# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MOE_FILL=piggyback (P.53): the expert replica copies set_replicas queues, carried by other layers' MoE calls.

A copy (chip src's home slot s of layer L -> chip dst's spare slot of L) is expert_slot_copy's run table s cut in
fabric-payload pieces (expert_slot_copy_chunk_bytes). The combine's port senders carry them
(ttnn.bringup.moe_block_main.set_combine_fill; routed_experts_combine's port writer, REC_FILL): chip src's port
facing the first hop toward dst (link 0) sends a piece only in a round that sent no row of its own, the call's rest
while the port drains. The host schedules the pieces, so progress needs no read-back:

- an entry (one set_replicas of one layer) waits one MoE call after it is queued (its unmap, a host write, and any
  superseded copy's pieces to the same slots land first: that call's start barrier is on the same routes);
- each MoE call carries ONE entry, the oldest whose layer is not the call's own (layer L's MoE never carries L's copy:
  it reads L's slots); a port's budget a call is MOE_FILL_PIECES pieces, split over its jobs (at most 4 a port);
- after the call that carried an entry's last pieces (each job's last piece ends with a flushed done increment on the
  destination), the entry is fenced; the next MoE call (any layer: its start barrier increments travel the same
  routes after the pieces) fences it, and the layer's slot map flips right after that call: a device copy of the map
  staged at set_replicas (no host transfer in the layer loop) into the router bias - a token never reaches an
  unfilled slot.
"""

from __future__ import annotations

import ttnn


def _pieces(table, chunk: int) -> int:
    return sum(-(-int(r[5]) // chunk) for r in table)


class FillEntry:
    """One set_replicas of one layer: jobs [(src, dst, table)] into every distinct weight set of the layer (its own and
    its views' with their own weights), then the layer's table `copies` mapped."""

    def __init__(self, owner, sets, jobs, totals, copies):
        self.owner = owner
        self.sets = sets  # [(w_gu, w_d, runs)]
        self.jobs = jobs
        self.totals = totals
        self.copies = copies
        self.set_i = 0
        self.done = [0] * len(jobs)  # (pieces carried of each job, current set)
        self.wait = 1  # (MoE calls to let pass before the first piece)
        self.fenced = False

    @property
    def layer(self) -> int:
        return self.owner.layer


class FillQueue:
    """The mesh's queue of FillEntry, and the per-call scheduling (before / after every MoE call)."""

    def __init__(self, mb, res, pieces: int, cap: int):
        self.mb, self.res = mb, res
        self.pieces, self.cap = int(pieces), int(cap)
        self.q: list[FillEntry] = []
        self.ports: dict = {}
        self.carried = None  # (entry, [(job index, p1)]) of the call in flight
        self.stats = {"calls": 0, "pieces": 0, "flips": 0}

    def port(self, src: int, dst: int) -> tuple:
        key = (src, dst)
        if key not in self.ports:
            self.ports[key] = (src, int(self.mb.combine_fill_port(self.res, src, dst)))
        return self.ports[key]

    def add(self, owner, sets, jobs, copies, chunk_bytes: int, tables) -> None:
        self.q = [e for e in self.q if e.owner is not owner]  # (a newer table supersedes an unfinished one)
        totals = [_pieces(tables[t], chunk_bytes) for _, _, t in jobs]
        self.q.append(FillEntry(owner, sets, jobs, totals, copies))

    def pending(self, owner=None) -> int:
        return sum(1 for e in self.q if owner is None or e.owner is owner)

    def before(self, layer: int) -> None:
        """Set the coming MoE call's jobs (layer: the call's own layer)."""
        e = next((e for e in self.q if not e.fenced and e.wait == 0 and e.layer != layer), None)
        if e is None:
            self.carried = None
            self.mb.set_combine_fill(self.res, True)
            return
        by_port: dict = {}
        for j, (src, dst, _) in enumerate(e.jobs):
            if e.done[j] < e.totals[j]:
                by_port.setdefault(self.port(src, dst), []).append(j)
        jobs, adv = [], []
        for js in by_port.values():
            js = js[: self.mb.COMBINE_FILL_MAX_JOBS]
            share = self.pieces if self.pieces > 0 else max(e.totals[j] for j in js)
            share = -(-share // len(js))
            for j in js:
                src, dst, t = e.jobs[j]
                p0 = e.done[j]
                p1 = min(e.totals[j], p0 + share)
                last = p1 == e.totals[j]
                jobs.append((src, dst, t, p0, p1, last))
                adv.append((j, p1))
        gu, d, runs = e.sets[e.set_i]
        self.mb.set_combine_fill(self.res, True, gu, d, runs, jobs, cap=self.cap)
        self.carried = (e, adv)
        self.stats["calls"] += 1
        self.stats["pieces"] += sum(p1 - p0 for _, _, _, p0, p1, _ in jobs)

    def after(self) -> None:
        """After the MoE call was enqueued: flip the entries an earlier call completed (this call fenced them), count
        this call's pieces, release the waits."""
        for e in [x for x in self.q if x.fenced]:
            e.owner.flip_replicas(e.copies)
            self.q.remove(e)
            self.stats["flips"] += 1
        if self.carried is not None:
            e, adv = self.carried
            for j, p1 in adv:
                e.done[j] = p1
            if all(dn >= t for dn, t in zip(e.done, e.totals)):
                e.set_i += 1
                e.done = [0] * len(e.jobs)
                e.fenced = e.set_i >= len(e.sets)
            self.carried = None
        for e in self.q:
            e.wait = max(0, e.wait - 1)
        self.mb.set_combine_fill(self.res, True)


def staged_map(mesh, like: ttnn.Tensor) -> ttnn.Tensor:
    """A device tensor of the router bias's spec (the slot map staged for a flip)."""
    return ttnn.allocate_tensor_on_device(like.spec, mesh)

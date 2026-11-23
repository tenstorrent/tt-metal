# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Engine glue of the capacity tiers (tail-park HY4_TAILPARK=1, host tier HY4_HOSTTIER=1; both default OFF, live files HY4_TAILPARK_FILE / HY4_HOSTTIER_FILE holding 0 / 1, read per call).

Pure python (numpy only): works on any engine-like object ``eng`` exposing ``ret`` (list of np.int32 chains or None), ``ret_t``, ``live``, ``pos``, ``pages``, ``pf_rows``, ``hit_len(r, toks)``, ``stash`` (dict, may be empty).
Hy4Engine (device) and the CPU fake engine of the tests / demo/scenario_capacity.py use the same class, so every policy path is unit tested without a device.

Calls made by the engine:
  plan(arrs, rows)          at the end of ``assign_rows``: records where a tail/host hit can be promoted and may move a request to the free row that holds its tail entry (cheapest, row-local copy).
  pre_prefill(prompts_by_row) at the top of ``prefill`` (after drain, before any xrow copy / hit computation): demote the chains the call will overwrite, promote hits, invalidate tails the new prompts need.
  note_round()              before every spec round: a decode that grows towards a tail invalidates it BEFORE it can write there.
"""
from __future__ import annotations

import time

import numpy as np

from models.demos.blackhole.hy4_preview.tt import hy4_chainstore as CS

PAGE = CS.PAGE


class Capacity:
    def __init__(self, eng, store, log=lambda s: None):
        self.eng, self.store, self.log = eng, store, log
        self.plan_by_row = {}  # row -> (entry id, h) from assign_rows
        self.pending = {}  # row -> tokens of the prompt being admitted (extent of rows in the current call)
        self.last = {}  # row -> dict(promoted=h, tier=..) of the last pre_prefill (usage / tests)
        store.extent_fn = self.extent
        store.live_fn = lambda r: bool(eng.live[r])

    # -- extent of a row: what its pages must keep
    def extent(self, r):
        e = self.eng
        a = e.ret[r]
        x = 0 if a is None else len(a)
        if e.live[r]:
            x = max(x, int(e.pos[r]) + 16)
        return max(x, self.pending.get(r, 0))

    def active(self):
        return self.store.active() and self.eng._pfx_on()

    # -- assign_rows hook
    def plan(self, arrs, rows):
        """arrs[i] = tokens of request i, rows[i] = the row chosen by the engine's policy. Returns rows (possibly moved) and fills ``plan_by_row``."""
        self.plan_by_row = {}
        if not self.active() or not self.store.entries:
            return list(rows)
        eng, out, taken = self.eng, list(rows), set(rows)
        for i, a in enumerate(arrs):
            e, h = self.store.match(a)
            r = out[i]
            if e is None or h <= 0:
                continue
            own = eng.hit_len(r, a)
            # a free row that holds the entry itself is the cheapest destination (row-local copy): move the request there when it is free and not used by another request of this call
            if e.tier == "tail" and e.row != r and not eng.live[e.row] and e.row not in taken:
                own_e = eng.hit_len(e.row, a)
                if self.store.worth_promoting(e, h, max(own_e, 0), e.row) and own_e < h:
                    taken.discard(r)
                    taken.add(e.row)
                    out[i] = r = e.row
                    own = own_e
            if self.store.worth_promoting(e, h, own, r):
                self.plan_by_row[r] = (e.eid, h)
        return out

    # -- prefill hook
    def pre_prefill(self, prompts_by_row, skip_rows=()):
        """prompts_by_row {row: tokens}. skip_rows: rows that have an xrow copy planned (they get no promote)."""
        self.last = {}
        eng, st = self.eng, self.store
        if not self.active():
            self.plan_by_row = {}
            return
        arrs = {r: np.asarray(p, dtype=np.int32) for r, p in prompts_by_row.items()}
        self.pending = {r: len(a) for r, a in arrs.items()}
        self.plan_by_row = (
            {}
        )  # the row choice was made by assign_rows; the entries are re-matched here (they may have changed)
        try:
            # (1) which entry (if any) is promoted on each row; decided BEFORE anything moves
            todo = {}
            for r, a in arrs.items():
                if r in skip_rows:
                    continue
                ent, hh = st.match(a)
                if ent is not None and st.worth_promoting(ent, hh, eng.hit_len(r, a), r):
                    todo[r] = (ent, hh)
            # (2) demote the chains this call overwrites (the row's own chain is the source; its new extent is the prompt)
            for r, a in arrs.items():
                chain = eng.ret[r]
                if chain is None or r in getattr(eng, "stash", {}):
                    continue
                if st.should_demote(chain, a):
                    st.demote(r, chain, new_extent_src=len(a))
            # (3) promote (largest first is irrelevant: distinct destination rows)
            for r, (ent, hh) in todo.items():
                if st.entries.get(ent.eid) is not ent:
                    continue  # dropped / evicted meanwhile
                eng.ret[r] = None  # the destination pages change: nothing retained unless the copy completes
                if st.promote(ent, r, hh):
                    eng.ret[r] = np.asarray(ent.tokens[:hh], dtype=np.int32).copy()
                    eng.ret_t[r] = time.time()
                    self.last[r] = dict(promoted=hh, tier=ent.tier)
        finally:
            # (4) rows about to be prefilled use pages [0, len(prompt)): tail entries below that are gone
            for r, a in arrs.items():
                st.invalidate_extent(r, len(a))
            self.pending = {}

    # -- round hook
    def note_round(self):
        eng = self.eng
        if not self.store.entries:
            return
        for r in range(len(eng.live)):
            if eng.live[r]:
                self.store.note_pos(r, int(eng.pos[r]))
            elif eng.ret[r] is not None:
                self.store.invalidate_extent(r, len(eng.ret[r]))  # a retained chain that grew (defensive)

    def reset(self):
        self.store.entries.clear()
        self.plan_by_row, self.pending, self.last = {}, {}, {}

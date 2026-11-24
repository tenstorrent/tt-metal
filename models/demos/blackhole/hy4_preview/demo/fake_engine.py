# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU fake of the Hy4 engine for the capacity scenarios (demo/scenario_capacity.py) and tests/test_hy4_capacity_cpu.py. No device.

``FakeCapEngine`` SUBCLASSES the real ``Hy4Engine`` without running its __init__ (object.__new__ pattern): the real ``assign_rows`` / ``_assign_rows_base`` / ``match_len`` / ``hit_len`` / ``_cap_pre_prefill`` /
``_pfx_on`` code runs, together with the real ChainStore + Capacity glue. Only the device parts are replaced:
  * the POOL is ``pool[row][page]`` = a tag (int) that a COLD prefill derives from the chained digest of the tokens up to that page (0 = never written, negative = a decode write): a hit that reads a wrong or stale
    page gives a different logits md5 / token stream than the cold run (this is what 'hit == cold' means here);
  * ``prefill`` mimics the real flow (cap pre-prefill, hits, lock-step start, writes pages from the start, retains the page-floored prompt), ``round`` emits one deterministic token per live row per round
    (a function of every pool page the row owns) and writes the decode page like the real decode, ``release`` parks the row.
  * ``FakeBackend`` moves tags (row-local / cross-row copy, to_host / from_host) with error injection (``fail_next``) and counts bytes.
"""
from __future__ import annotations

import hashlib

import numpy as np

from models.demos.blackhole.hy4_preview.tt import hy4_chainstore as CS
from models.demos.blackhole.hy4_preview.tt.hy4_blocks_store import digests_of
from models.demos.blackhole.hy4_preview.tt.hy4_capacity import Capacity
from models.demos.blackhole.hy4_preview.tt.hy4_serve_engine import ROWS, Hy4Engine

PAGE = 512


class FakeBackend(CS.Backend):
    def __init__(self, eng):
        self.eng = eng
        self.no_headroom = False  # can_copy reports no DRAM headroom for every cross-row copy
        self.fail_next = 0  # number of upcoming calls that raise
        self.skip_ok = 0  # that many calls succeed first (failure in the middle of a sliced transfer)
        self.calls = []
        self.pages_moved = 0

    def can_copy(self, src_row, dst_row, npages):
        if self.no_headroom and src_row != dst_row:
            return False, "injected: no headroom"
        return True, ""

    def _maybe_fail(self, what):
        if self.skip_ok > 0:
            self.skip_ok -= 1
            return
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError(f"injected {what} failure")

    def copy_pages(self, src_row, src_page0, dst_row, dst_page0, npages):
        self._maybe_fail("copy")
        P = self.eng.pool
        if src_row == dst_row and not (src_page0 + npages <= dst_page0 or dst_page0 + npages <= src_page0):
            raise ValueError("overlapping row-local copy")
        if min(src_page0, dst_page0) < 0 or max(src_page0, dst_page0) + npages > self.eng.pages:
            raise ValueError("page range outside the pool")
        P[dst_row][dst_page0 : dst_page0 + npages] = P[src_row][src_page0 : src_page0 + npages].copy()
        self.calls.append(("copy", src_row, src_page0, dst_row, dst_page0, npages))
        self.pages_moved += npages

    def to_host(self, row, page0, npages):
        self._maybe_fail("to_host")
        self.calls.append(("to_host", row, page0, npages))
        self.pages_moved += npages
        return self.eng.pool[row][page0 : page0 + npages].copy()

    def from_host(self, payload, row, page0, npages):
        self._maybe_fail("from_host")
        self.eng.pool[row][page0 : page0 + npages] = payload[:npages]
        self.calls.append(("from_host", row, page0, npages))
        self.pages_moved += npages


def tag(dig):
    return int(dig[:12], 16) + 1


class FakeRes:
    def __init__(self, logits):
        self.logits = logits


class FakeCapEngine(Hy4Engine):
    K = 3
    S = 4

    def __init__(
        self,
        pages=2056,
        pfx=True,
        tail=True,
        host=False,
        min_spill=2048,
        host_gb=1.0,
        costs=None,
        live_headroom=4096,
        slice_mb=0,
    ):
        self.pages, self.pfx, self.pf_rows, self.pfx_redo = pages, pfx, PAGE, True
        self.xrow, self.xrow_on, self.xrow_plan = None, False, {}
        self.blocks, self.blocks_on, self.stash_on, self.stash = None, False, False, {}
        self.loss_log, self.hist, self.hist_max = False, [], 0
        self.ret, self.ret_t, self.live = [None] * ROWS, [0.0] * ROWS, [False] * ROWS
        self.pos, self.last_cached = [0] * ROWS, {}
        self.pfx_stats = dict(calls=0, hit_calls=0, hit_tokens=0, prompt_tokens=0, computed_tokens=0)
        self.pool = [np.zeros(pages, dtype=np.int64) for _ in range(ROWS)]
        self.toks = {}
        self.round_no = 0
        self.tail_on, self.host_on, self.tail_stats = tail, host, {}
        self.backend = FakeBackend(self)
        store = CS.ChainStore(
            ROWS,
            pages,
            self.backend,
            lambda r: 0,
            costs=costs or CS.Costs(),
            tail=lambda: self.tail_on and CS.live_file_on("HY4_TAILPARK_FILE"),
            host=lambda: self.host_on and CS.live_file_on("HY4_HOSTTIER_FILE"),
            min_spill=min_spill,
            host_cap_tokens=int(host_gb * 1e9 / CS.BYTES_PER_TOKEN),
            live_headroom_tokens=live_headroom,
            slice_mb=slice_mb,
            yield_fn=lambda *a: self._cap_do_yield(*a),
            log=lambda s: None,
        )
        self.cap = Capacity(self, store)
        self.pp = _PP(self)
        self.prefill_log = []

    # ---- real-engine API
    def drain(self):
        pass

    def _logits(self, r, n):
        """what the lm_head would see: a function of every pool page of the prompt + the tail tokens (a wrong page changes it)"""
        h = hashlib.md5(np.asarray(self.pool[r][: (n + PAGE - 1) // PAGE], dtype=np.int64).tobytes())
        h.update(np.asarray(self.toks[r][(n // PAGE) * PAGE :], dtype=np.int64).tobytes())
        return h.hexdigest()

    def prefill(self, prompts_by_row, sampling=None, rids=None, _sub=False):
        rows = sorted(prompts_by_row)
        self.last_cached = {}
        A = self.pf_rows
        arrs = {r: np.asarray(prompts_by_row[r], dtype=np.int32) for r in rows}
        plan, self.xrow_plan = self.xrow_plan, {}
        if self.cap is not None and self.pfx:
            self._cap_pre_prefill(rows, prompts_by_row, plan)
        hits = {r: self.hit_len(r, arrs[r]) for r in rows}
        st = [(h - A if h > 0 else h) for h in hits.values()]
        start = max(0, min(st)) if st else 0
        self.last_cached = {r: min(hits[r], start) for r in rows}
        for r in rows:
            self.ret[r] = None
        logits = {}
        for r in rows:
            a = arrs[r]
            digs = digests_of(a)
            # pages below ``start`` are READ from the pool (the hit); pages from ``start`` are computed and written
            for p in range(len(digs)):
                if p >= start // PAGE:
                    self.pool[r][p] = tag(digs[p])
            if len(a) % PAGE:  # the partial last page (written by prefill, not retained)
                last = digs[-1] if digs else ""
                self.pool[r][len(digs)] = tag(
                    hashlib.md5((last + a[len(digs) * PAGE :].tobytes().hex()).encode()).hexdigest()
                )
            self.toks[r] = a
            logits[r] = self._logits(r, len(a))
        self.prefill_log.append(dict(rows=rows, start=start, hits=dict(hits), lens={r: len(arrs[r]) for r in rows}))
        for r in rows:
            n = (len(arrs[r]) // PAGE) * PAGE
            self.ret[r] = arrs[r][:n].copy() if n else None
            self.ret_t[r] = float(len(self.prefill_log))
            self.live[r] = True
            self.pos[r] = len(arrs[r])
        self.pfx_stats["calls"] += 1
        self.pfx_stats["prompt_tokens"] += sum(len(a) for a in arrs.values())
        self.pfx_stats["computed_tokens"] += sum(len(a) - start for a in arrs.values())
        self._lg = logits
        self.pp.prefill(None, keep_rows=rows, start=start)  # the demo's md5 tap wraps this
        return {r: int(hashlib.md5(logits[r].encode()).hexdigest()[:6], 16) for r in rows}

    def _tok(self, r):
        h = hashlib.md5(np.asarray(self.pool[r][: self.pos[r] // PAGE], dtype=np.int64).tobytes())
        h.update(str(self.pos[r]).encode())
        return int(h.hexdigest()[:6], 16)

    def round(self, caps, want_sampled, **k):
        if self.cap is not None:
            self.cap.note_round()
        out = []
        for r in range(ROWS):
            if self.live[r]:
                t = self._tok(r)
                self.pos[r] += 1
                self.pool[r][self.pos[r] // PAGE] = -(
                    1 + r
                )  # the decode writes its page (a stale tail below would be destroyed)
                out.append(dict(a=[t], m=0, pos=self.pos[r], drafts=[], tok=t))
            else:
                out.append(dict(a=[0], m=0, pos=self.pos[r], drafts=[], tok=0))
        self.round_no += 1
        return out

    def release(self, row):
        self.live[row] = False

    def advance(self, row, n):
        """n decode rounds of ``row`` only (others idle): used to grow a decode into a tail."""
        for _ in range(n):
            self.round([0] * ROWS, False)


class _T:
    def __init__(self, s):
        self.b = bytes.fromhex(s)

    def numpy(self):
        return np.frombuffer(self.b, dtype=np.uint8)


class _PP:
    """stand-in of Hy4PoolPrefill: returns the logits (md5 of what the last engine.prefill computed from the pool pages) so the demo's tap records lg_md5 as on the device."""

    def __init__(self, eng):
        self.eng = eng

    def prefill(self, prompts, keep_rows=None, start=0, **k):
        lg = [None] * ROWS
        for r in keep_rows or []:
            lg[r] = _T(self.eng._lg[r])
        return FakeRes(lg)

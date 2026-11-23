# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 CAPACITY tiers of the prefix cache (agent hy4cap, HY4_PORT_PLAN 3.1 A1 tail-park + A2 host tier). Pure python + numpy, no ttnn/torch: the engine glue (hy4_serve_engine) and the CPU tests share this file.

Problem (RESULT_HY4PFX / HY4_PORT_PLAN 1.2): every mesh row keeps ONE retained chain (``Hy4Engine.ret[r]``); a new prompt on that row overwrites it (cumulative loss 296,448 tokens in runs pfxdp and pfxstash).
The pool is allocated for max_model_len (1M tokens, ~2056 pages) per row while chains are <= ~460k, so the pages above the high-water mark of every row are idle (~2.1M tokens over 4 rows).

Three tiers (one ``ChainStore`` object):
  ROW   (tier 0, unchanged)  ``Hy4Engine.ret[r]``: the chain whose pool pages 0..n of row r are valid.
  TAIL  (tier 1, HY4_TAILPARK=1)  a chain copied into IDLE pool pages [page0, page0 + n) of ANY row (its own or another one): a page allocator per row over [floor(r), pages), floor(r) = pages needed by that
        row's retained chain / live request + lookahead. A request that needs the pages (admitted prompt, or a decode that grows into them: ``note_pos``) INVALIDATES the entry first (drop, never stale).
  HOST  (tier 2, HY4_HOSTTIER=1)  a chain read to host RAM (bytes in a backend-defined payload), capacity in tokens.
Keys = the chained sha256 digests of the tokens (hy4_blocks_store.digests_of): ``match`` walks the digests of the arriving prompt (longest stored depth), never compares token arrays.
Demote = at the moment a row's chain is about to be overwritten (chain >= min_spill tokens, not a prefix of the arriving prompt): choose the cheapest feasible destination by the tier cost functions
(``Costs``: spill + restore must beat the recompute), evicting lower-value entries (value = recompute seconds saved = tokens; ties oldest first) when full; a tail victim goes to host when that is on.
Promote = on a hit, before the suffix prefill: copy the pages back to row pages 0..h (same REDO / MTP-pair flow as every other hit: the engine then sees ``ret[r]`` = the promoted tokens).
Any backend exception = cold recompute (entry dropped, destination chain cleared first): never a wrong hit. The backend (device or fake pool) is the only thing that touches data.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from models.demos.blackhole.hy4_preview.tt.hy4_blocks_store import digests_of

PAGE = 512
PAGE_BYTES_CHIP = 3_750_912  # [C] bytes of one 512-token page on one chip
BYTES_PER_TOKEN = 58608  # [C] kv_geometry.py SCALED_FP8, one mesh row, 79 layers


# ------------------------------------------------------------------------------------------------ switches
def _file_flag(env_file, default=True):
    f = os.environ.get(env_file)
    if f:
        try:
            with open(f) as fh:
                return fh.read().strip() == "1"
        except (OSError, ValueError):
            pass
    return default


def live_file_on(env_file):
    """live switch file: True when unset / unreadable, else its content == 1."""
    return _file_flag(env_file)


def tail_enabled():
    """HY4_TAILPARK=1 and the live switch file HY4_TAILPARK_FILE (if set) holding 1."""
    return os.environ.get("HY4_TAILPARK", "0") == "1" and _file_flag("HY4_TAILPARK_FILE")


def host_enabled():
    """HY4_HOSTTIER=1 and the live switch file HY4_HOSTTIER_FILE (if set) holding 1."""
    return os.environ.get("HY4_HOSTTIER", "0") == "1" and _file_flag("HY4_HOSTTIER_FILE")


# ------------------------------------------------------------------------------------------------ costs
class Costs:
    """seconds per token of each path. Defaults: recompute 3.4 ms [M] (RESULT_HY4PFX / port plan 1.1); tail copy across rows 0.55 s / 65536 tokens [M/E] (hy4_xrow.copy_cost_s, B45x gate 3);
    tail copy inside a row [E] 1/3 of that (no all_gather); host path = BYTES_PER_TOKEN / bandwidth [E, E1 measures it].
    """

    def __init__(self, recompute=3.4e-3, xrow=0.55 / 65536, local=0.55 / 65536 / 3, host_bw=2e9, fixed=0.2):
        self.recompute, self.xrow, self.local, self.host_bw, self.fixed = recompute, xrow, local, host_bw, fixed

    @classmethod
    def from_env(cls):
        e = os.environ.get
        return cls(
            recompute=float(e("HY4_CAP_RECOMPUTE_S_PER_TOK", "3.4e-3")),
            xrow=float(e("HY4_CAP_XROW_S_PER_TOK", str(0.55 / 65536))),
            local=float(e("HY4_CAP_LOCAL_S_PER_TOK", str(0.55 / 65536 / 3))),
            host_bw=float(e("HY4_HOSTTIER_BW_GBPS", "2")) * 1e9,
        )

    def recompute_s(self, tokens):
        return tokens * self.recompute

    def move_s(self, tier, tokens, same_row):
        """one direction (spill == restore)."""
        if tier == "tail":
            return self.fixed + tokens * (self.local if same_row else self.xrow)
        return self.fixed + tokens * BYTES_PER_TOKEN / self.host_bw


class Entry:
    __slots__ = ("eid", "tokens", "digs", "tier", "row", "page0", "npages", "payload", "t", "hits")

    def __init__(self, eid, tokens, tier, row=None, page0=None, payload=None, t=0.0):
        self.eid, self.tokens, self.tier, self.row, self.page0, self.payload, self.t = (
            eid,
            tokens,
            tier,
            row,
            page0,
            payload,
            t,
        )
        self.digs = digests_of(tokens)
        self.npages = len(tokens) // PAGE
        self.hits = 0

    def __repr__(self):
        loc = f"row{self.row}@{self.page0}" if self.tier == "tail" else "host"
        return f"<chain#{self.eid} {len(self.tokens)}tok {self.tier} {loc}>"


class Backend:
    """data mover. All methods raise on any error (the store then recomputes cold). supports_cross_row False: only same-row tail placements are offered."""

    supports_cross_row = True

    def copy_pages(self, src_row, src_page0, dst_row, dst_page0, npages):
        raise NotImplementedError

    def to_host(self, row, page0, npages):
        raise NotImplementedError

    def from_host(self, payload, row, page0, npages):
        raise NotImplementedError


class ChainStore:
    def __init__(
        self,
        rows,
        pages,
        backend,
        extent_fn,
        costs=None,
        tail=True,
        host=False,
        min_spill=32768,
        host_cap_tokens=3_000_000,
        lookahead_tokens=1024,
        live_headroom_tokens=65536,
        min_gain=1024,
        clock=time.time,
        log=lambda s: None,
        live_fn=None,
        slice_mb=None,
        yield_fn=None,
    ):
        """extent_fn(r) -> tokens of row r that must stay untouched (retained chain / live position), live_fn(r) -> row r decodes (a tail on a live row keeps live_headroom_tokens above the live extent).
        tail / host: callables or bools (live switches)."""
        self.rows, self.pages, self.be, self.extent_fn = rows, int(pages), backend, extent_fn
        self.costs = costs or Costs()
        self._tail, self._host = tail, host
        self.min_spill, self.host_cap, self.look = min_spill, host_cap_tokens, lookahead_tokens
        self.live_head, self.min_gain = live_headroom_tokens, min_gain
        self.clock, self.log = clock, log
        self.risk_p0 = float(
            os.environ.get("HY4_CAP_RISK_P0", "0.3")
        )  # [E] invalidation risk of a tail with no slack; 0 = place by copy cost only
        self.risk_tokens = float(os.environ.get("HY4_CAP_RISK_TOKENS", "200000"))
        self.live_fn = live_fn or (lambda r: False)
        # HY4_HOSTTIER_SLICE_MB (per chip, default 0 = one whole-chain transfer = all rows' decode stalls for the transfer time): spill / restore in slices of that many MB per chip (3.75 MB per chip per page
        # [C]); ``yield_fn(kind, i, n_slices, nbytes)`` (set by the engine / plugin: run decode rounds) is called BETWEEN slices; the per-event log line reports bytes, ms, GB/s and the largest single stall.
        self.slice_mb = float(os.environ.get("HY4_HOSTTIER_SLICE_MB", "256")) if slice_mb is None else float(slice_mb)
        self.yield_fn = yield_fn
        self.xfers = []  # bounded log of transfer events (dicts)
        self.entries = {}  # eid -> Entry
        self._eid = 0
        self.stats = dict(
            demoted_tail=0,
            demoted_host=0,
            promoted_tail=0,
            promoted_host=0,
            invalidated=0,
            evicted_tail=0,
            evicted_host=0,
            dropped=0,
            failed=0,
            skipped_cost=0,
            skipped_dup=0,
            skipped_full=0,
            move_s=0.0,
            tokens_moved=0,
            host_xfer_bytes=0,
        )

    # ---- switches
    def tail_on(self):
        return bool(self._tail() if callable(self._tail) else self._tail)

    def host_on(self):
        return bool(self._host() if callable(self._host) else self._host)

    def active(self):
        return self.tail_on() or self.host_on()

    # ---- bookkeeping helpers
    def host_tokens(self):
        return sum(len(e.tokens) for e in self.entries.values() if e.tier == "host")

    def tail_tokens(self, row=None):
        return sum(len(e.tokens) for e in self.entries.values() if e.tier == "tail" and (row is None or e.row == row))

    def floor_pages(self, r, extent_tokens=None):
        """first page of row r that a tail entry may use."""
        ext = self.extent_fn(r) if extent_tokens is None else extent_tokens
        if self.live_fn(r):
            ext += self.live_head
        return -(-(int(ext) + self.look) // PAGE)

    def _free_gaps(self, r, floor):
        """free page intervals [(lo, hi)) of row r above ``floor``, highest first."""
        occ = sorted((e.page0, e.page0 + e.npages) for e in self.entries.values() if e.tier == "tail" and e.row == r)
        gaps, lo = [], floor
        for a, b in occ:
            if a > lo:
                gaps.append((lo, a))
            lo = max(lo, b)
        if lo < self.pages:
            gaps.append((lo, self.pages))
        return gaps[::-1]

    def _alloc(self, r, n, floor):
        """top-down first fit: the highest page0 of a gap of >= n pages (keeps the low pages, where the row grows, free)."""
        for lo, hi in self._free_gaps(r, floor):
            if hi - lo >= n:
                return hi - n
        return None

    def slice_pages(self):
        return 0 if self.slice_mb <= 0 else max(1, int(self.slice_mb * 1e6 // PAGE_BYTES_CHIP))

    def _slices(self, n):
        sp = self.slice_pages()
        if sp <= 0 or n <= sp:
            return [(0, n)]
        return [(k, min(sp, n - k)) for k in range(0, n, sp)]

    def _event(self, kind, row, tokens, parts_ms, nsl):
        tot = sum(parts_ms)
        nb = tokens * BYTES_PER_TOKEN
        ev = dict(
            kind=kind,
            tier="host",
            row=row,
            bytes=nb,
            ms=tot * 1e3,
            gbps=nb / max(tot, 1e-9) / 1e9,
            slices=nsl,
            max_stall_ms=max(parts_ms) * 1e3 if parts_ms else 0.0,
        )
        self.xfers.append(ev)
        del self.xfers[:-256]
        self.log(
            f"chainstore xfer: {kind} host row {row} {nb / 1e9:.2f} GB in {ev['ms']:.0f} ms = {ev['gbps']:.2f} GB/s row-aggregate, {nsl} slice(s), max single stall {ev['max_stall_ms']:.0f} ms (all rows' decode frozen for each slice; the yield between slices runs decode rounds)"
        )
        self.stats["host_xfer_bytes"] += nb
        return ev

    def _yield(self, kind, i, n, nbytes):
        if self.yield_fn is not None:
            self.yield_fn(kind, i, n, nbytes)

    def _spill(self, row, page0, npages, kind="spill"):
        """read npages of ``row`` to host in slices; payload = [(page offset, npages, backend payload)]. Raises on error."""
        parts, ms = [], []
        sl = self._slices(npages)
        for i, (k, n) in enumerate(sl):
            t0 = time.perf_counter()
            parts.append((k, n, self.be.to_host(row, page0 + k, n)))
            ms.append(time.perf_counter() - t0)
            if i + 1 < len(sl):
                self._yield("spill", i, len(sl), n * PAGE * BYTES_PER_TOKEN)
        self._event(kind, row, npages * PAGE, ms, len(sl))
        return parts

    def _restore(self, e, dst_row, npages):
        ms = []
        parts = [p for p in e.payload if p[0] < npages]
        for i, (k, n, pl) in enumerate(parts):
            n = min(n, npages - k)
            t0 = time.perf_counter()
            self.be.from_host(pl, dst_row, k, n)
            ms.append(time.perf_counter() - t0)
            if i + 1 < len(parts):
                self._yield("restore", i, len(parts), n * PAGE * BYTES_PER_TOKEN)
        self._event("restore", dst_row, npages * PAGE, ms, len(parts))

    def _drop(self, e, why):
        self.entries.pop(e.eid, None)
        self.stats["dropped"] += 1
        self.log(f"chainstore: drop {e} ({why})")

    # ---- invalidation (a request needs the pages)
    def invalidate_extent(self, r, tokens_needed):
        """a request will use pool pages [0, tokens_needed + lookahead) of row r: drop every tail entry of that row that starts below it. Returns the dropped entries."""
        lim = -(-(int(tokens_needed) + self.look) // PAGE)
        out = [e for e in list(self.entries.values()) if e.tier == "tail" and e.row == r and e.page0 < lim]
        for e in out:
            self.entries.pop(e.eid, None)
            self.stats["invalidated"] += 1
            self.log(f"chainstore: invalidate {e} (row {r} needs {tokens_needed} tokens)")
        return out

    def note_pos(self, r, pos, advance=1024):
        """decode of row r is at ``pos`` and may write up to pos + advance + lookahead before the next check: invalidate entries below that."""
        return self.invalidate_extent(r, int(pos) + advance)

    # ---- lookup
    def match(self, tokens, max_hit=None):
        """best entry for ``tokens``: (entry, h) with h = chunk aligned hit tokens (at least one token left to prefill), or (None, 0). Keys = chained digests."""
        digs = digests_of(tokens)
        cap = max(0, (len(tokens) - 1) // PAGE)
        best, bh = None, 0
        for e in self.entries.values():
            if (e.tier == "tail" and not self.tail_on()) or (e.tier == "host" and not self.host_on()):
                continue
            n, m = 0, min(len(digs), len(e.digs), cap)
            while n < m and digs[n] == e.digs[n]:
                n += 1
            h = n * PAGE
            if max_hit is not None:
                h = min(h, max_hit)
            if h > bh or (h == bh and best is not None and h > 0 and e.t > best.t):
                best, bh = e, h
        return best, bh

    def covers(self, tokens):
        """an existing entry already holds ``tokens`` (as a prefix of its chain)."""
        digs = digests_of(tokens)
        for e in self.entries.values():
            if len(e.digs) >= len(digs) and e.digs[: len(digs)] == digs:
                return e
        return None

    # ---- demote
    def should_demote(self, chain, prompt):
        """chain (np tokens of row r's retained prefix) would be overwritten by ``prompt``: long enough and the prompt does not extend it (prefix match of the whole chain)."""
        if chain is None or len(chain) < max(self.min_spill, PAGE) or not self.active():
            return False
        n = min(len(chain), len(prompt))
        ne = (np.asarray(chain[:n]) != np.asarray(prompt[:n])).nonzero()[0]
        m = int(ne[0]) if len(ne) else n
        return m < len(chain)

    def _candidates(self, src_row, n, new_extent_src, tokens):
        """feasible placements [(cost_s, tier, row, page0)] cheapest first. The source row's floor uses the extent of its NEW request."""
        out = []
        if self.tail_on():
            for r in range(self.rows):
                if r != src_row and not self.be.supports_cross_row:
                    continue
                fl = self.floor_pages(r, new_extent_src if r == src_row else None)
                p0 = self._alloc(r, n, fl)
                if p0 is None or (r == src_row and p0 < n):  # source pages [0, n) and destination must be disjoint
                    continue
                # risk of being invalidated before the next hit: a row whose floor is close to the entry (a long request or a growing decode would need the pages) loses the chain = a recompute
                slack = max(0, p0 - fl) * PAGE
                risk = self.risk_p0 * math.exp(-slack / self.risk_tokens) * (1.5 if self.live_fn(r) else 1.0)
                out.append(
                    (
                        self.costs.move_s("tail", tokens, r == src_row) + risk * self.costs.recompute_s(tokens),
                        "tail",
                        r,
                        p0,
                    )
                )
        if self.host_on() and tokens <= self.host_cap:
            out.append((self.costs.move_s("host", tokens, False), "host", None, None))
        out.sort(key=lambda x: (x[0], x[1] != "tail", x[2] if x[2] is not None else 99))
        return out

    def _evict_one(self, incoming_tokens, tier):
        """evict the least valuable entry of ``tier`` if it is worth less than the incoming chain; a tail victim moves to host when that is on. True when something was freed."""
        vs = [e for e in self.entries.values() if e.tier == tier]
        if not vs:
            return False
        v = min(vs, key=lambda e: (len(e.tokens), e.t))
        if len(v.tokens) >= incoming_tokens:
            return False
        self.entries.pop(v.eid)
        self.stats["evicted_" + tier] += 1
        self.log(f"chainstore: evict {v} for a {incoming_tokens}-token chain")
        if tier == "tail" and self.host_on() and len(v.tokens) <= self.host_cap:
            try:  # the victim's pages are still intact: read them to host (best effort)
                self._room_host(len(v.tokens))
                payload = self._spill(v.row, v.page0, v.npages, "evict-cascade")
                ne = Entry(self._next(), v.tokens, "host", payload=payload, t=v.t)
                self.entries[ne.eid] = ne
                self.stats["demoted_host"] += 1
            except Exception as ex:  # noqa: BLE001
                self.stats["failed"] += 1
                self.log(f"chainstore: cascade to host FAILED ({type(ex).__name__}: {str(ex)[:100]}): dropped")
        return True

    def _room_host(self, tokens):
        while self.host_tokens() + tokens > self.host_cap:
            vs = [e for e in self.entries.values() if e.tier == "host"]
            if not vs:
                break
            v = min(vs, key=lambda e: (len(e.tokens), e.t))
            self.entries.pop(v.eid)
            self.stats["evicted_host"] += 1

    def _next(self):
        self._eid += 1
        return self._eid

    def demote(self, src_row, chain, new_extent_src=0):
        """park ``chain`` (np tokens, valid in pool pages 0..n of src_row) in the tail / host before src_row's pages are overwritten. Returns the Entry or None (not parked: cold). Never raises."""
        n = len(chain) // PAGE
        chain = np.asarray(chain[: n * PAGE], dtype=np.int32)
        if n == 0 or not self.active():
            return None
        ex = self.covers(chain)
        if ex is not None:
            ex.t = self.clock()
            self.stats["skipped_dup"] += 1
            return ex
        rec = self.costs.recompute_s(len(chain))
        ok = lambda: [
            c for c in self._candidates(src_row, n, new_extent_src, len(chain)) if 2 * c[0] < rec * (1 + self.risk_p0)
        ]
        cands = ok()
        while not cands and (self._evict_one(len(chain), "tail") or self._evict_one(len(chain), "host")):
            cands = ok()
        if not cands:
            self.stats[
                "skipped_cost" if self._candidates(src_row, n, new_extent_src, len(chain)) else "skipped_full"
            ] += 1
            return None
        for cost, tier, r, p0 in cands:
            t0 = time.perf_counter()
            try:
                if tier == "tail":
                    self.be.copy_pages(src_row, 0, r, p0, n)
                    e = Entry(self._next(), chain, "tail", row=r, page0=p0, t=self.clock())
                else:
                    self._room_host(len(chain))
                    payload = self._spill(src_row, 0, n, "spill")
                    e = Entry(self._next(), chain, "host", payload=payload, t=self.clock())
            except Exception as ex_:  # noqa: BLE001
                self.stats["failed"] += 1
                self.log(
                    f"chainstore: demote to {tier} FAILED ({type(ex_).__name__}: {str(ex_)[:100]}): next candidate"
                )
                continue
            dt = time.perf_counter() - t0
            for o in [
                o for o in self.entries.values() if len(o.digs) <= len(e.digs) and e.digs[: len(o.digs)] == o.digs
            ]:
                self.entries.pop(o.eid, None)  # entries this chain extends are subsumed
            self.entries[e.eid] = e
            self.stats["demoted_" + tier] += 1
            self.stats["move_s"] += dt
            self.stats["tokens_moved"] += len(chain)
            self.log(
                f"chainstore: demoted {len(chain)} tokens of row {src_row} -> {e} in {dt * 1e3:.0f} ms (predicted {cost:.2f} s)"
            )
            return e
        return None

    # ---- promote
    def promote(self, e, dst_row, h):
        """copy the first h tokens (pages) of entry e into pool pages 0..h/512 of dst_row. True on success; on any error the entry is dropped and False returned (caller: cold; the caller cleared ret[dst] before)."""
        n = h // PAGE
        if h <= 0 or h % PAGE or n > e.npages:
            return False
        if e.tier == "tail" and e.row == dst_row and e.page0 < n:
            self._drop(e, "overlaps its destination")
            return False
        t0 = time.perf_counter()
        try:
            if e.tier == "tail":
                self.be.copy_pages(e.row, e.page0, dst_row, 0, n)
            else:
                self._restore(e, dst_row, n)
        except Exception as ex_:  # noqa: BLE001
            self.stats["failed"] += 1
            self._drop(e, f"promote failed {type(ex_).__name__}: {str(ex_)[:100]}")
            return False
        dt = time.perf_counter() - t0
        e.hits += 1
        e.t = self.clock()
        self.stats["promoted_" + e.tier] += 1
        self.stats["move_s"] += dt
        self.stats["tokens_moved"] += h
        self.log(f"chainstore: promoted {h} tokens of {e} -> row {dst_row} in {dt * 1e3:.0f} ms")
        return True

    def worth_promoting(self, e, h, own_hit, dst_row):
        gain = h - own_hit
        if gain < self.min_gain:
            return False
        return self.costs.move_s(e.tier, h, e.tier == "tail" and e.row == dst_row) < self.costs.recompute_s(gain)

    def snapshot(self):
        return dict(
            stats=dict(self.stats),
            tail_entries=sum(1 for e in self.entries.values() if e.tier == "tail"),
            host_entries=sum(1 for e in self.entries.values() if e.tier == "host"),
            tail_tokens=self.tail_tokens(),
            host_tokens=self.host_tokens(),
        )

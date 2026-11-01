# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU block index of the device-resident, block-paged prefix cache (APC_PAGED_DESIGN.md, milestone M1a). Pure Python + numpy: no ttnn, no torch.

What it owns (per mesh row): a refcounted pool of 128-token pages, the key-slot, halo-A and halo-B slot free lists, a hash index of 512-token chunk-blocks (chained sha256 over the KEY tokens of the
prefix), the LRU / leaf-first eviction, the reservation of a free user on the row that holds a hit, and the accounting of who holds what. The device side (kernels, stores) only ever sees the ids
it hands out (``INTERFACE_INDEX`` in ``APC_PAGED_INDEX_INTERFACE.md``).

Request life: ``plan_hit`` (scheduler thread: pins the hit path, reserves a user) -> ``bind`` (model thread, start of ``prefill_forward``) -> ``page_table`` -> ``begin_chunk`` / ``finish_chunk`` per 512 chunk
-> ``release``. Everything is guarded by one RLock (the scheduler thread and the model thread both call in); listeners are called after the lock is released.
"""
import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

CHUNK = 512
PAGE = 128


class OutOfPages(RuntimeError):
    """No free page on the row and nothing evictable."""


def _as_i64(tokens):
    a = np.asarray(tokens)
    return np.ascontiguousarray(a.reshape(-1).astype(np.int64, copy=False))


class Chain:
    """Incremental chained sha256 over a token array at every multiple of ``chunk``: ``digests[i]`` is the key of the prefix ``[0, chunk * (i + 1))``. Without a salt the digests equal
    ``PrefixStore._chain`` (hexdigest of one sha256 fed with the int64 bytes of every chunk). ``salt`` (mm_cache_salt / cache_salt bytes) is fed first.
    """

    def __init__(self, chunk=CHUNK, salt=b""):
        self.chunk, self.salt = int(chunk), bytes(salt or b"")
        self.h = hashlib.sha256()
        if self.salt:
            self.h.update(b"salt\x00" + self.salt)
        self.digests = []

    def extend(self, tokens, top):
        """feed the chunks of ``tokens`` (1-D, int64-convertible) up to ``top`` tokens (a multiple of chunk is used); -> self"""
        t = (
            tokens
            if isinstance(tokens, np.ndarray) and tokens.dtype == np.int64 and tokens.ndim == 1
            else _as_i64(tokens)
        )
        c = self.chunk
        top = min(int(top), t.shape[0]) // c * c
        while (len(self.digests) + 1) * c <= top:
            n = len(self.digests)
            self.h.update(np.ascontiguousarray(t[n * c : (n + 1) * c]).tobytes())
            self.digests.append(self.h.copy().hexdigest())
        return self


def chain_digests(tokens, top=None, chunk=CHUNK, salt=b""):
    t = _as_i64(tokens)
    return Chain(chunk, salt).extend(t, t.shape[0] if top is None else top).digests


def zone_start(prompt_len, zone, chunk=CHUNK):
    """p0: first chunk start of a prompt of ``prompt_len`` tokens that runs layers 21-39 (bounded replay ``zone`` = DSV41_PF_REPLAY)."""
    return max(0, (int(prompt_len) - int(zone)) // chunk * chunk)


@dataclass
class HitPlan:
    """A reservation (``plan_hit``) or a binding (``bind``). ``user`` is the GLOBAL physical user id (row * users_per_row + in-row index)."""

    hit_tokens: int = 0
    row: int = -1
    user: int = -1
    block_ids: tuple = ()
    page_ids: tuple = ()
    key_slots: tuple = ()
    halo_a: int = -1
    halo_b: int = -1
    xrow_src: int = (
        -1
    )  # cross-row hit (DSV41_APC_PAGED_XROW): the mesh row that HOLDS the blocks (block_ids / key_slots / halo_* name slots of THAT row, page_ids are fresh private pages of ``row``)

    @property
    def src_row(
        self,
    ):  # row-local cache (M1): the data lives on the row that runs the request; cross-row (M3): on ``xrow_src``
        return self.row if self.xrow_src < 0 else self.xrow_src


@dataclass
class ChunkPlan:
    req_id: object
    row: int
    block_idx: int
    key_slot: int = -1
    halo_a: int = -1
    halo_b: int = -1
    full: bool = False
    dup: bool = False  # the block exists already: this call can only add halo slots
    _blk: object = field(default=None, repr=False)
    _done: bool = field(default=False, repr=False)
    zone: int = 0  # p0 of the prompt this chunk belongs to (recorded with halo B)


class _Block:
    __slots__ = (
        "id",
        "row",
        "idx",
        "key",
        "parent",
        "pages",
        "key_slot",
        "halo_a",
        "halo_b",
        "b_zone",
        "refs",
        "children",
        "halo_pins",
        "state",
    )

    def __init__(self, bid, row, idx, key, parent, pages, key_slot):
        self.id, self.row, self.idx, self.key, self.parent = bid, row, idx, key, parent
        self.pages, self.key_slot = pages, key_slot
        self.halo_a = self.halo_b = -1
        self.b_zone = (
            1 << 60
        )  # zone start (p0) of the prompt whose replay wrote halo B: B is only valid for a new prompt with p0_new >= b_zone
        self.refs = self.children = self.halo_pins = 0
        self.state = "pending"  # pending -> live -> gone


class _Row:
    def __init__(self, pages, keys, ha, hb):
        self.free_pages = list(range(pages - 1, -1, -1))
        self.free_key = list(range(keys - 1, -1, -1))
        self.free_a = list(range(ha - 1, -1, -1))
        self.free_b = list(range(hb - 1, -1, -1))
        self.blocks = {}  # digest -> live block
        self.pending = {}  # digest -> block being computed
        self.lru = OrderedDict()  # block id -> block, live blocks, oldest first
        self.a_lru = OrderedDict()
        self.b_lru = OrderedDict()
        self.n_pages = pages


class _Req:
    def __init__(self, req_id, state, row, user, deadline):
        self.id, self.state, self.row, self.user, self.deadline = req_id, state, row, user, deadline
        self.plan = None
        self.held = []  # blocks this request holds a ref on
        self.halo_pins = []
        self.pages = []
        self.priv = set()
        self.cps = []
        self.chain = None
        self.prompt_len = 0
        self.xsrc = None  # cross-row hit: the donor blocks (also in ``held`` until ``xrow_finish``)


def _locked(fn):
    """hold the index lock for the call; listeners get the events after the lock is released."""

    def run(self, *a, **k):
        out = [None]
        err = [None]
        with self._lock:
            try:
                out[0] = fn(self, *a, **k)
            except BaseException as e:  # noqa: BLE001 - re-raised after the events are flushed
                err[0] = e
            ev, self._events = self._events, []
        for e in ev:
            for cb in list(self._listeners):
                try:
                    cb(e)
                except Exception:  # a broken listener must not break the cache
                    pass
        if err[0] is not None:
            raise err[0]
        return out[0]

    run.__name__ = fn.__name__
    run.__doc__ = fn.__doc__
    return run


class BlockIndex:
    def __init__(
        self,
        rows=4,
        users_per_row=2,
        pages_per_row=16386,
        chunk=CHUNK,
        page=PAGE,
        halo_a=128,
        halo_b=48,
        key_slots=2048,
        zone=2560,
        margin=128,
        signature="",
        ttl_s=60.0,
        clock=time.monotonic,
        free_users=None,
        halo_thin=1,
        halo_budget=0,
        strict_spans=False,
        strict_zone_equal=False,
        xrow=False,
        xrow_min=2048,
        xrow_max=65536,
    ):
        if chunk % page:
            raise ValueError(f"chunk {chunk} is not a multiple of the page {page}")
        self.rows, self.U, self.chunk, self.page = int(rows), int(users_per_row), int(chunk), int(page)
        self.ppc = self.chunk // self.page  # pages per chunk-block
        self.zone, self.margin, self.signature = int(zone), int(margin), signature
        self.ttl, self.clock, self._free_users_fn = float(ttl_s), clock, free_users
        self.halo_thin, self.strict_spans = max(1, int(halo_thin)), bool(strict_spans)
        self.halo_budget = int(
            halo_budget
        )  # > 0: at most about this many halo A records per prompt (thin = ceil(blocks / budget), the tip always): a long cold prompt must not wipe the halo store of its row
        self.strict_zone_equal = bool(
            strict_zone_equal
        )  # halo B valid only if b_zone == p0_new (exact), instead of b_zone <= p0_new
        self.caps = dict(pages=int(pages_per_row), key=int(key_slots), a=int(halo_a), b=int(halo_b))
        # cross-row hits (M3): a request whose best hit sits on a row without a free user is planned on a row that has one and the blocks are COPIED there (``xrow_*``); only when the copy gains at
        # least ``xrow_min`` tokens over the best local hit
        self.xrow, self.xrow_min, self.xrow_max = (
            bool(xrow),
            int(xrow_min),
            int(xrow_max),
        )  # (xrow_max: the host-staged copy is not worth it, and reads too many slots, beyond this many tokens)
        self._lock = threading.RLock()
        self._events = []
        self._listeners = []
        self._rows = [_Row(pages_per_row, key_slots, halo_a, halo_b) for _ in range(self.rows)]
        self._reqs = {}
        self._reserved = {}  # global user -> req id (planned)
        self._busy = {}  # global user -> req id (bound)
        self._lost = OrderedDict()
        self._bid = 0
        self.counters = dict(
            lookups=0, hits=0, hit_tokens=0, lost=0, binds=0, misses_bound=0, commits=0, dup_commits=0, upgrades=0,
            ev_block=0, ev_halo_a=0, ev_halo_b=0, out_of_pages=0, resets=0, expired=0, busy_loss=0, busy_loss_tokens=0, halo_loss=0, halo_loss_tokens=0,
            xrow_plans=0, xrow_tokens=0, xrow_done=0, xrow_failed=0, xrow_nopages=0,
        )  # fmt: skip

    # ---- listeners ---------------------------------------------------------------------------------------------------------------------------------------
    def add_listener(self, fn):
        self._listeners.append(fn)

    def _emit(self, kind, row, ids, why):
        self._events.append(dict(kind=kind, row=row, ids=list(ids), why=why))

    # ---- helpers (lock held) ----------------------------------------------------------------------------------------------------------------------------
    def _row_of(self, user):
        return int(user) // self.U

    def set_signature(self, sig):
        with self._lock:
            if sig == self.signature:
                return
            self.signature = sig
        self.reset()

    def _free_users(self, row, fin_users=()):
        """free users of ``row``. ``fin_users``: users whose request the scheduler already finished but the model has not released yet (the release rides on the next step's SchedulerOutput, which carries
        the new request too): they are free for a new reservation (the model releases them before it prefills the new request)
        """
        base = (
            list(self._free_users_fn(row))
            if self._free_users_fn is not None
            else list(range(row * self.U, (row + 1) * self.U))
        )
        base += [u for u in fin_users if u not in base]
        return [
            u
            for u in base
            if u not in self._reserved and (u not in self._busy or u in fin_users) and self._row_of(u) == row
        ]

    def _expire(self):
        now = self.clock()
        for rid in [r.id for r in self._reqs.values() if r.state == "planned" and r.deadline < now]:
            self._drop(rid)
            self._mark_lost(rid)
            self.counters["expired"] += 1

    def _mark_lost(self, rid):
        self._lost[rid] = True
        while len(self._lost) > 4096:
            self._lost.popitem(last=False)

    def _touch(self, row, blk, halos=True):
        """most recently used: the block, and (``halos``) its halo slots (only the boundary a hit resumes from is touched, as in the simulation)"""
        row.lru.move_to_end(blk.id)
        if halos and blk.halo_a >= 0:
            row.a_lru.move_to_end(blk.id)
        if halos and blk.halo_b >= 0:
            row.b_lru.move_to_end(blk.id)

    def _evict_block(self, ri, blk, why="lru"):
        row = self._rows[ri]
        row.blocks.pop(blk.key, None)
        row.lru.pop(blk.id, None)
        row.a_lru.pop(blk.id, None)
        row.b_lru.pop(blk.id, None)
        if blk.parent is not None:
            blk.parent.children -= 1
        row.free_key.append(blk.key_slot)
        if blk.halo_a >= 0:
            row.free_a.append(blk.halo_a)
        if blk.halo_b >= 0:
            row.free_b.append(blk.halo_b)
        blk.halo_a = blk.halo_b = -1
        row.free_pages.extend(reversed(blk.pages))
        blk.state = "gone"
        self.counters["ev_block"] += 1
        self._emit("block", ri, [blk.id], why)

    def _evict_lru_block(self, ri):
        row = self._rows[ri]
        for blk in row.lru.values():  # oldest first: the first leaf nobody holds
            if blk.refs == 0 and blk.children == 0:
                self._evict_block(ri, blk)
                return True
        return False

    def _alloc_pages(self, ri, n):
        row = self._rows[ri]
        while len(row.free_pages) < n:
            if not self._evict_lru_block(ri):
                self.counters["out_of_pages"] += 1
                raise OutOfPages(f"row {ri}: {n} pages wanted, {len(row.free_pages)} free, nothing evictable")
        return [row.free_pages.pop() for _ in range(n)]

    def _alloc_key(self, ri):
        row = self._rows[ri]
        while not row.free_key:
            if not self._evict_lru_block(ri):
                return -1
        return row.free_key.pop()

    def _drop_halo_b(self, ri, blk, why="lru"):
        row = self._rows[ri]
        row.b_lru.pop(blk.id, None)
        row.free_b.append(blk.halo_b)
        blk.halo_b = -1
        self.counters["ev_halo_b"] += 1
        self._emit("halo_b", ri, [blk.id], why)

    def _drop_halo_a(self, ri, blk, why="lru"):
        row = self._rows[ri]
        if blk.halo_b >= 0:
            self._drop_halo_b(ri, blk, why)
        row.a_lru.pop(blk.id, None)
        row.free_a.append(blk.halo_a)
        blk.halo_a = -1
        self.counters["ev_halo_a"] += 1
        self._emit("halo_a", ri, [blk.id], why)

    def _alloc_halo_a(self, ri):
        row = self._rows[ri]
        if not row.free_a:
            for blk in row.a_lru.values():
                if blk.halo_pins == 0:
                    self._drop_halo_a(ri, blk)
                    break
        return row.free_a.pop() if row.free_a else -1

    def _alloc_halo_b(self, ri):
        row = self._rows[ri]
        if not row.free_b:
            for blk in row.b_lru.values():
                if blk.halo_pins == 0:
                    self._drop_halo_b(ri, blk)
                    break
        return row.free_b.pop() if row.free_b else -1

    def _lookup_row(self, ri, digests, num_tokens, max_hit, spans=None):
        """-> (hit tokens, [blocks 1..hit/chunk]) on row ``ri`` (no pins)."""
        row = self._rows[ri]
        top = min(int(num_tokens) - self.margin, int(max_hit) if max_hit is not None else 1 << 60)
        top = max(top, 0) // self.chunk * self.chunk
        p0 = zone_start(num_tokens, self.zone, self.chunk)
        best, path, cur = 0, [], []
        self._depth = 0
        for i in range(1, top // self.chunk + 1):
            blk = row.blocks.get(digests[i - 1])
            if blk is None:
                break
            cur.append(blk)
            self._depth = len(cur)
            s = i * self.chunk
            if (
                blk.halo_a >= 0
                and (s <= p0 or (blk.halo_b >= 0 and self._b_ok(blk, p0)))
                and self._boundary_ok(s, spans)
            ):
                best, path = s, list(cur)
        return best, path

    def _b_ok(self, blk, p0):
        return blk.b_zone == p0 if self.strict_zone_equal else blk.b_zone <= p0

    def _boundary_ok(self, b, spans):
        if not (self.strict_spans and spans):
            return True
        return not any(st < b < st + n for st, n, *_ in spans)

    def _digests(self, key_tokens, top, salt, spans):
        t = _as_i64(key_tokens)
        return Chain(self.chunk, salt).extend(t, top).digests

    # ---- reservation ----------------------------------------------------------------------------------------------------------------------------------
    def _drop(self, rid):
        """forget a request completely (planned or bound): holds, pins, pages, user, pending chunks."""
        req = self._reqs.pop(rid, None)
        if req is None:
            return None
        for cp in req.cps:
            self._abort_cp(req, cp)
        for blk in req.halo_pins:
            blk.halo_pins -= 1
        for blk in req.held:
            blk.refs -= 1
            if blk.state == "dead" and blk.refs == 0:
                self._rows[blk.row].free_pages.extend(reversed(blk.pages))
                blk.state = "gone"
        row = self._rows[req.row] if req.row >= 0 else None
        if row is not None:
            row.free_pages.extend(sorted(req.priv, reverse=True))
        self._reserved.pop(req.user, None) if self._reserved.get(req.user) == rid else None
        self._busy.pop(req.user, None) if self._busy.get(req.user) == rid else None
        return req

    @_locked
    def plan_hit(self, req_id, key_tokens, num_tokens, max_hit=None, salt=b"", spans=None, finished_ids=()):
        """``finished_ids``: request ids the scheduler finished since the model was last told (their users count as free, see ``_free_users``)"""
        self._expire()
        fin_users = {r.user for r in (self._reqs.get(i) for i in finished_ids) if r is not None and r.state == "bound"}
        self._drop(req_id)
        self._lost.pop(req_id, None)
        self.counters["lookups"] += 1
        num_tokens = int(num_tokens)
        top = (
            max(min(num_tokens - self.margin, int(max_hit) if max_hit is not None else 1 << 60), 0)
            // self.chunk
            * self.chunk
        )
        if top <= 0 or not self.rows:
            return HitPlan()
        chain = Chain(self.chunk, salt).extend(_as_i64(key_tokens), top)
        digests = chain.digests
        best = (0, -1, None, None)  # hit, row, path, free users
        deep = (0, -1, None)  # the deepest hit of any row, free user or not (cross-row copy source)
        any_hit = (
            depth
        ) = 0  # diagnostics: the deepest hit of any row (ignoring free users) and the deepest block chain (ignoring halos)
        for ri in range(self.rows):
            free = self._free_users(ri, fin_users)
            hit, path = self._lookup_row(ri, digests, num_tokens, max_hit, spans)
            any_hit = max(any_hit, hit)
            if hit > deep[0]:
                deep = (hit, ri, path)
            depth = max(depth, self._depth * self.chunk)
            if not free:
                continue
            if hit > best[0] or (hit == best[0] and hit > 0 and len(free) > len(best[3])):
                best = (hit, ri, path, free)
        hit, ri, path, free = best
        if any_hit > hit:
            self.counters["busy_loss"] += 1
            self.counters["busy_loss_tokens"] += any_hit - hit
        if depth > any_hit:
            self.counters["halo_loss"] += 1
            self.counters["halo_loss_tokens"] += depth - any_hit
        if self.xrow and deep[0] - hit >= self.xrow_min and deep[0] <= self.xrow_max:
            xp = self._plan_xrow(req_id, chain, num_tokens, deep, ri, fin_users)
            if xp is not None:
                return xp
        if not hit:
            return HitPlan()
        row = self._rows[ri]
        user = min(free)
        req = _Req(req_id, "planned", ri, user, self.clock() + self.ttl)
        req.chain = chain
        req.prompt_len = num_tokens
        for blk in path:
            blk.refs += 1
            req.held.append(blk)
            self._touch(row, blk, halos=blk is path[-1])
        tip = path[-1]
        tip.halo_pins += 1
        req.halo_pins.append(tip)
        req.plan = HitPlan(
            hit_tokens=hit,
            row=ri,
            user=user,
            block_ids=tuple(b.id for b in path),
            page_ids=tuple(p for b in path for p in b.pages),
            key_slots=tuple(b.key_slot for b in path),
            halo_a=tip.halo_a,
            halo_b=tip.halo_b
            if hit > zone_start(num_tokens, self.zone, self.chunk)
            else -1,  # (lookup guarantees b_zone <= p0 then)
        )
        self._reserved[user] = req_id
        self._reqs[req_id] = req
        self.counters["hits"] += 1
        self.counters["hit_tokens"] += hit
        return req.plan

    def _plan_xrow(self, req_id, chain, num_tokens, deep, best_row, fin_users):
        """cross-row hit: the deepest hit (``deep``) is on row ``sri`` that has no free user (or a shallower one is local). Reserve a user on another row, allocate fresh private pages there for the whole hit
        and pin the donor blocks; the model copies the data (pages, key slots, tip halos) and calls ``xrow_finish``. -> HitPlan or None (no receiving row / no pages).
        """
        dhit, sri, spath = deep
        dst = best_row if best_row >= 0 and best_row != sri else -1
        if dst < 0:
            cands = [(len(self._free_users(r, fin_users)), r) for r in range(self.rows) if r != sri]
            cands = [c for c in cands if c[0] > 0]
            if not cands:
                return None
            dst = max(cands)[1]
        free = self._free_users(dst, fin_users)
        if not free:
            return None
        try:
            pages = self._alloc_pages(dst, dhit // self.page)
        except OutOfPages:
            self.counters["xrow_nopages"] += 1
            return None
        user = min(free)
        req = _Req(req_id, "planned", dst, user, self.clock() + self.ttl)
        req.chain, req.prompt_len = chain, num_tokens
        req.priv.update(pages)
        srow = self._rows[sri]
        for blk in spath:
            blk.refs += 1
            req.held.append(blk)
            self._touch(srow, blk, halos=blk is spath[-1])
        tip = spath[-1]
        tip.halo_pins += 1
        req.halo_pins.append(tip)
        req.xsrc = list(spath)
        req.plan = HitPlan(
            hit_tokens=dhit,
            row=dst,
            user=user,
            block_ids=tuple(b.id for b in spath),
            page_ids=tuple(pages),
            key_slots=tuple(b.key_slot for b in spath),
            halo_a=tip.halo_a,
            halo_b=tip.halo_b if dhit > zone_start(num_tokens, self.zone, self.chunk) else -1,
            xrow_src=sri,
        )
        self._reserved[user] = req_id
        self._reqs[req_id] = req
        self.counters["xrow_plans"] += 1
        self.counters["xrow_tokens"] += dhit
        return req.plan

    @_locked
    def xrow_source(self, req_id):
        """the donor blocks of a cross-row plan: [(block id, page ids, key slot, halo A slot, halo B slot, b_zone)] in chain order, or None"""
        req = self._reqs.get(req_id)
        if req is None or req.xsrc is None:
            return None
        return [(b.id, tuple(b.pages), b.key_slot, b.halo_a, b.halo_b, b.b_zone) for b in req.xsrc]

    @_locked
    def xrow_finish(self, req_id):
        """after the model copied the data and published every receiving block (``begin_chunk(copy=...)`` + ``finish_chunk``): unpin the donor, hold the receiving blocks and rewrite the plan to the
        receiving row's own slots (a normal local hit). -> that HitPlan, or None when a receiving block is missing (the caller drops the request: recompute from 0).
        """
        req = self._reqs.get(req_id)
        if req is None or req.xsrc is None:
            return None
        for blk in req.xsrc:
            blk.refs -= 1
            if blk in req.held:
                req.held.remove(blk)
            if blk.state == "dead" and blk.refs == 0:
                self._rows[blk.row].free_pages.extend(reversed(blk.pages))
                blk.state = "gone"
        for blk in req.halo_pins:
            blk.halo_pins -= 1
        req.halo_pins = []
        req.xsrc = None
        old = req.plan
        n = old.hit_tokens // self.chunk
        row = self._rows[req.row]
        path = [row.blocks.get(d) for d in req.chain.digests[:n]]
        if not path or any(b is None or b.state != "live" for b in path) or path[-1].halo_a < 0:
            self.counters["xrow_failed"] += 1
            return None
        for blk in path:
            if blk not in req.held:
                blk.refs += 1
                req.held.append(blk)
        self._touch(row, path[-1])
        for blk in path:
            self._touch(row, blk, halos=blk is path[-1])
        tip = path[-1]
        tip.halo_pins += 1
        req.halo_pins.append(tip)
        req.plan = HitPlan(
            hit_tokens=old.hit_tokens,
            row=req.row,
            user=req.user,
            block_ids=tuple(b.id for b in path),
            page_ids=old.page_ids,
            key_slots=tuple(b.key_slot for b in path),
            halo_a=tip.halo_a,
            halo_b=tip.halo_b if old.halo_b >= 0 else -1,
        )
        self.counters["xrow_done"] += 1
        return req.plan

    @_locked
    def cancel(self, req_id):
        req = self._reqs.get(req_id)
        if req is not None and req.state == "planned":
            self._drop(req_id)

    @_locked
    def cancel_user(self, user):
        """drop the UNBOUND reservation that holds ``user`` (a model-level caller needs the user for another request); the reserving request is counted as lost"""
        rid = self._reserved.get(int(user))
        if rid is not None:
            self._drop(rid)
            self._mark_lost(rid)

    @_locked
    def bind(self, req_id, start_pos, user=None, prompt_len=None):
        self._expire()
        start_pos = int(start_pos)
        req = self._reqs.get(req_id)
        if req is not None and req.state == "bound":
            return None  # a continuation (the model's tracker owns it) or a duplicate call
        if req is not None:  # planned
            plan = req.plan
            if start_pos == plan.hit_tokens and (user is None or int(user) == plan.user):
                req.state = "bound"
                self._reserved.pop(plan.user, None)
                self._busy[plan.user] = req_id
                req.pages = list(plan.page_ids)
                req.deadline = float("inf")
                if prompt_len is not None:
                    req.prompt_len = int(prompt_len)
                self.counters["binds"] += 1
                return plan
            self._drop(req_id)
            self._mark_lost(req_id)
            self.counters["lost"] += 1
            req = None
            if start_pos > 0:
                return None
        elif req_id in self._lost:
            self._lost.pop(req_id)
            self.counters["lost"] += 1
            if start_pos > 0:
                return None
        if start_pos != 0 or user is None:
            return None
        user = int(user)
        if user in self._reserved or user in self._busy or not 0 <= self._row_of(user) < self.rows:
            raise ValueError(f"user {user} is reserved, busy or outside the rows")
        req = _Req(req_id, "bound", self._row_of(user), user, float("inf"))
        req.plan = HitPlan(row=req.row, user=user)
        req.prompt_len = int(prompt_len or 0)
        self._reqs[req_id] = req
        self._busy[user] = req_id
        self.counters["binds"] += 1
        self.counters["misses_bound"] += 1
        return req.plan

    @_locked
    def restored(self, req_id):
        req = self._reqs.get(req_id)
        if req is not None:
            for blk in req.halo_pins:
                blk.halo_pins -= 1
            req.halo_pins = []

    # ---- pages ----------------------------------------------------------------------------------------------------------------------------------------
    @_locked
    def page_table(self, req_id, total_tokens):
        req = self._reqs.get(req_id)
        if req is None or req.state != "bound":
            raise KeyError(f"request {req_id!r} is not bound")
        n = -(-int(total_tokens) // self.page)
        if n > len(req.pages):
            new = self._alloc_pages(req.row, n - len(req.pages))
            req.pages.extend(new)
            req.priv.update(new)
        return list(req.pages)

    @_locked
    def rollback(self, req_id, total_tokens):
        req = self._reqs.get(req_id)
        if req is None:
            return
        n = -(-int(total_tokens) // self.page)
        while len(req.pages) > n and req.pages[-1] in req.priv:
            p = req.pages.pop()
            req.priv.discard(p)
            self._rows[req.row].free_pages.append(p)

    # ---- chunk commit ---------------------------------------------------------------------------------------------------------------------------------
    @_locked
    def begin_chunk(self, req_id, start, end, key_tokens, prompt_len, salt=b"", spans=None, copy=None):
        """``copy`` (cross-row copy of a donor block, see ``_plan_xrow``): (want halo A, want halo B, b_zone of the donor's halo B); the halo policy is replaced by it, an existing block is left alone"""
        req = self._reqs.get(req_id)
        if req is None or req.state != "bound":
            raise KeyError(f"request {req_id!r} is not bound")
        start, end, c = int(start), int(end), self.chunk
        if end % c or start != end - c or end > int(prompt_len) or not self._boundary_ok(end, spans):
            return None
        i = end // c
        if len(req.pages) < i * self.ppc:
            raise ValueError(
                f"page table of {req_id!r} has {len(req.pages)} pages, the chunk ending at {end} needs {i * self.ppc}"
            )
        if req.chain is None or req.chain.salt != bytes(salt or b""):
            req.chain = Chain(c, salt)
        req.chain.extend(_as_i64(key_tokens), end)
        key = req.chain.digests[i - 1]
        ri = req.row
        row = self._rows[ri]
        z_new = zone_start(prompt_len, self.zone, c)
        full = start >= z_new
        tip = end == int(prompt_len) // c * c
        thin = self.halo_thin
        if self.halo_budget > 0:
            thin = max(thin, -(-(int(prompt_len) // c) // self.halo_budget))
        want_halo = tip or (i % thin == 0)
        if copy is not None:
            want_halo = bool(copy[0])
        blk = row.blocks.get(key)
        if blk is not None and copy is not None:
            self._touch(row, blk, halos=False)
            # the receiving row already holds this block (e.g. the shared system prompt published by another request) but possibly WITHOUT the halo the donor's tip carries: add it
            # (the halo-only path below), else ``xrow_finish`` finds the tip without a halo and the request recomputes from 0
            if not (want_halo and (blk.halo_a < 0 or (copy[1] and blk.halo_b < 0))):
                return None
        if blk is not None:  # dup: only halos can be added
            self._touch(row, blk, halos=want_halo)
            self.counters["dup_commits"] += 1
            if not want_halo:
                return None
            ha = hb = -1
            if blk.halo_a < 0:
                ha = self._alloc_halo_a(ri)
                if ha < 0:
                    return None
            if copy is None and full and blk.halo_b >= 0 and blk.b_zone > z_new and blk.halo_pins == 0:
                self._drop_halo_b(
                    ri, blk, "zone"
                )  # B written by a prompt whose zone started later is valid for fewer prompts: replace it
            if (full if copy is None else bool(copy[1])) and blk.halo_b < 0 and (blk.halo_a >= 0 or ha >= 0):
                hb = self._alloc_halo_b(ri)
            if ha < 0 and hb < 0:
                return None
            blk.refs += 1  # (protects the block until finish/abort)
            req.held.append(blk)
            cp = ChunkPlan(req_id, ri, i, -1, ha, hb, full, True, blk, zone=z_new if copy is None else int(copy[2]))
            req.cps.append(cp)
            return cp
        if key in row.pending:
            return None
        parent = row.blocks.get(req.chain.digests[i - 2]) if i > 1 else None
        if i > 1 and parent is None:
            parent = row.pending.get(req.chain.digests[i - 2])
            if parent is None:
                return None
        ks = self._alloc_key(ri)
        if ks < 0:
            return None
        if parent is not None and parent.state not in ("live", "pending"):  # (the key allocation evicted it)
            row.free_key.append(ks)
            return None
        ha = self._alloc_halo_a(ri) if want_halo else -1
        if copy is None:
            hb = self._alloc_halo_b(ri) if (full and ha >= 0) else -1
        else:
            hb = self._alloc_halo_b(ri) if (copy[1] and ha >= 0) else -1
        self._bid += 1
        blk = _Block(self._bid, ri, i, key, parent, list(req.pages[(i - 1) * self.ppc : i * self.ppc]), ks)
        blk.refs = 1  # the creator
        blk.halo_a, blk.halo_b = ha, hb
        row.pending[key] = blk
        blk.b_zone = z_new if copy is None else int(copy[2])
        cp = ChunkPlan(req_id, ri, i, ks, ha, hb, full, False, blk, zone=z_new if copy is None else int(copy[2]))
        req.cps.append(cp)
        return cp

    def _abort_cp(self, req, cp):
        if cp._done:
            return
        cp._done = True
        row = self._rows[cp.row]
        blk = cp._blk
        if cp.dup:
            blk.refs -= 1
            if blk in req.held:
                req.held.remove(blk)
            if cp.halo_a >= 0:
                row.free_a.append(cp.halo_a)
            if cp.halo_b >= 0:
                row.free_b.append(cp.halo_b)
            return
        if row.pending.get(blk.key) is blk:
            del row.pending[blk.key]
        if blk.state == "pending":
            row.free_key.append(blk.key_slot)
            if blk.halo_a >= 0:
                row.free_a.append(blk.halo_a)
            if blk.halo_b >= 0:
                row.free_b.append(blk.halo_b)
            blk.halo_a = blk.halo_b = -1
            blk.state = "gone"

    @_locked
    def abort_chunk(self, cp):
        req = self._reqs.get(cp.req_id)
        if req is not None and cp in req.cps:
            req.cps.remove(cp)
            self._abort_cp(req, cp)

    @_locked
    def finish_chunk(self, cp):
        """publish the chunk. -> True when it is now visible to lookups (False: dropped: the same key was published first, the parent is gone, or the cache was reset)."""
        req = self._reqs.get(cp.req_id)
        if req is None or cp._done:
            return False
        req.cps.remove(cp)
        cp._done = True
        ri = cp.row
        row = self._rows[ri]
        blk = cp._blk
        if cp.dup:
            blk.refs -= 1
            req.held.remove(blk)
            if blk.state != "live":
                row.free_a.extend([cp.halo_a] if cp.halo_a >= 0 else [])
                row.free_b.extend([cp.halo_b] if cp.halo_b >= 0 else [])
                return False
            # (two requests may have planned a record for the same block while it had none: the first to finish attaches its slot, the other gives its slot back instead of overwriting,
            # which leaked the first one; a record a pending hit pinned is never replaced)
            if cp.halo_a >= 0:
                if blk.halo_a >= 0:
                    row.free_a.append(cp.halo_a)
                else:
                    blk.halo_a = cp.halo_a
                    row.a_lru[blk.id] = blk
            if cp.halo_b >= 0:
                if blk.halo_a < 0:  # (the block lost its A record meanwhile: B needs A)
                    row.free_b.append(cp.halo_b)
                elif blk.halo_b >= 0 and (blk.b_zone <= cp.zone or blk.halo_pins > 0):
                    row.free_b.append(cp.halo_b)  # (the attached record is at least as widely valid / is pinned)
                else:
                    if blk.halo_b >= 0:  # (replace a record written by a prompt whose zone started later)
                        row.b_lru.pop(blk.id, None)
                        row.free_b.append(blk.halo_b)
                    blk.halo_b = cp.halo_b
                    blk.b_zone = cp.zone
                    row.b_lru[blk.id] = blk
            self._touch(row, blk)
            self.counters["upgrades"] += 1
            return True
        ok = (
            blk.state == "pending"
            and row.blocks.get(blk.key) is None
            and (blk.parent is None or blk.parent.state == "live")
        )
        if row.pending.get(blk.key) is blk:
            del row.pending[blk.key]
        if not ok:
            if blk.state == "pending":
                row.free_key.append(blk.key_slot)
                if blk.halo_a >= 0:
                    row.free_a.append(blk.halo_a)
                if blk.halo_b >= 0:
                    row.free_b.append(blk.halo_b)
                blk.halo_a = blk.halo_b = -1
                blk.state = "gone"
            return False
        blk.state = "live"
        row.blocks[blk.key] = blk
        row.lru[blk.id] = blk
        if blk.parent is not None:
            blk.parent.children += 1
        if blk.halo_a >= 0:
            row.a_lru[blk.id] = blk
        if blk.halo_b >= 0:
            row.b_lru[blk.id] = blk
        for p in blk.pages:
            req.priv.discard(p)
        req.held.append(blk)  # (its ref was taken at begin_chunk)
        self.counters["commits"] += 1
        return True

    # ---- release / reset ------------------------------------------------------------------------------------------------------------------------------
    @_locked
    def release(self, req_id):
        self._drop(req_id)

    @_locked
    def release_user(self, user):
        rid = self._busy.get(int(user))
        if rid is not None:
            self._drop(rid)

    @_locked
    def reset(self):
        self.counters["resets"] += 1
        for rid in [r.id for r in self._reqs.values() if r.state == "planned"]:
            self._drop(rid)
            self._mark_lost(rid)
        for ri, row in enumerate(self._rows):
            gone = []
            for blk in list(row.lru.values()) + list(row.pending.values()):
                row.free_key.append(blk.key_slot)
                if blk.halo_a >= 0:
                    row.free_a.append(blk.halo_a)
                if blk.halo_b >= 0:
                    row.free_b.append(blk.halo_b)
                blk.halo_a = blk.halo_b = -1
                if blk.refs == 0 and blk.state == "live":
                    row.free_pages.extend(reversed(blk.pages))
                    blk.state = "gone"
                else:
                    blk.state = (
                        "dead" if blk.state == "live" else "gone"
                    )  # (pending: its pages are still the creator's private pages)
                gone.append(blk.id)
            row.blocks.clear()
            row.pending.clear()
            row.lru.clear()
            row.a_lru.clear()
            row.b_lru.clear()
            if gone:
                self._emit("block", ri, gone, "reset")

    # ---- introspection --------------------------------------------------------------------------------------------------------------------------------
    @_locked
    def peek(self, key_tokens, num_tokens, max_hit=None, salt=b""):
        """lookup without pins or user constraints -> (hit tokens, row) of the deepest hit (lowest row on a tie)"""
        num_tokens = int(num_tokens)
        top = max(min(num_tokens - self.margin, int(max_hit) if max_hit is not None else 1 << 60), 0)
        digests = Chain(self.chunk, salt).extend(_as_i64(key_tokens), top).digests
        best = (0, -1)
        for ri in range(self.rows):
            hit, _ = self._lookup_row(ri, digests, num_tokens, max_hit)
            if hit > best[0]:
                best = (hit, ri)
        return best

    @_locked
    def reserved_users(self, row):
        return sorted(u for u in self._reserved if self._row_of(u) == row)

    @_locked
    def reservation(self, req_id):
        """the HitPlan of an UNBOUND reservation of ``req_id`` (None: none / bound / lost)"""
        req = self._reqs.get(req_id)
        return req.plan if req is not None and req.state == "planned" else None

    @_locked
    def reserved_all(self):
        """set of the global users reserved for a hit (planned, not bound yet): a model that places other requests must not claim them"""
        return set(self._reserved)

    @_locked
    def free_users(self, row):
        return sorted(self._free_users(row))

    @_locked
    def stats(self):
        out = dict(self.counters)
        out["rows"] = [
            dict(
                pages_free=len(r.free_pages), key_free=len(r.free_key), halo_a_free=len(r.free_a), halo_b_free=len(r.free_b),
                blocks=len(r.blocks), pending=len(r.pending),
            )
            for r in self._rows
        ]  # fmt: skip
        out["requests"] = len(self._reqs)
        return out

    @_locked
    def check(self):
        """invariants (tests): every page / slot is in exactly one place, refcounts match the holders."""
        holders = {}
        for r in self._reqs.values():
            for b in r.held:
                holders[b.id] = holders.get(b.id, 0) + 1
        for ri, row in enumerate(self._rows):
            pages = list(row.free_pages)
            keys, a, b = list(row.free_key), list(row.free_a), list(row.free_b)
            seen_blocks = list(row.lru.values()) + list(row.pending.values())
            seen_ids = {x.id for x in seen_blocks}
            dead = {bl.id: bl for r in self._reqs.values() for bl in r.held if bl.row == ri and bl.state == "dead"}
            for blk in seen_blocks:
                if blk.state == "live":  # (a pending block's pages are still its creator's private pages)
                    pages += blk.pages
                keys.append(blk.key_slot)
                if blk.halo_a >= 0:
                    a.append(blk.halo_a)
                if blk.halo_b >= 0:
                    b.append(blk.halo_b)
                    assert blk.halo_a >= 0, "halo B without halo A"
            for blk in dead.values():
                pages += blk.pages
            for r in self._reqs.values():
                if r.row == ri:
                    pages += list(r.priv)
                for cp in r.cps:
                    if cp.row == ri and cp.dup:
                        if cp.halo_a >= 0:
                            a.append(cp.halo_a)
                        if cp.halo_b >= 0:
                            b.append(cp.halo_b)
            assert len(pages) == len(set(pages)), f"row {ri}: a page is double booked"
            assert len(pages) == row.n_pages, f"row {ri}: {len(pages)} pages accounted of {row.n_pages}"
            for name, ids, cap in (
                ("key", keys, self.caps["key"]),
                ("halo_a", a, self.caps["a"]),
                ("halo_b", b, self.caps["b"]),
            ):
                assert (
                    len(ids) == len(set(ids)) == cap
                ), f"row {ri}: {name} slots {len(ids)} unique {len(set(ids))} of {cap}"
            for blk in row.lru.values():
                assert blk.refs == holders.get(
                    blk.id, 0
                ), f"block {blk.id}: refs {blk.refs} vs holders {holders.get(blk.id, 0)}"
                kids = sum(1 for x in row.lru.values() if x.parent is blk)
                assert blk.children == kids, f"block {blk.id}: children {blk.children} vs {kids}"
            assert set(row.a_lru) <= seen_ids and set(row.b_lru) <= seen_ids
        return True


class ApcPagedHook:
    """The object the model registers with the plugin (``vllm_tt_plugin.prefix_hooks.set_prefix_hook``): the scheduler asks it for the hit of a new request (scheduler thread)."""

    def __init__(self, index, key_tokens_fn=None):
        self.index, self.key_tokens_fn = index, key_tokens_fn

    def plan_hit(self, request_id, tokens, num_tokens, max_hit=None, mm=None, salt=b"", finished_ids=()):
        """-> (hit_tokens, row). ``tokens``: the prompt token ids; ``mm``: opaque multimodal info for ``key_tokens_fn`` (images never reach this today)."""
        if self.key_tokens_fn is not None:
            kt = self.key_tokens_fn(tokens, mm)
            if kt is None:
                return 0, -1
        else:
            kt = tokens
        plan = self.index.plan_hit(request_id, kt, num_tokens, max_hit=max_hit, salt=salt, finished_ids=finished_ids)
        return plan.hit_tokens, plan.row

    def cancel(self, request_id):
        self.index.cancel(request_id)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview block index of the paged prefix cache and the pool admission policy (design task kv-design; CPU only: pure Python + numpy, no ttnn/torch).

``Hy4BlockIndex`` is the DeepSeek-V4.1 ``BlockIndex`` (tt/apc_paged.py of the DSV4 tree, reused as is: tt-reuse-first) with Hy4 constants. What changes and why (DESIGN.md s3):
  * page = chunk = 512 tokens (one page holds the 64-token slice of every column, see kv_geometry.py), so ``ppc`` = 1 and a published block IS one pool page;
  * no halo / tail record store: halo A is a bookkeeping flag with exactly one slot per page (never scarce), halo B does not exist (cap 0), ``zone`` = 0 makes the zone rule always true;
  * index keys live inside the page (22 key layers), so key slot i and pool page i are the same budget: ``key_slots`` = ``pages_per_row`` (the key free list can never run out before the page list);
  * hit rule: deepest boundary ``s <= e - 1`` on the 512 grid (``margin`` = 1; DSV4 needed 128 for its window replay); never a hit at s == 0;
  * cross-row copy has no 65536 cap (``xrow_max`` = 1M): the d2d rows-mode copy is cheaper than any recompute at every length.
``PoolAdmission`` is the oversubscribed-pool admission policy (DESIGN.md s4).
"""
import importlib.util
import os
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.environ.get("HY4_APC_BASE", os.path.join(_HERE, "..", "..", "deepseek_v41_flash", "tt", "apc_paged.py"))


def _load_base():
    spec = importlib.util.spec_from_file_location("dsv4_apc_paged_base", _BASE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AP = _load_base()
BlockIndex, HitPlan, ChunkPlan, OutOfPages, Chain, chain_digests = (
    AP.BlockIndex,
    AP.HitPlan,
    AP.ChunkPlan,
    AP.OutOfPages,
    AP.Chain,
    AP.chain_digests,
)
_locked = AP._locked

CHUNK = 512
PAGE = 512
MAX_CTX = 1 << 20
_FORBIDDEN = (
    "halo_a",
    "halo_b",
    "key_slots",
    "zone",
    "margin",
    "chunk",
    "page",
    "halo_thin",
    "halo_budget",
    "halo_spacing",
    "halo_tip",
)


class Hy4BlockIndex(BlockIndex):
    def __init__(self, rows=4, users_per_row=4, pages_per_row=2147, **kw):
        bad = [k for k in _FORBIDDEN if k in kw]
        if bad:
            raise TypeError(f"Hy4BlockIndex fixes {bad} (no halo store, page = chunk = 512)")
        kw.setdefault("xrow_max", MAX_CTX)
        kw.setdefault("xrow_min", 2048)
        super().__init__(
            rows=rows,
            users_per_row=users_per_row,
            pages_per_row=pages_per_row,
            chunk=CHUNK,
            page=PAGE,
            halo_a=pages_per_row,
            halo_b=0,
            key_slots=pages_per_row,
            zone=0,
            margin=1,
            halo_thin=1,
            **kw,
        )

    @_locked
    def begin_chunk(
        self, req_id, start, end, key_tokens, prompt_len, salt=b"", spans=None, copy=None, force_halo=False
    ):
        """as the DSV4 index; a cross-row copy chunk always publishes the block as a hit point (halo A is the 'resumable' flag of EVERY block here, there is no halo store): copy = (True, False, 0)"""
        if copy is not None:
            copy = (True, False, 0)
        return super().begin_chunk(
            req_id, start, end, key_tokens, prompt_len, salt=salt, spans=spans, copy=copy, force_halo=force_halo
        )

    # ---- pool accounting (admission) ----------------------------------------------------------------------------------------------------------------
    @_locked
    def free_pages(self, row):
        return len(self._rows[row].free_pages)

    @_locked
    def reclaimable_pages(self, row):
        """pages of published blocks nobody holds and no held block builds on (the leaf-first LRU can free all of them)"""
        r = self._rows[row]
        pinned = set()
        for blk in r.lru.values():
            b = blk
            while b is not None and blk.refs > 0 and b.id not in pinned:
                pinned.add(b.id)
                b = b.parent
        return sum(len(b.pages) for b in r.lru.values() if b.id not in pinned)

    @_locked
    def held_pages(self, req_id):
        req = self._reqs.get(req_id)
        return 0 if req is None else len(req.pages) if req.state == "bound" else len(req.plan.page_ids)

    @_locked
    def row_of(self, req_id):
        req = self._reqs.get(req_id)
        return -1 if req is None else req.row

    @_locked
    def alloc_pages_now(self, row, n):
        """reserve ``n`` private pages of ``row`` (evicting cached blocks leaf-first); used by the model for decode growth. OutOfPages if the row cannot"""
        return self._alloc_pages(row, n)


# ---- oversubscribed pool admission ---------------------------------------------------------------------------------------------------------------------
class PoolExhausted(RuntimeError):
    def __init__(self, row, victim):
        super().__init__(f"row {row}: pool exhausted, preempt {victim!r}")
        self.row, self.victim = row, victim


@dataclass
class Promise:
    row: int
    pages: int  # pages the request may still grow to (prompt + decode headroom), INCLUDING shared hit pages
    giant: bool


@dataclass
class PoolAdmission:
    """Per-row promise accounting on top of ``Hy4BlockIndex``.

    avail(row) = free pages + reclaimable cached pages - sum over requests on the row of (promised pages - pages held now).
    A request is admitted on a row iff its NEW pages (promise minus shared hit pages) <= avail(row): a promise is never broken, so decode growth inside the promise cannot fail and no running
    request is preempted while every request stays inside its promise. A request whose promise exceeds ``entitlement`` pages (a 'giant', default 262144 tokens = the B=16 guarantee) is admitted only
    on a row with enough room for all of it, so at the guaranteed points (users x ctx <= pool) nothing waits, and 1M users are resident only as many as the pool holds (one per row on an empty row).
    """

    index: Hy4BlockIndex
    entitlement_pages: int = 512  # 262144 tokens
    headroom_tokens: int = 8192  # decode growth promised beyond the prompt (capped by max_new_tokens)
    promises: dict = field(default_factory=dict)
    draining: set = field(
        default_factory=set
    )  # rows that admit nothing but giants' head-of-line request (anti-starvation, see start_drain)

    # ---- sizing ------------------------------------------------------------------------------------------------------------------------------------
    def total_pages(self, prompt_len, max_new):
        grow = min(int(max_new), self.headroom_tokens) if max_new is not None else self.headroom_tokens
        n = -(-(int(prompt_len) + max(grow, 1)) // CHUNK)
        return min(n, MAX_CTX // CHUNK)

    def outstanding(self, row):
        out = 0
        for rid, p in self.promises.items():
            if p.row == row:
                out += max(0, p.pages - self.index.held_pages(rid))
        return out

    def avail(self, row):
        return self.index.free_pages(row) + self.index.reclaimable_pages(row) - self.outstanding(row)

    # ---- admission ---------------------------------------------------------------------------------------------------------------------------------
    def can_admit(self, row, prompt_len, max_new, hit_tokens=0, protect_pages=0):
        """-> (ok, new_pages). ``protect_pages`` of the hit are reclaimable blocks that the hit itself pins (they stop being reclaimable once it is planned)"""
        total = self.total_pages(prompt_len, max_new)
        new = max(0, total - hit_tokens // CHUNK)
        return new <= self.avail(row) - protect_pages, new

    def admit(self, req_id, row, prompt_len, max_new, hit_tokens=0):
        """record the promise (call right after plan_hit / the model's own placement, before bind). -> True; False = does not fit (caller keeps the request waiting)"""
        total = self.total_pages(prompt_len, max_new)
        hit_pages = hit_tokens // CHUNK
        # a hit pins its blocks: those are no longer reclaimable (if they were), so count them against the room
        pinned_by_hit = min(hit_pages, self.index.reclaimable_pages(row))
        ok, _ = self.can_admit(row, prompt_len, max_new, hit_tokens, protect_pages=pinned_by_hit)
        if not ok:
            return False
        self.promises[req_id] = Promise(row, total, total > self.entitlement_pages)
        return True

    def pick_row(self, prompt_len, max_new, hit_tokens=0, hit_row=-1, rows_with_free_user=()):
        """row for a request: the hit row if it can take it, else the row with the most room among rows with a free user. -> row or -1 (wait). Draining rows only take giants."""
        total = self.total_pages(prompt_len, max_new)
        giant = total > self.entitlement_pages
        order = []
        if hit_row >= 0 and hit_row in rows_with_free_user:
            order.append(hit_row)
        order += sorted((r for r in rows_with_free_user if r != hit_row), key=lambda r: -self.avail(r))
        for r in order:
            if r in self.draining and not giant:
                continue
            h = hit_tokens if r == hit_row else 0
            if self.can_admit(r, prompt_len, max_new, h)[0]:
                return r
        return -1

    def start_drain(self, row):
        """a giant has waited too long: the row takes no new non-giant request until the giant is admitted (``end_drain``)"""
        self.draining.add(row)

    def end_drain(self, row):
        self.draining.discard(row)

    def release(self, req_id):
        self.promises.pop(req_id, None)

    # ---- decode growth ---------------------------------------------------------------------------------------------------------------------------
    def grow(self, req_id, total_tokens):
        """the model needs pages for ``total_tokens`` of ``req_id``. Inside the promise: always succeeds (pages are free or reclaimable by construction). Beyond it: extend the promise if the row has the room,
        else raise PoolExhausted(victim): the request on the row with the largest overshoot above the entitlement (a giant), the requester itself when nobody is above it (preempt = recompute with an APC hit).
        """
        p = self.promises[req_id]
        need = -(-int(total_tokens) // CHUNK)
        if need <= p.pages:
            return list(self.index.page_table(req_id, total_tokens))
        extra = need - p.pages
        if extra <= self.avail(p.row):
            p.pages = need
            p.giant = need > self.entitlement_pages
            return list(self.index.page_table(req_id, total_tokens))
        over = [
            (q.pages - self.entitlement_pages, rid)
            for rid, q in self.promises.items()
            if q.row == p.row and q.pages > self.entitlement_pages
        ]
        victim = max(over)[1] if over else req_id
        raise PoolExhausted(p.row, victim)

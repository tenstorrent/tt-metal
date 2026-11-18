# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 SHADOW block store (agent b45b, brief B45_3; CPU only: pure Python + numpy, no ttnn).

Why shadow and not a page-table indirection (DESIGN.md of b45/blocks): the pool is addressed by page position (identity) in more than three kernel families (indexer score, select/gather/id_filter chain, prefill
pf_id_filter/pf_stair/slice_copy, MTP pool). The cheap variant keeps every one of them untouched: each mesh row has the identity ACTIVE region (pages 0..P-1, today's pool, holds the chain being decoded /
the retained chain ``ret[row]``) plus a SHADOW region of ``shadow_pages`` extra pages that holds published blocks (prefix tree keyed by the chained sha256 digest of the tokens up to the block end).
  * publish (copy-out): before the active pages of the retained chain are overwritten by another prompt, its blocks that the shadow does not hold are copied active page i -> shadow page (``plan_out``);
  * hit (copy-in): a prompt whose prefix matches blocks of the shadow deeper than the active chain gets those blocks copied shadow -> active page i (``plan_hit``), then prefills from the hit boundary.
Both are page-granular row-local DRAM copies (``(src_page, dst_page)`` lists; the device op is a page-list copy over the 79 kv pools + 22 key pools). Sharing: a block is stored once per row however many chains use it.
With ``shadow_pages == 0`` the store reproduces the row-local ``ret[]`` semantics of hy4_serve_engine (hit_len = chunk-aligned match, at least one token left to prefill).
Policy: free list first; then leaf-first LRU over blocks that are neither pinned (in-flight copy) nor tip-protected (the last chain of each of the last ``tip_sessions`` sessions of the row); optional cost weight
(``depth_weight`` seconds-equivalent per block of depth: deep blocks are costlier to recompute, default 0 = plain LRU).
"""
import hashlib

import numpy as np

CHUNK = 512


def digests_of(tokens, salt=b"", chunk=CHUNK):
    """chained digests: d[i] keys the prefix [0, chunk*(i+1)) (same chain as apc_paged.chain_digests)"""
    t = np.ascontiguousarray(np.asarray(tokens).reshape(-1).astype(np.int64, copy=False))
    h = hashlib.sha256()
    if salt:
        h.update(b"salt\x00" + bytes(salt))
    out = []
    for i in range(t.shape[0] // chunk):
        h.update(np.ascontiguousarray(t[i * chunk : (i + 1) * chunk]).tobytes())
        out.append(h.copy().hexdigest())
    return out


class _Blk:
    __slots__ = ("digest", "parent", "page", "depth", "children", "last", "pins")

    def __init__(self, digest, parent, page, depth, last):
        self.digest, self.parent, self.page, self.depth, self.last = digest, parent, page, depth, last
        self.children = 0
        self.pins = 0


class HitPlan:
    """start: first token to prefill (chunk aligned); copy_in: [(shadow_page, active_page)] to run BEFORE the prefill; active_blocks / shadow_blocks: how many blocks of the hit come from where; pin: token for ``unpin``"""

    def __init__(self, start, copy_in, active_blocks, shadow_blocks, pin):
        self.start, self.copy_in, self.active_blocks, self.shadow_blocks, self.pin = (
            start,
            copy_in,
            active_blocks,
            shadow_blocks,
            pin,
        )


class ShadowStore:
    def __init__(
        self, rows=4, shadow_pages=0, chunk=CHUNK, tip_sessions=1, depth_weight=0.0, margin_tokens=1, shadow_base=0
    ):
        # shadow_base: first pool page of the shadow region (= users_per_row * pages_per_user: HY4_B8_ON32 puts user slot u at pages [u*ppu,(u+1)*ppu), the shadow after both)
        self.shadow_base = int(shadow_base)
        self.rows, self.shadow_pages, self.chunk = rows, int(shadow_pages), chunk
        self.tip_sessions, self.depth_weight, self.margin = tip_sessions, float(depth_weight), margin_tokens
        self._blk = [dict() for _ in range(rows)]  # per row: digest -> _Blk
        self._free = [list(range(self.shadow_pages - 1, -1, -1)) for _ in range(rows)]
        self._tips = [dict() for _ in range(rows)]  # per row: session -> leaf digest (insertion order = age)
        self._clock = 0
        self.stats = dict(published=0, evicted=0, skipped=0, hit_blocks_shadow=0, hit_blocks_active=0)

    # ---- queries
    def _tick(self):
        self._clock += 1
        return self._clock

    def shadow_chain(self, row, digs):
        """consecutive blocks of ``digs`` held by the shadow of ``row`` (list of _Blk)"""
        out = []
        for d in digs:
            b = self._blk[row].get(d)
            if b is None:
                break
            out.append(b)
        return out

    def used(self, row):
        return len(self._blk[row])

    def free_count(self, row):
        return len(self._free[row])

    def max_hit_blocks(self, n_tokens):
        """deepest boundary s <= n - margin on the chunk grid (never a hit at s == 0 is enforced by the caller: 0 blocks)"""
        return max(0, (int(n_tokens) - self.margin) // self.chunk)

    def plan_hit(self, row, tokens, resident=None, salt=b"", max_hit_blocks=None, active_base=0):
        """reusable prefix of ``tokens`` on ``row``. ``resident``: token array whose pool pages 0..len-1 are valid in the ACTIVE region of the row (the retained chain), or None.
        Active blocks are free (already in place); deeper shadow blocks are copied in. The hit is the deeper of the two (never both mixed except the active prefix + the shadow continuation).
        """
        digs = digests_of(tokens, salt, self.chunk)
        cap = self.max_hit_blocks(len(tokens))
        if max_hit_blocks is not None:
            cap = min(cap, max_hit_blocks)
        a = 0
        if resident is not None and len(resident):
            n = min(len(resident), cap * self.chunk, len(tokens))
            ne = (np.asarray(resident[:n]) != np.asarray(tokens[:n])).nonzero()[0]
            a = (int(ne[0]) if len(ne) else n) // self.chunk
        chain = self.shadow_chain(row, digs[:cap])
        m = len(chain)
        t = self._tick()
        copy = []
        pins = []
        if m > a:
            for i in range(a, m):  # blocks a..m-1 come from the shadow
                copy.append((self.shadow_base + chain[i].page, active_base + i))
                chain[i].pins += 1
                pins.append(chain[i])
            self.stats["hit_blocks_shadow"] += m - a
        else:
            m = a
        self.stats["hit_blocks_active"] += min(a, m)
        for b in chain:
            b.last = t
        return HitPlan(m * self.chunk, copy, min(a, m), max(0, m - a), pins)

    def unpin(self, plan):
        for b in plan.pin:
            b.pins -= 1
        plan.pin = []

    # ---- publish
    def _protected(self, row):
        prot = set()
        for leaf in self._tips[row].values():
            b = self._blk[row].get(leaf)
            while b is not None and b.digest not in prot:
                prot.add(b.digest)
                b = b.parent
        return prot

    def _victim(self, row, prot):
        best, bk = None, None
        for b in self._blk[row].values():
            if b.children or b.pins or b.digest in prot:
                continue
            k = b.last - self.depth_weight * b.depth
            if bk is None or k < bk:
                best, bk = b, k
        return best

    def _evict(self, row, b):
        del self._blk[row][b.digest]
        if b.parent is not None:
            b.parent.children -= 1
        self._free[row].append(b.page)
        self.stats["evicted"] += 1

    def _page(self, row, prot):
        if self._free[row]:
            return self._free[row].pop()
        v = self._victim(row, prot)
        if v is None:
            return None
        self._evict(row, v)
        return self._free[row].pop()

    def set_tip(self, row, session, tokens, salt=b""):
        """the last chain of ``session`` on ``row`` is protected from eviction (only the newest ``tip_sessions`` sessions are)"""
        digs = digests_of(tokens, salt, self.chunk)
        t = self._tips[row]
        t.pop(session, None)
        if digs:
            t[session] = digs[-1]
        while len(t) > self.tip_sessions:
            t.pop(next(iter(t)))

    def plan_out(self, row, resident, salt=b"", session=None, keep_from=0, active_base=0):
        """publish the blocks of the ACTIVE chain ``resident`` (about to be overwritten) that the shadow does not hold: [(active_page, shadow_page)] parent first. Stops at the first block that cannot get a page
        (blocks beyond it would have no parent). ``keep_from``: blocks below this index are not published (the new prompt keeps them in place, no overwrite). Run the copies BEFORE the overwrite.
        """
        if self.shadow_pages == 0 or resident is None or len(resident) < self.chunk:
            return []
        digs = digests_of(resident, salt, self.chunk)
        t = self._tick()
        prot = self._protected(row)
        # the chain being published is itself protected while we allocate (never evict own ancestors)
        parent, plan = None, []
        for i, d in enumerate(digs):
            b = self._blk[row].get(d)
            if b is not None:
                b.last = t
                parent = b
                prot.add(d)
                continue
            if i < keep_from:
                break  # shadow lacks an ancestor that stays active: cannot publish a rootless block
            pg = self._page(row, prot)
            if pg is None:
                self.stats["skipped"] += len(digs) - i
                break
            nb = _Blk(d, parent, pg, i + 1, t)
            if parent is not None:
                parent.children += 1
            self._blk[row][d] = nb
            prot.add(d)
            parent = nb
            plan.append((active_base + i, self.shadow_base + pg))
            self.stats["published"] += 1
        if session is not None:
            self.set_tip(row, session, resident, salt)
        return plan

    # ---- invariants
    def check(self):
        for r in range(self.rows):
            pages = [b.page for b in self._blk[r].values()] + self._free[r]
            assert sorted(pages) == list(range(self.shadow_pages)), "page leak / double use"
            for b in self._blk[r].values():
                assert b.parent is None or b.parent.digest in self._blk[r], "orphan block"
                kids = sum(1 for c in self._blk[r].values() if c.parent is b)
                assert kids == b.children

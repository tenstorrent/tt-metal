# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model wiring of the device-resident block-paged prefix cache (APC_PAGED_DESIGN.md, milestone M1b; switch ``DSV41_APC_PAGED=1``, default off).

Pieces (built by other lanes): ``tt/apc_paged.py`` (CPU ``BlockIndex``: hashes, refcounted pool pages, key / halo slots, LRU, reservations) and ``tt/apc_paged_store.py`` (device ``KeyStore`` /
``HaloStore`` + the three copy kernels). This module glues them into ``DSV41Model``:

* ``IndexAllocator``: the pool's per-row ``PageAllocator`` replaced by a view of the index. The index owns EVERY page of the row (but the 2 idle pages per user, see ``IDLE_PAGES``): a request's
  page table is ``[shared pages of the hit, private pages]``, a committed chunk block hands its 4 pages to the cache (no copy), release drops the refcounts.
* ``PagedApc``: build of the stores (once, before the first trace capture, + warm-up of the 3 kernel programs), request binding (hit / miss), the restore of a hit (key_load + halo restore, device
  local), the commit of every finished 512-token chunk (key_save + halo save, enqueued BEHIND the chunk's replay on CQ 0: the saves of chunk k precede the replay of chunk k+1 in the queue, so no
  save runs inside the trace and no trace layout changes), eviction through the index (a freed slot / page is reused only by commands enqueued later).

Hit rules (index): grid 512, ``s <= e - 128``, halo A at s, halo B valid if ``s <= p0(e)`` or ``b_zone <= p0(e)`` (``DSV41_APC_PAGED_ZONE_EQ=1``: ``==``). Only text prompts take part (image prompts use the
host snapshot path and never commit blocks). Needs: the single chunk trace, chunk 512, ``Up == U`` (one prefill slot per user), see ``supported``.
"""

import itertools
import os
import time

import numpy as np
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import apc_paged as AP
from models.demos.blackhole.deepseek_v41_flash.tt import apc_paged_store as APS
from models.demos.blackhole.deepseek_v41_flash.tt.kv_paged import PageAllocator

IDLE_PAGES = 2  # pages of an idle user (released / never used): the decode loop grows every user to ceil((0 + 1 + 128) / 128) = 2 pages
GRID = 512


def enabled():
    return APS.enabled()


def extra_pool_pages(users_per_row):
    """pages the pool needs on top of the per-user capacity: the idle pages of the users (they are outside the index)"""
    return IDLE_PAGES * int(users_per_row) if enabled() else 0


def supported(model, chunk):
    """-> (ok, why not). The wiring needs: the single chunk trace (one persistent key slab / halo set), the 512 grid as the model chunk, one prefill slot per user."""
    from models.demos.blackhole.deepseek_v41_flash.tt import prefill_single as PSG

    if not PSG.enabled():
        return False, "needs the single chunk trace (DSV41_PF_SINGLE=1)"
    if int(chunk) != GRID:
        return False, f"needs the chunk {GRID} (DSV41_VLLM_CHUNK), got {chunk}"
    if model.Up != model.U:
        return False, f"needs one prefill slot per user (Up {model.Up} != U {model.U})"
    return True, ""


class IndexAllocator(PageAllocator):
    """``PageAllocator`` interface (admit / grow / rollback / release / page_table / pages / length) of ONE mesh row on top of the shared ``BlockIndex``. User keys are the global user ids."""

    def __init__(self, pa, row, page_tokens=128, max_pages=None):
        self.pa, self.row = pa, row
        self.page_tokens = page_tokens
        self.max_pages = max_pages or pa.idx.caps["pages"]
        self.num_pages = pa.idx.caps["pages"] + IDLE_PAGES * pa.model.U
        self.pages, self.length = {}, {}

    def free_pages(self):
        return self.pa.idx.stats()["rows"][self.row]["pages_free"]

    def idle_pages(self, user, n):
        base = self.pa.idx.caps["pages"] + IDLE_PAGES * (int(user) % self.pa.model.U)
        return [base + i for i in range(n)]

    def admit(self, user, tokens, reserve_tokens=0):
        assert user not in self.pages, f"user {user} already admitted"
        self.pages[user], self.length[user] = [], 0
        try:
            self.grow(user, tokens, reserve_tokens)
        except MemoryError:
            self.release(user)
            raise
        return self.pages[user]

    def grow(self, user, tokens, reserve_tokens=0):
        need = self._need(tokens + reserve_tokens)
        if need > self.max_pages:
            raise MemoryError(f"user {user}: {need} pages exceed the per-user limit {self.max_pages}")
        have = self.pages[user]
        rid = self.pa.rid.get(user)
        if rid is None:
            if need <= IDLE_PAGES:  # an idle user (never prefilled / released): its two private pages outside the index
                if len(have) < need:
                    have[:] = self.idle_pages(user, need)
                self.length[user] = max(self.length[user], tokens)
                return have
            rid = self.pa.auto_bind(
                user, tokens
            )  # a prefill / test that did not name its request: an anonymous one (never commits blocks)
        if len(have) < need or (have and have[0] >= self.pa.idx.caps["pages"]):
            try:
                have[:] = self.pa.idx.page_table(rid, tokens + reserve_tokens)
            except AP.OutOfPages as e:
                raise MemoryError(str(e)) from e
        self.length[user] = max(self.length[user], tokens)
        return have

    def rollback(self, user, tokens, keep_reserve_tokens=0):
        assert tokens <= self.length[user]
        self.length[user] = tokens
        rid = self.pa.rid.get(user)
        if rid is None:
            return
        self.pa.idx.rollback(rid, tokens + keep_reserve_tokens)
        self.pages[user][:] = self.pa.idx.page_table(rid, 0)

    def release(self, user):
        self.pa.drop_user(user)
        self.pages.pop(user, None)
        self.length.pop(user, None)

    def adopt(self, user, pages):
        """a hit: the first pages of the table are the shared pages of the plan"""
        self.pages[user] = list(pages)
        self.length[user] = len(pages) * self.page_tokens


class PagedApc:
    def __init__(self, model, chunk=GRID, log=print):
        ok, why = supported(model, chunk)
        if not ok:
            raise RuntimeError(why)
        self.model, self.chunk, self.log = model, int(chunk), log
        rows, U = model.rows, model.U
        from models.demos.blackhole.deepseek_v41_flash.tt import prefill_model as PM

        self.zone = PM.replay_tokens() if PM.replay_tokens() > 0 else 1 << 40
        ha, hb, kb = APS.env_sizes()
        self.sizes = dict(halo_a=ha, halo_b=hb, key=kb)
        self.free_fn = None  # row -> iterable of free users (set by the serving adapter: users no request occupies)
        self.idx = AP.BlockIndex(
            rows=rows,
            users_per_row=U,
            pages_per_row=model.num_pages - IDLE_PAGES * U,
            chunk=GRID,
            page=128,
            halo_a=ha,
            halo_b=hb,
            key_slots=kb,
            zone=self.zone,
            margin=128,
            signature="",
            free_users=self._free_users,
            strict_zone_equal=os.environ.get("DSV41_APC_PAGED_ZONE_EQ", "0") == "1",
            halo_budget=int(os.environ.get("DSV41_APC_PAGED_HALO_BUDGET", "32")),
            xrow=os.environ.get("DSV41_APC_PAGED_XROW", "0")
            == "1",  # M3: cross-row copy of the blocks of a busy row (default off)
            xrow_min=int(os.environ.get("DSV41_APC_PAGED_XROW_MIN", "2048")),
            xrow_max=int(os.environ.get("DSV41_APC_PAGED_XROW_MAX", "65536")),
        )
        self.rid = {}  # user -> bound request id
        self.auto = set()  # users whose request is anonymous (no commits)
        self.noc = set()  # users whose request never commits (image prompt)
        self.plen = {}  # user -> total prompt length (zone / tip rule)
        self.ks = self.hs = None
        self.host_fallback = os.environ.get("DSV41_APC_PAGED_HOST", "0") == "1"
        self._auto_n = itertools.count()
        self.stats = dict(
            hits=0,
            hit_tokens=0,
            lost=0,
            restores=0,
            restore_s=0.0,
            commits=0,
            commit_s=0.0,
            saves=0,
            xrow=0,
            xrow_tokens=0,
            xrow_s=0.0,
            xrow_failed=0,
        )
        self.ktok = {}  # user -> (torch tensor id, numpy view) of the prompt tokens (key tokens of the chain)
        self.image_token_id = None
        self.hook = AP.ApcPagedHook(self.idx, self._key_tokens)

    # ---- plugin hook ------------------------------------------------------------------------------------------------------------------
    def _key_tokens(self, tokens, mm):
        t = np.asarray(tokens)
        if self.image_token_id is not None and (t == self.image_token_id).any():
            return None  # image prompts: the host snapshot path decides (and nothing is committed for them)
        return t

    def _free_users(self, row):
        if self.free_fn is not None:
            return list(self.free_fn(row))
        U = self.model.U
        return [u for u in range(row * U, (row + 1) * U) if u not in self.rid]

    def set_signature(self, sig):
        self.idx.set_signature(sig)

    def install(self):
        """replace the allocators of the pool (the model is idle: every user gets its idle pages)"""
        pool = self.model.pool
        pool.allocs = [IndexAllocator(self, r, max_pages=pool.max_pages) for r in range(pool.rows)]
        for b in range(pool.B):
            pool.admit(b, 1)
        pool.sync_page_table()

    # ---- stores (before the first trace capture) --------------------------------------------------------------------------------------
    def halos(self):
        d = {}
        for lid, pl in self.model.prefill_model.layers:
            if pl.pa.halo is not None:
                d[lid] = pl.pa.halo
        return d

    def owners(self):
        """({layer: ratio}, {layer: key slab tensor}) of the index-key owners of the single-trace slab"""
        ow, slabs = {}, {}
        for lid, pl in self.model.prefill_model.layers:
            pa = pl.pa
            sp = getattr(pa, "sparse", None)
            if sp is None or not getattr(sp, "dyn_on", False):
                continue
            ix = sp.indexer
            if ix is not None and sp.owner and ix.key_owner is None and (getattr(ix, "sh", None) or {}).get("single"):
                ow[lid], slabs[lid] = pa.ratio, ix.keys
        return ow, slabs

    def build(self):
        """allocate the stores and compile the kernels' programs; idempotent. Call after the chunk's persistent tensors exist (``compile_dyn``) and BEFORE the first trace capture."""
        if self.ks is not None:
            return
        pm = self.model.prefill_model
        assert pm.dyn is not None, "the stores are built after the prefill context was allocated (compile_dyn)"
        from models.demos.blackhole.deepseek_v41_flash.tt.prefill_dyn import memitem

        t0 = time.perf_counter()
        ow, slabs = self.owners()
        halos = self.halos()
        assert ow and set(ow) <= {2, 8, 14, 20}, (
            sorted(halos),
            ow,
        )  # (a model built with a subset of the layers keeps only those halos / owners)
        with memitem(self.model.md, "apc_paged_store"):
            self.ks = APS.KeyStore(self.model.md, self.model.U, ow, chunk=self.chunk, K=self.sizes["key"])
            self.hs = APS.HaloStore(
                self.model.md,
                self.model.U,
                halo_a=self.sizes["halo_a"],
                halo_b=self.sizes["halo_b"],
                layers=sorted(halos),
            )
        self.ks.warm(slabs)
        self.hs.warm(halos)
        ttnn.synchronize_device(self.model.md)
        sb = APS.store_bytes_per_chip(self.sizes["halo_a"], self.sizes["halo_b"], self.sizes["key"])
        mib = (sb["halo_a"] + sb["halo_b"] + self.sizes["key"] * sb["key_block_bytes"]) / 2**20
        self.log(
            f"APC paged: stores built (halo A {self.sizes['halo_a']} + B {self.sizes['halo_b']} slots, {self.sizes['key']} key blocks per mesh row = {mib:.0f} MiB per chip), kernels warmed in {time.perf_counter() - t0:.1f} s"
        )

    # ---- request binding --------------------------------------------------------------------------------------------------------------
    def auto_bind(self, user, tokens=0):
        rid = f"~auto{next(self._auto_n)}"
        self._bind(user, rid, 0, tokens, auto=True)
        return rid

    def _bind(self, user, rid, start, plen, auto=False, nocommit=False):
        try:
            self.idx.bind(rid, start, user=user, prompt_len=plen)
        except (
            ValueError
        ):  # the user is reserved for another request (a model-level caller that did not dodge it): that reservation is lost
            self.idx.cancel_user(user)
            self.idx.bind(rid, start, user=user, prompt_len=plen)
        self.rid[user] = rid
        self.plen[user] = int(plen or 0)
        if auto:
            self.auto.add(user)
        else:
            self.auto.discard(user)
        if nocommit:
            self.noc.add(user)
        else:
            self.noc.discard(user)

    def owns(self, user):
        """True for a text request this cache serves and feeds: the host snapshot tier then neither restores into it nor snapshots it. A host restore turns the request anonymous (``restore_users`` releases and
        re-admits the user), so its chunks are never committed and the paged cache starves (measured: all commits on the one row that computed cold); with ``DSV41_APC_PAGED_HOST=1`` the host tier keeps
        serving paged-managed requests too (index miss -> host snapshot), at that price. Image and anonymous users always use the host tier.
        """
        return not self.host_fallback and user in self.rid and user not in self.auto and user not in self.noc

    def drop_user(self, user):
        """the request of ``user`` is over (release_user / a new prompt of the user): refcounts and private pages go back to the index"""
        self.idx.release_user(user)
        rid = self.rid.pop(user, None)
        if rid is not None:
            self.idx.release(rid)
        self.auto.discard(user)
        self.noc.discard(user)
        self.plen.pop(user, None)
        self.ktok.pop(user, None)

    def bind_new(self, user, rid, plen, nocommit=False):
        """start of a new prompt at position 0 (a miss, or a lost reservation): the allocator's ``admit`` follows"""
        if rid is None:
            rid = f"~auto{next(self._auto_n)}"
            self._bind(user, rid, 0, plen, auto=True)
            return
        self.idx.release(rid)
        self._bind(user, rid, 0, plen, nocommit=nocommit)

    def sync_zone(self):
        """the bounded-replay length can be switched live (``DSV41_PF_REPLAY_FILE``): the zone rule of the index follows it, every record of the old rule is dropped"""
        from models.demos.blackhole.deepseek_v41_flash.tt import prefill_model as PM

        z = PM.replay_tokens() if PM.replay_tokens() > 0 else 1 << 40
        if z != self.zone:
            self.log(
                f"APC paged: bounded replay R {self.zone} -> {z}: the cache is reset (halo B records follow the zone)"
            )
            self.zone = z
            self.idx.zone = z
            self.idx.reset()

    def prepare_hits(self, items, rids, tots, rep=False, vis=None):
        """Items with ``start > 0`` and a request id that is not bound yet are prefix hits: bind the reservation to its user. -> (items with the lost hits reset to start 0, {user: HitPlan})."""
        out, hits = [], {}
        for b, t, st, e in items:
            rid = rids.get(b) if rids else None
            if (
                st > 0
                and rid is not None
                and self.rid.get(b) != rid
                and not (vis and b in vis)
                and self.idx.reservation(rid) is not None
            ):  # (a start > 0 without a reservation is a continuation of the model's own tracker / a request the index does not know: the legacy handling)
                self.model.pool.release(b)  # (the previous request of the user)
                tot = tots.get(b)
                plan = None
                if (
                    tot is not None or not rep
                ):  # (bounded replay needs the total length of the prompt: the zone of the hit)
                    try:
                        plan = self.idx.bind(rid, st, user=b, prompt_len=tot if tot is not None else e)
                    except ValueError:
                        plan = None
                    if (
                        plan is not None and plan.xrow_src >= 0
                    ):  # cross-row hit: copy the donor row's blocks into this row first
                        plan = self.xrow_copy(b, rid, plan, t, e)
                if plan is not None and plan.hit_tokens == st:
                    self.rid[b], self.plen[b] = rid, int(tot if tot is not None else e)
                    self.auto.discard(b)
                    self.noc.discard(b)
                    self.model.pool.allocs[b // self.model.U].adopt(b, plan.page_ids)
                    hits[b] = plan
                    self.stats["hits"] += 1
                    self.stats["hit_tokens"] += st
                    out.append((b, t, st, e))
                    continue
                self.stats["lost"] += 1
                self.idx.cancel(rid)
                self.log(
                    f"APC paged: request {rid} lost its reservation (hit {st} tokens): recomputing from position 0 (bind: {plan})"
                )
                out.append((b, t, 0, e))
                continue
            out.append((b, t, st, e))
        return out, hits

    # ---- hit restore --------------------------------------------------------------------------------------------------------------------
    def restore(self, hits):
        """enqueue the device-local copies of the hits (key_load, halo restore A / B); the shared pages are already in the page tables"""
        if not hits:
            return 0.0
        t0 = time.perf_counter()
        n = self.model.rows * self.model.U
        kl, ha, hb = [None] * n, [None] * n, [None] * n
        for b, plan in hits.items():
            kl[b] = list(plan.key_slots)
            ha[b] = plan.halo_a
            hb[b] = plan.halo_b if plan.halo_b >= 0 else None
        _, slabs = self.owners()
        halos = self.halos()
        self.ks.set_load(kl)
        self.hs.set_restore("A", ha)
        self.ks.load(slabs)
        self.hs.restore("A", halos)
        if any(x is not None for x in hb) and "B" in self.hs.store:
            self.hs.set_restore("B", hb)
            self.hs.restore("B", halos)
        for b in hits:
            self.idx.restored(self.rid[b])
        dt = time.perf_counter() - t0
        self.stats["restores"] += len(hits)
        self.stats["restore_s"] += dt
        return dt

    # ---- cross-row hit (M3, DSV41_APC_PAGED_XROW=1): host-staged copy of the blocks of a donor row into the row of the new request ---------------------------------------
    def warm_xrow(self, user=0, tokens=1024):
        """Compile-before-capture for the cross-row copy (``xrow_copy``): the same device ops on an idle ``user`` with zero / self-copied data, no index involved: the pool gather, the pool row scatter
        (written back with what was read: no state change), the key-store slice + read-back of slot 0 and the halo slice + read-back. The save kernels are warmed by ``build``. -> seconds (0.0 when off).
        """
        if self.ks is None or not self.idx.xrow:
            return 0.0
        from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as PO
        from models.demos.blackhole.deepseek_v41_flash.tt import prefix_cache as PC

        t0 = time.perf_counter()
        m = self.model
        cols, b = m.cols, int(user)
        r = b // m.U
        try:
            m.release_user(b)
            m.pool.release(b)
            m.pool.admit(b, tokens + 1)
            m.pool.sync_page_table()
            for s_ in PO.SOURCES:
                if s_ not in m.layer_ids:
                    continue
                n_ = tokens // PO.SRC_RATIO[s_]
                phys = m.pool.phys_rows(b, s_, torch.arange(n_)).long()
                lat = PC.gather_pool_rows(m, r, phys)
                PC.scatter_rows_mesh(m, m.pool.pool, {r: (phys, lat)})
            pend = []
            for lid in self.ks.store:
                R = self.ks.geom[lid].nrb * 32
                part = ttnn.slice(self.ks.store[lid], [0, 0, 0, 0], [1, 1, R, APS.IDIM])
                pend.append(
                    (
                        part,
                        [
                            ttnn.from_device(ttnn.get_device_tensors(part)[r * cols + c], blocking=False)
                            for c in range(cols)
                        ],
                    )
                )
            hp = []
            for pn in self.hs.store:
                NL = len(self.hs.layers[pn])
                sl = ttnn.slice(self.hs.store[pn], [0, 0, 0, 0], [NL, 1, APS.WINDOW, APS.HEAD_DIM])
                hp.append((sl, ttnn.from_device(ttnn.get_device_tensors(sl)[r * cols], blocking=False)))
            ttnn.synchronize_device(m.md)
            for part, hs_ in pend:
                for h in hs_:
                    ttnn.to_torch(h)
                ttnn.deallocate(part)
            for sl, h in hp:
                ttnn.to_torch(h)
                ttnn.deallocate(sl)
            if (
                os.environ.get("DSV41_WARM_XROW_SAVE", "0") == "1"
            ):  # (before the captures: see ``warm_xrow_late``; this variant hung a server under load in 2 of 2 tests)
                self._warm_xrow_save()
        finally:
            m.release_user(b)
        ttnn.synchronize_device(m.md)
        dt = time.perf_counter() - t0
        self.log(f"APC paged: cross-row copy ops warmed on an idle user in {dt:.1f} s")
        return dt

    def _warm_xrow_save(self):
        """the save kernels on the STAGING tensors of the copy (their sharding / layout differ from the persistent slabs ``build`` warmed them with) with all-SKIP tables: no data is touched"""
        m = self.model
        n_u = m.rows * m.U
        shard = ttnn.ShardTensor2dMesh(m.md, dims=(0, 1), mesh_shape=(m.rows, m.cols))
        rep0 = ttnn.ShardTensor2dMesh(m.md, dims=(0, None), mesh_shape=(m.rows, m.cols))
        keep = []
        self.ks.set_save([None] * n_u)
        stage = {}
        for lid in self.ks.store:
            R = self.ks.geom[lid].nrb * 32
            stage[lid] = ttnn.from_torch(
                torch.zeros(n_u, m.cols, R, APS.IDIM, dtype=torch.bfloat16),
                device=m.md,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=shard,
            )
        self.ks.save(stage)
        keep += list(stage.values())
        for pn in self.hs.store:
            self.hs.set_save(pn, [None] * n_u)
            st_ = {
                lid: ttnn.from_torch(
                    torch.zeros(n_u, 1, APS.WINDOW, APS.HEAD_DIM, dtype=torch.bfloat16),
                    device=m.md,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=rep0,
                )
                for lid in self.hs.layers[pn]
            }
            self.hs.save(pn, st_)
            keep += list(st_.values())
        ttnn.synchronize_device(m.md)
        for x in keep:
            ttnn.deallocate(x)

    def warm_xrow_late(self):
        """``_warm_xrow_save`` AFTER every trace was captured (DSV41_WARM_XROW_LATE=1; the end of the vLLM warm-up)"""
        if self.ks is None or not self.idx.xrow or os.environ.get("DSV41_WARM_XROW_LATE", "0") != "1":
            return 0.0
        t0 = time.perf_counter()
        self._warm_xrow_save()
        dt = time.perf_counter() - t0
        self.log(f"APC paged: cross-row save kernels warmed after the captures in {dt:.1f} s")
        return dt

    def xrow_copy(self, b, rid, plan, tokens, e):
        """The best hit of request ``rid`` lives on mesh row ``plan.xrow_src`` (no free user there); user ``b`` is on another row and holds fresh private pages (``plan.page_ids``). Read the donor blocks
        to the host (latents of the pool pages, key store slots, the tip's halo slots), write them into this row (latents scatter into the new pages, keys / halos through ``key_save`` /
        ``halo_copy`` from staging tensors into freshly allocated store slots), publish the blocks in this row's index and return the now LOCAL plan (``xrow_finish``), which the usual ``restore`` then
        executes. Eager launches between replays on CQ 0, no CCL. -> HitPlan, or None (the request is dropped from the index, the caller recomputes from 0).
        """
        from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as PO
        from models.demos.blackhole.deepseek_v41_flash.tt import prefix_cache as PC
        from models.demos.blackhole.deepseek_v41_flash.tt.kv_paged import PageLayout

        t0 = time.perf_counter()
        m = self.model
        md, rows, cols, U = m.md, m.rows, m.cols, m.U
        sri, dst, hit, C = plan.xrow_src, b // U, plan.hit_tokens, GRID
        try:
            src = self.idx.xrow_source(rid)
            assert src, "no donor blocks"
            nb = len(src)
            tok = np.ascontiguousarray(torch.as_tensor(tokens).long().reshape(-1).numpy())
            tp = [time.perf_counter()]
            # 1. read the donor blocks (row sri)
            pt = torch.tensor([p for _, pages, *_ in src for p in pages], dtype=torch.long)
            lat = {}
            for s_ in PO.SOURCES:
                if s_ in m.layer_ids:
                    n_ = hit // PO.SRC_RATIO[s_]
                    phys = PageLayout().phys_rows(pt, PO.SOURCES.index(s_), torch.arange(n_)).long()
                    lat[s_] = PC.gather_pool_rows(m, sri, phys)
            pend = []
            for i, (_, _, kslot, _, _, _) in enumerate(src):
                for lid in self.ks.store:
                    R = self.ks.geom[lid].nrb * 32
                    part = ttnn.slice(self.ks.store[lid], [kslot, 0, 0, 0], [kslot + 1, 1, R, APS.IDIM])
                    pend.append(
                        (
                            lid,
                            i,
                            [
                                ttnn.from_device(ttnn.get_device_tensors(part)[sri * cols + c], blocking=False)
                                for c in range(cols)
                            ],
                            part,
                        )
                    )
            tip = src[-1]
            hpend = {}
            for pn in self.hs.store:
                slot = tip[3] if pn == "A" else tip[4]
                if slot >= 0:
                    NL = len(self.hs.layers[pn])
                    sl = ttnn.slice(
                        self.hs.store[pn], [slot * NL, 0, 0, 0], [(slot + 1) * NL, 1, APS.WINDOW, APS.HEAD_DIM]
                    )
                    hpend[pn] = (ttnn.from_device(ttnn.get_device_tensors(sl)[sri * cols], blocking=False), sl)
            ttnn.synchronize_device(md)
            keys = {}
            for lid, i, hs_, part in pend:
                keys[(lid, i)] = [ttnn.to_torch(h).reshape(-1, APS.IDIM).clone() for h in hs_]
                ttnn.deallocate(part)
            halos = {}
            for pn, (h, sl) in hpend.items():
                halos[pn] = ttnn.to_torch(h).reshape(-1, APS.WINDOW, APS.HEAD_DIM).clone()
                ttnn.deallocate(sl)
            tp.append(time.perf_counter())
            # 2. this row: the page table first, then the latents into the new pages
            m.pool.allocs[dst].adopt(b, plan.page_ids)
            m.pool.sync_page_table()
            for s_, l_ in lat.items():
                n_ = l_.shape[0]
                PC.scatter_rows_mesh(m, m.pool.pool, {dst: (m.pool.phys_rows(b, s_, torch.arange(n_)).long(), l_)})
            tp.append(time.perf_counter())
            # 3. one copy chunk per donor block: key store slot (+ the tip's halo slots) in this row, filled by the save kernels from staging tensors
            cps = []
            for i, (_, _, _, ha, hb, bz) in enumerate(src):
                last = i == nb - 1
                cp = self.idx.begin_chunk(
                    rid, i * C, (i + 1) * C, tok, e, copy=(last and ha >= 0, last and hb >= 0, bz)
                )
                if cp is not None:
                    cps.append((i, cp))
            shard = ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(rows, cols))
            rep0 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
            keep = []
            for i, cp in cps:
                if cp.key_slot >= 0:
                    vals = [None] * (rows * U)
                    vals[b] = cp.key_slot
                    self.ks.set_save(vals)
                    stage = {}
                    for lid in self.ks.store:
                        R = self.ks.geom[lid].nrb * 32
                        host = torch.zeros(rows * U, cols, R, APS.IDIM, dtype=torch.bfloat16)
                        for c in range(cols):
                            host[b, c] = keys[(lid, i)][c]
                        stage[lid] = ttnn.from_torch(
                            host,
                            device=md,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=shard,
                        )
                    self.ks.save(stage)
                    keep += list(stage.values())
                for pn, slot in (("A", cp.halo_a), ("B", cp.halo_b)):
                    if slot >= 0 and pn in halos:
                        vals = [None] * (rows * U)
                        vals[b] = slot
                        self.hs.set_save(pn, vals)
                        st_ = {}
                        for j, lid in enumerate(self.hs.layers[pn]):
                            host = torch.zeros(rows * U, 1, APS.WINDOW, APS.HEAD_DIM, dtype=torch.bfloat16)
                            host[b, 0] = halos[pn][j]
                            st_[lid] = ttnn.from_torch(
                                host,
                                device=md,
                                dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                mesh_mapper=rep0,
                            )
                        self.hs.save(pn, st_)
                        keep += list(st_.values())
                # (the save kernels read their tables and staging tensors when they RUN: the table of the next block is only written after this launch is enqueued, h2d is ordered on the queue)
            ttnn.synchronize_device(md)
            for x in keep:
                ttnn.deallocate(x)
            for _, cp in cps:
                self.idx.finish_chunk(cp)
            new = self.idx.xrow_finish(rid)
            if new is None:
                raise RuntimeError("a receiving block is missing after the copy")
        except Exception as ex:  # (the request recomputes from 0; the index forgets it)
            self.stats["xrow_failed"] += 1
            self.log(
                f"APC paged: cross-row copy for request {rid} (donor row {sri} -> row {dst}, {hit} tokens) failed: {ex!r}"
            )
            self.idx.release(rid)
            return None
        t1 = time.perf_counter()
        dt = t1 - t0
        self.stats["xrow"] += 1
        self.stats["xrow_tokens"] += hit
        self.stats["xrow_s"] += dt
        self.log(
            f"APC paged: cross-row hit {hit} tokens copied row {sri} -> row {dst} in {dt * 1e3:.0f} ms ({nb} blocks; read {(tp[1] - tp[0]) * 1e3:.0f} ms, latent scatter {(tp[2] - tp[1]) * 1e3:.0f} ms, store writes {(t1 - tp[2]) * 1e3:.0f} ms)"
        )
        return new

    def engram_history(self, hits, items):
        """the n-gram history of the hit users on the host (pure function of the tokens, as ``prefix_cache.restore_users``)"""
        m = self.model
        if m.host_rows is None:
            return
        st = m.hasher.st
        for b, t, s0, e in items:
            if b in hits:
                st.cache[b, :s0] = st.token_map[t[:s0]]

    # ---- commit of the chunks ---------------------------------------------------------------------------------------------------------
    def commit(self, entries):
        """``entries``: [(user, tokens (torch 1-D), s_end, e)] of the windows just ENQUEUED (every window of the call that ended at s_end <= e). Saves the keys + halos of the blocks the index wants and
        publishes them. Called on the main thread right after the replay was enqueued: the saves are queued behind the trace and before the next replay.
        """
        if not entries or self.ks is None:
            return
        t0 = time.perf_counter()
        n = self.model.rows * self.model.U
        kv, va, vb = [None] * n, [None] * n, [None] * n
        cps = []
        for b, t, s_end, e in entries:
            rid = self.rid.get(b)
            if rid is None or b in self.auto or b in self.noc or s_end > e or s_end % GRID:
                continue
            ent = self.ktok.get(b)
            if ent is None or ent[0] is not t:
                ent = (t, np.ascontiguousarray(t.numpy()) if t.dtype == torch.int64 else t.long().numpy())
                self.ktok[b] = ent
            try:
                cp = self.idx.begin_chunk(rid, s_end - GRID, s_end, ent[1], self.plen.get(b) or e)
            except (KeyError, ValueError) as ex:
                self.log(f"APC paged: no commit for user {b} at {s_end}: {ex}")
                continue
            if cp is None:
                continue
            cps.append(cp)
            kv[b] = cp.key_slot if cp.key_slot >= 0 else None
            va[b] = cp.halo_a if cp.halo_a >= 0 else None
            vb[b] = cp.halo_b if cp.halo_b >= 0 else None
        if not cps:
            return
        _, slabs = self.owners()
        halos = self.halos()
        if any(x is not None for x in kv):
            self.ks.set_save(kv)
            self.ks.save(slabs)
        if any(x is not None for x in va):
            self.hs.set_save("A", va)
            self.hs.save("A", halos)
        if any(x is not None for x in vb) and "B" in self.hs.store:
            self.hs.set_save("B", vb)
            self.hs.save("B", halos)
        for cp in cps:
            self.idx.finish_chunk(cp)
        self.stats["commits"] += len(cps)
        self.stats["commit_s"] += time.perf_counter() - t0

    # ---- lifecycle ---------------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.idx.reset()

    def line(self):
        s = self.idx.stats()
        r0 = s["rows"][0]
        return (
            f"APC paged: lookups {s['lookups']} hits {s['hits']} ({s['hit_tokens']} tokens) lost {s['lost']} commits {s['commits']} dup {s['dup_commits']} evictions block/haloA/haloB "
            f"{s['ev_block']}/{s['ev_halo_a']}/{s['ev_halo_b']} out_of_pages {s['out_of_pages']}; busy-row loss {s['busy_loss']} ({s['busy_loss_tokens']} tok), halo loss {s['halo_loss']} ({s['halo_loss_tokens']} tok); row0 free pages {r0['pages_free']} keys {r0['key_free']} A {r0['halo_a_free']} B {r0['halo_b_free']} blocks {r0['blocks']}; "
            f"restores {self.stats['restores']} ({1e3 * self.stats['restore_s']:.1f} ms), commit {1e3 * self.stats['commit_s']:.1f} ms"
        )

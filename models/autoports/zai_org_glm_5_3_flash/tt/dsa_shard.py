# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-sharded DSA state + split-K decode over the chips of a TP mesh (doc/ttnn/14-production-tp8-megakernel/dsa/
served_1m_memory_plan.md lever B-q; coordinator ruling 2026-10-22 18:30Z). Host side: the owner / page mapping, the
sharded allocation of the latent pool and the pooled-key store, the float references (replicated vs sharded step, the
exact candidate merge, the cross-chip log-sum-exp merge) and the per-chip byte budget. The device side is
tt/tp_dsa_shard.py (the planner) + kernels/mla/dsa_shard.hpp and the MLA_DSA_SHARD paths of the owner kernels.

Unit / page format (the ruling): a shard UNIT is 128 tokens = one pooled-key page (POOL_PAGE = 32 pools of KPOOL = 4)
= 2 vLLM blocks of 64. Unit u of a user lives on chip u mod C (interleaved: every context length balances and nothing
re-shards as the context grows). Position p -> owner (p // 128) % C, local position (p // 128 // C) * 128 + p % 128.
Global pool g (= p // 4) -> owner (g // 32) % C, local pool (g // 32 // C) * 32 + g % 32: so on its owner a pool keeps
the SAME in-page index, and the kernels' local pool ids address the local page table with no translation (the S1 gather
of kernels/mla/att_dsa_dm0.cpp is unchanged). Block ids: a global block id b of a position range in unit u satisfies
b % C == u % C and the local block id is b // C (the allocator hands unit u blocks from the pool {b : b % C == u % C},
Stage 9's "vLLM block -> (chip, local block)" rule); the device reads per-chip page tables of LOCAL block ids.

Split-K decode step of one DSA layer (per token, C chips, H = 64 heads, H / C per chip):
  I4 / I5 local: every chip scores ITS pools, exact local top-k (k = TOPK = 512, the kernel's k) by the radix select;
  candidate exchange: all-gather of the local top-k keys (order-preserving uint32 of the fp32 score) with their chunk
  core; global threshold T_g = the k-th largest key of the union (exact: the global top-k of the union of the local
  top-ks is the global top-k; ties resolved in (chip, chunk core, local order), the local select's own tie order);
  S1 split-K: every chip attends ITS selected pools (+ the open pool's tail rows on the owner of the open pool) with
  ALL H heads (q all-gathered); partial (m, l, O) per head in fp32;
  merge: the partials of a head go to its owner (the TP head split), merged with the in-chip partial-tree math
  (M = max m_c; a_c = exp(m_c - M); l = sum a_c l_c; O = sum diag(a_c) O_c), o = O / l on the head owner.
The latents never leave their owner chip (only queries, candidate keys and 64-head partials cross the fabric)."""

from __future__ import annotations

import math
import os
import struct

import torch

UNIT = 128  # tokens per shard unit = one pooled-key page (KPOOL * POOL_PAGE) = 2 vLLM blocks of 64
KPOOL = 4  # tokens per DSA pool (kernels/mla/dsa_common.hpp KPOOL)
POOL_PAGE = 32  # pools per pooled-key page (tt/prefill_tp.py POOL_PAGE)
TOPK = 512  # the kernel's k (dsa_common.hpp TOPK)
DENSE_MAX = 2051  # dense-exact bound (dsa_common.hpp DENSE_MAX)
KV_LORA = 512
INDEX_DIM = 128
LMCACHE_CHUNK_UNITS = 2  # LMCache chunk = 256 tokens = 2 units (served_1m_memory_plan.md, page / unit format)
SKIP = 0xFFFFFFFF  # kernels/prefill/rows_to.cpp: a tile / row the scatter leaves alone (tiles_to)
assert UNIT == KPOOL * POOL_PAGE


def enabled() -> bool:
    """GLM53_DSA_SHARD=1: the sharded pool + split-K decode phase (default off)"""
    return os.environ.get("GLM53_DSA_SHARD", "0") == "1"


class ShardMap:
    """Positions / pools / blocks of a user's context -> (owner chip, local index) for C chips (the ruling's unit rule)."""

    def __init__(self, C: int, block: int = 64):
        assert C >= 1 and UNIT % block == 0, (C, block)
        self.C, self.block = int(C), int(block)
        self.bpu = UNIT // block  # blocks per unit

    # ---- positions
    def owner(self, p: int) -> int:
        return (p // UNIT) % self.C

    def local_pos(self, p: int) -> int:
        return (p // UNIT // self.C) * UNIT + p % UNIT

    def local_cap(self, cap: int) -> int:
        """local tokens one slot of `cap` global tokens needs on EVERY chip (the same on all chips: one program)"""
        units = -(-cap // UNIT)
        return -(-units // self.C) * UNIT

    def n_local_rows(self, p: int, r: int) -> int:
        """rows of chip r among positions 0..p (the dense-exact path attends them all)"""
        n = p + 1
        fu, rem = divmod(n, UNIT)
        full = max(0, (fu - r + self.C - 1) // self.C)  # units u < fu with u % C == r
        return full * UNIT + (rem if fu % self.C == r else 0)

    # ---- pools (pooled-key rows)
    def n_closed_local(self, p: int, r: int) -> int:
        """closed pools (< (p + 1) // 4) that live on chip r"""
        n = (p + 1) // KPOOL
        fu, rem = divmod(n, POOL_PAGE)
        full = max(0, (fu - r + self.C - 1) // self.C)
        return full * POOL_PAGE + (rem if fu % self.C == r else 0)

    def pool_owner(self, g: int) -> int:
        return (g // POOL_PAGE) % self.C

    def local_pool(self, g: int) -> int:
        return (g // POOL_PAGE // self.C) * POOL_PAGE + g % POOL_PAGE

    def global_pool(self, r: int, lq: int) -> int:
        return ((lq // POOL_PAGE) * self.C + r) * POOL_PAGE + lq % POOL_PAGE

    def local_pool_rows(self, cap: int) -> int:
        """pooled-key rows per slot per chip (Tp of mk_mla.DsaIndexer.pool_rows on the local capacity)"""
        return -(-(self.local_cap(cap) // KPOOL) // 32) * 32

    # ---- blocks (the owner invariant b % C == owner, local id b // C)
    def local_blocks(self, nb_global: int) -> int:
        return -(-nb_global // self.C)

    def block_owner(self, b: int) -> int:
        return b % self.C

    def local_block(self, b: int) -> int:
        return b // self.C

    def global_block(self, r: int, lb: int) -> int:
        return lb * self.C + r

    def page_table_local(self, pt: torch.Tensor, r: int, bps_local: int | None = None) -> torch.Tensor:
        """a slot-major GLOBAL page table [rows, bps] (column j = the block of positions [j B, (j + 1) B)) -> chip r's
        LOCAL table [rows, bps_local] (column j = local block j of the slot's local positions); entries beyond the
        global table are 0 (never read: the local position bound follows cur_pos). Vectorised (B = 32 x 256 local
        columns x 8 chips per publish)."""
        pt = pt.to(torch.int64)
        rows, bps = pt.shape
        if bps_local is None:
            bps_local = self.local_cap(bps * self.block) // self.block
        j = torch.arange(bps_local)
        jg = ((j // self.bpu) * self.C + r) * self.bpu + j % self.bpu
        keep = jg < bps
        out = torch.zeros(rows, bps_local, dtype=torch.int64)
        if keep.any():
            b = pt[:, jg[keep]]
            if ((b % self.C) != r).any():
                bad = int(jg[keep][torch.nonzero((b % self.C) != r)[0, 1]])
                raise ValueError(f"page table: a block of unit column {bad} is not owned by chip {r}")
            out[:, keep] = b // self.C
        return out.to(torch.int32)

    def page_tables_local_rows(self, pt: torch.Tensor, bps_local: int) -> list:
        """[page_table_local(pt, r, bps_local) for r < C] in one gather (bitwise the same tables): the per-chip loop is
        ~12 small torch ops per chip, which dominates a one-row publish (cf3h-d lever A). An off-owner entry falls back
        to the per-chip loop, which raises its error."""
        pt = pt.to(torch.int64)
        rows, bps = pt.shape
        key = (bps, int(bps_local))
        cache = self.__dict__.setdefault("_local_idx", {})
        if key not in cache:
            j = torch.arange(bps_local)
            jg = ((j // self.bpu) * self.C)[None, :] + torch.arange(self.C)[:, None]
            jg = jg * self.bpu + (j % self.bpu)[None, :]  # [C, bps_local]: page_table_local's jg for every chip
            keep = jg < bps
            cache[key] = (torch.where(keep, jg, 0), keep)
        idx, keep = cache[key]
        g = pt[:, idx]  # [rows, C, bps_local]
        if bool((keep[None] & (g % self.C != torch.arange(self.C)[None, :, None])).any()):
            return [self.page_table_local(pt, r, bps_local) for r in range(self.C)]  # raises the per-chip error
        out = torch.where(keep[None], g // self.C, 0).to(torch.int32)
        return [out[:, r].contiguous() for r in range(self.C)]

    def page_tables_local(self, pt: torch.Tensor, bps_local: int | None = None) -> torch.Tensor:
        """every chip's local table stacked: [C, rows, bps_local] int32 (ShardTensorToMesh on dim 0)"""
        return torch.stack([self.page_table_local(pt, r, bps_local) for r in range(self.C)])

    # ---- the TP prefill's owner-only writes (tt/prefill_tp.py under GLM53_DSA_SHARD; tiles_to with a per-chip index)
    def prefill_lat_idx(self, blocks: torch.Tensor, n_valid: int, wt: int = KV_LORA // 32) -> torch.Tensor:
        """the chunk's latent tiles -> each chip's paged cache (kernels/prefill/rows_to.cpp over tile pages, SKIP =
        0xFFFFFFFF): blocks [c / block] = the PHYSICAL global block of every page of the chunk (g % C = its owner, local
        block g // C); the chunk's latent [1, 1, c, 512] TILE has c / 32 tile rows of wt tiles; tile row k goes to its
        page's owner chip only, at local tile (local block x tiles per block + row within the block) x wt + w, and only
        while it holds a valid row (32 k < n_valid: a padding tile row of the bucket is never written anywhere -- the
        replicated path wrote the bucket's pad pages through the page-table tail). Returns [C, c / 32 x wt] int64."""
        blocks = blocks.to(torch.int64).reshape(-1)
        tpb = self.block // 32  # tile rows per block
        n_rows = int(blocks.numel()) * tpb
        out = torch.full((self.C, n_rows * wt), SKIP, dtype=torch.int64)
        for k in range(n_rows):
            if 32 * k >= n_valid:
                break
            g = int(blocks[k // tpb])
            r, lb = g % self.C, g // self.C
            d0 = (lb * tpb + k % tpb) * wt
            out[r, k * wt : (k + 1) * wt] = torch.arange(d0, d0 + wt)
        return out

    def prefill_pool_idx(self, c: int, chunk_start: int, n_valid: int, slot: int, pps_local: int,
                         wt: int = INDEX_DIM // 32) -> torch.Tensor:  # fmt: skip
        """the bucket's pooled-key tile rows (one per 128-token unit: [1, 1, c / 4, 128] TILE = c / 128 rows of wt
        tiles) -> each chip's LOCAL pooled-key store (rows x pps_local pages of wt tiles; slot s = pages s pps_local ..):
        unit u of the chunk is global unit chunk_start / 128 + u, owner (that) % C, local unit (that) // C; written only
        on its owner and only while it holds a valid row (128 u < n_valid; the bucket's other units SKIP). Returns
        [C, max(1, c / 128) x wt] int64 (the device table's width)."""
        assert chunk_start % UNIT == 0 and n_valid <= c, (chunk_start, n_valid, c)
        n_units = -(-int(n_valid) // UNIT)
        out = torch.full((self.C, max(1, c // UNIT) * wt), SKIP, dtype=torch.int64)
        return self._pool_idx_into(out, 0, chunk_start // UNIT, n_units, slot, pps_local, wt)

    def _pool_idx_into(self, out, u0_src: int, u0_global: int, n_units: int, slot: int, pps_local: int, wt: int):
        for u in range(n_units):
            gu = u0_global + u
            r, lu = gu % self.C, gu // self.C
            assert lu < pps_local, (gu, lu, pps_local)
            d0 = (slot * pps_local + lu) * wt
            out[r, (u0_src + u) * wt : (u0_src + u + 1) * wt] = torch.arange(d0, d0 + wt)
        return out

    def prefill_pool_idx_paged(self, blocks: torch.Tensor, n_valid: int, wt: int = INDEX_DIM // 32) -> torch.Tensor:
        """GLM53_DSA_PAGED_POOLS (12's paged_pooled_keys.md rev 3): the chunk's pooled keys PADDED per 64-token block
        ([1, 1, c / 2, 128] TILE: block k's 16 pooled rows in rows 0-15 of tile row k, rows 16-31 zero) -> each chip's
        paged store (page = local latent block, wt tiles): tile row k goes to block k's owner chip g % C at page
        g // C only, while it holds a valid row (64 k < n_valid). blocks [c / 64] = the PHYSICAL global block of every
        page of the chunk. Returns [C, c / 64 x wt] int64 (SKIP elsewhere)."""
        blocks = blocks.to(torch.int64).reshape(-1)
        out = torch.full((self.C, int(blocks.numel()) * wt), SKIP, dtype=torch.int64)
        for k in range(int(blocks.numel())):
            if self.block * k >= n_valid:
                break
            g = int(blocks[k])
            r, lb = g % self.C, g // self.C
            out[r, k * wt : (k + 1) * wt] = torch.arange(lb * wt, (lb + 1) * wt)
        return out

    def packed_pool_idx_paged(self, t: int, segs: list, wt: int = INDEX_DIM // 32) -> torch.Tensor:
        """GLM53_DSA_PAGED_POOLS: the PACKED prefill's padded pooled-key tile rows (one per 64-token block): segs = [(offset
        o, n_valid, the slot's PHYSICAL page row)], o on 64-row boundaries -> [C, t / 64 x wt] int64 (as
        prefill_pool_idx_paged per segment)"""
        out = torch.full((self.C, t // self.block * wt), SKIP, dtype=torch.int64)
        for o, nv, row in segs:
            assert o % self.block == 0, (o, self.block)
            row = row.to(torch.int64).reshape(-1)
            for p in range(-(-int(nv) // self.block)):
                g = int(row[p])
                r, lb = g % self.C, g // self.C
                src = o // self.block + p
                out[r, src * wt : (src + 1) * wt] = torch.arange(lb * wt, (lb + 1) * wt)
        return out

    def packed_lat_idx(self, t: int, segs: list, wt: int = KV_LORA // 32) -> torch.Tensor:
        """the PACKED prefill's latent tiles (tt/prefill_tp.py write_packed): segs = [(offset o, n_valid, the slot's
        PHYSICAL page row)], segment rows [o, o + n_valid) on 32-row boundaries -> [C, t / 32 x wt] int64 (SKIP off the
        owner / past the valid rows, as prefill_lat_idx)"""
        tpb = self.block // 32
        out = torch.full((self.C, t // 32 * wt), SKIP, dtype=torch.int64)
        for o, nv, row in segs:
            assert o % 32 == 0, o
            row = row.to(torch.int64).reshape(-1)
            for p in range(-(-int(nv) // 32)):
                g = int(row[p // tpb])
                r, lb = g % self.C, g // self.C
                d0 = (lb * tpb + p % tpb) * wt
                src = o // 32 + p
                out[r, src * wt : (src + 1) * wt] = torch.arange(d0, d0 + wt)
        return out

    def packed_pool_idx(self, t: int, segs: list, pps_local: int, wt: int = INDEX_DIM // 32) -> torch.Tensor:
        """the PACKED prefill's pooled-key tile rows: segs = [(offset o, n_valid, slot)], o on 128-row boundaries ->
        [C, t / 128 x wt] int64 (segment unit u = the slot's global unit u: owner u % C, local u // C)"""
        out = torch.full((self.C, t // UNIT * wt), SKIP, dtype=torch.int64)
        for o, nv, slot in segs:
            assert o % UNIT == 0, (o, UNIT)
            self._pool_idx_into(out, o // UNIT, 0, -(-int(nv) // UNIT), int(slot), pps_local, wt)
        return out

    # ---- the TP prefill's owner-aware READS: local page_gather -> all_gather (rank order) -> this static un-permute
    def gather_table(self, n_pages: int, pages_local: int, ppu: int) -> torch.Tensor:
        """page_gather table [1, n_pages] int32 (tt/prefill_tp.py page_gather: dst page i <- src page table[i]) that
        rebuilds the GLOBAL order of a store from the chips' local stores all-gathered in rank order (block r = chip
        r's pages_local pages): a page = 1 / ppu of a 128-token unit (latent blocks of 64: ppu = 2; pooled-key pages:
        ppu = 1); global page i is unit i // ppu, owner (i // ppu) % C, local page ((i // ppu) // C) ppu + i % ppu"""
        i = torch.arange(n_pages)
        u = i // ppu
        src = (u % self.C) * pages_local + (u // self.C) * ppu + i % ppu
        assert int(src.max()) < self.C * pages_local, (n_pages, pages_local, ppu)
        return src.to(torch.int32).reshape(1, -1)

    # ---- lever B-q (the split-K prefill, GLM53_DSA_BQ): the per-chip id filter and the owner merge, as the device does
    def bq_local_ids(self, ids: torch.Tensor, r: int, width: int | None = None) -> tuple:
        """kernels/prefill/expand_rows.cpp SHARD mode: a row's global token ids (valid prefix, SKIP tail) -> chip r's
        owned keys as LOCAL row ids (same order), the SKIP tail, and the count; an empty row gets one dummy key (local
        row 0). ids [R, W] int64 -> (local ids [R, W], counts [R])"""
        R, W = ids.shape
        width = width or W
        out = torch.full((R, width), SKIP, dtype=torch.int64)
        counts = torch.zeros(R, dtype=torch.int64)
        for i in range(R):
            row = ids[i]
            v = row[row != SKIP]
            mine = v[(v // UNIT) % self.C == r]
            loc = (mine // UNIT // self.C) * UNIT + mine % UNIT
            n = int(loc.numel())
            counts[i] = n
            if n == 0:
                out[i, 0] = 0
            else:
                out[i, :n] = loc
        return out, counts

    @staticmethod
    def bq_stats(q: torch.Tensor, keys: torch.Tensor, scale: float, bf16_stats: bool = False) -> tuple:
        """one device's sparse_sdpa_stats on its keys: (o normalised [H, D], m [H] = the UNSCALED score max,
        l [H] = sum exp((s - m) scale)); keys [n, D] (n >= 1), q [H, D]. bf16_stats: m / l rounded as the kernel's bf16
        running tiles would"""
        s = q.float() @ keys.float().T  # [H, n]
        m = s.max(dim=-1).values
        if bf16_stats:
            m = m.to(torch.bfloat16).float()
        pr = torch.exp((s - m[:, None]) * scale)
        l = pr.sum(-1)
        if bf16_stats:
            l = l.to(torch.bfloat16).float()
        o = (pr @ keys.float()) / l[:, None]
        return o, m, l

    @staticmethod
    def bq_merge(parts: list, scale: float) -> torch.Tensor:
        """the owner merge of C devices' (o_c, m_c, l_c, valid_c) in device order (tt/prefill_tp.py MlaPrefill._bq):
        M = max_c m_c, w_c = exp((m_c - M) scale) l_c / sum_c' ..., o = sum_c w_c o_c; an invalid partial (no owned
        key: the dummy) weighs 0"""
        ms = torch.stack([torch.where(v, m, torch.full_like(m, -float("inf"))) for _, m, _, v in parts])  # [C, H]
        ls = torch.stack([torch.where(v, l, torch.zeros_like(l)) for _, _, l, v in parts])
        M = ms.max(0).values
        w = torch.exp((ms - M) * scale) * ls  # [C, H]
        w = w / w.sum(0)
        o = torch.zeros_like(parts[0][0])
        for c, (oc, _, _, v) in enumerate(parts):
            o = o + torch.where(v[:, None], oc * w[c][:, None], torch.zeros_like(oc))
        return o

    def null_row(self, bps: int) -> torch.Tensor:
        """the physical page-table row of an idle slot: column j -> its owner's null block (global id = the owner;
        tt/owner_blocks.py OwnerBlockMap.null)"""
        return (torch.arange(bps) // self.bpu) % self.C

    def shard_rows(self, rows: torch.Tensor, r: int, cap: int | None = None) -> torch.Tensor:
        """rows [L, D] of a context in position order -> chip r's rows in LOCAL order [local_cap(L), D] (zero past L)"""
        L = int(rows.shape[0])
        cap = cap or L
        lc = self.local_cap(cap)
        out = torch.zeros(lc, *rows.shape[1:], dtype=rows.dtype)
        units = -(-L // UNIT)
        for u in range(r, units, self.C):
            n = min(UNIT, L - u * UNIT)
            lu = u // self.C
            out[lu * UNIT : lu * UNIT + n] = rows[u * UNIT : u * UNIT + n]
        return out

    def shard_pools(self, pools: torch.Tensor, r: int, tp_local: int) -> torch.Tensor:
        """pooled-key rows [Tp, 128] (row g = pool g) -> chip r's [tp_local, 128] (row lq = local pool lq)"""
        out = torch.zeros(tp_local, *pools.shape[1:], dtype=pools.dtype)
        pages = -(-int(pools.shape[0]) // POOL_PAGE)
        for u in range(r, pages, self.C):
            n = min(POOL_PAGE, int(pools.shape[0]) - u * POOL_PAGE)
            lu = u // self.C
            out[lu * POOL_PAGE : lu * POOL_PAGE + n] = pools[u * POOL_PAGE : u * POOL_PAGE + n]
        return out


# ------------------------------------------------------------------------------------------------- page tables
def local_rows(value: torch.Tensor, rows: int, C: int, bpu: int, bps_local: int) -> list:
    """physical global page-table rows [n, bps] (block g of column j on chip g % C, local block g // C) -> [chip r's LOCAL
    rows [rows, bps_local] for r < C] (ShardMap.page_table_local; rows past n: every column's owner null block = local
    block 0); raises on an off-owner entry. Row i of every output depends on row i of `value` only."""
    value = value.to(torch.int64)
    full = (torch.arange(value.shape[1]) // bpu % C).expand(rows, -1).clone()  # null rows: each column's owner
    full[: value.shape[0]] = value
    return ShardMap(C, UNIT // bpu).page_tables_local_rows(full, bps_local)


class LocalTableCache:
    """GLM53_PT_INCREMENTAL (default 1; cf3h-d lever A, coordinator 2026-10-24T05:28:50Z): the host half of a sharded
    page-table publish. The last published global table and its C local tables are kept for one device buffer (`key`,
    compared by identity); a publish re-localizes only the rows that differ and reuses the others as they are, so the
    result is bitwise the full localization (local_rows is row-wise). A first publish, a new key or a new shape runs
    the full path. 0: always the full path (the pre-cf3h-d behaviour)."""

    def __init__(self):
        self.key, self.glob, self.local = None, None, None
        self.full = self.rows_relocalized = 0

    def update(self, key: tuple, value: torch.Tensor, rows: int, localize) -> tuple:
        """-> (the C local tables [rows, bps_local], the changed global rows: a list, or None after a full path).
        localize(value_rows, n_rows) = local_rows of those rows"""
        value = torch.as_tensor(
            value
        )  # compared in the caller's dtype (int32 from the publish): half the bytes of int64
        same_key = self.key is not None and len(self.key) == len(key) and all(a is b for a, b in zip(self.key, key))
        if (os.environ.get("GLM53_PT_INCREMENTAL", "1") == "1" and same_key and self.glob.shape == value.shape
                and self.glob.dtype == value.dtype):  # fmt: skip
            changed = torch.nonzero((value != self.glob).any(dim=1)).flatten()
            if changed.numel():
                sub = localize(value[changed], int(changed.numel()))  # raises before the cache changes
                for lt, st in zip(self.local, sub):
                    lt[changed] = st
                self.rows_relocalized += int(changed.numel())
            changed = changed.tolist()
        else:
            self.local, changed = localize(value, rows), None
            self.full += 1
        self.key, self.glob = key, value.clone()
        return self.local, changed


# ------------------------------------------------------------------------------------------------- references
def f32_key(x: torch.Tensor) -> torch.Tensor:
    """order-preserving uint32 key of fp32 values (dsa_common.hpp key_of), as int64"""
    b = x.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    neg = b >= 0x80000000
    return torch.where(neg, (~b) & 0xFFFFFFFF, b | 0x80000000)


def core_of_pool(g: torch.Tensor, chunks: int) -> torch.Tensor:
    """the chunk core that holds pool g on a chip whose owners take pooled-key tiles pt = c, c + C, ... (dsa_common.hpp
    pool_of): core = (g // 32) % chunks, local index = (g // 32 // chunks) * 32 + g % 32"""
    return (g // POOL_PAGE) % chunks


def local_index_of_pool(g: torch.Tensor, chunks: int) -> torch.Tensor:
    return (g // POOL_PAGE // chunks) * POOL_PAGE + g % POOL_PAGE


def select_topk(keys: torch.Tensor, order: torch.Tensor, k: int = TOPK):
    """exact top-k of `keys` (int64 order keys) with ties broken by the smaller `order` value (int64, unique): returns
    the selected positions (indices into keys, in ascending index order) and the threshold key T (k-th largest)"""
    n = int(keys.numel())
    if n <= k:
        return torch.arange(n), int(keys.min()) if n else 0
    # lexicographic (key desc, order asc): a stable sort by order, then a stable sort by key descending
    i1 = torch.sort(order, stable=True).indices
    i2 = torch.sort(keys[i1], descending=True, stable=True).indices
    top = i1[i2[:k]]
    T = int(keys[top].min())
    return torch.sort(top).values, T


def replicated_select(scores: torch.Tensor, chunks: int, k: int = TOPK):
    """the replicated kernel's selection on one chip: all pools, ties in (chunk core, local index) order"""
    g = torch.arange(scores.numel())
    order = core_of_pool(g, chunks) * (1 << 24) + local_index_of_pool(g, chunks)
    sel, T = select_topk(f32_key(scores), order, k)
    return sel, T


def sharded_select(scores: torch.Tensor, sm: ShardMap, chunks: int, k: int = TOPK):
    """the sharded step's selection: local top-k per chip, candidate all-gather, exact global merge with ties in
    (chip, chunk core, local index) order -> (global pool ids selected (ascending), T_g, per-chip local selections,
    per-chip local thresholds)"""
    g_all = torch.arange(scores.numel())
    keys = f32_key(scores)
    owner = (g_all // POOL_PAGE) % sm.C
    cand_keys, cand_order, cand_g, local, T_l = [], [], [], [], []
    for r in range(sm.C):
        mine = torch.nonzero(owner == r).reshape(-1)
        lq = (mine // POOL_PAGE // sm.C) * POOL_PAGE + mine % POOL_PAGE  # ShardMap.local_pool, vectorised
        order = core_of_pool(lq, chunks) * (1 << 24) + local_index_of_pool(lq, chunks)
        sel, T = select_topk(keys[mine], order, k)  # local top-k (indices into mine)
        local.append(mine[sel])
        T_l.append(T)
        cand_keys.append(keys[mine][sel])
        cand_order.append(r * (1 << 40) + order[sel])  # (chip, core, local index): the global tie order
        cand_g.append(mine[sel])
    ck, co, cg = torch.cat(cand_keys), torch.cat(cand_order), torch.cat(cand_g)
    sel, T_g = select_topk(ck, co, k)
    return torch.sort(cg[sel]).values, T_g, local, T_l


def _lazy_rows(ids: torch.Tensor, d: int = KV_LORA, seed: int = 0) -> torch.Tensor:
    """deterministic random latent rows per token id (only the attended rows are ever materialised: a 1M context is
    sampled, not stored)"""
    out = torch.empty(ids.numel(), d, dtype=torch.float32)
    for i, v in enumerate(ids.tolist()):
        gen = torch.Generator().manual_seed(seed * 1_000_003 + v)
        out[i] = torch.randn(d, generator=gen)
    return out


def attention_partial(q: torch.Tensor, K: torch.Tensor, scale: float):
    """q [H, D], K [n, D] -> (m [H], l [H], O [H, D]) fp32 (the kernel's partial: S = q K^T scale, m = rowmax,
    E = exp(S - m), l = rowsum E, O = E K); n = 0 -> (-1e30, 0, 0)"""
    H = q.shape[0]
    if K.shape[0] == 0:
        return torch.full((H,), -1.0e30), torch.zeros(H), torch.zeros(H, q.shape[1])
    S = (q @ K.T) * scale
    m = S.max(dim=1).values
    E = torch.exp(S - m[:, None])
    return m, E.sum(dim=1), E @ K


def merge_partials(parts: list):
    """the cross-chip merge = the in-chip combine (kernels/mla/att_c.cpp combine), folded in chip order: M = max(m_a,
    m_b), a = exp(m_a - M), b = exp(m_b - M), l = a l_a + b l_b, O = diag(a) O_a + diag(b) O_b"""
    m, l, O = parts[0]
    for m2, l2, O2 in parts[1:]:
        M = torch.maximum(m, m2)
        a, b = torch.exp(m - M), torch.exp(m2 - M)
        l = a * l + b * l2
        O = a[:, None] * O + b[:, None] * O2
        m = M
    return m, l, O


def bf16_split(x: torch.Tensor, terms: int) -> torch.Tensor:
    """the fabric payload format of a partial: `terms` bf16 terms whose sum reproduces x (1: bf16(x); 2: hi + lo with
    lo = bf16(x - hi), 16 significant bits; 3: + bf16(x - hi - lo), exact for normal fp32)"""
    acc = torch.zeros_like(x)
    rem = x.clone()
    for _ in range(terms):
        t = rem.to(torch.bfloat16).float()
        acc = acc + t
        rem = rem - t
    return acc


def selected_rows(sel_pools: torch.Tensor, p: int):
    """token ids attended for a selection of closed pools at position p: 4 rows per pool + the open pool's tail rows
    (positions 4 (p // 4) .. p; the current token's own latent included, as the kernel's tail)"""
    rows = (sel_pools[:, None] * KPOOL + torch.arange(KPOOL)[None, :]).reshape(-1)
    tail = torch.arange((p // KPOOL) * KPOOL, p + 1)
    return torch.cat([rows, tail])


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a * b).sum() / d)


def reference_case(L: int, B: int, C: int, *, chunks: int = 64, heads: int = 64, k: int = TOPK, seed: int = 0,
                   ties: int = 0, split_terms: tuple = (1, 2, 3), scale: float = 1.0 / math.sqrt(576.0)) -> dict:  # fmt: skip
    """the mock of one DSA layer step at context L (B users all at position L - 1): replicated selection + attention
    vs the sharded step (local top-k, exact merge, split-K partials, LSE merge in chip order). `ties`: pools whose
    score is copied from another pool (deliberate ties at random; the tie ORDER differs between the two layouts, so the
    selected sets are compared under both rules). Returns the per-user worst numbers."""
    gen = torch.Generator().manual_seed(seed)
    sm = ShardMap(C)
    p = L - 1
    assert p + 1 > DENSE_MAX, "a sparse position"
    n_closed = (p + 1) // KPOOL
    out = dict(L=L, B=B, C=C, k=k, ties=ties, n_closed=n_closed, sel_bitwise=True, sel_bitwise_sharded_rule=True,
               T_bitwise=True, pcc_min={t: 1.0 for t in split_terms}, maxabs={t: 0.0 for t in split_terms},
               pcc_merge_fp32=1.0, local_union_holds=True)  # fmt: skip
    for b in range(B):
        scores = torch.randn(n_closed, generator=gen)
        if ties:
            src = torch.randint(0, n_closed, (ties,), generator=gen)
            dst = torch.randint(0, n_closed, (ties,), generator=gen)
            scores[dst] = scores[src]
        sel_rep, T_rep = replicated_select(scores, chunks, k)
        sel_sh, T_g, local, T_l = sharded_select(scores, sm, chunks, k)
        # the exact-merge invariant: every globally selected pool is in its owner's local top-k
        loc_all = torch.cat(local)
        out["local_union_holds"] &= bool(torch.isin(sel_sh, loc_all).all())
        out["T_bitwise"] &= T_g == T_rep
        out["sel_bitwise"] &= bool(sel_sh.numel() == sel_rep.numel() and torch.equal(sel_sh, sel_rep))
        # the sharded tie rule applied to the FULL score list (one chip, (chip, core, local) order) must give the same set
        g_all = torch.arange(n_closed)
        lq = (g_all // POOL_PAGE // C) * POOL_PAGE + g_all % POOL_PAGE
        order = (
            ((g_all // POOL_PAGE) % C) * (1 << 40)
            + core_of_pool(lq, chunks) * (1 << 24)
            + local_index_of_pool(lq, chunks)
        )
        sel_rule, _ = select_topk(f32_key(scores), order, k)
        out["sel_bitwise_sharded_rule"] &= bool(torch.equal(sel_sh, sel_rule))
        out["tie_set_diff"] = max(out.get("tie_set_diff", 0), int(k - torch.isin(sel_sh, sel_rep).sum()))
        # attention: all heads over the selected rows (+ the tail): the one-chip exact attention over the SAME set
        # (the sharded tie rule's exact top-k; equal to the replicated set whenever no tie sits at the threshold) vs
        # the split-K partials merged across chips
        q = torch.randn(heads, KV_LORA, generator=gen)
        K_rep = _lazy_rows(selected_rows(sel_rule, p), seed=seed)
        m, l, O = attention_partial(q, K_rep, scale)
        o_rep = O / l[:, None]
        if not torch.equal(sel_rule, sel_rep):  # info: what the replicated kernel's tie order would have produced
            m2, l2, O2 = attention_partial(q, _lazy_rows(selected_rows(sel_rep, p), seed=seed), scale)
            out["pcc_vs_replicated_tie_rule"] = min(
                out.get("pcc_vs_replicated_tie_rule", 1.0), pcc(O2 / l2[:, None], o_rep)
            )
        parts = []
        for r in range(C):
            mine = (
                sel_sh[torch.tensor([sm.pool_owner(int(v)) == r for v in sel_sh], dtype=torch.bool)]
                if sel_sh.numel()
                else sel_sh
            )
            rows = (mine[:, None] * KPOOL + torch.arange(KPOOL)[None, :]).reshape(-1)
            if sm.owner(p) == r:  # the open pool's tail rows live on the owner of the open pool
                rows = torch.cat([rows, torch.arange((p // KPOOL) * KPOOL, p + 1)])
            parts.append(attention_partial(q, _lazy_rows(rows, seed=seed), scale))
        mg, lg, Og = merge_partials(parts)
        o_sh = Og / lg[:, None]
        out["pcc_merge_fp32"] = min(out["pcc_merge_fp32"], pcc(o_sh, o_rep))
        # the fabric payload: the peer's partial crosses as bf16 terms (the head owner's own partial stays fp32)
        for terms in split_terms:
            own = sm.C - 1  # the last chip folds in the others: its own partial exact, the rest through the payload
            payload = [(bf16_split(m_, terms), bf16_split(l_, terms), bf16_split(O_, terms)) if r != own else (m_, l_, O_)
                       for r, (m_, l_, O_) in enumerate(parts)]  # fmt: skip
            mg, lg, Og = merge_partials(payload)
            o_t = Og / lg[:, None]
            out["pcc_min"][terms] = min(out["pcc_min"][terms], pcc(o_t, o_rep))
            out["maxabs"][terms] = max(out["maxabs"][terms], float((o_t - o_rep).abs().max()))
    return out


# ------------------------------------------------------------------------------------------------- byte budget
def budget(L: int, C: int, *, layers: int = 11, users: int = 1, B: int = 32, block: int = 64, bf16_latent: bool = True,
           t_rows: int = 4) -> dict:  # fmt: skip
    """bytes per chip of the DSA state for `users` users at context L on C chips (+ the split-K exchange tensors of a
    t_rows launch): replicated (today) vs sharded (this module). Latent: 1 KiB per token per layer (bf16 [64, 512]
    blocks, + block rounding); pooled keys: 64 B per token per layer (256 B per pool); ring 16 KiB per slot per layer.
    """
    sm = ShardMap(C, block)
    lat_tok = KV_LORA * 2 if bf16_latent else KV_LORA
    nb = -(-L // block)
    rep_latent = nb * block * lat_tok * layers * users
    sh_latent = sm.local_blocks(nb) * block * lat_tok * layers * users
    tp = -(-(-(-L // block) * block // KPOOL) // 32) * 32
    rep_pools = tp * INDEX_DIM * 2 * layers * users
    sh_pools = sm.local_pool_rows(L) * INDEX_DIM * 2 * layers * users
    ring = B * 32 * 2 * INDEX_DIM * 2 * layers
    # split-K exchange per layer per launch of t rows (TP2 v0 format, tt/tp_dsa_shard.py): the AG block = per token
    # [32 rows q (H / C heads, padded) | 32 rows candidates] x 512 bf16 = 64 KiB per chip, landed x C; the RS block =
    # per token per peer 3 x 32-row chunks (O hi, O lo, stats) = 96 KiB
    ag = t_rows * 2 * 32 * KV_LORA * 2 * C
    rs = t_rows * 3 * 32 * KV_LORA * 2 * (C - 1) * 2
    return dict(L=L, C=C, layers=layers, users=users,
                replicated=dict(latent=rep_latent, pooled=rep_pools, ring=ring, total=rep_latent + rep_pools + ring),
                sharded=dict(latent=sh_latent, pooled=sh_pools, ring=ring, exchange_per_layer=ag + rs,
                             total=sh_latent + sh_pools + ring + (ag + rs) * layers))  # fmt: skip


def gib(n: int) -> float:
    return round(n / 2**30, 3)


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]

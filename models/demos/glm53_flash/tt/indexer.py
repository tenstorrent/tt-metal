# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash DSA indexer with k-pool compression (``Glm5NextTextIndexer``), single device.

HF semantics (index_kpool = 4, index_topk = 2048, always_select_tail):
    q  = wq_b(q_resid)                         [T, 32, 128]
    k  = LayerNorm(wk(x))                      [T, 128]
    g  = x @ gate^T                            [T, 128]      (k-pool gate logits)
    pool p (tokens 4p..4p+3, complete pools only):
        pk[p] = sum_j softmax_j(g[4p+j] + ape[j]) * k[4p+j]              [128]
    score[s, p] = sum_h relu(q[s,h] . pk[p] * 128^-0.5) * w[s,h] * 32^-0.5, w = weights_proj(x)
    pool p is selectable by query s iff 4p+3 <= s; top-512 pools -> their 4 tokens (2048 ids)
    + the incomplete tail tokens 4*floor((s+1)/4) .. s (<= 3 ids), -1 elsewhere.
For s < 2048 every visible token is selected and the attention is exact (the dense path).

Device composite (design 5.3 / 8.3). The index state is maintained on EVERY append (dense and
sparse prefill, dense and sparse decode) so a user can cross the 2048 boundary on the dense path:
  * pooled keys: paged cache ``pool_cache [max_batch*pool_blocks, 1, 32, 128]`` bf16 TILE with an
    identity page table (user u's pools are contiguous); viewed ``[max_batch, 1, pools, 128]`` for
    ``indexer_score_dsa(cache_batch_idx=u)``;
  * raw rings: the k and gate logits of the current pool's tokens live in one 32-row block per user
    (``ring_k`` / ``ring_g`` ``[max_batch, 1, 32, 128]``, row = position % 4). Decode writes the new
    token's row with ``paged_update_cache(update_idxs=position % 4)`` and then rewrites pool
    ``position // 4`` from rows 0..3: while that pool is incomplete it is not selectable
    (4*(a//4)+3 > a), and the write at a % 4 == 3 is the exact pooled key;
  * scoring: ``ttnn.experimental.indexer_score_dsa(key_compression_ratio=4)`` (causal by pool);
  * selection: ``ttnn.experimental.topk_large_indices(k=512)`` over the pool scores returns the
    pool ids, expanded on device to the 4 token ids of each pool (exact fp32 arithmetic);
  * the tail ids are built from the query positions and appended; the row is padded to a
    multiple of 128 (sparse_sdpa's k_chunk) with the 0xFFFFFFFF sentinel.
"""

from __future__ import annotations

import os

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.glm53_flash.custom_ops.index_score_decode.op import IndexScoreDecode
from models.demos.glm53_flash.custom_ops.topk_rows.op import TopkRows
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.matmul_configs import prefill_linear
from models.demos.glm53_flash.tt.mesh import from_torch_host_layout, lazy_const
from models.demos.glm53_flash.tt.vllm_config import PREFIX_BUCKET_ENV

TILE = 32
STATIC_PREFILL_ENV = "GLM53_INDEX_STATIC_PREFILL"  # "1" (default): shape-static eager maintenance of aligned chunks


def ring_selector(S: int, T: int, P: int = 4) -> torch.Tensor:
    """``[1,1,32,T]`` one-hot selector of a chunk's ring rows: row ``j < P`` selects the last chunk position
    ``p <= S-1`` with ``p % P == j`` (a zero row when the chunk has no such position: that ring slot then holds
    zeros, harmless because its pool is incomplete until decode rewrites the slot); rows ``>= P`` are zero.
    Applied as an exact matmul over the index stems ``[1,1,T,D]`` it yields the ring block ``[1,1,32,D]`` in one
    program whose shape depends on ``T`` only (the traced prefill's tail write, and the eager static maintenance)."""
    sel = torch.zeros(1, 1, TILE, T)
    for j in range(P):
        p = S - 1 - ((S - 1 - j) % P)
        if p >= 0:
            sel[0, 0, j, p] = 1.0
    return sel


SENTINEL = 0xFFFFFFFF
DECODE_SCORE_SLACK = 8  # pools per user hidden by the traced decode scorer's fixed chunk start (TILE // index_kpool)
# pooled-key cache dtype: bf16 (default) or bfp8_b (GLM53_POOL_DTYPE=bfloat8_b: halves the decode scorer's read,
# the design's budget term; the selected sets are checked against the bf16 cache in tests/custom_ops)
POOL_DTYPE = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}[os.environ.get("GLM53_POOL_DTYPE", "bfloat16")]


def _upload(device, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT):
    return from_torch_host_layout(t, dtype, layout, device)  # host-side layout: no Tilize program per shape


def _compute(device, fidelity=ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=False):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


def height_shard_rows(device, rows: ttnn.Tensor) -> ttnn.Tensor:
    """[1, B, 1, W] -> height-sharded on B cores (the paged_update_cache input contract)."""
    B, W = rows.shape[1], rows.shape[-1]
    grid = device.compute_with_storage_grid_size()
    cfg = ttnn.create_sharded_memory_config(
        shape=(TILE, W),
        core_grid=ttnn.num_cores_to_corerangeset(B, grid, row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.to_memory_config(rows, cfg)


class TtIndexCache:
    """Per-layer pooled-key cache + raw rings for ``max_batch`` slots.

    User-partitioned layout (``scratch`` set, see doc/mesh/sparse_1m.md section 3): the chip holds
    ``users_per_chip`` local users in slots ``[0, users_per_chip)`` plus the prefill scratch slot ``scratch``
    (= ``max_batch - 1``). Decode writes the local slots (page-table views of the first ``users_per_chip``
    rows); prefill writes the scratch slot on every column chip and, on the owner column, the user's slot
    (per-user sharded page tables), and reads the scratch slot everywhere.

    ``dummy`` (the heads2users prefill, sparse_1m.md section 7): no scratch slot; ``max_batch`` = the local
    users; the non-owner columns of a prefill write into dummies instead: the pool page table and the ring page
    table get one extra row (index ``max_batch``) that points every block at the LAST pool block of the last
    slot (never a valid pool: ``pools_per_user`` then leaves >= 32 spare pools above the valid ids) and at one
    extra ring block; every column reads slot ``user % users_per_chip`` (the owner's holds the user)."""

    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        *,
        max_batch: int,
        max_seq_len: int,
        mesh=None,
        users_per_chip: int | None = None,
        scratch: int | None = None,
        dummy: bool = False,
    ):
        self.device = device
        self.mesh = mesh
        self.D = cfg.index_head_dim
        self.P = cfg.index_kpool
        self.max_batch = max_batch
        self.dummy = bool(dummy)
        assert not (self.dummy and scratch is not None), "scratch slot and dummy writes are alternatives"
        self.partition = scratch is not None or self.dummy
        self.users_per_chip = max_batch if users_per_chip is None else users_per_chip
        self.scratch = scratch
        self.dummy_slot: int | None = max_batch if self.dummy else None  # the extra page-table row / ring block
        pools = -(-max_seq_len // self.P)
        # tile-aligned rows per user, plus DECODE_SCORE_SLACK pools the traced decode scorer can never select:
        # it scores with a fixed chunk start in the last 32-token block of the allocation (see
        # TtDSAIndexer.score_decode_row), under which the final 8 pools of the row are causally hidden; and at
        # least index_select_pools visible pools, the width the decode top-k ranks for every user (a short
        # allocation then holds 544 pool rows = 136 KB per user)
        pools = max(pools, cfg.index_select_pools)
        # rounded to 256 pools (8 tiles): the batched decode scorer streams the keys in 8-tile chunks; the dummy
        # layout keeps a whole spare block (32 pools) above the valid ids: the prefill dummy pool block
        slack = max(DECODE_SCORE_SLACK, TILE) if self.dummy else DECODE_SCORE_SLACK
        self.pools_per_user = -(-(pools + slack) // (8 * TILE)) * (8 * TILE)
        self.pool_blocks = self.pools_per_user // TILE
        assert not self.dummy or self.pools_per_user - TILE >= pools, (self.pools_per_user, pools)
        self.pool_dtype = POOL_DTYPE
        self.pool_cache = ttnn.zeros(
            (max_batch * self.pool_blocks, 1, TILE, self.D),
            dtype=self.pool_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pt = torch.arange(max_batch * self.pool_blocks, dtype=torch.int32).reshape(max_batch, self.pool_blocks)
        ring_ids = torch.arange(max_batch, dtype=torch.int32).reshape(max_batch, 1)
        n_ring = max_batch
        if self.dummy:
            # the extra row: every block of a non-owner's prefill chunk lands on the last block of the last slot
            # (the spare pools above every valid id); the extra ring block is the ring dummy
            pt = torch.cat([pt, torch.full((1, self.pool_blocks), max_batch * self.pool_blocks - 1, dtype=torch.int32)])
            ring_ids = torch.cat([ring_ids, torch.tensor([[max_batch]], dtype=torch.int32)])
            n_ring = max_batch + 1
        self._pt_host, self._ring_ids_host = pt, ring_ids
        self.page_table = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        # raw rings: one 32-row block per user, row = position % P (rows >= P unused)
        self.ring_k = ttnn.zeros((n_ring, 1, TILE, self.D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.ring_g = ttnn.zeros((n_ring, 1, TILE, self.D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.ring_page_table = ttnn.from_torch(ring_ids, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        # constants for the prefill tail writes (one row per call): slot ids and single-user page tables,
        # uploaded once here so no host->device write happens inside a prefill pass
        rm = lambda t: ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        self._slot_ids = [rm(torch.tensor([j], dtype=torch.int32)) for j in range(self.P)]
        self._user_pts = [rm(torch.tensor([[u]], dtype=torch.int32)) for u in range(n_ring)]
        # decode page-table views: the paged ops pair batch entry b with page-table row b, so a decode of the
        # users_per_chip local users needs tables of exactly that many rows (the scratch slot / dummy row is left out)
        B = self.users_per_chip
        if self.partition:
            self._decode_pt = ttnn.slice(self.page_table, [0, 0], [B, self.pool_blocks])
            self._decode_ring_pt = ttnn.slice(self.ring_page_table, [0, 0], [B, 1])
        else:
            self._decode_pt, self._decode_ring_pt = self.page_table, self.ring_page_table
        self._prefill_views: dict[tuple, ttnn.Tensor] = {}

    # ------------------------------------------------------ user-partitioned prefill
    def prefill_slot(self, user: int) -> int:
        """The slot every column chip READS during the prefill of ``user``: the scratch slot (scratch layout), the
        user's local slot (dummy layout: real on the owner column, discarded elsewhere), or the user."""
        if self.scratch is not None:
            return self.scratch
        return user % self.users_per_chip if self.dummy else user

    def _owner_rows(self, user: int, table_host: torch.Tensor) -> torch.Tensor:
        """[cols, W] host rows: row c = the user's slot row of ``table_host`` on its owner column, the scratch (or
        dummy) row elsewhere."""
        cols = self.mesh.cols
        owner, slot = user // self.users_per_chip, user % self.users_per_chip
        other = self.scratch if self.scratch is not None else self.dummy_slot
        return torch.stack([table_host[slot] if c == owner else table_host[other] for c in range(cols)])

    def _pool_view_rows(self, user: int) -> list[torch.Tensor]:
        """The host pool page-table rows a prefill of ``user`` fills (batch_idx 0): its own row when not partitioned;
        partitioned: the per-chip owner/scratch (or owner/dummy) rows and, with a scratch slot, the scratch row."""
        pt = self._pt_host
        if not self.partition:
            return [pt[user : user + 1]]
        rows = [self._owner_rows(user, pt)]
        if self.scratch is not None:
            rows.append(pt[self.scratch : self.scratch + 1])
        return rows

    def _upload_pool_view(self, rows: torch.Tensor, per_chip: bool) -> ttnn.Tensor:
        if per_chip:
            return self.mesh.upload(rows, ttnn.int32, cols=0, layout=ttnn.ROW_MAJOR_LAYOUT)
        if self.mesh is not None:
            return self.mesh.upload(rows, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.from_torch(rows.contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    def prefill_pool_views(self, user: int, b0: int = 0) -> list[ttnn.Tensor]:
        """Page-table views ([1, pool_blocks], batch_idx 0) a prefill of ``user`` fills from pool block ``b0`` on
        (``_pool_view_rows``). ``b0 == 0``: the user's views, built once. ``b0 > 0``: the rows shifted by ``b0`` on the
        host and padded back to the full width with the row's last block (never written: the pools past the write
        are zero-padded, not stored), uploaded for this call (the caller frees them): the width never changes, so the
        fill runs one program for every chunk start (the earlier device slice was a program per start block)."""
        per_chip = self.partition
        if b0 == 0:
            # a user's views are built on its first prefill: under live traces that is a chunk transient (GLMMesh.lazy)
            key = ("pools", user)
            make = lambda: [
                self._upload_pool_view(r, per_chip and i == 0) for i, r in enumerate(self._pool_view_rows(user))
            ]
            return lazy_const(self.mesh, self._prefill_views, key, make)
        if not 0 < b0 < self.pool_blocks:
            raise ValueError(f"pool start block {b0} outside [1, {self.pool_blocks})")
        out = []
        for i, r in enumerate(self._pool_view_rows(user)):
            shifted = torch.cat([r[:, b0:], r[:, -1:].expand(-1, b0)], dim=1).contiguous()
            out.append(self._upload_pool_view(shifted, per_chip and i == 0))
        return out

    # ------------------------------------------------------------------ waves (per-row chunks)
    def _wave_target_rows(self, table_host: torch.Tensor, user: int, idle: tuple[int, ...]) -> list[torch.Tensor]:
        """Per mesh row r the ``[cols, W]`` host rows of ``table_host`` a wave's prefill of slot ``user`` writes on row
        r's chips: the user's owner / dummy (or scratch) rows, or on an idle row (no member) a harmless target on
        every column: the dummy row (heads2users) or the scratch row; an unpartitioned cache has none."""
        cols = self.mesh.cols
        if not self.partition:
            if idle:
                raise NotImplementedError("a wave with idle rows needs a partitioned index (dummy / scratch rows)")
            return [table_host[user : user + 1].expand(cols, -1) for _ in range(self.mesh.rows)]
        other = self.scratch if self.scratch is not None else self.dummy_slot
        member = self._owner_rows(user, table_host)
        return [table_host[other : other + 1].expand(cols, -1) if r in idle else member for r in range(self.mesh.rows)]

    def _wave_upload(self, per_row: list[torch.Tensor], shifts: tuple[int, ...]) -> ttnn.Tensor:
        """``[rows * cols, W]`` int32 ROW_MAJOR, one row per chip (row-major chip order): mesh row r's rows shifted by
        ``shifts[r]`` blocks (padded back to the width with the last block, never written)."""
        rows = []
        for r, t in enumerate(per_row):
            b0 = int(shifts[r])
            rows.append(t if b0 == 0 else torch.cat([t[:, b0:], t[:, -1:].expand(-1, b0)], dim=1))
        host = torch.cat(rows, dim=0).contiguous()
        return self.mesh.upload(host, ttnn.int32, rows=0, cols=0, layout=ttnn.ROW_MAJOR_LAYOUT)

    def write_pools_wave(self, pooled: ttnn.Tensor, user: int, start_pools: tuple[int, ...], idle: tuple[int, ...]):
        """``write_pools`` for a wave: mesh row r writes ``pooled`` (its own ``[1,1,n,D]``) at ITS pool start
        ``start_pools[r]`` (tile-aligned: a wave chunk starts at a multiple of 32 * P) through per-chip page-table
        views; an idle row writes into its dummy / scratch pools."""
        if any(sp % TILE for sp in start_pools):
            raise ValueError(f"wave pool starts {start_pools} must be multiples of {TILE} (chunk starts of 32 * P)")
        n = pooled.shape[-2]
        n_pad = -(-n // TILE) * TILE
        if n_pad != n:
            pooled = ttnn.pad(pooled, [(0, 0), (0, 0), (0, n_pad - n), (0, 0)], value=0.0)
        if pooled.dtype != self.pool_dtype:
            pooled = ttnn.typecast(pooled, self.pool_dtype)
        shifts = tuple(sp // TILE for sp in start_pools)
        views = [self._wave_upload(self._wave_target_rows(self._pt_host, user, idle), shifts)]
        if self.partition and self.scratch is not None:
            scratch = self._pt_host[self.scratch : self.scratch + 1].expand(self.mesh.cols, -1)
            views.append(self._wave_upload([scratch] * self.mesh.rows, shifts))
        for view in views:
            ttnn.experimental.paged_fill_cache(self.pool_cache, pooled, view, batch_idx=0)
            ttnn.deallocate(view)

    def write_ring_block_user_wave(self, k_tile: ttnn.Tensor, g_tile: ttnn.Tensor, user: int, idle: tuple[int, ...]):
        """``write_ring_block_user`` for a wave: every row writes its own ring block of slot ``user`` (an idle row its
        dummy / scratch ring block)."""
        tables = [self._wave_upload(self._wave_target_rows(self._ring_ids_host, user, idle), (0,) * self.mesh.rows)]
        if self.partition and self.scratch is not None:
            scratch = self._ring_ids_host[self.scratch : self.scratch + 1].expand(self.mesh.cols, -1)
            tables.append(self._wave_upload([scratch] * self.mesh.rows, (0,) * self.mesh.rows))
        for pt in tables:
            for ring, tile in ((self.ring_k, k_tile), (self.ring_g, g_tile)):
                ttnn.experimental.paged_fill_cache(ring, tile, pt, batch_idx=0)
            ttnn.deallocate(pt)

    def _pool_table_row(self, user: int) -> ttnn.Tensor:
        """[1, pool_blocks] row ``user`` of the identity pool page table (a device slice, cached by the caller)."""
        return ttnn.slice(self.page_table, [user, 0], [user + 1, self.pool_blocks])

    def prefill_ring_tables(self, user: int) -> list[ttnn.Tensor]:
        """[1, 1] ring page tables a prefill tail write of ``user`` goes to (owner/scratch or owner/dummy per chip,
        plus the scratch block with a scratch slot)."""
        if not self.partition:
            return [self._user_pts[user]]
        key = ("ring", user)
        own = lazy_const(
            self.mesh,
            self._prefill_views,
            key,
            lambda: self.mesh.upload(
                self._owner_rows(user, self._ring_ids_host), ttnn.int32, cols=0, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
        )
        if self.scratch is None:
            return [own]
        return [own, self._user_pts[self.scratch]]

    def pools_batched(self) -> ttnn.Tensor:
        """[max_batch, 1, pools_per_user, D] view for scoring (taken after the latest writes)."""
        return ttnn.reshape(self.pool_cache, [self.max_batch, 1, self.pools_per_user, self.D])

    def write_pools(self, pooled: ttnn.Tensor, user: int, start_pool: int = 0, slot=None):
        """pooled [1,1,n,D] bf16 -> pools [start_pool, start_pool + n) of ``user``. The write is
        tile-granular (paged_fill_cache copies whole 32-pool tiles from the block ``start_pool // 32``
        on), so a start inside a tile first re-reads that tile's earlier pools from the cache and the
        rows past ``n`` are zero-padded (they are pools not yet complete, never selectable).
        ``slot`` (``trace_prefill.PrefillSlot``, start_pool == 0): the slot index comes from its device
        tensors (trace-replayable for any user)."""
        n = pooled.shape[-2]
        b0, r0 = start_pool // TILE, start_pool % TILE
        if slot is not None and start_pool:
            raise ValueError("a device-slot pool write starts at pool 0")
        if r0:
            blk = (
                self.prefill_slot(user) * self.pool_blocks + b0
            )  # identity page table: a slot's pool blocks are contiguous
            prefix = ttnn.reshape(
                ttnn.slice(self.pool_cache, [blk, 0, 0, 0], [blk + 1, 1, r0, self.D]), [1, 1, r0, self.D]
            )
            if prefix.dtype != ttnn.bfloat16:
                prefix = ttnn.typecast(prefix, ttnn.bfloat16)
            pooled = ttnn.concat([prefix, pooled], dim=-2)
            ttnn.deallocate(prefix)
            n += r0
        n_pad = -(-n // TILE) * TILE
        if n_pad != n:
            pooled = ttnn.pad(pooled, [(0, 0), (0, 0), (0, n_pad - n), (0, 0)], value=0.0)
        if pooled.dtype != self.pool_dtype:  # paged_fill_cache wants the cache dtype (paged_update_cache casts itself)
            pooled = ttnn.typecast(pooled, self.pool_dtype)
        if slot is not None:
            ttnn.experimental.paged_fill_cache(
                self.pool_cache, pooled, self.page_table, batch_idx_tensor=slot.cache_idx
            )
            if self.scratch is not None:
                ttnn.experimental.paged_fill_cache(
                    self.pool_cache, pooled, self.page_table, batch_idx_tensor=slot.scratch_idx
                )
        elif b0 == 0 and not self.partition:
            ttnn.experimental.paged_fill_cache(self.pool_cache, pooled, self.page_table, batch_idx=user)
        else:
            # fixed-width page-table view(s) of the target slot's blocks from b0 on: one fill program for every start
            for view in self.prefill_pool_views(user, b0):
                ttnn.experimental.paged_fill_cache(self.pool_cache, pooled, view, batch_idx=0)
                if b0:
                    ttnn.deallocate(view)

    def update_pools(self, pooled_rows: ttnn.Tensor, pool_ids: ttnn.Tensor):
        """pooled_rows [1,B,1,D] (one row per user slot), pool_ids [B] int32 -> pool rows (all users)."""
        sh = height_shard_rows(self.device, pooled_rows)
        ttnn.experimental.paged_update_cache(
            self.pool_cache, sh, update_idxs_tensor=pool_ids, page_table=self._decode_pt
        )
        ttnn.deallocate(sh)

    def write_ring_rows(
        self, k_rows: ttnn.Tensor, g_rows: ttnn.Tensor, slots: ttnn.Tensor, page_table: ttnn.Tensor | None = None
    ):
        """k_rows/g_rows [1,N,1,D] bf16, slots [N] int32 (position % P). Row n goes to ring block
        ``page_table[n, 0]`` (default: user n), row ``slots[n]``."""
        pt = self._decode_ring_pt if page_table is None else page_table
        for ring, rows in ((self.ring_k, k_rows), (self.ring_g, g_rows)):
            sh = height_shard_rows(self.device, rows)
            ttnn.experimental.paged_update_cache(ring, sh, update_idxs_tensor=slots, page_table=pt)
            ttnn.deallocate(sh)

    def write_ring_block(self, k_tile: ttnn.Tensor, g_tile: ttnn.Tensor, slot):
        """k_tile / g_tile [1,1,32,D] bf16 (row j = the ring row of slot j, rows >= P unused) -> the whole
        ring block of the slot ``slot`` names on device (the traced prefill's tail write: one tile copy
        per ring instead of one row write per tail position)."""
        for ring, tile in ((self.ring_k, k_tile), (self.ring_g, g_tile)):
            ttnn.experimental.paged_fill_cache(ring, tile, self.ring_page_table, batch_idx_tensor=slot.cache_idx)
            if self.scratch is not None:
                ttnn.experimental.paged_fill_cache(ring, tile, self.ring_page_table, batch_idx_tensor=slot.scratch_idx)

    def write_ring_block_user(self, k_tile: ttnn.Tensor, g_tile: ttnn.Tensor, user: int):
        """``write_ring_block`` for a host-side ``user``: the ring block(s) a prefill of that user writes
        (``prefill_ring_tables``: its own block, or the owner / scratch blocks per chip when partitioned)."""
        for pt in self.prefill_ring_tables(user):
            for ring, tile in ((self.ring_k, k_tile), (self.ring_g, g_tile)):
                ttnn.experimental.paged_fill_cache(ring, tile, pt, batch_idx=0)

    def rings(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Rows 0..P-1 of every decode user's rings: ([B,1,P,D], [B,1,P,D]), B = the local users."""
        B = self.users_per_chip
        return (
            ttnn.slice(self.ring_k, [0, 0, 0, 0], [B, 1, self.P, self.D]),
            ttnn.slice(self.ring_g, [0, 0, 0, 0], [B, 1, self.P, self.D]),
        )


class TtDSAIndexer(LightweightModule):
    def __init__(self, device, cfg: GLM53FlashConfig, idx_state_dict: dict[str, torch.Tensor], mesh=None):
        """``idx_state_dict``: raw tensors of ``self_attn.indexer.`` (prefix stripped). ``mesh`` (``tt/mesh.py``
        ``GLMMesh``): needed by the slab-local sequence-parallel prefill, whose per-row query positions are
        uploaded as row-sharded constants (``select(row_starts=...)``)."""
        self.device = device
        self.mesh = mesh
        self.cfg = cfg
        self.Hi, self.D, self.P = cfg.index_n_heads, cfg.index_head_dim, cfg.index_kpool
        sd = idx_state_dict
        # softmax scale folded into wq_b, the n_heads^-0.5 into weights_proj (relu is positively homogeneous)
        self.w_qb = _upload(
            device, (sd["wq_b.weight"].float() * cfg.index_softmax_scale).t(), ttnn.bfloat16
        )  # [1536, Hi*D]
        self.w_k = _upload(device, sd["wk.weight"].t(), ttnn.bfloat16)  # [4096, D]
        self.knorm_w = _upload(device, sd["k_norm.weight"].float().reshape(1, 1, 1, -1), ttnn.bfloat16)
        self.knorm_b = _upload(device, sd["k_norm.bias"].float().reshape(1, 1, 1, -1), ttnn.bfloat16)
        self.w_weights = _upload(
            device, (sd["weights_proj.weight"].float() * self.Hi**-0.5).t(), ttnn.bfloat16
        )  # [4096, Hi]
        self.w_gate = _upload(device, sd["index_kpool_compress_gate"].t(), ttnn.bfloat16)  # [4096, D]
        self.ape = _upload(
            device, sd["index_kpool_compress_ape"].float().reshape(1, 1, self.P, self.D), ttnn.float32
        )  # [1,1,P,D]
        self.hifi2 = _compute(device, ttnn.MathFidelity.HiFi2, packer_l1_acc=True)
        self.fp32 = _compute(device)
        self.static_maintenance_calls = 0  # eager chunks that took the shape-static path (tests)
        self.score_cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=0)
        self._grid = device.compute_with_storage_grid_size()
        self._mesh_rows = tuple(device.shape)[0] if isinstance(device, ttnn.MeshDevice) else 1
        self._prefill_scorer: IndexScoreDecode | None = None
        self._topk_rows: TopkRows | None = None  # per-row bounded top-k (the multi-row prefill selection)
        # test hook: emulate mesh row r of a multi-row mesh on one chip (the shared scorer's chunk start + r*Sq
        # offset applied explicitly, the selection bounded per row as on the mesh)
        self.row_offset_sim = 0
        # the scored width of a prefill sub-block rounds up to the prefix bucket in pools (GLM53_PREFIX_BUCKET / P,
        # 0 = exact): the crop of the score block and the custom scorer's column count then key their programs on
        # the bucket, not on the chunk's position (the 2026-10-28 served 64K prompts: the chunk at 32768 was a new
        # crop width under the sealed program cache). The extra columns are causally invisible (-inf) to every query.
        self.prefix_bucket_pools = int(os.environ.get(PREFIX_BUCKET_ENV, "0")) // self.P
        self._const_cache: dict[tuple, ttnn.Tensor] = {}
        self.stage_timer = None  # TtDSA.set_stage_timer installs a StageTimer (test instrumentation)

    def _mark(self, name: str):
        if self.stage_timer is not None:
            self.stage_timer.mark(name)

    # ------------------------------------------------------------- stems
    def keys_and_gates(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """x [1,1,T,4096] bf16 -> (k [1,1,T,D] bf16 (LayerNorm'd), g [1,1,T,D] bf16 gate logits). The narrow
        stems run through ``prefill_linear`` (fixed K blocking) so a position's pooled key is the same whatever the
        chunk length it was prefilled in (the single-shot / chunked equality; decode rows keep the default)."""
        k = prefill_linear(x, self.w_k, self._grid, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        k = ttnn.layer_norm(k, weight=self.knorm_w, bias=self.knorm_b, epsilon=1e-6, compute_kernel_config=self.fp32)
        g = prefill_linear(x, self.w_gate, self._grid, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        return k, g

    def _pool_groups(self, kp: ttnn.Tensor, gp: ttnn.Tensor) -> ttnn.Tensor:
        """kp, gp [X, n, P, D] bf16 -> pooled [X, n, 1, D] bf16 (softmax over the P tokens of a pool)."""
        kf = ttnn.typecast(kp, ttnn.float32)
        logits = ttnn.add(ttnn.typecast(gp, ttnn.float32), self.ape)  # + ape[j] per in-pool position
        prob = ttnn.softmax(logits, dim=-2, compute_kernel_config=self.fp32)
        pooled = ttnn.sum(ttnn.multiply(prob, kf), dim=-2, keepdim=True)
        return ttnn.typecast(pooled, ttnn.bfloat16)

    def pool(self, k: ttnn.Tensor, g: ttnn.Tensor) -> ttnn.Tensor:
        """k, g [1,1,n*P,D] (complete pools only) -> pooled keys [1,1,n,D] bf16."""
        n = k.shape[-2] // self.P
        pooled = self._pool_groups(ttnn.reshape(k, [1, n, self.P, self.D]), ttnn.reshape(g, [1, n, self.P, self.D]))
        return ttnn.reshape(pooled, [1, 1, n, self.D])

    def pool_rings(self, ring_k: ttnn.Tensor, ring_g: ttnn.Tensor) -> ttnn.Tensor:
        """ring_k, ring_g [B,1,P,D] -> pooled rows [1,B,1,D] bf16 (one pooled key per user)."""
        B = ring_k.shape[0]
        return ttnn.reshape(self._pool_groups(ring_k, ring_g), [1, B, 1, self.D])

    def queries_and_weights(
        self,
        x: ttnn.Tensor,
        q_resid: ttnn.Tensor,
        w: ttnn.Tensor | None = None,
        *,
        rows_form: bool = False,
        free_view: bool = False,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """-> q [1,Hi,T,D] bf16 (scale folded), w [1,1,T,Hi] bf16 (n_heads^-0.5 folded; ``w`` given =
        already computed by the caller's fused stem). ``rows_form``: q as [1,T,Hi,D] instead (query t's heads as
        the rows of its tile row: the custom scorer's layout, ``select(q_rows_form=True)``). ``free_view`` (decode,
        T <= 32, not ``rows_form``): the head-major [1,Hi,T,D] as a free view of the projection's packed tile row
        (tile j of [1,1,T,Hi*D] is tile j % (D/32) of head j // (D/32): the same tile sequence), instead of the
        reshape copy + transpose (the ``GLM53_DSA_DECODE_STATE`` block)."""
        T = x.shape[-2]
        if T > TILE and os.environ.get("GLM53_PREFILL_STEMS", "fixed") == "fixed":
            # M-independent (fixed K blocking) like the k / gate stems: the index queries decide the top-k, and a
            # one-ulp difference between chunk / slab lengths flips near-tie selections (tt/dsa.py ``_stem``)
            q = prefill_linear(
                q_resid, self.w_qb, self._grid, row_block=1024, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2
            )
        else:
            q = ttnn.linear(q_resid, self.w_qb, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,T,Hi*D]
        if free_view and not rows_form and T <= TILE:
            q = ttnn.experimental.view(q, [1, self.Hi, T, self.D])  # [1,Hi,T,D]: the packed tiles re-read
        else:
            q = ttnn.reshape(q, [1, T, self.Hi, self.D])  # [1,T,Hi,D]
            if not rows_form:
                q = ttnn.transpose(q, 1, 2)  # [1,Hi,T,D]
        if w is None:  # [1,1,T,Hi]; fixed K blocking for prefill rows (see keys_and_gates)
            w = prefill_linear(x, self.w_weights, self._grid, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        return q, w

    # ------------------------------------------------------- maintenance
    def maintain_prefill(
        self,
        x: ttnn.Tensor | None,
        cache: TtIndexCache,
        user: int,
        seq_len: int,
        start: int = 0,
        slot=None,
        ring_sel: ttnn.Tensor | None = None,
        kg=None,
    ):
        """Maintain the index for a prefill chunk holding positions ``[start, start + seq_len)`` of
        ``user`` (x row i = position start + i; rows past ``seq_len`` are padding): write every pool
        that becomes complete and leave the rows of the still-incomplete last pool in the ring.

        The chunk boundary may fall inside a pool (``start % P != 0``): that pool's earlier rows are in
        the ring (written as the previous chunk's tail), so it is completed from the ring rows plus
        this chunk's first ``P - start % P`` rows; a chunk that ends inside it only extends the ring.
        With ``start == 0`` this is the single-shot maintenance (pools from 0, tail rows to slots 0..).

        ``slot`` + ``ring_sel`` (the traced fixed-length prefill, tt/trace_prefill.py; start == 0): the
        prompt length is not a host value, so EVERY pool of the padded chunk is written (the pools past
        the prompt are rewritten from the ring by decode before they become selectable: pool p is
        selectable only once position 4p+3 was decoded) and the ring block gets the rows ``ring_sel``
        selects (the last prompt position of each ``% P`` class, chosen on the host as a one-hot
        ``[1,1,32,T_pad]`` selector and applied with an exact HiFi4 matmul); the slot index comes from
        ``slot``'s device tensors.
        ``kg = (k_idx, g_idx)``: the chunk's stems ``[1,1,T_pad,D]`` already computed by the caller (the
        slab-local sequence-parallel prefill gathers them over the mesh rows; ``x`` is then unused); they
        are consumed (deallocated) here either way."""
        P, D = self.P, self.D
        k_idx, g_idx = self.keys_and_gates(x) if kg is None else kg  # [1,1,T_pad,D]
        if slot is not None:
            T = k_idx.shape[-2]
            if start or T % (TILE * P):
                raise ValueError(f"traced index maintenance: start == 0 and T_pad % {TILE * P} == 0 (got {start}, {T})")
            pooled = self.pool(k_idx, g_idx)  # [1,1,T_pad/P,D]: every pool of the padded chunk
            cache.write_pools(pooled, user, start_pool=0, slot=slot)
            ttnn.deallocate(pooled)
            k_tile = ttnn.matmul(ring_sel, k_idx, dtype=ttnn.bfloat16, compute_kernel_config=self.fp32)  # [1,1,32,D]
            g_tile = ttnn.matmul(ring_sel, g_idx, dtype=ttnn.bfloat16, compute_kernel_config=self.fp32)
            cache.write_ring_block(k_tile, g_tile, slot)
            for t in (k_idx, g_idx, k_tile, g_tile):
                ttnn.deallocate(t)
            return
        T = k_idx.shape[-2]
        if (
            os.environ.get(STATIC_PREFILL_ENV, "1") == "1"
            and start % (TILE * P) == 0
            and T % (TILE * P) == 0
            and start // P + T // P <= cache.pools_per_user
        ):
            # Shape-static maintenance (every served chunk: starts at a multiple of 32*P, padded to a warmed bucket).
            # The offset path below pools only the complete pools: it slices the stems to lead + n_pools*P rows and
            # reshapes THAT, a kernel variant per prompt length (% P) that is built at request time, after the trace
            # captures (the 2026-10-28 served image hang: tt-triage = reader/writer_reshape_tiled stuck on every
            # chip at the first 286-token request, the warm-up prompt being 288). Here every op's shape depends on
            # T only, exactly like the traced maintenance.
            self._maintain_static(cache, user, k_idx, g_idx, seq_len, start)
            return
        end = start + seq_len
        r = start % P  # rows of pool start // P already in the ring
        lead = min((P - r) % P, seq_len)  # this chunk's rows of that straddling pool
        pools, first_pool = [], None
        if r and lead < P - r:
            # the chunk ends inside the straddling pool: extend the ring at slots r .. r+lead-1
            self._write_tail_rows(cache, user, k_idx, g_idx, row0=0, n=lead, slot0=r)
            ttnn.deallocate(k_idx)
            ttnn.deallocate(g_idx)
            return
        if r:
            # complete the straddling pool start // P from the ring rows [0, r) + the chunk rows [0, lead)
            s_ = cache.prefill_slot(user)
            rk = ttnn.reshape(ttnn.slice(cache.ring_k, [s_, 0, 0, 0], [s_ + 1, 1, r, D]), [1, 1, r, D])
            rg = ttnn.reshape(ttnn.slice(cache.ring_g, [s_, 0, 0, 0], [s_ + 1, 1, r, D]), [1, 1, r, D])
            kp = ttnn.concat([rk, ttnn.slice(k_idx, [0, 0, 0, 0], [1, 1, lead, D])], dim=-2)
            gp = ttnn.concat([rg, ttnn.slice(g_idx, [0, 0, 0, 0], [1, 1, lead, D])], dim=-2)
            pools.append(self.pool(kp, gp))  # [1,1,1,D]
            first_pool = start // P
            for t in (rk, rg, kp, gp):
                ttnn.deallocate(t)
        a0 = start + lead  # first pool-aligned position of the chunk (== end when nothing follows)
        n_pools = (end - a0) // P
        if n_pools > 0:
            pools.append(
                self.pool(
                    ttnn.slice(k_idx, [0, 0, lead, 0], [1, 1, lead + n_pools * P, D]),
                    ttnn.slice(g_idx, [0, 0, lead, 0], [1, 1, lead + n_pools * P, D]),
                )
            )
            if first_pool is None:
                first_pool = a0 // P
        if pools:
            pooled = pools[0] if len(pools) == 1 else ttnn.concat(pools, dim=-2)
            cache.write_pools(pooled, user, start_pool=first_pool)
        # rows of the incomplete last pool [a0 + n_pools*P, end) -> ring slots 0..tail-1 (pos % P == slot)
        tail = end - (a0 + n_pools * P)
        self._write_tail_rows(cache, user, k_idx, g_idx, row0=lead + n_pools * P, n=tail, slot0=0)
        ttnn.deallocate(k_idx)
        ttnn.deallocate(g_idx)

    def maintain_prefill_wave(
        self, cache: TtIndexCache, user: int, k_idx, g_idx, starts: tuple[int, ...], valids: tuple[int, ...]
    ):
        """The shape-static maintenance for a wave (``doc/mesh/row_ownership.md``): mesh row r's stems ``k_idx`` /
        ``g_idx`` ``[1,1,T,D]`` are ITS chunk ``[starts[r], starts[r] + valids[r])`` (0 valid rows on an idle row);
        every pool of the padded chunk is written from the row's own pool ``starts[r] // P`` (a multiple of 32: wave
        chunks start at multiples of 32 * P), the ring block gets the row's last row of each ``% P`` class through a
        ROW-SHARDED one-hot selector (``[rows, 1, 32, T]``: row r's ``ring_selector(valids[r], T)``, zeros on an idle
        row). The program set depends on T only."""
        P = self.P
        T = k_idx.shape[-2]
        rows = self.mesh.rows
        if len(starts) != rows or len(valids) != rows:
            raise ValueError(f"{len(starts)} starts / {len(valids)} valids for a {rows}-row mesh")
        if T % (TILE * P) or any(s0 % (TILE * P) for s0 in starts):
            raise ValueError(f"wave index maintenance: T {T} and every start {starts} must be multiples of {TILE * P}")
        if max(starts) // P + T // P > cache.pools_per_user:
            raise ValueError(f"wave chunk past the pool capacity: {max(starts)} + {T} > {cache.pools_per_user * P}")
        idle = tuple(r for r, v in enumerate(valids) if v == 0)
        pooled = self.pool(k_idx, g_idx)  # [1,1,T/P,D]
        cache.write_pools_wave(pooled, user, tuple(s0 // P for s0 in starts), idle)
        ttnn.deallocate(pooled)
        sel = torch.cat([ring_selector(v, T, P) if v else torch.zeros(1, 1, TILE, T) for v in valids], dim=0)
        sel_dev = self.mesh.upload(sel, ttnn.bfloat16, rows=0)  # [1,1,32,T] per chip: its row's selector
        k_tile = ttnn.matmul(sel_dev, k_idx, dtype=ttnn.bfloat16, compute_kernel_config=self.fp32)  # [1,1,32,D]
        g_tile = ttnn.matmul(sel_dev, g_idx, dtype=ttnn.bfloat16, compute_kernel_config=self.fp32)
        cache.write_ring_block_user_wave(k_tile, g_tile, user, idle)
        for t in (sel_dev, k_idx, g_idx, k_tile, g_tile):
            ttnn.deallocate(t)
        self.static_maintenance_calls += 1

    def _maintain_static(self, cache: TtIndexCache, user: int, k_idx, g_idx, seq_len: int, start: int):
        """The traced path's maintenance with host-side ``user`` / ``seq_len`` (``start % 128 == 0``, ``T_pad % 128
        == 0``): every pool of the padded chunk is written from pool ``start // P`` on (the pools past the prompt
        are rewritten from the ring by decode before they become selectable: pool p is selectable only once
        position 4p+3 was decoded) and the ring block gets the chunk's last row of each ``% P`` class through the
        host-built one-hot selector (a buffer upload, no program) and an exact matmul."""
        P = self.P
        T = k_idx.shape[-2]
        pooled = self.pool(k_idx, g_idx)  # [1,1,T/P,D]
        cache.write_pools(pooled, user, start_pool=start // P)
        ttnn.deallocate(pooled)
        sel = ring_selector(seq_len, T, P)
        sel_dev = (
            self.mesh.upload(sel, ttnn.bfloat16) if self.mesh is not None else _upload(self.device, sel, ttnn.bfloat16)
        )
        k_tile = ttnn.matmul(sel_dev, k_idx, dtype=ttnn.bfloat16, compute_kernel_config=self.fp32)  # [1,1,32,D]
        g_tile = ttnn.matmul(sel_dev, g_idx, dtype=ttnn.bfloat16, compute_kernel_config=self.fp32)
        cache.write_ring_block_user(k_tile, g_tile, user)
        for t in (sel_dev, k_idx, g_idx, k_tile, g_tile):
            ttnn.deallocate(t)
        self.static_maintenance_calls += 1

    def _write_tail_rows(self, cache: TtIndexCache, user: int, k_idx, g_idx, *, row0: int, n: int, slot0: int):
        """Chunk rows row0 .. row0+n-1 -> ring block of ``user``, slots slot0 .. slot0+n-1, written ONE
        row per call: paged_update_cache gives every batch entry its own core and each core
        read-modify-writes the whole 32-row tile of the block, so two rows of the same block in one
        call would race."""
        D = self.D
        for j in range(n):
            row = row0 + j
            k_row = ttnn.reshape(ttnn.slice(k_idx, [0, 0, row, 0], [1, 1, row + 1, D]), [1, 1, 1, D])
            g_row = ttnn.reshape(ttnn.slice(g_idx, [0, 0, row, 0], [1, 1, row + 1, D]), [1, 1, 1, D])
            for pt in cache.prefill_ring_tables(user):
                cache.write_ring_rows(k_row, g_row, cache._slot_ids[slot0 + j], page_table=pt)

    def maintain_decode(
        self, x: ttnn.Tensor, cache: TtIndexCache, pool_ids: ttnn.Tensor, ring_slots: ttnn.Tensor, kg=None
    ):
        """Batched, trace-safe: x [1,1,B,4096] (token at position a per user), pool_ids = a // P,
        ring_slots = a % P ([B] int32 device tensors). Writes the ring rows, then pool a // P.
        ``kg = (k_raw, g)``: the raw (un-normed) key and gate projections [1,1,B,D] when the caller's
        fused stem already computed them (the k LayerNorm is applied here)."""
        if kg is None:
            B = x.shape[-2]
            k_idx, g_idx = self.keys_and_gates(x)  # [1,1,B,D]
        else:
            k_raw, g_idx = kg
            B = k_raw.shape[-2]
            k_idx = ttnn.layer_norm(
                k_raw, weight=self.knorm_w, bias=self.knorm_b, epsilon=1e-6, compute_kernel_config=self.fp32
            )
            ttnn.deallocate(k_raw)
        cache.write_ring_rows(
            ttnn.reshape(k_idx, [1, B, 1, self.D]), ttnn.reshape(g_idx, [1, B, 1, self.D]), ring_slots
        )
        ttnn.deallocate(k_idx)
        ttnn.deallocate(g_idx)
        rk, rg = cache.rings()  # [B,1,P,D]
        pooled = self.pool_rings(rk, rg)  # [1,B,1,D]
        ttnn.deallocate(rk)
        ttnn.deallocate(rg)
        cache.update_pools(pooled, pool_ids)
        ttnn.deallocate(pooled)

    # ---------------------------------------------------------- selection
    def select(
        self,
        q: ttnn.Tensor,
        w: ttnn.Tensor,
        cache: TtIndexCache,
        user: int,
        *,
        query_start: int,
        num_pools_valid: int,
        row: int | None = None,
        q_rows_form: bool = False,
        row_starts: tuple[int, ...] | None = None,
        pos_floor: int = 0,
    ) -> ttnn.Tensor:
        """Top-2048 token ids (+ tail, sentinel-padded to a multiple of 128) for queries at absolute
        positions ``query_start + s``. Returns [1,1,T,W] uint32 ROW_MAJOR. With ``row`` only that query
        row of the (tile-aligned) block is selected: the scores are sliced to the row before the 4x
        expansion and the top-k, so a decode query costs one row of the key-wide pass instead of the
        32 rows of its padded block; the result is [1,1,1,W] (identical to row ``row`` of the full one).
        ``q_rows_form``: ``q`` is [1,T,Hi,D] and the scores come from the custom scorer (``_score_prefill_rows``:
        per-row causal bounds, no per-device chunk offset, hence the form for multi-row meshes).
        ``row_starts`` (slab-local sequence-parallel prefill, custom scorer only): one query start per MESH
        ROW -- mesh row r's queries sit at ``max(row_starts[r] + s, pos_floor)``; the causal bounds and the
        tail ids are then built per row and uploaded row-sharded (replicated over the columns), so every
        row selects for its own slab with one program. ``pos_floor`` clamps the positions of rows that are
        attended by the dense path anyway (a mixed chunk's superset rows): they then select a valid full
        pool prefix instead of a short one, and their result is discarded by the caller.

        The prefill loop of ``TtDSA`` uses the two halves separately (``select_pools`` per score sub-block,
        ``expand_pools`` once per attention block); this composition is the decode / single-call form."""
        T = q.shape[1] if q_rows_form else q.shape[-2]
        positions = self._row_positions(T, query_start, row_starts, pos_floor)  # host [R, T] int64 (R = rows or 1)
        top_p = self._rank_pools(
            q,
            w,
            cache,
            user,
            T=T,
            positions=positions,
            query_start=query_start,
            num_pools_valid=num_pools_valid,
            row=row,
            q_rows_form=q_rows_form,
            row_starts=row_starts,
        )
        tail = self._tail_ids(T, query_start, positions=positions)  # [1,1,T,P-1] uint32 (cached per geometry)
        if row is not None:
            tail = ttnn.slice(tail, [0, 0, row, 0], [1, 1, row + 1, self.P - 1])
            T = 1
        return self._expand_pools(top_p, tail, T, free_tail=row is not None)

    def select_pools(
        self,
        q: ttnn.Tensor,
        w: ttnn.Tensor,
        cache: TtIndexCache,
        user: int,
        *,
        query_start: int,
        num_pools_valid: int,
        q_rows_form: bool = False,
        row_starts: tuple[int, ...] | None = None,
        pos_floor: int = 0,
    ) -> ttnn.Tensor:
        """The score + top-k half of ``select`` for a prefill sub-block: the ranked pool ids [1,1,T,512] uint32
        ROW_MAJOR of the T queries (descending score, each row over its own causal prefix), BEFORE the token
        expansion. The expansion (``expand_pools``) is linear in the rows and needs no score block, so the caller
        batches it over several sub-blocks (``TtDSA._sparse_rows``); the score block ([T, pools] bf16, 268 MB per
        512 rows at 1 M positions) lives only inside this call. Same arguments as ``select`` (no ``row``)."""
        T = q.shape[1] if q_rows_form else q.shape[-2]
        positions = self._row_positions(T, query_start, row_starts, pos_floor)
        return self._rank_pools(
            q,
            w,
            cache,
            user,
            T=T,
            positions=positions,
            query_start=query_start,
            num_pools_valid=num_pools_valid,
            q_rows_form=q_rows_form,
            row_starts=row_starts,
        )

    def expand_pools(
        self, top_p: ttnn.Tensor, *, query_start: int, row_starts: tuple[int, ...] | None = None, pos_floor: int = 0
    ) -> ttnn.Tensor:
        """The expansion half of ``select``: ranked pool ids [1,1,T,512] uint32 (consumed; the concatenation of
        consecutive ``select_pools`` sub-blocks starting at ``query_start``) -> the index rows [1,1,T,W] uint32
        ROW_MAJOR (4 token ids per pool, the tail ids of each query's incomplete pool, sentinel padding to a
        multiple of 128). ``row_starts`` / ``pos_floor`` as in ``select`` for the block's first row: the tail ids
        of row r of the mesh are built from ``max(row_starts[r] + s, pos_floor)``."""
        T = top_p.shape[-2]
        positions = self._row_positions(T, query_start, row_starts, pos_floor)
        tail = self._tail_ids(T, query_start, positions=positions)
        return self._expand_pools(top_p, tail, T, free_tail=False)

    def _rank_pools(
        self,
        q: ttnn.Tensor,
        w: ttnn.Tensor,
        cache: TtIndexCache,
        user: int,
        *,
        T: int,
        positions: torch.Tensor,
        query_start: int,
        num_pools_valid: int,
        row: int | None = None,
        q_rows_form: bool = False,
        row_starts: tuple[int, ...] | None = None,
    ) -> ttnn.Tensor:
        """Scores + top-512 pools of the T queries: [1,1,T,512] uint32 ROW_MAJOR ranked pool ids ([1,1,1,512] with
        ``row``). The score block is released here."""
        # Multi-row meshes with the shared scorer: the op treats the mesh rows as sequence-parallel ranks
        # (``seq_shard_axes=[0]``: row r scores with chunk start + r*Sq; the queries are REPLICATED over the
        # tensor-parallel columns). The v1 prefill replicates the prompt over the rows too, so row r >= 1 scores
        # a SUPERSET of its queries' causal pools (up to (start + r*Sq + t + 1)/P instead of (start + t + 1)/P;
        # row 0 exact); the per-row bounded top-k (``_topk_bounded``) then ranks every row over its exact
        # causal prefix only, so every row selects what a single chip selects. ``extra`` = the largest offset in
        # query rows (``row_offset_sim`` emulates row r on one chip); the scored width covers every row's start.
        # Slab-local rows (``row_starts``: mesh row r's queries start at row_starts[r], a contiguous slab of the
        # chunk per row, sub-blocks of T rows inside it): the op's offset r*T is a superset of every row's true
        # start when the chunk start is max_r(row_starts[r] - r*T) (exact on the row attaining it, the last one
        # for equal slabs), so the same bounded top-k with the rows' TRUE bounds (row-sharded) selects exactly.
        bounded = False
        if row_starts is not None and not q_rows_form:
            R = len(row_starts)
            op_start = max(s - r * T for r, s in enumerate(row_starts))
            bounded = R > 1
            # every row's chunk start (op_start + r*T) must lie inside the scored width: cover the last row's end
            kv_len = self._kv_len(cache, max(num_pools_valid, (op_start + R * T) // self.P))
        else:
            op_start = query_start + self.row_offset_sim * T
            extra = 0 if q_rows_form else self._extra_rows()
            bounded = bool(extra)
            kv_len = self._kv_len(cache, num_pools_valid, extra * T)
        if self.prefix_bucket_pools:  # one scored width per prefix bucket (served: no program per chunk position)
            kv_len = min(cache.pools_per_user, -(-kv_len // self.prefix_bucket_pools) * self.prefix_bucket_pools)
        if q_rows_form:
            scores_rm = self._score_prefill_rows(q, w, cache, user, positions=positions, kv_len=kv_len)
            self._mark("index_score")
        else:
            if bounded and row is not None:
                raise NotImplementedError(
                    "per-row decode selection on a multi-row mesh (the traced decode selects on device)"
                )
            scores = ttnn.experimental.indexer_score_dsa(
                q,
                cache.pools_batched(),
                w,
                chunk_start_idx=op_start,
                key_compression_ratio=self.P,
                program_config=self.score_cfg,
                cache_batch_idx=user,
                kv_len=kv_len,
                seq_shard_axes=[0],
            )  # [1,1,T,pools_per_user] bf16, -inf where not visible
            self._mark("index_score")
            # The op scores the allocated width (pools_per_user, the KV pool size: 262 176 at 1 M positions); only
            # the first kv_len columns can be selected, so the expansion + top-k work on that prefix and scale with
            # the user's context rather than the pool size. The crop and the layout change are ONE pass
            # (``untilize_with_unpadding``, end indices inclusive): the former ``ttnn.slice`` + ``to_layout`` pair
            # moved the [T, kv_len] block twice (two 268 MB copies per 512-row sub-block at 1 M); pure data
            # movement either way, bit-identical.
            # indexer_score_dsa emits a ROW_MAJOR block (measured 2026-10-26: untilize_with_unpadding rejects it), so
            # the one-pass crop applies only to a tile-major block; a row-major block takes the plain slice.
            if os.environ.get("GLM53_INDEXER_DEBUG") == "1":  # the shapes the sealed program cache keys on
                logger.info(
                    f"[indexer] rank_pools T={T} kv_len={kv_len} scores={tuple(scores.shape)} {scores.layout} "
                    f"op_start={op_start} num_pools_valid={num_pools_valid} row_starts={row_starts} bounded={bounded}"
                )
            if kv_len < scores.shape[-1]:
                if scores.layout == ttnn.TILE_LAYOUT:
                    scores_rm = ttnn.untilize_with_unpadding(scores, [0, 0, T - 1, kv_len - 1])
                else:
                    scores_rm = ttnn.slice(scores, [0, 0, 0, 0], [1, 1, T, kv_len])
            elif scores.layout == ttnn.TILE_LAYOUT:
                scores_rm = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT)
            else:
                scores_rm = scores
            if scores_rm is not scores:
                ttnn.deallocate(scores)
        row_slice = None
        if row is not None:
            row_slice = ttnn.slice(scores_rm, [0, 0, row, 0], [1, 1, row + 1, scores_rm.shape[-1]])
        self._mark("score_untilize")
        # Every query handed here must see >= index_topk pooled tokens (prefill rows >= 2048, decode positions
        # > 2050): then the top-k block is a full valid prefix and sparse_sdpa, which reads a row up to its
        # FIRST sentinel, sees the appended tail ids. Shorter contexts belong to the dense path (TtDSA).
        k_pools = self.cfg.index_select_pools
        assert kv_len >= k_pools, f"sparse selection needs >= {k_pools} scored pools, got {kv_len}"
        # top-512 POOLS over the pool scores (a 4x-expanded token-level top-2048 selects the same whole
        # pools, at 4x the top-k width plus a single-core repeat_interleave of the row: 10.6 ms vs 1.2 ms
        # per decode row at 262 K pools, and a [S, 4W] bf16 intermediate in prefill)
        if bounded:
            top_p = self._topk_bounded(scores_rm, T, kv_len, positions)  # [1,1,T,512]: each row's exact prefix
        else:
            top_p = ttnn.experimental.topk_large_indices(
                scores_rm if row_slice is None else row_slice, k=k_pools, valid_length=kv_len
            )  # [1,1,T,512] uint32 pool ids (never a sentinel: >= 512 visible pools is the caller's contract)
        self._mark("topk")
        for t in (scores_rm, row_slice):
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)
        return top_p

    def _expand_pools(self, top_p: ttnn.Tensor, tail: ttnn.Tensor, T: int, *, free_tail: bool) -> ttnn.Tensor:
        """Ranked pool ids [1,1,T,512] (consumed) + tail ids [1,1,T,P-1] -> index rows [1,1,T,W] uint32."""
        k_eff = self.cfg.index_topk
        # pool id p -> token ids 4p + j: exact in fp32 (ids < 2^24), [1,1,T,2048] uint32
        top_f = ttnn.typecast(top_p, ttnn.float32)
        top_rep = ttnn.repeat_interleave(top_f, self.P, dim=-1)
        top_tok = ttnn.add(ttnn.multiply(top_rep, float(self.P)), self._pool_offsets(T))
        top = ttnn.typecast(top_tok, ttnn.uint32)
        for t in (top_p, top_f, top_rep, top_tok):
            ttnn.deallocate(t)
        self._mark("score_expand")
        width = k_eff + self.P - 1
        pad = (-width) % 128
        parts = [top, tail]
        if pad:
            parts.append(self._sentinels(T, pad))
        idx = ttnn.concat(parts, dim=-1)
        ttnn.deallocate(top)
        if free_tail:
            ttnn.deallocate(tail)
        self._mark("ids_concat")
        return idx

    @property
    def prefill_scorer_custom(self) -> bool:
        """``GLM53_PREFILL_SCORER``: ``auto`` / ``shared`` (default: the shared ``indexer_score_dsa``, on multi-row
        meshes with the per-row bounded top-k of ``select``) or ``custom`` (the prefill form of
        ``custom_ops/index_score_decode``: every query row re-streams the slot's pooled keys, O(T x N) DRAM traffic:
        10.9 s per 16 K-token chunk per layer at 1 M positions against 0.5 s for the shared op, doc/mesh/prefill_1m_cost.md).
        """
        mode = os.environ.get("GLM53_PREFILL_SCORER", "auto")
        assert mode in ("auto", "custom", "shared"), mode
        return mode == "custom"

    def _row_positions(
        self, T: int, query_start: int, row_starts: tuple[int, ...] | None, pos_floor: int
    ) -> torch.Tensor:
        """Absolute query positions as a host int64 [R, T]: one row (``query_start + s``) in the single-device
        form, one per mesh row (``max(row_starts[r] + s, pos_floor)``) in the slab-local form (R = mesh rows)."""
        s = torch.arange(T, dtype=torch.int64)
        if row_starts is None:
            return (query_start + s).reshape(1, T)
        if self.mesh is None or len(row_starts) != self.mesh.rows:
            raise ValueError(f"row_starts {row_starts} must list one start per mesh row ({self.mesh})")
        return torch.clamp(torch.tensor(row_starts, dtype=torch.int64).reshape(-1, 1) + s, min=pos_floor)

    def _const(self, t: torch.Tensor, dtype, per_row: bool) -> ttnn.Tensor:
        """Upload a host constant: replicated (``per_row=False``, or one chip) or with dim 0 split over the mesh
        rows and replicated over the columns (the slab-local prefill's per-row geometry; dim 0 = R entries)."""
        if per_row and self.mesh is not None and self.mesh.rows > 1:
            return self.mesh.upload(t, dtype, rows=0, layout=ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    def _extra_rows(self) -> int:
        """The largest chunk-start offset of a mesh row in query rows (rows - 1 on a multi-row mesh; the emulated
        row on one chip)."""
        return max(self._mesh_rows - 1, self.row_offset_sim)

    def _kv_len(self, cache: TtIndexCache, num_pools_valid: int, extra_tokens: int = 0) -> int:
        """The scored width in pools: one spare tile beyond the valid pools (the op requires the (32-aligned) chunk
        start to lie inside kv_len*P tokens, and a decode query at position a may sit in a block starting at
        a - a%32 >= 4*pools; the spare rows are causally invisible to the real query: pool p visible iff
        4p+3 <= a), plus the pools covering the row-offset superset of a multi-row mesh (``extra_tokens``)."""
        return min(cache.pools_per_user, max(TILE, -(-(num_pools_valid + 1 + extra_tokens // self.P) // TILE) * TILE))

    def prefill_rows_form(self, cache: TtIndexCache, *, query_start: int, T: int, num_pools_valid: int) -> bool:
        """Whether a prefill sub-block of ``T`` queries at ``query_start`` goes through the custom scorer
        (``queries_and_weights(rows_form=True)`` + ``_score_prefill_rows``): the GLM53_PREFILL_SCORER=custom mode,
        or a sub-block whose multi-row superset would start past the pooled-key allocation (the last chunks of a
        prompt filling the whole allocation: the shared op needs every row's chunk start inside the scored width)."""
        if self.prefill_scorer_custom:
            return True
        extra = self._extra_rows()
        return bool(extra) and (query_start + extra * T) // self.P >= self._kv_len(cache, num_pools_valid, extra * T)

    def _topk_bounded(self, scores_rm: ttnn.Tensor, T: int, kv_len: int, positions: torch.Tensor) -> ttnn.Tensor:
        """Top-512 pools of every score row over its own causal prefix (pools < floor((position + 1) / P),
        clamped to [512, kv_len]) through ``custom_ops/topk_rows``: [1,1,T,kv_len] bf16 ROW_MAJOR -> [1,1,T,512]
        uint32. The columns a mesh row's offset adds beyond a row's bound are never ranked. ``positions`` [R, T]
        (``_row_positions``): R = the mesh rows gives every row its own bounds (row-sharded upload)."""
        if self._topk_rows is None:
            self._topk_rows = TopkRows(self.device, k=self.cfg.index_select_pools)
        k = self.cfg.index_select_pools
        valid = self._bounds_const(
            "topk_bounds", positions, T, lambda p: ((p + 1) // self.P).clamp(k, kv_len), kv_len
        )  # [T, 1] per chip
        out = ttnn.zeros((T, 1, 1, k), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        self._topk_rows(ttnn.reshape(scores_rm, [T, 1, 1, kv_len]), valid, out)
        return ttnn.reshape(out, [1, 1, T, k])

    def _bounds_const(self, name: str, positions: torch.Tensor, T: int, bound, limit: int) -> ttnn.Tensor:
        """The per-row bound tensor [R*T, 1] uint32 (row-sharded over the mesh rows when R > 1) of
        ``bound(positions)``, cached per geometry like ``_tail_ids``: the positions of a sub-block are an
        arithmetic sequence per row clamped at the floor, so (T, first, last, R, limit) identifies them, and a
        prompt's chunks revisit the same keys at every layer. Replaces a host->device upload per sub-block
        per layer (a blocking write on the dispatch path) with a dictionary lookup."""
        R = positions.shape[0]
        key = (name, T, tuple(positions[:, 0].tolist()), tuple(positions[:, -1].tolist()), R, limit)
        return lazy_const(
            self.mesh,
            self._const_cache,
            key,
            lambda: self._const(bound(positions).to(torch.int32).reshape(R * T, 1), ttnn.uint32, per_row=R > 1),
        )

    def _score_prefill_rows(
        self, q: ttnn.Tensor, w: ttnn.Tensor, cache: TtIndexCache, user: int, *, positions: torch.Tensor, kv_len: int
    ) -> ttnn.Tensor:
        """Pool scores of T prefill queries through the batched decode scorer (``custom_ops/index_score_decode``):
        query t is one "user" of the op (its Hi = 32 index heads as the rows of its [32, D] block, ``q`` [1,T,Hi,D]),
        every user scores the same slot ``user`` over the first ``kv_len`` (256-rounded) pools, and the causal bound
        of row t (pools < floor((position + 1) / P)) is written as -inf by the op's per-row mask -- no per-device
        chunk offset, so mesh rows holding replicated or independent prompts score alike (the shared op gives row
        r the start + r*Sq of a sequence-parallel ring); ``positions`` [R, T] (host, ``_row_positions``) with R =
        mesh rows gives every row its own bounds (row-sharded upload). [1,1,T,cols] bf16 ROW_MAJOR, cols >= kv_len."""
        T = q.shape[1]
        W = cache.pools_per_user
        if self._prefill_scorer is None:
            self._prefill_scorer = IndexScoreDecode(self.device)
        unit = TILE * self._prefill_scorer.KC
        cols = min(W, -(-kv_len // unit) * unit)
        valid_rows = self._bounds_const(
            "score_bounds", positions, T, lambda p: torch.clamp((p + 1) // self.P, max=cols), cols
        )  # [T, 1] per chip
        out = ttnn.zeros((T, 1, 1, cols), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        self._prefill_scorer(q, w, cache.pools_batched(), out, valid_rows=valid_rows, slot=user, cols=cols)
        return ttnn.reshape(out, [1, 1, T, cols])

    # ------------------------------------------------- traced multi-user decode
    def decode_chunk_start(self, cache: TtIndexCache) -> int:
        """The fixed chunk start of the traced decode scorer: the last 32-token block of the user's pool
        allocation. ``indexer_score_dsa`` masks pool t for query row s iff t >= floor((start + s + 1) / P), so
        row 0 sees every pool below ``pools_per_user - 8``: the kernel's causal mask is thereby disabled and
        the per-user causal bound (the user's complete pools) is applied by ``select_decode_pools`` instead.
        A constant per cache geometry, so the captured program keeps it across replays."""
        return cache.pools_per_user * self.P - TILE

    def score_decode_row(
        self, q_block: ttnn.Tensor, w_block: ttnn.Tensor, cache: TtIndexCache, user: int
    ) -> ttnn.Tensor:
        """Pool scores of ONE decode query against ``user``'s pooled keys, trace-safe: ``q_block`` [1,Hi,32,D]
        and ``w_block`` [1,1,32,Hi] bf16 TILE carry the query in ROW 0 (the other rows are scored and
        dropped); returns [1,1,1,pools_per_user] bf16 ROW_MAJOR with every pool below the slack visible
        (finite): the user's incomplete / not yet written pools score against their current (finite)
        content and are excluded by the top-k's valid length, never by this op."""
        W = cache.pools_per_user
        scores = ttnn.experimental.indexer_score_dsa(
            q_block,
            cache.pools_batched(),
            w_block,
            chunk_start_idx=self.decode_chunk_start(cache),
            key_compression_ratio=self.P,
            program_config=self.score_cfg,
            cache_batch_idx=user,
            kv_len=W,
            seq_shard_axes=[0],
        )  # [1,1,32,W] bf16 row-major
        row = ttnn.slice(scores, [0, 0, 0, 0], [1, 1, 1, W])
        ttnn.deallocate(scores)
        return row

    def select_decode_pools(self, score_row: ttnn.Tensor, valid_len: ttnn.Tensor) -> ttnn.Tensor:
        """Top ``index_select_pools`` pool ids (descending score) of the [B,1,1,W] score rows over their first
        ``valid_len`` columns: a 1-element uint32 device tensor (one bound for every row) or a [B, 1] tensor
        (one bound per row), read at dispatch (the users' complete pools, trace-replay safe). Returns
        [B,1,1,512] uint32 ROW_MAJOR."""
        return ttnn.experimental.topk_large_indices(
            score_row, k=self.cfg.index_select_pools, valid_length_tensor=valid_len
        )

    def _pool_offsets(self, T: int) -> ttnn.Tensor:
        """fp32 [1,1,T,index_topk] of the in-pool offsets 0..P-1 repeated (cached per T)."""
        key = ("pool_offsets", T)

        def make() -> ttnn.Tensor:
            P, K = self.P, self.cfg.index_topk
            offs = torch.arange(P, dtype=torch.float32).repeat(K // P).reshape(1, 1, 1, K).expand(1, 1, T, K)
            return ttnn.from_torch(
                offs.contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )

        return lazy_const(self.mesh, self._const_cache, key, make)

    def _tail_ids(self, T: int, query_start: int, positions: torch.Tensor | None = None) -> ttnn.Tensor:
        """Token ids of the incomplete tail pool of each query (position a = query_start + s, or row r of
        ``positions`` [R, T] on mesh row r): 4*floor((a+1)/4) + j for j < (a+1) % 4, else the sentinel. Built
        once per geometry from host ints (chunk geometry, not data) and cached on device (row-sharded when R
        = the mesh rows > 1)."""
        if positions is None:
            positions = (torch.arange(T, dtype=torch.int64) + query_start).reshape(1, T)
        R = positions.shape[0]
        key = ("tail", T, tuple(positions[:, 0].tolist()), tuple(positions[:, -1].tolist()), R)

        def make() -> ttnn.Tensor:
            P = self.P
            rows = positions.reshape(R, 1, T, 1)  # a
            base = (rows + 1) // P * P
            ids = torch.stack([base + j for j in range(P - 1)], dim=-1).reshape(R, 1, T, P - 1)
            ids = torch.where(ids <= rows, ids, torch.full_like(ids, SENTINEL))
            return self._const(ids, ttnn.uint32, per_row=R > 1)  # [1,1,T,P-1] per chip

        return lazy_const(self.mesh, self._const_cache, key, make)

    def _sentinels(self, T: int, n: int) -> ttnn.Tensor:
        key = ("sentinel", T, n)
        return lazy_const(
            self.mesh,
            self._const_cache,
            key,
            lambda: ttnn.from_torch(
                torch.full((1, 1, T, n), SENTINEL, dtype=torch.int64),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            ),
        )

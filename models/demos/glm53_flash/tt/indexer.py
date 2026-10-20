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

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.glm53_flash.tt.config import GLM53FlashConfig

TILE = 32
SENTINEL = 0xFFFFFFFF
DECODE_SCORE_SLACK = 8  # pools per user hidden by the traced decode scorer's fixed chunk start (TILE // index_kpool)


def _upload(device, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


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
    (per-user sharded page tables), and reads the scratch slot everywhere."""

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
    ):
        self.device = device
        self.mesh = mesh
        self.D = cfg.index_head_dim
        self.P = cfg.index_kpool
        self.max_batch = max_batch
        self.partition = scratch is not None
        self.users_per_chip = max_batch if users_per_chip is None else users_per_chip
        self.scratch = scratch
        pools = -(-max_seq_len // self.P)
        # tile-aligned rows per user, plus DECODE_SCORE_SLACK pools the traced decode scorer can never select:
        # it scores with a fixed chunk start in the last 32-token block of the allocation (see
        # TtDSAIndexer.score_decode_row), under which the final 8 pools of the row are causally hidden; and at
        # least index_select_pools visible pools, the width the decode top-k ranks for every user (a short
        # allocation then holds 544 pool rows = 136 KB per user)
        pools = max(pools, cfg.index_select_pools)
        # rounded to 256 pools (8 tiles): the batched decode scorer streams the keys in 8-tile chunks
        self.pools_per_user = -(-(pools + DECODE_SCORE_SLACK) // (8 * TILE)) * (8 * TILE)
        self.pool_blocks = self.pools_per_user // TILE
        self.pool_cache = ttnn.zeros(
            (max_batch * self.pool_blocks, 1, TILE, self.D),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pt = torch.arange(max_batch * self.pool_blocks, dtype=torch.int32).reshape(max_batch, self.pool_blocks)
        self.page_table = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        # raw rings: one 32-row block per user, row = position % P (rows >= P unused)
        self.ring_k = ttnn.zeros(
            (max_batch, 1, TILE, self.D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.ring_g = ttnn.zeros(
            (max_batch, 1, TILE, self.D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.ring_page_table = ttnn.from_torch(
            torch.arange(max_batch, dtype=torch.int32).reshape(max_batch, 1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        # constants for the prefill tail writes (one row per call): slot ids and single-user page tables,
        # uploaded once here so no host->device write happens inside a prefill pass
        rm = lambda t: ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        self._slot_ids = [rm(torch.tensor([j], dtype=torch.int32)) for j in range(self.P)]
        self._user_pts = [rm(torch.tensor([[u]], dtype=torch.int32)) for u in range(max_batch)]
        # decode page-table views: the paged ops pair batch entry b with page-table row b, so a decode of the
        # users_per_chip local users needs tables of exactly that many rows (the scratch slot is left out)
        B = self.users_per_chip
        if self.partition:
            self._decode_pt = ttnn.slice(self.page_table, [0, 0], [B, self.pool_blocks])
            self._decode_ring_pt = ttnn.slice(self.ring_page_table, [0, 0], [B, 1])
        else:
            self._decode_pt, self._decode_ring_pt = self.page_table, self.ring_page_table
        self._prefill_views: dict[tuple, ttnn.Tensor] = {}

    # ------------------------------------------------------ user-partitioned prefill
    def prefill_slot(self, user: int) -> int:
        """The slot every column chip READS during the prefill of ``user`` (the scratch slot when partitioned)."""
        return self.scratch if self.partition else user

    def _owner_rows(self, user: int, table_host: torch.Tensor) -> torch.Tensor:
        """[cols, W] host rows: row c = the user's slot row of ``table_host`` on its owner column, the scratch row elsewhere."""
        cols = self.mesh.cols
        owner, slot = user // self.users_per_chip, user % self.users_per_chip
        return torch.stack([table_host[slot] if c == owner else table_host[self.scratch] for c in range(cols)])

    def prefill_pool_views(self, user: int) -> list[ttnn.Tensor]:
        """Page-table views ([1, pool_blocks], batch_idx 0) a prefill of ``user`` fills: its own row when not
        partitioned; partitioned: the per-chip owner/scratch row and the (replicated) scratch row."""
        key = ("pools", user)
        if not self.partition:
            if key not in self._prefill_views:
                self._prefill_views[key] = self._pool_table_row(user)
            return [self._prefill_views[key]]
        if key not in self._prefill_views:
            pt = torch.arange(self.max_batch * self.pool_blocks, dtype=torch.int32).reshape(
                self.max_batch, self.pool_blocks
            )
            self._prefill_views[key] = self.mesh.upload(
                self._owner_rows(user, pt), ttnn.int32, cols=0, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            if ("pools", "scratch") not in self._prefill_views:
                self._prefill_views[("pools", "scratch")] = self.mesh.upload(
                    pt[self.scratch : self.scratch + 1], ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                )
        return [self._prefill_views[key], self._prefill_views[("pools", "scratch")]]

    def _pool_table_row(self, user: int) -> ttnn.Tensor:
        """[1, pool_blocks] row ``user`` of the identity pool page table (a device slice, cached by the caller)."""
        return ttnn.slice(self.page_table, [user, 0], [user + 1, self.pool_blocks])

    def prefill_ring_tables(self, user: int) -> list[ttnn.Tensor]:
        """[1, 1] ring page tables a prefill tail write of ``user`` goes to (owner/scratch per chip + scratch)."""
        if not self.partition:
            return [self._user_pts[user]]
        key = ("ring", user)
        if key not in self._prefill_views:
            ids = torch.arange(self.max_batch, dtype=torch.int32).reshape(self.max_batch, 1)
            self._prefill_views[key] = self.mesh.upload(
                self._owner_rows(user, ids), ttnn.int32, cols=0, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        return [self._prefill_views[key], self._user_pts[self.scratch]]

    def pools_batched(self) -> ttnn.Tensor:
        """[max_batch, 1, pools_per_user, D] view for scoring (taken after the latest writes)."""
        return ttnn.reshape(self.pool_cache, [self.max_batch, 1, self.pools_per_user, self.D])

    def write_pools(self, pooled: ttnn.Tensor, user: int, start_pool: int = 0):
        """pooled [1,1,n,D] bf16 -> pools [start_pool, start_pool + n) of ``user``. The write is
        tile-granular (paged_fill_cache copies whole 32-pool tiles from the block ``start_pool // 32``
        on), so a start inside a tile first re-reads that tile's earlier pools from the cache and the
        rows past ``n`` are zero-padded (they are pools not yet complete, never selectable)."""
        n = pooled.shape[-2]
        b0, r0 = start_pool // TILE, start_pool % TILE
        if r0:
            blk = (
                self.prefill_slot(user) * self.pool_blocks + b0
            )  # identity page table: a slot's pool blocks are contiguous
            prefix = ttnn.reshape(
                ttnn.slice(self.pool_cache, [blk, 0, 0, 0], [blk + 1, 1, r0, self.D]), [1, 1, r0, self.D]
            )
            pooled = ttnn.concat([prefix, pooled], dim=-2)
            ttnn.deallocate(prefix)
            n += r0
        n_pad = -(-n // TILE) * TILE
        if n_pad != n:
            pooled = ttnn.pad(pooled, [(0, 0), (0, 0), (0, n_pad - n), (0, 0)], value=0.0)
        if b0 == 0 and not self.partition:
            ttnn.experimental.paged_fill_cache(self.pool_cache, pooled, self.page_table, batch_idx=user)
        else:
            # page-table view(s) of the target slot's blocks from b0 on (device slices: no host upload in the pass)
            for full in self.prefill_pool_views(user):
                view = full if b0 == 0 else ttnn.slice(full, [0, b0], [1, self.pool_blocks])
                ttnn.experimental.paged_fill_cache(self.pool_cache, pooled, view, batch_idx=0)
                if view is not full:
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

    def rings(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Rows 0..P-1 of every decode user's rings: ([B,1,P,D], [B,1,P,D]), B = the local users."""
        B = self.users_per_chip
        return (
            ttnn.slice(self.ring_k, [0, 0, 0, 0], [B, 1, self.P, self.D]),
            ttnn.slice(self.ring_g, [0, 0, 0, 0], [B, 1, self.P, self.D]),
        )


class TtDSAIndexer(LightweightModule):
    def __init__(self, device, cfg: GLM53FlashConfig, idx_state_dict: dict[str, torch.Tensor]):
        """``idx_state_dict``: raw tensors of ``self_attn.indexer.`` (prefix stripped)."""
        self.device = device
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
        self.score_cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=0)
        self._mesh_rows = tuple(device.shape)[0] if isinstance(device, ttnn.MeshDevice) else 1
        self._const_cache: dict[tuple, ttnn.Tensor] = {}
        self.stage_timer = None  # TtDSA.set_stage_timer installs a StageTimer (test instrumentation)

    def _mark(self, name: str):
        if self.stage_timer is not None:
            self.stage_timer.mark(name)

    # ------------------------------------------------------------- stems
    def keys_and_gates(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """x [1,1,T,4096] bf16 -> (k [1,1,T,D] bf16 (LayerNorm'd), g [1,1,T,D] bf16 gate logits)."""
        k = ttnn.linear(x, self.w_k, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        k = ttnn.layer_norm(k, weight=self.knorm_w, bias=self.knorm_b, epsilon=1e-6, compute_kernel_config=self.fp32)
        g = ttnn.linear(x, self.w_gate, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
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
        self, x: ttnn.Tensor, q_resid: ttnn.Tensor, w: ttnn.Tensor | None = None
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """-> q [1,Hi,T,D] bf16 (scale folded), w [1,1,T,Hi] bf16 (n_heads^-0.5 folded; ``w`` given =
        already computed by the caller's fused stem)."""
        T = x.shape[-2]
        q = ttnn.linear(q_resid, self.w_qb, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,T,Hi*D]
        q = ttnn.transpose(ttnn.reshape(q, [1, T, self.Hi, self.D]), 1, 2)  # [1,Hi,T,D]
        if w is None:
            w = ttnn.linear(x, self.w_weights, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,T,Hi]
        return q, w

    # ------------------------------------------------------- maintenance
    def maintain_prefill(self, x: ttnn.Tensor, cache: TtIndexCache, user: int, seq_len: int, start: int = 0):
        """Maintain the index for a prefill chunk holding positions ``[start, start + seq_len)`` of
        ``user`` (x row i = position start + i; rows past ``seq_len`` are padding): write every pool
        that becomes complete and leave the rows of the still-incomplete last pool in the ring.

        The chunk boundary may fall inside a pool (``start % P != 0``): that pool's earlier rows are in
        the ring (written as the previous chunk's tail), so it is completed from the ring rows plus
        this chunk's first ``P - start % P`` rows; a chunk that ends inside it only extends the ring.
        With ``start == 0`` this is the single-shot maintenance (pools from 0, tail rows to slots 0..)."""
        P, D = self.P, self.D
        k_idx, g_idx = self.keys_and_gates(x)  # [1,1,T_pad,D]
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
    ) -> ttnn.Tensor:
        """Top-2048 token ids (+ tail, sentinel-padded to a multiple of 128) for queries at absolute
        positions ``query_start + s``. Returns [1,1,T,W] uint32 ROW_MAJOR. With ``row`` only that query
        row of the (tile-aligned) block is selected: the scores are sliced to the row before the 4x
        expansion and the top-k, so a decode query costs one row of the key-wide pass instead of the
        32 rows of its padded block; the result is [1,1,1,W] (identical to row ``row`` of the full one)."""
        T = q.shape[-2]
        # one spare tile beyond the valid pools: the op requires the (32-aligned) chunk start to lie inside
        # kv_len*P tokens, and a decode query at position a may sit in a block starting at a - a%32 >= 4*pools.
        # The spare rows are causally invisible to the real query (pool p visible iff 4p+3 <= a).
        kv_len = min(cache.pools_per_user, max(TILE, -(-(num_pools_valid + 1) // TILE) * TILE))
        # On a mesh the op treats every device as a sequence-parallel rank by default (device r along
        # the whole-mesh ring scores with chunk_start + r*Sq); the queries here are REPLICATED over the
        # tensor-parallel columns, so the SP axis is the row axis (extent 1 on a single-row mesh ->
        # every chip uses chunk_start as-is). Multi-row meshes are the Phase-3 SP prefill engine.
        if self._mesh_rows != 1:
            raise NotImplementedError("sparse selection on a multi-row mesh is not wired (sequence-parallel rows)")
        scores = ttnn.experimental.indexer_score_dsa(
            q,
            cache.pools_batched(),
            w,
            chunk_start_idx=query_start,
            key_compression_ratio=self.P,
            program_config=self.score_cfg,
            cache_batch_idx=user,
            kv_len=kv_len,
            seq_shard_axes=[0],
        )  # [1,1,T,pools_per_user] bf16, -inf where not visible
        self._mark("index_score")
        # The op scores the allocated width (pools_per_user, the KV pool size: 262 176 at 1 M positions); only
        # the first kv_len columns can be selected, so the untilize + expansion + top-k work on that prefix
        # and scale with the user's context rather than the pool size.
        scores_valid = scores
        if kv_len < scores.shape[-1]:
            scores_valid = ttnn.slice(scores, [0, 0, 0, 0], [1, 1, T, kv_len])
        scores_rm = ttnn.to_layout(scores_valid, ttnn.ROW_MAJOR_LAYOUT)
        tail = self._tail_ids(T, query_start)  # [1,1,T,P-1] uint32 (cached per geometry)
        row_slice = None
        if row is not None:
            row_slice = ttnn.slice(scores_rm, [0, 0, row, 0], [1, 1, row + 1, scores_rm.shape[-1]])
            tail = ttnn.slice(tail, [0, 0, row, 0], [1, 1, row + 1, self.P - 1])
            T = 1
        self._mark("score_untilize")
        # Every query handed here must see >= index_topk pooled tokens (prefill rows >= 2048, decode positions
        # > 2050): then the top-k block is a full valid prefix and sparse_sdpa, which reads a row up to its
        # FIRST sentinel, sees the appended tail ids. Shorter contexts belong to the dense path (TtDSA).
        k_eff, k_pools = self.cfg.index_topk, self.cfg.index_select_pools
        assert kv_len >= k_pools, f"sparse selection needs >= {k_pools} scored pools, got {kv_len}"
        # top-512 POOLS over the pool scores (a 4x-expanded token-level top-2048 selects the same whole
        # pools, at 4x the top-k width plus a single-core repeat_interleave of the row: 10.6 ms vs 1.2 ms
        # per decode row at 262 K pools, and a [S, 4W] bf16 intermediate in prefill)
        top_p = ttnn.experimental.topk_large_indices(
            scores_rm if row_slice is None else row_slice, k=k_pools, valid_length=kv_len
        )  # [1,1,T,512] uint32 pool ids (never a sentinel: >= 512 visible pools is the caller's contract)
        self._mark("topk")
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
        for t in (scores, scores_valid, scores_rm, row_slice) + ((tail,) if row is not None else ()):
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)
        self._mark("ids_concat")
        return idx

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
        if key not in self._const_cache:
            P, K = self.P, self.cfg.index_topk
            offs = torch.arange(P, dtype=torch.float32).repeat(K // P).reshape(1, 1, 1, K).expand(1, 1, T, K)
            self._const_cache[key] = ttnn.from_torch(
                offs.contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
        return self._const_cache[key]

    def _tail_ids(self, T: int, query_start: int) -> ttnn.Tensor:
        """Token ids of the incomplete tail pool of each query (position a = query_start + s):
        4*floor((a+1)/4) + j for j < (a+1) % 4, else the sentinel. Built once per (T, query_start)
        geometry from host ints (chunk geometry, not data) and cached on device."""
        key = ("tail", T, query_start)
        if key not in self._const_cache:
            P = self.P
            rows = torch.arange(T, dtype=torch.int64).reshape(1, 1, T, 1) + query_start  # a
            base = (rows + 1) // P * P
            ids = torch.stack([base + j for j in range(P - 1)], dim=-1).reshape(1, 1, T, P - 1)
            ids = torch.where(ids <= rows, ids, torch.full_like(ids, SENTINEL))
            self._const_cache[key] = ttnn.from_torch(
                ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
        return self._const_cache[key]

    def _sentinels(self, T: int, n: int) -> ttnn.Tensor:
        key = ("sentinel", T, n)
        if key not in self._const_cache:
            self._const_cache[key] = ttnn.from_torch(
                torch.full((1, 1, T, n), SENTINEL, dtype=torch.int64),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
        return self._const_cache[key]

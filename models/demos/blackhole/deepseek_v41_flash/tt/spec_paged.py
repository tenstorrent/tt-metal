# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Speculative verification (n = 1 + k token rows per user) on the PAGED KV pool of ``paged_ops.PagedKVPool`` (stage 1: <= 512 selected entries, no indexer).

Every (user, j) row is a "virtual user": ``paged_kv_step`` is called with ``nq = n`` (user = row // n, position of the row = base + j) so ONE op

  * writes the n window rows of a user into its ring region (slot = pos % ring_rows, ring_rows >= 128 + k so a rejected speculative write never overwrites an entry the
    next round still needs),
  * (compressed owners) writes the latent of every GROUP-COMPLETING row into the shared page pool (even positions of ratio-2 layers write nothing),
  * builds one ``sparse_sdpa`` index row per virtual user: the window rows <= pos (causal inside the block, no mask) + the compressed entries <= pos.

The block compression of ratio-2 layers (previous-token chain inside the block, ``prev_cs`` commit by the accepted count) is ``SpecCompressedAttention``'s.
"""

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P
from models.demos.blackhole.deepseek_v41_flash.tt.attention import HEAD_DIM
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import DIM, HEADS, DSV41DecodeIndexer
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain
from models.demos.blackhole.deepseek_v41_flash.tt.paged_attention import (
    DSV41PagedAttention,
    DSV41PagedCompressedAttention,
)
from models.demos.blackhole.deepseek_v41_flash.tt.spec_attention import SpecCompressedAttention
from models.demos.blackhole.deepseek_v41_flash.tt.spec_attention import _PagedMixin as _SpecFinish

RING_SPEC = 160  # 128 + margin (k <= 5 needs 133; 160 keeps the ring a multiple of 32)


class SpecIndexer(DSV41DecodeIndexer):
    """Multi-query decode indexer (matmul backend) for verify blocks: R = U * n token rows (user-major, row u*n + j), ONE key slab per USER.

    Queries of the n rows of a user are stacked on the head axis ([U, 1, 32 n, 128] @ K_u^T), the per-row head weights contract the 32-row groups, and the selection
    is the base class's ``select`` with per-ROW valid counts (``st["nvalid"]``, causal inside the block: row j only ranks entries completed by its own position).
    Keys of the block's group-completing rows are written first (one ``paged_update_cache`` per block index, others skipped with -1).
    """

    def __init__(self, *args, n=2, **kw):
        super().__init__(*args, **kw)
        assert self.backend == "matmul"
        self.n = n
        self.R = self.T * n

    def project(self, x, qr, st):
        U, n, R = self.T, self.n, self.R
        w = ttnn.linear(x, self.wproj, compute_kernel_config=self.ckc, core_grid=ttnn.CoreGrid(y=2, x=8))  # [1,1,R,32]
        q = ttnn.linear(
            qr, self.wq_b, compute_kernel_config=self.ckc, core_grid=ttnn.CoreGrid(y=2, x=8)
        )  # [1,1,R,4096]
        q = ttnn.reshape(q, (R, 1, HEADS, DIM))
        q = ttnn.addcmul(ttnn.multiply(q, st["C"]), ttnn.matmul(q, self.P, compute_kernel_config=self.ckc), st["S"])
        if self.fp4_q:
            x4 = ttnn.reshape(ttnn.typecast(q, ttnn.float32), (R, 1, HEADS * 4, 32))
            q = ttnn.typecast(ttnn.reshape(self._fp4_blocks(x4), (R, 1, HEADS, DIM)), ttnn.bfloat16)
        q = ttnn.reshape(q, (U, 1, n * HEADS, DIM))  # the n rows of a user stacked on the head axis
        return q, ttnn.reshape(w, (R, 1, 1, HEADS))

    def score(self, q, w):
        U, n, R = self.T, self.n, self.R
        s = ttnn.matmul(
            q, self.k_cache, transpose_b=True, activation="relu", compute_kernel_config=self.ckc
        )  # [U,1,32n,T]
        s = ttnn.reshape(s, (R, 1, HEADS, self.n_alloc))
        s = ttnn.matmul(w, s, compute_kernel_config=self.ckc)  # [R,1,1(32),T]
        return ttnn.to_layout(ttnn.slice(s, [0, 0, 0, 0], [R, 1, 1, self.n_alloc]), ttnn.ROW_MAJOR_LAYOUT)

    def _ent_per_j(self, st):
        """int32 [U] row-major per block index j: key slab entry written by row j (pos // ratio) when the row completes a group, else -1 (skip)."""
        U, n, R, r = self.T, self.n, self.R, self.ratio
        posf = ttnn.typecast(ttnn.to_layout(ttnn.reshape(st["pos"], [1, 1, 1, R]), ttnn.TILE_LAYOUT), ttnn.float32)
        p1 = ttnn.add(posf, 1.0)
        comp = ttnn.eq(ttnn.multiply(ttnn.floor(ttnn.multiply(p1, 1.0 / r)), float(r)), p1)  # (pos + 1) % r == 0
        ent1 = ttnn.multiply(ttnn.add(ttnn.floor(ttnn.multiply(posf, 1.0 / r)), 1.0), comp)  # entry + 1, 0 = skip
        m = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(ent1, ttnn.ROW_MAJOR_LAYOUT), [U, n]), ttnn.TILE_LAYOUT
        )  # [U, n]
        out = []
        for j in range(n):
            col = ttnn.slice(m, [0, j], [U, j + 1])  # [U,1]
            v = ttnn.subtract(col, 1.0)
            out.append(ttnn.reshape(ttnn.to_layout(ttnn.typecast(v, ttnn.int32), ttnn.ROW_MAJOR_LAYOUT), [U]))
        return out

    def append_key_block(self, lat_pre, st):
        """lat_pre [1,1,R,512] bf16 tile: the block's pooled latents BEFORE RoPE; writes the keys of the group-completing rows."""
        U, n, R = self.T, self.n, self.R
        k = ttnn.linear(lat_pre, self.wk, compute_kernel_config=self.ckc)  # [1,1,R,128]
        k = ttnn.rms_norm(ttnn.typecast(k, ttnn.bfloat16), weight=self.k_norm_w, epsilon=self.k_eps)
        k = ttnn.addcmul(ttnn.multiply(k, st["iCg"]), ttnn.matmul(k, self.P, compute_kernel_config=self.ckc), st["iSg"])
        k = ttnn.typecast(
            self._fp4_blocks(ttnn.reshape(ttnn.typecast(k, ttnn.float32), (1, 1, R * 4, 32))), ttnn.bfloat16
        )
        k_rm = ttnn.reshape(ttnn.to_layout(ttnn.reshape(k, (1, 1, R, DIM)), ttnn.ROW_MAJOR_LAYOUT), (U, n, DIM))
        ents = self._ent_per_j(st)
        for j in range(n):
            kj = ttnn.slice(k_rm, [0, j, 0], [U, j + 1, DIM])  # [U,1,128] row-major
            kj = ttnn.to_layout(ttnn.reshape(kj, (1, U, 1, DIM)), ttnn.TILE_LAYOUT)  # one tile per user, key in row 0
            kj = ttnn.to_memory_config(kj, self._kcfg)
            ttnn.experimental.paged_update_cache(self.k_cache, kj, update_idxs_tensor=ents[j], page_table=None)
            ttnn.deallocate(kj)


class SpecPagedWindowAttention(_SpecFinish, DSV41PagedAttention):
    def __init__(self, md, mesh_config, ccl, w, freqs_cis, kvpool, ring_slot, users_per_row=4, n=2):
        super().__init__(md, mesh_config, ccl, w, freqs_cis, kvpool, ring_slot, users_per_row=users_per_row * n)
        self.U, self.n, self.nq = users_per_row, n, n

    def commit(self, m=None):
        pass


class SpecPagedCompressedAttention(_SpecFinish, DSV41PagedCompressedAttention):
    per_row_rope = (
        True  # verify blocks hold rows at different positions: no fused rows-layout RoPE (one angle for all rows)
    )

    def __init__(
        self,
        md,
        mesh_config,
        ccl,
        w,
        freqs_cis,
        ratio,
        comp_w,
        kvpool,
        ring_slot,
        src_layer,
        users_per_row=4,
        n=2,
        source=None,
        indexer=None,
    ):
        super().__init__(
            md,
            mesh_config,
            ccl,
            w,
            freqs_cis,
            ratio,
            comp_w,
            kvpool,
            ring_slot,
            src_layer,
            users_per_row=users_per_row * n,
            source=source,
            indexer=indexer,
        )
        self.U, self.n, self.nq = users_per_row, n, n
        self.cs_block = None

    _compress_block = SpecCompressedAttention._compress_block
    commit = SpecCompressedAttention.commit

    def init_state(self, B):
        """Empty compressor state: prev_cs = [kv 0 | score -1e9] (position 0 pools only itself)."""
        if self.source is None and self.ratio == 2:
            cs = torch.cat([torch.zeros(B, HEAD_DIM), torch.full((B, HEAD_DIM), -1e9)], dim=-1).reshape(
                1, 1, B, 2 * HEAD_DIM
            )
            self.prev_cs = ttnn.from_torch(
                cs.contiguous(),
                device=self.mesh_device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(2, None), mesh_shape=(self.rows, self.cols)),
            )

    def _compress_keep_pre(self, x, st):
        """block compression; keeps the pooled latent BEFORE RoPE (``_lat_pre``): the indexer key is derived from it."""
        self._lat_pre = None
        orig = self._rope_rows

        def capture(lat, c, s, name="SW"):
            self._lat_pre = ttnn.clone(lat)  # the fused RoPE rotates ``lat`` in place
            return orig(lat, c, s, name)

        self._rope_rows = capture
        try:
            return self._compress_block(x, st)
        finally:
            del self._rope_rows

    def forward(self, x, st):
        owner = self.source is None
        lat = (
            (self._compress_keep_pre(x, st) if self.indexer is not None else self._compress_block(x, st))
            if owner
            else self.source.last_lat
        )
        self.last_lat = lat
        q, kv, k = self._qkv(x, st)
        ttnn.deallocate(kv)
        ttnn.deallocate(k)
        if (
            self.indexer is not None
        ):  # index-source layer: (owners: write the block's keys,) then the per-row top-512 shared with the next layers of this kind
            if owner:
                self.indexer.append_key_block(self._lat_pre, st)
            st["topk_ids"] = self.indexer.forward(x, self._last_qr, st)
        o = self._paged_attend(q, lat if owner else None, st, self.ratio, P.SRC_OFF[self.src_layer], P.WINDOW + 512)
        return self._finish(o, st)


class SpecPagedChain(DSV41DecodeChain):
    """Layer builder for paged spec verification: one pool (users_per_row users per mesh row, ring_rows = 160), every attention layer a ring slot."""

    def __init__(self, mesh_device, users_per_row=4, n=2, ctx=512, log=print):
        super().__init__(
            mesh_device, users_per_row=users_per_row * n, max_comp=32, log=log
        )  # base paged flag stays off (own pool below)
        self.U, self.n, self.ctx = users_per_row, n, ctx
        self.n_users = self.rows * users_per_row
        pages_per_user = -(-ctx // P.PAGE_TOKENS)
        self.kvpool = P.PagedKVPool(
            mesh_device,
            users_per_row,
            num_pages=users_per_row * pages_per_user,
            n_ring_layers=40,
            max_ctx=ctx,
            ring_rows=RING_SPEC,
        )
        for b in range(self.n_users):
            self.kvpool.admit(b, ctx)
        self.kvpool.sync_page_table()
        self.ring_slots = 0
        self.use_indexer = (
            ctx > 512
        )  # ratio-1 layers hold <= 512 selected entries without the indexer (ratio-2: 1024 positions)
        self.index_owner = {}

    def _indexer(self, L, meta, w):
        if not (self.use_indexer and "indexer" in w):
            return None
        iw = w["indexer"]
        r = meta["ratio"]
        idx = SpecIndexer(
            self.md,
            iw,
            w["freqs_cis"],
            users_per_row=self.U,
            n_alloc=-(-(self.ctx // r + 32) // 32) * 32,
            ratio=r,
            key_dtype=ttnn.bfloat8_b,
            fp4_q=True,
            backend="matmul",
            n=self.n,
        )
        if meta["is_kv_source"]:
            idx.set_key_weights(iw["wk"], iw["k_norm"])
            idx.load_keys(torch.zeros(self.n_users, 0, 128))
            self.index_owner[L] = idx
        else:
            idx.k_cache = self.index_owner[meta["kv_source"]].k_cache  # layers 24..36 score against layer 20's keys
        return idx

    def build_layer(self, L, chain, w=None):
        from models.demos.blackhole.deepseek_v41_flash.tt.layer import DSV41Layer
        from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer

        w = w if w is not None else load_layer(L, max_seq_len=self.ctx + 64, with_indexer=self.use_indexer)
        meta = w["meta"]
        slot, self.ring_slots = self.ring_slots, self.ring_slots + 1
        kw = dict(users_per_row=self.U, n=self.n)
        pool = self.kvpool
        if meta["ratio"] == 0:
            attn = SpecPagedWindowAttention(
                self.md, self.mesh_config, self.ccl, w["attn"], w["freqs_cis"], pool, slot, **kw
            )
        elif meta["is_kv_source"]:
            attn = SpecPagedCompressedAttention(
                self.md,
                self.mesh_config,
                self.ccl,
                w["attn"],
                w["freqs_cis"],
                meta["ratio"],
                w["compressor"],
                pool,
                slot,
                L,
                indexer=self._indexer(L, meta, w),
                **kw,
            )
            attn.init_state(self.n_users)
            self.sources[L] = attn
        else:
            attn = SpecPagedCompressedAttention(
                self.md,
                self.mesh_config,
                self.ccl,
                w["attn"],
                w["freqs_cis"],
                meta["ratio"],
                None,
                pool,
                slot,
                meta["kv_source"],
                source=self.sources[meta["kv_source"]],
                indexer=self._indexer(L, meta, w),
                **kw,
            )
        layer = DSV41Layer(
            self.md,
            self.mesh_config,
            self.ccl,
            attn,
            w["norms"],
            w["mhc"],
            w["moe"],
            gate_bias_shift=chain["gate_cutoff"],
            users_per_row=self.T,
            moe_buffers=self.moe_buffers,
        )
        if self.moe_buffers is None:
            self.moe_buffers = layer.moe.decode.buffers
        return layer, attn

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
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain
from models.demos.blackhole.deepseek_v41_flash.tt.paged_attention import (
    DSV41PagedAttention,
    DSV41PagedCompressedAttention,
)
from models.demos.blackhole.deepseek_v41_flash.tt.spec_attention import SpecCompressedAttention
from models.demos.blackhole.deepseek_v41_flash.tt.spec_attention import _PagedMixin as _SpecFinish

RING_SPEC = 160  # 128 + margin (k <= 5 needs 133; 160 keeps the ring a multiple of 32)


class SpecPagedWindowAttention(_SpecFinish, DSV41PagedAttention):
    def __init__(self, md, mesh_config, ccl, w, freqs_cis, kvpool, ring_slot, users_per_row=4, n=2):
        super().__init__(md, mesh_config, ccl, w, freqs_cis, kvpool, ring_slot, users_per_row=users_per_row * n)
        self.U, self.n, self.nq = users_per_row, n, n

    def commit(self, m=None):
        pass


class SpecPagedCompressedAttention(_SpecFinish, DSV41PagedCompressedAttention):
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
            indexer=None,
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

    def forward(self, x, st):
        lat = self._compress_block(x, st) if self.source is None else self.source.last_lat
        self.last_lat = lat
        q, kv, k = self._qkv(x, st)
        ttnn.deallocate(kv)
        ttnn.deallocate(k)
        o = self._paged_attend(
            q, lat if self.source is None else None, st, self.ratio, P.SRC_OFF[self.src_layer], P.WINDOW + 512
        )
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

    def build_layer(self, L, chain, w=None):
        from models.demos.blackhole.deepseek_v41_flash.tt.layer import DSV41Layer
        from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer

        w = w if w is not None else load_layer(L, max_seq_len=self.ctx + 64)
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

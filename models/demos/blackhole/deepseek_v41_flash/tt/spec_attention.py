# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention for speculative VERIFICATION of blocks of ``n = 1 + k`` consecutive positions per user (k = 1..5, a runtime parameter).

Token rows (the "virtual users" of the existing decode path) are ordered user-major: row ``u * n + j`` is block index j of user u,
position ``base[u] + j``. Everything per-token (projections, RoPE, mask rows) is exactly ``DSV41Attention`` / ``DSV41CompressedAttention``
with ``users_per_row = U * n`` token rows. What changes is the KV cache: ONE paged cache per layer ``[U * pages_per_user, 1, page, 512]`` whose
page table gives all n rows of a user the SAME pages, so

  * ``paged_update_cache`` writes the n new K/V rows (positions base..base+k) of a user into its pages,
  * ``paged_scaled_dot_product_attention_decode`` with per-row ``cur_pos`` makes the block causal (row j sees positions <= base + j,
    all of which were written before the SDPA call).

Compressed layers: ring of R = 128 + RING_MARGIN slots + compressed slots in one page; ratio-2 pooling of (previous, current) token with the
previous token of row j = the compressor input of row j-1 (row 0: ``prev_cs``, carried across rounds); rows at even positions write their
junk latent to a trash slot (see spec_state.py); ``commit(m)`` selects the new ``prev_cs`` after the accept decision.
"""

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attention import (
    HEAD_DIM,
    NH,
    WINDOW,
    DSV41Attention,
    DSV41CompressedAttention,
)
from models.demos.blackhole.deepseek_v41_flash.tt.spec_state import RING_MARGIN


def make_page_table(U, n, ppu, device, mesh_rows, mesh_cols, shuffle=False):
    """Page table [rows*U*n, ppu] int32 row-major, sharded over mesh rows (token rows). Row (u, j) -> the pages of user u (same for all j)."""
    perm = torch.randperm(U * ppu) if shuffle else torch.arange(U * ppu)
    per_user = (
        perm.view(U, ppu).to(torch.int32).repeat_interleave(n, dim=0)
    )  # [U*n, ppu], page ids local to this device row's cache
    full = per_user.repeat(mesh_rows, 1)
    return ttnn.from_torch(
        full,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=(0, None), mesh_shape=(mesh_rows, mesh_cols)),
    )


class _PagedMixin:
    def _init_paged(self, U, n, page, total):
        self.U, self.n = U, n
        self.page, self.ppu = page, total // page
        assert total % page == 0
        self.pt = make_page_table(U, n, self.ppu, self.mesh_device, self.rows, self.cols)
        self.cache = self._up(
            torch.zeros(U * self.ppu, 1, page, HEAD_DIM)
        )  # replicated bf16 paged cache (per device: its row's users)

    def _finish(self, o, st):
        """``DSV41Attention._finish`` for any T: ``nlp_concat_heads_decode`` needs a RECTANGULAR core grid for its sharded input (T = 8, 16, 24 or < 8 work;
        T = 10, 12, 20, ... fail with 'bad optional access'): pad the token rows to a multiple of 8 first (the padded rows are ignored).
        """
        T = self.T
        if T <= 8 or T % 8 == 0 or T > 32:
            return super()._finish(o, st)
        Tp = -(-T // 8) * 8
        o = self._rope_heads(o, st["Ch"], st["nSh"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.pad(o, [(0, 0), (0, Tp - T), (0, 0), (0, 0)], 0.0)
        cfg = ttnn.create_sharded_memory_config(
            shape=(32, HEAD_DIM),
            core_grid=ttnn.num_cores_to_corerangeset(Tp, ttnn.CoreCoord(8, 8), row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        c = ttnn.experimental.nlp_concat_heads_decode(ttnn.to_memory_config(o, cfg), num_heads=NH)
        c = ttnn.to_memory_config(c, ttnn.DRAM_MEMORY_CONFIG)
        part = self._lin(self._lin(c, self.wo_a, "OA"), self.wo_b, "OB")
        out = self.mesh_config.allreduce(part, self.ccl, axis=1)
        return ttnn.reshape(out, (1, 1, T, 5120), (1, 1, 32, 5120))

    def _write_paged(self, row, idx_j):
        """row: [1,T,*,512] L1-sharded K/V rows; idx_j: per block index j an int32 [T] index tensor (-1 = skip the row), see SpecStepState."""
        for idx in idx_j:
            ttnn.experimental.paged_update_cache(self.cache, row, update_idxs_tensor=idx, page_table=self.pt)

    def _seed_cache(self, full):
        """full [rows*U, total, 512] host (per-user linear cache) -> paged cache (user u of device row r owns pages [u*ppu, (u+1)*ppu))."""
        rows, cols, U, ppu, page = self.rows, self.cols, self.U, self.ppu, self.page
        t = full.reshape(rows * U * ppu, 1, page, HEAD_DIM)
        self.cache = ttnn.from_torch(
            t.contiguous(),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, None), mesh_shape=(rows, cols)),
        )


class SpecWindowAttention(_PagedMixin, DSV41Attention):
    def __init__(
        self, mesh_device, mesh_config, ccl_manager, w, freqs_cis, users_per_row=4, n=2, max_seq=256, page=256
    ):
        super().__init__(
            mesh_device, mesh_config, ccl_manager, w, freqs_cis, users_per_row=users_per_row * n, max_seq=32
        )
        self.max_seq = max_seq
        self._init_paged(users_per_row, n, page, max_seq)

    def load_window(self, kv_rows):
        """kv_rows [rows*U, S, 512] (positions 0..S-1)."""
        full = torch.zeros(kv_rows.shape[0], self.max_seq, HEAD_DIM)
        full[:, : kv_rows.shape[1]] = kv_rows
        self._seed_cache(full)

    def commit(self, m=None):
        pass

    def forward(self, x, st):
        q, kv, k = self._qkv(x, st)
        self._write_paged(kv, st["pos_j"])
        ttnn.deallocate(kv)
        ttnn.deallocate(k)
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.cache,
            self.cache,
            page_table_tensor=self.pt,
            cur_pos_tensor=st["pos"],
            sliding_window_size=WINDOW,
            attention_sink=self.sinks,
            scale=self.scale,
            program_config=self._sdpa_cfg(128),
            compute_kernel_config=self.ckc_sdpa,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self._finish(o, st)


class SpecCompressedAttention(_PagedMixin, DSV41CompressedAttention):
    def __init__(
        self,
        mesh_device,
        mesh_config,
        ccl_manager,
        w,
        freqs_cis,
        ratio,
        comp_w,
        users_per_row=4,
        n=2,
        max_comp=128,
        index_topk=512,
        source=None,
    ):
        R = WINDOW + RING_MARGIN
        super().__init__(
            mesh_device,
            mesh_config,
            ccl_manager,
            w,
            freqs_cis,
            ratio,
            comp_w,
            users_per_row=users_per_row * n,
            max_comp=max_comp,
            index_topk=index_topk,
            source=source,
        )
        self.R = R
        total = R + max_comp
        self._k_chunk = next(c for c in (128, 64, 32) if total % c == 0)
        self._init_paged(users_per_row, n, total, total)  # one page per user
        self.cs_block = None

    @property
    def comp_cache(self):
        raise NotImplementedError

    def _seed(self, B, window_kv, S, comp_kv):
        """Linear host cache [B, R + max_comp, 512] from the reference ring (slot = pos % 128, S tokens prefilled) and compressed latents."""
        R = self.R
        full = torch.zeros(B, R + self.max_comp, HEAD_DIM)
        for y in range(max(0, S - WINDOW), S):
            full[:, y % R] = window_kv[:, y % WINDOW]
        if comp_kv is not None:
            full[:, R : R + comp_kv.shape[1]] = comp_kv
        return full

    def load_state(self, window_kv, comp_kv, kv_state, score_state, S=None):
        """window_kv [B,128,512] reference ring after S prefilled tokens, comp_kv [B,Lc,512], kv_state/score_state [B,ratio,512]."""
        md = self.mesh_device
        B = window_kv.shape[0]
        if self.source is not None:
            full = self._seed(B, window_kv, S, None)
            self._seed_cache(full)
            # copy the owner's compressed slots on device: concat [ring | comp] along the page dim
            ring = ttnn.slice(self.cache, [0, 0, 0, 0], [self.U * self.ppu, 1, self.R, HEAD_DIM])
            comp = ttnn.slice(
                self.source.cache, [0, 0, self.R, 0], [self.U * self.ppu, 1, self.R + self.max_comp, HEAD_DIM]
            )
            self.cache = ttnn.concat([ring, comp], dim=2)
            return
        self._seed_cache(self._seed(B, window_kv, S, comp_kv))
        if self.ratio > 1:
            prev = (S - 1) % self.ratio
            cs = torch.cat([kv_state[:, prev], score_state[:, prev]], dim=-1).float().reshape(1, 1, B, 2 * HEAD_DIM)
            self.prev_cs = ttnn.from_torch(
                cs.contiguous(),
                device=md,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(self.rows, self.cols)),
            )

    def _compress_block(self, x, st):
        """x [1,1,T,5120] (T = U*n rows) -> RoPE'd latent rows [1,1,T,512]; ratio 2 also records ``cs_block`` for ``commit``."""
        U, n, T = self.U, self.n, self.T
        if self.ratio == 1:
            lat = ttnn.rms_norm(self._lin(x, self.c_wkv, "CP"), weight=self.c_norm, epsilon=self.eps)
        else:
            cs = self._lin(x, self.c_wcat, "CP", dtype=ttnn.float32)  # [1,1,T,1024] fp32 = [kv | score]
            cs3 = ttnn.reshape(ttnn.to_layout(cs, ttnn.ROW_MAJOR_LAYOUT), [U, n, 2 * HEAD_DIM])
            prev = ttnn.reshape(ttnn.to_layout(self.prev_cs, ttnn.ROW_MAJOR_LAYOUT), [U, 1, 2 * HEAD_DIM])
            prev_all = (
                prev if n == 1 else ttnn.concat([prev, ttnn.slice(cs3, [0, 0, 0], [U, n - 1, 2 * HEAD_DIM])], dim=1)
            )
            a = ttnn.to_layout(ttnn.reshape(prev_all, [1, 1, T, 2 * HEAD_DIM]), ttnn.TILE_LAYOUT)
            self.cs_block = cs3
            sl = lambda t, lo: ttnn.slice(t, [0, 0, 0, lo], [1, 1, T, lo + HEAD_DIM])
            d = ttnn.subtract(a, cs)
            pooled = ttnn.addcmul(sl(cs, 0), sl(d, 0), sl(ttnn.sigmoid(d), HEAD_DIM))
            lat = ttnn.rms_norm(ttnn.typecast(pooled, ttnn.bfloat16), weight=self.c_norm, epsilon=self.eps)
        return self._rope_rows(lat, st["Cg"], st["Sg"])

    def commit(self, m=None):
        """After the accept decision: ``prev_cs`` <- compressor input of block index m[u] of every user (m None = all accepted: index n-1).
        m: int32/float tensor [U] on device would select via one-hot (M3); the host-int form is for tests."""
        if self.ratio != 2 or self.source is not None:
            return
        U, n = self.U, self.n
        if m is None:
            sel = ttnn.slice(self.cs_block, [0, n - 1, 0], [U, n, 2 * HEAD_DIM])
        else:  # m: [U,n,1] fp32 row-major one-hot over block index
            prod = ttnn.multiply(
                ttnn.to_layout(self.cs_block, ttnn.TILE_LAYOUT), ttnn.to_layout(m, ttnn.TILE_LAYOUT)
            )  # [U,n,1024] * [U,n,1]
            sel = ttnn.to_layout(ttnn.sum(prod, dim=1, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy(ttnn.to_layout(ttnn.reshape(sel, [1, 1, U, 2 * HEAD_DIM]), ttnn.TILE_LAYOUT), self.prev_cs)

    def forward(self, x, st):
        lat = self._compress_block(x, st) if self.source is None else self.source.last_lat
        self.last_lat = lat
        q, kv, k = self._qkv(x, st, lat)
        self._write_paged(kv, st["pos_ring_j"])
        self._write_paged(k, st["comp_idx_j"])
        ttnn.deallocate(kv)
        ttnn.deallocate(k)
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.cache,
            self.cache,
            page_table_tensor=self.pt,
            is_causal=False,
            attn_mask=st["mask"],
            attention_sink=self.sinks,
            scale=self.scale,
            program_config=self._sdpa_cfg(self._k_chunk),
            compute_kernel_config=self.ckc_sdpa,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self._finish(o, st)

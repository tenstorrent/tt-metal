# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSA (DeepSeek sparse attention, MLA with NoPE) layer for GLM-5.3-Flash on one device.

HF ``Glm5NextTextAttention`` (qk_rope_head_dim = 0, so no rotary anywhere):
    q = q_b(rmsnorm(q_a(x)))                      [T, 64, 256]
    c = rmsnorm(kv_a(x))                           [T, 512]   (the latent, cached per token)
    k_h = c @ Wkb1[h]^T, v_h = c @ Wkb2[h]^T       (kv_b = [64 x (256 k | 256 v), 512])
    o = softmax(q k^T / 16) v -> concat heads -> o_proj
The indexer selects the top 512 complete key pools (2048 tokens) plus the incomplete tail, so a
query at position a sees every visible token while floor((a+1)/4) <= 512, i.e. a <= 2050: there
the layer is exact causal attention (dense path). Beyond, the pooled-key indexer picks the keys
(sparse path, ``TtDSAIndexer`` + ``sparse_sdpa``).

Device design (section 5.3 of the design doc, single chip = Phase 1):
  * prefill: the latent is written to the paged cache with ``paged_fill_cache`` and the pooled-key
    index is maintained. Query rows [0, 2048) run the dense causal SDPA on materialized K/V; rows
    from 2048 on (they always see >= 2048 pooled tokens, so the top-k row is a full valid prefix)
    run the indexer selection + ``sparse_sdpa`` on the absorbed q. A prompt may arrive in
    consecutive chunks (``start`` = the chunk's first absolute position, a block multiple): the
    chunk's latent fills the pages from block ``start // 32``, the index continues from ``start``
    (the ring holds the previous chunk's tail) and the chunk rows attend over the full prefix, the
    dense rows through the windowed causal SDPA (K/V from the cached latent of [0, start) + the
    chunk's, Q offset ``start``) and the sparse rows exactly as in the single-shot path.
  * decode: absorbed MLA. q_lat[h] = q[h] @ Wkb1[h] (256 -> 512), then
    ``paged_flash_multi_latent_attention_decode`` over the latent cache for every user slot
    (exact through position 2050), then o[h] = o_lat[h] @ Wkb2[h]^T (512 -> 256), concat heads,
    o_proj. Users at position >= 2051 get their row replaced by the per-user indexer +
    ``sparse_sdpa`` result (host chunk geometry, not traced). The latent of the new token is
    appended with ``paged_update_cache`` first and the index is maintained for every user.
  * ``sparse_sdpa`` reads an index row up to its FIRST sentinel, so every row handed to it must be
    a contiguous valid prefix: that holds exactly because only queries with >= 2048 visible pooled
    tokens reach it.

Cache: ``[max_blocks, 1, block_size, 512]`` bf16, paged; page_table ``[B, max_blocks_per_seq]`` int32.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.indexer import TtDSAIndexer, TtIndexCache, height_shard_rows

TILE = 32
CORES_PER_USER_DECODE = 4


def _compute(device, fidelity=ttnn.MathFidelity.HiFi2, fp32_acc=True, packer_l1_acc=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


def _upload(device, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def decode_position_tensors(device, host_positions: list[int], kpool: int):
    """(positions, positions // kpool, positions % kpool) as [B] int32 ROW_MAJOR device tensors."""

    def up(vals):
        return ttnn.from_torch(
            torch.tensor(vals, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

    return up(host_positions), up([p // kpool for p in host_positions]), up([p % kpool for p in host_positions])


class TtLatentCache:
    """Paged latent cache for one DSA layer (and its page table) plus the pooled-key index, device-owned."""

    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        *,
        max_batch: int,
        max_seq_len: int,
        block_size: int = 32,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.block_size = block_size
        self.max_batch = max_batch
        self.blocks_per_seq = -(-max_seq_len // block_size)
        self.max_blocks = self.blocks_per_seq * max_batch
        self.dtype = dtype
        self.latent_dim = cfg.kv_lora_rank
        self.cache = ttnn.zeros(
            (self.max_blocks, 1, block_size, self.latent_dim),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # identity page table [B, blocks_per_seq]: user b owns blocks [b*bps, (b+1)*bps)
        pt = torch.arange(self.max_blocks, dtype=torch.int32).reshape(max_batch, self.blocks_per_seq)
        self.set_page_table(pt)
        self.rows_per_user = self.blocks_per_seq * block_size
        # pooled-key index (its own identity table; user u's pools are contiguous regardless of the latent table)
        self.index = TtIndexCache(device, cfg, max_batch=max_batch, max_seq_len=self.rows_per_user)

    def set_page_table(self, page_table: torch.Tensor):
        """Any [max_batch, blocks_per_seq] mapping (vLLM-style); every block id must be unique."""
        assert tuple(page_table.shape) == (self.max_batch, self.blocks_per_seq), page_table.shape
        assert page_table.unique().numel() == page_table.numel(), "page table blocks must be unique"
        self.page_table_host = page_table.to(torch.int32).contiguous()
        self.page_table = ttnn.from_torch(
            self.page_table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )

    def fill(self, latent: ttnn.Tensor, user: int, start: int = 0):
        """latent [1, 1, T_pad, 512] (T_pad % block_size == 0) -> pages of ``user`` from position
        ``start`` (a block multiple): page-table view from block ``start // block_size`` on."""
        src = latent if latent.dtype == self.dtype else ttnn.typecast(latent, self.dtype)
        if start == 0:
            ttnn.experimental.paged_fill_cache(self.cache, src, self.page_table, batch_idx=user)
            return
        if start % self.block_size:
            raise ValueError(f"fill start {start} must be a multiple of the block size {self.block_size}")
        b0 = start // self.block_size
        view = ttnn.slice(self.page_table, [user, b0], [user + 1, self.blocks_per_seq])  # device slice, no upload
        ttnn.experimental.paged_fill_cache(self.cache, src, view, batch_idx=0)
        ttnn.deallocate(view)

    def update(self, latent_rows: ttnn.Tensor, positions: ttnn.Tensor):
        """latent_rows [1, B, 1, 512] (one token per user), positions [B] int32 row-major."""
        sh = height_shard_rows(self.device, latent_rows)
        ttnn.experimental.paged_update_cache(self.cache, sh, update_idxs_tensor=positions, page_table=self.page_table)
        ttnn.deallocate(sh)

    def user_rows(self, user: int, n_rows: int) -> ttnn.Tensor:
        """The first ``n_rows`` (rounded up to whole blocks) latent rows of ``user`` in page-table
        order as a fresh TILE [1, 1, rows, 512] tensor."""
        n_blocks = min(self.blocks_per_seq, -(-n_rows // self.block_size))
        ids = self.page_table_host[user, :n_blocks].tolist()
        if ids == list(range(ids[0], ids[0] + n_blocks)):  # contiguous (identity-like) fast path
            blocks = ttnn.slice(self.cache, [ids[0], 0, 0, 0], [ids[0] + n_blocks, 1, self.block_size, self.latent_dim])
        else:
            parts = [ttnn.slice(self.cache, [b, 0, 0, 0], [b + 1, 1, self.block_size, self.latent_dim]) for b in ids]
            blocks = ttnn.concat(parts, dim=0)
            for p in parts:
                ttnn.deallocate(p)
        return ttnn.reshape(blocks, [1, 1, n_blocks * self.block_size, self.latent_dim])

    def user_rows_rm(self, user: int, n_rows: int) -> ttnn.Tensor:
        """``user_rows`` as a fresh ROW_MAJOR [1, 1, rows, 512] tensor (the sparse_sdpa kv contract)."""
        rows = self.user_rows(user, n_rows)
        rm = ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(rows)
        return rm


class TtDSA(LightweightModule):
    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        attn_state_dict: dict[str, torch.Tensor],
        *,
        layer_idx: int,
        weight_dtype=ttnn.bfloat8_b,
    ):
        """``attn_state_dict``: raw checkpoint tensors of ``self_attn.`` (prefix stripped, fp8 dequantized)."""
        self.device = device
        self.cfg = cfg
        self.layer_idx = layer_idx
        H, Dq, Dv, L = cfg.num_attention_heads, cfg.qk_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
        self.H, self.Dq, self.Dv, self.L = H, Dq, Dv, L
        self.scale = cfg.mla_softmax_scale
        sd = attn_state_dict
        self.w_qa = _upload(device, sd["q_a_proj.weight"].t(), weight_dtype)  # [4096,1536]
        self.qa_norm = _upload(device, sd["q_a_layernorm.weight"].float().reshape(1, 1, 1, -1), ttnn.bfloat16)
        self.w_qb = _upload(device, sd["q_b_proj.weight"].t(), weight_dtype)  # [1536, H*256]
        self.w_kva = _upload(device, sd["kv_a_proj_with_mqa.weight"].t(), weight_dtype)  # [4096, 512]
        self.kva_norm = _upload(device, sd["kv_a_layernorm.weight"].float().reshape(1, 1, 1, -1), ttnn.bfloat16)
        wkb = sd["kv_b_proj.weight"].reshape(H, Dq + Dv, L)  # [H, 512, L]
        wkb1 = wkb[:, :Dq, :]  # k_nope[h] = c @ wkb1[h]^T   -> [H, 256, L]
        wkb2 = wkb[:, Dq:, :]  # v[h]      = c @ wkb2[h]^T   -> [H, 256, L]
        # prefill: K = c @ Wk^T for all heads at once: [T, L] @ [L, H*256]; same for V
        self.w_k_all = _upload(
            device, wkb1.transpose(0, 1).reshape(Dq, H, L).permute(2, 1, 0).reshape(L, H * Dq), weight_dtype
        )
        self.w_v_all = _upload(device, wkb2.permute(2, 0, 1).reshape(L, H * Dv), weight_dtype)
        # decode (absorbed): per-head [H, 256, L] for q -> latent, and [H, L, 256] for latent -> v space
        self.w_absorb_q = _upload(device, wkb1.reshape(1, H, Dq, L), weight_dtype)
        self.w_absorb_o = _upload(device, wkb2.transpose(1, 2).reshape(1, H, L, Dv), weight_dtype)
        self.w_o = _upload(device, sd["o_proj.weight"].t(), weight_dtype)  # [H*256, 4096]
        self.indexer = TtDSAIndexer(
            device, cfg, {k[len("indexer.") :]: v for k, v in sd.items() if k.startswith("indexer.")}
        )
        # prefill rows from here on use the sparse path (they see >= 2048 pooled tokens: full index rows)
        self.sparse_row_start = cfg.index_topk  # 2048
        # decode: dense attention is HF-exact while floor((a+1)/4) <= index_topk/4, i.e. a <= 2050
        self.dense_exact_max_pos = cfg.index_topk + cfg.index_kpool - 2  # 2050
        self.hifi2 = _compute(device)
        self.hifi4 = _compute(device, ttnn.MathFidelity.HiFi4, packer_l1_acc=False)
        grid = device.compute_with_storage_grid_size()
        self.sdpa_prefill_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=128, exp_approx_mode=False
        )
        self.full_grid = grid
        self._decode_cfgs: dict[int, ttnn.SDPAProgramConfig] = {}
        self._cu_windows: dict[int, ttnn.Tensor] = {}  # chunked-prefill SDPA window bounds per key length
        self.sdpa_compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def _decode_sdpa_config(self, batch: int) -> ttnn.SDPAProgramConfig:
        """Flash MLA decode splits each user's heads over cores and tree-reduces the partial results;
        its c_19 staging CB is sized (cores_per_user - 1) * (out + 2) fp32 tiles: 36 tiles * 4 KB * 11
        = 1.6 MB at 12 cores/user with the 512-wide latent. 4 cores/user keeps the op at ~1.2 MB."""
        if batch not in self._decode_cfgs:
            want = min(self.full_grid.x * self.full_grid.y, CORES_PER_USER_DECODE * batch)
            x = min(self.full_grid.x, want)
            y = max(1, min(self.full_grid.y, want // x))
            self._decode_cfgs[batch] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(x, y),
                q_chunk_size=0,
                k_chunk_size=32,
                exp_approx_mode=False,
            )
        return self._decode_cfgs[batch]

    # ---------------------------------------------------------------- shared stems
    def _q_resid(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,4096] -> q_resid [1,1,T,1536] bf16 (q_a -> rmsnorm)."""
        qa = ttnn.linear(x, self.w_qa, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        return ttnn.rms_norm(qa, epsilon=self.cfg.rms_norm_eps, weight=self.qa_norm, compute_kernel_config=self.hifi4)

    def _q(self, q_resid: ttnn.Tensor) -> ttnn.Tensor:
        """q_resid [1,1,T,1536] -> q [1,1,T,H*256] bf16 (q_b)."""
        return ttnn.linear(q_resid, self.w_qb, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)

    def _latent(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,T,4096] -> c [1,1,T,512] bf16 (kv_a -> rmsnorm)."""
        c = ttnn.linear(x, self.w_kva, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        return ttnn.rms_norm(c, epsilon=self.cfg.rms_norm_eps, weight=self.kva_norm, compute_kernel_config=self.hifi4)

    def _absorb_q(self, q: ttnn.Tensor) -> ttnn.Tensor:
        """q [1,1,T,H*Dq] -> q_lat [1,H,T,L] bf16 (per-head q @ Wkb1)."""
        T = q.shape[-2]
        qh = ttnn.transpose(ttnn.reshape(q, [1, T, self.H, self.Dq]), 1, 2)  # [1,H,T,Dq]
        q_lat = ttnn.matmul(qh, self.w_absorb_q, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        ttnn.deallocate(qh)
        return q_lat

    def _o_lat_to_heads(self, o_lat: ttnn.Tensor) -> ttnn.Tensor:
        """o_lat [1,H,T,L] -> concat-heads v-space [1,1,T,H*Dv] (per-head @ Wkb2^T)."""
        T = o_lat.shape[-2]
        o = ttnn.matmul(o_lat, self.w_absorb_o, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,H,T,Dv]
        return ttnn.reshape(ttnn.transpose(o, 1, 2), [1, 1, T, self.H * self.Dv])

    def _o_proj(self, o_heads: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(o_heads, self.w_o, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)

    def _cu_window(self, n_keys: int) -> ttnn.Tensor:
        """``cu_window_seqlens`` = [0, n_keys] for the windowed (prefix-aware) causal SDPA: one window
        over the whole prefix. Chunk geometry, not data: built once per key length and cached."""
        if n_keys not in self._cu_windows:
            self._cu_windows[n_keys] = ttnn.from_torch(
                torch.tensor([0, n_keys], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
        return self._cu_windows[n_keys]

    def _dense_attention(
        self, q: ttnn.Tensor, c: ttnn.Tensor, prefix: ttnn.Tensor | None = None, start: int = 0
    ) -> ttnn.Tensor:
        """Causal attention of q [1,1,T,H*Dq] (rows at absolute positions ``start + i``) over the latent
        of the full prefix: ``prefix`` [1,1,start,L] (the user's cached rows before this chunk, None
        when start == 0) followed by this chunk's c [1,1,T,L]; K/V are materialized from the latent.
        With a prefix the SDPA runs in windowed mode (one window over all keys, Q offset ``start``),
        which builds the causal mask on device at the global positions and is bit-identical to the
        plain causal kernel's rows [start, start + T). Returns concat-heads v-space [1,1,T,H*Dv]."""
        T = q.shape[-2]
        H, Dq, Dv = self.H, self.Dq, self.Dv
        if prefix is not None:
            c = ttnn.concat([prefix, c], dim=-2)  # [1,1,start+T,L]
        S = c.shape[-2]
        k = ttnn.linear(c, self.w_k_all, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,S,H*Dq]
        v = ttnn.linear(c, self.w_v_all, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,S,H*Dv]
        if prefix is not None:
            ttnn.deallocate(c)
        qh = ttnn.transpose(ttnn.reshape(q, [1, T, H, Dq]), 1, 2)
        kh = ttnn.transpose(ttnn.reshape(k, [1, S, H, Dq]), 1, 2)
        vh = ttnn.transpose(ttnn.reshape(v, [1, S, H, Dv]), 1, 2)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        windowed = {} if start == 0 else {"cu_window_seqlens": self._cu_window(S), "windowed_q_token_offset": start}
        o = ttnn.transformer.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            is_causal=True,
            scale=self.scale,
            program_config=self.sdpa_prefill_cfg,
            compute_kernel_config=self.sdpa_compute,
            **windowed,
        )  # [1,H,T,Dv]
        ttnn.deallocate(qh)
        ttnn.deallocate(kh)
        ttnn.deallocate(vh)
        return ttnn.reshape(ttnn.transpose(o, 1, 2), [1, 1, T, H * Dv])

    # --------------------------------------------------------------------- prefill
    def prefill_forward(
        self, x: ttnn.Tensor, cache: TtLatentCache, user: int, seq_len: int, start: int = 0
    ) -> ttnn.Tensor:
        """One user, one chunk: x row i is the token at absolute position ``start + i``
        (``start % block_size == 0``; ``start == 0`` = a new prompt). x [1,1,T_pad,4096] with
        T_pad % block_size == 0 and the logical ``seq_len`` <= T_pad; returns [1,1,T_pad,4096] (rows
        past seq_len are garbage). Fills the user's latent pages [start, start + T_pad) and maintains
        the pooled-key index for positions [start, start + seq_len) (the ring must hold the previous
        chunk's tail rows, i.e. chunks of a prompt are prefilled in order). Rows at absolute positions
        < 2048 run the dense causal attention over the full prefix (K/V from the cached latent rows +
        this chunk's); rows from 2048 on use the indexer + sparse_sdpa, exactly as the single-shot path."""
        T = x.shape[-2]
        if start % cache.block_size:
            raise ValueError(f"start {start} must be a multiple of the cache block size {cache.block_size}")
        end = start + seq_len
        split = min(T, max(0, self.sparse_row_start - start))  # chunk rows [0, split) are dense
        q_resid = self._q_resid(x)
        q = self._q(q_resid)  # [1,1,T,H*Dq]
        c = self._latent(x)  # [1,1,T,L]
        cache.fill(c, user, start=start)
        self.indexer.maintain_prefill(x, cache.index, user, seq_len, start=start)
        prefix = cache.user_rows(user, start) if start and split else None  # [1,1,start,L]
        if split == T:
            ttnn.deallocate(q_resid)
            o = self._dense_attention(q, c, prefix, start)
            ttnn.deallocate(q)
            ttnn.deallocate(c)
            if prefix is not None:
                ttnn.deallocate(prefix)
            return self._o_proj(o)
        # dense rows [0, split) (absolute positions [start, 2048))
        o_dense = None
        if split:
            o_dense = self._dense_attention(
                ttnn.slice(q, [0, 0, 0, 0], [1, 1, split, self.H * self.Dq]),
                ttnn.slice(c, [0, 0, 0, 0], [1, 1, split, self.L]),
                prefix,
                start,
            )
            if prefix is not None:
                ttnn.deallocate(prefix)
        ttnn.deallocate(c)
        # sparse rows [split, T) (absolute positions from max(start, 2048))
        idx = self.indexer
        x_s = ttnn.slice(x, [0, 0, split, 0], [1, 1, T, x.shape[-1]]) if split else x
        qr_s = ttnn.slice(q_resid, [0, 0, split, 0], [1, 1, T, q_resid.shape[-1]]) if split else q_resid
        q_i, w_i = idx.queries_and_weights(x_s, qr_s)
        ttnn.deallocate(q_resid)
        indices = idx.select(q_i, w_i, cache.index, user, query_start=start + split, num_pools_valid=end // idx.P)
        ttnn.deallocate(q_i)
        ttnn.deallocate(w_i)
        q_lat = self._absorb_q(
            ttnn.slice(q, [0, 0, split, 0], [1, 1, T, self.H * self.Dq]) if split else q
        )  # [1,H,T-split,L]
        ttnn.deallocate(q)
        q_rm = ttnn.to_layout(q_lat, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(q_lat)
        kv_rm = cache.user_rows_rm(user, start + T)
        o_rm = ttnn.transformer.sparse_sdpa(
            q_rm,
            kv_rm,
            indices,
            v_dim=self.L,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=self.scale,
            k_chunk_size=128,
        )  # [1,H,T-split,L] row-major
        ttnn.deallocate(q_rm)
        ttnn.deallocate(indices)
        ttnn.deallocate(kv_rm)
        o_lat = ttnn.to_layout(o_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(o_rm)
        o_sparse = self._o_lat_to_heads(o_lat)  # [1,1,T-split,H*Dv]
        ttnn.deallocate(o_lat)
        if o_dense is None:  # the whole chunk lies in the sparse region
            return self._o_proj(o_sparse)
        o = ttnn.concat([o_dense, o_sparse], dim=-2)  # [1,1,T,H*Dv]
        ttnn.deallocate(o_dense)
        ttnn.deallocate(o_sparse)
        return self._o_proj(o)

    # ---------------------------------------------------------------------- decode
    def decode_forward(
        self,
        x: ttnn.Tensor,
        cache: TtLatentCache,
        positions: ttnn.Tensor,
        pool_ids: ttnn.Tensor,
        ring_slots: ttnn.Tensor,
        host_positions: list[int] | None = None,
    ) -> ttnn.Tensor:
        """x [1,1,B,4096] (one token per user at position a = positions[b]); ``pool_ids`` = a // 4 and
        ``ring_slots`` = a % 4 ([B] int32 device tensors, host-derived like ``positions``). Appends the
        latent, maintains the pooled-key index, then attends over [0, a] with the dense flash-MLA decode
        for every slot (HF-exact through position 2050, trace-safe). With ``host_positions`` (the same
        values), users at a >= 2051 get their row replaced by the per-user indexer + sparse_sdpa result
        (host chunk geometry, not traced). Without ``host_positions`` a user beyond 2050 is attended
        densely, which is NOT HF semantics: FunctionalDecoder enforces the routing."""
        B = x.shape[-2]
        q_resid = self._q_resid(x)
        q = self._q(q_resid)  # [1,1,B,H*Dq]
        c = self._latent(x)  # [1,1,B,L]
        cache.update(ttnn.reshape(c, [1, B, 1, self.L]), positions)
        ttnn.deallocate(c)
        self.indexer.maintain_decode(x, cache.index, pool_ids, ring_slots)
        q_lat = self._absorb_q(q)  # [1,H,B,L]
        ttnn.deallocate(q)
        q_dec = ttnn.transpose(q_lat, 1, 2)  # [1,B,H,L] (decode layout: batch, padded heads, dim)
        o_lat = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_dec,
            cache.cache,
            head_dim_v=self.L,
            page_table_tensor=cache.page_table,
            cur_pos_tensor=positions,
            scale=self.scale,
            program_config=self._decode_sdpa_config(B),
            compute_kernel_config=self.sdpa_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,B,H,L]
        ttnn.deallocate(q_dec)
        oh = ttnn.transpose(o_lat, 1, 2)  # [1,H,B,L]
        ttnn.deallocate(o_lat)
        sparse_users = (
            [u for u, a in enumerate(host_positions) if a > self.dense_exact_max_pos] if host_positions else []
        )
        if sparse_users:
            sparse_rows = self._decode_sparse_rows(x, q_resid, q_lat, cache, host_positions, sparse_users)
            rows = [
                sparse_rows[u] if u in sparse_rows else ttnn.slice(oh, [0, 0, u, 0], [1, self.H, u + 1, self.L])
                for u in range(B)
            ]
            oh_new = rows[0] if B == 1 else ttnn.concat(rows, dim=-2)
            ttnn.deallocate(oh)
            oh = oh_new
        ttnn.deallocate(q_resid)
        ttnn.deallocate(q_lat)
        return self._o_proj(self._o_lat_to_heads(oh))

    def _decode_sparse_rows(self, x, q_resid, q_lat, cache: TtLatentCache, positions: list[int], users: list[int]):
        """Per-user indexer selection + sparse_sdpa rows {u: [1,H,1,L]} for the listed users (positions
        > dense_exact_max_pos, so every index row is a full valid prefix)."""
        idx = self.indexer
        D = idx.D
        q_i, w_i = idx.queries_and_weights(x, q_resid)  # [1,Hi,B,D], [1,1,B,Hi]
        out = {}
        for u in users:
            a = positions[u]
            # indexer_score_dsa wants a tile-aligned query block and chunk start: score the 32 positions
            # [a - a%32, a - a%32 + 32) with this user's query placed at row r = a % 32 (zero rows elsewhere,
            # scored but discarded) and keep row r of the selection
            start, r = a - a % TILE, a % TILE
            qu = self._place_row(ttnn.slice(q_i, [0, 0, u, 0], [1, idx.Hi, u + 1, D]), r)
            wu = self._place_row(ttnn.slice(w_i, [0, 0, u, 0], [1, 1, u + 1, idx.Hi]), r)
            indices32 = idx.select(qu, wu, cache.index, u, query_start=start, num_pools_valid=(a + 1) // idx.P)
            indices = ttnn.slice(indices32, [0, 0, r, 0], [1, 1, r + 1, indices32.shape[-1]])
            q_rm = ttnn.to_layout(ttnn.slice(q_lat, [0, 0, u, 0], [1, self.H, u + 1, self.L]), ttnn.ROW_MAJOR_LAYOUT)
            kv_rm = cache.user_rows_rm(u, a + 1)
            o_rm = ttnn.transformer.sparse_sdpa(
                q_rm,
                kv_rm,
                indices,
                v_dim=self.L,
                kv_format=ttnn.transformer.SparseKVFormat.BF16,
                scale=self.scale,
                k_chunk_size=128,
            )  # [1,H,1,L]
            ttnn.deallocate(kv_rm)
            ttnn.deallocate(q_rm)
            ttnn.deallocate(indices32)
            out[u] = ttnn.to_layout(o_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(q_i)
        ttnn.deallocate(w_i)
        return out

    def selected_ids(self, x: ttnn.Tensor, cache: TtLatentCache, user: int, position: int) -> list[int]:
        """Test hook: the raw index row (token ids, 0xFFFFFFFF sentinels) the indexer selects for one
        query ``x`` [1,1,1,4096] (the normalized attention input) of ``user`` at ``position`` on the
        CURRENT index cache (read only)."""
        idx = self.indexer
        q_resid = self._q_resid(x)
        q_i, w_i = idx.queries_and_weights(x, q_resid)
        start, r = position - position % TILE, position % TILE
        qu = self._place_row(q_i, r)
        wu = self._place_row(w_i, r)
        ids32 = idx.select(qu, wu, cache.index, user, query_start=start, num_pools_valid=(position + 1) // idx.P)
        ids = ttnn.to_torch(ttnn.slice(ids32, [0, 0, r, 0], [1, 1, r + 1, ids32.shape[-1]])).flatten().tolist()
        for t in (q_resid, q_i, w_i, ids32):
            ttnn.deallocate(t)
        return [int(i) for i in ids]

    def _place_row(self, row: ttnn.Tensor, r: int) -> ttnn.Tensor:
        """[.., 1, W] -> [.., 32, W] with ``row`` at row r and zeros elsewhere (tile pad cannot front-pad)."""
        shape = list(row.shape)
        parts = []
        if r:
            parts.append(
                ttnn.zeros(shape[:-2] + [r, shape[-1]], dtype=row.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
            )
        parts.append(row)
        if TILE - 1 - r:
            parts.append(
                ttnn.zeros(
                    shape[:-2] + [TILE - 1 - r, shape[-1]], dtype=row.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
                )
            )
        return parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=-2)

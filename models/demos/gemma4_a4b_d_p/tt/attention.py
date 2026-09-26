# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 attention (chunked prefill) on a 1x4 mesh, TP by head: TtSlidingAttention below, TtGlobalAttention (full
attention, 2 KV heads x 512, each KV head on 2 chips, paged-shaped cache + chunked SDPA) at the end of the file.

Sliding-window attention:

Chip c holds Q heads 4c..4c+3 and KV heads 2c, 2c+1 (GQA 2:1 inside the chip, no CCL until o_proj).

    qkv   = x @ Wqkv_c                     fused per-chip [H, (4 + 2 + 2) * 256]
    q,k,v = nlp_create_qkv_heads
    q     = rope(rms(q) * q_norm)          per-head RMS over D, HiFi4 + fp32 acc
    k     = rope(rms(k) * k_norm)
    v     = rms(v)                         unscaled
    cache[start:start+S] = k, v            fill_cache(update_idx=start), full-length cache [1, 2, max_seq, D] per chip
    tail  = cache[start-hist:start]        hist = min(window, start): the previous window, read back from the cache
    sdpa  = SDPA(causal, sliding_window=1024, scale 1.0) over the square [tail | chunk]; Q is front-padded by hist
            filler rows whose outputs are dropped (causal, so they never touch the kept rows)
    out   = all_reduce(concat_heads(sdpa) @ Wo_c)   row-parallel o_proj

Adapted from models/demos/gemma4/tt/attention/prefill.py (sliding branch with the sliding_tail concat) and
models/demos/ernie45_d_p/tt/attention.py (fused per-chip QKV, row-parallel o_proj, all_reduce). Unlike gemma4's
stash, the tail comes from the full-length cache, so a chunk is a pure function of (x, cache).
"""

from __future__ import annotations

import torch

import ttnn

NUM_CHIPS = 4
TILE = 32


def _hifi4():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


def _hifi2():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def rope_tables(inv_freq: torch.Tensor, start: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half cos/sin [length, D] for positions [start, start+length), computed in fp32."""
    pos = torch.arange(start, start + length, dtype=torch.float64)
    freqs = pos[:, None] * inv_freq.double()[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().float(), emb.sin().float()


class TtKVCacheSliding:
    """Full-length K and V for one layer: [1, 8, max_seq, D] sharded by head -> [1, 2, max_seq, D] per chip."""

    def __init__(self, mesh, num_kv_heads: int, head_dim: int, max_seq: int, dtype=ttnn.bfloat16):
        assert max_seq % TILE == 0
        self.mesh, self.nkv, self.d, self.max_seq, self.dtype = mesh, num_kv_heads, head_dim, max_seq, dtype
        self.k = self._dev(torch.zeros(1, num_kv_heads, max_seq, head_dim))
        self.v = self._dev(torch.zeros(1, num_kv_heads, max_seq, head_dim))

    def _dev(self, t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=1),
        )

    def load_prefix(self, key: torch.Tensor, value: torch.Tensor, length: int) -> None:
        """Host key/value [8, >= length, D]: positions [0, length) are copied, the rest is zero."""

        def full(t):
            f = torch.zeros(1, self.nkv, self.max_seq, self.d)
            f[0, :, :length] = t[:, :length].float()
            return f

        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)
        self.k, self.v = self._dev(full(key)), self._dev(full(value))

    def to_torch(self, length: int) -> dict:
        def host(t):
            parts = [ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(t)]  # each [1, 2, max_seq, D]
            return torch.cat(parts, dim=1)[0, :, :length]

        return {"key": host(self.k), "value": host(self.v)}

    def free(self):
        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)


class TtSlidingAttention:
    def __init__(self, mesh, cfg, w, inv_freq: torch.Tensor, window: int, eps: float = 1e-6):
        """cfg: Gemma4TextConfig; w: reference LayerWeights (wq, wk, wv, wo, q_norm, k_norm)."""
        self.mesh, self.window, self.eps = mesh, window, eps
        self.inv_freq = inv_freq
        n, D = NUM_CHIPS, cfg.head_dim
        self.d = D
        self.nq, self.nkv = cfg.num_attention_heads // n, cfg.num_key_value_heads // n
        wq, wk, wv = w.wq.float().T, w.wk.float().T, w.wv.float().T  # [H, out]
        fused = [
            torch.cat(
                [
                    wq[:, c * self.nq * D : (c + 1) * self.nq * D],
                    wk[:, c * self.nkv * D : (c + 1) * self.nkv * D],
                    wv[:, c * self.nkv * D : (c + 1) * self.nkv * D],
                ],
                dim=-1,
            )
            for c in range(n)
        ]
        self.wqkv = self._shard(torch.stack(fused)[:, None], dim=0)  # [4, 1, H, 2048] -> [1, 1, H, 2048] per chip
        self.wo = self._shard(w.wo.float().T.contiguous()[None, None], dim=-2)  # [4096, H] -> [1024, H] per chip
        self.q_norm = self._replicate(w.q_norm.float().reshape(1, 1, 1, -1))
        self.k_norm = self._replicate(w.k_norm.float().reshape(1, 1, 1, -1))
        self._rope = {}

    def _shard(self, t, dim):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=dim),
        )

    def _replicate(self, t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _rope_tables(self, start: int, seq: int):
        key = (start, seq)
        if key not in self._rope:
            for v in self._rope.values():
                ttnn.deallocate(v[0])
                ttnn.deallocate(v[1])
            cos, sin = rope_tables(self.inv_freq, start, seq)
            self._rope = {key: (self._replicate(cos[None, None]), self._replicate(sin[None, None]))}
        return self._rope[key]

    def _head_norm(self, t, weight):
        shape = t.shape
        flat = ttnn.reshape(t, (1, 1, shape[1] * shape[2], shape[3]))
        y = ttnn.rms_norm(
            flat, weight=weight, epsilon=self.eps, compute_kernel_config=_hifi4(), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return ttnn.reshape(y, shape)

    def _sdpa_program_config(self, seq: int):
        q = 256 if seq >= 256 else TILE * max(1, seq // TILE)
        k = min(128, self.window // 2)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            q_chunk_size=q,
            k_chunk_size=k,
            exp_approx_mode=False,
        )

    def __call__(self, x: ttnn.Tensor, start: int, cache: TtKVCacheSliding) -> ttnn.Tensor:
        """x: replicated [1, 1, S, H] TILE (attn_norm output), queries at [start, start+S). Returns replicated
        [1, 1, S, H] (all-reduced). Writes this chunk's K/V into cache positions [start, start+S)."""
        seq = x.shape[-2]
        D, nq = self.d, self.nq
        assert start % TILE == 0 and seq % TILE == 0

        qkv = ttnn.linear(x, self.wqkv, compute_kernel_config=_hifi2(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=nq, num_kv_heads=self.nkv, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)

        cos, sin = self._rope_tables(start, seq)
        qn = self._head_norm(q, self.q_norm)
        ttnn.deallocate(q)
        q = ttnn.experimental.rotary_embedding(qn, cos, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qn)
        kn = self._head_norm(k, self.k_norm)
        ttnn.deallocate(k)
        k = ttnn.experimental.rotary_embedding(kn, cos, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(kn)
        vn = self._head_norm(v, None)
        ttnn.deallocate(v)
        v = vn

        ttnn.fill_cache(cache.k, k, batch_idx=0, update_idx=start)
        ttnn.fill_cache(cache.v, v, batch_idx=0, update_idx=start)

        hist = min(((self.window + TILE - 1) // TILE) * TILE, start)
        if hist:
            k_tail = ttnn.slice(cache.k, [0, 0, start - hist, 0], [1, self.nkv, start, D])
            v_tail = ttnn.slice(cache.v, [0, 0, start - hist, 0], [1, self.nkv, start, D])
            if seq >= hist:
                q_pad = ttnn.slice(q, [0, 0, 0, 0], [1, nq, hist, D])
            else:
                q_pad = ttnn.zeros([1, nq, hist, D], dtype=q.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh)
            q_cat = ttnn.concat([q_pad, q], dim=2)
            k_cat = ttnn.concat([k_tail, k], dim=2)
            v_cat = ttnn.concat([v_tail, v], dim=2)
            for t in (q_pad, k_tail, v_tail, q, k, v):
                ttnn.deallocate(t)
        else:
            q_cat, k_cat, v_cat = q, k, v

        full = ttnn.transformer.scaled_dot_product_attention(
            q_cat,
            k_cat,
            v_cat,
            is_causal=True,
            scale=1.0,
            sliding_window_size=self.window,
            program_config=self._sdpa_program_config(hist + seq),
            compute_kernel_config=_hifi4(),
        )
        for t in (q_cat, k_cat, v_cat):
            ttnn.deallocate(t)
        if hist:
            attn = ttnn.slice(full, [0, 0, hist, 0], [1, nq, hist + seq, D])
            ttnn.deallocate(full)
        else:
            attn = full

        a = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        o = ttnn.linear(a, self.wo, compute_kernel_config=_hifi2(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(a)
        out = ttnn.all_reduce(o, cluster_axis=1)
        ttnn.deallocate(o)
        return out


# --------------------------------------------------------------------------------------
# Global (full-attention) layers
# --------------------------------------------------------------------------------------

KV_BLOCK = 64  # page size of the paged-shaped global cache (identity page table)


class TtKVCacheGlobal:
    """Full-length K and V for one global layer. 2 KV heads on 4 chips: chip c holds KV head c // 2.

    Per chip the cache is a paged-shaped [max_seq / B, 1, B, D] tensor. With one head per chip that is bit-identical to a
    contiguous [1, 1, max_seq, D] cache, so the identity page table turns paged_fill_cache into "write at offset" and lets
    chunked_scaled_dot_product_attention read the prefix directly (as in ernie45_d_p/tt/attention.py).
    """

    def __init__(self, mesh, num_kv_heads: int, head_dim: int, max_seq: int, dtype=ttnn.bfloat16):
        assert NUM_CHIPS % num_kv_heads == 0
        self.max_seq = -(-max_seq // KV_BLOCK) * KV_BLOCK
        self.mesh, self.nkv, self.d, self.dtype = mesh, num_kv_heads, head_dim, dtype
        self.rep = NUM_CHIPS // num_kv_heads
        self.nb = self.max_seq // KV_BLOCK
        z = torch.zeros(num_kv_heads, self.max_seq, head_dim)
        self.k, self.v = self._dev(z), self._dev(z)
        self.page_table_host = torch.arange(self.nb, dtype=torch.int32)[None]
        self.page_table = self._pt(self.page_table_host)
        self._chunk_pt = {}

    def _pt(self, t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _dev(self, t: torch.Tensor) -> ttnn.Tensor:
        """Host [nkv, max_seq, D] -> device paged [nb, 1, B, D] per chip, chip c = head c // rep."""
        per_chip = t.repeat_interleave(self.rep, dim=0)  # [4, max_seq, D]
        paged = per_chip.reshape(NUM_CHIPS, self.nb, KV_BLOCK, self.d).transpose(0, 1).contiguous()  # [nb, 4, B, D]
        return ttnn.from_torch(
            paged.to(torch.bfloat16),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=1),
        )

    def chunk_page_table(self, start: int, seq: int):
        key = (start, seq)
        if key not in self._chunk_pt:
            assert start % KV_BLOCK == 0 and seq % KV_BLOCK == 0
            for v in self._chunk_pt.values():
                ttnn.deallocate(v)
            self._chunk_pt = {
                key: self._pt(self.page_table_host[:, start // KV_BLOCK : (start + seq) // KV_BLOCK].contiguous())
            }
        return self._chunk_pt[key]

    def load_prefix(self, key: torch.Tensor, value: torch.Tensor, length: int) -> None:
        """Host key/value [nkv, >= length, D]: positions [0, length) are copied, the rest is zero."""

        def full(t):
            f = torch.zeros(self.nkv, self.max_seq, self.d)
            f[:, :length] = t[:, :length].float()
            return f

        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)
        self.k, self.v = self._dev(full(key)), self._dev(full(value))

    def to_torch(self, length: int) -> dict:
        def host(t):
            parts = ttnn.get_device_tensors(t)
            heads = [ttnn.to_torch(parts[h * self.rep]).float() for h in range(self.nkv)]  # each [nb, 1, B, D]
            return torch.stack([p[:, 0].reshape(-1, self.d) for p in heads])[:, :length]

        return {"key": host(self.k), "value": host(self.v)}

    def free(self):
        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)
        ttnn.deallocate(self.page_table)
        for v in self._chunk_pt.values():
            ttnn.deallocate(v)
        self._chunk_pt = {}


class TtGlobalAttention(TtSlidingAttention):
    """Gemma-4 global attention (chunked prefill) on a 1x4 mesh, TP by head, GQA 8:1 with each KV head on 2 chips.

    Chip c holds Q heads 4c..4c+3 and KV head c // 2. There is no v_proj (attention_k_eq_v):
        qkv  = x @ [Wq_c | Wk_h | Wk_h]         per-chip fused [H, (4 + 1 + 1) * 512], the k head twice (raw K and raw V)
        q    = rope(rms(q) * q_norm)            proportional RoPE theta 1e6 (inv_freq zero past 25%: cos 1, sin 0 there)
        k    = rope(rms(k_raw) * k_norm)
        v    = rms(k_raw)                       unscaled, not roped
        cache[start:start+S] = k, v             paged_fill_cache, identity page table
        sdpa = SDPA causal (chunk 0) / chunked SDPA over cache[0:start+S] (later chunks), scale 1.0, no window
        out  = all_reduce(concat_heads(sdpa) @ Wo_c)   row-parallel o_proj
    """

    def __init__(self, mesh, cfg, w, inv_freq: torch.Tensor, eps: float = 1e-6):
        """cfg: Gemma4TextConfig; w: namespace with wq [16*512, H], wk [2*512, H], wo [H, 16*512], q_norm, k_norm."""
        self.mesh, self.window, self.eps = mesh, None, eps
        self.inv_freq = inv_freq
        n = NUM_CHIPS
        D = cfg.global_head_dim or cfg.head_dim
        self.d = D
        nkv_total = cfg.num_global_key_value_heads
        self.nq, self.nkv = cfg.num_attention_heads // n, 1
        self.rep = n // nkv_total
        wq, wk = w.wq.float().T, w.wk.float().T  # [H, out]
        fused = []
        for c in range(n):
            h = c // self.rep
            kh = wk[:, h * D : (h + 1) * D]
            fused.append(torch.cat([wq[:, c * self.nq * D : (c + 1) * self.nq * D], kh, kh], dim=-1))
        self.wqkv = self._shard(torch.stack(fused)[:, None], dim=0)  # [1, 1, H, 6 * 512] per chip
        self.wo = self._shard(w.wo.float().T.contiguous()[None, None], dim=-2)  # [8192, H] -> [2048, H] per chip
        self.q_norm = self._replicate(w.q_norm.float().reshape(1, 1, 1, -1))
        self.k_norm = self._replicate(w.k_norm.float().reshape(1, 1, 1, -1))
        self._rope = {}

    def _sdpa_program_config(self, seq: int, start: int = 0):
        # Head dim 512: 128 x 128 chunks keep the Q/K/V/QK circular buffers well inside L1.
        q = k = 128
        while q > TILE and seq % q:
            q //= 2
        k = q
        if start:  # chunked SDPA: chunk_start must be a multiple of both chunk sizes
            lowbit = start & -start
            q, k = min(q, lowbit), min(k, lowbit)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            q_chunk_size=q,
            k_chunk_size=k,
            exp_approx_mode=False,
        )

    def __call__(self, x: ttnn.Tensor, start: int, cache: TtKVCacheGlobal) -> ttnn.Tensor:
        """x: replicated [1, 1, S, H] TILE (attn_norm output), queries at [start, start+S). Returns replicated
        [1, 1, S, H] (all-reduced). Writes this chunk's K/V into cache positions [start, start+S)."""
        seq = x.shape[-2]
        assert start % KV_BLOCK == 0 and seq % KV_BLOCK == 0

        qkv = ttnn.linear(x, self.wqkv, compute_kernel_config=_hifi2(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.nq, num_kv_heads=1, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)

        cos, sin = self._rope_tables(start, seq)
        qn = self._head_norm(q, self.q_norm)
        ttnn.deallocate(q)
        q = ttnn.experimental.rotary_embedding(qn, cos, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qn)
        kn = self._head_norm(k, self.k_norm)
        ttnn.deallocate(k)
        k = ttnn.experimental.rotary_embedding(kn, cos, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(kn)
        vn = self._head_norm(v, None)
        ttnn.deallocate(v)
        v = vn

        pt = cache.chunk_page_table(start, seq)
        ttnn.experimental.paged_fill_cache(cache.k, k, pt, batch_idx=0)
        ttnn.experimental.paged_fill_cache(cache.v, v, pt, batch_idx=0)

        prog = self._sdpa_program_config(seq, start)
        if start == 0:
            attn = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=1.0, program_config=prog, compute_kernel_config=_hifi4()
            )
        else:
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q,
                input_tensor_k=cache.k,
                input_tensor_v=cache.v,
                page_table_tensor=cache.page_table,
                chunk_start_idx=start,
                scale=1.0,
                program_config=prog,
                compute_kernel_config=_hifi4(),
            )
        for t in (q, k, v):
            ttnn.deallocate(t)

        a = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        o = ttnn.linear(a, self.wo, compute_kernel_config=_hifi2(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(a)
        out = ttnn.all_reduce(o, cluster_axis=1)
        ttnn.deallocate(o)
        return out

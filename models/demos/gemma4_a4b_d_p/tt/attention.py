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

import os

import torch

import ttnn

NUM_CHIPS = 4
TILE = 32

# SDPA presets (env GEMMA4_SDPA_CFG, default "A").
#   base: the bring-up config (HiFi4 + fp32 dest acc, which runs SDPA's non-streaming kernel on Blackhole, exact exp,
#         sliding q256/k128, global q128/k128).
#   A:    HiFi2, fp32 dest acc off (streaming kernel), packer_l1_acc off, approx exp, full grid; the compute settings
#         of ernie45_d_p/tt/attention.py SDPA_PRESETS["A"]. ERNIE's q256/k512 does not fit L1 here (streaming CBs):
#         sliding D 256 fits at most q256/k256 (q256/k512 needs 2.1 MB); global D 512 fits q128/k128 or q160/k128
#         (q256 only with k32, which is slower). Global SDPA is causal, so it gets no KV chain forwarding and each Q
#         chunk streams the whole K/V prefix from DRAM; the larger q160 chunk cuts that traffic (44 -> 35 ms at 51k).
SDPA_PRESETS = {
    "base": dict(fidelity="HiFi4", fp32=True, packer_l1=False, exp_approx=False, sliding=(256, 128), glob=(128, 128)),
    "A": dict(fidelity="HiFi2", fp32=False, packer_l1=False, exp_approx=True, sliding=(256, 256), glob=(160, 128)),
}


def sdpa_settings() -> dict:
    """The active SDPA preset; GEMMA4_SDPA_CFG selects it, GEMMA4_SDPA_{SQ,SK,GQ,GK} override the chunk sizes."""
    name = os.environ.get("GEMMA4_SDPA_CFG", "A")
    c = dict(SDPA_PRESETS[name], name=name)
    env = os.environ.get
    if env("GEMMA4_SDPA_SQ") or env("GEMMA4_SDPA_SK"):
        c["sliding"] = (int(env("GEMMA4_SDPA_SQ", c["sliding"][0])), int(env("GEMMA4_SDPA_SK", c["sliding"][1])))
    if env("GEMMA4_SDPA_GQ") or env("GEMMA4_SDPA_GK"):
        c["glob"] = (int(env("GEMMA4_SDPA_GQ", c["glob"][0])), int(env("GEMMA4_SDPA_GK", c["glob"][1])))
    return c


def sdpa_compute_config():
    c = sdpa_settings()
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, c["fidelity"]),
        math_approx_mode=False,
        fp32_dest_acc_en=c["fp32"],
        packer_l1_acc=c["packer_l1"],
    )


def _fit_chunk(size: int, seq: int, start: int = 0) -> int:
    """size if it divides seq and start, else the largest power of two <= min(size, 128) (>= TILE) that does."""
    if seq % size == 0 and start % size == 0:
        return size
    size = 1 << (min(size, 128).bit_length() - 1)
    while size > TILE and (seq % size or start % size):
        size //= 2
    return size


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

    def set_rope_tables(self, cos_full: ttnn.Tensor, sin_full: ttnn.Tensor) -> None:
        """Use device cos/sin tables [1, 1, max_seq, D] built once at load (shared by all layers of this RoPE type);
        each chunk then slices its rows on the device instead of uploading new tables."""
        self.rope_full = (cos_full, sin_full)

    def _chunk_rope(self, start: int, seq: int):
        """(cos, sin, owned): owned tensors are per-chunk slices the caller frees after use."""
        full = getattr(self, "rope_full", None)
        if full is None:
            return (*self._rope_tables(start, seq), False)
        cos = ttnn.slice(full[0], [0, 0, start, 0], [1, 1, start + seq, full[0].shape[-1]])
        sin = ttnn.slice(full[1], [0, 0, start, 0], [1, 1, start + seq, full[1].shape[-1]])
        return cos, sin, True

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
        c = sdpa_settings()
        qc, kc = c["sliding"]
        q = qc if seq >= qc else TILE * max(1, seq // TILE)
        k = min(kc, self.window // 2)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            q_chunk_size=q,
            k_chunk_size=k,
            exp_approx_mode=c["exp_approx"],
        )

    def __call__(self, x: ttnn.Tensor, start: int, cache: TtKVCacheSliding, kv_sink=None) -> ttnn.Tensor:
        """x: replicated [1, 1, S, H] TILE (attn_norm output), queries at [start, start+S). Returns replicated
        [1, 1, S, H] (all-reduced). Writes this chunk's K/V into cache positions [start, start+S).
        kv_sink(k, v), if given, also receives this chunk's per-chip K/V [1, nkv_local, S, D] (the serving cache)."""
        seq = x.shape[-2]
        D, nq = self.d, self.nq
        assert start % TILE == 0 and seq % TILE == 0

        qkv = ttnn.linear(x, self.wqkv, compute_kernel_config=_hifi2(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=nq, num_kv_heads=self.nkv, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)

        cos, sin, rope_owned = self._chunk_rope(start, seq)
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
        if rope_owned:
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
        if kv_sink is not None:
            kv_sink(k, v)

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
            compute_kernel_config=sdpa_compute_config(),
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
        if getattr(self, "device_page_slices", False):
            # Serving path: slice the resident identity table on the device (no per-chunk host upload).
            assert start % KV_BLOCK == 0 and seq % KV_BLOCK == 0
            if key not in self._chunk_pt:
                for v in self._chunk_pt.values():
                    ttnn.deallocate(v)
                self._chunk_pt = {
                    key: ttnn.slice(self.page_table, [0, start // KV_BLOCK], [1, (start + seq) // KV_BLOCK])
                }
            return self._chunk_pt[key]
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
        c = sdpa_settings()
        # chunked SDPA: chunk_start must be a multiple of both chunk sizes
        q, k = _fit_chunk(c["glob"][0], seq, start), _fit_chunk(c["glob"][1], seq, start)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            q_chunk_size=q,
            k_chunk_size=k,
            exp_approx_mode=c["exp_approx"],
        )

    def __call__(self, x: ttnn.Tensor, start: int, cache: TtKVCacheGlobal, kv_sink=None) -> ttnn.Tensor:
        """x: replicated [1, 1, S, H] TILE (attn_norm output), queries at [start, start+S). Returns replicated
        [1, 1, S, H] (all-reduced). Writes this chunk's K/V into cache positions [start, start+S).
        kv_sink(k, v), if given, also receives this chunk's per-chip K/V [1, 1, S, D] (the serving cache)."""
        seq = x.shape[-2]
        assert start % KV_BLOCK == 0 and seq % KV_BLOCK == 0

        qkv = ttnn.linear(x, self.wqkv, compute_kernel_config=_hifi2(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.nq, num_kv_heads=1, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)

        cos, sin, rope_owned = self._chunk_rope(start, seq)
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
        if rope_owned:
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
        if kv_sink is not None:
            kv_sink(k, v)

        pt = cache.chunk_page_table(start, seq)
        ttnn.experimental.paged_fill_cache(cache.k, k, pt, batch_idx=0)
        ttnn.experimental.paged_fill_cache(cache.v, v, pt, batch_idx=0)

        prog = self._sdpa_program_config(seq, start)
        if start == 0:
            attn = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=1.0, program_config=prog, compute_kernel_config=sdpa_compute_config()
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
                compute_kernel_config=sdpa_compute_config(),
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

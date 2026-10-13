# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 sliding-window attention for prefill on the 8x4 Galaxy mesh.

Layout: the chunk is replicated on every mesh row; along the 4 columns the heads are split (TP=4). Column c holds
Q heads 4c..4c+3 and KV heads 2c, 2c+1 (GQA 2:1), so attention is local to each chip. o_proj is row-parallel and is
followed by an all_reduce over the columns (cluster_axis=1). Every row computes the same result.

Per chunk at absolute positions [start, start+S):
  q = rope(q_norm(x Wq)), k = rope(k_norm(x Wk)), v = rms(x Wv)   (Gemma RMSNorm x * w, eps 1e-6; V has no weight)
  K/V are written into the bfp8 cache at [start, start+S), then SDPA (causal, window 1024, scale 1.0) runs over the
  square [cache tail (start-hist, start) | chunk]. Q is front-padded with hist dummy rows whose outputs are dropped.
"""

import torch

import ttnn

from .settings import settings

_FIDELITY = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi2": ttnn.MathFidelity.HiFi2, "LoFi": ttnn.MathFidelity.LoFi}

TP_AXIS = 1
TILE = 32


def _kernel_config(mesh, fidelity: str, fp32_acc: bool):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=_FIDELITY[fidelity],
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=False,
    )


def _col_mapper(mesh, dim):
    """Shard tensor dim `dim` over the mesh columns (TP), replicate over the rows."""
    return ttnn.ShardTensor2dMesh(mesh, tuple(mesh.shape), dims=(None, dim))


class TtKVCache:
    """Per-layer K/V cache [1, Hkv, max_seq, D] bfp8, heads split over the mesh columns, replicated over the rows.
    K is stored post k_norm + RoPE, V post the unscaled RMS (the reference state's layout)."""

    def __init__(self, mesh, num_kv_heads: int, head_dim: int, max_seq: int, k_prefix=None, v_prefix=None):
        self.mesh, self.max_seq = mesh, max_seq
        tp = mesh.shape[TP_AXIS]
        assert num_kv_heads % tp == 0, f"{num_kv_heads} kv heads over TP={tp}"
        self.k = self._upload(num_kv_heads, head_dim, k_prefix)
        self.v = self._upload(num_kv_heads, head_dim, v_prefix)

    def _upload(self, hkv, d, prefix):
        full = torch.zeros(1, hkv, self.max_seq, d, dtype=torch.float32)
        if prefix is not None and prefix.shape[1] > 0:
            full[0, :, : prefix.shape[1]] = prefix.float()
        return ttnn.from_torch(
            full,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_col_mapper(self.mesh, 1),
        )

    def deallocate(self):
        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)


class TtSlidingAttention:
    kv_repeat = 1  # chips per KV head along the TP columns

    def __init__(
        self,
        mesh,
        wq,
        wk,
        wv,
        wo,
        q_norm,
        k_norm,
        *,
        num_heads,
        num_kv_heads,
        head_dim,
        window,
        rope_inv_freq,
        eps=1e-6,
        scale=1.0,
    ):
        self.mesh = mesh
        tp = mesh.shape[TP_AXIS]
        assert num_heads % tp == 0 and num_kv_heads % tp == 0
        self.nh, self.nkv, self.d = num_heads // tp, num_kv_heads // tp, head_dim
        self.window, self.eps, self.scale = window, eps, scale
        self.inv_freq = rope_inv_freq.float()

        # Fused per-chip QKV weight [in, (4 Q | 2 K | 2 V) heads * D], chip blocks stacked along the output dim.
        qd, kd = self.nh * head_dim, self.nkv * head_dim
        blocks = []
        for c in range(tp):
            blocks += [wq[c * qd : (c + 1) * qd], wk[c * kd : (c + 1) * kd], wv[c * kd : (c + 1) * kd]]
        wqkv = torch.cat(blocks, dim=0).t().contiguous()  # [hidden, tp * (qd + 2 kd)]
        self.wqkv = self._weight(wqkv, dim=-1)
        self.wo = self._weight(wo.t().contiguous(), dim=-2)  # [nh*D, hidden], rows split by Q head
        self.q_norm = self._replicated(q_norm.float().reshape(1, 1, 1, -1))
        self.k_norm = self._replicated(k_norm.float().reshape(1, 1, 1, -1))

        self.mm_cfg = _kernel_config(mesh, settings.get("ATTN_MATMUL_FIDELITY"), settings.get("ATTN_MATMUL_FP32_ACC"))
        self.norm_cfg = _kernel_config(mesh, "HiFi4", True)
        self.sdpa_cfg = _kernel_config(mesh, settings.get("SDPA_FIDELITY"), settings.get("SDPA_FP32_ACC"))
        self.q_chunk, self.k_chunk = settings.get("SDPA_Q_CHUNK"), settings.get("SDPA_K_CHUNK")
        self.cos = self.sin = None
        self.rope_len = 0

    def _weight(self, w, dim):
        return ttnn.from_torch(
            w.float().reshape(1, 1, *w.shape),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_col_mapper(self.mesh, dim),
        )

    def _replicated(self, t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def ensure_rope(self, max_seq: int):
        """Build the rotate-half cos/sin tables [1, 1, max_seq, D] once (load time); forward slices them on device."""
        if max_seq <= self.rope_len:
            return
        pos = torch.arange(max_seq, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        if self.cos is not None:
            ttnn.deallocate(self.cos)
            ttnn.deallocate(self.sin)
        self.cos = self._replicated(emb.cos().reshape(1, 1, max_seq, -1), ttnn.float32)
        self.sin = self._replicated(emb.sin().reshape(1, 1, max_seq, -1), ttnn.float32)
        self.rope_len = max_seq

    def make_cache(self, max_seq: int, k_prefix=None, v_prefix=None) -> TtKVCache:
        """Device cache for this layer from host prefixes [Hkv, n, D] (each KV head on kv_repeat adjacent columns)."""
        r = self.kv_repeat
        k_prefix, v_prefix = (
            p.repeat_interleave(r, dim=0) if p is not None and r > 1 else p for p in (k_prefix, v_prefix)
        )
        return TtKVCache(self.mesh, self.nkv * self.mesh.shape[TP_AXIS], self.d, max_seq, k_prefix, v_prefix)

    def _head_norm(self, t, weight):
        shape = t.shape
        flat = ttnn.reshape(t, (1, 1, shape[1] * shape[2], shape[3]))
        out = ttnn.rms_norm(flat, weight=weight, epsilon=self.eps, compute_kernel_config=self.norm_cfg)
        return ttnn.reshape(out, shape)

    @staticmethod
    def _fp32(t):
        out = ttnn.typecast(t, ttnn.float32)
        ttnn.deallocate(t)
        return out

    def _norm_rope(self, t, weight, cos, sin, dtype):
        """fp32 [1, h, S, D] -> rope(rms_norm(t) * w) in `dtype`."""
        tn = self._head_norm(t, weight)
        ttnn.deallocate(t)
        tr = ttnn.experimental.rotary_embedding(tn, cos, sin, compute_kernel_config=self.norm_cfg)
        ttnn.deallocate(tn)
        out = ttnn.typecast(tr, dtype)
        ttnn.deallocate(tr)
        return out

    def __call__(self, x, cache: TtKVCache, start: int):
        """x: [1, 1, S, hidden] bf16 replicated; positions [start, start+S). Returns [1, 1, S, hidden] replicated."""
        s = x.shape[-2]
        end = start + s
        assert start % TILE == 0 and s % TILE == 0, f"start {start} and S {s} must be tile aligned"
        assert end <= cache.max_seq and end <= self.rope_len, f"chunk end {end} past cache/rope ({cache.max_seq})"
        nh, nkv, d = self.nh, self.nkv, self.d

        xqkv = ttnn.linear(x, self.wqkv, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv, num_heads=nh, num_kv_heads=nkv, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(xqkv)

        # Norms and RoPE in fp32 (HiFi4 + fp32 dest). K/V must reach bfp8 from real fp32 values: the device rounds
        # bf16 -> bfp8 ties away from zero (bf16 values sit exactly on bfp8 ties), which grew K norms by ~0.3 % and
        # the attention output by ~0.4 %; fp32 -> bfp8 is unbiased. rotary_embedding's default config adds ~0.2 %.
        q, k, v = (self._fp32(t) for t in (q, k, v))
        cos = ttnn.slice(self.cos, [0, 0, start, 0], [1, 1, end, d])
        sin = ttnn.slice(self.sin, [0, 0, start, 0], [1, 1, end, d])
        q = self._norm_rope(q, self.q_norm, cos, sin, ttnn.bfloat16)
        k8 = self._norm_rope(k, self.k_norm, cos, sin, ttnn.bfloat8_b)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        vn = self._head_norm(v, None)
        ttnn.deallocate(v)
        v8 = ttnn.typecast(vn, ttnn.bfloat8_b)
        ttnn.deallocate(vn)

        # Write the chunk into the cache (bfp8), then read back the window tail + chunk as K/V for SDPA.
        ttnn.fill_cache(cache.k, k8, batch_idx=0, update_idx=start)
        ttnn.fill_cache(cache.v, v8, batch_idx=0, update_idx=start)
        ttnn.deallocate(k8)
        ttnn.deallocate(v8)
        hist = min(start, ((self.window + TILE - 1) // TILE) * TILE)
        k_win = ttnn.slice(cache.k, [0, 0, start - hist, 0], [1, nkv, end, d])
        v_win = ttnn.slice(cache.v, [0, 0, start - hist, 0], [1, nkv, end, d])
        if hist:
            q_pad = ttnn.slice(q, [0, 0, 0, 0], [1, nh, hist, d])  # dummy queries, outputs dropped
            q_sq = ttnn.concat([q_pad, q], dim=2)
            ttnn.deallocate(q_pad)
            ttnn.deallocate(q)
        else:
            q_sq = q
        n = hist + s
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            q_chunk_size=min(self.q_chunk, n),
            k_chunk_size=min(self.k_chunk, n),
            exp_approx_mode=False,
        )
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_sq,
            k_win,
            v_win,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.window,
            program_config=program_config,
            compute_kernel_config=self.sdpa_cfg,
        )
        ttnn.deallocate(q_sq)
        ttnn.deallocate(k_win)
        ttnn.deallocate(v_win)
        if hist:
            full = attn
            attn = ttnn.slice(full, [0, 0, hist, 0], [1, nh, n, d])
            ttnn.deallocate(full)

        concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        partial = ttnn.linear(concat, self.wo, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg)
        ttnn.deallocate(concat)
        out = ttnn.all_reduce(partial, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(partial)
        return out


class TtGlobalAttention(TtSlidingAttention):
    """Gemma-4 global (full causal) attention: 16 Q heads x 512, 2 KV heads x 512, no v_proj.

    Column c holds Q heads 4c..4c+3 and KV head c // 2 (each KV head lives on 2 adjacent columns), so attention stays
    local to each chip (GQA 4:1). The per-chip fused weight is [Wq_c | Wk_{c//2}] and `nlp_create_qkv_heads(kv_tied)`
    reads raw K twice: K = rope(k_norm(k_raw)), V = rms(k_raw) without a weight. RoPE is the rotate-half op with the
    proportional inv_freq (zeros past 25 %: cos 1, sin 0 there). SDPA is chunked causal SDPA over the whole cache seen as
    pages of KV_BLOCK rows with an identity page table (one head per chip, so the paged view is a zero-cost reshape of
    the contiguous [1, 1, max_seq, D] cache).
    """

    kv_repeat = 2
    KV_BLOCK = 64

    def __init__(
        self, mesh, wq, wk, wo, q_norm, k_norm, *, num_heads, num_kv_heads, head_dim, rope_inv_freq, eps=1e-6, scale=1.0
    ):
        self.mesh = mesh
        tp = mesh.shape[TP_AXIS]
        assert num_heads % tp == 0 and num_kv_heads * self.kv_repeat == tp, (num_heads, num_kv_heads, tp)
        self.nh, self.nkv, self.d = num_heads // tp, 1, head_dim
        self.window, self.eps, self.scale = None, eps, scale
        self.inv_freq = rope_inv_freq.float()

        qd = self.nh * head_dim
        blocks = []
        for c in range(tp):
            h = c // self.kv_repeat
            blocks += [wq[c * qd : (c + 1) * qd], wk[h * head_dim : (h + 1) * head_dim]]
        self.wqkv = self._weight(torch.cat(blocks, dim=0).t().contiguous(), dim=-1)  # [hidden, tp * (qd + D)]
        self.wo = self._weight(wo.t().contiguous(), dim=-2)  # [nh*D, hidden], rows split by Q head
        self.q_norm = self._replicated(q_norm.float().reshape(1, 1, 1, -1))
        self.k_norm = self._replicated(k_norm.float().reshape(1, 1, 1, -1))

        self.mm_cfg = _kernel_config(mesh, settings.get("ATTN_MATMUL_FIDELITY"), settings.get("ATTN_MATMUL_FP32_ACC"))
        self.norm_cfg = _kernel_config(mesh, "HiFi4", True)
        self.sdpa_cfg = _kernel_config(mesh, settings.get("SDPA_FIDELITY"), settings.get("SDPA_FP32_ACC"))
        self.q_chunk, self.k_chunk = settings.get("GLOBAL_SDPA_Q_CHUNK"), settings.get("GLOBAL_SDPA_K_CHUNK")
        self.cos = self.sin = None
        self.rope_len = 0
        self.page_table = None

    def ensure_rope(self, max_seq: int):
        """RoPE tables and the identity page table for max_seq, built once (load time)."""
        grow = max_seq > self.rope_len
        super().ensure_rope(max_seq)
        if grow or self.page_table is None:
            assert max_seq % self.KV_BLOCK == 0, f"max_seq {max_seq} not a multiple of {self.KV_BLOCK}"
            if self.page_table is not None:
                ttnn.deallocate(self.page_table)
            self.page_table = ttnn.from_torch(
                torch.arange(max_seq // self.KV_BLOCK, dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

    def __call__(self, x, cache: TtKVCache, start: int):
        """x: [1, 1, S, hidden] bf16 replicated; positions [start, start+S). Returns [1, 1, S, hidden] replicated."""
        s = x.shape[-2]
        end = start + s
        assert start % TILE == 0 and s % TILE == 0, f"start {start} and S {s} must be tile aligned"
        assert end <= cache.max_seq <= self.rope_len, f"chunk end {end} / cache {cache.max_seq} / rope {self.rope_len}"
        nh, d = self.nh, self.d

        xqkv = ttnn.linear(x, self.wqkv, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=nh,
            num_kv_heads=1,
            transpose_k_heads=False,
            kv_tied=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        # fp32 norms + RoPE; K/V reach bfp8 from fp32 (see TtSlidingAttention.__call__).
        q, k, v = (self._fp32(t) for t in (q, k, v))
        cos = ttnn.slice(self.cos, [0, 0, start, 0], [1, 1, end, d])
        sin = ttnn.slice(self.sin, [0, 0, start, 0], [1, 1, end, d])
        q = self._norm_rope(q, self.q_norm, cos, sin, ttnn.bfloat16)
        k8 = self._norm_rope(k, self.k_norm, cos, sin, ttnn.bfloat8_b)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        vn = self._head_norm(v, None)
        ttnn.deallocate(v)
        v8 = ttnn.typecast(vn, ttnn.bfloat8_b)
        ttnn.deallocate(vn)

        ttnn.fill_cache(cache.k, k8, batch_idx=0, update_idx=start)
        ttnn.fill_cache(cache.v, v8, batch_idx=0, update_idx=start)
        ttnn.deallocate(k8)
        ttnn.deallocate(v8)

        pages = (cache.max_seq // self.KV_BLOCK, 1, self.KV_BLOCK, d)  # zero-cost view of the contiguous cache
        k_paged = ttnn.reshape(cache.k, pages)
        v_paged = ttnn.reshape(cache.v, pages)
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            q_chunk_size=min(self.q_chunk, s),
            k_chunk_size=min(self.k_chunk, s),
            exp_approx_mode=False,
        )
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            input_tensor_q=q,
            input_tensor_k=k_paged,
            input_tensor_v=v_paged,
            page_table_tensor=self.page_table,
            chunk_start_idx=start,
            scale=self.scale,
            program_config=program_config,
            compute_kernel_config=self.sdpa_cfg,
        )
        ttnn.deallocate(q)

        concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        partial = ttnn.linear(concat, self.wo, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg)
        ttnn.deallocate(concat)
        out = ttnn.all_reduce(partial, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(partial)
        return out

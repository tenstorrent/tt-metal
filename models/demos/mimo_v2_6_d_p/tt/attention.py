# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 full attention (chunked prefill) on a 1x4 mesh, TP by head, no CCL until o_proj.

64 Q heads / 4 KV heads, QK head dim 192, V head dim 128, partial rotate-half RoPE on dims [0, 64) (theta 1e7),
scale 192^-0.5, no sink. Chip r holds Q heads 16r..16r+15 and KV head r (GQA 16:1 inside the chip):

    qkv   = x @ Wqkv_r                    fused per chip [H, 16*192 + 192 + 192]; the V rows are x attention_value_scale
                                           and zero-padded 128 -> 192 so every head has one head dim (SDPA needs Dv == D)
    q,k,v = nlp_create_qkv_heads          [1, 16, S, 192], [1, 1, S, 192], [1, 1, S, 192]
    q,k   = concat(rope(x[..., :64]), x[..., 64:])   cos/sin [1, 1, max_seq, 64] built once, sliced per chunk
    cache[start:start+S] = k, v           paged_fill_cache into a paged-shaped cache, identity page table
    sdpa  = SDPA causal (chunk 0) / chunked SDPA over cache[0:start+S] (later chunks), scale 192^-0.5
    out   = all_reduce(concat_heads(sdpa) @ Wo_r)   row-parallel o_proj; Wo_r has zero rows for the 64 V pad dims

Adapted from models/demos/gemma4_a4b_d_p/tt/attention.py (TtGlobalAttention + TtKVCacheGlobal). The forward does no
host transfer: RoPE tables and the page table live on the device and are sliced there per chunk.
"""

from __future__ import annotations

import math
import os

import torch

import ttnn

NUM_CHIPS = 4
TILE = 32
KV_BLOCK = 64  # page size of the paged-shaped cache (identity page table)

# SDPA presets (env MIMO_SDPA_CFG). "A" is the streaming Blackhole config (ernie45_d_p / gemma4_a4b_d_p preset A:
# HiFi2, fp32 dest acc off, approx exp); head dim 192 fits q256/k256 in L1 (known issue: L1 at large head_dim).
# "base" is HiFi4 + fp32 dest acc (non-streaming kernel), exact exp.
SDPA_PRESETS = {
    "base": dict(fidelity="HiFi4", fp32=True, exp_approx=False, chunks=(128, 128)),
    "A": dict(fidelity="HiFi2", fp32=False, exp_approx=True, chunks=(256, 256)),
}


def sdpa_settings(sliding: bool = False) -> dict:
    """Full layers: env MIMO_SDPA_CFG (default "A"). Sliding layers: env MIMO_SLIDING_SDPA_CFG (default "base"):
    with the sink ~28 logits above the row max, the output scales like exp(max - sink), so any relative error in the
    QK scores is amplified. HiFi2 (norms biased toward zero) and bf16 dest (scores ~-450 unscaled rounded to steps
    of 2-4) put the per-head output ratio off by 3-6%; HiFi4 + fp32 acc gives rel 0.005 (C.sliding_moe.attention)."""
    if sliding:
        name = os.environ.get("MIMO_SLIDING_SDPA_CFG", "base")
    else:
        name = os.environ.get("MIMO_SDPA_CFG", "A")
    return dict(SDPA_PRESETS[name], name=name)


def _sdpa_compute_config(sliding: bool = False):
    c = sdpa_settings(sliding)
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, c["fidelity"]),
        math_approx_mode=False,
        fp32_dest_acc_en=c["fp32"],
        packer_l1_acc=False,
    )


def _hifi4():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


def _fit_chunk(size: int, seq: int, start: int = 0) -> int:
    """The largest power of two <= size (>= TILE) that divides seq and start (chunked SDPA needs chunk_start % q/k == 0)."""
    size = 1 << (size.bit_length() - 1)
    while size > TILE and (seq % size or start % size):
        size //= 2
    return size


def rope_tables(inv_freq: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half cos/sin [length, R] for positions [0, length), computed in fp64."""
    pos = torch.arange(0, length, dtype=torch.float64)
    freqs = pos[:, None] * inv_freq.double()[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().float(), emb.sin().float()


def _tp_qkv_o(mesh, wqkv: torch.Tensor, wo: torch.Tensor, dims, value_scale, dtype, q_scale: float = 1.0):
    """Per-chip fused qkv [H, (nq + 2 nkv) * D] and row-parallel o_proj [nq * D, H] (device, sharded over chips).

    Chip r holds Q heads nq*r.., KV heads nkv*r..; Q rows x q_scale; V rows x value_scale and zero-padded Dv -> D;
    o_proj gets zero rows for the pad dims so the padded SDPA output feeds it unchanged."""
    n = NUM_CHIPS
    hq, hkv, d, dv = dims
    assert hq % n == 0 and hkv % n == 0
    nq, nkv = hq // n, hkv // n
    H = wqkv.shape[1]
    wq, wk, wv = wqkv.float().split([hq * d, hkv * d, hkv * dv], dim=0)
    if value_scale is not None:
        wv = wv * value_scale
    if q_scale != 1.0:
        wq = wq * q_scale
    wv_pad = torch.zeros(hkv, d, H)
    wv_pad[:, :dv] = wv.reshape(hkv, dv, H)
    wv_pad = wv_pad.reshape(hkv * d, H)
    fused = []
    for r in range(n):
        q_r = wq[r * nq * d : (r + 1) * nq * d]
        k_r = wk[r * nkv * d : (r + 1) * nkv * d]
        v_r = wv_pad[r * nkv * d : (r + 1) * nkv * d]
        fused.append(torch.cat([q_r, k_r, v_r], dim=0).T)  # [H, (nq + 2 nkv) * d]
    wo_t = wo.float().T.reshape(hq, dv, -1)  # [Hq, Dv, H]
    wo_pad = torch.zeros(hq, d, wo_t.shape[-1])
    wo_pad[:, :dv] = wo_t

    def shard(t, dim):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        )

    return shard(torch.stack(fused)[:, None], 0), shard(wo_pad.reshape(1, 1, hq * d, -1), -2)


class TtKVCacheFull:
    """Full-length K and V for one full-attention layer, KV head r on chip r (4 heads, 4 chips).

    Per chip a paged-shaped [max_seq / B, 1, B, D] tensor (bit-identical to a contiguous [1, 1, max_seq, D] cache with
    one head), D = 192 for both K and V (V zero-padded from 128). The identity page table is resident; per-chunk slices
    are cut on the device."""

    def __init__(self, mesh, num_kv_heads: int, head_dim: int, v_head_dim: int, max_seq: int, dtype=ttnn.bfloat16):
        assert num_kv_heads == NUM_CHIPS
        self.max_seq = -(-max_seq // KV_BLOCK) * KV_BLOCK
        self.mesh, self.nkv, self.d, self.dv, self.dtype = mesh, num_kv_heads, head_dim, v_head_dim, dtype
        self.nb = self.max_seq // KV_BLOCK
        z = torch.zeros(num_kv_heads, self.max_seq, head_dim)
        self.k, self.v = self._dev(z), self._dev(z)
        self.page_table = ttnn.from_torch(
            torch.arange(self.nb, dtype=torch.int32)[None],
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self._chunk_pt = {}

    def _dev(self, t: torch.Tensor) -> ttnn.Tensor:
        """Host [nkv, max_seq, D] -> device paged [nb, 1, B, D] per chip (chip r = head r)."""
        paged = t.reshape(NUM_CHIPS, self.nb, KV_BLOCK, self.d).transpose(0, 1).contiguous()  # [nb, 4, B, D]
        return ttnn.from_torch(
            paged.to(torch.bfloat16),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=1),
        )

    def chunk_page_table(self, start: int, seq: int) -> ttnn.Tensor:
        """Rows [start / B, (start + seq) / B) of the resident identity table, sliced on the device."""
        assert start % KV_BLOCK == 0 and seq % KV_BLOCK == 0
        key = (start, seq)
        if key not in self._chunk_pt:
            for v in self._chunk_pt.values():
                ttnn.deallocate(v)
            self._chunk_pt = {key: ttnn.slice(self.page_table, [0, start // KV_BLOCK], [1, (start + seq) // KV_BLOCK])}
        return self._chunk_pt[key]

    def load_prefix(self, key: torch.Tensor, value: torch.Tensor, length: int) -> None:
        """Host key [nkv, >= length, 192], value [nkv, >= length, 128]: positions [0, length) copied, rest zero."""

        def full(t):
            f = torch.zeros(self.nkv, self.max_seq, self.d)
            f[:, :length, : t.shape[-1]] = t[:, :length].float()
            return f

        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)
        self.k, self.v = self._dev(full(key)), self._dev(full(value))

    def to_torch(self, length: int) -> dict:
        def host(t, d):
            parts = ttnn.get_device_tensors(t)
            heads = [ttnn.to_torch(parts[h]).float() for h in range(self.nkv)]  # each [nb, 1, B, D]
            return torch.stack([p[:, 0].reshape(-1, self.d) for p in heads])[:, :length, :d]

        return {"key": host(self.k, self.d), "value": host(self.v, self.dv)}

    def free(self):
        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)
        ttnn.deallocate(self.page_table)
        for v in self._chunk_pt.values():
            ttnn.deallocate(v)
        self._chunk_pt = {}


class TtFullAttention:
    def __init__(
        self,
        mesh,
        wqkv: torch.Tensor,
        wo: torch.Tensor,
        dims: tuple[int, int, int, int],
        inv_freq: torch.Tensor,
        max_seq: int,
        value_scale: float | None,
        dtype=ttnn.bfloat16,
    ):
        """wqkv: dequantized fused [q; k; v] [Hq*D + Hkv*D + Hkv*Dv, H] in global order (reference/weights.qkv_weight);
        wo: [H, Hq*Dv]; dims: (Hq, Hkv, D, Dv); inv_freq: [R/2] RoPE frequencies (R = rotated dims)."""
        n = NUM_CHIPS
        hq, hkv, d, dv = dims
        assert hkv == n and hq % n == 0
        self.mesh, self.d, self.dv = mesh, d, dv
        self.nq = hq // n
        self.rope_dim = 2 * inv_freq.shape[0]
        # Rounded to fp32: the chunked SDPA binding takes scale as noconvert, and nanobind then rejects a Python float
        # that fp32 cannot hold exactly (192^-0.5 is not), with an "incompatible function arguments" TypeError.
        self.scale = float(torch.tensor(d**-0.5, dtype=torch.float32).item())
        self.wqkv, self.wo = _tp_qkv_o(mesh, wqkv, wo, dims, value_scale, dtype)
        cos, sin = rope_tables(inv_freq, -(-max_seq // TILE) * TILE)
        self.cos = self._replicate(cos[None, None])
        self.sin = self._replicate(sin[None, None])
        self.max_seq = cos.shape[0]

    def _shard(self, t, dim, dtype):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=dtype,
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

    def _partial_rope(self, t, cos, sin):
        """Rotate-half RoPE on dims [0, R) of t [1, h, S, D]; dims [R, D) pass through."""
        _, h, s, d = t.shape
        r = self.rope_dim
        tr = ttnn.slice(t, [0, 0, 0, 0], [1, h, s, r])
        tp = ttnn.slice(t, [0, 0, 0, r], [1, h, s, d])
        rot = ttnn.experimental.rotary_embedding(tr, cos, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tr)
        out = ttnn.concat([rot, tp], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rot)
        ttnn.deallocate(tp)
        return out

    def _sdpa_program_config(self, seq: int, start: int):
        c = sdpa_settings()
        q, k = _fit_chunk(c["chunks"][0], seq, start), _fit_chunk(c["chunks"][1], seq, start)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            q_chunk_size=q,
            k_chunk_size=k,
            exp_approx_mode=c["exp_approx"],
        )

    def __call__(self, x: ttnn.Tensor, start: int, cache: TtKVCacheFull, kv_sink=None) -> ttnn.Tensor:
        """x: replicated [1, 1, S, H] TILE (attn_norm output), queries at [start, start+S). Returns replicated
        [1, 1, S, H] (all-reduced). Writes this chunk's K/V into cache positions [start, start+S).
        kv_sink(k, v), if given, also receives this chunk's per-chip K/V [1, 1, S, 192] (V padded)."""
        seq = x.shape[-2]
        assert start % KV_BLOCK == 0 and seq % KV_BLOCK == 0 and start + seq <= self.max_seq

        qkv = ttnn.linear(x, self.wqkv, compute_kernel_config=_hifi4(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.nq, num_kv_heads=1, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)

        r = self.rope_dim
        cos = ttnn.slice(self.cos, [0, 0, start, 0], [1, 1, start + seq, r])
        sin = ttnn.slice(self.sin, [0, 0, start, 0], [1, 1, start + seq, r])
        qr = self._partial_rope(q, cos, sin)
        ttnn.deallocate(q)
        kr = self._partial_rope(k, cos, sin)
        ttnn.deallocate(k)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        q, k = qr, kr

        if kv_sink is not None:
            kv_sink(k, v)
        pt = cache.chunk_page_table(start, seq)
        ttnn.experimental.paged_fill_cache(cache.k, k, pt, batch_idx=0)
        ttnn.experimental.paged_fill_cache(cache.v, v, pt, batch_idx=0)

        prog = self._sdpa_program_config(seq, start)
        if start == 0:
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=self.scale,
                program_config=prog,
                compute_kernel_config=_sdpa_compute_config(),
            )
        else:
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                cache.k,
                cache.v,
                cache.page_table,
                int(start),
                scale=self.scale,
                program_config=prog,
                compute_kernel_config=_sdpa_compute_config(),
            )
        for t in (q, k, v):
            ttnn.deallocate(t)

        a = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, S, 16*192]
        ttnn.deallocate(attn)
        o = ttnn.linear(a, self.wo, compute_kernel_config=_hifi4(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(a)
        out = ttnn.all_reduce(o, cluster_axis=1)
        ttnn.deallocate(o)
        return out

    def free(self):
        for t in (self.wqkv, self.wo, self.cos, self.sin):
            ttnn.deallocate(t)


# --------------------------------------------------------------------------------------
# Sliding-window layers (window 128, per-head sink)
# --------------------------------------------------------------------------------------


class TtKVCacheSliding:
    """Full-length K and V for one sliding layer: contiguous [1, 8, max_seq, D] sharded by head, [1, 2, max_seq, D] per
    chip (KV heads 2r, 2r+1 on chip r), D = 192 for K and V (V zero-padded from 128). Written with fill_cache at the
    chunk offset; the attention reads the previous window rows back with ttnn.slice."""

    def __init__(self, mesh, num_kv_heads: int, head_dim: int, v_head_dim: int, max_seq: int, dtype=ttnn.bfloat16):
        assert num_kv_heads % NUM_CHIPS == 0
        self.max_seq = -(-max_seq // TILE) * TILE
        self.mesh, self.nkv, self.d, self.dv, self.dtype = mesh, num_kv_heads, head_dim, v_head_dim, dtype
        z = torch.zeros(1, num_kv_heads, self.max_seq, head_dim)
        self.k, self.v = self._dev(z), self._dev(z)

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
        """Host key [nkv, >= length, 192], value [nkv, >= length, 128]: positions [0, length) copied, rest zero."""

        def full(t):
            f = torch.zeros(1, self.nkv, self.max_seq, self.d)
            f[0, :, :length, : t.shape[-1]] = t[:, :length].float()
            return f

        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)
        self.k, self.v = self._dev(full(key)), self._dev(full(value))

    def to_torch(self, length: int) -> dict:
        def host(t, d):
            parts = [ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(t)]  # each [1, nkv/4, max_seq, D]
            return torch.cat(parts, dim=1)[0, :, :length, :d]

        return {"key": host(self.k, self.d), "value": host(self.v, self.dv)}

    def free(self):
        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)


class TtSlidingAttention(TtFullAttention):
    """MiMo-V2 sliding-window attention (window W = 128, key j visible to query i iff i - W < j <= i) with a per-head
    sink logit, TP by KV head: chip r holds Q heads 16r..16r+15 and KV heads 2r, 2r+1 (GQA 8:1 on the chip).

        qkv   = x @ Wqkv_r                    [H, 16*192 + 2*192 + 2*192] (V x value_scale, padded 128 -> 192)
        q,k,v = nlp_create_qkv_heads          [1, 16, S, 192], [1, 2, S, 192], [1, 2, S, 192]
        q,k   = partial RoPE dims [0, 64), theta 1e4 (tables built once, sliced per chunk on the device)
        cache[start:start+S] = k, v           fill_cache at the chunk offset
        tail  = cache[start - T : start]      T = min(round_up(W, 32), start) rows read back on the device
        sdpa  = SDPA(causal, sliding_window_size=W, attention_sink) over [tail | chunk]; Q front-padded by T rows
                 (their outputs are dropped). Sink stored pre-divided by the scale: the kernel scales the sink logit,
                 HF does not (as gpt_oss_d_p weights.py).
        out   = all_reduce(concat_heads(sdpa[T:]) @ Wo_r)

    Adapted from gemma4_a4b_d_p TtSlidingAttention (window tail concat) and gpt_oss_d_p (sinks)."""

    def __init__(
        self,
        mesh,
        wqkv: torch.Tensor,
        wo: torch.Tensor,
        dims: tuple[int, int, int, int],
        inv_freq: torch.Tensor,
        max_seq: int,
        value_scale: float | None,
        window: int,
        sink: torch.Tensor | None,
        dtype=ttnn.bfloat16,
    ):
        n = NUM_CHIPS
        hq, hkv, d, dv = dims
        assert hq % n == 0 and hkv % n == 0
        self.mesh, self.d, self.dv, self.window = mesh, d, dv, int(window)
        self.nq, self.nkv = hq // n, hkv // n
        self.rope_dim = 2 * inv_freq.shape[0]
        # SDPA scale: a power of two, the rest folded into the Q rows. The kernel folds the sink as
        # exp((sink - max) * scale) with the scale truncated to bf16 (compute_streaming.hpp, scale_fp32 >> 16):
        # 192^-0.5 -> 0.07178 (-0.54%), and with the row max ~390 (unscaled) below the sink that shrinks the sink
        # logit by ~0.15 and inflates the output ~14%. A power of two is exact in bf16.
        true_scale = d**-0.5
        self.scale = 2.0 ** math.floor(math.log2(true_scale))
        q_scale = true_scale / self.scale
        self.wqkv, self.wo = _tp_qkv_o(mesh, wqkv, wo, dims, value_scale, dtype, q_scale=q_scale)
        self.sink = None
        if sink is not None:
            s = (sink.float() / self.scale).reshape(1, hq, 1, 1)
            self.sink = self._shard(s, dim=1, dtype=ttnn.bfloat16)  # [1, 16, 1, 1] per chip
        cos, sin = rope_tables(inv_freq, -(-max_seq // TILE) * TILE)
        self.cos = self._replicate(cos[None, None])
        self.sin = self._replicate(sin[None, None])
        self.max_seq = cos.shape[0]

    def _sliding_program_config(self, total: int):
        c = sdpa_settings(sliding=True)
        q = _fit_chunk(c["chunks"][0], total)
        k = _fit_chunk(min(c["chunks"][1], max(TILE, self.window)), total)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            q_chunk_size=q,
            k_chunk_size=k,
            exp_approx_mode=c["exp_approx"],
        )

    def __call__(self, x: ttnn.Tensor, start: int, cache: TtKVCacheSliding, kv_sink=None) -> ttnn.Tensor:
        """x: replicated [1, 1, S, H] TILE (attn_norm output), queries at [start, start+S). Returns replicated
        [1, 1, S, H] (all-reduced). Writes this chunk's K/V into cache positions [start, start+S).
        kv_sink(k, v), if given, also receives this chunk's per-chip K/V [1, 2, S, 192] (V padded)."""
        seq = x.shape[-2]
        D, nq, nkv = self.d, self.nq, self.nkv
        assert start % TILE == 0 and seq % TILE == 0 and start + seq <= min(self.max_seq, cache.max_seq)

        qkv = ttnn.linear(x, self.wqkv, compute_kernel_config=_hifi4(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=nq, num_kv_heads=nkv, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)

        r = self.rope_dim
        cos = ttnn.slice(self.cos, [0, 0, start, 0], [1, 1, start + seq, r])
        sin = ttnn.slice(self.sin, [0, 0, start, 0], [1, 1, start + seq, r])
        qr = self._partial_rope(q, cos, sin)
        ttnn.deallocate(q)
        kr = self._partial_rope(k, cos, sin)
        ttnn.deallocate(k)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        q, k = qr, kr

        if kv_sink is not None:
            kv_sink(k, v)
        ttnn.fill_cache(cache.k, k, batch_idx=0, update_idx=start)
        ttnn.fill_cache(cache.v, v, batch_idx=0, update_idx=start)

        hist = min(-(-self.window // TILE) * TILE, start)
        if hist:
            k_tail = ttnn.slice(cache.k, [0, 0, start - hist, 0], [1, nkv, start, D])
            v_tail = ttnn.slice(cache.v, [0, 0, start - hist, 0], [1, nkv, start, D])
            if seq >= hist:
                q_pad = ttnn.slice(q, [0, 0, 0, 0], [1, nq, hist, D])
            else:
                q_pad = ttnn.zeros([1, nq, hist, D], dtype=q.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh)
            q_cat = ttnn.concat([q_pad, q], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            k_cat = ttnn.concat([k_tail, k], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            v_cat = ttnn.concat([v_tail, v], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for t in (q_pad, k_tail, v_tail, q, k, v):
                ttnn.deallocate(t)
        else:
            q_cat, k_cat, v_cat = q, k, v

        full = ttnn.transformer.scaled_dot_product_attention(
            q_cat,
            k_cat,
            v_cat,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.window,
            attention_sink=self.sink,
            program_config=self._sliding_program_config(hist + seq),
            compute_kernel_config=_sdpa_compute_config(sliding=True),
        )
        for t in (q_cat, k_cat, v_cat):
            ttnn.deallocate(t)
        if hist:
            attn = ttnn.slice(full, [0, 0, hist, 0], [1, nq, hist + seq, D])
            ttnn.deallocate(full)
        else:
            attn = full

        a = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, S, 16*192]
        ttnn.deallocate(attn)
        o = ttnn.linear(a, self.wo, compute_kernel_config=_hifi4(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(a)
        out = ttnn.all_reduce(o, cluster_axis=1)
        ttnn.deallocate(o)
        return out

    def free(self):
        super().free()
        if self.sink is not None:
            ttnn.deallocate(self.sink)

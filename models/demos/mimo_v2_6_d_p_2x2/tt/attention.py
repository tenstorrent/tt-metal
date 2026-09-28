# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 full attention (chunked prefill) on the 2x2 mesh, TP=4 by head over the flattened mesh, no CCL until o_proj.

2x2 port of models/demos/mimo_v2_6_d_p/tt/attention.py (1x4). Changed: chip d = 2*row + col (ShardTensorToMesh over
the 2x2 mesh, row-major device order) holds TP rank d, and the o_proj reduce is ttnn.all_reduce(cluster_axis=None),
which on a non-line mesh runs axis 1 then axis 0 (two 2-chip stages). Everything else is the prior's module. The
sliding-window classes (TtKVCacheSliding, TtSlidingAttention) are ported the same way.

64 Q heads / 4 KV heads, QK head dim 192, V head dim 128, partial rotate-half RoPE on dims [0, 64) (theta 1e7),
scale 192^-0.5, no sink. Chip r holds Q heads 16r..16r+15 and KV head r (GQA 16:1 inside the chip). V at its real
head dim 128 end to end (the default):

    q     = x @ Wq_r                      [H, 16*192]
    kv    = x @ Wkv_r                     [H, 1 * (192 + 128)], rows per KV head [k_h | v_h]; V rows x attention_value_scale
    q_rot, q_pass = nlp_create_q_heads_split(q, 16, 64)     [1, 16, S, 64], [1, 16, S, 128] (the RoPE split for free)
    k, v  = nlp_create_q_heads_split(kv, 1, 192)            [1, 1, S, 192], [1, 1, S, 128]
    q,k   = concat(rope(x[..., :64]), x[..., 64:])   cos/sin [1, 1, max_seq, 64] built once, sliced per chunk
    cache[start:start+S] = k, v           paged_fill_cache into paged-shaped caches (K 192 wide, V 128), identity page table
    sdpa  = ttnn.bringup SDPA causal (chunk 0) / chunked SDPA over cache[0:start+S] (later chunks), scale 192^-0.5;
            the fork takes V narrower than K and returns [1, 16, S, 128]
    out   = all_reduce(concat_heads(sdpa) @ Wo_r)   row-parallel o_proj, K = 16*128 = 2048 per chip; axis 1 then axis 0

Head split, measured per chip at S 5120 (device us, 1 KV head / 2 KV heads, qkv matmul + head split + the Q RoPE
split): one fused [q|k|v] matmul + nlp_create_qkv_heads + slice Q to [0:64] / [64:192] (the padded path) 3996 / 4278;
the same + a slice of V to 128: 4001 / 4295; one fused unpadded matmul + two column slices + two
nlp_create_q_heads_split 4022 / 4314; two matmuls (Q; per-head interleaved KV) + two nlp_create_q_heads_split
3723 / 4131 (chosen: the Q matmul at N 3072 runs 3190 us against 3637 us for the fused N 3392/3456, the KV matmul
345 / 722, and the Q heads come out already split for RoPE); three matmuls (Q, K, V) 3911 / 5366 (a permute per
K/V for 2 heads).

MIMO_V_PAD=1 restores the padded path: fused [H, 16*192 + 192 + 192] qkv with V zero-padded 128 -> 192,
nlp_create_qkv_heads, V caches 192 wide, ttnn.transformer SDPA with V 192, and o_proj with zero rows for the 64 pad
dims of every head (K 3072 per chip). It is read when the modules and caches are built.

Adapted from models/demos/gemma4_a4b_d_p/tt/attention.py (TtGlobalAttention + TtKVCacheGlobal). The forward does no
host transfer: RoPE tables and the page table live on the device and are sliced there per chunk.
"""

from __future__ import annotations

import math
import os

import torch

import ttnn
from models.demos.common.bringup.testing import profiler

NUM_CHIPS = 4  # 2x2 mesh, TP rank d on chip d = 2*row + col
TILE = 32
KV_BLOCK = 64  # page size of the paged-shaped cache (identity page table)

# SDPA presets (env MIMO_SDPA_CFG / MIMO_SLIDING_SDPA_CFG). fp32 dest acc turns off the streaming SDPA kernel on
# Blackhole (sdpa_program_factory.cpp:75); every preset except "base" keeps it off.
#   "base": the bring-up config, HiFi4 + fp32 dest acc (non-streaming kernel), exact exp, q128/k128.
#   "A":    config A of ernie45_d_p / gemma4_a4b_d_p: HiFi2, fp32 dest off, approx exp. Full layers use q512/k128: the
#           chunked causal SDPA gets no KV chain forwarding, so every Q chunk streams the whole prefix from DRAM, and the
#           larger Q chunk halves that traffic (51k prefix: q256/k256 67.8 ms, q512/k128 54.3 ms, q512/k64 75.3 ms for 2
#           layers). q512/k256 (1.78 MB) and q1024/k64 (1.91 MB) exceed L1 (1.57 MB) at head_dim 192.
#   "S":    the sliding (sink) layers: streaming kernel with HiFi4 and exact exp. Config A fails the frozen sliding
#           component test (row norm ratio 1.0504 > 1.05); S passes (rel 0.0136, ratio [0.964, 1.046]; base 0.0086,
#           [0.975, 1.021]) and is as fast as A there (4 layers: 2.5 ms vs 19.7 ms base). The kernel's only speed
#           lever at window 128 is streaming; fidelity and approx exp do not change the time.
SDPA_PRESETS = {
    "base": dict(fidelity="HiFi4", fp32=True, exp_approx=False, chunks=(128, 128)),
    "A": dict(fidelity="HiFi2", fp32=False, exp_approx=True, chunks=(512, 128)),
    "S": dict(fidelity="HiFi4", fp32=False, exp_approx=False, chunks=(128, 128)),
    # Preset A at HiFi4 (the 2x2 default before P.1, per the HiFi4 rule). MIMO_SDPA_CFG=A4 selects it for comparison.
    "A4": dict(fidelity="HiFi4", fp32=False, exp_approx=True, chunks=(512, 128)),
}
# P.1 (owner decision 2026-09-28): full-attention SDPA (layers 0, 5) runs preset A (HiFi2), as the 1x4 prior ends.
# An explicit owner exception to the HiFi4 rule for this one op; every other matmul stays at HiFi4.
FULL_SDPA_DEFAULT = "A"


# Profile sub-sections inside attention (qkv, rope, kv_write, kv_tail, sdpa, o_proj, ccl). signpost is a no-op unless
# the bring-up profiler is enabled. MIMO_ATTN_SIGNPOSTS=0 makes attention one profile section again.
ATTN_SIGNPOSTS = os.environ.get("MIMO_ATTN_SIGNPOSTS", "1") != "0"


def v_pad_enabled() -> bool:
    """MIMO_V_PAD=1: V zero-padded to the QK head dim (the path before ttnn.bringup SDPA took a narrow V)."""
    return os.environ.get("MIMO_V_PAD", "0") == "1"


def _sp(name: str) -> None:
    if ATTN_SIGNPOSTS:
        profiler.signpost(f"attention.{name}")


def sdpa_settings(sliding: bool = False) -> dict:
    """Full layers: env MIMO_SDPA_CFG (default "A"; "A4" = the HiFi4 variant). Sliding layers: env MIMO_SLIDING_SDPA_CFG (default "S", or "base"
    when MIMO_SDPA_CFG=base, so that one variable restores the whole bring-up config). With the sink ~28 logits above
    the row max, the sliding output scales like exp(max - sink), so any relative error in the QK scores is amplified:
    bf16 dest is most of S's extra error over base, HiFi2 on top pushes the norm ratio to the limit.
    Env MIMO_[SLIDING_]SDPA_Q / _K override the chunk sizes (sweeps)."""
    if sliding:
        full = os.environ.get("MIMO_SDPA_CFG", FULL_SDPA_DEFAULT)
        name = os.environ.get("MIMO_SLIDING_SDPA_CFG", "base" if full == "base" else "S")
    else:
        name = os.environ.get("MIMO_SDPA_CFG", FULL_SDPA_DEFAULT)
    c = dict(SDPA_PRESETS[name], name=name)
    pre = "MIMO_SLIDING_SDPA" if sliding else "MIMO_SDPA"
    env = os.environ.get
    if env(f"{pre}_Q") or env(f"{pre}_K"):  # chunk-size overrides for sweeps
        c["chunks"] = (int(env(f"{pre}_Q", c["chunks"][0])), int(env(f"{pre}_K", c["chunks"][1])))
    return c


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


def _tp_weights(
    mesh, wqkv: torch.Tensor, wo: torch.Tensor, dims, value_scale, dtype, q_scale: float = 1.0, v_pad=False
):
    """Per-chip projection weights (device, sharded over chips); chip r holds Q heads nq*r.., KV heads nkv*r..; Q rows
    x q_scale, V rows x value_scale. Returns (wq, wkv, wo):
      * V at 128 (default): wq [H, nq * D], wkv [H, nkv * (D + Dv)] with rows per KV head [k_h | v_h], row-parallel
        o_proj [nq * Dv, H];
      * v_pad: wq = the fused [H, (nq + 2 nkv) * D] qkv with V zero-padded Dv -> D, wkv None, o_proj [nq * D, H] with
        zero rows for the pad dims so the padded SDPA output feeds it unchanged."""
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

    def shard(t, dim):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        )

    if not v_pad:
        wk_h, wv_h = wk.reshape(hkv, d, H), wv.reshape(hkv, dv, H)
        wkv = torch.cat([wk_h, wv_h], dim=1)  # [Hkv, D + Dv, H]: per head [k_h | v_h], KV heads in chip order
        q_t = torch.stack([wq[r * nq * d : (r + 1) * nq * d].T for r in range(n)])  # [n, H, nq * D]
        kv_t = torch.stack([wkv[r * nkv : (r + 1) * nkv].reshape(-1, H).T for r in range(n)])  # [n, H, nkv*(D+Dv)]
        wo_t = wo.float().T.reshape(1, 1, hq * dv, -1)  # [1, 1, Hq * Dv, H], rows head-major like concat_heads
        return shard(q_t[:, None], 0), shard(kv_t[:, None], 0), shard(wo_t, -2)

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
    return shard(torch.stack(fused)[:, None], 0), None, shard(wo_pad.reshape(1, 1, hq * d, -1), -2)


class TtKVCacheFull:
    """Full-length K and V for one full-attention layer, KV head d on chip d = 2*row + col (4 heads, 2x2 mesh).

    Per chip a paged-shaped [max_seq / B, 1, B, W] tensor (bit-identical to a contiguous [1, 1, max_seq, W] cache with
    one head), W = 192 for K and 128 for V (192, zero-padded, under MIMO_V_PAD=1). The identity page table is resident;
    per-chunk slices are cut on the device."""

    def __init__(self, mesh, num_kv_heads: int, head_dim: int, v_head_dim: int, max_seq: int, dtype=ttnn.bfloat16):
        assert num_kv_heads == NUM_CHIPS
        self.max_seq = -(-max_seq // KV_BLOCK) * KV_BLOCK
        self.mesh, self.nkv, self.d, self.dv, self.dtype = mesh, num_kv_heads, head_dim, v_head_dim, dtype
        self.nb = self.max_seq // KV_BLOCK
        self.vw = head_dim if v_pad_enabled() else v_head_dim  # V width on the device
        self.k = self._dev(torch.zeros(num_kv_heads, self.max_seq, head_dim))
        self.v = self._dev(torch.zeros(num_kv_heads, self.max_seq, self.vw))
        self.page_table = ttnn.from_torch(
            torch.arange(self.nb, dtype=torch.int32)[None],
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self._chunk_pt = {}

    def _dev(self, t: torch.Tensor) -> ttnn.Tensor:
        """Host [nkv, max_seq, W] -> device paged [nb, 1, B, W] per chip (chip r = head r)."""
        paged = t.reshape(NUM_CHIPS, self.nb, KV_BLOCK, t.shape[-1]).transpose(0, 1).contiguous()  # [nb, 4, B, W]
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

        def full(t, w):
            f = torch.zeros(self.nkv, self.max_seq, w)
            f[:, :length, : t.shape[-1]] = t[:, :length].float()
            return f

        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)
        self.k, self.v = self._dev(full(key, self.d)), self._dev(full(value, self.vw))

    def to_torch(self, length: int) -> dict:
        def host(t, d):
            parts = ttnn.get_device_tensors(t)
            heads = [ttnn.to_torch(parts[h]).float() for h in range(self.nkv)]  # each [nb, 1, B, W]
            return torch.stack([p[:, 0].reshape(-1, p.shape[-1]) for p in heads])[:, :length, :d]

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
        self.nq, self.nkv = hq // n, 1
        self.rope_dim = 2 * inv_freq.shape[0]
        self.v_pad = v_pad_enabled()
        self.vw = d if self.v_pad else dv
        # Rounded to fp32: the chunked SDPA binding takes scale as noconvert, and nanobind then rejects a Python float
        # that fp32 cannot hold exactly (192^-0.5 is not), with an "incompatible function arguments" TypeError.
        self.scale = float(torch.tensor(d**-0.5, dtype=torch.float32).item())
        self.wq, self.wkv, self.wo = _tp_weights(mesh, wqkv, wo, dims, value_scale, dtype, v_pad=self.v_pad)
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
        """Rotate-half RoPE on dims [0, R) of t [1, h, S, D]; dims [R, D) pass through. t may also be the pair
        (t[..., :R], t[..., R:]) already split (taken over and freed)."""
        if isinstance(t, tuple):
            tr, tp = t
        else:
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

    def _qkv_rope(self, x, start: int, seq: int):
        """x [1, 1, S, H] -> q [1, nq, S, D], k [1, nkv, S, D] (both post-RoPE), v [1, nkv, S, V width]."""
        _sp("qkv")
        if self.v_pad:
            qkv = ttnn.linear(x, self.wq, compute_kernel_config=_hifi4(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv,
                num_heads=self.nq,
                num_kv_heads=self.nkv,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv)
        else:
            qp = ttnn.linear(x, self.wq, compute_kernel_config=_hifi4(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kvp = ttnn.linear(x, self.wkv, compute_kernel_config=_hifi4(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            q = tuple(  # (q[..., :R], q[..., R:]): the RoPE split comes out of the head split
                ttnn.experimental.nlp_create_q_heads_split(
                    qp, num_heads=self.nq, split_head_dim=self.rope_dim, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            )
            k, v = ttnn.experimental.nlp_create_q_heads_split(
                kvp, num_heads=self.nkv, split_head_dim=self.d, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(qp)
            ttnn.deallocate(kvp)

        _sp("rope")
        r = self.rope_dim
        cos = ttnn.slice(self.cos, [0, 0, start, 0], [1, 1, start + seq, r])
        sin = ttnn.slice(self.sin, [0, 0, start, 0], [1, 1, start + seq, r])
        qr = self._partial_rope(q, cos, sin)
        if not isinstance(q, tuple):
            ttnn.deallocate(q)
        kr = self._partial_rope(k, cos, sin)
        ttnn.deallocate(k)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        return qr, kr, v

    def _sdpa_ops(self):
        """(scaled_dot_product_attention, chunked_scaled_dot_product_attention): the ttnn.bringup fork takes a V
        narrower than K; MIMO_V_PAD=1 keeps the source ops (V padded to K's width)."""
        m = ttnn.transformer if self.v_pad else ttnn.bringup
        return m.scaled_dot_product_attention, m.chunked_scaled_dot_product_attention

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
        kv_sink(k, v), if given, also receives this chunk's per-chip K [1, 1, S, 192] and V [1, 1, S, 128] (192,
        zero-padded, under MIMO_V_PAD=1)."""
        seq = x.shape[-2]
        assert start % KV_BLOCK == 0 and seq % KV_BLOCK == 0 and start + seq <= self.max_seq
        assert cache.vw == self.vw, f"KV cache V width {cache.vw} != attention's {self.vw} (MIMO_V_PAD changed?)"

        q, k, v = self._qkv_rope(x, start, seq)

        _sp("kv_write")
        if kv_sink is not None:
            kv_sink(k, v)
        pt = cache.chunk_page_table(start, seq)
        ttnn.experimental.paged_fill_cache(cache.k, k, pt, batch_idx=0)
        ttnn.experimental.paged_fill_cache(cache.v, v, pt, batch_idx=0)

        _sp("sdpa")
        prog = self._sdpa_program_config(seq, start)
        sdpa, chunked_sdpa = self._sdpa_ops()
        if start == 0:
            attn = sdpa(
                q,
                k,
                v,
                is_causal=True,
                scale=self.scale,
                program_config=prog,
                compute_kernel_config=_sdpa_compute_config(),
            )
        else:
            attn = chunked_sdpa(
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

        _sp("o_proj")
        a = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, S, 16*128]
        ttnn.deallocate(attn)
        o = ttnn.linear(a, self.wo, compute_kernel_config=_hifi4(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(a)
        _sp("ccl")
        # 2x2: cluster_axis=None reduces over axis 1 then axis 0 (all_reduce.cpp), i.e. all 4 TP ranks.
        out = ttnn.all_reduce(o, cluster_axis=None)
        ttnn.deallocate(o)
        return out

    def free(self):
        for t in (self.wq, self.wkv, self.wo, self.cos, self.sin):
            if t is not None:
                ttnn.deallocate(t)


# --------------------------------------------------------------------------------------
# Sliding-window layers (window 128, per-head sink)
# --------------------------------------------------------------------------------------


class TtKVCacheSliding:
    """Full-length K and V for one sliding layer: contiguous [1, 8, max_seq, W] sharded by head, [1, 2, max_seq, W] per
    chip (KV heads 2d, 2d+1 on chip d = 2*row + col), W = 192 for K and 128 for V (192, zero-padded, under MIMO_V_PAD=1). Written
    with fill_cache at the chunk offset; the attention reads the previous window rows back with ttnn.slice."""

    def __init__(self, mesh, num_kv_heads: int, head_dim: int, v_head_dim: int, max_seq: int, dtype=ttnn.bfloat16):
        assert num_kv_heads % NUM_CHIPS == 0
        self.max_seq = -(-max_seq // TILE) * TILE
        self.mesh, self.nkv, self.d, self.dv, self.dtype = mesh, num_kv_heads, head_dim, v_head_dim, dtype
        self.vw = head_dim if v_pad_enabled() else v_head_dim  # V width on the device
        self.k = self._dev(torch.zeros(1, num_kv_heads, self.max_seq, head_dim))
        self.v = self._dev(torch.zeros(1, num_kv_heads, self.max_seq, self.vw))

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

        def full(t, w):
            f = torch.zeros(1, self.nkv, self.max_seq, w)
            f[0, :, :length, : t.shape[-1]] = t[:, :length].float()
            return f

        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)
        self.k, self.v = self._dev(full(key, self.d)), self._dev(full(value, self.vw))

    def to_torch(self, length: int) -> dict:
        def host(t, d):
            parts = [ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(t)]  # each [1, nkv/4, max_seq, W]
            return torch.cat(parts, dim=1)[0, :, :length, :d]

        return {"key": host(self.k, self.d), "value": host(self.v, self.dv)}

    def free(self):
        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)


class TtSlidingAttention(TtFullAttention):
    """MiMo-V2 sliding-window attention (window W = 128, key j visible to query i iff i - W < j <= i) with a per-head
    sink logit, TP by KV head: chip d = 2*row + col holds Q heads 16d..16d+15, KV heads 2d, 2d+1 and sink values
    16d..16d+15 (GQA 8:1 on the chip).

        q, kv = x @ Wq_r, x @ Wkv_r           [H, 16*192], [H, 2 * (192 + 128)] (V x value_scale), as TtFullAttention
        q,k,v = nlp_create_q_heads_split x2   [1, 16, S, 64] + [1, 16, S, 128], [1, 2, S, 192], [1, 2, S, 128]
        q,k   = partial RoPE dims [0, 64), theta 1e4 (tables built once, sliced per chunk on the device)
        cache[start:start+S] = k, v           fill_cache at the chunk offset
        tail  = cache[start - T : start]      T = min(round_up(W, 32), start) rows read back on the device
        sdpa  = ttnn.bringup SDPA(causal, sliding_window_size=W, attention_sink) over [tail | chunk], V 128; Q
                 front-padded by T rows (their outputs are dropped). Sink stored pre-divided by the scale: the kernel scales the sink logit,
                 HF does not (as gpt_oss_d_p weights.py).
        out   = all_reduce(concat_heads(sdpa[T:]) @ Wo_r)   o_proj K 16*128 per chip; cluster_axis None (axis 1, axis 0)

    MIMO_V_PAD=1: the padded path as in TtFullAttention (V 192 everywhere, ttnn.transformer SDPA).
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
        self.v_pad = v_pad_enabled()
        self.vw = d if self.v_pad else dv
        # SDPA scale: a power of two, the rest folded into the Q rows. The kernel folds the sink as
        # exp((sink - max) * scale) with the scale truncated to bf16 (compute_streaming.hpp, scale_fp32 >> 16):
        # 192^-0.5 -> 0.07178 (-0.54%), and with the row max ~390 (unscaled) below the sink that shrinks the sink
        # logit by ~0.15 and inflates the output ~14%. A power of two is exact in bf16.
        true_scale = d**-0.5
        self.scale = 2.0 ** math.floor(math.log2(true_scale))
        q_scale = true_scale / self.scale
        self.wq, self.wkv, self.wo = _tp_weights(
            mesh, wqkv, wo, dims, value_scale, dtype, q_scale=q_scale, v_pad=self.v_pad
        )
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
        kv_sink(k, v), if given, also receives this chunk's per-chip K [1, 2, S, 192] and V [1, 2, S, 128] (192,
        zero-padded, under MIMO_V_PAD=1)."""
        seq = x.shape[-2]
        D, VW, nq, nkv = self.d, self.vw, self.nq, self.nkv
        assert start % TILE == 0 and seq % TILE == 0 and start + seq <= min(self.max_seq, cache.max_seq)
        assert cache.vw == VW, f"KV cache V width {cache.vw} != attention's {VW} (MIMO_V_PAD changed?)"

        q, k, v = self._qkv_rope(x, start, seq)

        _sp("kv_write")
        if kv_sink is not None:
            kv_sink(k, v)
        ttnn.fill_cache(cache.k, k, batch_idx=0, update_idx=start)
        ttnn.fill_cache(cache.v, v, batch_idx=0, update_idx=start)

        _sp("kv_tail")
        hist = min(-(-self.window // TILE) * TILE, start)
        if hist:
            k_tail = ttnn.slice(cache.k, [0, 0, start - hist, 0], [1, nkv, start, D])
            v_tail = ttnn.slice(cache.v, [0, 0, start - hist, 0], [1, nkv, start, VW])
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

        _sp("sliding_sdpa")
        full = self._sdpa_ops()[0](
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
            attn = ttnn.slice(full, [0, 0, hist, 0], [1, nq, hist + seq, VW])
            ttnn.deallocate(full)
        else:
            attn = full

        _sp("o_proj")
        a = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, S, 16*128]
        ttnn.deallocate(attn)
        o = ttnn.linear(a, self.wo, compute_kernel_config=_hifi4(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(a)
        _sp("ccl")
        # 2x2: cluster_axis=None reduces over axis 1 then axis 0 (all_reduce.cpp), i.e. all 4 TP ranks.
        out = ttnn.all_reduce(o, cluster_axis=None)
        ttnn.deallocate(o)
        return out

    def free(self):
        super().free()
        if self.sink is not None:
            ttnn.deallocate(self.sink)

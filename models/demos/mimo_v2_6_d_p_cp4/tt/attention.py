# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 full attention (chunked prefill) under context parallelism CP=4 on the 1x4 mesh, TP=1.

64 Q heads / 4 KV heads, QK head dim 192, V head dim 128, partial rotate-half RoPE on dims [0, 64) (theta 1e7), scale
192^-0.5, no sink. Every chip holds the whole qkv_proj and o_proj and computes all heads for its own contiguous S/4
rows of the chunk (chip c: positions start + c*S/4 + j), so there is no CCL after o_proj:

    q     = x @ Wq                     [H, 64*192]
    kv    = x @ Wkv                    [H, 4 * (192 + 192)], rows per KV head [k_h | v_h, zero-padded 128 -> 192];
                                       V rows x attention_value_scale
    q_rot, q_pass = nlp_create_q_heads_split(q, 64, 64)   [1, 64, S/4, 64], [1, 64, S/4, 128]
    k, v  = nlp_create_q_heads_split(kv, 4, 192)          [1, 4, S/4, 192] each
    q,k   = concat(rope(x[..., :64]), x[..., 64:])        cos/sin in chunk-major row order per chip, built once at
                                                          load per chunk size, sliced on device at local row start/4
    cache = update_padded_kv_cache(k / v, kv_actual_global=start, cluster_axis=1)   chunk-major CP ring cache
    attn  = ring_joint_scaled_dot_product_attention(q, cache_k, cache_v; causal, chunked prefill, kv_actual_isl=start,
            logical_n=start+S, Linear topology on axis 1, full-capacity persistent gather buffers, HiFi4 + fp32 dest)
            -> [1, 64, S/4, 192] (V padded)
    out   = concat_heads(attn[..., :128]) @ Wo    o_proj [64*128, H] (the pad dims are sliced off first:
                                       nlp_concat_heads of 64 x 192 overflows L1)

V is padded to 192 because the ring op's tensor-V mode is used with V head dim == QK head dim (known issue "ring_joint
SDPA: V head dim ..."); the state read-back returns V[..., :128].

Chunk-major ring cache (models/demos/gemma4_d_p/tt/attention/ring_prefill.py): per chip [1, 4, max_seq/4, W]; local
row (n*L + j) on chip r holds global position n*C + r*L + j, C the chunk, L = C/4. The layout depends on C, so a cache
is bound to one chunk size (the first chunk run, or the one given when it is built).

Adapted from models/demos/mimo_v2_6_d_p/tt/attention.py (TtFullAttention: weights, value-scale fold, RoPE split,
MIMO_V_PAD path) and gemma4_d_p / gpt_oss_d_p (ring cache, ring SDPA). The forward does no host transfer.
"""

from __future__ import annotations

import torch

import ttnn

from .ccl import RingCCL

TILE = 32
Q_CHUNK, K_CHUNK = 64, 256  # as Gemma-4's global ring layers (q64 optimal there, k256 for non-sliding layers)


def _hifi4():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


def rope_tables(inv_freq: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half cos/sin [len(positions), R], computed in fp64."""
    freqs = positions.double()[:, None] * inv_freq.double()[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().float(), emb.sin().float()


def chunk_major_positions(chip: int, cp: int, chunk: int, rows: int) -> torch.Tensor:
    """Global position of each of chip ``chip``'s first ``rows`` local rows for chunk size ``chunk``."""
    L = chunk // cp
    m = torch.arange(rows, dtype=torch.int64)
    return (m // L) * chunk + chip * L + m % L


def _replicate(mesh, t: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _per_chip(mesh, t: torch.Tensor, dim: int, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
    )


class TtKVCacheRing:
    """K and V for one full-attention layer in the chunk-major CP ring layout, bf16, V zero-padded to the QK head dim.

    Per chip [1, nkv, max_seq/cp, W] (batch = slot * num_layers + layer = 0). The full-capacity ring-gather buffers
    ([1, nkv, max_seq, W], replicated) come from the shared RingCCL and are allocated here, at build time."""

    def __init__(
        self,
        mesh,
        ccl: RingCCL,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        max_seq: int,
        chunk: int | None = None,
        dtype=ttnn.bfloat16,
    ):
        cp = ccl.cp
        assert max_seq % (cp * TILE) == 0, f"max_seq {max_seq} must split into {cp} tile-aligned slices"
        self.mesh, self.ccl, self.cp = mesh, ccl, cp
        self.nkv, self.d, self.dv, self.dtype = num_kv_heads, head_dim, v_head_dim, dtype
        self.max_seq, self.local = max_seq, max_seq // cp
        self.vw = head_dim  # V width on the device (padded)
        self.chunk = None
        self._pending = None
        self.k = self._zeros()
        self.v = self._zeros()
        self.buf_k = ccl.gather_buffer("ring_k", num_kv_heads, max_seq, head_dim, dtype)
        self.buf_v = ccl.gather_buffer("ring_v", num_kv_heads, max_seq, head_dim, dtype)
        if chunk is not None:
            self.bind_chunk(chunk)

    def _zeros(self):
        return ttnn.zeros(
            [1, self.nkv, self.local, self.d],
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def bind_chunk(self, chunk: int) -> None:
        """Fix the chunk size of the layout (and write a prefix loaded before the chunk size was known)."""
        if self.chunk is not None:
            assert self.chunk == chunk, f"ring cache laid out for chunk {self.chunk}, called with chunk {chunk}"
            return
        assert chunk % (self.cp * TILE) == 0 and self.max_seq % chunk == 0, (chunk, self.max_seq)
        self.chunk = chunk
        if self._pending is not None:
            key, value, length = self._pending
            self._pending = None
            self._write_prefix(key, value, length)

    def load_prefix(self, key: torch.Tensor, value: torch.Tensor, length: int) -> None:
        """Host key [nkv, >= length, 192], value [nkv, >= length, 128]: positions [0, length) copied, rest zero.
        Before the chunk size is bound, the prefix is kept on the host and written when it is."""
        if self.chunk is None:
            self._pending = (key, value, length)
            return
        self._write_prefix(key, value, length)

    def _layout(self, t: torch.Tensor, length: int) -> torch.Tensor:
        """Host [nkv, >= length, w] -> [1, nkv, cp * local, W]: chip r's chunk-major slab at rows [r*local, ...)."""
        out = torch.zeros(1, self.nkv, self.cp * self.local, self.d)
        for r in range(self.cp):
            pos = chunk_major_positions(r, self.cp, self.chunk, self.local)
            ok = pos < length
            out[0, :, r * self.local : (r + 1) * self.local][:, ok, : t.shape[-1]] = t[:, pos[ok]].float()
        return out

    def _write_prefix(self, key, value, length):
        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)
        self.k = _per_chip(self.mesh, self._layout(key, length), dim=2, dtype=self.dtype)
        self.v = _per_chip(self.mesh, self._layout(value, length), dim=2, dtype=self.dtype)

    def to_torch(self, length: int) -> dict:
        if self.chunk is None:
            if self._pending is not None:
                key, value, n = self._pending
                assert length <= n
                return {"key": key[:, :length].float(), "value": value[:, :length].float()}
            return {"key": torch.zeros(self.nkv, length, self.d), "value": torch.zeros(self.nkv, length, self.dv)}
        p = torch.arange(length, dtype=torch.int64)
        L = self.chunk // self.cp
        chip, row = (p % self.chunk) // L, (p // self.chunk) * L + p % L

        def host(t, w):
            parts = torch.stack(
                [ttnn.to_torch(x).float()[0] for x in ttnn.get_device_tensors(t)]
            )  # [cp, nkv, local, W]
            return parts[chip, :, row, :w].transpose(0, 1).contiguous()  # [nkv, length, w]

        return {"key": host(self.k, self.d), "value": host(self.v, self.dv)}

    def free(self):
        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)


class TtFullAttention:
    def __init__(
        self,
        mesh,
        ccl: RingCCL,
        wqkv: torch.Tensor,
        wo: torch.Tensor,
        dims: tuple[int, int, int, int],
        inv_freq: torch.Tensor,
        max_seq: int,
        value_scale: float | None,
        chunk_sizes,
        dtype=ttnn.bfloat16,
    ):
        """wqkv: dequantized fused [q; k; v] [Hq*D + Hkv*D + Hkv*Dv, H] in global order (reference/weights.qkv_weight);
        wo: [H, Hq*Dv]; dims: (Hq, Hkv, D, Dv); inv_freq: [R/2]; chunk_sizes: every chunk size (global tokens) the
        module will run, for the chunk-major RoPE tables."""
        hq, hkv, d, dv = dims
        self.mesh, self.ccl, self.cp = mesh, ccl, ccl.cp
        self.nq, self.nkv, self.d, self.dv = hq, hkv, d, dv
        self.rope_dim = 2 * inv_freq.shape[0]
        # fp32-exact scale (192^-0.5 rounded once), as the prior.
        self.scale = float(torch.tensor(d**-0.5, dtype=torch.float32).item())

        H = wqkv.shape[1]
        wq, wk, wv = wqkv.float().split([hq * d, hkv * d, hkv * dv], dim=0)
        if value_scale is not None:
            wv = wv * value_scale
        wv_pad = torch.zeros(hkv, d, H)
        wv_pad[:, :dv] = wv.reshape(hkv, dv, H)
        wkv = torch.cat([wk.reshape(hkv, d, H), wv_pad], dim=1).reshape(hkv * 2 * d, H)  # per head [k_h | v_h pad]
        self.wq = _replicate(mesh, wq.T[None, None], dtype)
        self.wkv = _replicate(mesh, wkv.T[None, None], dtype)
        self.wo = _replicate(mesh, wo.float().T.reshape(1, 1, hq * dv, H), dtype)  # rows head-major, as concat_heads

        self.max_seq = -(-max_seq // TILE) * TILE
        self.tables = {}
        for c in sorted({int(c) for c in chunk_sizes}):
            assert c % (self.cp * TILE) == 0, f"chunk {c} must split into {self.cp} tile-aligned slices"
            L = c // self.cp
            rows = -(-self.max_seq // c) * L
            cs = [rope_tables(inv_freq, chunk_major_positions(r, self.cp, c, rows)) for r in range(self.cp)]
            cos = torch.stack([t[0] for t in cs])[:, None]  # [cp, 1, rows, R]
            sin = torch.stack([t[1] for t in cs])[:, None]
            self.tables[c] = (_per_chip(mesh, cos, dim=0), _per_chip(mesh, sin, dim=0))

        self.program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ccl.sdpa_grid,
            q_chunk_size=Q_CHUNK,
            k_chunk_size=K_CHUNK,
            exp_approx_mode=False,
        )

    def new_cache(self, max_seq: int, chunk: int | None = None) -> TtKVCacheRing:
        return TtKVCacheRing(self.mesh, self.ccl, self.nkv, self.d, self.dv, max_seq, chunk)

    def _partial_rope(self, parts, cos, sin):
        tr, tp = parts
        rot = ttnn.experimental.rotary_embedding(tr, cos, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tr)
        out = ttnn.concat([rot, tp], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rot)
        ttnn.deallocate(tp)
        return out

    def __call__(self, x: ttnn.Tensor, start: int, cache: TtKVCacheRing) -> ttnn.Tensor:
        """x: chip c's CP slice [1, 1, S/4, H] TILE (attn_norm output) of the chunk at [start, start + S). Returns the
        same shape and sharding. Writes this chunk's K/V into the ring cache."""
        L = x.shape[-2]
        C = L * self.cp
        assert start % C == 0 and start + C <= min(self.max_seq, cache.max_seq), (start, C, cache.max_seq)
        assert C in self.tables, f"chunk {C} has no RoPE table (built for {sorted(self.tables)})"
        assert cache.chunk == C, f"ring cache laid out for chunk {cache.chunk}, called with {C} (bind_chunk first)"
        mc = ttnn.DRAM_MEMORY_CONFIG

        qp = ttnn.linear(x, self.wq, compute_kernel_config=_hifi4(), memory_config=mc)
        kvp = ttnn.linear(x, self.wkv, compute_kernel_config=_hifi4(), memory_config=mc)
        q = tuple(
            ttnn.experimental.nlp_create_q_heads_split(
                qp, num_heads=self.nq, split_head_dim=self.rope_dim, memory_config=mc
            )
        )
        k_parts = ttnn.experimental.nlp_create_q_heads_split(
            kvp, num_heads=self.nkv, split_head_dim=self.d, memory_config=mc
        )
        ttnn.deallocate(qp)
        ttnn.deallocate(kvp)
        k, v = k_parts

        r = self.rope_dim
        cos_t, sin_t = self.tables[C]
        row = start // self.cp  # the same local row on every chip (chunk-major)
        cos = ttnn.slice(cos_t, [0, 0, row, 0], [1, 1, row + L, r])
        sin = ttnn.slice(sin_t, [0, 0, row, 0], [1, 1, row + L, r])
        q = self._partial_rope(q, cos, sin)
        k_split = (
            ttnn.slice(k, [0, 0, 0, 0], [1, self.nkv, L, r]),
            ttnn.slice(k, [0, 0, 0, r], [1, self.nkv, L, self.d]),
        )
        ttnn.deallocate(k)
        k = self._partial_rope(k_split, cos, sin)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

        for c_t, t in ((cache.k, k), (cache.v, v)):
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache=c_t,
                input=t,
                slot_idx=0,
                layer_idx=0,
                num_layers=1,
                kv_actual_global=int(start),
                cluster_axis=self.ccl.cp_axis,
            )
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            cache.k,
            cache.v,
            None,
            None,
            None,
            persistent_output_buffer_k=cache.buf_k,
            persistent_output_buffer_v=cache.buf_v,
            joint_strategy="rear",
            logical_n=int(start + C),
            program_config=self.program_config,
            scale=self.scale,
            compute_kernel_config=_hifi4(),
            dim=2,
            multi_device_global_semaphore=self.ccl.ring_semaphores,
            num_links=self.ccl.num_links,
            cluster_axis=self.ccl.cp_axis,
            mesh_device=self.mesh,
            topology=self.ccl.topology,
            ccl_core_grid_offset=self.ccl.ccl_core_grid_offset,
            use_column_major_ccl=True,
            is_causal=True,
            is_balanced=False,
            kv_cache_batch_idx=0,
            # No kv_actual_isl: its KV-pad rotation needs streaming compute (fp32 dest off). Chunk starts are
            # chunk-aligned here, so the plain chunked path (Q offset from logical_n and the shapes) is exact.
        )
        ttnn.deallocate(q)

        # Drop the V pad dims before the head concat: nlp_concat_heads of 64 x 192 overflows L1 (1.68 MB CBs).
        attn_v = ttnn.slice(attn, [0, 0, 0, 0], [1, self.nq, L, self.dv], memory_config=mc)
        ttnn.deallocate(attn)
        a = ttnn.experimental.nlp_concat_heads(attn_v, memory_config=mc)  # [1, 1, S/4, 64*128]
        ttnn.deallocate(attn_v)
        out = ttnn.linear(a, self.wo, compute_kernel_config=_hifi4(), memory_config=mc)
        ttnn.deallocate(a)
        return out

    def free(self):
        for t in (self.wq, self.wkv, self.wo):
            ttnn.deallocate(t)
        for cos, sin in self.tables.values():
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
        self.tables = {}

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

Sliding layers (TtSlidingAttention, below) reuse the projections, RoPE and ring cache, exchange a 128-row halo with
all_gather + mesh_partition and run plain sliding SDPA locally (see its docstring for why not the ring op).

Adapted from models/demos/mimo_v2_6_d_p/tt/attention.py (TtFullAttention: weights, value-scale fold, RoPE split,
MIMO_V_PAD path) and gemma4_d_p / gpt_oss_d_p (ring cache, ring SDPA). The forward does no host transfer.
"""

from __future__ import annotations

import math
import os

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
    """K and V for one attention layer in the chunk-major CP ring layout, V zero-padded to the QK head dim.

    Per chip [1, nkv, max_seq/cp, W] (batch = slot * num_layers + layer = 0). The ring-gather buffers
    ([1, nkv, gather_seq, W], replicated; full layers gather_seq = max_seq, sliding layers the compact halo) come from
    the shared RingCCL and are allocated here, at build time (none when gather_seq is 0: the sliding layers exchange
    their halo themselves). bf16."""

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
        gather_seq: int | None = None,
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
        gs = max_seq if gather_seq is None else gather_seq
        self.buf_k = self.buf_v = None  # gather_seq 0: no ring gather (sliding layers exchange their own halo)
        if gs:
            self.buf_k = ccl.gather_buffer("ring_k", num_kv_heads, gs, head_dim, dtype)
            self.buf_v = ccl.gather_buffer("ring_v", num_kv_heads, gs, head_dim, dtype)
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
        q_scale: float = 1.0,
    ):
        """wqkv: dequantized fused [q; k; v] [Hq*D + Hkv*D + Hkv*Dv, H] in global order (reference/weights.qkv_weight);
        wo: [H, Hq*Dv]; dims: (Hq, Hkv, D, Dv); inv_freq: [R/2]; chunk_sizes: every chunk size (global tokens) the
        module will run, for the chunk-major RoPE tables; q_scale multiplies the Q rows (sliding layers fold the true
        scale over the power-of-two SDPA scale there)."""
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
        if q_scale != 1.0:
            wq = wq * q_scale
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
        q, k, v = self._qkv_rope(x, start, L, C)
        self._write_cache(cache, k, v, start)
        attn = self._attend(q, cache, start, C)
        ttnn.deallocate(q)
        return self._o_proj(attn, L)

    def _qkv_rope(self, x, start: int, L: int, C: int):
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
        return q, k, v

    def _write_cache(self, cache: TtKVCacheRing, k, v, start: int) -> None:
        for c_t, t in ((cache.k, k), (cache.v, v)):
            if t.dtype != c_t.dtype:
                tc = ttnn.typecast(t, c_t.dtype)
                ttnn.deallocate(t)
                t = tc
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache=c_t,
                input=t,
                slot_idx=0,
                layer_idx=0,
                num_layers=1,
                kv_actual_global=int(start),
                cluster_axis=self.ccl.cp_axis,
            )
            ttnn.deallocate(t)

    def _ring_kwargs(self, cache: TtKVCacheRing, start: int, C: int) -> dict:
        return dict(
            persistent_output_buffer_k=cache.buf_k,
            persistent_output_buffer_v=cache.buf_v,
            joint_strategy="rear",
            logical_n=int(start + C),
            program_config=self.program_config,
            scale=self.scale,
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
        )

    def _attend(self, q, cache: TtKVCacheRing, start: int, C: int):
        attn, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            cache.k,
            cache.v,
            None,
            None,
            None,
            compute_kernel_config=_hifi4(),
            # No kv_actual_isl: its KV-pad rotation needs streaming compute (fp32 dest off). Chunk starts are
            # chunk-aligned here, so the plain chunked path (Q offset from logical_n and the shapes) is exact.
            **self._ring_kwargs(cache, start, C),
        )
        return attn

    def _o_proj(self, attn, L: int):
        mc = ttnn.DRAM_MEMORY_CONFIG
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


# ---------------------------------------------------------------------------------------------------------------------
# Sliding-window layers (window 128, per-head sink)
# ---------------------------------------------------------------------------------------------------------------------

# Sliding SDPA presets (env MIMO_SLIDING_SDPA_CFG), as the prior: "S" (default, owner rule) HiFi4, fp32 dest off
# (streaming), exact exp; "base" HiFi4 + fp32 dest. q128/k128.
SLIDING_PRESETS = {"S": False, "base": True}
SLIDING_Q_CHUNK, SLIDING_K_CHUNK = 128, 128


def _sliding_compute_config():
    fp32 = SLIDING_PRESETS[os.environ.get("MIMO_SLIDING_SDPA_CFG", "S")]
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False
    )


class TtSlidingAttention(TtFullAttention):
    """MiMo-V2 sliding-window attention (key j visible to query i iff i - W < j <= i, W = 128) with a per-head sink,
    CP=4 over the sequence, TP=1 (all 64 Q / 8 KV heads and the whole qkv / o_proj on every chip, no CCL after o_proj).
    Chip r holds rows [r*L, (r+1)*L) of the chunk (L = S/4) and needs the T = 128 positions before them (the halo):
    chip r > 0 the tail of chip r-1's slice of this chunk, chip 0 the tail of chip 3's slice of the previous chunk
    (in chip 3's ring cache). The halo is exchanged with one all_gather; the attention is local:

        q, kv, RoPE, cache write   as TtFullAttention (RoPE theta 1e4, chunk-major ring cache, bf16, V padded)
        blk   = concat([k|v][L-T:L], cache[k|v][row-T:row])   [1, 8, 2T, 384] (A_r: this slice's tail, B_r: the
                                                              previous chunk's tail on this chip; row = start/4)
        g     = all_gather(blk, dim 2, axis 1)                [1, 8, 8T, 384] = [A0 B0 A1 B1 A2 B2 A3 B3], every chip
        halo  = mesh_partition(concat(B3, A0, A1, A2), dim 2, axis 1)   chip r gets block r: [1, 8, T, 384]
        sdpa  = SDPA(concat(q[:T], q), concat(halo_k, k), concat(halo_v, v); causal, sliding_window_size=W,
                attention_sink, scale 2^-4) -> rows [T, T + L)  (Q front-padded by T rows, their outputs dropped)
                chunk 0 (no history): chip 0's halo is masked instead (attn_mask built at load, non-causal)
        out   = concat_heads(sdpa[..., :128]) @ Wo

    Why not ring_joint_scaled_dot_product_attention (the plan): on this 1x4 FABRIC_2D mesh its Linear-topology halo
    wrap (chip 3 -> chip 0 over 3 hops) hangs, and with Topology.Ring it runs but scores rel L2 0.032 on the layer-1
    golden against 0.013 for this plain SDPA on the same Q/K/V (its sliding path also needs bfp8 K/V and fp32 dest
    off). Scale: SDPA scale 2^-4 with 192^-0.5 / 2^-4 folded into the Q rows, sink pre-divided by 2^-4 (exact in bf16;
    the kernel folds the sink with a bf16-truncated scale, known issue). Adapted from the prior's TtSlidingAttention."""

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
        window: int,
        sink: torch.Tensor | None,
        dtype=ttnn.bfloat16,
    ):
        hq, hkv, d, dv = dims
        true_scale = d**-0.5
        scale = 2.0 ** math.floor(math.log2(true_scale))
        super().__init__(
            mesh,
            ccl,
            wqkv,
            wo,
            dims,
            inv_freq,
            max_seq,
            value_scale,
            chunk_sizes,
            dtype,
            q_scale=true_scale / scale,
        )
        self.scale = scale
        self.window = int(window)
        self.halo = -(-self.window // TILE) * TILE
        assert self.halo >= self.window - 1
        for c in self.tables:
            assert self.halo <= c // self.cp, f"chunk {c}: slice {c // self.cp} shorter than the halo {self.halo}"
        self.sink = None
        if sink is not None:
            self.sink = _replicate(mesh, (sink.float() / scale).reshape(1, hq, 1, 1))  # [1, 64, 1, 1] bf16
        self.program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
            q_chunk_size=SLIDING_Q_CHUNK,
            k_chunk_size=SLIDING_K_CHUNK,
            exp_approx_mode=False,
        )
        # Chunk-0 masks (chip 0 has no history), per chunk size, shared by every sliding layer.
        self.first_masks = {
            c: ccl.constant(("sliding_first_mask", c, self.window), self._first_mask(c)) for c in self.tables
        }

    def _first_mask(self, chunk: int):
        """Builder of the [1, 1, T + L, T + L] additive mask per chip for chunk 0: the window (in the front-padded
        coordinates) on every chip, and on chip 0 no halo column (its padded rows keep their diagonal so no row is
        empty; those rows are dropped)."""

        def build():
            T, L = self.halo, chunk // self.cp
            n = T + L
            i = torch.arange(n)[:, None]
            j = torch.arange(n)[None, :]
            ok = (j <= i) & (j > i - self.window)
            masks = []
            for r in range(self.cp):
                m = ok & ((j >= T) | (j == i)) if r == 0 else ok
                masks.append(torch.where(m, 0.0, float("-inf")))
            return _per_chip(self.mesh, torch.stack(masks)[:, None], dim=0)

        return build

    def new_cache(self, max_seq: int, chunk: int | None = None) -> TtKVCacheRing:
        return TtKVCacheRing(self.mesh, self.ccl, self.nkv, self.d, self.dv, max_seq, chunk, gather_seq=0)

    def _halo(self, k, v, cache: TtKVCacheRing, start: int):
        """Each chip's predecessor tail [1, nkv, T, 2D] (K | V) from one all_gather (see the class docstring)."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        T, L, D, nkv = self.halo, k.shape[-2], self.d, self.nkv
        row = start // self.cp
        b0 = max(row - T, 0)  # chunk 0: any rows (chip 0 masks its halo, chips 1..3 use A)
        cur = ttnn.concat(
            [ttnn.slice(k, [0, 0, L - T, 0], [1, nkv, L, D]), ttnn.slice(v, [0, 0, L - T, 0], [1, nkv, L, D])],
            dim=3,
            memory_config=mc,
        )
        prev = ttnn.concat(
            [
                ttnn.slice(cache.k, [0, 0, b0, 0], [1, nkv, b0 + T, D]),
                ttnn.slice(cache.v, [0, 0, b0, 0], [1, nkv, b0 + T, D]),
            ],
            dim=3,
            memory_config=mc,
        )
        blk = ttnn.concat([cur, prev], dim=2, memory_config=mc)  # [1, nkv, 2T, 2D] = [A_r; B_r]
        ttnn.deallocate(cur)
        ttnn.deallocate(prev)
        g = ttnn.all_gather(blk, dim=2, cluster_axis=self.ccl.cp_axis, memory_config=mc)  # [A0 B0 A1 B1 A2 B2 A3 B3]
        ttnn.deallocate(blk)
        cp = self.cp
        parts = [ttnn.slice(g, [0, 0, (2 * (cp - 1) + 1) * T, 0], [1, nkv, 2 * cp * T, 2 * D])]  # B_{cp-1}
        parts += [ttnn.slice(g, [0, 0, 2 * c * T, 0], [1, nkv, (2 * c + 1) * T, 2 * D]) for c in range(cp - 1)]
        ttnn.deallocate(g)
        rolled = ttnn.concat(parts, dim=2, memory_config=mc)  # [B3 A0 A1 A2], the same on every chip
        for t in parts:
            ttnn.deallocate(t)
        halo = ttnn.mesh_partition(rolled, dim=2, cluster_axis=self.ccl.cp_axis, memory_config=mc)  # chip r: block r
        ttnn.deallocate(rolled)
        hk = ttnn.slice(halo, [0, 0, 0, 0], [1, nkv, T, D], memory_config=mc)
        hv = ttnn.slice(halo, [0, 0, 0, D], [1, nkv, T, 2 * D], memory_config=mc)
        ttnn.deallocate(halo)
        return hk, hv

    def __call__(self, x: ttnn.Tensor, start: int, cache: TtKVCacheRing) -> ttnn.Tensor:
        """x: chip c's CP slice [1, 1, S/4, H] TILE (attn_norm output) of the chunk at [start, start + S). Returns the
        same shape and sharding. Writes this chunk's K/V into the ring cache."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        L = x.shape[-2]
        C = L * self.cp
        assert start % C == 0 and start + C <= min(self.max_seq, cache.max_seq), (start, C, cache.max_seq)
        assert C in self.tables, f"chunk {C} has no RoPE table (built for {sorted(self.tables)})"
        assert cache.chunk == C, f"ring cache laid out for chunk {cache.chunk}, called with {C} (bind_chunk first)"
        T = self.halo
        q, k, v = self._qkv_rope(x, start, L, C)
        hk, hv = self._halo(k, v, cache, start)  # reads the previous chunk's tail before this chunk is written
        k_cat = ttnn.concat([hk, k], dim=2, memory_config=mc)
        v_cat = ttnn.concat([hv, v], dim=2, memory_config=mc)
        ttnn.deallocate(hk)
        ttnn.deallocate(hv)
        self._write_cache(cache, k, v, start)  # frees k, v
        q_pad = ttnn.slice(q, [0, 0, 0, 0], [1, self.nq, T, self.d])
        q_cat = ttnn.concat([q_pad, q], dim=2, memory_config=mc)
        ttnn.deallocate(q_pad)
        ttnn.deallocate(q)
        first = start == 0
        full = ttnn.transformer.scaled_dot_product_attention(
            q_cat,
            k_cat,
            v_cat,
            is_causal=not first,
            attn_mask=self.first_masks[C] if first else None,
            sliding_window_size=None if first else self.window,
            scale=self.scale,
            attention_sink=self.sink,
            program_config=self.program_config,
            compute_kernel_config=_sliding_compute_config(),
        )
        for t in (q_cat, k_cat, v_cat):
            ttnn.deallocate(t)
        attn = ttnn.slice(full, [0, 0, T, 0], [1, self.nq, T + L, self.d], memory_config=mc)
        ttnn.deallocate(full)
        return self._o_proj(attn, L)

    def free(self):
        super().free()
        if self.sink is not None:
            ttnn.deallocate(self.sink)
            self.sink = None

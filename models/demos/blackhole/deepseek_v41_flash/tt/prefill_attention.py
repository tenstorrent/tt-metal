# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash PREFILL attention (S-wide) that wraps an already built decode attention object.

Reuses the decode weights (wqkv, norms, wq_b / wo_a / wo_b column shards, rope matrix, compressor weights) so prefill costs no
extra weight DRAM, and leaves the decode state (window ring + compressed cache in ``attn.cache``, ratio-2 ``prev_cs``) exactly as
the existing decode loop expects. See docs/superpowers/specs/2026-10-02-dsv41-device-prefill-design.md.

Layout per mesh row: ``U`` users x ``Sp`` (padded prompt length, multiple of 64) tokens, user-major, as ``h [1,1,U*Sp,D]`` bf16;
heads are split over the 8 mesh columns exactly like decode. The padded positions (>= S) compute garbage that nothing real reads
(causal) and that the decode masks hide.
"""

import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attention import (
    HEAD_DIM,
    LOCAL_HEADS,
    Q_LORA,
    WINDOW,
    DSV41CompressedAttention,
)

NEG = -1e9
AR_ROWS = int(os.environ.get("DSV41_PF_AR_ROWS", "128"))
_MASKS = {}


def pad_len(S, mult=64):
    return -(-S // mult) * mult


class DSV41PrefillAttention:
    def __init__(self, attn, attn_sink: torch.Tensor, q_chunk=128, k_chunk=128):
        """attn: built DSV41Attention / DSV41CompressedAttention (decode); attn_sink: [64] host tensor of the layer."""
        self.a = attn
        self.md = attn.mesh_device
        self.U = attn.T
        self.compressed = isinstance(attn, DSV41CompressedAttention)
        self.ratio = attn.ratio if self.compressed else 0
        self.q_chunk, self.k_chunk = q_chunk, k_chunk
        md = self.md
        rows, cols = attn.rows, attn.cols
        # sink for the prefill SDPA: [1, 8 local heads, 1, 1] per column, pre-divided by the softmax scale (the kernel multiplies by it)
        sink = (attn_sink.float() / attn.scale).reshape(1, cols * LOCAL_HEADS, 1, 1)
        self.sink = ttnn.from_torch(
            sink,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 1), mesh_shape=(rows, cols)),
        )
        self.last_lat = (
            None  # [U,1,Sc_pad,512] latents of the last prefill (owner layers; readers use ``source.pre.last_lat``)
        )
        self._tabs = {}
        self.ckc_sdpa = attn.ckc_sdpa
        self.tap = None  # dict -> keeps intermediates (qh, kv, o_raw, c, part) for diagnostics

    # ---- constants ----------------------------------------------------------------------------------------
    def _up(self, t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t.contiguous(),
            device=self.md,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
        )

    def rope_tabs(self, positions: torch.Tensor):
        """Full-width tables for the given positions -> (C, S, -S) device tensors [1,1,n,512]."""
        key = tuple(positions.tolist())
        if key not in self._tabs:
            c, s = self.a._rope_inputs(positions)
            n = positions.shape[0]
            self._tabs[key] = tuple(self._up(t.reshape(1, 1, n, HEAD_DIM)) for t in (c, s, -s))
        return self._tabs[key]

    def mask(self, S, Sp):
        """Additive mask [1,1,Sp,Sp+Scp] over keys [kv | latents] for a compressed layer (cached per (S, Sp, ratio))."""
        key = (S, Sp, self.ratio, id(self.md))
        if key not in _MASKS:
            r = self.ratio
            Scp = Sp // r
            i = torch.arange(Sp).view(-1, 1)
            j = torch.arange(Sp).view(1, -1)
            m = torch.where((j <= i) & (j > i - WINDOW), 0.0, NEG)
            jc = torch.arange(Scp).view(1, -1)
            mc = torch.where((jc < (i + 1) // r) & (jc < S // r), 0.0, NEG)
            full = torch.cat([m, mc], dim=1).reshape(1, 1, Sp, Sp + Scp)
            _MASKS[key] = self._up(full)
        return _MASKS[key]

    # ---- helpers ------------------------------------------------------------------------------------------
    def _rope(self, x, tabs):
        c, s, _ = tabs
        return ttnn.add(
            ttnn.multiply(x, c), ttnn.multiply(ttnn.matmul(x, self.a.Pf, compute_kernel_config=self.a.ckc), s)
        )

    def _rope_inv(self, x, tabs):
        c, _, ns = tabs
        return ttnn.add(
            ttnn.multiply(x, c), ttnn.multiply(ttnn.matmul(x, self.a.Pf, compute_kernel_config=self.a.ckc), ns)
        )

    def _compress(self, h, S, Sp):
        """-> (lat [U,1,Scp,512] bf16 RoPE'd latents, cs [1,1,R,1024] fp32 or None). Only complete groups (j < S // ratio) are valid."""
        a, U, r = self.a, self.U, self.ratio
        R = U * Sp
        Scp = Sp // r
        cs = None
        if r == 1:
            lat = ttnn.rms_norm(ttnn.linear(h, a.c_wkv, compute_kernel_config=a.ckc), weight=a.c_norm, epsilon=a.eps)
        else:
            cs = ttnn.linear(
                h, a.c_wcat, compute_kernel_config=a.ckc, dtype=ttnn.float32
            )  # [1,1,R,1024] = [kv | score]
            pairs = ttnn.to_layout(
                ttnn.reshape(ttnn.to_layout(cs, ttnn.ROW_MAJOR_LAYOUT), [1, 1, R // 2, 2 * 2 * HEAD_DIM]),
                ttnn.TILE_LAYOUT,
            )
            ga = ttnn.slice(pairs, [0, 0, 0, 0], [1, 1, R // 2, 2 * HEAD_DIM])  # first token of each group
            gb = ttnn.slice(pairs, [0, 0, 0, 2 * HEAD_DIM], [1, 1, R // 2, 4 * HEAD_DIM])
            d = ttnn.subtract(ga, gb)
            sl = lambda t, lo: ttnn.slice(t, [0, 0, 0, lo], [1, 1, R // 2, lo + HEAD_DIM])
            pooled = ttnn.addcmul(sl(gb, 0), sl(d, 0), sl(ttnn.sigmoid(d), HEAD_DIM))
            lat = ttnn.rms_norm(ttnn.typecast(pooled, ttnn.bfloat16), weight=a.c_norm, epsilon=a.eps)
        lat = ttnn.reshape(lat, [U, 1, Scp, HEAD_DIM])
        lat = self._rope(lat, self.rope_tabs(torch.arange(Scp) * r))
        return lat, cs

    # ---- forward ------------------------------------------------------------------------------------------
    def forward(self, h, S, write_state=True):
        """h [1,1,U*Sp,D] bf16 tile (normed attention input, user-major) -> [1,1,U*Sp,D] bf16 replicated over columns.
        S = real prompt length (<= Sp). Writes the decode state of this layer when ``write_state``."""
        a, U = self.a, self.U
        R = h.shape[2]
        Sp = R // U
        assert Sp % 64 == 0 and S <= Sp
        tabs = self.rope_tabs(torch.arange(Sp))
        y = ttnn.linear(h, a.wqkv, compute_kernel_config=a.ckc)  # [1,1,R,1792]
        qr = ttnn.rms_norm(ttnn.slice(y, [0, 0, 0, 0], [1, 1, R, Q_LORA]), weight=a.q_norm, epsilon=a.eps)
        kvn = ttnn.rms_norm(
            ttnn.slice(y, [0, 0, 0, Q_LORA], [1, 1, R, Q_LORA + HEAD_DIM]), weight=a.kv_norm, epsilon=a.eps
        )
        ttnn.deallocate(y)
        q = ttnn.linear(qr, a.wq_b, compute_kernel_config=a.ckc)  # [1,1,R,8*512]
        ttnn.deallocate(qr)
        q = ttnn.reshape(q, [U, 1, Sp, LOCAL_HEADS * HEAD_DIM])
        qh, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=LOCAL_HEADS, num_kv_heads=0, transpose_k_heads=False
        )
        ttnn.deallocate(q)
        qh = self._rope(qh, tabs)
        kv = self._rope(ttnn.reshape(kvn, [U, 1, Sp, HEAD_DIM]), tabs)  # [U,1,Sp,512]
        if self.tap is not None:
            self.tap.update(qh=qh, kv=kv)

        kw = {}
        keys = kv
        lat = cs = None
        if self.compressed:
            if a.source is None:
                lat, cs = self._compress(h, S, Sp)
                self.last_lat = lat
            else:
                lat = a.source.prefill.last_lat
            keys = ttnn.concat([kv, lat], dim=2)
            kw = dict(is_causal=False, attn_mask=self.mask(S, Sp))
        else:
            kw = dict(is_causal=True, sliding_window_size=WINDOW)
        o = ttnn.transformer.scaled_dot_product_attention(
            qh,
            keys,
            keys,
            scale=a.scale,
            attention_sink=self.sink,
            compute_kernel_config=self.ckc_sdpa,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.md.compute_with_storage_grid_size(),
                q_chunk_size=self.q_chunk,
                k_chunk_size=self.k_chunk,
                exp_approx_mode=False,
            ),
            **kw,
        )
        if self.tap is None:
            ttnn.deallocate(qh)
        else:
            self.tap["o_raw"] = o
        if self.compressed:
            ttnn.deallocate(keys)
        if write_state:
            self._write_state(kv, lat if (self.compressed and a.source is None) else None, cs, h, S, Sp)
        if self.tap is None:
            ttnn.deallocate(kv)
        o = self._rope_inv(o, tabs)
        c = ttnn.experimental.nlp_concat_heads(o)  # [U,1,Sp,8*512]
        if self.tap is None:
            ttnn.deallocate(o)
        zero = ttnn.zeros(
            [U, 1, Sp, HEAD_DIM], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.md
        )  # decode wo_a has 512 zero rows in front
        c = ttnn.reshape(ttnn.concat([zero, c], dim=3), [1, 1, R, (LOCAL_HEADS + 1) * HEAD_DIM])
        part = ttnn.linear(ttnn.linear(c, a.wo_a, compute_kernel_config=a.ckc), a.wo_b, compute_kernel_config=a.ckc)
        if self.tap is not None:
            self.tap.update(c=c, part=ttnn.clone(part))
        return self._allreduce(part)

    def _allreduce(self, part):
        """All-reduce over the 8 mesh columns in pieces of <= AR_ROWS rows. A single reduce_scatter + all_gather of [1,1,512,5120] bf16
        silently corrupts one tile-row block (sometimes NaN) on this build (tests/test_prefill_attn_debug.py); pieces of 256 rows or
        less are exact."""
        a, R = self.a, part.shape[2]
        if R <= AR_ROWS:
            return a.mesh_config.allreduce(part, a.ccl, axis=1)
        pieces = [
            a.mesh_config.allreduce(
                ttnn.slice(part, [0, 0, i, 0], [1, 1, min(i + AR_ROWS, R), part.shape[3]]), a.ccl, axis=1
            )
            for i in range(0, R, AR_ROWS)
        ]
        ttnn.deallocate(part)
        out = ttnn.concat(pieces, dim=2)
        for p in pieces:
            ttnn.deallocate(p)
        return out

    # ---- decode state -------------------------------------------------------------------------------------
    def _write_state(self, kv, lat, cs, h, S, Sp):
        """Window ring (+ compressed latents, prev_cs) into the decode cache. Supports S <= WINDOW (ring slot p = p) for now."""
        a, U = self.a, self.U
        assert S <= WINDOW, "S > 128 needs the ring rotation (not implemented yet)"
        ring = ttnn.slice(kv, [0, 0, 0, 0], [U, 1, min(Sp, WINDOW), HEAD_DIM])
        if ring.shape[2] < WINDOW:
            ring = ttnn.pad(ring, [(0, 0), (0, 0), (0, WINDOW - ring.shape[2]), (0, 0)], 0.0)
        if not self.compressed:
            full = ring
            if a.cache.shape[2] > WINDOW:
                full = ttnn.pad(ring, [(0, 0), (0, 0), (0, a.cache.shape[2] - WINDOW), (0, 0)], 0.0)
        else:
            src = a if a.source is None else a.source
            comp = src.prefill.last_lat
            Scp = comp.shape[2]
            if Scp >= a.max_comp:
                comp = ttnn.slice(comp, [0, 0, 0, 0], [U, 1, a.max_comp, HEAD_DIM])
            else:
                comp = ttnn.pad(comp, [(0, 0), (0, 0), (0, a.max_comp - Scp), (0, 0)], 0.0)
            full = ttnn.concat([ring, comp], dim=2)
        ttnn.copy(ttnn.typecast(full, a.cache.dtype) if full.dtype != a.cache.dtype else full, a.cache)
        if cs is not None:  # ratio 2: the compressor's "previous token" state = [kv | score] of the last prompt token
            R = cs.shape[2]
            last = ttnn.slice(ttnn.reshape(cs, [U, 1, Sp, 2 * HEAD_DIM]), [0, 0, S - 1, 0], [U, 1, S, 2 * HEAD_DIM])
            last = ttnn.reshape(last, [1, 1, U, 2 * HEAD_DIM])
            if a.prev_cs is None:
                a.prev_cs = ttnn.clone(last)
            else:
                ttnn.copy(last, a.prev_cs)

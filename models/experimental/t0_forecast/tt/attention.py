# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN pre-norm attention sublayers of t0: time (causal, RoPE + xPos) and group (across a series' rows).

reference : t0/torch/model/layers/time_attention.py, group_attention.py, rope.py
    xn = RMSNorm(x) * norm
    time:   q, k, v = heads(xn @ Wqkv + b)                  # rows (three, heads, head_dim)
            q, k = RMSNorm_hd(q) * q_norm, RMSNorm_hd(k) * k_norm     # QK-norm BEFORE RoPE
            q = q * cos_q + rot(q) * sin_q; k likewise       # interleaved pairs, xPos folded into the tables
            ctx = SDPA(q, k, v), causal (or the uploaded mask), scale 1 / sqrt(head_dim)
    group:  one row per series: x + xn @ (Wo Wv) + (Wo bv + bo)    # softmax over one key is 1; one matmul
            else SDPA over a series' G * S_pad tokens as they lie, (B, H, G * S_pad, hd), with a same-slot mask
            (the read-out cut, S not tile-aligned: SDPA over the G rows at every patch, (S_pad * B, H, G, hd))
    return x + ctx @ Wo + bo
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from models.experimental.t0_forecast.tt.common import (
    TtT0Precision,
    linear_minimal,
    linear_residual,
    rms_norm,
    upload,
    upload_bias,
    upload_linear,
    upload_ones,
    upload_norm,
)
from models.experimental.t0_forecast.tt.host import pair_rotation
from models.experimental.t0_forecast.tt.weights import TIME, AttentionWeights

_MAX_SDPA_CHUNK = 256
# SDPA with a user mask over 768 keys was wrong at chunk 256 (PCC 0.985) and right at 128 (R2-1 micro, jobs 582, 583).
_MAX_MASKED_CHUNK = 128
# QKV on minimal_matmul: blocks M 16 / K 8 / N 4, subblock 4 x 1 (tests/perf/sweep_matmul.py at M = 98304,
# doc/optimize/iter_01/matmul_sweep_qkv*.json: 4.24 ms against 12.67 ms for the auto 1D config).
QKV_BLOCKS = (16, 8, 4, 4, 1)
WO_BLOCKS = (16, 8, 4, 4, 1)


@dataclass(frozen=True)
class TtAttentionInputs:
    """Device tensors shared by every attention sublayer of one forward."""

    rows: int  # R
    padded_seq: int  # S_pad, the patches of a row padded to a whole tile
    group_size: int  # G
    rope: tuple  # (cos_q, sin_q, cos_k, sin_k), each (1, 1, S_pad, hd); pad positions rotate by identity
    time_mask: object | None  # (R, 1, S_pad, S_pad) additive; None = causal
    group_mask: object | None  # (S_pad * R / G, 1, G, G) additive; None = every row of a series visible
    packed_group_mask: object | None = None  # (1 or R / G, 1, G * S_pad, G * S_pad) additive: same slot (+ group_mask)


def sdpa_chunk(padded_len: int, limit: int = _MAX_SDPA_CHUNK) -> int:
    """Largest tile multiple up to ``limit`` dividing the padded length."""
    return max(c for c in range(32, min(padded_len, limit) + 1, 32) if padded_len % c == 0)


class TtAttention:
    def __init__(self, device, weights: AttentionWeights, precision: TtT0Precision):
        self.device = device
        self.precision = precision
        self.kind = weights.kind
        self.num_heads = weights.num_heads
        self.head_dim = weights.head_dim
        self.d = weights.num_heads * weights.head_dim
        self.eps = 1e-8
        self._norm = upload_norm(device, weights.norm)
        self._wqkv = upload_linear(device, weights.wqkv, precision.weight_dtype("qkv"))
        self._bqkv = upload_bias(device, weights.bqkv)
        if weights.kind != TIME:
            # One row per series: softmax over one key is 1, so the sublayer is x + Wo (Wv xn + bv) + bo, one matmul
            # with W = Wo Wv and b = Wo bv + bo (fp32 at load, one bf16 rounding instead of two).
            wv, bv = weights.part(2)
            self._wvo = upload_linear(device, weights.wo @ wv, precision.weight_dtype("wo"))
            self._bvo = upload_bias(device, weights.wo @ bv + weights.bo)
        self._q_norm = upload_norm(device, weights.q_norm)
        self._k_norm = upload_norm(device, weights.k_norm)
        self._wo = upload_linear(device, weights.wo, precision.weight_dtype("wo"))
        self._bo = upload_bias(device, weights.bo)
        self._ones = upload_ones(device, self.d)
        if weights.kind == TIME:
            self._trans = upload(device, pair_rotation(32).reshape(1, 1, 32, 32))

    def __call__(self, x, inputs: TtAttentionInputs):
        """Flat ``(1, 1, R * S_pad, d)`` -> ``x + attention``, same shape; ``x`` is borrowed."""
        import ttnn

        xn = rms_norm(x, self._norm, self.eps, self.precision)
        if self.kind != TIME and inputs.group_size == 1:
            w, b, ctx = self._wvo, self._bvo, xn
        else:
            w, b = self._wo, self._bo
            ctx = self._time(xn, inputs) if self.kind == TIME else self._group(xn, inputs)
            ttnn.deallocate(xn)
        y = linear_residual(ctx, w, x, self._ones, self.precision, role="wo", blocks=WO_BLOCKS, bias=b)
        ttnn.deallocate(ctx)
        return y

    def _heads(self, qkv):
        """``(batch, 1, seq, 3d)`` (tile-aligned seq) or ``(batch, seq, 3d)`` -> q, k, v ``(batch, H, seq, hd)``
        with q and k RMS-normed per head."""
        import ttnn

        if len(qkv.shape) == 4:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=self.num_heads, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        else:
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                qkv, num_heads=self.num_heads, transpose_key=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        ttnn.deallocate(qkv)
        qn = rms_norm(q, self._q_norm, self.eps, self.precision)
        kn = rms_norm(k, self._k_norm, self.eps, self.precision)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        return qn, kn, v

    def _rope(self, t, cos, sin):
        import ttnn

        out = ttnn.experimental.rotary_embedding_llama(
            t,
            cos,
            sin,
            self._trans,
            is_decode_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.precision.exact_kernel_config(),
        )
        ttnn.deallocate(t)
        return out

    def _sdpa(self, q, k, v, *, mask, is_causal: bool, concat_heads: bool = False, chunk_limit=_MAX_SDPA_CHUNK):
        import ttnn

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=sdpa_chunk(q.padded_shape[-2], chunk_limit),
            k_chunk_size=sdpa_chunk(k.padded_shape[-2], chunk_limit),
            exp_approx_mode=False,
        )
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=is_causal,
            scale=1.0 / math.sqrt(self.head_dim),
            program_config=program_config,
            compute_kernel_config=self.precision.compute_kernel_config(fp32_acc=True),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_concat_heads=concat_heads,
        )
        for t in (q, k, v):
            ttnn.deallocate(t)
        return ctx

    def _time(self, xn, inputs: TtAttentionInputs):
        import ttnn

        r, s = inputs.rows, inputs.padded_seq
        qkv = linear_minimal(xn, self._wqkv, self.precision, role="qkv", blocks=QKV_BLOCKS, bias=self._bqkv)
        q, k, v = self._heads(ttnn.reshape(qkv, (r, 1, s, 3 * self.d)))
        cos_q, sin_q, cos_k, sin_k = inputs.rope
        q, k = self._rope(q, cos_q, sin_q), self._rope(k, cos_k, sin_k)
        ctx = self._sdpa(q, k, v, mask=inputs.time_mask, is_causal=inputs.time_mask is None, concat_heads=True)
        return ttnn.reshape(ctx, (1, 1, r * s, self.d))

    def _group(self, xn, inputs: TtAttentionInputs):
        """Attention across the G rows of each series at every patch slot."""
        import ttnn

        r, s, g, d = inputs.rows, inputs.padded_seq, inputs.group_size, self.d
        qkv = linear_minimal(xn, self._wqkv, self.precision, role="qkv", blocks=QKV_BLOCKS, bias=self._bqkv)
        if inputs.packed_group_mask is not None:
            # A series' rows are contiguous: (B, 1, G * S_pad, 3d) is a free view, no permutes, no G -> 32 padding.
            q, k, v = self._heads(ttnn.reshape(qkv, (r // g, 1, g * s, 3 * d)))
            ctx = self._sdpa(
                q,
                k,
                v,
                mask=inputs.packed_group_mask,
                is_causal=False,
                concat_heads=True,
                chunk_limit=_MAX_MASKED_CHUNK,
            )
            return ttnn.reshape(ctx, (1, 1, r * s, d))
        qkv = ttnn.reshape(qkv, (r, s, 3 * d))
        slot_major = ttnn.permute(qkv, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qkv)
        blocks = ttnn.reshape(slot_major, (s * r // g, g, 3 * d))
        if g % 32:
            # The reshape leaves stale rows in the tile padding; zero them so no key reads garbage.
            blocks = ttnn.fill_implicit_tile_padding(blocks, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = self._heads(blocks)
        merged = ttnn.reshape(
            self._sdpa(q, k, v, mask=inputs.group_mask, is_causal=False, concat_heads=True), (s, r, d)
        )
        row_major = ttnn.permute(merged, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(merged)
        return ttnn.reshape(row_major, (1, 1, r * s, d))

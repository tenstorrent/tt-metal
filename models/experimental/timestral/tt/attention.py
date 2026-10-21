# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN gated attention sublayer (time or variate axis), pre-norm, no residual.

reference : models/experimental/timestral/reference/src/timestral/layers.py (``Attention``)
    xn = RMSNorm(x)                                   # gain folded into Wqkv and Wg
    q, k, v = heads(xn @ Wqkv)                        # rows reordered to (qkv, heads, head_dim)
    time:    q, k = RoPE(q, k); ctx = SDPA(q, k, v)   # causal, or the uploaded mask
    variate: one row per series -> ctx = v exactly (softmax over one key)
             else SDPA over each series' rows, (S * B, H, V, Dh)
    return (merge(ctx) * sigmoid(xn @ Wg)) @ Wo
"""

from __future__ import annotations

from dataclasses import dataclass

from models.experimental.timestral.tt.common import TILE, TtTimestralPrecision, linear, rms_norm, upload_linear
from models.experimental.timestral.tt.weights import AttentionWeights

_MAX_SDPA_CHUNK = 256


@dataclass(frozen=True)
class TtAttentionInputs:
    """Device tensors shared by every attention sublayer of one forward."""

    cos: object  # (1 or R, 1, S_pad, Dh)
    sin: object
    time_mask: object | None  # (R, 1, S_pad, S_pad) additive; None = causal
    variate_mask: object | None  # (S_pad * R / V, 1, V, V) additive; None = all rows of a series visible
    rows: int  # R
    padded_seq: int  # S_pad
    group_size: int  # V
    shared_positions: bool  # cos/sin broadcast over rows


def sdpa_chunk(padded_len: int) -> int:
    """Largest tile multiple up to 256 that divides the padded sequence."""
    return max(c for c in range(32, min(padded_len, _MAX_SDPA_CHUNK) + 1, 32) if padded_len % c == 0)


class TtAttention:
    def __init__(self, device, weights: AttentionWeights, precision: TtTimestralPrecision):
        self.device = device
        self.precision = precision
        self.num_heads = weights.num_heads
        self.head_dim = weights.head_dim
        self.d_model = weights.num_heads * weights.head_dim
        self.is_time = weights.is_time
        self.eps = weights.eps
        dtype = precision.weight_dtype()
        self._wqkv = upload_linear(device, weights.wqkv, dtype)
        self._wv = None if weights.is_time else upload_linear(device, weights.wv, dtype)
        self._wg = upload_linear(device, weights.wg, dtype)
        self._wo = upload_linear(device, weights.wo, dtype)

    def __call__(self, x, inputs: TtAttentionInputs):
        """Flat device ``(1, 1, R * S_pad, D)`` -> same shape; ``x`` is borrowed."""
        import ttnn

        p = self.precision
        xn = rms_norm(x, self.eps, p)
        gate = linear(xn, self._wg, p, activation="sigmoid")
        if self.is_time:
            merged = self._time_context(xn, inputs)
        elif inputs.group_size == 1:
            merged = linear(xn, self._wv, p)
        else:
            merged = self._variate_context(xn, inputs)
        ttnn.deallocate(xn)
        gated = ttnn.mul(merged, gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(merged)
        ttnn.deallocate(gate)
        out = linear(gated, self._wo, p)
        ttnn.deallocate(gated)
        return out

    def _split_heads(self, qkv):
        import ttnn

        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self.num_heads, transpose_key=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)
        return q, k, v

    def _sdpa(self, q, k, v, *, mask, is_causal: bool):
        import ttnn

        chunk = sdpa_chunk(q.padded_shape[-2])
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        )
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=is_causal,
            scale=self.head_dim**-0.5,
            program_config=program_config,
            compute_kernel_config=self.precision.compute_kernel_config(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        merged = ttnn.transformer.concatenate_heads(ctx, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ctx)
        return merged

    def _rope(self, t, inputs: TtAttentionInputs):
        """``t * cos + rotate_half(t) * sin`` on ``(R, H, S_pad, Dh)``; consumes ``t``."""
        import ttnn

        if inputs.shared_positions:
            out = ttnn.experimental.rotary_embedding(t, inputs.cos, inputs.sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(t)
            return out
        b, h, s, dh = t.shape
        half = dh // 2
        x1 = ttnn.slice(t, (0, 0, 0, 0), (b, h, s, half))
        x2 = ttnn.slice(t, (0, 0, 0, half), (b, h, s, dh))
        rotated = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        out = ttnn.add(ttnn.mul(t, inputs.cos), ttnn.mul(rotated, inputs.sin), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(t)
        ttnn.deallocate(rotated)
        return out

    def _time_context(self, xn, inputs: TtAttentionInputs):
        import ttnn

        r, s, d = inputs.rows, inputs.padded_seq, self.d_model
        qkv = ttnn.reshape(linear(xn, self._wqkv, self.precision), (r, s, 3 * d))
        q, k, v = self._split_heads(qkv)
        q, k = self._rope(q, inputs), self._rope(k, inputs)
        merged = self._sdpa(q, k, v, mask=inputs.time_mask, is_causal=inputs.time_mask is None)
        return ttnn.reshape(merged, (1, 1, r * s, d))

    def _variate_context(self, xn, inputs: TtAttentionInputs):
        """Attention across the V rows of each series at every time slot: ``(S_pad * B, H, V, Dh)``."""
        import ttnn

        r, s, d, g = inputs.rows, inputs.padded_seq, self.d_model, inputs.group_size
        qkv = ttnn.reshape(linear(xn, self._wqkv, self.precision), (r, s, 3 * d))
        slot_major = ttnn.permute(qkv, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qkv)
        blocks = ttnn.reshape(slot_major, (s * r // g, g, 3 * d))
        if inputs.variate_mask is not None and g % TILE:
            # The reshape leaves stale rows in the tile padding; with a user mask SDPA reads those key rows.
            blocks = ttnn.fill_implicit_tile_padding(blocks, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = self._split_heads(blocks)
        merged = self._sdpa(q, k, v, mask=inputs.variate_mask, is_causal=False)
        merged = ttnn.reshape(merged, (s, r, d))
        row_major = ttnn.permute(merged, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(merged)
        return ttnn.reshape(row_major, (1, 1, r * s, d))

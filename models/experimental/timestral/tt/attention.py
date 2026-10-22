# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN gated attention sublayer (time or variate axis), pre-norm.

reference : models/experimental/timestral/reference/src/timestral/layers.py (``Attention``)
    xn = RMSNorm(x)                                   # gain folded into Wqkv and Wg
    q, k, v = heads(xn @ Wqkv)                        # rows reordered to (qkv, heads, head_dim)
    time:    q, k = RoPE(q, k); ctx = SDPA(q, k, v)   # causal, or the uploaded mask
             q, k, v and xn @ Wg come out of one [Wq | Wk | Wv | Wg] matmul as four flat tensors;
             one fused op per tensor splits the heads head-major (1, H, R * S_pad, Dh) and applies
             RoPE to q, k (pair order), and SDPA runs on the (H, R, S_pad, Dh) view of that layout
    variate: one row per series -> ctx = v exactly (softmax over one key), [Wv | Wg] in one matmul
             else SDPA over each series' rows, (S * B, H, V, Dh)
    return (merge(ctx) * sigmoid(xn @ Wg)) @ Wo       # the sigmoid inside the MUL where Wg was packed

With ``SplitStreams`` the context and decoder tokens are two flat streams. Context queries attend
causally on (H, R, L_pad, Dh); the decoder queries of ``rows_per_block`` rows fill one tile and attend
over those rows' context keys followed by their decoder keys, a block mask keeping each on its row.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.timestral.tt.common import (
    TILE,
    TtTimestralPrecision,
    linear,
    linear_residual,
    linear_split,
    rms_norm,
    take_slots,
    upload,
    upload_linear,
)
from models.experimental.timestral.tt.weights import AttentionWeights

_MAX_SDPA_CHUNK = 256


@dataclass(frozen=True)
class TtAttentionInputs:
    """Device tensors shared by every attention sublayer of one forward."""

    cos: object  # (1, 1, R * S_pad, Dh), one row per token, head_dim in rope_pair_order
    sin: object
    unit_stats: object  # (1, 1, R * S_pad, TILE) row sums of x^2 that make the head-split norm scale exactly 1
    time_mask: object | None  # (1, R, S_pad, S_pad) additive; None = causal
    variate_mask: object | None  # (S_pad * R / V, 1, V, V) additive; None = all rows of a series visible
    rows: int  # R
    padded_seq: int  # S_pad
    group_size: int  # V


@dataclass(frozen=True)
class DecoderSlots:
    """Token slots ``[start, start + count)`` of every row, the only ones the head reads, and the inputs
    of the sublayers that run on those slots alone."""

    start: int
    count: int
    inputs: TtAttentionInputs  # padded_seq = count; no RoPE tables or time mask

    def take(self, x, inputs: TtAttentionInputs):
        """Flat ``(1, 1, R * S_pad, D)`` laid out by ``inputs`` -> ``(1, 1, R * count, D)``; ``x`` is borrowed."""
        return take_slots(x, inputs.rows, inputs.padded_seq, self.start, self.count)


@dataclass(frozen=True)
class SplitStreams:
    """Inputs of the split layout: context tokens ``(1, 1, R * L_pad, D)`` and decoder tokens ``(1, 1, R * h, D)``."""

    context: TtAttentionInputs  # padded_seq = L_pad
    decoder: TtAttentionInputs  # padded_seq = h
    cross_mask: object  # (1, 1 or R / b, b * h, b * (L_pad + h)) additive, b = rows_per_block
    rows_per_block: int  # b = TILE / h


def pair_rotation(n: int) -> torch.Tensor:
    """``(n, n)`` matrix with ``t @ R`` mapping each ``rope_pair_order`` pair ``(a_i, b_i)`` to ``(-b_i, a_i)``."""
    rotation = torch.zeros(n, n)
    rotation[torch.arange(0, n, 2), torch.arange(1, n, 2)] = 1.0
    rotation[torch.arange(1, n, 2), torch.arange(0, n, 2)] = -1.0
    return rotation


def sdpa_chunk(padded_len: int) -> int:
    """Largest tile multiple up to 256 that divides the padded sequence."""
    return max(c for c in range(32, min(padded_len, _MAX_SDPA_CHUNK) + 1, 32) if padded_len % c == 0)


class TtAttention:
    def __init__(self, device, weights: AttentionWeights, precision: TtTimestralPrecision, *, kv_only: bool = False):
        """``kv_only``: also keep ``[Wk | Wv]`` for ``split_time(context_out=False)``."""
        self.device = device
        self.precision = precision
        self.num_heads = weights.num_heads
        self.head_dim = weights.head_dim
        self.d_model = weights.num_heads * weights.head_dim
        self.is_time = weights.is_time
        self.eps = weights.eps
        dtype = precision.weight_dtype()
        d = self.d_model
        # The gate's pre-activation rides along as the last chunk of the split projections.
        if weights.is_time:
            self._wqkvg = upload_linear(device, torch.cat([weights.wqkv, weights.wg]), dtype)
            self._wkv = upload_linear(device, weights.wqkv[d:], dtype) if kv_only else None
            self._rope_trans = upload(device, pair_rotation(TILE).reshape(1, 1, TILE, TILE))
        else:
            self._wqkv = upload_linear(device, weights.wqkv, dtype)
            self._wvg = upload_linear(device, torch.cat([weights.wv, weights.wg]), dtype)
            self._wg = upload_linear(device, weights.wg, dtype)
        self._wo = upload_linear(device, weights.wo, precision.weight_dtype(residual=True))
        self._ones = upload(device, torch.ones(1, d))

    def __call__(self, x, inputs: TtAttentionInputs, residual=None, decoder: DecoderSlots | None = None):
        """Flat device ``(1, 1, R * S_pad, D)`` -> same shape, plus ``residual`` (added in the wo
        matmul) when given; ``x`` and ``residual`` are borrowed. With ``decoder`` the gate, wo and
        residual run on the decoder slots only and the output is ``(1, 1, R * count, D)``."""
        import ttnn

        xn = rms_norm(x, self.eps, self.precision)
        merged, gate_logits = self._context(xn, inputs)
        if gate_logits is not None:
            ttnn.deallocate(xn)
            xn = None
        if decoder is None:
            return self._output(merged, xn, residual, gate_logits)
        # Nothing after the context mixes token slots.
        full = [t for t in (merged, xn, gate_logits) if t is not None]
        merged, xn, gate_logits = (None if t is None else decoder.take(t, inputs) for t in (merged, xn, gate_logits))
        for t in full:
            ttnn.deallocate(t)
        if residual is None:
            return self._output(merged, xn, None, gate_logits)
        residual = decoder.take(residual, inputs)
        out = self._output(merged, xn, residual, gate_logits)
        ttnn.deallocate(residual)
        return out

    def split_time(self, x_c, x_d, streams: SplitStreams, *, context_out: bool = True):
        """Time attention plus residual on both streams: ``(x_c, x_d)`` -> ``(y_c, y_d)``; borrows both.
        Without ``context_out`` the context stream only supplies keys and values and ``y_c`` is None."""
        import ttnn

        p, ctx, dec = self.precision, streams.context, streams.decoder
        xn_c = rms_norm(x_c, self.eps, p)
        if context_out:
            q_c, k_c, v_c, g_c = linear_split(xn_c, self._wqkvg, p, 4)
        else:
            k_c, v_c = linear_split(xn_c, self._wkv, p, 2)
        ttnn.deallocate(xn_c)
        k_c, v_c = self._heads(k_c, ctx, rope=True), self._heads(v_c, ctx, rope=False)
        xn_d = rms_norm(x_d, self.eps, p)
        q_d, k_d, v_d, g_d = linear_split(xn_d, self._wqkvg, p, 4)
        ttnn.deallocate(xn_d)
        q_d, k_d, v_d = (
            self._heads(q_d, dec, rope=True),
            self._heads(k_d, dec, rope=True),
            self._heads(v_d, dec, rope=False),
        )
        y_d = self._output(self._decoder_attention(q_d, k_c, v_c, k_d, v_d, streams), None, x_d, g_d)
        if not context_out:
            ttnn.deallocate(k_c)
            ttnn.deallocate(v_c)
            return None, y_d
        merged_c = self._time_sdpa(self._heads(q_c, ctx, rope=True), k_c, v_c, ctx)
        return self._output(merged_c, None, x_c, g_c), y_d

    def _output(self, merged, xn, residual, gate_logits=None):
        """``(merged * sigmoid(gate_logits)) @ Wo``, plus ``residual`` when given; without ``gate_logits``
        they are ``xn @ Wg``. Consumes ``merged``, ``xn`` and ``gate_logits``."""
        import ttnn

        p = self.precision
        if gate_logits is None:
            gate = linear(xn, self._wg, p, activation="sigmoid")
            ttnn.deallocate(xn)
            gated = ttnn.mul(merged, gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(gate)
        else:
            gated = ttnn.mul(
                merged,
                gate_logits,
                input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(gate_logits)
        ttnn.deallocate(merged)
        if residual is None:
            out = linear(gated, self._wo, p)
        else:
            out = linear_residual(gated, self._wo, p, residual, self._ones)
        ttnn.deallocate(gated)
        return out

    def _context(self, xn, inputs: TtAttentionInputs):
        """``(merged, gate_logits)``; ``gate_logits`` is None where the gate is computed from ``xn``."""
        if self.is_time:
            return self._time_context(xn, inputs)
        if inputs.group_size == 1:
            return tuple(linear_split(xn, self._wvg, self.precision, 2))
        return self._variate_context(xn, inputs), None

    def _split_heads(self, qkv):
        import ttnn

        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self.num_heads, transpose_key=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)
        return q, k, v

    def _sdpa(self, q, k, v, *, mask, is_causal: bool, concat_heads: bool = True):
        import ttnn

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=sdpa_chunk(q.padded_shape[-2]),
            k_chunk_size=sdpa_chunk(k.padded_shape[-2]),
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
            output_concat_heads=concat_heads,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        return ctx

    def _heads(self, t, inputs: TtAttentionInputs, *, rope: bool):
        """Flat ``(1, 1, R * S_pad, D)`` -> head-major ``(1, H, R * S_pad, Dh)``, rotated when ``rope``;
        consumes ``t``. The op is an RMSNorm; ``unit_stats`` make its scale exactly 1."""
        import ttnn

        rope_args = {}
        if rope:
            rope_args = dict(transformation_mat=self._rope_trans, rope_cos=inputs.cos, rope_sin=inputs.sin)
        out = ttnn.experimental.wan_fused_rmsnorm_post_allgather(
            t,
            inputs.unit_stats,
            epsilon=0.0,
            num_heads_per_device=self.num_heads,
            # Below HiFi4 the +-1 rotation matmul drops input mantissa bits.
            compute_kernel_config=self.precision.compute_kernel_config(math_fidelity=ttnn.MathFidelity.HiFi4),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **rope_args,
        )
        ttnn.deallocate(t)
        return out

    def _time_context(self, xn, inputs: TtAttentionInputs):
        q, k, v, gate_logits = linear_split(xn, self._wqkvg, self.precision, 4)
        merged = self._time_sdpa(
            self._heads(q, inputs, rope=True),
            self._heads(k, inputs, rope=True),
            self._heads(v, inputs, rope=False),
            inputs,
        )
        return merged, gate_logits

    def _time_sdpa(self, q, k, v, inputs: TtAttentionInputs):
        """Head-major ``(1, H, R * S_pad, Dh)`` q, k, v -> merged ``(1, 1, R * S_pad, D)``; consumes them."""
        import ttnn

        r, s, h, dh = inputs.rows, inputs.padded_seq, self.num_heads, self.head_dim
        q, k, v = (ttnn.reshape(t, (h, r, s, dh)) for t in (q, k, v))
        ctx = self._sdpa(q, k, v, mask=inputs.time_mask, is_causal=inputs.time_mask is None, concat_heads=False)
        return self._concat_heads(ctx, r * s)

    def _decoder_attention(self, q_d, k_c, v_c, k_d, v_d, streams: SplitStreams):
        """Head-major decoder q, k, v and context k, v -> merged decoder ``(1, 1, R * h, D)``; consumes the
        decoder tensors and borrows the context ones."""
        import ttnn

        h, dh, b = self.num_heads, self.head_dim, streams.rows_per_block
        blocks, lc, ld = streams.context.rows // b, streams.context.padded_seq, streams.decoder.padded_seq

        def block_keys(context, decoder):
            keys = ttnn.concat(
                [ttnn.reshape(context, (h, blocks, b * lc, dh)), ttnn.reshape(decoder, (h, blocks, b * ld, dh))],
                dim=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(decoder)
            return keys

        k, v = block_keys(k_c, k_d), block_keys(v_c, v_d)
        q = ttnn.reshape(q_d, (h, blocks, b * ld, dh))
        ctx = self._sdpa(q, k, v, mask=streams.cross_mask, is_causal=False, concat_heads=False)
        return self._concat_heads(ctx, streams.decoder.rows * ld)

    def _concat_heads(self, ctx, tokens: int):
        """SDPA output viewed ``(1, H, tokens, Dh)`` -> ``(1, 1, tokens, D)``; consumes ``ctx``."""
        import ttnn

        h, dh = self.num_heads, self.head_dim
        merged = ttnn.experimental.nlp_concat_heads(
            ttnn.reshape(ctx, (1, h, tokens, dh)), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(ctx)
        return merged

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

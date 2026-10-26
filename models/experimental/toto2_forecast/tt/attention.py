# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN Toto 2.0 attention sublayers (pre-norm, tau residual folded to ``beta * x + f``).

reference : toto2/toto2/model.py @ 9213d2c (``SelfAttention``, ``SelfAttentionTransformerLayer``), folded by
``tt/weights.py``
    time:    xn = RMSNorm(x); q, k, v = heads(xn Wqkv + b)          (R, H, S_pad, 64), PerDimScale in the q rows
             q, k = RoPE + xPos with separate q / k tables (rotary_embedding_llama, identity on dims 32-63)
             a = SDPA(q, k, v, scale 1/64, causal or the uploaded group mask)
    variate: V = 1 -> a = xn Wv + bv (softmax over one key);  else SDPA across the V variates of each token slot
    x <- beta * x + a Wo + bo
"""

from __future__ import annotations

from dataclasses import dataclass

from models.experimental.toto2_forecast.tt.common import (
    TILE,
    Toto2Precision,
    pad_to_tile,
    linear,
    rms_norm,
    scaled_residual,
    upload_bias,
    upload_linear,
)
from models.experimental.toto2_forecast.tt.weights import LayerWeights, Toto2Config

_MAX_SDPA_CHUNK = 256
# SDPA with a user mask is wrong at chunk 256 (PLAN section 9 F9, D2 condition 2): cap masked calls at 128.
_MAX_MASKED_SDPA_CHUNK = 128


def sdpa_chunk(padded_len: int, masked: bool = False) -> int:
    """Largest tile multiple up to 256 (128 with a user mask) that divides the padded sequence."""
    cap = _MAX_MASKED_SDPA_CHUNK if masked else _MAX_SDPA_CHUNK
    return max(c for c in range(32, min(padded_len, cap) + 1, 32) if padded_len % c == 0)


@dataclass(frozen=True)
class TimeInputs:
    """Device tensors shared by the time layers of one forward."""

    rows: int  # R = B * V
    padded_seq: int  # S_pad, tokens per row on device
    cos_q: object  # (1, 1, S_pad, 64)
    sin_q: object
    cos_k: object
    sin_k: object
    trans_mat: object  # (1, 1, 32, 32)
    mask: object | None  # (R, 1, S_pad, S_pad) additive; None = causal


@dataclass(frozen=True)
class VariateInputs:
    batch: int  # B
    variates: int  # V
    slots: int  # token slots per row at the variate layer
    mask: object | None  # (B * slots, 1, V_pad, V_pad) additive, pad keys -inf; None = all variates visible


class TtTimeAttention:
    def __init__(self, device, lw: LayerWeights, config: Toto2Config, precision: Toto2Precision):
        self.device = device
        self.precision = precision
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.eps = config.norm_eps
        self.beta = lw.beta_attn
        dtype = precision.weight_dtype()
        self._w_qkv = upload_linear(device, lw.w_qkv, dtype)
        self._b_qkv = upload_bias(device, lw.b_qkv)
        self._w_o = upload_linear(device, lw.w_o, precision.weight_dtype("out"))
        self._b_o = upload_bias(device, lw.b_o)

    def __call__(self, x, inputs: TimeInputs):
        """Flat ``(1, 1, R * S_pad, d)`` -> same shape; consumes ``x``."""
        import ttnn

        p = self.precision
        r, sp = inputs.rows, inputs.padded_seq
        xn = rms_norm(x, self.eps, p)
        qkv = linear(xn, self._w_qkv, p, bias=self._b_qkv)
        ttnn.deallocate(xn)
        qkv = ttnn.reshape(qkv, (r, 1, sp, qkv.shape[-1]))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        q = self._rope(q, inputs.cos_q, inputs.sin_q, inputs.trans_mat)
        k = self._rope(k, inputs.cos_k, inputs.sin_k, inputs.trans_mat)
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=inputs.mask,
            is_causal=inputs.mask is None,
            scale=1.0 / self.head_dim,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=sdpa_chunk(sp, inputs.mask is not None),
                k_chunk_size=sdpa_chunk(sp, inputs.mask is not None),
                exp_approx_mode=False,
            ),
            compute_kernel_config=p.sdpa_config(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for t in (q, k, v):
            ttnn.deallocate(t)
        merged = ttnn.experimental.nlp_concat_heads(ctx, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ctx)
        merged = ttnn.reshape(merged, (1, 1, r * sp, merged.shape[-1]))
        out = linear(merged, self._w_o, p, bias=self._b_o)
        ttnn.deallocate(merged)
        y = scaled_residual(out, x, self.beta)
        ttnn.deallocate(x)
        return y

    def _rope(self, t, cos, sin, trans_mat):
        import ttnn

        out = ttnn.experimental.rotary_embedding_llama(
            t, cos, sin, trans_mat, compute_kernel_config=self.precision.rope_config()
        )
        ttnn.deallocate(t)
        return out


class TtVariateAttention:
    def __init__(self, device, lw: LayerWeights, config: Toto2Config, precision: Toto2Precision):
        self.device = device
        self.precision = precision
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.eps = config.norm_eps
        self.beta = lw.beta_attn
        d = config.d_model
        dtype = precision.weight_dtype()
        self._w_v = upload_linear(device, lw.w_qkv[2 * d :], dtype)
        self._b_v = upload_bias(device, lw.b_qkv[2 * d :])
        self._w_qkv = upload_linear(device, lw.w_qkv, dtype)
        self._b_qkv = upload_bias(device, lw.b_qkv)
        self._w_o = upload_linear(device, lw.w_o, precision.weight_dtype("out"))
        self._b_o = upload_bias(device, lw.b_o)

    def __call__(self, x, inputs: VariateInputs):
        """Flat ``(1, 1, B * V * slots, d)`` (rows item-major, then variate, then slot) -> same; consumes ``x``."""
        import ttnn

        p = self.precision
        xn = rms_norm(x, self.eps, p)
        if inputs.variates == 1:
            merged = linear(xn, self._w_v, p, bias=self._b_v)
        else:
            merged = self._across_variates(xn, inputs)
        ttnn.deallocate(xn)
        out = linear(merged, self._w_o, p, bias=self._b_o)
        ttnn.deallocate(merged)
        y = scaled_residual(out, x, self.beta)
        ttnn.deallocate(x)
        return y

    def _across_variates(self, xn, inputs: VariateInputs):
        """SDPA over the V variates of each (item, token slot). With a series-id mask the variate axis is padded
        to a whole tile with zero rows and the mask covers all 32 keys (pad keys -inf): SDPA with a user mask
        does not mask the tile padding of a key axis that is not tile-aligned (PLAN section 9, F1). Without a
        mask SDPA pads the logical V keys itself."""
        import ttnn

        p = self.precision
        b, v, n = inputs.batch, inputs.variates, inputs.slots
        qkv = linear(xn, self._w_qkv, p, bias=self._b_qkv)
        width = qkv.shape[-1]
        slot_major = ttnn.permute(
            ttnn.reshape(qkv, (b, v, n, width)), (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)
        blocks = ttnn.reshape(slot_major, (b * n, 1, v, width))
        padded = inputs.mask is not None and v % TILE != 0
        if padded:
            zeros = ttnn.zeros(
                (b * n, 1, pad_to_tile(v) - v, width),
                dtype=blocks.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            whole = ttnn.concat([blocks, zeros], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(zeros)
            ttnn.deallocate(slot_major)
            blocks = whole
        q, k, vv = ttnn.experimental.nlp_create_qkv_heads(
            blocks,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(blocks)
        # No program config: the default SDPA chunk is 32 here, under the masked cap of 128 (F9, D3 defect 6).
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            vv,
            attn_mask=inputs.mask,
            is_causal=False,
            scale=1.0 / self.head_dim,
            compute_kernel_config=p.sdpa_config(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for t in (q, k, vv):
            ttnn.deallocate(t)
        merged = ttnn.experimental.nlp_concat_heads(ctx, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ctx)
        d = merged.shape[-1]
        if padded:
            real = ttnn.slice(merged, (0, 0, 0, 0), (b * n, 1, v, d), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(merged)
            merged = real
        row_major = ttnn.permute(ttnn.reshape(merged, (b, n, v, d)), (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(merged)
        return ttnn.reshape(row_major, (1, 1, b * v * n, d))

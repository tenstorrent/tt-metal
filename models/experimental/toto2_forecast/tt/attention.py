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

import os
from dataclasses import dataclass

import torch

from models.experimental.toto2_forecast.tt.common import (
    TILE,
    Toto2Precision,
    pad_to_tile,
    linear,
    rms_norm,
    scaled_residual,
    upload,
    upload_bias,
    upload_linear,
)
from models.experimental.toto2_forecast.tt.weights import LayerWeights, Toto2Config

# Round 1 screening toggle (R1-6): SDPA writes the heads side by side, no nlp_concat_heads.
_SDPA_CONCAT_HEADS = os.environ.get("TOTO2_SDPA_CONCAT_HEADS", "1") == "1"
# R1-4: QKV, out_proj and (2.5B-FT) FFN down on minimal_matmul (TOTO2_MINIMAL_MATMUL=0: ttnn.linear, for A/B). Blocks (M, K, N, subblock_h, subblock_w) in
# tiles by d_model, the fastest of job 770's sweep at M = 98304 (accuracy): 2.5B-FT QKV 44.32 -> 12.03 ms, out_proj
# 7.22 -> 4.70 ms; 313m QKV 12.77 -> 4.27 ms, out_proj 2.41 -> 1.62 ms per call; 2.5B-FT down (K = 171 tiles)
# 15.82 -> 9.19 ms (job 781). 313m down has no block that divides its K tiles and stays on ttnn.linear.
_MINIMAL_MATMUL = os.environ.get("TOTO2_MINIMAL_MATMUL", "1") == "1"
_MM_BLOCKS = {
    2048: {"qkv": (8, 32, 2, 2, 2), "out": (8, 16, 4, 2, 2)},
    1024: {"qkv": (8, 32, 2, 2, 2), "out": (8, 32, 2, 2, 2)},
}
_MM_DOWN_BLOCKS = {2048: (8, 9, 4, 1, 4)}


def mm_config(device, d_model: int, role: str, k: int | None = None):
    """The R1-4 ``MinimalMatmulConfig`` of a role (``qkv`` / ``out`` / ``down``), None when off or when no block
    fits; K_block is cut to a divisor of the K tiles (``k``, default d_model) for qkv / out."""
    import ttnn

    if not _MINIMAL_MATMUL:
        return None
    k_tiles = (k or d_model) // TILE
    if role == "down":
        blocks = _MM_DOWN_BLOCKS.get(d_model)
        if blocks is None or k_tiles % blocks[1]:
            return None
    else:
        blocks = _MM_BLOCKS.get(d_model, _MM_BLOCKS[1024])[role]
    mb, kb, nb, sh, sw = blocks
    kb = max(b for b in range(1, min(kb, k_tiles) + 1) if k_tiles % b == 0)
    grid = device.compute_with_storage_grid_size()
    return ttnn.MinimalMatmulConfig(
        M_block_size=mb,
        K_block_size=kb,
        N_block_size=nb,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
    )


def epilogue_config(config, k_tiles: int):
    """``config`` with K_block cut to the largest divisor of ``k_tiles`` up to half (env
    ``TOTO2_EPILOGUE_KDIV``, default 2): the fused residual's extra circular buffers overflow L1 at the R1-4
    blocks (1594752 > 1572864 B on 313m out_proj)."""
    import ttnn

    if config is None:
        return None
    cap = max(1, config.K_block_size // int(os.environ.get("TOTO2_EPILOGUE_KDIV", "2")))
    return ttnn.MinimalMatmulConfig(
        M_block_size=config.M_block_size,
        K_block_size=max(b for b in range(1, cap + 1) if k_tiles % b == 0),
        N_block_size=config.N_block_size,
        subblock_h=config.subblock_h,
        subblock_w=config.subblock_w,
        compute_with_storage_grid_size=config.compute_with_storage_grid_size,
    )


def mm_linear(x, weight, precision: Toto2Precision, config, *, bias=None):
    """``x @ weight (+ bias)`` through ``minimal_matmul`` (``config`` None: ``ttnn.linear``)."""
    import ttnn

    if config is None:
        return linear(x, weight, precision, bias=bias)
    return ttnn.experimental.minimal_matmul(
        x,
        weight,
        bias_tensor=bias,
        config=config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=precision.matmul_config(),
    )


# R2-2 screening switch: residual add in the minimal_matmul epilogue (needs folded betas, R2-1).
_EPILOGUE_RESIDUAL = os.environ.get("TOTO2_EPILOGUE_RESIDUAL", "1") == "1"


def mm_linear_residual(x, weight, precision: Toto2Precision, configs, residual, beta: float, ones, *, bias=None):
    """``beta * residual + x @ weight (+ bias)``; consumes neither input. With ``beta == 1``, a ``minimal_matmul``
    config and matching dtypes the add runs in the matmul epilogue (``dit_minimal_matmul_addcmul_fused`` with a
    row of ones); otherwise a separate add. ``configs``: (plain, epilogue) ``MinimalMatmulConfig``."""
    import ttnn

    config, config_ep = configs

    if (
        _EPILOGUE_RESIDUAL
        and beta == 1.0
        and config is not None
        and ones is not None
        and x.dtype == weight.dtype == residual.dtype
    ):
        return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            x,
            weight,
            1.0,
            residual,
            ones,
            bias,
            config=config_ep,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=precision.matmul_config(),
        )
    return scaled_residual(mm_linear(x, weight, precision, config, bias=bias), residual, beta)


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
        self.eps = config.norm_eps if lw.eps_attn is None else lw.eps_attn
        self.beta = lw.beta_attn
        dtype = precision.weight_dtype()
        self._w_qkv = upload_linear(device, lw.w_qkv, dtype)
        self._b_qkv = upload_bias(device, lw.b_qkv)
        self._w_o = upload_linear(device, lw.w_o, precision.weight_dtype("out"))
        self._b_o = upload_bias(device, lw.b_o)
        d = config.d_model
        self._qkv_config = mm_config(device, d, "qkv")
        self._o_config = mm_config(device, d, "out")
        self._ones = upload(device, torch.ones(1, d)) if self.beta == 1.0 else None
        self._o_config_ep = epilogue_config(self._o_config, d // TILE)

    def __call__(self, x, inputs: TimeInputs):
        """Flat ``(1, 1, R * S_pad, d)`` -> same shape; consumes ``x``."""
        import ttnn

        p = self.precision
        r, sp = inputs.rows, inputs.padded_seq
        xn = rms_norm(x, self.eps, p)
        qkv = mm_linear(xn, self._w_qkv, p, self._qkv_config, bias=self._b_qkv)
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
            output_concat_heads=_SDPA_CONCAT_HEADS,
        )
        for t in (q, k, v):
            ttnn.deallocate(t)
        if _SDPA_CONCAT_HEADS:
            merged = ctx
        else:
            merged = ttnn.experimental.nlp_concat_heads(ctx, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(ctx)
        merged = ttnn.reshape(merged, (1, 1, r * sp, merged.shape[-1]))
        y = mm_linear_residual(merged, self._w_o, p, (self._o_config, self._o_config_ep), x, self.beta, self._ones, bias=self._b_o)
        ttnn.deallocate(merged)
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
        self.eps = config.norm_eps if lw.eps_attn is None else lw.eps_attn
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
        row_major = ttnn.permute(
            ttnn.reshape(merged, (b, n, v, d)), (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(merged)
        return ttnn.reshape(row_major, (1, 1, b * v * n, d))

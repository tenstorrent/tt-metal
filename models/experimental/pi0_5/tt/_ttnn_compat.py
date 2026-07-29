# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fallbacks for pi0.5 fused TTNN ops when running against a main-branch build."""

from __future__ import annotations


import ttnn


def nlp_create_qkv_heads_rope(
    xqkv,
    cos,
    sin,
    num_heads: int,
    num_kv_heads: int,
    *,
    memory_config=None,
):
    mem = memory_config or ttnn.L1_MEMORY_CONFIG
    # The fused op is now tile-aware (it validates Ht == 1 against the operand's own tile height,
    # sizes every CB from its tensor's tile, preserves the input page config on its outputs, and
    # hashes the tile dims), so it is used at ANY tile height. One dispatch instead of three.
    if hasattr(ttnn.experimental, "nlp_create_qkv_heads_rope"):
        return ttnn.experimental.nlp_create_qkv_heads_rope(xqkv, cos, sin, num_heads, num_kv_heads, memory_config=mem)
    # Fallback: nlp_create_qkv_heads requires a sharded (non-width-sharded) output when its input is
    # sharded. The DECODE_ALL path feeds a WIDTH-SHARDED qkv; convert it to interleaved first so the
    # interleaved-in -> interleaved-out path is taken and the requested `mem` config is honored.
    if xqkv.is_sharded():
        xqkv = ttnn.sharded_to_interleaved(xqkv, mem)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        xqkv,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        memory_config=mem,
    )
    q = ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=mem)
    k = ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=mem)
    return q, k, v


def concat_heads_matmul(attn_out, weight, *, memory_config=None, program_config=None, dtype=None):
    mem = memory_config or ttnn.L1_MEMORY_CONFIG
    out_dtype = dtype or ttnn.bfloat16
    if hasattr(ttnn.experimental, "concat_heads_matmul"):
        kwargs = {"memory_config": mem}
        if program_config is not None:
            kwargs["program_config"] = program_config
        return ttnn.experimental.concat_heads_matmul(attn_out, weight, **kwargs)
    heads = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=mem)
    kwargs = {"dtype": out_dtype, "memory_config": mem}
    if program_config is not None:
        kwargs["program_config"] = program_config
    out = ttnn.linear(heads, weight, **kwargs)
    ttnn.deallocate(heads)
    return out


def concat_heads_matmul_decode(
    attn_out,
    weight,
    *,
    output_dtype=None,
    compute_kernel_config=None,
    reshard_cores=None,
    residual=None,
    gate=None,
):
    if hasattr(ttnn.experimental, "concat_heads_matmul_decode"):
        return ttnn.experimental.concat_heads_matmul_decode(
            attn_out,
            weight,
            output_dtype=output_dtype,
            compute_kernel_config=compute_kernel_config,
            reshard_cores=reshard_cores,
            residual=residual,
            gate=gate,
        )
    out = concat_heads_matmul(
        attn_out,
        weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=output_dtype or ttnn.bfloat16,
    )
    if residual is not None and gate is not None:
        out = ttnn.addcmul(residual, gate, out, memory_config=ttnn.L1_MEMORY_CONFIG)
    return out


def kv_sdpa(
    q,
    k,
    v,
    *,
    attn_mask=None,
    scale=None,
    past_k=None,
    past_v=None,
    compute_kernel_config=None,
    max_kv_chunk_tiles=None,
    kv_splits=None,
    prefix_valid_tiles=None,
):
    # The tiny-tile kv_sdpa dropped the mask path our earlier version had (cb_mask_in + a
    # use_provided_mask compile-time gate) and now hard-fails:
    #   TT_FATAL: kv_sdpa FlashFused two-source path does not support an attention mask
    # Callers that genuinely need a mask (the DRAM ExpertChunkSlice expert in ttnn_gemma, which
    # feeds the 28-chip and pipeline_1x8_v2 paths) fall back to the general SDPA below rather than
    # silently dropping the mask, which would change numerics wherever the mask is not all-zero
    # (e.g. the keep_padded phantom -1e4 band). Restoring the fused mask path on top of the
    # tiny-tile two-phase chunking is stage 8; see TINY_TILE_INTEGRATION_PLAN.md.
    if hasattr(ttnn, "kv_sdpa") and attn_mask is None:
        kwargs = {"attn_mask": attn_mask, "scale": scale}
        if past_k is not None:
            kwargs["past_k"] = past_k
            kwargs["past_v"] = past_v
        if compute_kernel_config is not None:
            kwargs["compute_kernel_config"] = compute_kernel_config
        if max_kv_chunk_tiles is not None:
            kwargs["max_kv_chunk_tiles"] = max_kv_chunk_tiles
        if kv_splits is not None:
            kwargs["kv_splits"] = kv_splits
        if prefix_valid_tiles:
            kwargs["prefix_valid_tiles"] = list(prefix_valid_tiles)
        return ttnn.kv_sdpa(q, k, v, **kwargs)

    if past_k is not None and past_v is not None:
        k = ttnn.concat([past_k, k], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        v = ttnn.concat([past_v, v], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)

    kwargs = {
        "attn_mask": attn_mask,
        "is_causal": False,
        "scale": scale,
        "memory_config": ttnn.L1_MEMORY_CONFIG,
    }
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config
    return ttnn.transformer.scaled_dot_product_attention(q, k, v, **kwargs)


def decode_all_supported() -> bool:
    """True when the ops the decode_all denoise path actually dispatches are present.

    The tiny-tile block projects through ``ttnn.experimental.matmul_decode`` (tiny-tile inputA,
    32x32 inputB). The older fused path used ``ttnn.matmul_decode`` + ``ttnn.gate_up_matmul_decode``;
    either is sufficient, so accept a build that exposes one family or the other.
    """
    return hasattr(ttnn.experimental, "matmul_decode") or (
        hasattr(ttnn, "matmul_decode") and hasattr(ttnn, "gate_up_matmul_decode")
    )

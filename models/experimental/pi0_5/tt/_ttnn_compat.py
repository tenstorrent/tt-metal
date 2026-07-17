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
    if hasattr(ttnn.experimental, "nlp_create_qkv_heads_rope"):
        return ttnn.experimental.nlp_create_qkv_heads_rope(xqkv, cos, sin, num_heads, num_kv_heads, memory_config=mem)
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


def kv_sdpa(q, k, v, *, attn_mask=None, scale=None, past_k=None, past_v=None, compute_kernel_config=None):
    if hasattr(ttnn, "kv_sdpa"):
        kwargs = {"attn_mask": attn_mask, "scale": scale}
        if past_k is not None:
            kwargs["past_k"] = past_k
            kwargs["past_v"] = past_v
        if compute_kernel_config is not None:
            kwargs["compute_kernel_config"] = compute_kernel_config
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
    return hasattr(ttnn, "matmul_decode") and hasattr(ttnn, "gate_up_matmul_decode")

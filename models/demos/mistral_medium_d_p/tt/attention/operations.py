# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 attention primitive ops.

No projection bias, no QK-norm, and FULL rotary (rotary_dim == head_dim == 128), so there is no
partial-rotary slice/concat.

The block closes with a **reduce-scatter, not an all-reduce** — see :func:`apply_reduce_scatter`.
"""

import ttnn

from .weights import AttentionWeights


def apply_qkv_projection(hidden_states, weights: AttentionWeights):
    """Fused QKV projection. No bias (Mistral projections are bias-free).

    Args:
        hidden_states: [1, 1, seq_len, hidden_size]
    Returns:
        Fused QKV [1, 1, seq_len, local_qkv_dim]
    """
    return ttnn.linear(hidden_states, weights.wqkv, dtype=ttnn.bfloat16)


def split_qkv_heads_prefill(xqkv_fused, num_heads: int, num_kv_heads: int):
    """Split fused QKV into Q/K/V head tensors (GQA).

    Returns (Q, K, V): [1, num_heads, seq_len, head_dim] / [1, num_kv_heads, seq_len, head_dim].
    """
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def apply_rope(
    tensor, rope_mats, transformation_mat, is_decode_mode: bool = False, kv_actual_global=None, cluster_axis=None
):
    """FULL rotary RoPE. The YaRN scaling (theta 1e6, factor 64, orig_max_pos 4096, beta 4/1) and the
    attention_factor (1.4158883) are baked into ``rope_mats`` at build time (tt/rope_tables.py), so
    this is a plain full rotation of the whole head.

    Two inner ops:
      * default (``kv_actual_global`` is None): ``rotary_embedding_llama`` with per-chunk cos/sin.
      * indexed (``kv_actual_global`` set): ``rotary_embedding_indexed`` — ``rope_mats`` carry the
        WHOLE-cache, block-cyclic, SP-sharded cos/sin built once, and the op derives this chunk's
        per-chip start row on-device. No per-chunk host reshard.
    """
    if kv_actual_global is not None:
        return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            tensor,
            rope_mats[0],
            rope_mats[1],
            transformation_mat,
            kv_actual_global=kv_actual_global,
            cluster_axis=cluster_axis,
        )
    return ttnn.experimental.rotary_embedding_llama(
        tensor, rope_mats[0], rope_mats[1], transformation_mat, is_decode_mode=is_decode_mode
    )


def concat_heads(tensor):
    """[1, n_heads, seq_len, head_dim] -> [1, 1, seq_len, n_heads*head_dim]."""
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights: AttentionWeights, activation_dtype):
    """Row-parallel output projection. No bias.

    Each chip contracts only its own Q-head slice, so the result is a PARTIAL sum over the hidden
    dim; :func:`apply_allreduce` completes it.
    """
    tensor = ttnn.typecast(tensor, ttnn.bfloat8_b)
    out = ttnn.matmul(tensor, weights.o_proj, dtype=activation_dtype)
    tensor.deallocate(True)
    return out


def apply_reduce_scatter(tensor, mesh_config, ccl_manager, hidden_size: int):
    """Close the block: complete the row-parallel partial sum AND land it as ``emb/tp``.

    ``o_proj`` is row-parallel, so each chip holds only its own Q-head slice of the contraction and
    emits a partial sum over the FULL hidden dim. A reduce-scatter both finishes that sum and
    scatters it to ``hidden_size / tp`` per chip — exactly the sharded residual's layout, so the
    caller adds it straight into the residual with no further communication.

    Deliberately NOT an all-reduce: the all-gather back to full emb belongs in front of the next
    norm (the decoder layer's job). Doing it here would move the same bytes twice.

    At ``tp == 1`` this is the identity, so the single-chip tests exercise the same path.

    Args:
        tensor: ``[1, 1, s_local, hidden_size]`` partial sum from o_proj
    Returns:
        ``[1, 1, s_local, hidden_size // tp]`` (or the input unchanged at tp == 1)
    """
    if mesh_config.tp == 1:
        return tensor

    # If weights.py padded o_proj's output for tile alignment, trim before scattering so the
    # per-chip shard is exactly hidden_size/tp. No-op for Mistral (12288/4 = 3072, tile-aligned).
    local_hidden = hidden_size // mesh_config.tp
    if ((local_hidden + 31) // 32) * 32 != local_hidden:
        shape = tensor.shape
        trimmed = ttnn.slice(
            tensor, starts=[0, 0, 0, 0], ends=[shape[0], shape[1], shape[2], hidden_size], steps=[1, 1, 1, 1]
        )
        tensor.deallocate(True)
        tensor = trimmed

    out = mesh_config.reduce_scatter(tensor, ccl_manager, dim=3, axis=mesh_config.tp_axis)
    tensor.deallocate(True)
    return out

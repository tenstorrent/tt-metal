# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B attention primitive ops.

Copied from ``gpt_oss_d_p/tt/attention/operations.py`` with the bias arguments removed (Llama's
projections are unbiased) and the fused matmul+reduce-scatter path dropped — it is gated off on
Blackhole in the donor anyway, and perf tuning is out of bring-up scope.

RoPE is FULL rotary (rotary_dim == head_dim == 128), so there is no partial-rotary slice/concat.
"""

import ttnn

from .weights import AttentionWeights


def apply_qkv_projection(hidden_states, weights: AttentionWeights):
    """[1, 1, S, hidden] -> fused [1, 1, S, local_q + local_k + local_v]. No bias."""
    return ttnn.linear(hidden_states, weights.wqkv, dtype=ttnn.bfloat16)


def split_qkv_heads_prefill(xqkv_fused, num_heads: int, num_kv_heads: int):
    """Split fused QKV into per-head tensors (GQA: ``num_heads`` Q, ``num_kv_heads`` K/V).

    Returns Q ``[1, num_heads, S, head_dim]`` and K/V ``[1, num_kv_heads, S, head_dim]``, where both
    counts are the LOCAL (per-TP-chip) counts: 8 Q heads and 2 KV heads at TP=4.
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
    """Apply full rotary RoPE. llama3 scaling is baked into ``rope_mats`` at build time (tt/rope.py).

    Two inner ops:
      * default (``kv_actual_global`` is None): ``rotary_embedding_llama`` with cos/sin already
        sliced to this chunk's positions — the single-shot / unit-test path.
      * indexed (``kv_actual_global`` set): ``rotary_embedding_indexed`` — ``rope_mats`` carry the
        WHOLE-cache, block-cyclic, SP-sharded cos/sin built once, and the op derives this chunk's
        per-chip start row on-device from ``kv_actual_global`` + the device's ``cluster_axis``
        coordinate (the same block-cyclic math the KV-cache writer uses). No host reshard.
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
    """[1, n_heads, S, head_dim] -> [1, 1, S, n_heads*head_dim] (the chip's local hidden slice)."""
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights: AttentionWeights, activation_dtype):
    """Row-parallel o_proj. No bias. Output is a per-chip PARTIAL sum; the caller must all-reduce."""
    tensor = ttnn.typecast(tensor, ttnn.bfloat8_b)
    out = ttnn.matmul(tensor, weights.o_proj, dtype=activation_dtype)
    tensor.deallocate(True)
    return out


def apply_allreduce(tensor, mesh_config, ccl_manager, hidden_size: int):
    """TP all-reduce over the row-parallel o_proj partials (no-op at TP == 1).

    Llama needs no padding trim here: ``hidden_size/tp`` is tile-aligned, so ``weights.py`` adds no
    alignment columns (see the assert there).
    """
    if mesh_config.tp > 1:
        # mesh_config.allreduce frees its input internally between reduce_scatter and all_gather;
        # do not deallocate again here.
        tensor = mesh_config.allreduce(tensor, ccl_manager, pad_size=0, axis=mesh_config.tp_axis)
    return tensor

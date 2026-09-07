# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-Medium-3.5 attention primitive ops.

Adapted from ``gpt_oss_d_p/tt/attention/operations.py``. RoPE is FULL rotary (rotary_dim ==
head_dim == 128, no partial-rotary slice/concat), there is no QK-norm, and — unlike the donor — the
QKV and O projections carry NO bias, so both are plain matmuls.

The donor's fused matmul + reduce-scatter o_proj path (``minimal_matmul_strided_reduce_scatter_async``)
is NOT carried over: it is gated off on Blackhole in the donor itself (a semaphore/overlap race
producing non-deterministic garbage) and the spec targets ``bh_galaxy``, so the only path it could
ever take here is the one that is already known-broken. Dropping it removes a per-M_tiles tuned
config table that would otherwise be dead code. The plain o_proj + all-reduce path is what runs.
"""

import ttnn

from .weights import AttentionWeights


def apply_qkv_projection(hidden_states, weights: AttentionWeights, dtype=ttnn.bfloat16, compute_kernel_config=None):
    """
    Apply the fused QKV projection (no bias — Ministral3 q/k/v are ``bias=False``).

    Args:
        hidden_states: Input tensor [1, 1, seq_len, hidden_size]
        weights: Attention weights container
        dtype: Output dtype
        compute_kernel_config: see ``utils.general_utils.get_matmul_compute_config`` — required in
            practice, because the default (bf16 destination accumulation) costs ~0.008 PCC on a
            12288-deep contraction

    Returns:
        Fused QKV tensor [1, 1, seq_len, local_q_dim + local_k_dim + local_v_dim]
    """
    return ttnn.linear(hidden_states, weights.wqkv, dtype=dtype, compute_kernel_config=compute_kernel_config)


def split_qkv_heads_prefill(xqkv_fused, num_heads: int, num_kv_heads: int):
    """
    Split fused QKV into separate head tensors for prefill (GQA: num_heads Q, num_kv_heads K/V).

    Args:
        xqkv_fused: Fused QKV tensor
        num_heads: Number of LOCAL Q heads (per TP shard)
        num_kv_heads: Number of LOCAL K/V heads (per TP shard)

    Returns:
        Tuple (Q, K, V): [1, num_heads, seq_len, head_dim] / [1, num_kv_heads, seq_len, head_dim]
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
    """
    Apply rotary position embedding — FULL rotary for Mistral (rotary_dim == head_dim == 128).

    The YaRN scaling (theta 1e6, factor 64, orig_max_pos 4096, truncated correction dims) is baked
    into the cos/sin at build time (see ``tt/rope.py``), so this op is a plain full rotation.

    Two inner ops:
      * default (``kv_actual_global`` is None): ``rotary_embedding_llama`` with a per-chunk cos/sin
        already sliced to this chunk's positions (the non-cache / single-shot path);
      * indexed (``kv_actual_global`` set): ``rotary_embedding_indexed`` — ``rope_mats`` carry the
        WHOLE-cache, block-cyclic-reordered, SP-sharded cos/sin (built once by ``tt/rope.py``), and
        the op derives this chunk's per-chip start row on-device from ``kv_actual_global`` + the
        device's ``cluster_axis`` coordinate (the same block-cyclic math the KV writer uses).

    Args:
        tensor: Input tensor (Q or K), [1, n_heads, seq_len, head_dim]
        rope_mats: (cos, sin), last dim = head_dim
        transformation_mat: RoPE transformation matrix
        is_decode_mode: False for prefill
        kv_actual_global: prior valid global KV length (tile-aligned). Set -> indexed on-device RoPE.
        cluster_axis: SP mesh axis the whole-cache cos/sin are sharded along (required when indexed).
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
    """
    Concatenate attention heads back to the (local) hidden dimension.

    Args:
        tensor: Attention output with separate heads [1, n_heads, seq_len, head_dim]

    Returns:
        [1, 1, seq_len, n_heads * head_dim]
    """
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights: AttentionWeights, activation_dtype, compute_kernel_config=None):
    """
    Apply the output projection (no bias — Ministral3 o_proj is ``bias=False``).

    The donor casts the SDPA output to bf8 before this matmul to halve its DRAM traffic. That is
    dropped here: measured, it is worth ~0.00002 PCC at seq 512 (0.999867 vs 0.999890), so the
    saving is real and the accuracy cost is negligible — but the bring-up default is the accurate
    one, and a perf pass can put the cast back with a number to justify it.

    Args:
        tensor: Attention output [1, 1, seq_len, local_hidden]
        weights: Attention weights container
        activation_dtype: Target dtype for the output
        compute_kernel_config: see ``utils.general_utils.get_matmul_compute_config``

    Returns:
        Row-parallel partial sum, to be all-reduced across TP by the caller
    """
    return ttnn.matmul(tensor, weights.o_proj, dtype=activation_dtype, compute_kernel_config=compute_kernel_config)


def apply_allreduce(tensor, mesh_config, ccl_manager, hidden_size: int):
    """
    Apply the tensor-parallel all-reduce (TP > 1), then strip the o_proj tile-alignment padding.

    o_proj is row-parallel, so each TP device holds a partial sum over its slice of the contraction
    dim; the all-reduce is what makes the result correct, not an optimization.

    Args:
        tensor: Row-parallel partial sums
        mesh_config: Mesh configuration
        ccl_manager: Communication manager
        hidden_size: Hidden size (for padding removal)
    """
    if mesh_config.tp <= 1:
        return tensor

    # `mesh_config.allreduce` frees its input internally between reduce_scatter and all_gather;
    # don't deallocate the original handle again here.
    tensor = mesh_config.allreduce(tensor, ccl_manager, pad_size=0, axis=mesh_config.tp_axis)

    local_hidden = hidden_size // mesh_config.tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32
    if padded_local_hidden != local_hidden:
        shape = tensor.shape
        tensor_sliced = ttnn.slice(
            tensor,
            starts=[0, 0, 0, 0],
            ends=[shape[0], shape[1], shape[2], hidden_size],
            steps=[1, 1, 1, 1],
        )
        tensor.deallocate(True)
        tensor = tensor_sliced
    return tensor

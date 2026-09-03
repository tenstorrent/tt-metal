# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B attention primitive ops: projections, GQA head split, RoPE, TP-collective tail.

Template: ``models/demos/gpt_oss_d_p/tt/attention/operations.py:14`` (``apply_qkv_projection``),
``:29`` (``split_qkv_heads_prefill``), ``:41`` (``nlp_create_qkv_heads``), ``:50`` (``apply_rope``),
``:87`` (``rotary_embedding_llama``), ``:79`` (the indexed variant), ``:92`` (``concat_heads``),
``:102`` (``nlp_concat_heads``), ``:105`` (``apply_output_projection``), ``:238``
(``apply_allreduce``), ``:252`` (the ``allreduce`` call with ``axis=mesh_config.tp_axis``).

**Deletions** (``03_OUTLINE.md`` §3.9):

* every bias add — ``bias=weights.wqkv_bias`` (``:25``) and the ``o_proj_bias`` add (``:120``);
  ``attention_bias: false``.
* the padding slice inside ``apply_allreduce`` (``:257-269``) — Llama's ``hidden/TP`` is always
  tile-aligned, so ``apply_allreduce`` reduces to the collective itself.
* the whole fused matmul+reduce-scatter path (``:126`` ``_FUSED_MM_RS_CONFIGS``, ``:131``
  ``is_shape_fused_mm_rs_supported``, ``:142`` ``apply_output_projection_fused_rs``, ``:214``
  ``apply_allgather_and_slice``). It is **gated off on Blackhole** anyway because the op races there
  (``:136``, comment ``:132-135``), and this is a Blackhole-only package.

**Change forced by ``DEC-011``:** with three separate Q/K/V weights, ``apply_qkv_projection``
returns ``(q, kv)`` where ``kv = ttnn.concat([k, v], -1)``, and ``split_qkv_heads_prefill`` uses
``nlp_create_qkv_heads``' **separate-KV form** — documented at
``ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads_nanobind.cpp:24``
("If optional ``input_kv`` tensor is provided, K and V will be created from ``input_kv`` and
``input`` should have shape [B, 1, S, head_dim * num_heads] instead"), argument at ``:28``.
``transpose_k_heads=False`` keeps K as ``[B, n_kv, S, head_dim]``, which is what SDPA wants.
"""

from __future__ import annotations

import ttnn

from .weights import AttentionWeights


def apply_qkv_projection(hidden_states, weights: AttentionWeights, compute_kernel_config=None):
    """Three separate linears, then one concat of K|V for the head-split op (``DEC-011``).

    Args:
        hidden_states: ``[1, 1, S_loc, hidden]``.
        weights: :class:`~.weights.AttentionWeights`.
        compute_kernel_config: passed to all three matmuls (``DEC-031``).

    Returns:
        ``(q, kv)`` — ``q [1, 1, S_loc, n_q_loc*head_dim]``,
        ``kv [1, 1, S_loc, 2*n_kv_loc*head_dim]``.
    """
    q = ttnn.linear(hidden_states, weights.wq, dtype=ttnn.bfloat16, compute_kernel_config=compute_kernel_config)
    k = ttnn.linear(hidden_states, weights.wk, dtype=ttnn.bfloat16, compute_kernel_config=compute_kernel_config)
    v = ttnn.linear(hidden_states, weights.wv, dtype=ttnn.bfloat16, compute_kernel_config=compute_kernel_config)
    kv = ttnn.concat([k, v], dim=-1)
    k.deallocate(True)
    v.deallocate(True)
    return q, kv


def split_qkv_heads_prefill(q, kv, num_heads: int, num_kv_heads: int):
    """GQA head split from the separate-KV form of ``nlp_create_qkv_heads``.

    Args:
        q: ``[1, 1, S_loc, num_heads*head_dim]``.
        kv: ``[1, 1, S_loc, 2*num_kv_heads*head_dim]`` (K then V on the last dim).
        num_heads: **local** Q head count (``mesh_config.shard_size(config.num_heads)``).
        num_kv_heads: **local** KV head count.

    Returns:
        ``(Q, K, V)`` — ``[1, num_heads, S_loc, head_dim]`` and two
        ``[1, num_kv_heads, S_loc, head_dim]``.
    """
    return ttnn.experimental.nlp_create_qkv_heads(
        q,
        kv,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def apply_rope(
    tensor, rope_mats, transformation_mat, is_decode_mode: bool = False, kv_actual_global=None, cluster_axis=None
):
    """Full rotary RoPE on Q or K. The llama3 scaling is already baked into ``rope_mats``.

    Two inner ops, exactly as the template (``operations.py:78-89``):

    * default (``kv_actual_global`` is None): ``ttnn.experimental.rotary_embedding_llama`` with
      per-chunk cos/sin already sliced to this chunk's positions — the one-shot prefill path built
      by ``tt/rope.build_prefill_rope``.
    * indexed (``kv_actual_global`` set): ``rotary_embedding_indexed`` — ``rope_mats`` carry the
      WHOLE-cache, block-cyclic, SP-sharded cos/sin from ``tt/rope.build_indexed_rope``, and the op
      derives this chunk's per-chip start row on-device. No per-chunk host reshard. (P7/P8.)

    Args:
        tensor: ``[1, n_heads, S_loc, head_dim]``.
        rope_mats: ``(cos, sin)``, last dim ``head_dim``, **Meta/interleaved** convention.
        transformation_mat: the ``[1, 1, 32, 32]`` matrix from ``tt/rope.build_transformation_mat``.
        is_decode_mode: always ``False`` in this prefill-only package.
        kv_actual_global: prior valid global KV length (tile-aligned) -> selects the indexed op.
        cluster_axis: the SP mesh axis the whole-cache cos/sin are sharded on (indexed only).
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
    """``[1, n_heads, S_loc, head_dim]`` -> ``[1, 1, S_loc, n_heads*head_dim]``."""
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights: AttentionWeights, activation_dtype, compute_kernel_config=None):
    """``o_proj``: ``[1, 1, S_loc, local_hidden]`` -> ``[1, 1, S_loc, hidden]`` (a partial sum at TP>1).

    No bias (``attention_bias: false``). The input is cast to ``bfloat8_b`` first, as the template
    does (``operations.py:117``), because ``o_proj`` itself is a ``bfloat8_b`` weight and the cast
    halves the matmul's activation bandwidth.
    """
    src = ttnn.typecast(tensor, ttnn.bfloat8_b)
    out = ttnn.matmul(src, weights.o_proj, dtype=activation_dtype, compute_kernel_config=compute_kernel_config)
    src.deallocate(True)
    return out


def apply_allreduce(tensor, mesh_config, ccl_manager):
    """Scheme A tail: TP all-reduce on ``cluster_axis=tp_axis``, ``dim=3``. No-op at TP=1.

    ``MeshConfig.allreduce`` frees its own input between the reduce-scatter and the all-gather
    (``tt/config.py:134``), so the caller must not deallocate ``tensor`` afterwards.
    """
    if mesh_config.tp > 1:
        return mesh_config.allreduce(tensor, ccl_manager, axis=mesh_config.tp_axis)
    return tensor


def apply_reduce_scatter(tensor, mesh_config, ccl_manager):
    """Scheme B tail (``DEC-018``): the reduce-scatter half only -> ``[1, 1, S_loc, hidden/tp]``."""
    if mesh_config.tp > 1:
        scattered = mesh_config.reduce_scatter(tensor, ccl_manager, dim=3, axis=mesh_config.tp_axis)
        tensor.deallocate(True)
        return scattered
    return tensor

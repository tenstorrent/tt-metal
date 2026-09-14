# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The per-op pieces of the attention block, each a thin, separately testable wrapper."""

from __future__ import annotations

import ttnn

from ..compute import matmul_compute_config
from .weights import AttentionWeights


def apply_qkv_projection(hidden_states, weights: AttentionWeights, compute_kernel_config=None):
    """``[1, 1, s, hidden]`` -> ``[1, 1, s, (n_q_local + 2*n_kv_local) * head_dim]``.

    One column-parallel matmul; Llama-3.1 has no q/k/v bias. The per-device weight was laid out as
    ``[q | k | v]`` at load (``weights.py``), which is what makes the single output splittable by
    ``nlp_create_qkv_heads``.
    """
    return ttnn.linear(
        hidden_states,
        weights.wqkv,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config or matmul_compute_config(),
    )


def split_qkv_heads(xqkv_fused, num_local_heads: int, num_local_kv_heads: int):
    """Fused QKV -> ``q [1, n_q_local, s, d]``, ``k``/``v`` ``[1, n_kv_local, s, d]``."""
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_local_heads,
        num_kv_heads=num_local_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def apply_rope(tensor, rope_mats, transformation_mat, *, kv_actual_global=None, cluster_axis=None):
    """Rotate Q or K in place of its own layout. Llama rotates the FULL head_dim — no partial slice.

    Two inner ops, same rotation:

    * ``kv_actual_global is None`` — ``rotary_embedding_llama`` with cos/sin already sliced to this
      chunk's positions and SP-sharded the same way the tokens are.
    * ``kv_actual_global`` set — ``rotary_embedding_indexed``: ``rope_mats`` carry the whole-cache
      block-cyclic SP-sharded table and the op derives this chunk's start row on device from
      ``kv_actual_global`` plus the chip's ``cluster_axis`` coordinate, which is the same block-cyclic
      arithmetic the KV-cache writer uses. No per-chunk host reshard, and chunk 0 and chunk N take
      one code path.
    """
    assert rope_mats[0].shape[-1] == tensor.shape[-1], (
        f"cos width {rope_mats[0].shape[-1]} != head_dim {tensor.shape[-1]}; Llama-3.1 has full "
        f"rotary, so a narrower table means the wrong model's rope tables were passed in"
    )
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
        tensor, rope_mats[0], rope_mats[1], transformation_mat, is_decode_mode=False
    )


def concat_heads(tensor):
    """``[1, n_q_local, s, d]`` -> ``[1, 1, s, n_q_local*d]``."""
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights: AttentionWeights, compute_kernel_config=None):
    """Row-parallel o_proj. Output is a PARTIAL sum over the TP shard until the all-reduce."""
    return ttnn.linear(
        tensor,
        weights.o_proj,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config or matmul_compute_config(),
    )


def apply_allreduce(tensor, mesh_config, ccl_manager):
    """Complete o_proj's partial sums across the TP columns, back to the replicated residual layout."""
    if mesh_config.tp <= 1:
        return tensor
    return mesh_config.all_reduce(tensor, ccl_manager)

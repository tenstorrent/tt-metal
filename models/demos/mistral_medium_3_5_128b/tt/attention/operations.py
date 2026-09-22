# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The small, individually testable steps of the attention block. Ported from
``gpt_oss_d_p/tt/attention/operations.py``.

Each function is one device op (or one op plus a reshape) so that ``prefill.attention_forward``
reads as the block's dataflow and the pieces can be exercised in isolation. Dropped relative to the
source: ``apply_qk_norm`` (no QK-norm in this model) and the bias adds (no biases).
"""

import ttnn
from models.demos.mistral_medium_3_5_128b.tt.compute import matmul_compute_kernel_config


def apply_qkv_projection(hidden_states, weights):
    """``[1, 1, tokens_local, hidden_size]`` -> fused QKV ``[1, 1, tokens_local, qkv_local]``.

    Column-parallel: no CCL, each chip produces its own heads.
    """
    return ttnn.linear(
        hidden_states, weights.wqkv, dtype=ttnn.bfloat16, compute_kernel_config=matmul_compute_kernel_config()
    )


def split_qkv_heads_prefill(xqkv_fused, num_heads_local: int, num_kv_heads_local: int):
    """Split the fused projection into ``(q, k, v)`` in head-major layout.

    Returns three tensors shaped ``[1, n_heads_local, tokens_local, head_dim]`` (q) and
    ``[1, n_kv_local, tokens_local, head_dim]`` (k, v). Undoes the per-shard interleave that
    :func:`~.weights.fuse_qkv_host` built in.
    """
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_heads_local,
        num_kv_heads=num_kv_heads_local,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def apply_rope(tensor, rope_mats, transformation_mat):
    """Rotary embedding in place of the sequence dim, via ``ttnn.experimental.rotary_embedding_llama``.

    ``rope_mats`` is the ``[cos, sin]`` pair from ``tt/rope.py``, already SP-sharded to match
    ``tensor``'s token rows and Meta-interleaved to match the permuted projection weights.
    """
    return ttnn.experimental.rotary_embedding_llama(
        tensor, rope_mats[0], rope_mats[1], transformation_mat, is_decode_mode=False
    )


def concat_heads(tensor):
    """``[1, n_heads_local, tokens_local, head_dim]`` -> ``[1, 1, tokens_local, n_heads_local*head_dim]``."""
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights, activation_dtype=ttnn.bfloat16):
    """Row-parallel ``o_proj``. Produces a **partial sum** over hidden; the caller must all-reduce.

    The fused matmul + reduce-scatter (``minimal_matmul_strided_reduce_scatter_async``) is gated
    off on Blackhole upstream, so the split matmul/all-reduce below is the only available path.

    The activation stays ``bfloat16``. The borrowed source downcast it to ``bfloat8_b`` first — a
    shape-tuned choice from a model whose L1 budget needed it, and one the binding spec forbids
    here (``dataformats.activations.default`` is ``bfloat16``; only weights and the KV cache are
    block-float). Downcasting re-quantizes the attention output on the residual path once per
    layer, which is the same accumulate-over-depth failure ``tt/compute.py`` describes.
    """
    return ttnn.matmul(
        tensor, weights.o_proj, dtype=activation_dtype, compute_kernel_config=matmul_compute_kernel_config()
    )


def apply_allreduce(tensor, mesh_config, ccl_manager):
    """Complete the row-parallel sum across the TP axis, restoring the hidden-replicated layout."""
    if mesh_config.tp == 1:
        return tensor
    # mesh_config.allreduce deallocates its input (see tt/config.py), so the caller must not reuse
    # `tensor` afterwards — every call site here assigns the return value over it.
    return mesh_config.allreduce(tensor, ccl_manager, axis=mesh_config.tp_axis)

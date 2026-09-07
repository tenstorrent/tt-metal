# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-Medium-3.5 chunked-prefill attention forward: dense GQA with full rotary (YaRN baked into
the cos/sin) and full-causal masking on every layer. No sinks, no sliding window, no partial rotary,
no QK-norm, no projection bias, no MSA/sparse path.

Adapted from ``gpt_oss_d_p/tt/attention/prefill.py`` — the cluster's ``sdpa_path_selection`` role.

Sequence-parallel *chunked* prefill is cache-backed from the first chunk: RingJointSDPA reads the
block-cyclic SP K/V cache. A one-shot request whose cache is exactly as long as its single chunk has
equal-sized Q and K/V slabs, which the ring reader does not support (it requires Q shorter than the
K/V cache), so that case keeps the exact all-gather / SDPA / reduce-scatter bootstrap.

Pipeline: QKV proj -> head split (GQA) -> full RoPE on Q,K -> KV-cache write -> SDPA -> concat heads
-> o_proj -> TP all-reduce.
"""

import ttnn
from models.demos.mistral_3_5_d_p.utils.general_utils import get_matmul_compute_config

from .config import AttentionConfig, ProgramConfig
from .dense_sp import dense_sp_attention
from .kv_cache import MistralKVCache, write_kv_chunk
from .operations import (
    apply_allreduce,
    apply_output_projection,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights

# Above this per-user chunk length the activations are carried in bf8 to bound live DRAM; below it
# bf16 keeps the residual stream's dynamic range. Same threshold as the donor.
_BF8_ACTIVATION_SEQ_THRESHOLD = 32 * 1024


def _run_sdpa(tt_q, tt_k, tt_v, config, program_config, mesh_device, seq_len):
    """Plain causal GQA SDPA — the one-shot / single-device path. Full-causal (no sliding window),
    no attention sink."""
    return ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        # Pass the configured scale explicitly so it cannot drift from config.scaling; it equals the
        # kernel default (1/sqrt(head_dim)).
        scale=config.scaling,
        program_config=program_config.get_prefill_sdpa_config(mesh_device, seq_len),
        compute_kernel_config=program_config.get_compute_kernel_config(),
    )


def attention_forward(
    hidden_states,
    rope_mats,
    weights: AttentionWeights,
    kv_cache,
    config: AttentionConfig,
    mesh_config,
    mesh_device,
    program_config: ProgramConfig,
    transformation_mat,
    position_idx,
    ccl_manager,
    user_id=0,
    batch_size=1,
    layer_idx=0,
    cached_len=0,
    indexed_rope=False,
):
    """
    Prefill forward pass (seq_len > 1).

    Args:
        hidden_states: Input tensor [1, 1, batch*seq_len, hidden_size]
        rope_mats: (cos, sin) with YaRN baked in, full head_dim wide. When ``indexed_rope`` is set
            these are the WHOLE-cache block-cyclic SP-sharded cos/sin built once by
            ``tt/rope.build_indexed_rope`` (not per-chunk).
        weights: Attention weights (fused QKV + o_proj; no biases)
        kv_cache: Optional :class:`~.kv_cache.MistralKVCache`. Required for sequence-parallel
            prefill; may be None only on the single-device unit-test path.
        config: Attention configuration
        mesh_config: Mesh parallelization config
        mesh_device: TTNN mesh device
        program_config: Model-specific program configs
        transformation_mat: Transformation matrix for RoPE
        position_idx: Position indices (unused in prefill)
        ccl_manager: Communication manager (only used when TP > 1 or SP > 1)
        user_id: cache slot index for the per-user cache write
        batch_size: number of users packed on the sequence dim
        layer_idx: this layer's index, for the per-layer cache write
        cached_len: valid prefix length already in the cache BEFORE this chunk (0 = first chunk)
        indexed_rope: use the on-device indexed RoPE

    Returns:
        Attention output [1, 1, batch*seq_len, hidden_size]
    """
    total_seq_len = hidden_states.shape[-2]
    hidden_size = hidden_states.shape[-1]
    seq_len = total_seq_len // batch_size  # Per-user sequence length
    activation_dtype = ttnn.bfloat8_b if seq_len > _BF8_ACTIVATION_SEQ_THRESHOLD else ttnn.bfloat16

    if seq_len <= 1:
        raise ValueError(f"Prefill mode requires seq_len>1, got {seq_len}. Use decode mode for single tokens.")

    # One compute config for both projections. fp32 destination accumulation is not optional at
    # these contraction depths — see get_matmul_compute_config for the measured difference.
    matmul_config = get_matmul_compute_config(mesh_device)

    # QKV projection (no bias)
    xqkv_fused = apply_qkv_projection(hidden_states, weights, compute_kernel_config=matmul_config)
    hidden_states.deallocate(True)  # Free input activations after projection

    # Reshape for batch: [1, 1, B*S, QKV] -> [B, 1, S, QKV]
    if batch_size > 1:
        xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len, -1])

    # Split into Q, K, V heads (GQA: local Q / local KV heads per TP shard)
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(xqkv_fused, num_local_heads, num_local_kv_heads)
    xqkv_fused.deallocate(True)

    # Full RoPE on Q and K.
    # indexed_rope: rope_mats are the WHOLE-cache block-cyclic SP-sharded cos/sin (built once); the
    # indexed op derives this chunk's per-chip start from kv_actual_global=cached_len + the device's
    # SP mesh coord on-device (no per-chunk host reshard). The per-user seq_len slice applies only to
    # the non-indexed multi-user path (the indexed rope carries the whole cache, never sliced here).
    rope_kv_actual = cached_len if indexed_rope else None
    rope_cluster_axis = mesh_config.sp_axis if indexed_rope else None
    if batch_size > 1 and not indexed_rope:
        rope_mats_sliced = [rope_mats[0][:, :, :seq_len, :], rope_mats[1][:, :, :seq_len, :]]
    else:
        rope_mats_sliced = rope_mats
    tt_q_orig, tt_k_orig = tt_q, tt_k
    tt_q = apply_rope(
        tt_q, rope_mats_sliced, transformation_mat, kv_actual_global=rope_kv_actual, cluster_axis=rope_cluster_axis
    )
    tt_k = apply_rope(
        tt_k, rope_mats_sliced, transformation_mat, kv_actual_global=rope_kv_actual, cluster_axis=rope_cluster_axis
    )
    tt_q_orig.deallocate(True)
    tt_k_orig.deallocate(True)

    # Per-layer KV cache write: post-RoPE K + raw V into the packed SP cache at this chunk's offset.
    # Single write point for all chunks; the cache-read path below reads the accumulated prefix.
    # tt_k / tt_v stay live (bf16) for the SDPA that follows; write_kv_chunk casts its own copy.
    if kv_cache is not None:
        assert isinstance(kv_cache, MistralKVCache), "kv_cache must be a MistralKVCache"
        write_kv_chunk(
            kv_cache,
            tt_k,
            tt_v,
            slot_idx=user_id,
            layer_idx=layer_idx,
            kv_actual=cached_len,
            sp_axis=mesh_config.sp_axis,
        )

    # --- Attention core ---
    if config.sequence_parallel and mesh_config.sp > 1:
        sp = mesh_config.sp
        assert kv_cache is not None, "SP prefill needs a KV cache"
        # Any non-first chunk is necessarily short-Q/long-K. Chunk 0 has that same valid cache-backed
        # shape only if the cache has capacity beyond its first chunk. For a one-shot request where
        # capacity == seq_len * sp, Q and K/V are equal-sized and the ring reader rejects it, so run
        # the exact replicated bootstrap instead.
        if cached_len > 0 or kv_cache.max_seq_len > seq_len * sp:
            tt_sdpa_out = dense_sp_attention(
                tt_q,
                kv_cache.k,
                kv_cache.v,
                tt_k,
                tt_v,
                kv_actual=cached_len,
                logical_n=cached_len + seq_len * sp,
                n_kv=config.num_kv_heads,
                cache_global=kv_cache.max_seq_len,
                head_dim=config.head_dim,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                program_config=program_config.get_ring_sdpa_config(mesh_device),
                compute_kernel_config=program_config.get_ring_compute_kernel_config(mesh_device),
                scale=config.scaling,
                cluster_axis=mesh_config.sp_axis,
                slot_idx=user_id,
                layer_idx=layer_idx,
                num_layers=kv_cache.num_layers,
                # The per-layer seam above already wrote this chunk's K/V into the cache.
                write_chunk=False,
            )
        else:
            full_seq_len = seq_len * sp
            tt_q_full = mesh_config.allgather(tt_q, ccl_manager, axis=mesh_config.sp_axis, dim=2)
            tt_k_full = mesh_config.allgather(tt_k, ccl_manager, axis=mesh_config.sp_axis, dim=2)
            tt_v_full = mesh_config.allgather(tt_v, ccl_manager, axis=mesh_config.sp_axis, dim=2)
            tt_q.deallocate(True)
            tt_k.deallocate(True)
            tt_v.deallocate(True)
            tt_q, tt_k, tt_v = tt_q_full, tt_k_full, tt_v_full
            tt_sdpa_out_full = _run_sdpa(tt_q, tt_k, tt_v, config, program_config, mesh_device, full_seq_len)
            tt_sdpa_out = ttnn.experimental.reduce_scatter_minimal_async(
                tt_sdpa_out_full,
                dim=2,
                multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(),
                num_links=ccl_manager.num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ccl_manager.topology,
                cluster_axis=mesh_config.sp_axis,
                barrier_semaphore=ccl_manager.get_barrier_semaphore(),
            )
            tt_sdpa_out_full.deallocate(True)
            # reduce_scatter SUMS across the SP rows, but each row computed the same full-sequence
            # result, so divide the sum back out.
            tt_sdpa_out_scaled = ttnn.multiply(tt_sdpa_out, 1.0 / sp)
            ttnn.deallocate(tt_sdpa_out)
            tt_sdpa_out = tt_sdpa_out_scaled
    elif cached_len > 0:
        # Single-device chunked cache-read: Q is the current chunk at global offset cached_len while
        # K/V span [0, cached_len+seq_len), so a plain is_causal SDPA (which assumes Q row 0 aligns
        # with K row 0) is off by cached_len and silently WRONG. The correct paths are the ring-joint
        # SDPA over the block-cyclic cache (used above whenever SP > 1) or a paged chunked SDPA.
        # Fail loud rather than return wrong output; the galaxy never takes this branch.
        raise NotImplementedError(
            "mistral_3_5_d_p: single-device (sp==1) chunked cache-read attention is not implemented — "
            "it needs a chunk-position-aware SDPA. KV-cache storage/write is supported and validated; "
            "run chunked prefill on the SP mesh, where the ring cache-read path handles it."
        )
    else:
        tt_sdpa_out = _run_sdpa(tt_q, tt_k, tt_v, config, program_config, mesh_device, seq_len)

    tt_q.deallocate(True)
    tt_k.deallocate(True)
    tt_v.deallocate(True)

    # Concat heads back to the (local) hidden dim
    tt_sdpa_out_pre_concat = tt_sdpa_out
    tt_sdpa_out = concat_heads(tt_sdpa_out)
    tt_sdpa_out_pre_concat.deallocate(True)

    # Flatten back for the output projection: [B, 1, S, H] -> [1, 1, B*S, H]
    if batch_size > 1:
        tt_sdpa_out = ttnn.reshape(tt_sdpa_out, [1, 1, total_seq_len, -1])

    # Output projection (row-parallel) + the TP all-reduce that completes its partial sums.
    tt_out = apply_output_projection(tt_sdpa_out, weights, activation_dtype, compute_kernel_config=matmul_config)
    tt_sdpa_out.deallocate(True)
    return apply_allreduce(tt_out, mesh_config, ccl_manager, hidden_size)

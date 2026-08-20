# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS chunked-prefill attention forward: GQA with full rotary (YaRN baked into the cos/sin),
attention sinks, and per-layer sliding-window vs full-causal masking. No MSA / sparse path, no
partial rotary, no QK-norm.

Sequence-parallel *chunked* prefill is cache-backed from the first chunk: RingJointSDPA reads the
block-cyclic SP K/V cache and exchanges only the bounded sliding halo when needed. A one-shot
request has equal-sized Q and K/V slabs, which RingJointSDPA sliding does not support, so it keeps
the exact all-gather / SDPA / reduce-scatter fallback.
"""

import ttnn

from .config import AttentionConfig, ProgramConfig
from .dense_sp import dense_sp_attention
from .kv_cache import GptOssKVCache, write_kv_chunk
from .operations import (
    apply_allgather_and_slice,
    apply_allreduce,
    apply_output_projection,
    apply_output_projection_fused_rs,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    is_shape_fused_mm_rs_supported,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights


def _run_sdpa(tt_q, tt_k, tt_v, weights, config, program_config, mesh_device, seq_len):
    """Single-chip GQA SDPA with sliding-window + attention-sink (the P1 dense path)."""
    return ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        # Pass the configured scale explicitly so QK scaling and the sink pre-division (both use
        # config.scaling) stay consistent for any config; it equals the kernel default (1/sqrt(head_dim)).
        scale=config.scaling,
        sliding_window_size=config.sliding_window,
        attention_sink=weights.sinks,
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
    Prefill forward pass — optimized for sequence processing (seq_len > 1).

    Pipeline: QKV proj (+bias) -> head split (GQA) -> full RoPE on Q,K -> optional KV-cache write
    -> SDPA (sliding_window + attention_sink) -> concat heads -> o_proj (+bias) -> TP allreduce.

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        rope_mats: Tuple/list of (cos, sin) matrices for RoPE (YaRN baked in, full head_dim wide).
            When ``indexed_rope`` is set these are the WHOLE-cache block-cyclic SP-sharded cos/sin
            built once by tt/rope.build_indexed_rope (not per-chunk).
        weights: Attention weights
        kv_cache: Optional GptOssKVCache (packed K/V). Required for sequence-parallel prefill;
            may be None only for the single-device unit-test path.
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
        cached_len: valid prefix length already in the cache BEFORE this chunk (0 = first/only chunk).
            Chunked SP requests, including chunk 0, use cache-backed RingJointSDPA; an equal-sized
            one-shot request uses the all-gather fallback required by the sliding operator.
        indexed_rope: use the on-device indexed RoPE (rope_mats are the whole-cache block-cyclic SP
            cos/sin; the op derives this chunk's rows from cached_len + the SP mesh coord).

    Returns:
        Attention output [batch, seq_len, hidden_size]
    """
    activation_dtype = ttnn.bfloat16
    total_seq_len = hidden_states.shape[-2]
    hidden_size = hidden_states.shape[-1]
    seq_len = total_seq_len // batch_size  # Per-user sequence length
    if seq_len > 32 * 1024:
        activation_dtype = ttnn.bfloat8_b
    else:
        activation_dtype = ttnn.bfloat16

    # Validate prefill mode
    if seq_len <= 1:
        raise ValueError(f"Prefill mode requires seq_len>1, got {seq_len}. Use decode mode for single tokens.")

    # QKV projection (+ fused bias)
    xqkv_fused = apply_qkv_projection(hidden_states, weights)
    hidden_states.deallocate(True)  # Free input activations after projection

    # Reshape for batch: [1, 1, B*S, QKV] -> [B, 1, S, QKV]
    if batch_size > 1:
        xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len, -1])

    # Split into Q, K, V heads (GQA: local Q / local KV heads per TP shard)
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)

    tt_q, tt_k, tt_v = split_qkv_heads_prefill(xqkv_fused, num_local_heads, num_local_kv_heads)
    xqkv_fused.deallocate(True)

    # Apply full RoPE on Q and K.
    # indexed_rope: rope_mats are the WHOLE-cache block-cyclic SP-sharded cos/sin (built once); the
    # indexed op derives this chunk's per-chip start from kv_actual_global=cached_len + the device's
    # SP mesh coord on-device (no per-chunk host reshard). The per-user seq_len slice only applies to
    # the non-indexed multi-user path (indexed rope carries the whole cache, never sliced here).
    rope_kv_actual = cached_len if indexed_rope else None
    rope_cluster_axis = mesh_config.sp_axis if indexed_rope else None
    if batch_size > 1 and not indexed_rope:
        rope_mats_sliced = [rope_mats[0][:, :, :seq_len, :], rope_mats[1][:, :, :seq_len, :]]
    else:
        rope_mats_sliced = rope_mats
    tt_q_orig = tt_q
    tt_k_orig = tt_k
    tt_q = apply_rope(
        tt_q,
        rope_mats_sliced,
        transformation_mat,
        is_decode_mode=False,
        kv_actual_global=rope_kv_actual,
        cluster_axis=rope_cluster_axis,
    )
    tt_k = apply_rope(
        tt_k,
        rope_mats_sliced,
        transformation_mat,
        is_decode_mode=False,
        kv_actual_global=rope_kv_actual,
        cluster_axis=rope_cluster_axis,
    )
    tt_q_orig.deallocate(True)
    tt_k_orig.deallocate(True)

    # Per-layer KV cache write: post-RoPE K + raw V into the packed SP cache at this chunk's offset
    # (cached_len). Single write point for all chunks; the cache-read path below then reads the
    # accumulated prefix. None in the unit test. tt_k / tt_v stay live (bf16) for the SDPA that follows;
    # write_kv_chunk casts its own copy to the cache dtype.
    if kv_cache is not None:
        assert isinstance(kv_cache, GptOssKVCache), "kv_cache must be a GptOssKVCache"
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
    # Chunked sequence-parallel prefill uses one cache-backed RingJointSDPA path from
    # chunk 0 onward. The first complete ring group is valid: device 0 clips its
    # left window at token zero and the remaining devices consume predecessors within
    # that same group. An equal-sized one-shot request cannot use sliding RingJointSDPA:
    # its reader requires Q to be shorter than the K/V cache, so retain the exact bootstrap.
    if config.sequence_parallel and mesh_config.sp > 1:
        sp = mesh_config.sp
        assert kv_cache is not None, "SP prefill needs a KV cache"
        # Any non-first chunk is necessarily short-Q/long-K. Chunk 0 has that same valid
        # cache-backed shape only if the request cache has capacity beyond its first chunk.
        # For one-shot max_seq_len == seq_len * sp, Q and K/V are equal and sliding RingJointSDPA
        # rejects it, so run the exact replicated bootstrap instead.
        use_cache_backed_ring = cached_len > 0 or kv_cache.max_seq_len > seq_len * sp
        if use_cache_backed_ring:
            grid = mesh_device.compute_with_storage_grid_size()
            sp_prog = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),  # carve the CCL column
                q_chunk_size=128,
                k_chunk_size=128,
                exp_approx_mode=False,
            )
            # fp32_dest_acc_en=False is required by the ring op's streaming-sink compute.
            sp_kcfg = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
            # Where this (user, layer) lives — the legacy packed cache, or (bounded_sliding_kv_cache)
            # the split full/sliding caches. batch_idx is the flat slot, so the op call below passes
            # slot_idx=batch_idx, layer_idx=0, num_layers=1 (the kernel linearizes identically).
            cache_k, cache_v, cache_batch_idx, cache_capacity, cache_bounded = kv_cache.layer_view(user_id, layer_idx)
            if cache_bounded:
                # PR1 ships allocation + the circular WRITE only: the ring cache-read cannot
                # un-rotate a CIRCULAR (bounded) sliding cache yet — that is the PR2 C++ change.
                # The cache-backed ring path serves every chunk (chunk 0 included), so bounded
                # sliding layers cannot take ANY chunked SP read in PR1. Fail loud instead of
                # reading garbage KV.
                raise NotImplementedError(
                    f"bounded_sliding_kv_cache: on-device cache-read of a bounded sliding layer "
                    f"(layer {layer_idx}, cached_len={cached_len}) is not supported yet — the ring "
                    f"cache-read un-rotation lands with PR2 (C++). PR1 supports allocation + the "
                    f"circular write only; validate via host readback (kv_cache_pcc_check)."
                )
            tt_sdpa_out = dense_sp_attention(
                tt_q,
                cache_k,
                cache_v,
                tt_k,
                tt_v,
                kv_actual=cached_len,
                logical_n=cached_len + seq_len * sp,
                n_kv=config.num_kv_heads,
                cache_global=cache_capacity,
                head_dim=config.head_dim,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                program_config=sp_prog,
                compute_kernel_config=sp_kcfg,
                scale=config.scaling,
                cluster_axis=mesh_config.sp_axis,
                attention_sink=weights.sinks,
                sliding_window_size=config.sliding_window,
                slot_idx=cache_batch_idx,
                layer_idx=0,
                num_layers=1,
                # The per-layer seam wrote current K/V into the cache before this call.
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
            tt_sdpa_out_full = _run_sdpa(tt_q, tt_k, tt_v, weights, config, program_config, mesh_device, full_seq_len)
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
            tt_sdpa_out_scaled = ttnn.multiply(tt_sdpa_out, 1.0 / sp)
            ttnn.deallocate(tt_sdpa_out)
            tt_sdpa_out = tt_sdpa_out_scaled
    elif cached_len > 0:
        # Chunked cache-read (current chunk attends the accumulated prefix) is not implemented yet.
        # The KV-cache STORAGE + write is done and validated (test_kv_cache_vs_ref); reading it back
        # for attention needs a chunk-position-aware SDPA, because Q is the current chunk at global
        # offset cached_len while K/V span [0, cached_len+seq_len) — plain is_causal SDPA (which assumes
        # Q row 0 aligns with K row 0) is off by cached_len and silently wrong. The correct paths are
        # the paged chunked_scaled_dot_product_attention (needs a paged KV cache + page table) or the
        # ring-joint dense SDPA over the block-cyclic cache (M3 dense_sp_attention). The GPT-OSS SP
        # runtime uses that path above; this single-device fallback fails loud rather than return wrong output.
        raise NotImplementedError(
            "gpt_oss_d_p: chunked cache-read attention (cached_len>0) is not implemented yet — needs a "
            "chunk-position-aware SDPA (paged chunked SDPA or ring-joint over the block-cyclic cache). "
            "KV-cache storage/write is supported and validated; only the single-device first chunk works."
        )
    else:
        tt_sdpa_out = _run_sdpa(tt_q, tt_k, tt_v, weights, config, program_config, mesh_device, seq_len)

    tt_q.deallocate(True)
    tt_k.deallocate(True)
    tt_v.deallocate(True)

    # Concat heads back to (local) hidden dim
    tt_sdpa_out_pre_concat = tt_sdpa_out
    tt_sdpa_out = concat_heads(tt_sdpa_out)
    tt_sdpa_out_pre_concat.deallocate(True)

    # Flatten back for output projection: [B, 1, S, H] -> [1, 1, B*S, H]
    if batch_size > 1:
        tt_sdpa_out = ttnn.reshape(tt_sdpa_out, [1, 1, total_seq_len, -1])

    # Output projection (+bias) + tensor-parallel allreduce.
    # When TP > 1 (and supported), use the fused matmul + reduce-scatter op; the trailing
    # all-gather + padding slice stay separate. The fused MM+RS op only supports Ring topology and
    # is gated off on Blackhole (see is_shape_fused_mm_rs_supported), so fall back to plain
    # o_proj + all-reduce otherwise.
    use_fused_rs = (
        mesh_config.tp > 1
        and is_shape_fused_mm_rs_supported(tt_sdpa_out)
        and ccl_manager.topology == ttnn.Topology.Ring
    )
    if use_fused_rs:
        rs_out = apply_output_projection_fused_rs(tt_sdpa_out, weights, mesh_config, ccl_manager)
        tt_sdpa_out.deallocate(True)
        tt_out_result = apply_allgather_and_slice(rs_out, mesh_config, ccl_manager, hidden_size)
    else:
        tt_out = apply_output_projection(tt_sdpa_out, weights, activation_dtype)
        tt_sdpa_out.deallocate(True)
        tt_out_result = apply_allreduce(tt_out, mesh_config, ccl_manager, hidden_size)
    return tt_out_result

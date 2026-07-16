# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS chunked-prefill attention forward. Mirrors ``minimax_m3/tt/attention/prefill.py`` and
``gpt_oss/tt/attention/prefill.py`` but simpler: no MSA / sparse path, no partial rotary, no
QK-norm. GQA with full rotary (YaRN baked into the cos/sin), attention sinks, and per-layer
sliding-window vs full-causal masking.

SP seam (P1): when the sequence is SP-sharded, AllGather K/V so each chip holds the full K/V, then
run single-chip SDPA. The native ring SDPA (sinks+sliding+halo CCL) swaps in later — see the SP branch.
"""

import ttnn

from .config import AttentionConfig, ProgramConfig
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
):
    """
    Prefill forward pass — optimized for sequence processing (seq_len > 1).

    Pipeline: QKV proj (+bias) -> head split (GQA) -> full RoPE on Q,K -> optional KV-cache write
    -> SDPA (sliding_window + attention_sink) -> concat heads -> o_proj (+bias) -> TP allreduce.

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        rope_mats: Tuple/list of (cos, sin) matrices for RoPE (YaRN baked in, full head_dim wide)
        weights: Attention weights
        kv_cache: Optional [k_cache, v_cache] pair; may be None (e.g. the unit test)
        config: Attention configuration
        mesh_config: Mesh parallelization config
        mesh_device: TTNN mesh device
        program_config: Model-specific program configs
        transformation_mat: Transformation matrix for RoPE
        position_idx: Position indices (unused in prefill)
        ccl_manager: Communication manager (only used when TP > 1 or SP > 1)
        user_id: cache slot index for the per-user cache write
        batch_size: number of users packed on the sequence dim
        layer_idx: this layer's index (informational)

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

    # Apply full RoPE on Q and K (use per-user seq_len positions when multi-user)
    if batch_size > 1:
        rope_mats_sliced = [rope_mats[0][:, :, :seq_len, :], rope_mats[1][:, :, :seq_len, :]]
    else:
        rope_mats_sliced = rope_mats
    tt_q_orig = tt_q
    tt_k_orig = tt_k
    tt_q = apply_rope(tt_q, rope_mats_sliced, transformation_mat, is_decode_mode=False)
    tt_k = apply_rope(tt_k, rope_mats_sliced, transformation_mat, is_decode_mode=False)
    tt_q_orig.deallocate(True)
    tt_k_orig.deallocate(True)

    # Optional KV-cache write (non-paged). None in the unit test. Post-RoPE K + raw V.
    # Write the cache in its own dtype (e.g. bf8_b) via a cast copy, but keep the bf16 tt_k/tt_v
    # live for this chunk's SDPA below (casting them in place would run attention at cache precision).
    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        k_for_cache = ttnn.typecast(tt_k, k_cache.dtype)
        v_for_cache = ttnn.typecast(tt_v, v_cache.dtype)
        if batch_size > 1:
            for b in range(batch_size):
                k_b = ttnn.slice(
                    k_for_cache, (b, 0, 0, 0), (b + 1, k_for_cache.shape[1], k_for_cache.shape[2], k_for_cache.shape[3])
                )
                v_b = ttnn.slice(
                    v_for_cache, (b, 0, 0, 0), (b + 1, v_for_cache.shape[1], v_for_cache.shape[2], v_for_cache.shape[3])
                )
                ttnn.fill_cache(k_cache, k_b, batch_idx=b)
                ttnn.fill_cache(v_cache, v_b, batch_idx=b)
                k_b.deallocate(True)
                v_b.deallocate(True)
        else:
            ttnn.fill_cache(k_cache, k_for_cache, batch_idx=user_id)
            ttnn.fill_cache(v_cache, v_for_cache, batch_idx=user_id)
        k_for_cache.deallocate(True)
        v_for_cache.deallocate(True)

    # --- Attention core ---
    if config.sequence_parallel and mesh_config.sp > 1:
        # PLACEHOLDER (SP>1): AllGather K/V to the full sequence, then single-chip SDPA. BROKEN as-is —
        # Q stays the local SP shard while K spans the full seq, so is_causal SDPA has no per-rank
        # position offset and the causal mask is wrong for rank>0. Not exercised by the single-chip PCC
        # test; replaced by the native ring SDPA (sinks+sliding+halo CCL) at P6 — validate before SP>1 use.
        tt_k_full = mesh_config.allgather(tt_k, ccl_manager, axis=mesh_config.sp_axis, dim=2)
        tt_v_full = mesh_config.allgather(tt_v, ccl_manager, axis=mesh_config.sp_axis, dim=2)
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        tt_k, tt_v = tt_k_full, tt_v_full
        tt_sdpa_out = _run_sdpa(tt_q, tt_k, tt_v, weights, config, program_config, mesh_device, seq_len)
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

    # o_proj (+bias) + TP reduce. TP>1 on Ring (and a supported shape) uses fused matmul+reduce-scatter
    # (then all-gather + slice off padding); else plain o_proj + all-reduce. Fused MM+RS is gated off
    # on Blackhole (see is_shape_fused_mm_rs_supported).
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

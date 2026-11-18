# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttnn

from ..batched_prefill import fill_batch_idx
from .config import AttentionConfig, ProgramConfig
from .operations import (
    apply_allgather_and_slice,
    apply_allreduce,
    apply_output_projection,
    apply_output_projection_allreduce_prefill,
    apply_output_projection_fused_rs,
    apply_qkv_projection,
    apply_rope,
    is_shape_fused_mm_rs_supported,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights

# (seq_len, sliding window, prefill_l1_end, tuned q / k chunks) -> index of the prefill SDPA candidate that builds
_SDPA_FIT = {}


def prefill_sequence_parallel_supported(mesh_config, weights: AttentionWeights, x) -> bool:
    """Whether the prefill attention of x ([1, 1, S, *]) can end in a reduce-scatter over the sequence: TP > 1 on the
    unpadded o_proj + all_reduce path (Blackhole; not the fused matmul + reduce-scatter op) and S % (32 * TP) == 0,
    the condition under which ttnn.all_reduce itself scatters the rows."""
    tp = mesh_config.tp
    return (
        tp > 1
        and weights.o_proj_prefill is not None
        and not is_shape_fused_mm_rs_supported(x)
        and x.padded_shape[-2] % (ttnn.TILE_SIZE * tp) == 0
    )


def prefill_forward(
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
    page_table,
    ccl_manager,
    user_id=0,
    batch_size=1,
    sequence_parallel=False,
    fused_decode=False,
):
    """
    Prefill forward pass - optimized for sequence processing (seq_len>1).

    sequence_parallel: return the o_proj partial sums reduce-scattered over the sequence (each device its S / TP
    rows) instead of all-reduced; needs the unpadded o_proj + all_reduce path (prefill_sequence_parallel_supported).

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        rope_mats: Tuple of (cos, sin) matrices for RoPE
        weights: Attention weights
        kv_cache: KV cache [k_cache, v_cache]
        config: Attention configuration
        mesh_config: Mesh parallelization config
        mesh_device: TTNN mesh device
        program_config: Model-specific program configs
        transformation_mat: Transformation matrix for RoPE
        position_idx: Position indices (unused in prefill)
        page_table: Page table for paged attention (optional)
        ccl_manager: Communication manager
        fused_decode: the model runs the fused decode path (gpt-oss-20b on QuietBox 2): the output all-reduce is
            ttnn.all_reduce (apply_allreduce use_all_reduce)

    Returns:
        Attention output [batch, seq_len, hidden_size]
    """
    activation_dtype = ttnn.bfloat16
    total_seq_len = hidden_states.shape[-2]
    hidden_size = hidden_states.shape[-1]
    seq_len = total_seq_len // batch_size  # Per-user sequence length
    # On the unpadded o_proj + all_reduce path (Blackhole, weights.o_proj_prefill) the o_proj output / all-reduce
    # payload is bf8 at every length (was bf16 up to 32K): the all-reduce moves half the bytes (measured [S, 2880]
    # 1.36 -> 0.94 ms at S = 8192). Up to 32K the decoder layer adds it to the residual into a bf16 output
    # (DecoderLayer.__call__); a bf8 post-attention residual fails the v4 per-prompt guard. Other paths keep bf16 up to
    # 32K. Both 32K gates use the per-user length seq_len (DecoderLayer.__call__ also divides by batch_size), not the
    # total token count.
    if seq_len > 32 * 1024 or weights.o_proj_prefill is not None:
        activation_dtype = ttnn.bfloat8_b
    else:
        activation_dtype = ttnn.bfloat16

    # Validate prefill mode
    if seq_len <= 1:
        raise ValueError(f"Prefill mode requires seq_len>1, got {seq_len}. Use decode mode for single tokens.")

    # QKV projection
    xqkv_fused = apply_qkv_projection(
        hidden_states,
        weights,
        program_config.get_prefill_proj_config(mesh_device, "qkv", hidden_states, weights.wqkv, mesh_config.tp),
    )
    hidden_states.deallocate(True)  # Free input activations after projection

    # Reshape for batch: [1, 1, B*S, QKV] -> [B, 1, S, QKV]
    if batch_size > 1:
        xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len, -1])

    # Split into Q, K, V heads
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)

    tt_q, tt_k, tt_v = split_qkv_heads_prefill(xqkv_fused, num_local_heads, num_local_kv_heads)
    xqkv_fused.deallocate(True)

    # Apply RoPE (use per-user seq_len positions; Model.prepare_inputs_prefill already slices them to seq_len)
    if batch_size > 1 and rope_mats[0].shape[-2] != seq_len:
        rope_mats_sliced = [rope_mats[0][:, :, :seq_len, :], rope_mats[1][:, :, :seq_len, :]]
    else:
        rope_mats_sliced = rope_mats
    tt_q_orig = tt_q
    tt_k_orig = tt_k
    tt_q = apply_rope(tt_q, rope_mats_sliced, transformation_mat, is_decode_mode=False)
    tt_k = apply_rope(tt_k, rope_mats_sliced, transformation_mat, is_decode_mode=False)
    tt_q_orig.deallocate(True)
    tt_k_orig.deallocate(True)

    # Fill KV cache
    k_cache, v_cache = kv_cache
    tt_k_pre_cast = tt_k
    tt_v_pre_cast = tt_v
    tt_k = ttnn.typecast(tt_k, k_cache.dtype)
    tt_v = ttnn.typecast(tt_v, v_cache.dtype)
    tt_k_pre_cast.deallocate(True)
    tt_v_pre_cast.deallocate(True)

    if page_table is not None:
        block_size = k_cache.shape[2]
        page_len = page_table.shape[-1] * block_size
        if batch_size > 1:
            # One batched paged fill: input batch b ([B, kv_heads, S, head_dim]) goes through page-table row b
            # (batch_idx_tensor 0..B-1); rows of padded users hold -1 and write nothing.
            assert page_table.shape[0] == batch_size, f"page table rows {page_table.shape[0]} != batch {batch_size}"
            batch_idx = fill_batch_idx(ccl_manager, batch_size)
            tt_k_fill = tt_k[:, :, :page_len, :] if page_len < tt_k.shape[2] else tt_k
            tt_v_fill = tt_v[:, :, :page_len, :] if page_len < tt_v.shape[2] else tt_v
            ttnn.experimental.paged_fill_cache(k_cache, tt_k_fill, page_table, batch_idx_tensor=batch_idx)
            ttnn.experimental.paged_fill_cache(v_cache, tt_v_fill, page_table, batch_idx_tensor=batch_idx)
            if tt_k_fill is not tt_k:
                tt_k_fill.deallocate(True)
                tt_v_fill.deallocate(True)
        else:
            tt_k_sliced = tt_k[:, :, :page_len, :] if page_len < tt_k.shape[2] else tt_k
            tt_v_sliced = tt_v[:, :, :page_len, :] if page_len < tt_v.shape[2] else tt_v
            ttnn.experimental.paged_fill_cache(k_cache, tt_k_sliced, page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(v_cache, tt_v_sliced, page_table, batch_idx=user_id)
            if page_len < tt_k.shape[2]:
                tt_k_sliced.deallocate(True)
            if page_len < tt_v.shape[2]:
                tt_v_sliced.deallocate(True)

    else:
        # Non-paged attention
        if batch_size > 1:
            for b in range(batch_size):
                k_b = ttnn.slice(tt_k, (b, 0, 0, 0), (b + 1, tt_k.shape[1], tt_k.shape[2], tt_k.shape[3]))
                v_b = ttnn.slice(tt_v, (b, 0, 0, 0), (b + 1, tt_v.shape[1], tt_v.shape[2], tt_v.shape[3]))
                ttnn.fill_cache(k_cache, k_b, batch_idx=b)
                ttnn.fill_cache(v_cache, v_b, batch_idx=b)
                k_b.deallocate(True)
                v_b.deallocate(True)
        else:
            ttnn.fill_cache(k_cache, tt_k, batch_idx=user_id)
            ttnn.fill_cache(v_cache, tt_v, batch_idx=user_id)

    # Scaled dot-product attention. With L1 held resident by a caller (prefill_l1_end), the tuned chunks' circular
    # buffers may not fit below it: the first candidate config that builds is used (remembered per shape and L1 end).
    candidates = program_config.prefill_sdpa_config_candidates(mesh_device, seq_len, config.sliding_window)
    c0 = candidates[0]
    fit_key = (seq_len, config.sliding_window, program_config.prefill_l1_end, c0.q_chunk_size, c0.k_chunk_size)
    fits = _SDPA_FIT
    i = fits.get(fit_key, 0)
    while True:
        try:
            tt_sdpa_out = ttnn.transformer.scaled_dot_product_attention(
                tt_q,
                tt_k,
                tt_v,
                is_causal=True,
                sliding_window_size=config.sliding_window,
                program_config=candidates[i],
                compute_kernel_config=program_config.get_prefill_sdpa_compute_kernel_config(mesh_device),
                attention_sink=weights.sinks,
                # Heads written side by side, [B, 1, S, nh * dh]: what nlp_concat_heads makes of the default layout,
                # without that op (measured 8K 182.6 -> 179.7 ms TTFT, bit-identical).
                output_concat_heads=True,
            )
            break
        except RuntimeError as e:  # (raised when the program is built, before anything is enqueued)
            if "clash with L1 buffers" not in str(e) or i + 1 >= len(candidates):
                raise
            i += 1
    if fits.get(fit_key) != i:
        fits[fit_key] = i
        if i:
            logger.info(
                f"prefill SDPA at {seq_len} tokens (window {config.sliding_window}): q / k chunks "
                f"{candidates[i].q_chunk_size} / {candidates[i].k_chunk_size} fit below the resident L1"
            )
    tt_q.deallocate(True)
    tt_k.deallocate(True)
    tt_v.deallocate(True)

    # Every o_proj path casts its input to bf8 first; casting here instead is the same per-tile cast.
    tt_sdpa_out_bf8 = ttnn.typecast(tt_sdpa_out, ttnn.bfloat8_b)
    tt_sdpa_out.deallocate(True)
    tt_sdpa_out = tt_sdpa_out_bf8

    # Flatten back for output projection: [B, 1, S, H] -> [1, 1, B*S, H]
    if batch_size > 1:
        tt_sdpa_out = ttnn.reshape(tt_sdpa_out, [1, 1, total_seq_len, -1])

    # Output projection + tensor-parallel allreduce.
    # When TP > 1 we use the fused matmul + reduce-scatter op; the trailing
    # all-gather + padding slice stay as separate ops. See
    # apply_output_projection_fused_rs for the per-shape tuned configs.
    if sequence_parallel:
        assert prefill_sequence_parallel_supported(
            mesh_config, weights, tt_sdpa_out
        ), "sequence-parallel prefill needs the unpadded o_proj + all_reduce path"
    if mesh_config.tp > 1 and is_shape_fused_mm_rs_supported(tt_sdpa_out):
        rs_out = apply_output_projection_fused_rs(tt_sdpa_out, weights, mesh_config, ccl_manager)
        tt_sdpa_out.deallocate(True)
        tt_out_result = apply_allgather_and_slice(rs_out, mesh_config, ccl_manager, hidden_size)
    elif mesh_config.tp > 1 and weights.o_proj_prefill is not None:
        tt_out_result = apply_output_projection_allreduce_prefill(
            tt_sdpa_out,
            weights,
            activation_dtype,
            mesh_config,
            ccl_manager,
            # apply_output_projection casts its input to bf8 before the matmul.
            program_config.get_prefill_proj_config(
                mesh_device, "o", tt_sdpa_out, weights.o_proj_prefill, mesh_config.tp, in0_dtype=ttnn.bfloat8_b
            ),
            scatter=sequence_parallel,
            batch_size=batch_size,
        )
        tt_sdpa_out.deallocate(True)
    else:
        tt_out = apply_output_projection(
            tt_sdpa_out,
            weights,
            activation_dtype,
            program_config=program_config.get_prefill_proj_config(
                mesh_device, "o", tt_sdpa_out, weights.o_proj, mesh_config.tp, in0_dtype=ttnn.bfloat8_b
            ),
        )
        tt_sdpa_out.deallocate(True)
        tt_out_result = apply_allreduce(tt_out, mesh_config, ccl_manager, hidden_size, use_all_reduce=fused_decode)
    return tt_out_result

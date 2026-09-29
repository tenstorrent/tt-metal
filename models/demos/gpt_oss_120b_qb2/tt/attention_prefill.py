# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.gpt_oss.tt.attention.config import AttentionConfig, ProgramConfig
from models.demos.gpt_oss.tt.attention.operations import (
    apply_allgather_and_slice,
    apply_allreduce,
    apply_output_projection_fused_rs,
    apply_rope,
    concat_heads,
    is_shape_fused_mm_rs_supported,
    split_qkv_heads_prefill,
)
from models.demos.gpt_oss.tt.attention.weights import AttentionWeights


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
    projection_input_dtype=ttnn.bfloat8_b,
    projection_compute_kernel_config=None,
    fill_seq_lens=None,
    chunk_start_idx=None,
    ring_tail_block=None,
    fill_start_idx=None,
):
    """
    Prefill forward pass - optimized for sequence processing (seq_len>1).

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

    # QKV projection
    xqkv_fused = ttnn.linear(
        hidden_states,
        weights.wqkv,
        bias=weights.wqkv_bias,
        dtype=ttnn.bfloat16,
        compute_kernel_config=projection_compute_kernel_config,
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

    # Apply RoPE (use per-user seq_len positions)
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

    # Fill KV cache
    k_cache, v_cache = kv_cache
    tt_k_pre_cast = tt_k
    tt_v_pre_cast = tt_v
    tt_k = ttnn.typecast(tt_k, k_cache.dtype)
    tt_v = ttnn.typecast(tt_v, v_cache.dtype)
    tt_k_pre_cast.deallocate(True)
    tt_v_pre_cast.deallocate(True)

    if chunk_start_idx:
        if batch_size != 1 or page_table is None:
            raise ValueError("a chunked prefill continuation needs a paged single-user call")
        tt_sdpa_out = _prefill_chunk_continuation(
            tt_q,
            tt_k,
            tt_v,
            k_cache,
            v_cache,
            page_table,
            int(chunk_start_idx),
            seq_len,
            weights,
            config,
            mesh_device,
            program_config,
            fill_seq_lens,
            ring_tail_block,
            fill_start_idx,
        )
        tt_q.deallocate(True)
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        return _finish_prefill(
            tt_sdpa_out,
            weights,
            mesh_config,
            mesh_device,
            ccl_manager,
            seq_len,
            batch_size,
            hidden_size,
            projection_input_dtype,
            projection_compute_kernel_config,
        )

    if page_table is not None:
        block_size = k_cache.shape[2]
        page_len = page_table.shape[-1] * block_size
        modulo = config.cache_position_modulo

        def ring_kwargs(fill_len):
            if modulo is not None and fill_len > modulo:
                return {"cache_position_modulo": modulo}
            return {}

        if batch_size > 1:
            # Per-user paged cache fill. The flattened approach (reshape batch into seq
            # + flattened page_table) produces wrong cache for users beyond the first —
            # paged_fill_cache doesn't correctly handle positions beyond the original
            # page_table's block count. Use per-user calls with batch_idx=0 instead.
            for b in range(batch_size):
                fill_len = page_len
                if fill_seq_lens is not None:
                    valid = int(fill_seq_lens[b]) if b < len(fill_seq_lens) else 0
                    if valid <= 0:
                        continue
                    fill_len = min(page_len, ((valid + block_size - 1) // block_size) * block_size)
                k_b = tt_k[b : b + 1, :, :, :]
                v_b = tt_v[b : b + 1, :, :, :]
                pt_b = page_table[b : b + 1, :]
                k_b_fill = k_b[:, :, :fill_len, :] if fill_len < k_b.shape[2] else k_b
                v_b_fill = v_b[:, :, :fill_len, :] if fill_len < v_b.shape[2] else v_b
                ttnn.experimental.paged_fill_cache(k_cache, k_b_fill, pt_b, batch_idx=0, **ring_kwargs(fill_len))
                ttnn.experimental.paged_fill_cache(v_cache, v_b_fill, pt_b, batch_idx=0, **ring_kwargs(fill_len))
        else:
            fill_len = min(page_len, tt_k.shape[2])
            if fill_seq_lens is not None:
                valid = int(fill_seq_lens[0])
                fill_len = min(fill_len, ((valid + block_size - 1) // block_size) * block_size)
            fill_off = 0 if modulo is not None or not fill_start_idx else int(fill_start_idx)
            if fill_off % block_size:
                raise ValueError(f"fill start {fill_off} is not a multiple of the block size {block_size}")
            if fill_off < fill_len:
                sliced = fill_off > 0 or fill_len < tt_k.shape[2]
                tt_k_sliced = tt_k[:, :, fill_off:fill_len, :] if sliced else tt_k
                tt_v_sliced = tt_v[:, :, fill_off:fill_len, :] if sliced else tt_v
                if fill_off:
                    fill_table = ttnn.slice(
                        page_table, [user_id, fill_off // block_size], [user_id + 1, fill_len // block_size]
                    )
                    fill_idx = 0
                else:
                    fill_table = page_table
                    fill_idx = user_id
                ttnn.experimental.paged_fill_cache(
                    k_cache, tt_k_sliced, fill_table, batch_idx=fill_idx, **ring_kwargs(fill_len)
                )
                ttnn.experimental.paged_fill_cache(
                    v_cache, tt_v_sliced, fill_table, batch_idx=fill_idx, **ring_kwargs(fill_len)
                )
                if fill_off:
                    fill_table.deallocate(True)
                if sliced:
                    tt_k_sliced.deallocate(True)
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

    # Scaled dot-product attention
    tt_sdpa_out = ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        sliding_window_size=config.sliding_window,
        program_config=program_config.get_prefill_sdpa_config(mesh_device, seq_len),
        compute_kernel_config=program_config.get_compute_kernel_config(),
        attention_sink=weights.sinks,
    )
    tt_q.deallocate(True)
    tt_k.deallocate(True)
    tt_v.deallocate(True)
    return _finish_prefill(
        tt_sdpa_out,
        weights,
        mesh_config,
        mesh_device,
        ccl_manager,
        seq_len,
        batch_size,
        hidden_size,
        projection_input_dtype,
        projection_compute_kernel_config,
    )


def _prefill_chunk_continuation(
    tt_q,
    tt_k,
    tt_v,
    k_cache,
    v_cache,
    page_table,
    chunk_start_idx,
    seq_len,
    weights,
    config,
    mesh_device,
    program_config,
    fill_seq_lens,
    ring_tail_block,
    fill_start_idx=None,
):
    """Attention for one later chunk of a prompt (positions [start, start + seq_len)).

    The chunk's K/V go into the paged cache first. A full-attention layer then runs
    the chunked SDPA over the whole cached prefix; with ``fill_start_idx`` it writes
    only the positions from there on, because the earlier ones are cached blocks
    other requests may share. A bounded-ring layer (sliding window) keeps only the
    last positions in its ring, so it attends a square [previous window | chunk]
    with the ordinary windowed SDPA: the window comes from the ring blocks named by
    ``ring_tail_block``, the chunk from the fresh K/V, and the ``window`` filler
    query rows in front are dropped. A negative ``ring_tail_block`` means the ring
    holds nothing for this prompt (a cached prefix computed elsewhere); the chunk
    then attends within itself, and the caller has started it far enough back for
    the window to be exact again by the first new token.
    """
    block_size = int(k_cache.shape[2])
    valid = seq_len if not fill_seq_lens else int(fill_seq_lens[0])
    fill_len = min(seq_len, ((valid + block_size - 1) // block_size) * block_size)
    tt_k_fill = tt_k[:, :, :fill_len, :] if fill_len < seq_len else tt_k
    tt_v_fill = tt_v[:, :, :fill_len, :] if fill_len < seq_len else tt_v
    modulo = config.cache_position_modulo
    first_block = chunk_start_idx // block_size
    if modulo is not None:
        window = int(config.sliding_window)
        if chunk_start_idx % block_size:
            raise ValueError(f"chunk start {chunk_start_idx} is not a multiple of the block size {block_size}")
        if ring_tail_block is None:
            raise ValueError("a bounded-ring layer needs ring_tail_block for a chunk continuation")
        cold_ring = int(ring_tail_block) < 0
        if seq_len < window and not cold_ring:
            raise ValueError(f"chunk of {seq_len} tokens is shorter than the {window}-token window")
        tail_blocks = window // block_size
        nkv = int(tt_k.shape[1])
        head_dim = int(tt_k.shape[3])

        def ring_tail(cache):
            start = [int(ring_tail_block), 0, 0, 0]
            end = [int(ring_tail_block) + tail_blocks, nkv, block_size, head_dim]
            if int(cache.shape[0]) % tail_blocks:
                blocks = ttnn.slice(cache, start, end)
            else:
                bounds = [
                    ttnn.from_torch(
                        torch.tensor(values, dtype=torch.int32),
                        device=mesh_device,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    )
                    for values in (start, end)
                ]
                blocks = ttnn.slice(
                    cache, bounds[0], bounds[1], slice_dim=0, num_devices=int(cache.shape[0]) // tail_blocks
                )
                for bound in bounds:
                    bound.deallocate(True)
            heads_first = ttnn.permute(blocks, (1, 0, 2, 3))
            blocks.deallocate(True)
            tail = ttnn.reshape(heads_first, (1, nkv, tail_blocks * block_size, head_dim))
            cast = ttnn.typecast(tail, tt_q.dtype)
            tail.deallocate(True)
            return cast

        ring_kwargs = {"cache_position_modulo": modulo} if fill_len > modulo else {}
        first_block = chunk_start_idx // block_size
        ring_table = ttnn.slice(page_table, [0, first_block], [1, first_block + fill_len // block_size])
        if cold_ring:
            ttnn.experimental.paged_fill_cache(k_cache, tt_k_fill, ring_table, batch_idx=0, **ring_kwargs)
            ttnn.experimental.paged_fill_cache(v_cache, tt_v_fill, ring_table, batch_idx=0, **ring_kwargs)
            ring_table.deallocate(True)
            return ttnn.transformer.scaled_dot_product_attention(
                tt_q,
                tt_k,
                tt_v,
                is_causal=True,
                sliding_window_size=window,
                program_config=program_config.get_prefill_sdpa_config(mesh_device, seq_len),
                compute_kernel_config=program_config.get_compute_kernel_config(),
                attention_sink=weights.sinks,
            )
        k_tail = ring_tail(k_cache)
        v_tail = ring_tail(v_cache)
        ttnn.experimental.paged_fill_cache(k_cache, tt_k_fill, ring_table, batch_idx=0, **ring_kwargs)
        ttnn.experimental.paged_fill_cache(v_cache, tt_v_fill, ring_table, batch_idx=0, **ring_kwargs)
        ring_table.deallocate(True)
        nqh = int(tt_q.shape[1])
        if seq_len > window:
            q_pad = ttnn.slice(tt_q, [0, 0, 0, 0], [1, nqh, window, head_dim])
        else:
            q_pad = ttnn.clone(tt_q, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q_cat = ttnn.concat([q_pad, tt_q], dim=2)
        q_pad.deallocate(True)
        k_chunk = ttnn.typecast(tt_k, tt_q.dtype)
        v_chunk = ttnn.typecast(tt_v, tt_q.dtype)
        k_cat = ttnn.concat([k_tail, k_chunk], dim=2)
        v_cat = ttnn.concat([v_tail, v_chunk], dim=2)
        k_chunk.deallocate(True)
        v_chunk.deallocate(True)
        k_tail.deallocate(True)
        v_tail.deallocate(True)
        square = ttnn.transformer.scaled_dot_product_attention(
            q_cat,
            k_cat,
            v_cat,
            is_causal=True,
            sliding_window_size=window,
            program_config=program_config.get_prefill_sdpa_config(mesh_device, window + seq_len),
            compute_kernel_config=program_config.get_compute_kernel_config(),
            attention_sink=weights.sinks,
        )
        q_cat.deallocate(True)
        k_cat.deallocate(True)
        v_cat.deallocate(True)
        out = ttnn.slice(square, [0, 0, window, 0], [1, nqh, window + seq_len, head_dim])
        square.deallocate(True)
        return out

    fill_off = 0 if not fill_start_idx else max(0, int(fill_start_idx) - chunk_start_idx)
    if fill_off % block_size:
        raise ValueError(f"fill start {fill_start_idx} is not a multiple of the block size {block_size}")
    if fill_off < fill_len:
        if fill_off:
            tt_k_fill = tt_k[:, :, fill_off:fill_len, :]
            tt_v_fill = tt_v[:, :, fill_off:fill_len, :]
        chunk_table = ttnn.slice(
            page_table, [0, first_block + fill_off // block_size], [1, first_block + fill_len // block_size]
        )
        ttnn.experimental.paged_fill_cache(k_cache, tt_k_fill, chunk_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(v_cache, tt_v_fill, chunk_table, batch_idx=0)
        chunk_table.deallocate(True)
    return ttnn.transformer.chunked_scaled_dot_product_attention(
        tt_q,
        k_cache,
        v_cache,
        page_table,
        chunk_start_idx,
        program_config=program_config.get_prefill_sdpa_config(mesh_device, seq_len),
        compute_kernel_config=program_config.get_compute_kernel_config(),
        attention_sink=weights.sinks,
        sliding_window_size=config.sliding_window,
    )


def _finish_prefill(
    tt_sdpa_out,
    weights,
    mesh_config,
    mesh_device,
    ccl_manager,
    seq_len,
    batch_size,
    hidden_size,
    projection_input_dtype,
    projection_compute_kernel_config,
):
    activation_dtype = ttnn.bfloat8_b if seq_len > 32 * 1024 else ttnn.bfloat16
    total_seq_len = seq_len * batch_size
    # Concat heads and apply output projection
    tt_sdpa_out_pre_concat = tt_sdpa_out
    tt_sdpa_out = concat_heads(tt_sdpa_out, is_decode_mode=False)
    tt_sdpa_out_pre_concat.deallocate(True)

    # Flatten back for output projection: [B, 1, S, H] -> [1, 1, B*S, H]
    if batch_size > 1:
        tt_sdpa_out = ttnn.reshape(tt_sdpa_out, [1, 1, total_seq_len, -1])

    # Output projection + tensor-parallel allreduce.
    # When TP > 1 we use the fused matmul + reduce-scatter op; the trailing
    # all-gather + padding slice stay as separate ops. See
    # apply_output_projection_fused_rs for the per-shape tuned configs.
    if mesh_config.tp > 1 and is_shape_fused_mm_rs_supported(tt_sdpa_out):
        rs_out = apply_output_projection_fused_rs(tt_sdpa_out, weights, mesh_config, ccl_manager)
        tt_sdpa_out.deallocate(True)
        tt_out_result = apply_allgather_and_slice(rs_out, mesh_config, ccl_manager, hidden_size)
    else:
        projection_input = ttnn.typecast(tt_sdpa_out, projection_input_dtype)
        tt_out = ttnn.matmul(
            projection_input,
            weights.o_proj,
            dtype=activation_dtype,
            compute_kernel_config=projection_compute_kernel_config,
        )
        projection_input.deallocate(True)
        ttnn.add(tt_out, weights.o_proj_bias, output_tensor=tt_out)
        tt_sdpa_out.deallocate(True)
        tt_out_result = apply_allreduce(tt_out, mesh_config, ccl_manager, hidden_size)
    return tt_out_result

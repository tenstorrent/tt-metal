# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Gated Attention.
"""

import ttnn


def rotate_half_ttnn(x):
    """Rotates half the hidden dims of the input."""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def apply_rotary_pos_emb_ttnn(q, k, cos, sin):
    """Apply RoPE to query and key tensors using TTNN ops."""
    if len(cos.shape) < 4:
        cos = ttnn.unsqueeze(cos, 1)  # [B, 1, T, D]
        sin = ttnn.unsqueeze(sin, 1)

    rotary_dim = cos.shape[-1]
    full_dim = q.shape[-1]

    q_rot = q[..., :rotary_dim]
    k_rot = k[..., :rotary_dim]

    q_embed = ttnn.add(
        ttnn.multiply(q_rot, cos),
        ttnn.multiply(rotate_half_ttnn(q_rot), sin),
    )
    k_embed = ttnn.add(
        ttnn.multiply(k_rot, cos),
        ttnn.multiply(rotate_half_ttnn(k_rot), sin),
    )

    if rotary_dim < full_dim:
        q_pass = q[..., rotary_dim:]
        k_pass = k[..., rotary_dim:]
        q_embed = ttnn.concat([q_embed, q_pass], dim=-1)
        k_embed = ttnn.concat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


def rms_norm_zero_centered_ttnn(x, weight, eps=1e-6):
    """
    Zero-centered RMSNorm using TTNN: x * rsqrt(mean(x^2) + eps) * (1 + weight).
    """
    x_sq = ttnn.multiply(x, x)
    variance = ttnn.mean(x_sq, dim=-1, keepdim=True)
    inv_rms = ttnn.rsqrt(ttnn.add(variance, eps))
    x_normed = ttnn.multiply(x, inv_rms)
    scale = ttnn.add(weight, 1.0)
    return ttnn.multiply(x_normed, scale)


def _get_sdpa_program_config(device, seq_len, q_seq_len=None):
    """Build SDPAProgramConfig with chunk sizes tuned to sequence length.

    For decode with pre-allocated cache (q_seq_len=1), use small chunks.
    For segmented prefill (q_seq_len < seq_len, both > 1), use small chunks.
    For regular prefill, scale chunks based on seq_len.
    """
    grid_size = device.compute_with_storage_grid_size()
    if q_seq_len is not None and q_seq_len <= 1 and seq_len >= 512:
        # Decode with large pre-allocated cache: use small chunks to avoid L1 OOM
        q_chunk = 32
        k_chunk = 64
    elif q_seq_len is not None and q_seq_len > 1 and seq_len > q_seq_len:
        # Segmented prefill: Q shorter than KV, use small chunks to avoid L1 OOM
        q_chunk = 64
        k_chunk = 64
    elif seq_len >= 8192:
        q_chunk = 64
        k_chunk = 64
    elif seq_len >= 4096:
        q_chunk = 128
        k_chunk = 128
    elif seq_len >= 2048:
        q_chunk = 128
        k_chunk = 128
    else:
        q_chunk = 64
        k_chunk = 64
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )


def _get_sdpa_compute_kernel_config():
    """WormholeComputeKernelConfig for SDPA -- HiFi2 with fp32 accumulation."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def gated_attention_forward_ttnn(
    hidden_states,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    q_norm_weight,
    k_norm_weight,
    cos,
    sin,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    device,
    norm_eps=1e-6,
    use_optimized_concat=False,
    past_key=None,
    past_value=None,
    compute_kernel_config=None,
    kv_cache_key=None,
    kv_cache_value=None,
    cache_pos=None,
    cache_len=None,
    memory_config=None,
    norm_weights_pre_offset=False,
    # Trace-compatible mode: save new K/V to buffers, write to staging pos, use full cache + mask
    trace_new_k_buf=None,
    trace_new_v_buf=None,
    trace_attn_mask=None,
    trace_kv_pad_zeros=None,
    trace_staging_pos=None,
    # Decode-optimized SDPA: pass cur_pos_tensor to use sdpa_decode (skips mask management)
    cur_pos_tensor=None,
):
    """
    TTNN forward pass for Gated Attention with KV cache support.

    Uses ttnn.transformer.scaled_dot_product_attention (FlashAttention-2 kernel)
    with SDPAProgramConfig for tiling and WormholeComputeKernelConfig for precision.
    The fused kernel handles GQA (num_q_heads != num_kv_heads) internally.

    Args:
        hidden_states: ttnn.Tensor [B, T, hidden_size]
        *_proj_weight: ttnn.Tensor weight matrices in [in_features, out_features] format
                       (transposed from PyTorch convention)
        q_norm_weight, k_norm_weight: ttnn.Tensor [head_dim]
        cos, sin: ttnn.Tensor [B, T, head_dim] rotary embeddings
        num_attention_heads: number of Q heads
        num_key_value_heads: number of KV heads
        head_dim: dimension per head
        device: ttnn device
        norm_eps: RMSNorm epsilon
        use_optimized_concat: if True, use ttnn.transformer.concatenate_heads
        past_key: ttnn.Tensor [B, H_kv, S_past, D] or None
        past_value: ttnn.Tensor [B, H_kv, S_past, D] or None

    Returns:
        output: ttnn.Tensor [B, T, hidden_size]
        new_key: ttnn.Tensor [B, H_kv, S_total, D] updated KV cache key
        new_value: ttnn.Tensor [B, H_kv, S_total, D] updated KV cache value
    """
    B = hidden_states.shape[0]
    T = hidden_states.shape[1]
    scaling = head_dim**-0.5

    # Q projection: 2x wide
    ckc = compute_kernel_config
    qg = ttnn.linear(hidden_states, q_proj_weight, compute_kernel_config=ckc, memory_config=memory_config)
    qg = ttnn.reshape(qg, [B, T, num_attention_heads, head_dim * 2])
    # Split into query and gate
    query_states, gate = ttnn.chunk(qg, 2, dim=-1)
    ttnn.deallocate(qg)
    gate = ttnn.reshape(gate, [B, T, num_attention_heads * head_dim])

    # Q norm + transpose to [B, H_q, T, D]
    if norm_weights_pre_offset:
        query_states = ttnn.rms_norm(query_states, weight=q_norm_weight, epsilon=norm_eps)
    else:
        query_states = rms_norm_zero_centered_ttnn(query_states, q_norm_weight, eps=norm_eps)
    query_states = ttnn.transpose(query_states, 1, 2)

    # K projection + norm + transpose to [B, H_kv, T, D]
    key_states = ttnn.linear(hidden_states, k_proj_weight, compute_kernel_config=ckc, memory_config=memory_config)
    key_states = ttnn.reshape(key_states, [B, T, num_key_value_heads, head_dim])
    if norm_weights_pre_offset:
        key_states = ttnn.rms_norm(key_states, weight=k_norm_weight, epsilon=norm_eps)
    else:
        key_states = rms_norm_zero_centered_ttnn(key_states, k_norm_weight, eps=norm_eps)
    key_states = ttnn.transpose(key_states, 1, 2)

    # V projection + transpose to [B, H_kv, T, D]
    value_states = ttnn.linear(hidden_states, v_proj_weight, compute_kernel_config=ckc, memory_config=memory_config)
    value_states = ttnn.reshape(value_states, [B, T, num_key_value_heads, head_dim])
    value_states = ttnn.transpose(value_states, 1, 2)

    # RoPE — for decode (T==1), use reshape (metadata-only) instead of unsqueeze (data movement)
    if T == 1 and len(cos.shape) == 3:
        cos_4d = ttnn.reshape(cos, [cos.shape[0], 1, 1, cos.shape[-1]])
        sin_4d = ttnn.reshape(sin, [sin.shape[0], 1, 1, sin.shape[-1]])
        query_states, key_states = apply_rotary_pos_emb_ttnn(query_states, key_states, cos_4d, sin_4d)
    else:
        query_states, key_states = apply_rotary_pos_emb_ttnn(query_states, key_states, cos, sin)

    # KV cache handling
    _use_sdpa_decode = False
    if cur_pos_tensor is not None and kv_cache_key is not None and T == 1:
        # Trace-compatible SDPA decode: write K/V at dynamic position, attend up to cur_pos.
        # Uses paged_update_cache (takes device tensor for index, trace-compatible) +
        # sdpa_decode (reads cache[:, :, :cur_pos+1, :], no mask needed).
        # Eliminates staging position, mask updates, and post-trace cache copies.
        #
        # paged_update_cache expects: input [1, N, seq, D], cache [N, 1, max_seq, D]
        # where N = batch * num_kv_heads. Reshape cache view for the write, SDPA reads
        # from the original [B, H_kv, max_seq, D] layout (same underlying data).
        N = B * num_key_value_heads
        max_seq = kv_cache_key.shape[2]
        k_for_cache = ttnn.reshape(key_states, [1, N, T, head_dim])
        v_for_cache = ttnn.reshape(value_states, [1, N, T, head_dim])
        # paged_update_cache requires HEIGHT_SHARDED input.
        # [1, N, 1, head_dim] in TILE_LAYOUT → 2D: height=N*32, width=head_dim.
        # Need N cores with shard [32, head_dim] each (one tile-row per core).
        _shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(N - 1, 0))})
        _shard_spec = ttnn.ShardSpec(_shard_grid, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR)
        _sharded_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec)
        k_for_cache = ttnn.to_memory_config(k_for_cache, _sharded_mc)
        v_for_cache = ttnn.to_memory_config(v_for_cache, _sharded_mc)
        kv_cache_key_reshaped = ttnn.reshape(kv_cache_key, [N, 1, max_seq, head_dim])
        kv_cache_value_reshaped = ttnn.reshape(kv_cache_value, [N, 1, max_seq, head_dim])
        # paged_update_cache expects one index per "batch" element (N = B * H_kv).
        # cur_pos_tensor must have N elements — all the same position, pre-allocated by caller.
        ttnn.experimental.paged_update_cache(kv_cache_key_reshaped, k_for_cache, update_idxs_tensor=cur_pos_tensor)
        ttnn.experimental.paged_update_cache(kv_cache_value_reshaped, v_for_cache, update_idxs_tensor=cur_pos_tensor)
        new_key = kv_cache_key
        new_value = kv_cache_value
        _use_sdpa_decode = True
        S_total = kv_cache_key.shape[2]
        is_causal = False
        segmented_attn_mask = None
    elif trace_new_k_buf is not None and trace_attn_mask is not None:
        # Legacy trace mode: staging position + mask (fallback when cur_pos_tensor not available)
        ttnn.copy(key_states, trace_new_k_buf)
        ttnn.copy(value_states, trace_new_v_buf)
        if trace_kv_pad_zeros is not None and trace_staging_pos is not None:
            k_padded = ttnn.concat([key_states, trace_kv_pad_zeros], dim=2)
            v_padded = ttnn.concat([value_states, trace_kv_pad_zeros], dim=2)
            ttnn.update_cache(kv_cache_key, k_padded, update_idx=trace_staging_pos)
            ttnn.update_cache(kv_cache_value, v_padded, update_idx=trace_staging_pos)
            ttnn.deallocate(k_padded)
            ttnn.deallocate(v_padded)
        key_states = kv_cache_key
        value_states = kv_cache_value
        new_key = kv_cache_key
        new_value = kv_cache_value
        S_total = kv_cache_key.shape[2]
        is_causal = False
        segmented_attn_mask = None
    elif kv_cache_key is not None and cache_pos is not None and cur_pos_tensor is not None and T == 1:
        # Decode with SDPA decode variant: write K/V at cache_pos, attend with cur_pos_tensor.
        # sdpa_decode reads cache[:, :, :cur_pos+1, :] internally — no slicing or mask needed.
        ttnn.copy(key_states, kv_cache_key[:, :, cache_pos : cache_pos + T, :])
        ttnn.copy(value_states, kv_cache_value[:, :, cache_pos : cache_pos + T, :])

        new_key = kv_cache_key
        new_value = kv_cache_value
        _use_sdpa_decode = True
        S_total = kv_cache_key.shape[2]
        is_causal = False
        segmented_attn_mask = None
    elif kv_cache_key is not None and cache_pos is not None:
        # Pre-allocated KV cache mode: write new K/V at cache_pos, read 0..cache_pos+T
        # Write new tokens into cache at position cache_pos
        # kv_cache_key shape: [B, H_kv, max_seq_len, D]
        ttnn.copy(key_states, kv_cache_key[:, :, cache_pos : cache_pos + T, :])
        ttnn.copy(value_states, kv_cache_value[:, :, cache_pos : cache_pos + T, :])

        # Read valid portion of cache for attention
        valid_len = cache_pos + T
        key_states = kv_cache_key[:, :, :valid_len, :]
        key_states = ttnn.to_layout(key_states, ttnn.TILE_LAYOUT)
        value_states = kv_cache_value[:, :, :valid_len, :]
        value_states = ttnn.to_layout(value_states, ttnn.TILE_LAYOUT)

        new_key = kv_cache_key
        new_value = kv_cache_value
        S_total = valid_len
        is_causal = T > 1  # Only causal during prefill
        segmented_attn_mask = None
    elif past_key is not None:
        # Legacy concat mode — used during segmented prefill
        past_len = past_key.shape[2]
        key_states = ttnn.concat([past_key, key_states], dim=2)
        value_states = ttnn.concat([past_value, value_states], dim=2)
        new_key = key_states
        new_value = value_states
        S_total = key_states.shape[2]
        if T > 1 and S_total > T:
            # Segmented prefill: Q_len != KV_len, need explicit causal mask
            # Each query position i (in current segment) can attend to:
            #   positions 0..past_len+i (all past tokens + tokens up to i in current segment)
            import torch as _torch

            row_idx = _torch.arange(T).unsqueeze(1)  # [T, 1]
            col_idx = _torch.arange(S_total).unsqueeze(0)  # [1, S_total]
            # Mask future positions: col > past_len + row means future token
            mask = (
                _torch.where(
                    col_idx > past_len + row_idx,
                    _torch.tensor(-1e4, dtype=_torch.bfloat16),
                    _torch.tensor(0.0, dtype=_torch.bfloat16),
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )  # [1, 1, T, S_total]
            segmented_attn_mask = ttnn.from_torch(
                mask,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            is_causal = False
        else:
            segmented_attn_mask = None
            is_causal = False
    else:
        # First call (prefill, no cache yet)
        new_key = key_states
        new_value = value_states
        S_total = key_states.shape[2]
        is_causal = True
        segmented_attn_mask = None

    # Fused scaled dot-product attention
    if _use_sdpa_decode:
        # Decode-optimized SDPA: reads cache[:, :, :cur_pos+1, :] internally.
        # sdpa_decode expects Q as [1, B, H, D] (not [B, H, 1, D] like regular SDPA).
        # Transpose from [B=1, H, T=1, D] → [T=1, B=1, H, D].
        q_decode = ttnn.transpose(query_states, 1, 2)  # [1, 1, 16, 256]
        # cur_pos_tensor has B*H_kv elements; sdpa_decode needs B elements.
        sdpa_pos = cur_pos_tensor[:B] if cur_pos_tensor.shape[0] > B else cur_pos_tensor
        attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode,
            kv_cache_key,
            kv_cache_value,
            cur_pos_tensor=sdpa_pos,
            scale=scaling,
            program_config=_get_sdpa_program_config(device, S_total, q_seq_len=T),
            compute_kernel_config=_get_sdpa_compute_kernel_config(),
        )
        # Transpose back to [B, H, T, D] for head concatenation
        attn_output = ttnn.transpose(attn_output, 1, 2)
    else:
        if trace_new_k_buf is not None:
            _attn_mask = trace_attn_mask
        elif past_key is not None and segmented_attn_mask is not None:
            _attn_mask = segmented_attn_mask
        else:
            _attn_mask = None
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=_attn_mask,
            is_causal=is_causal,
            scale=scaling,
            program_config=_get_sdpa_program_config(device, S_total, q_seq_len=T),
            compute_kernel_config=_get_sdpa_compute_kernel_config(),
        )
    ttnn.deallocate(query_states)

    # Convert from [B, H, T, D] back to [B, T, H*D]
    if use_optimized_concat:
        attn_output = ttnn.transformer.concatenate_heads(attn_output)
    else:
        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, [B, T, num_attention_heads * head_dim])

    # Apply sigmoid gate
    gate = ttnn.sigmoid(gate)
    attn_output = ttnn.multiply(attn_output, gate)
    ttnn.deallocate(gate)

    # Output projection
    attn_output = ttnn.linear(attn_output, o_proj_weight, compute_kernel_config=ckc, memory_config=memory_config)

    return attn_output, new_key, new_value

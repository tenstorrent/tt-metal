# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Gated Attention.
"""

import os as _os

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


def apply_rotary_pos_emb_fused(q, k, cos, sin, memory_config=None):
    """Apply RoPE to query and key tensors using the fused ttnn.experimental.rotary_embedding_hf op.

    Prefill-only (T > 1): q/k must be [1, n_heads, seq_len, head_dim] TILE bf16; cos/sin must be
    [1, 1, seq_len, rotary_dim] (reshaped here from [1, seq_len, rotary_dim] if 3-D) TILE bf16,
    with cos/sin seq_len >= q/k seq_len. rotary_embedding_hf implements HF rotate_half semantics
    directly, so this replaces the hand-rolled slice/neg/concat/mul/add of apply_rotary_pos_emb_ttnn.

    When rotary_dim == head_dim (full rotary), the op runs directly on q and k. Otherwise (partial
    rotary, e.g. Qwen3.5's rotary_dim=64 of head_dim=256), the op runs only on the rotary slice and
    the untouched pass-through dims are concatenated back on: slice + rotary + slice + concat = 4
    ops per tensor (8 total), vs. ~10 ops for the legacy elementwise path.
    """
    if len(cos.shape) == 3:
        cos = ttnn.reshape(cos, [cos.shape[0], 1, cos.shape[1], cos.shape[2]])  # metadata only
        sin = ttnn.reshape(sin, [sin.shape[0], 1, sin.shape[1], sin.shape[2]])

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim == head_dim:
        q_embed = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False, memory_config=memory_config)
        k_embed = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False, memory_config=memory_config)
        return q_embed, k_embed

    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    q_rot_embed = ttnn.experimental.rotary_embedding_hf(
        q_rot, cos, sin, is_decode_mode=False, memory_config=memory_config
    )
    ttnn.deallocate(q_rot)
    q_embed = ttnn.concat([q_rot_embed, q_pass], dim=-1, memory_config=memory_config)
    ttnn.deallocate(q_rot_embed)
    ttnn.deallocate(q_pass)

    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]
    k_rot_embed = ttnn.experimental.rotary_embedding_hf(
        k_rot, cos, sin, is_decode_mode=False, memory_config=memory_config
    )
    ttnn.deallocate(k_rot)
    k_embed = ttnn.concat([k_rot_embed, k_pass], dim=-1, memory_config=memory_config)
    ttnn.deallocate(k_rot_embed)
    ttnn.deallocate(k_pass)

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


def _get_sdpa_program_config(device, seq_len, q_seq_len=None, chunk_start_idx=None):
    """Build SDPAProgramConfig with chunk sizes tuned to sequence length.

    For chunked paged prefill (chunk_start_idx >= 0), chunk sizes follow the
    tt_transformers reference (model_config.py:1433). For chunk_start_idx > 0,
    chunk sizes must divide chunk_start_idx (uses lowest-set-bit trick).
    For decode with pre-allocated cache (q_seq_len=1), use small chunks.
    For segmented prefill (q_seq_len < seq_len, both > 1), use small chunks.
    For regular prefill, scale chunks based on seq_len.
    """
    grid_size = device.compute_with_storage_grid_size()
    if chunk_start_idx is not None and chunk_start_idx == 0:
        # First chunk of paged prefill: no divisibility constraint.
        # Swept on P150 at S=4096 (8x256 q, 2x256 kv, causal): 64/64 2.81 ms, 128/128 1.52 ms,
        # 256/128 1.07 ms; 256/256 does not fit L1. QWEN36_SDPA_PREFILL_CHUNKS="q,k" overrides.
        env = _os.environ.get("QWEN36_SDPA_PREFILL_CHUNKS")
        if env:
            q_chunk, k_chunk = (int(v) for v in env.split(","))
        elif seq_len >= 4096:
            q_chunk, k_chunk = 256, 128
        elif seq_len >= 2048:
            q_chunk, k_chunk = 128, 128
        else:
            q_chunk, k_chunk = 64, 64
    elif chunk_start_idx is not None and chunk_start_idx > 0:
        # Subsequent chunks of paged prefill: chunk sizes must divide chunk_start_idx.
        # (x & -x) extracts the largest power-of-2 factor of x.
        # Use conservative sizes on Blackhole to avoid L1 clashes.
        if seq_len >= 2048:
            q_chunk = min(64, chunk_start_idx & -chunk_start_idx)
            k_chunk = min(64, chunk_start_idx & -chunk_start_idx)
        else:
            q_chunk = min(64, chunk_start_idx & -chunk_start_idx)
            k_chunk = min(64, chunk_start_idx & -chunk_start_idx)
        assert chunk_start_idx % q_chunk == 0, f"chunk_start_idx={chunk_start_idx} not divisible by q_chunk={q_chunk}"
    elif q_seq_len is not None and q_seq_len <= 1 and seq_len >= 512:
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


def _get_flexible_sdpa_program_config(device):
    """Fixed SDPAProgramConfig for the FLEXIBLE chunked SDPA (chunk_start_idx supplied as a
    runtime device tensor). The chunk size must divide every chunk_start (all multiples of the
    2048-token GDN chunk) so ONE program serves all chunk positions — required so a single
    captured trace can be replayed per chunk in chunk-outer prefill.

    Default 128 (not 64): 128 still divides 2048, and a microbench showed q/k=128 is ~2x faster
    than 64 on the prefill SDPA shape (16 q-heads / 4 kv-heads, d=256, KV~32k) — numerically exact
    (only tiling granularity changes). Opt back to 64 with QWEN9B_SDPA_QK64=1 if 128 ever clashes
    with L1 in the paged chunked-SDPA path on a given prompt length.
    """
    qk = 64 if _os.environ.get("QWEN9B_SDPA_QK64") == "1" else 128
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=qk,
        k_chunk_size=qk,
        exp_approx_mode=False,
    )


def _get_paged_sdpa_decode_program_config(device, max_seq_len):
    """Build SDPAProgramConfig for paged_scaled_dot_product_attention_decode.

    Uses small chunk sizes to avoid L1 OOM on Blackhole.
    Note: do NOT pass compute_kernel_config to paged_sdpa_decode on Blackhole —
    the default is compatible; custom configs can cause incorrect output.
    """
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=32,
        k_chunk_size=64,
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


def _get_qknorm_compute_kernel_config(device):
    """Device compute-kernel config for the fast-path q/k rms_norm calls, gated behind
    QWEN36_ATTN_QKNORM_HIFI2 (default "1" as of step1-F10c phase 2 -- PCC-validated, see the call
    site's comment; set to "0" for the legacy op-default HiFi4).

    Built the way models/common/rmsnorm.py's compute_kernel_config_hifi2 is (HiFi2,
    fp32_dest_acc_en, packer_l1_acc=True), but via ttnn.init_device_compute_kernel_config for the
    device arch, matching models/tt_dit/layers/normalization.py's pattern. gemma4's
    attention/operations.py deliberately keeps HiFi4 for q/k norm accuracy; this HiFi2 override was
    PCC-gated (val_b + compare_logits + gt_check where runnable) before being defaulted on here.
    """
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
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
    # Pre-allocated sharded buffers for paged_update_cache (avoids allocation during trace)
    paged_k_buf=None,
    paged_v_buf=None,
    # Paged attention with page_table (for vLLM integration)
    page_table=None,
    paged_kv_cache_key=None,  # DRAM paged cache [num_blocks, num_kv_heads, block_size, head_dim]
    paged_kv_cache_value=None,  # DRAM paged cache [num_blocks, num_kv_heads, block_size, head_dim]
    # Paged prefill: fill K/V into paged cache, attend via chunked SDPA
    chunk_page_table=None,  # [1, num_blocks_in_chunk] int32 — blocks for this chunk only
    chunk_start_idx=None,  # int — absolute position of this chunk in the full sequence
    chunk_start_idx_tensor=None,  # device tensor [1] int32 — runtime offset for trace-replay (flexible SDPA)
    prefill_progcfg_fn=None,
    # De-interleaved/packed prefill fast path (see weights.py load_attention_weights): when all
    # three are given and T > 1, replaces the q/k/v projection + head-split block below with two
    # matmuls (q, packed k|v) plus ttnn.experimental.nlp_create_qkv_heads. Decode (T == 1) and
    # callers that don't pass these (e.g. other models) keep using the plain path.
    q_deint_weight=None,
    gate_deint_weight=None,
    kv_packed_weight=None,
    # F9: single fused [Q|K|V] weight (weights.py AttentionWeights.qkv_fused). When given (and
    # QWEN36_ATTN_FUSED_QKV != "0"), replaces the q_deint_weight + kv_packed_weight two-matmul
    # step with one matmul + the single-input form of ttnn.experimental.nlp_create_qkv_heads.
    # Falls back to the q_deint_weight/kv_packed_weight two-matmul path when None.
    qkv_fused_weight=None,
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

    # memory_config is only threaded into the extra prefill-only ops added for F3 (fast-path q/k
    # rms_norm, fused rotary + its concats, concatenate_heads, gate sigmoid/multiply, chunked SDPA
    # output) when T > 1. Those ops were never passed memory_config for decode (T == 1) before this
    # change (only the linears/nlp_create_qkv_heads/o_proj were, via the memory_config param
    # directly) -- gating on T > 1 keeps decode byte-for-byte identical to before.
    _prefill_mc = memory_config if T > 1 else None

    # Q projection: 2x wide
    ckc = compute_kernel_config

    def _pc(x_in, w_in):
        if prefill_progcfg_fn is None or T <= 1:
            return None
        return prefill_progcfg_fn(
            T, x_in.shape[-1], w_in.shape[-1], x_in.dtype, w_in.dtype, getattr(ckc, "fp32_dest_acc_en", True)
        )

    # F9: fused single-matmul Q|K|V path, gated on qkv_fused_weight being available and the
    # QWEN36_ATTN_FUSED_QKV env override (default on). Falls back to the F3 two-matmul
    # (q_deint + kv_packed) path when qkv_fused_weight is None or the override is "0".
    _fused_qkv = qkv_fused_weight is not None and _os.environ.get("QWEN36_ATTN_FUSED_QKV", "1") != "0"
    if (
        gate_deint_weight is not None
        and (_fused_qkv or (q_deint_weight is not None and kv_packed_weight is not None))
        and T > 1
        and T % 32 == 0
    ):  # nlp_create_qkv_heads needs a tile-aligned seq
        # Fast path (prefill only): weight columns were de-interleaved once at load time
        # (weights.py load_attention_weights), so head-splitting is a single fused
        # ttnn.experimental.nlp_create_qkv_heads instead of a copying reshape + chunk, a separate
        # gate reshape, two more K/V reshapes, and three transposes.
        if _fused_qkv:
            # F9: one matmul for the fused [Q-block | K-block | V-block] weight (each block
            # head-major, see weights.py) instead of separate q and kv_packed matmuls.
            qkv = ttnn.linear(
                hidden_states,
                qkv_fused_weight,
                compute_kernel_config=ckc,
                memory_config=memory_config,
                program_config=_pc(hidden_states, qkv_fused_weight),
            )  # [B, T, H*Dh + 2*Hkv*Dh]
        else:
            q = ttnn.linear(
                hidden_states,
                q_deint_weight,
                compute_kernel_config=ckc,
                memory_config=memory_config,
                program_config=_pc(hidden_states, q_deint_weight),
            )  # [B, T, H*Dh]
            kv = ttnn.linear(
                hidden_states,
                kv_packed_weight,
                compute_kernel_config=ckc,
                memory_config=memory_config,
                program_config=_pc(hidden_states, kv_packed_weight),
            )  # [B, T, 2*Hkv*Dh]
        gate = ttnn.linear(
            hidden_states,
            gate_deint_weight,
            compute_kernel_config=ckc,
            memory_config=memory_config,
            program_config=_pc(hidden_states, gate_deint_weight),
        )  # [B, T, H*Dh] flat, used as-is later
        if _fused_qkv:
            qkv4 = ttnn.reshape(qkv, [B, 1, T, num_attention_heads * head_dim + 2 * num_key_value_heads * head_dim])
            query_states, key_states, value_states = ttnn.experimental.nlp_create_qkv_heads(
                qkv4,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                transpose_k_heads=False,
                memory_config=memory_config,
            )  # [B,H,T,Dh], [B,Hkv,T,Dh], [B,Hkv,T,Dh]
            ttnn.deallocate(qkv)
        else:
            q4 = ttnn.reshape(q, [B, 1, T, num_attention_heads * head_dim])  # metadata only
            kv4 = ttnn.reshape(kv, [B, 1, T, 2 * num_key_value_heads * head_dim])
            query_states, key_states, value_states = ttnn.experimental.nlp_create_qkv_heads(
                q4,
                kv4,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                transpose_k_heads=False,
                memory_config=memory_config,
            )  # [B,H,T,Dh], [B,Hkv,T,Dh], [B,Hkv,T,Dh]
            ttnn.deallocate(q)
            ttnn.deallocate(kv)

        # Q/K norm on head-major tensors: rms_norm reduces over the last dim (Dh), which is the
        # same dim being normalized whether heads sit at dim 2 (old path, pre-transpose) or dim 1
        # (here) — numerically identical, so no transpose is needed before or after.
        # QWEN36_ATTN_QKNORM_HIFI2 (default "1" as of step1-F10c phase 2, 2026-09-21): the op
        # runs its default HiFi4 only if this is set to "0". Data (val_b T=2048/4096: -1.6/-2.1ms;
        # compare_logits vs logits_after_f10b.pt: PCC=0.999920/0.999810 at T=512/1024, argmax
        # match at T=512/1024/2048/4096; traced_4k demo TTFT 0.162s, no regression) supports
        # defaulting this on -- see patches/step1_f10c_kvbf8_qknorm_lmhead.patch. Set to "0" to
        # restore the legacy HiFi4 q/k rms_norm for A/B.
        _qknorm_ckc = (
            _get_qknorm_compute_kernel_config(device)
            if _os.environ.get("QWEN36_ATTN_QKNORM_HIFI2", "1") != "0"
            else None
        )
        if norm_weights_pre_offset:
            query_states = ttnn.rms_norm(
                query_states,
                weight=q_norm_weight,
                epsilon=norm_eps,
                memory_config=_prefill_mc,
                compute_kernel_config=_qknorm_ckc,
            )
            key_states = ttnn.rms_norm(
                key_states,
                weight=k_norm_weight,
                epsilon=norm_eps,
                memory_config=_prefill_mc,
                compute_kernel_config=_qknorm_ckc,
            )
        else:
            query_states = rms_norm_zero_centered_ttnn(query_states, q_norm_weight, eps=norm_eps)
            key_states = rms_norm_zero_centered_ttnn(key_states, k_norm_weight, eps=norm_eps)
    else:
        qg = ttnn.linear(
            hidden_states,
            q_proj_weight,
            compute_kernel_config=ckc,
            memory_config=memory_config,
            program_config=_pc(hidden_states, q_proj_weight),
        )
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
        key_states = ttnn.linear(
            hidden_states,
            k_proj_weight,
            compute_kernel_config=ckc,
            memory_config=memory_config,
            program_config=_pc(hidden_states, k_proj_weight),
        )
        key_states = ttnn.reshape(key_states, [B, T, num_key_value_heads, head_dim])
        if norm_weights_pre_offset:
            key_states = ttnn.rms_norm(key_states, weight=k_norm_weight, epsilon=norm_eps)
        else:
            key_states = rms_norm_zero_centered_ttnn(key_states, k_norm_weight, eps=norm_eps)
        key_states = ttnn.transpose(key_states, 1, 2)

        # V projection + transpose to [B, H_kv, T, D]
        value_states = ttnn.linear(
            hidden_states,
            v_proj_weight,
            compute_kernel_config=ckc,
            memory_config=memory_config,
            program_config=_pc(hidden_states, v_proj_weight),
        )
        value_states = ttnn.reshape(value_states, [B, T, num_key_value_heads, head_dim])
        value_states = ttnn.transpose(value_states, 1, 2)

    # RoPE — for decode (T==1), use reshape (metadata-only) instead of unsqueeze (data movement).
    # For prefill (T>1, tile-aligned), use the fused rotary_embedding_hf op
    # (apply_rotary_pos_emb_fused): ~4 ops/tensor (slice + rotary + slice + concat) instead of the
    # ~10-op hand-rolled rotate_half. QWEN36_ROPE_LEGACY=1 forces the legacy elementwise path for A/B.
    _rope_legacy = _os.environ.get("QWEN36_ROPE_LEGACY") == "1"
    if T == 1 and len(cos.shape) == 3:
        cos_4d = ttnn.reshape(cos, [cos.shape[0], 1, 1, cos.shape[-1]])
        sin_4d = ttnn.reshape(sin, [sin.shape[0], 1, 1, sin.shape[-1]])
        query_states, key_states = apply_rotary_pos_emb_ttnn(query_states, key_states, cos_4d, sin_4d)
    elif T > 1 and T % 32 == 0 and not _rope_legacy:
        query_states, key_states = apply_rotary_pos_emb_fused(
            query_states, key_states, cos, sin, memory_config=_prefill_mc
        )
    else:
        query_states, key_states = apply_rotary_pos_emb_ttnn(query_states, key_states, cos, sin)

    # KV cache handling
    _use_sdpa_decode = False
    _paged_sdpa_done = False
    if paged_kv_cache_key is not None and page_table is not None and T > 1 and chunk_page_table is not None:
        # Paged prefill: fill K/V into paged cache, then chunked SDPA.
        # Q/K/V stay bfloat16 by default — no typecast. Production models (Qwen3_VL) typecast to
        # bfloat8_b, but on Blackhole P150 this previously caused L1 clashes in downstream ops
        # (head concat, gate multiply, output projection) whose programs were compiled
        # for bfloat16. The core fix (paged KV) eliminates the growing-concat L1 issue;
        # QWEN36_ATTN_KV_BF8=1 (default "0", see below) re-enables the bfloat8_b typecast for
        # A/B testing; the downstream concat/gate/o_proj path is left untouched (it reads
        # whatever dtype SDPA emits), matching the "leave the SDPA output dtype as it comes
        # out" note this flag was added under.
        _kv_bf8 = _os.environ.get("QWEN36_ATTN_KV_BF8", "0") == "1"

        # Slice K/V to page_len to handle tile-padded tensors.
        block_size_cache = paged_kv_cache_key.shape[2]
        page_len = chunk_page_table.shape[1] * block_size_cache
        key_fill = key_states[:, :, :page_len, :] if page_len < key_states.shape[2] else key_states
        value_fill = value_states[:, :, :page_len, :] if page_len < value_states.shape[2] else value_states

        if _kv_bf8:
            # Typecast to whatever dtype the paged cache itself was allocated with (matches
            # tt_transformers/tt/attention.py's k_heads_1KSD_8b/v_heads_1VSD_8b pattern,
            # ~1141/1153: `ttnn.typecast(k_heads, dtype=keys.dtype)`). The cache is bfloat16
            # unless model.py's allocate_kv_caches (~2743) was run with QWEN_SDPA_BF8=1, so
            # this flag only yields an actual bfloat8_b cast when QWEN_SDPA_BF8=1 too --
            # otherwise it typecasts bfloat16 -> bfloat16 (a harmless no-op copy).
            key_fill = ttnn.typecast(key_fill, dtype=paged_kv_cache_key.dtype)
            value_fill = ttnn.typecast(value_fill, dtype=paged_kv_cache_value.dtype)

        ttnn.experimental.paged_fill_cache(paged_kv_cache_key, key_fill, chunk_page_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(paged_kv_cache_value, value_fill, chunk_page_table, batch_idx=0)
        ttnn.deallocate(key_states)
        ttnn.deallocate(value_states)

        # QWEN36_ATTN_KV_BF8 also typecasts Q to bfloat8_b right before SDPA (matches
        # tt_transformers/tt/attention.py ~1218: `ttnn.typecast(q_heads, dtype=... or
        # ttnn.bfloat8_b)`). query_states itself is left unmodified -- it's deallocated
        # unconditionally further below -- so this is a separate tensor used only here.
        _q_for_sdpa = ttnn.typecast(query_states, dtype=ttnn.bfloat8_b) if _kv_bf8 else query_states

        if chunk_start_idx_tensor is not None:
            # Flexible chunked SDPA: chunk_start_idx is a runtime device tensor and the
            # program config is fixed (q/k_chunk=64), so a single captured trace replays
            # for every chunk position (chunk-outer per-chunk prefill).
            attn_output = ttnn.transformer.chunked_scaled_dot_product_attention(
                _q_for_sdpa,
                paged_kv_cache_key,
                paged_kv_cache_value,
                page_table,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                scale=scaling,
                memory_config=_prefill_mc,
                program_config=_get_flexible_sdpa_program_config(device),
                compute_kernel_config=_get_sdpa_compute_kernel_config(),
            )
        else:
            attn_output = ttnn.transformer.chunked_scaled_dot_product_attention(
                _q_for_sdpa,
                paged_kv_cache_key,
                paged_kv_cache_value,
                page_table,
                chunk_start_idx,
                scale=scaling,
                memory_config=_prefill_mc,
                program_config=_get_sdpa_program_config(device, T, chunk_start_idx=chunk_start_idx),
                compute_kernel_config=_get_sdpa_compute_kernel_config(),
            )

        new_key = paged_kv_cache_key
        new_value = paged_kv_cache_value
        _paged_sdpa_done = True
        S_total = paged_kv_cache_key.shape[0] * paged_kv_cache_key.shape[2]
        is_causal = False
        segmented_attn_mask = None
    elif page_table is not None and paged_kv_cache_key is not None and T == 1:
        # Paged attention decode: write K/V via page_table, attend via paged SDPA.
        # paged_update_cache with page_table expects input [1, B, num_kv_heads, head_dim],
        # where page_table.shape[0] == input.shape[1] == B.
        # This is different from the non-paged path which uses [1, B*num_kv_heads, T, head_dim].
        k_for_paged = ttnn.reshape(key_states, [1, B, num_key_value_heads, head_dim])
        v_for_paged = ttnn.reshape(value_states, [1, B, num_key_value_heads, head_dim])
        # Shard to L1 (HEIGHT_SHARDED on B cores) for paged_update_cache
        _shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(B - 1, 0))})
        _shard_spec = ttnn.ShardSpec(_shard_grid, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR)
        _sharded_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec)
        k_sharded = ttnn.to_memory_config(k_for_paged, _sharded_mc)
        v_sharded = ttnn.to_memory_config(v_for_paged, _sharded_mc)
        # Write K/V into paged cache via page_table
        ttnn.experimental.paged_update_cache(
            paged_kv_cache_key, k_sharded, update_idxs_tensor=cur_pos_tensor, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            paged_kv_cache_value, v_sharded, update_idxs_tensor=cur_pos_tensor, page_table=page_table
        )
        # Paged SDPA decode
        q_decode = ttnn.transpose(query_states, 1, 2)  # [B, H_q, 1, D] -> [B, 1, H_q, D] = [1, B, H_q, D] for B=1
        # paged_scaled_dot_product_attention_decode requires Q in DRAM when not sharded
        # (sdpa_decode_device_operation.cpp:89). transpose preserves L1 layout from query_states.
        q_decode = ttnn.to_memory_config(q_decode, ttnn.DRAM_MEMORY_CONFIG)
        attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_decode,
            paged_kv_cache_key,
            paged_kv_cache_value,
            cur_pos_tensor=cur_pos_tensor,
            page_table_tensor=page_table,
            is_causal=True,
            scale=scaling,
            program_config=_get_paged_sdpa_decode_program_config(
                device, paged_kv_cache_key.shape[0] * paged_kv_cache_key.shape[2]
            ),
        )
        attn_output = ttnn.transpose(attn_output, 1, 2)  # back to [B, H_q, 1, D]
        new_key = paged_kv_cache_key
        new_value = paged_kv_cache_value
        _paged_sdpa_done = True  # Skip the bottom SDPA section (attn_output already computed)
        S_total = paged_kv_cache_key.shape[0] * paged_kv_cache_key.shape[2]
        is_causal = False
        segmented_attn_mask = None
    elif cur_pos_tensor is not None and kv_cache_key is not None and T == 1:
        # Trace-compatible SDPA decode: write K/V at dynamic position, attend up to cur_pos.
        # Uses paged_update_cache (takes device tensor for index, trace-compatible) +
        # sdpa_decode (reads cache[:, :, :cur_pos+1, :], no mask needed).
        #
        # paged_update_cache expects: input [1, N, seq, D] HEIGHT_SHARDED, cache [N, 1, max_seq, D].
        # N = batch * num_kv_heads. Pre-allocated sharded buffers (paged_k_buf, paged_v_buf)
        # are provided by the caller to avoid allocating during trace capture.
        N = B * num_key_value_heads
        max_seq = kv_cache_key.shape[2]
        k_for_cache = ttnn.reshape(key_states, [1, N, T, head_dim])
        v_for_cache = ttnn.reshape(value_states, [1, N, T, head_dim])
        # Convert to sharded for paged_update_cache. Use pre-allocated sharded buffers
        # when available (trace path) to avoid allocation during trace capture.
        if paged_k_buf is not None:
            k_sharded = ttnn.interleaved_to_sharded(
                k_for_cache, paged_k_buf.memory_config(), preallocated_output=paged_k_buf
            )
            v_sharded = ttnn.interleaved_to_sharded(
                v_for_cache, paged_v_buf.memory_config(), preallocated_output=paged_v_buf
            )
        else:
            # Eager path: allocate on the fly (not inside trace)
            _shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(N - 1, 0))})
            _shard_spec = ttnn.ShardSpec(_shard_grid, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR)
            _sharded_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec)
            k_sharded = ttnn.to_memory_config(k_for_cache, _sharded_mc)
            v_sharded = ttnn.to_memory_config(v_for_cache, _sharded_mc)
        kv_cache_key_reshaped = ttnn.reshape(kv_cache_key, [N, 1, max_seq, head_dim])
        kv_cache_value_reshaped = ttnn.reshape(kv_cache_value, [N, 1, max_seq, head_dim])
        ttnn.experimental.paged_update_cache(kv_cache_key_reshaped, k_sharded, update_idxs_tensor=cur_pos_tensor)
        ttnn.experimental.paged_update_cache(kv_cache_value_reshaped, v_sharded, update_idxs_tensor=cur_pos_tensor)
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

    # Fused scaled dot-product attention (skipped if paged SDPA already computed above)
    if _paged_sdpa_done:
        pass  # attn_output already set by paged SDPA decode path
    elif _use_sdpa_decode:
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
        attn_output = ttnn.transformer.concatenate_heads(attn_output, memory_config=_prefill_mc)
    else:
        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, [B, T, num_attention_heads * head_dim])

    # Apply sigmoid gate. F6-E: prefill (T>1) fuses sigmoid(gate) into the multiply's
    # second-operand activation (one op instead of two); decode (T==1, _prefill_mc is always None
    # there) always takes the legacy two-op path unchanged. QWEN36_ATTN_GATE_FUSED=0 restores the
    # legacy two-op path at any T.
    if T > 1 and _os.environ.get("QWEN36_ATTN_GATE_FUSED", "1") != "0":
        attn_output = ttnn.multiply(
            attn_output, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], memory_config=_prefill_mc
        )
    else:
        gate = ttnn.sigmoid(gate, memory_config=_prefill_mc)
        attn_output = ttnn.multiply(attn_output, gate, memory_config=_prefill_mc)
    ttnn.deallocate(gate)

    # Output projection
    attn_output = ttnn.linear(
        attn_output,
        o_proj_weight,
        compute_kernel_config=ckc,
        memory_config=memory_config,
        program_config=_pc(attn_output, o_proj_weight),
    )

    return attn_output, new_key, new_value

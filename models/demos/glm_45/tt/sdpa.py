# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Scaled Dot Product Attention (SDPA) implementation aligned with HF reference.

This module provides a custom SDPA implementation that supports:
- Sliding window/causal masking
- KV cache management for efficient autoregressive generation
- Group query attention (GQA) with key-value head broadcasting
- Stable softmax computation for numerical stability
"""

import ttnn


def softmax(x: ttnn.Tensor, stable=False):
    """
    Performs Softmax on a TTNN tensor with optional numerical stability.

    Args:
        x: Input tensor of shape (batch, heads, seq_len, kv_len)
        stable: If True, subtracts max value before exp for numerical stability

    Returns:
        Softmax-normalized tensor with same shape as input

    Note:
        The hardcoded dimension 3 is the last dimension (kv_len) which is the
        standard softmax dimension for attention scores.
    """
    if stable:
        sumsW = ttnn.max(x, -1)
        sumsW = ttnn.unsqueeze(sumsW, -1)
        z = ttnn.subtract(x, sumsW)  # x-max(x)
    else:
        z = x
    numerator = ttnn.exp(z)  # exp(z)
    # Hardcoded dimension 3 represents the kv_len dimension for attention scores
    denom1 = ttnn.sum(numerator, 3)  # torch.sum(x, 3)
    denom = ttnn.reciprocal(denom1)
    denom = ttnn.unsqueeze(denom, -1)
    output = ttnn.multiply(numerator, denom)

    return output


def sdpa(
    tt_q: ttnn.Tensor,
    tt_k: ttnn.Tensor,
    tt_v: ttnn.Tensor,
    tt_sink: ttnn.Tensor,
    sm_scale: float,
    tt_mask: ttnn.Tensor = None,
    tt_cache: ttnn.Tensor = None,
    position_idx: int = None,
) -> ttnn.Tensor:
    """
    Perform scaled dot-product attention with causal/sliding window masking.

    This implementation supports Group Query Attention (GQA) where the number of
    key-value heads (nkv) may be less than the number of query heads (nh).

    Args:
        tt_q: Query tensor of shape (num_tokens, 1, nh, dim)
        tt_k: Key tensor of shape (num_tokens, nkv, dim)
        tt_v: Value tensor of shape (num_tokens, nkv, dim)
        tt_sink: Deprecated/unused. Present for API compatibility.
        sm_scale: Scaling factor for attention scores (typically 1/sqrt(head_dim))
        tt_mask: Optional attention mask for sliding window, shape (1, nh, num_tokens, kv_len)
        tt_cache: Optional KV cache tuple [k_cache, v_cache] for autoregressive generation
        position_idx: Current position in sequence for cache slicing (used during prefill)

    Returns:
        Tuple of:
            - out: Attention output tensor, shape [1, 1, num_tokens, hidden_dim]
            - tt_cache: Updated KV cache [k_cache, v_cache]

    Note:
        The hardcoded dimensions in reshapes follow attention tensor layouts:
        - dim=2 is the sequence/cache length dimension
        - dim=3 is the feature/hidden dimension
    """

    assert tt_q.shape[-1] == tt_k.shape[-1] == tt_v.shape[-1], "Head dimension mismatch between Q, K, V"

    num_tokens, _, nh, dim = tt_q.shape
    _, nkv, _ = tt_k.shape

    # Reshape queries for GQA: group query heads by KV heads
    tt_q = ttnn.permute(tt_q, [1, 2, 0, 3])  # (1, nh, num_tokens, dim)
    tt_q = ttnn.reshape(tt_q, [nkv, nh // nkv, num_tokens, dim])

    # Reshape keys and values for cache management
    tt_k = ttnn.transpose(tt_k, 0, 1)  # [nkv, num_tokens, dim]
    tt_k = ttnn.unsqueeze(tt_k, 1)  # [nkv, 1, num_tokens, dim]

    tt_v = ttnn.transpose(tt_v, 0, 1)  # [nkv, num_tokens, dim]
    tt_v = ttnn.unsqueeze(tt_v, 1)  # [nkv, 1, num_tokens, dim]

    # KV Cache management
    if tt_cache is None:
        tt_cache = [tt_k, tt_v]  # Initialize cache for first token
    else:
        tt_k_back, tt_v_back = tt_cache

        # Slice cache to position_idx if provided (useful during prefill)
        if position_idx is not None:
            assert position_idx <= tt_k_back.shape[2], "position_idx exceeds cache length"
            tt_k_back = tt_k_back[:, :, :position_idx, :]
            tt_v_back = tt_v_back[:, :, :position_idx, :]

        # Concatenate cached tensors with current K/V along sequence dimension (dim=2)
        tt_k = ttnn.concat([tt_k_back, tt_k], dim=2)  # (nkv, 1, cache_len + num_tokens, dim)
        tt_v = ttnn.concat([tt_v_back, tt_v], dim=2)  # (nkv, 1, cache_len + num_tokens, dim)

        tt_cache = [tt_k, tt_v]  # Update cache with concatenated tensors

    kv_len = tt_k.shape[2]  # Total sequence length including cache

    # Broadcast K/V across query heads for GQA
    tt_k = ttnn.repeat(tt_k, [1, nh // nkv, 1, 1])  # (nkv, nh // nkv, kv_len, dim)
    tt_k = ttnn.transpose(tt_k, -1, -2)  # (nkv, nh // nkv, dim, kv_len) for matmul

    tt_v = ttnn.repeat(tt_v, [1, nh // nkv, 1, 1])  # (nkv, nh // nkv, kv_len, dim)

    # Compute attention scores: Q @ K^T
    tt_qk = ttnn.matmul(tt_q, tt_k)  # (nkv, nh // nkv, num_tokens, kv_len)
    tt_qk *= sm_scale  # Scale by 1/sqrt(head_dim)

    # Apply sliding window mask if provided
    if tt_mask is not None:
        tt_qk += tt_mask  # Masked positions get -inf

    # Softmax - using custom implementation for stability
    tt_qk = softmax(tt_qk, stable=True)  # (nkv, nh // nkv, num_tokens, kv_len)

    # Compute attention output: softmax(QK^T) @ V
    out = ttnn.matmul(tt_qk, tt_v)  # (nkv, nh // nkv, num_tokens, dim)
    out = ttnn.reshape(out, [1, nh, num_tokens, dim])

    # Concatenate attention heads back into single hidden dimension
    out = ttnn.experimental.nlp_concat_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, num_tokens, dim * nh]

    return out, tt_cache

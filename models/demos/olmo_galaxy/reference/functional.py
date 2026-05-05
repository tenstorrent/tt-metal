# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Standalone functional implementations for OLMo-3.1-32B verification.
These are reference implementations for TTNN PCC testing.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ==============================================================================
# RMSNorm
# ==============================================================================
def rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm forward pass.

    Args:
        x: Input tensor [..., hidden_size]
        weight: Learnable scale parameter [hidden_size]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor [..., hidden_size]
    """
    input_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * weight).to(input_dtype)


# ==============================================================================
# YaRN RoPE (OLMo-3.1-32B uses YaRN scaling)
# ==============================================================================
def yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> float:
    """Find the correction dimension for YaRN interpolation."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    """Find the correction range for YaRN interpolation."""
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(
    min_val: float,
    max_val: float,
    dim: int,
) -> torch.Tensor:
    """Create a linear ramp mask for YaRN interpolation."""
    if min_val == max_val:
        max_val += 0.001  # Prevent division by zero
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """Get the mscale value for YaRN attention scaling."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def compute_yarn_freqs(
    dim: int,
    max_position_embeddings: int,
    base: float = 10000.0,
    scaling_factor: float = 1.0,
    original_max_position_embeddings: int = 8192,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    attention_factor: float = 1.0,
) -> Tuple[torch.Tensor, float]:
    """Compute YaRN frequency tensor for OLMo-style RoPE.

    Args:
        dim: Head dimension
        max_position_embeddings: Maximum sequence length
        base: RoPE theta base
        scaling_factor: YaRN scaling factor
        original_max_position_embeddings: Original context length before extension
        beta_fast: YaRN beta_fast parameter
        beta_slow: YaRN beta_slow parameter
        attention_factor: Attention scaling factor (maps to mscale)

    Returns:
        Tuple of (inv_freq tensor, mscale value)
    """
    # Compute base inv_freq
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # Find correction range
    low, high = yarn_find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings)

    # Create interpolation mask
    inv_freq_mask = 1 - yarn_linear_ramp_mask(low, high, dim // 2)

    # Apply YaRN interpolation
    inv_freq_extrapolated = inv_freq / scaling_factor
    inv_freq_interpolated = inv_freq / scaling_factor
    inv_freq = inv_freq_interpolated * (1 - inv_freq_mask) + inv_freq_extrapolated * inv_freq_mask

    # Compute mscale (attention_factor in OLMo maps to this)
    mscale = attention_factor

    return inv_freq, mscale


def precompute_yarn_freqs_cis(
    dim: int,
    end: int,
    theta: float = 500000.0,
    scaling_factor: float = 8.0,
    original_max_position_embeddings: int = 8192,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    attention_factor: float = 1.2079441541679836,
) -> Tuple[torch.Tensor, float]:
    """Precompute YaRN RoPE frequencies for OLMo-3.1-32B.

    Args:
        dim: Head dimension (128 for OLMo)
        end: Maximum sequence length
        theta: RoPE theta (500000 for OLMo)
        scaling_factor: YaRN factor (8.0 for OLMo)
        original_max_position_embeddings: Original context (8192 for OLMo)
        beta_fast: YaRN beta_fast (32.0 for OLMo)
        beta_slow: YaRN beta_slow (1.0 for OLMo)
        attention_factor: YaRN attention_factor (1.2079 for OLMo)

    Returns:
        Tuple of (freqs_cis complex tensor, mscale)
    """
    inv_freq, mscale = compute_yarn_freqs(
        dim=dim,
        max_position_embeddings=end,
        base=theta,
        scaling_factor=scaling_factor,
        original_max_position_embeddings=original_max_position_embeddings,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        attention_factor=attention_factor,
    )

    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return freqs_cis, mscale


def precompute_freqs_cos_sin(
    dim: int,
    end: int,
    theta: float = 500000.0,
    use_yarn: bool = True,
    scaling_factor: float = 8.0,
    original_max_position_embeddings: int = 8192,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    attention_factor: float = 1.2079441541679836,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Precompute cos/sin for RoPE.

    Returns:
        Tuple of (cos, sin, mscale) tensors
    """
    if use_yarn:
        freqs_cis, mscale = precompute_yarn_freqs_cis(
            dim=dim,
            end=end,
            theta=theta,
            scaling_factor=scaling_factor,
            original_max_position_embeddings=original_max_position_embeddings,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            attention_factor=attention_factor,
        )
        cos = freqs_cis.real
        sin = freqs_cis.imag
    else:
        # Standard RoPE (for comparison)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(end, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        mscale = 1.0

    return cos, sin, mscale


def apply_rotary_emb_complex(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings using complex multiplication.

    Args:
        xq: Query tensor [batch, seq_len, n_heads, head_dim]
        xk: Key tensor [batch, seq_len, n_kv_heads, head_dim]
        freqs_cis: Precomputed frequencies [seq_len, head_dim//2]

    Returns:
        Rotated (xq, xk) tensors
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs_cis for broadcasting
    ndim = xq_.ndim
    shape = [1] * ndim
    shape[1] = freqs_cis.shape[0]  # seq_len
    shape[-1] = freqs_cis.shape[1]  # head_dim // 2
    freqs_cis = freqs_cis.view(*shape)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dimensions."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_cos_sin(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings using cos/sin (HuggingFace style).

    Args:
        q: Query tensor [batch, seq_len, n_heads, head_dim]
        k: Key tensor [batch, seq_len, n_kv_heads, head_dim]
        cos: Cosine values [seq_len, head_dim//2] or [batch, seq_len, head_dim//2]
        sin: Sine values [seq_len, head_dim//2] or [batch, seq_len, head_dim//2]
        position_ids: Optional position indices

    Returns:
        Rotated (q, k) tensors
    """
    # Expand cos/sin to full head_dim
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    # Reshape for broadcasting
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(2)  # [batch, seq_len, 1, head_dim]
        sin = sin[position_ids].unsqueeze(2)
    else:
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(2)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# ==============================================================================
# Attention (GQA with optional sliding window)
# ==============================================================================
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match Q heads for GQA.

    Args:
        x: KV tensor [batch, seq_len, n_kv_heads, head_dim]
        n_rep: Number of times to repeat each KV head

    Returns:
        Repeated tensor [batch, seq_len, n_heads, head_dim]
    """
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def create_sliding_window_mask(
    seq_len: int,
    sliding_window: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> torch.Tensor:
    """Create a sliding window causal mask.

    Args:
        seq_len: Sequence length
        sliding_window: Window size (tokens outside window are masked)
        dtype: Output dtype
        device: Output device

    Returns:
        Mask tensor [1, 1, seq_len, seq_len]
    """
    # Start with causal mask
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device), diagonal=1)

    # Apply sliding window (mask tokens more than window_size positions ago)
    if sliding_window is not None and sliding_window > 0:
        sliding_mask = torch.tril(
            torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device), diagonal=-sliding_window - 1
        )
        mask = mask + sliding_mask

    return mask.unsqueeze(0).unsqueeze(0)


def attention_forward(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_heads: int = 40,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    sliding_window: Optional[int] = None,
    mscale: float = 1.0,
) -> torch.Tensor:
    """Standalone attention forward pass for OLMo-3.1-32B.

    Args:
        x: Input tensor [batch, seq_len, hidden_size]
        wq: Q projection weight [n_heads * head_dim, hidden_size]
        wk: K projection weight [n_kv_heads * head_dim, hidden_size]
        wv: V projection weight [n_kv_heads * head_dim, hidden_size]
        wo: Output projection weight [hidden_size, n_heads * head_dim]
        cos: Cosine values for RoPE [seq_len, head_dim//2]
        sin: Sine values for RoPE [seq_len, head_dim//2]
        n_heads: Number of query heads (40 for OLMo)
        n_kv_heads: Number of KV heads (8 for OLMo)
        head_dim: Dimension per head (128 for OLMo)
        sliding_window: Optional sliding window size (4096 for sliding layers)
        mscale: Attention scale from YaRN

    Returns:
        Output tensor [batch, seq_len, hidden_size]
    """
    batch, seq_len, hidden = x.shape
    n_rep = n_heads // n_kv_heads

    # QKV projections (no bias in OLMo)
    q = F.linear(x, wq)  # [batch, seq_len, n_heads * head_dim]
    k = F.linear(x, wk)  # [batch, seq_len, n_kv_heads * head_dim]
    v = F.linear(x, wv)  # [batch, seq_len, n_kv_heads * head_dim]

    # Reshape
    q = q.view(batch, seq_len, n_heads, head_dim)
    k = k.view(batch, seq_len, n_kv_heads, head_dim)
    v = v.view(batch, seq_len, n_kv_heads, head_dim)

    # Apply RoPE
    q, k = apply_rotary_emb_cos_sin(q, k, cos[:seq_len], sin[:seq_len])

    # Repeat KV for GQA
    k = repeat_kv(k, n_rep)  # [batch, seq_len, n_heads, head_dim]
    v = repeat_kv(v, n_rep)

    # Transpose for attention
    q = q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Scaled dot-product attention with optional sliding window
    scale = (head_dim**-0.5) * mscale
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Create and apply mask
    mask = create_sliding_window_mask(seq_len, sliding_window, dtype=scores.dtype, device=scores.device)
    scores = scores + mask

    # Softmax and attention
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    output = torch.matmul(attn_weights, v)

    # Reshape and project
    output = output.transpose(1, 2).contiguous().view(batch, seq_len, n_heads * head_dim)
    return F.linear(output, wo)


def attention_forward_decode(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    start_pos: int,
    n_heads: int = 40,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    sliding_window: Optional[int] = None,
    mscale: float = 1.0,
    q_norm_weight: Optional[torch.Tensor] = None,
    k_norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention forward pass for decode mode with KV cache.

    Args:
        x: Input tensor [batch, 1, hidden_size] (single token)
        wq, wk, wv, wo: Projection weights
        cos, sin: RoPE embeddings
        cache_k: KV cache for keys [batch, max_seq, n_kv_heads, head_dim]
        cache_v: KV cache for values [batch, max_seq, n_kv_heads, head_dim]
        start_pos: Current position in sequence
        n_heads: Number of Q heads
        n_kv_heads: Number of KV heads
        head_dim: Head dimension
        sliding_window: Sliding window size (None for full attention)
        mscale: YaRN attention scaling factor
        q_norm_weight: Optional QK-norm weight for Q [n_heads * head_dim]
        k_norm_weight: Optional QK-norm weight for K [n_kv_heads * head_dim]
        norm_eps: Epsilon for QK-norm

    Returns:
        Tuple of (output, updated_cache_k, updated_cache_v)
    """
    batch, seq_len, hidden = x.shape
    assert seq_len == 1, f"Decode mode requires seq_len=1, got {seq_len}"
    n_rep = n_heads // n_kv_heads

    # QKV projections
    q = F.linear(x, wq)  # [batch, 1, n_heads * head_dim]
    k = F.linear(x, wk)  # [batch, 1, n_kv_heads * head_dim]
    v = F.linear(x, wv)

    # Apply QK-norm on flat Q/K before head splitting (OLMo style)
    if q_norm_weight is not None:
        q = rmsnorm_forward(q, q_norm_weight, norm_eps)
    if k_norm_weight is not None:
        k = rmsnorm_forward(k, k_norm_weight, norm_eps)

    q = q.view(batch, 1, n_heads, head_dim)
    k = k.view(batch, 1, n_kv_heads, head_dim)
    v = v.view(batch, 1, n_kv_heads, head_dim)

    # Apply RoPE at current position
    q, k = apply_rotary_emb_cos_sin(q, k, cos[start_pos : start_pos + 1], sin[start_pos : start_pos + 1])

    # Update KV cache
    cache_k = cache_k.clone()
    cache_v = cache_v.clone()
    cache_k[:, start_pos : start_pos + 1] = k
    cache_v[:, start_pos : start_pos + 1] = v

    # Get cached keys/values up to current position
    keys = cache_k[:, : start_pos + 1]  # [batch, start_pos+1, n_kv_heads, head_dim]
    values = cache_v[:, : start_pos + 1]

    # Repeat KV for GQA
    keys = repeat_kv(keys, n_rep)
    values = repeat_kv(values, n_rep)

    # Transpose for attention
    q = q.transpose(1, 2)  # [batch, n_heads, 1, head_dim]
    keys = keys.transpose(1, 2)  # [batch, n_heads, kv_len, head_dim]
    values = values.transpose(1, 2)

    # Attention with YaRN mscale
    scale = (head_dim**-0.5) * mscale
    scores = torch.matmul(q, keys.transpose(-2, -1)) * scale  # [batch, n_heads, 1, kv_len]

    # Apply sliding window mask if needed
    if sliding_window is not None:
        kv_len = keys.shape[2]
        positions = torch.arange(kv_len, device=scores.device)
        distance = start_pos - positions
        mask = torch.where(
            distance >= sliding_window,
            torch.full_like(distance, float("-inf"), dtype=scores.dtype),
            torch.zeros_like(distance, dtype=scores.dtype),
        )
        scores = scores + mask.view(1, 1, 1, kv_len)

    # Softmax and attention
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    output = torch.matmul(attn_weights, values)

    # Reshape and project
    output = output.transpose(1, 2).contiguous().view(batch, 1, n_heads * head_dim)
    output = F.linear(output, wo)

    return output, cache_k, cache_v


def init_kv_cache(
    batch_size: int,
    max_seq_len: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Initialize empty KV cache tensors.

    Args:
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        n_kv_heads: Number of KV heads
        head_dim: Head dimension
        dtype: Data type
        device: Device

    Returns:
        Tuple of (cache_k, cache_v) tensors
    """
    cache_k = torch.zeros(batch_size, max_seq_len, n_kv_heads, head_dim, dtype=dtype, device=device)
    cache_v = torch.zeros(batch_size, max_seq_len, n_kv_heads, head_dim, dtype=dtype, device=device)
    return cache_k, cache_v


# ==============================================================================
# SwiGLU MLP
# ==============================================================================
def swiglu_mlp_forward(
    x: torch.Tensor,
    w1: torch.Tensor,  # gate_proj
    w2: torch.Tensor,  # down_proj
    w3: torch.Tensor,  # up_proj
) -> torch.Tensor:
    """SwiGLU MLP forward pass.

    OLMo uses SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        x: Input tensor [batch, seq_len, hidden_size]
        w1: Gate projection weight [intermediate_size, hidden_size]
        w2: Down projection weight [hidden_size, intermediate_size]
        w3: Up projection weight [intermediate_size, hidden_size]

    Returns:
        Output tensor [batch, seq_len, hidden_size]
    """
    gate = F.silu(F.linear(x, w1))  # gate_proj with SiLU activation
    up = F.linear(x, w3)  # up_proj
    return F.linear(gate * up, w2)  # down_proj


# ==============================================================================
# Decoder Block
# ==============================================================================
def decoder_block_forward(
    x: torch.Tensor,
    attention_norm_weight: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_heads: int = 40,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    sliding_window: Optional[int] = None,
    mscale: float = 1.0,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """OLMo decoder block forward pass (pre-norm architecture).

    Args:
        x: Input tensor [batch, seq_len, hidden_size]
        attention_norm_weight: Input layernorm weight
        ffn_norm_weight: Post-attention layernorm weight
        wq, wk, wv, wo: Attention weights
        w1, w2, w3: MLP weights
        cos, sin: RoPE embeddings
        n_heads: Number of Q heads (40)
        n_kv_heads: Number of KV heads (8)
        head_dim: Head dimension (128)
        sliding_window: Sliding window size (4096 or None)
        mscale: YaRN mscale
        norm_eps: Norm epsilon

    Returns:
        Output tensor [batch, seq_len, hidden_size]
    """
    # Pre-norm attention
    h = x
    h_normed = rmsnorm_forward(h, attention_norm_weight, norm_eps)
    attn_out = attention_forward(
        h_normed,
        wq,
        wk,
        wv,
        wo,
        cos,
        sin,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        mscale=mscale,
    )
    h = h + attn_out

    # Pre-norm MLP
    h_normed = rmsnorm_forward(h, ffn_norm_weight, norm_eps)
    mlp_out = swiglu_mlp_forward(h_normed, w1, w2, w3)
    h = h + mlp_out

    return h


def decoder_block_forward_decode(
    x: torch.Tensor,
    attention_norm_weight: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    start_pos: int,
    n_heads: int = 40,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    sliding_window: Optional[int] = None,
    mscale: float = 1.0,
    norm_eps: float = 1e-6,
    q_norm_weight: Optional[torch.Tensor] = None,
    k_norm_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """OLMo decoder block forward (decode mode with KV cache).

    Returns:
        Tuple of (output, updated_cache_k, updated_cache_v)
    """
    h = x
    h_normed = rmsnorm_forward(h, attention_norm_weight, norm_eps)
    attn_out, cache_k, cache_v = attention_forward_decode(
        h_normed,
        wq,
        wk,
        wv,
        wo,
        cos,
        sin,
        cache_k,
        cache_v,
        start_pos,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        mscale=mscale,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        norm_eps=norm_eps,
    )
    h = h + attn_out

    h_normed = rmsnorm_forward(h, ffn_norm_weight, norm_eps)
    mlp_out = swiglu_mlp_forward(h_normed, w1, w2, w3)
    h = h + mlp_out

    return h, cache_k, cache_v


# ==============================================================================
# Embedding
# ==============================================================================
def embedding_forward(
    input_ids: torch.Tensor,
    embed_weight: torch.Tensor,
) -> torch.Tensor:
    """Token embedding lookup.

    Args:
        input_ids: Token IDs [batch, seq_len]
        embed_weight: Embedding weight [vocab_size, hidden_size]

    Returns:
        Embeddings [batch, seq_len, hidden_size]
    """
    return F.embedding(input_ids, embed_weight)


# ==============================================================================
# LM Head
# ==============================================================================
def lm_head_forward(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> torch.Tensor:
    """Language model head (logits projection).

    Args:
        hidden_states: Hidden states [batch, seq_len, hidden_size]
        lm_head_weight: LM head weight [vocab_size, hidden_size]

    Returns:
        Logits [batch, seq_len, vocab_size]
    """
    return F.linear(hidden_states, lm_head_weight)


# ==============================================================================
# Full Model Forward (for golden output generation)
# ==============================================================================
def olmo_forward(
    input_ids: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    n_layers: int = 64,
    n_heads: int = 40,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    sliding_window: int = 4096,
    norm_eps: float = 1e-6,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    mscale: float = 1.0,
) -> torch.Tensor:
    """Full OLMo-3.1-32B forward pass.

    Args:
        input_ids: Token IDs [batch, seq_len]
        state_dict: Model weights dictionary
        n_layers: Number of transformer layers (64)
        n_heads: Number of Q heads (40)
        n_kv_heads: Number of KV heads (8)
        head_dim: Head dimension (128)
        sliding_window: Sliding window size (4096)
        norm_eps: Norm epsilon (1e-6)
        cos, sin: Precomputed RoPE embeddings
        mscale: YaRN mscale

    Returns:
        Logits [batch, seq_len, vocab_size]
    """
    # Embedding
    h = embedding_forward(input_ids, state_dict["model.embed_tokens.weight"])

    # Layer types: 3 sliding + 1 full, repeated
    layer_types = []
    for i in range(n_layers):
        if (i + 1) % 4 == 0:
            layer_types.append(None)  # Full attention
        else:
            layer_types.append(sliding_window)  # Sliding window

    # Transformer layers
    for i in range(n_layers):
        prefix = f"model.layers.{i}"
        h = decoder_block_forward(
            h,
            attention_norm_weight=state_dict[f"{prefix}.input_layernorm.weight"],
            ffn_norm_weight=state_dict[f"{prefix}.post_attention_layernorm.weight"],
            wq=state_dict[f"{prefix}.self_attn.q_proj.weight"],
            wk=state_dict[f"{prefix}.self_attn.k_proj.weight"],
            wv=state_dict[f"{prefix}.self_attn.v_proj.weight"],
            wo=state_dict[f"{prefix}.self_attn.o_proj.weight"],
            w1=state_dict[f"{prefix}.mlp.gate_proj.weight"],
            w2=state_dict[f"{prefix}.mlp.down_proj.weight"],
            w3=state_dict[f"{prefix}.mlp.up_proj.weight"],
            cos=cos,
            sin=sin,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            sliding_window=layer_types[i],
            mscale=mscale,
            norm_eps=norm_eps,
        )

    # Final norm and LM head
    h = rmsnorm_forward(h, state_dict["model.norm.weight"], norm_eps)
    logits = lm_head_forward(h, state_dict["lm_head.weight"])

    return logits

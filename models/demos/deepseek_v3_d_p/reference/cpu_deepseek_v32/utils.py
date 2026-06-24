# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU-compatible utilities for the DeepSeek V3.2 reference (Indexer + MLA).

Contains:
  * CPU equivalents of the tilelang CUDA kernels: ``act_quant_cpu``,
    ``fp8_index_cpu``, ``rotate_activation_cpu`` (Walsh-Hadamard transform).
  * Rotary-positional-embedding helpers shared by both layers:
    ``precompute_freqs_cis`` (YaRN-scaled) and ``apply_rotary_emb`` (interleaved
    for MLA, non-interleaved for the Indexer).

These are pure-PyTorch and free of any model-class imports, so ``model.py`` can
import them without a circular dependency.
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F

# ===== Kernel equivalents (mock tilelang kernels) =====


def act_quant_cpu(x: torch.Tensor, block_size: int = 128, scale_fmt: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CPU-compatible version of act_quant (FP8 quantization).

    Mirrors the reference tilelang kernel (kernel.py:act_quant_kernel):
    1. Computing block-wise absolute maximum (clamped to 1e-4)
    2. Computing the per-block scale. When ``scale_fmt`` is set (e.g. "ue8m0"),
       the scale is rounded up to a power of two, matching ``fast_round_scale``.
    3. Quantizing ``x / scale`` onto the FP8 E4M3 grid (rounding through
       float8_e4m3fn so the FP8 quantization error is simulated), stored as
       bfloat16 for CPU-friendly downstream matmuls.

    Args:
        x: Input tensor [*, N] where N % block_size == 0
        block_size: Block size for quantization (default: 128)
        scale_fmt: Scale format. When not None, scales are rounded to a power
            of two (ue8m0-style), matching the reference round_scale path.

    Returns:
        Tuple of (quantized_tensor, scale_factors)
        - quantized_tensor: bfloat16 (simulating FP8 E4M3)
        - scale_factors: float32
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    # FP8 E4M3 range
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    N = x.size(-1)
    original_shape = x.shape

    # Reshape to [*, num_blocks, block_size]
    x_reshaped = x.to(torch.float32).view(-1, N // block_size, block_size)

    # Compute absolute maximum per block
    amax = x_reshaped.abs().max(dim=-1, keepdim=True)[0]  # [*, num_blocks, 1]
    amax = torch.clamp(amax, min=1e-4)  # Avoid division by zero

    # Compute scale factors
    if scale_fmt is not None:
        # ue8m0: round the scale up to the nearest power of two.
        # Equivalent to fast_round_scale = fast_pow2(fast_log2_ceil(amax * fp8_max_inv)).
        scale = torch.pow(2.0, torch.ceil(torch.log2(amax * fp8_max_inv)))  # [*, num_blocks, 1]
    else:
        scale = amax * fp8_max_inv  # [*, num_blocks, 1]

    # Quantize: divide by scale, clamp to FP8 range, and round onto the FP8
    # E4M3 grid so the quantization error is simulated (matches the on-device
    # float8_e4m3fn cache). Stored as bfloat16 for CPU-friendly matmuls.
    x_quantized = torch.clamp(x_reshaped / scale, fp8_min, fp8_max)
    x_quantized = x_quantized.to(torch.float8_e4m3fn).to(torch.bfloat16)

    # Reshape back to original shape
    x_quantized = x_quantized.view(original_shape)
    scale = scale.squeeze(-1).view(*original_shape[:-1], N // block_size).to(torch.float32)

    return x_quantized, scale


def fp8_index_cpu(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    CPU-compatible version of fp8_index kernel.

    Computes index scores: Q @ K^T with FP8 scaling and ReLU.

    The CUDA kernel does:
    1. Q @ K^T (FP8 matmul)
    2. ReLU(logits) * q_scale (per-head weights)
    3. Sum over heads
    4. Multiply by k_scale

    Args:
        q: Query tensor [B, L, H, D] in bfloat16 (simulating FP8)
        q_s: Query scale/weights [B, L, H] in float32 (combined q_scale * weights)
        k: Key cache [B, C, D] in bfloat16 (simulating FP8)
        k_s: Key scales [B, C] in float32

    Returns:
        index_score: [B, L, C] in float32 (summed over heads)
    """
    # Q @ K^T per head: [B, L, H, D] x [B, C, D] -> [B, L, H, C]
    index_score = torch.einsum("blhd,bcd->blhc", q.float(), k.float())

    # Apply ReLU, then per-head weights (q_scale)
    index_score = F.relu(index_score)
    index_score = index_score * q_s.unsqueeze(-1)  # [B, L, H, C] * [B, L, H, 1]

    # Sum over heads: [B, L, H, C] -> [B, L, C] (matches reduce_sum in the kernel)
    index_score = index_score.sum(dim=2)  # [B, L, C]

    # Apply k_scale after the head sum: [B, L, C] * [B, 1, C]
    index_score = index_score * k_s.unsqueeze(1)  # [B, L, C]

    return index_score


def rotate_activation_cpu(x: torch.Tensor, scale: float = None) -> torch.Tensor:
    """
    CPU-compatible Hadamard transform (Walsh-Hadamard transform).

    Applies fast Walsh-Hadamard transform recursively.

    Args:
        x: Input tensor [..., D] where D must be a power of 2
        scale: Scaling factor (default: D ** -0.5)

    Returns:
        Transformed tensor with same shape as input
    """
    assert x.dtype == torch.bfloat16, "Input must be bfloat16"

    hidden_size = x.size(-1)

    # Check if hidden_size is power of 2
    if (hidden_size & (hidden_size - 1)) != 0:
        raise ValueError(f"Hidden size {hidden_size} must be a power of 2 for Hadamard transform")

    if scale is None:
        scale = hidden_size**-0.5

    # Convert to float32 for computation
    x_f32 = x.to(torch.float32)

    # Flatten all dimensions except last
    original_shape = x_f32.shape
    x_flat = x_f32.reshape(-1, hidden_size)  # [N, D]

    # Apply Hadamard transform
    x_transformed = _hadamard_transform_recursive(x_flat)

    # Scale
    x_transformed = x_transformed * scale

    # Reshape back and convert to bfloat16
    x_transformed = x_transformed.reshape(original_shape).to(torch.bfloat16)

    return x_transformed


def _hadamard_transform_recursive(x: torch.Tensor) -> torch.Tensor:
    """
    Recursive implementation of Fast Walsh-Hadamard Transform.

    Args:
        x: Input tensor [N, D] where D is a power of 2

    Returns:
        Transformed tensor [N, D]
    """
    N, D = x.shape

    if D == 1:
        return x

    # Split into two halves
    half = D // 2
    x_left = x[:, :half]
    x_right = x[:, half:]

    # Recursively apply to each half
    y_left = _hadamard_transform_recursive(x_left)
    y_right = _hadamard_transform_recursive(x_right)

    # Combine: Hadamard matrix [[1, 1], [1, -1]]
    result = torch.cat(
        [
            y_left + y_right,  # Top half
            y_left - y_right,  # Bottom half
        ],
        dim=1,
    )

    return result


# ===== Rotary positional embeddings (shared by Indexer + MLA) =====


def precompute_freqs_cis(args, apply_yarn: bool = None) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args: Model arguments containing positional embedding parameters
            (``qk_rope_head_dim``, ``max_seq_len``, ``beta_fast``, ``beta_slow``,
            ``rope_theta``, ``rope_factor``, ``original_seq_len``).
        apply_yarn: whether to apply the YaRN frequency rescaling. ``None`` (default) keeps the
            official gate (``max_seq_len > original_seq_len``); pass ``True``/``False`` to force it.
            The YaRN ramp itself depends only on ``original_seq_len``/``rope_factor`` — NOT
            ``max_seq_len`` — so forcing it lets the table length (``max_seq_len``) be sized to the
            sequence independently of whether long-context scaling is active.

    Returns:
        torch.Tensor: Precomputed complex exponential values [seq_len, rope_head_dim//2]
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """Computes the correction dimension for rotary positional embedding."""
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """Computes the range of correction dimensions."""
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim):
        """Computes a linear ramp function for smoothing."""
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Compute base frequencies
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # Apply correction for extended sequence lengths (YaRN). The gate is on max_seq_len by default
    # (official behaviour); callers that size the table to a short sequence but still want the model's
    # long-context scaling pass apply_yarn=True.
    do_yarn = (seqlen > args.original_seq_len) if apply_yarn is None else apply_yarn
    if do_yarn:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # Create position indices and compute outer product
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)  # [seq_len, rope_head_dim//2]

    # Convert to complex exponentials
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    """
    Applies rotary positional embeddings.

    Args:
        x: Input tensor
        freqs_cis: Precomputed complex exponential values
        interleaved: Whether to use interleaved format (MLA uses True, the
            Indexer uses False)

    Returns:
        Tensor with rotary embeddings applied
    """
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)

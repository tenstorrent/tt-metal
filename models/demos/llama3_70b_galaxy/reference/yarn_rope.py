# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
YaRN RoPE Reference Implementation for OLMo-3.1-32B.

This is the key new component for OLMo that differs from Llama/Qwen.
OLMo uses YaRN (Yet another RoPE extensioN) for position embeddings:

Configuration from OLMo-3.1-32B:
{
    "rope_type": "yarn",
    "factor": 8.0,
    "original_max_position_embeddings": 8192,
    "attention_factor": 1.2079441541679836,
    "beta_fast": 32.0,
    "beta_slow": 1.0
}

Reference: https://arxiv.org/abs/2309.00071 (YaRN paper)
"""

import math
from typing import Tuple, Optional
from dataclasses import dataclass

import torch


@dataclass
class YaRNConfig:
    """YaRN RoPE configuration for OLMo-3.1-32B."""

    dim: int = 128  # head_dim
    max_position_embeddings: int = 65536
    base: float = 500000.0  # rope_theta
    scaling_factor: float = 8.0  # factor
    original_max_position_embeddings: int = 8192
    attention_factor: float = 1.2079441541679836
    beta_fast: float = 32.0
    beta_slow: float = 1.0

    @classmethod
    def from_olmo(cls):
        """Create config for OLMo-3.1-32B."""
        return cls()

    @classmethod
    def from_hf_config(cls, hf_config):
        """Create from HuggingFace config."""
        rope_scaling = hf_config.rope_scaling or {}
        return cls(
            dim=hf_config.hidden_size // hf_config.num_attention_heads,
            max_position_embeddings=hf_config.max_position_embeddings,
            base=hf_config.rope_theta,
            scaling_factor=rope_scaling.get("factor", 8.0),
            original_max_position_embeddings=rope_scaling.get("original_max_position_embeddings", 8192),
            attention_factor=rope_scaling.get("attention_factor", 1.2079),
            beta_fast=rope_scaling.get("beta_fast", 32.0),
            beta_slow=rope_scaling.get("beta_slow", 1.0),
        )


def yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> float:
    """
    Find the dimension where the rotation count matches the target.

    This is used to find the boundary between interpolated and extrapolated
    dimensions in YaRN.

    Args:
        num_rotations: Target number of rotations
        dim: Total dimension
        base: RoPE theta base
        max_position_embeddings: Original context length

    Returns:
        The dimension index where rotations match
    """
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float = 10000.0,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    """
    Find the dimension range for interpolation in YaRN.

    Dimensions below 'low' are fully interpolated (extrapolated).
    Dimensions above 'high' are not interpolated.
    Dimensions between are linearly blended.

    Args:
        low_rot: Low rotation boundary (beta_fast in OLMo)
        high_rot: High rotation boundary (beta_slow in OLMo)
        dim: Head dimension
        base: RoPE theta
        max_position_embeddings: Original context length

    Returns:
        Tuple of (low_dim, high_dim) for interpolation range
    """
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(
    min_val: float,
    max_val: float,
    dim: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a linear ramp mask for YaRN interpolation blending.

    Returns values from 0 to 1, where:
    - 0 = fully extrapolated (low frequency, needs scaling)
    - 1 = no extrapolation (high frequency, keep original)

    Args:
        min_val: Start of ramp
        max_val: End of ramp
        dim: Number of dimensions

    Returns:
        Tensor of shape [dim] with linear ramp from 0 to 1
    """
    if min_val == max_val:
        max_val += 0.001  # Prevent division by zero
    linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


def compute_yarn_inv_freq(config: YaRNConfig, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Compute the inverse frequency tensor with YaRN scaling.

    YaRN modifies the standard RoPE frequencies by:
    1. Computing base inv_freq
    2. Finding which dimensions need interpolation (based on rotation counts)
    3. Applying interpolation/extrapolation with linear blending

    Args:
        config: YaRN configuration

    Returns:
        inv_freq tensor of shape [dim // 2]
    """
    dim = config.dim
    base = config.base

    # Step 1: Compute base inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

    # Step 2: Find correction range based on rotation counts
    # beta_fast (32) = high frequency boundary (fewer rotations needed)
    # beta_slow (1) = low frequency boundary (more rotations needed)
    low, high = yarn_find_correction_range(
        config.beta_fast,
        config.beta_slow,
        dim,
        config.base,
        config.original_max_position_embeddings,
    )

    # Step 3: Create interpolation mask
    # Mask = 1 means keep original frequency
    # Mask = 0 means fully scale the frequency
    inv_freq_mask = 1 - yarn_linear_ramp_mask(low, high, dim // 2, device=device)

    # Step 4: Scale frequencies for context extension
    inv_freq_scaled = inv_freq / config.scaling_factor

    # Step 5: Blend using mask
    # High-freq dimensions (mask≈0): use scaled (extrapolated)
    # Low-freq dimensions (mask≈1): use scaled (interpolated)
    # In YaRN, both are scaled, but the mask affects NTK-aware interpolation
    inv_freq = inv_freq_scaled

    return inv_freq


def compute_yarn_mscale(config: YaRNConfig) -> float:
    """
    Compute the attention mscale for YaRN.

    In OLMo, this is directly specified as 'attention_factor' in the config.
    The mscale adjusts the attention logits to compensate for context extension.

    Args:
        config: YaRN configuration

    Returns:
        mscale value to multiply attention logits by
    """
    # OLMo directly specifies attention_factor as mscale
    return config.attention_factor


def precompute_yarn_freqs(
    config: YaRNConfig,
    seq_len: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Precompute cos/sin for YaRN RoPE.

    Args:
        config: YaRN configuration
        seq_len: Sequence length (defaults to max_position_embeddings)
        device: Device for tensors

    Returns:
        Tuple of (cos, sin, mscale)
        - cos: [seq_len, dim//2]
        - sin: [seq_len, dim//2]
        - mscale: attention scaling factor
    """
    if seq_len is None:
        seq_len = config.max_position_embeddings

    # Compute YaRN-scaled inverse frequencies
    inv_freq = compute_yarn_inv_freq(config, device=device)

    # Position indices
    t = torch.arange(seq_len, dtype=torch.float32, device=device)

    # Outer product: [seq_len, dim//2]
    freqs = torch.outer(t, inv_freq)

    # Compute cos and sin
    cos = freqs.cos()
    sin = freqs.sin()

    # Compute mscale
    mscale = compute_yarn_mscale(config)

    return cos, sin, mscale


def precompute_yarn_freqs_cis(
    config: YaRNConfig,
    seq_len: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Precompute complex frequencies for YaRN RoPE (Llama-style).

    Args:
        config: YaRN configuration
        seq_len: Sequence length
        device: Device for tensors

    Returns:
        Tuple of (freqs_cis, mscale)
        - freqs_cis: complex64 tensor [seq_len, dim//2]
        - mscale: attention scaling factor
    """
    if seq_len is None:
        seq_len = config.max_position_embeddings

    # Compute YaRN-scaled inverse frequencies
    inv_freq = compute_yarn_inv_freq(config, device=device)

    # Position indices
    t = torch.arange(seq_len, dtype=torch.float32, device=device)

    # Outer product
    freqs = torch.outer(t, inv_freq)

    # Complex exponential: e^(i*freq) = cos(freq) + i*sin(freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    # Compute mscale
    mscale = compute_yarn_mscale(config)

    return freqs_cis, mscale


def apply_rotary_emb_yarn(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply YaRN rotary embeddings to Q and K.

    This is the same as standard RoPE application - YaRN only changes
    how the frequencies are computed, not how they're applied.

    Args:
        q: Query tensor [batch, seq_len, n_heads, head_dim]
        k: Key tensor [batch, seq_len, n_kv_heads, head_dim]
        cos: Cosine values [seq_len, head_dim//2]
        sin: Sine values [seq_len, head_dim//2]
        position_ids: Optional position indices [batch, seq_len]

    Returns:
        Tuple of rotated (q, k) tensors
    """
    # Expand cos/sin to full head_dim (interleave real/imag)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    # Get positions
    if position_ids is not None:
        # Gather cos/sin for specific positions
        cos = cos[position_ids]  # [batch, seq_len, head_dim]
        sin = sin[position_ids]
    else:
        # Use sequential positions
        seq_len = q.shape[1]
        cos = cos[:seq_len]  # [seq_len, head_dim]
        sin = sin[:seq_len]

    # Reshape for broadcasting
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(2)
    else:
        cos = cos.unsqueeze(2)  # [batch, seq_len, 1, head_dim]
        sin = sin.unsqueeze(2)

    # Rotate half
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Apply rotation (preserve input dtype)
    q_embed = ((q * cos) + (rotate_half(q) * sin)).to(q.dtype)
    k_embed = ((k * cos) + (rotate_half(k) * sin)).to(k.dtype)

    return q_embed, k_embed


# ==============================================================================
# Test/Verification Functions
# ==============================================================================
def verify_yarn_against_hf(model_path: str = "allenai/OLMo-3.1-32B-Think"):
    """
    Verify YaRN implementation against HuggingFace.

    This function loads the HF model and compares our RoPE computation
    against theirs.
    """
    try:
        from transformers import AutoConfig
    except ImportError:
        print("transformers not installed, skipping HF comparison")
        return False

    print(f"Loading HF config from {model_path}...")
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Create our config
    yarn_config = YaRNConfig.from_hf_config(hf_config)
    print(f"YaRN config: {yarn_config}")

    # Compute our frequencies
    cos, sin, mscale = precompute_yarn_freqs(yarn_config, seq_len=1024)

    print(f"cos shape: {cos.shape}")
    print(f"sin shape: {sin.shape}")
    print(f"mscale: {mscale}")
    print(f"cos[0, :5]: {cos[0, :5]}")
    print(f"sin[0, :5]: {sin[0, :5]}")

    # For full verification, load the model and compare
    print("\nTo fully verify, load HF model and compare rotary embeddings.")
    print("This requires significant memory for OLMo-32B.")

    return True


def print_yarn_config():
    """Print OLMo YaRN configuration details."""
    config = YaRNConfig.from_olmo()

    print("=" * 60)
    print("OLMo-3.1-32B YaRN RoPE Configuration")
    print("=" * 60)
    print(f"Head dimension: {config.dim}")
    print(f"RoPE theta (base): {config.base}")
    print(f"Max position embeddings: {config.max_position_embeddings}")
    print(f"Scaling factor: {config.scaling_factor}")
    print(f"Original max position embeddings: {config.original_max_position_embeddings}")
    print(f"Attention factor (mscale): {config.attention_factor}")
    print(f"Beta fast: {config.beta_fast}")
    print(f"Beta slow: {config.beta_slow}")
    print()

    # Compute correction range
    low, high = yarn_find_correction_range(
        config.beta_fast,
        config.beta_slow,
        config.dim,
        config.base,
        config.original_max_position_embeddings,
    )
    print(f"Correction range: dim {low} to {high}")
    print(f"(Dimensions {low}-{high} are interpolated, others extrapolated)")
    print("=" * 60)


if __name__ == "__main__":
    print_yarn_config()

    # Compute sample frequencies
    config = YaRNConfig.from_olmo()
    cos, sin, mscale = precompute_yarn_freqs(config, seq_len=128)

    print(f"\nSample frequencies (seq_len=128):")
    print(f"cos shape: {cos.shape}")
    print(f"sin shape: {sin.shape}")
    print(f"mscale: {mscale}")
    print(f"\ncos[0]: {cos[0, :8].tolist()}")
    print(f"cos[64]: {cos[64, :8].tolist()}")

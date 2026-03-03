# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
RoPE and MROPE implementations for Qwen3-TTS.

MROPE (Multimodal RoPE) is used by the Talker model with sections [24, 20, 20]
for temporal, height, and width dimensions.

Standard RoPE is used by the CodePredictor model.
"""

from typing import Optional, Tuple

import torch

import ttnn


def compute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1000000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute standard RoPE frequencies (cos and sin).

    Args:
        head_dim: Dimension of each attention head
        max_seq_len: Maximum sequence length
        theta: Base for frequency computation
        device: Device to create tensors on

    Returns:
        Tuple of (cos, sin) tensors of shape [max_seq_len, head_dim]
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    position_ids = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(position_ids, inv_freq)
    # Interleave for TTNN format: [cos0, cos0, cos1, cos1, ...]
    cos = torch.stack([freqs.cos(), freqs.cos()], dim=-1).flatten(-2)
    sin = torch.stack([freqs.sin(), freqs.sin()], dim=-1).flatten(-2)
    return cos, sin


def compute_mrope_frequencies(
    head_dim: int,
    max_seq_len: int,
    mrope_section: Tuple[int, int, int] = (24, 20, 20),
    theta: float = 1000000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute MROPE (Multimodal RoPE) frequencies for 3D positions.

    MROPE divides the head_dim into sections for different position types:
    - Section 0 (temporal): First mrope_section[0] * 2 dimensions
    - Section 1 (height): Next mrope_section[1] * 2 dimensions
    - Section 2 (width): Last mrope_section[2] * 2 dimensions

    Args:
        head_dim: Dimension of each attention head (must equal sum of sections * 2)
        max_seq_len: Maximum sequence length
        mrope_section: Tuple of (temporal, height, width) section sizes
        theta: Base for frequency computation
        device: Device to create tensors on

    Returns:
        Tuple of (cos, sin) tensors of shape [3, max_seq_len, head_dim]
        Each tensor contains frequencies for all 3 position types
    """
    assert sum(mrope_section) * 2 == head_dim, f"MROPE sections {mrope_section} don't match head_dim {head_dim}"

    cos_list = []
    sin_list = []

    # Compute frequencies for each section
    section_start = 0
    for section_idx, section_size in enumerate(mrope_section):
        section_dim = section_size * 2  # Each section handles half the frequencies

        # Compute inverse frequencies for this section
        inv_freq = 1.0 / (theta ** (torch.arange(0, section_dim, 2, dtype=torch.float32, device=device) / section_dim))
        position_ids = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(position_ids, inv_freq)

        # Interleave for TTNN format
        section_cos = torch.stack([freqs.cos(), freqs.cos()], dim=-1).flatten(-2)
        section_sin = torch.stack([freqs.sin(), freqs.sin()], dim=-1).flatten(-2)

        cos_list.append(section_cos)
        sin_list.append(section_sin)

    # Concatenate sections to form full head_dim
    # Shape: [max_seq_len, head_dim]
    full_cos = torch.cat(cos_list, dim=-1)
    full_sin = torch.cat(sin_list, dim=-1)

    # Expand for 3 position types: [3, max_seq_len, head_dim]
    # In practice, each position type would have its own position_ids
    # For now, we return the same frequencies for all 3 types
    cos_3d = full_cos.unsqueeze(0).expand(3, -1, -1)
    sin_3d = full_sin.unsqueeze(0).expand(3, -1, -1)

    return cos_3d, sin_3d


def get_rope_tensors(
    device,
    head_dim: int,
    seq_len: int,
    position_ids: torch.Tensor,
    theta: float = 1000000.0,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Get RoPE cos/sin tensors for TTNN attention.

    Args:
        device: TTNN device
        head_dim: Head dimension
        seq_len: Sequence length
        position_ids: Position indices [seq_len]
        theta: RoPE theta parameter

    Returns:
        Tuple of (cos, sin) TTNN tensors of shape [1, 1, seq_len, head_dim]
    """
    # Compute full frequency table - ensure it covers all positions
    max_pos = max(seq_len * 2, position_ids.max().item() + 1)
    cos_full, sin_full = compute_rope_frequencies(head_dim, max_pos, theta)

    # Gather for specific positions
    cos_gathered = cos_full[position_ids]  # [seq_len, head_dim]
    sin_gathered = sin_full[position_ids]

    # Reshape for TTNN: [1, 1, seq_len, head_dim]
    cos_gathered = cos_gathered.unsqueeze(0).unsqueeze(0)
    sin_gathered = sin_gathered.unsqueeze(0).unsqueeze(0)

    # Convert to TTNN
    cos_ttnn = ttnn.from_torch(
        cos_gathered.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_ttnn = ttnn.from_torch(
        sin_gathered.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return cos_ttnn, sin_ttnn


def get_mrope_tensors(
    device,
    head_dim: int,
    seq_len: int,
    temporal_ids: torch.Tensor,
    height_ids: torch.Tensor,
    width_ids: torch.Tensor,
    mrope_section: Tuple[int, int, int] = (24, 20, 20),
    theta: float = 1000000.0,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Get MROPE cos/sin tensors for TTNN attention.

    For MROPE, each position has 3 components (temporal, height, width),
    and different sections of head_dim are rotated by different position types.

    Args:
        device: TTNN device
        head_dim: Head dimension
        seq_len: Sequence length
        temporal_ids: Temporal position indices [seq_len]
        height_ids: Height position indices [seq_len]
        width_ids: Width position indices [seq_len]
        mrope_section: Section sizes for (temporal, height, width)
        theta: RoPE theta parameter

    Returns:
        Tuple of (cos, sin) TTNN tensors of shape [1, 1, seq_len, head_dim]
    """
    max_pos = max(temporal_ids.max().item(), height_ids.max().item(), width_ids.max().item()) + 1
    max_pos = max(max_pos, seq_len * 2)

    # Compute frequencies for each section
    cos_sections = []
    sin_sections = []

    position_ids_list = [temporal_ids, height_ids, width_ids]

    for section_idx, (section_size, pos_ids) in enumerate(zip(mrope_section, position_ids_list)):
        section_dim = section_size * 2

        # Compute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, section_dim, 2, dtype=torch.float32) / section_dim))
        freqs_full = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)

        # Gather for positions
        freqs = freqs_full[pos_ids]  # [seq_len, section_dim // 2]

        # Interleave
        section_cos = torch.stack([freqs.cos(), freqs.cos()], dim=-1).flatten(-2)
        section_sin = torch.stack([freqs.sin(), freqs.sin()], dim=-1).flatten(-2)

        cos_sections.append(section_cos)
        sin_sections.append(section_sin)

    # Concatenate sections: [seq_len, head_dim]
    cos_combined = torch.cat(cos_sections, dim=-1)
    sin_combined = torch.cat(sin_sections, dim=-1)

    # Reshape for TTNN: [1, 1, seq_len, head_dim]
    cos_combined = cos_combined.unsqueeze(0).unsqueeze(0)
    sin_combined = sin_combined.unsqueeze(0).unsqueeze(0)

    # Convert to TTNN
    cos_ttnn = ttnn.from_torch(
        cos_combined.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_ttnn = ttnn.from_torch(
        sin_combined.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return cos_ttnn, sin_ttnn


def get_transformation_mat(head_dim: int, device) -> ttnn.Tensor:
    """
    Get the transformation matrix for rotary_embedding_llama.

    Args:
        head_dim: Head dimension
        device: TTNN device

    Returns:
        Transformation matrix tensor
    """
    from models.tt_transformers.tt.common import get_rot_transformation_mat

    trans_mat = get_rot_transformation_mat(dhead=head_dim)
    trans_mat_ttnn = ttnn.from_torch(
        trans_mat,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return trans_mat_ttnn


def rearrange_to_interleaved(x: torch.Tensor) -> torch.Tensor:
    """
    Rearrange tensor from non-interleaved to interleaved RoPE format.

    Qwen3-TTS uses non-interleaved RoPE that pairs dimensions (i, i+64).
    TTNN rotary_embedding_llama uses interleaved format that pairs (2i, 2i+1).

    This function rearranges:
        [..., d0, d1, ..., d63, d64, d65, ..., d127]
    To:
        [..., d0, d64, d1, d65, ..., d63, d127]

    Args:
        x: Tensor with shape [..., head_dim] where head_dim is even

    Returns:
        Rearranged tensor with same shape
    """
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    x1 = x[..., :half_dim]  # dims 0-63
    x2 = x[..., half_dim:]  # dims 64-127

    # Interleave: [d0, d64, d1, d65, ...]
    return torch.stack([x1, x2], dim=-1).flatten(-2)


def rearrange_to_noninterleaved(x: torch.Tensor) -> torch.Tensor:
    """
    Rearrange tensor from interleaved to non-interleaved RoPE format.
    Inverse of rearrange_to_interleaved.

    This function rearranges:
        [..., d0, d64, d1, d65, ..., d63, d127]
    To:
        [..., d0, d1, ..., d63, d64, d65, ..., d127]

    Args:
        x: Tensor with shape [..., head_dim] where head_dim is even

    Returns:
        Rearranged tensor with same shape
    """
    x1 = x[..., 0::2]  # even indices: d0, d1, d2, ...
    x2 = x[..., 1::2]  # odd indices: d64, d65, d66, ...

    return torch.cat([x1, x2], dim=-1)


def compute_mrope_cos_sin_for_ttnn(
    cos_mrope: torch.Tensor,
    sin_mrope: torch.Tensor,
    mrope_section: Tuple[int, int, int] = (24, 20, 20),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert MROPE cos/sin tensors to format suitable for TTNN rotary_embedding_llama.

    The official qwen_tts MROPE with interleaved=True produces cos/sin that need
    special processing. This function applies the MROPE interleaving and then
    converts to TTNN's expected format (pair-wise interleaved).

    Args:
        cos_mrope: MROPE cos tensor [3, batch, seq, head_dim]
        sin_mrope: MROPE sin tensor [3, batch, seq, head_dim]
        mrope_section: MROPE section sizes [24, 20, 20]

    Returns:
        Tuple of (cos, sin) tensors in TTNN format [batch, seq, head_dim]
    """
    modality_num = len(mrope_section)
    dim = cos_mrope.shape[-1]  # 128

    def apply_mrope_interleaving(x, modality_num, mrope_section):
        """Apply MROPE interleaving from official qwen_tts."""
        x_t = x[0].clone()
        for i, n in enumerate(mrope_section[1:], 1):
            beg_idx = i
            end_idx = n * modality_num
            x_t[..., beg_idx:end_idx:modality_num] = x[beg_idx, ..., beg_idx:end_idx:modality_num]
        return x_t

    # Apply MROPE interleaving (produces HF format: [c0, c1, ..., c63, c0, c1, ..., c63])
    cos_hf = torch.cat([apply_mrope_interleaving(cos_mrope[..., : dim // 2], modality_num, mrope_section)] * 2, dim=-1)
    sin_hf = torch.cat([apply_mrope_interleaving(sin_mrope[..., : dim // 2], modality_num, mrope_section)] * 2, dim=-1)

    # Convert HF format to TTNN format (pair-wise interleaved: [c0, c0, c1, c1, ...])
    half_dim = dim // 2
    cos_unique = cos_hf[..., :half_dim]
    sin_unique = sin_hf[..., :half_dim]

    cos_ttnn = torch.repeat_interleave(cos_unique, repeats=2, dim=-1)
    sin_ttnn = torch.repeat_interleave(sin_unique, repeats=2, dim=-1)

    return cos_ttnn, sin_ttnn

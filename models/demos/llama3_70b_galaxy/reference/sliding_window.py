# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Sliding Window Attention Reference Implementation for OLMo-3.1-32B.

OLMo uses a hybrid attention pattern:
- 48 layers with sliding window (window_size=4096)
- 16 layers with full attention

Pattern: [sliding, sliding, sliding, full] repeated 16 times

This module provides reference implementations for:
1. Sliding window mask creation
2. Per-layer attention type determination
3. Attention with sliding window support
"""

from typing import Optional, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SlidingWindowConfig:
    """Configuration for OLMo sliding window attention."""

    n_layers: int = 64
    sliding_window: int = 4096
    # Pattern: 3 sliding + 1 full, repeated
    pattern_size: int = 4  # Every 4th layer is full attention

    def get_layer_type(self, layer_id: int) -> str:
        """Get attention type for a specific layer."""
        if (layer_id + 1) % self.pattern_size == 0:
            return "full_attention"
        return "sliding_attention"

    def get_sliding_window_size(self, layer_id: int) -> Optional[int]:
        """Get sliding window size for a layer (None for full attention)."""
        if self.get_layer_type(layer_id) == "full_attention":
            return None
        return self.sliding_window

    def get_all_layer_types(self) -> List[str]:
        """Get list of layer types for all layers."""
        return [self.get_layer_type(i) for i in range(self.n_layers)]


def create_causal_mask(
    seq_len: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a standard causal attention mask.

    Args:
        seq_len: Sequence length
        dtype: Output dtype (typically float32 or bfloat16)
        device: Output device

    Returns:
        Mask tensor [seq_len, seq_len] where future positions are -inf
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device), diagonal=1)
    return mask


def create_sliding_window_mask(
    seq_len: int,
    sliding_window: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a sliding window causal attention mask.

    Tokens can only attend to:
    - Themselves
    - Tokens up to `sliding_window - 1` positions before them
    - Total window size is `sliding_window` tokens

    Args:
        seq_len: Sequence length
        sliding_window: Window size (e.g., 4096 for OLMo)
        dtype: Output dtype
        device: Output device

    Returns:
        Mask tensor [seq_len, seq_len]

    Example for seq_len=8, sliding_window=4:
        Position 0 attends to: [0]               (1 token)
        Position 1 attends to: [0, 1]            (2 tokens)
        Position 2 attends to: [0, 1, 2]         (3 tokens)
        Position 3 attends to: [0, 1, 2, 3]      (4 tokens = window)
        Position 4 attends to: [1, 2, 3, 4]      (4 tokens)
        Position 5 attends to: [2, 3, 4, 5]      (4 tokens)
        Position 6 attends to: [3, 4, 5, 6]      (4 tokens)
        Position 7 attends to: [4, 5, 6, 7]      (4 tokens)
    """
    # Start with causal mask
    mask = create_causal_mask(seq_len, dtype, device)

    # Add sliding window constraint
    # Mask positions more than `sliding_window - 1` positions ago
    # (i.e., keep only the last `sliding_window` tokens)
    positions = torch.arange(seq_len, device=device)
    # distance[i, j] = i - j (how far back j is from i)
    distance = positions.unsqueeze(1) - positions.unsqueeze(0)

    # Mask where distance >= sliding_window (outside the window)
    window_mask = torch.where(
        distance >= sliding_window,
        torch.full_like(mask, float("-inf")),
        torch.zeros_like(mask),
    )

    return mask + window_mask


def create_combined_mask(
    seq_len: int,
    sliding_window: Optional[int],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create appropriate mask based on attention type.

    Args:
        seq_len: Sequence length
        sliding_window: Window size (None for full attention)
        dtype: Output dtype
        device: Output device

    Returns:
        Mask tensor [1, 1, seq_len, seq_len] ready for attention
    """
    if sliding_window is None:
        mask = create_causal_mask(seq_len, dtype, device)
    else:
        mask = create_sliding_window_mask(seq_len, sliding_window, dtype, device)

    return mask.unsqueeze(0).unsqueeze(0)


def create_decode_mask(
    current_pos: int,
    kv_seq_len: int,
    sliding_window: Optional[int],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create mask for decode mode (single query position).

    During decode, we have a single query attending to all cached KV positions.

    Args:
        current_pos: Current decode position
        kv_seq_len: Total KV cache length (current_pos + 1)
        sliding_window: Window size (None for full attention)
        dtype: Output dtype
        device: Output device

    Returns:
        Mask tensor [1, 1, 1, kv_seq_len]
    """
    # For decode, query position is always current_pos
    # KV positions are 0 to current_pos

    mask = torch.zeros(1, 1, 1, kv_seq_len, dtype=dtype, device=device)

    if sliding_window is not None:
        # Mask positions more than sliding_window before current_pos
        positions = torch.arange(kv_seq_len, device=device)
        distance = current_pos - positions

        # Positions outside window get -inf
        outside_window = distance >= sliding_window
        mask[0, 0, 0, outside_window] = float("-inf")

    return mask


def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sliding_window: Optional[int] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute attention with optional sliding window.

    Args:
        q: Query tensor [batch, n_heads, seq_len, head_dim]
        k: Key tensor [batch, n_heads, kv_len, head_dim]
        v: Value tensor [batch, n_heads, kv_len, head_dim]
        sliding_window: Window size (None for full attention)
        scale: Attention scale (default: 1/sqrt(head_dim))

    Returns:
        Output tensor [batch, n_heads, seq_len, head_dim]
    """
    batch, n_heads, seq_len, head_dim = q.shape
    kv_len = k.shape[2]

    if scale is None:
        scale = head_dim**-0.5

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Create and apply mask
    if seq_len == kv_len:
        # Prefill mode: full causal or sliding window mask
        mask = create_combined_mask(seq_len, sliding_window, dtype=scores.dtype, device=scores.device)
    else:
        # Decode mode: single query
        assert seq_len == 1, f"Expected seq_len=1 for decode, got {seq_len}"
        current_pos = kv_len - 1
        mask = create_decode_mask(current_pos, kv_len, sliding_window, dtype=scores.dtype, device=scores.device)

    scores = scores + mask

    # Softmax and weighted sum
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    output = torch.matmul(attn_weights, v)

    return output


def compute_effective_context(
    position: int,
    sliding_window: int,
) -> int:
    """
    Compute effective context length at a given position.

    For sliding window attention, the effective context is min(position + 1, window).

    Args:
        position: Current position (0-indexed)
        sliding_window: Window size

    Returns:
        Number of tokens this position can attend to
    """
    return min(position + 1, sliding_window)


def get_attention_pattern_summary(config: SlidingWindowConfig) -> dict:
    """
    Get summary of attention pattern.

    Returns:
        Dict with counts and positions of each attention type
    """
    layer_types = config.get_all_layer_types()

    sliding_layers = [i for i, t in enumerate(layer_types) if t == "sliding_attention"]
    full_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]

    return {
        "total_layers": config.n_layers,
        "sliding_count": len(sliding_layers),
        "full_count": len(full_layers),
        "sliding_window": config.sliding_window,
        "sliding_layers": sliding_layers[:5] + ["..."] + sliding_layers[-5:]
        if len(sliding_layers) > 10
        else sliding_layers,
        "full_layers": full_layers,
        "pattern": "3 sliding + 1 full, repeated",
    }


def print_attention_pattern(config: SlidingWindowConfig):
    """Print the attention pattern for visualization."""
    print("=" * 60)
    print("OLMo-3.1-32B Attention Pattern")
    print("=" * 60)

    summary = get_attention_pattern_summary(config)
    print(f"Total layers: {summary['total_layers']}")
    print(f"Sliding window layers: {summary['sliding_count']} (window={summary['sliding_window']})")
    print(f"Full attention layers: {summary['full_count']}")
    print(f"Pattern: {summary['pattern']}")
    print()
    print("Full attention layers:", summary["full_layers"])
    print("=" * 60)


# ==============================================================================
# Verification Functions
# ==============================================================================
def verify_mask_correctness(seq_len: int = 16, sliding_window: int = 4):
    """
    Verify sliding window mask is correct by visual inspection.

    Prints the mask pattern for small sequence lengths.
    """
    print(f"\nSliding Window Mask (seq_len={seq_len}, window={sliding_window}):")
    print("-" * 50)

    mask = create_sliding_window_mask(seq_len, sliding_window)

    # Convert to 0/1 for visualization
    visual = torch.where(mask == float("-inf"), torch.zeros_like(mask), torch.ones_like(mask))

    for i in range(seq_len):
        row = ""
        for j in range(seq_len):
            if visual[i, j] == 1:
                row += "o "  # Can attend
            else:
                row += ". "  # Cannot attend
        attending_to = int(visual[i].sum().item())
        print(f"Pos {i:2d}: {row} (attends to {attending_to} tokens)")

    print("-" * 50)
    print("o = can attend, . = masked")


def verify_decode_mask(kv_len: int = 16, sliding_window: int = 4):
    """
    Verify decode mask is correct for different current positions.
    """
    print(f"\nDecode Masks (kv_len={kv_len}, window={sliding_window}):")
    print("-" * 50)

    for current_pos in [0, 3, 7, 15]:
        if current_pos >= kv_len:
            continue

        mask = create_decode_mask(current_pos, current_pos + 1, sliding_window)
        visual = torch.where(mask == float("-inf"), torch.zeros_like(mask), torch.ones_like(mask))

        row = ""
        for j in range(current_pos + 1):
            if visual[0, 0, 0, j] == 1:
                row += "o "
            else:
                row += ". "

        attending_to = int(visual.sum().item())
        print(f"Pos {current_pos:2d}: {row} (attends to {attending_to} tokens)")

    print("-" * 50)


if __name__ == "__main__":
    # Print pattern
    config = SlidingWindowConfig()
    print_attention_pattern(config)

    # Verify masks
    verify_mask_correctness(seq_len=16, sliding_window=4)
    verify_decode_mask(kv_len=16, sliding_window=4)

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Functional stubs for Qwen2.5-VL modules that match input/output shapes.
These are lightweight implementations for testing and development.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


def qwen2_rms_norm(x: torch.Tensor, state_dict: Dict, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm with weight scaling."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * state_dict["weight"]


def _mlp_forward(x: torch.Tensor, state_dict: Dict) -> torch.Tensor:
    """Common MLP forward pass for both Qwen2MLP and Qwen2_5_VLMLP."""
    gate = F.silu(F.linear(x, state_dict["gate_proj"]["weight"], state_dict["gate_proj"].get("bias")))
    up = F.linear(x, state_dict["up_proj"]["weight"], state_dict["up_proj"].get("bias"))
    return F.linear(gate * up, state_dict["down_proj"]["weight"], state_dict["down_proj"].get("bias"))


def qwen2_mlp(x: torch.Tensor, state_dict: Dict) -> torch.Tensor:
    """Qwen2MLP forward pass."""
    return _mlp_forward(x, state_dict)


def qwen2_5_vl_mlp(x: torch.Tensor, state_dict: Dict) -> torch.Tensor:
    """Qwen2_5_VLMLP forward pass."""
    return _mlp_forward(x, state_dict)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def qwen2_5_vl_vision_sdpa_attention(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_dict: Dict,
    num_heads: int,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Vision self-attention with rotary embeddings."""
    print("\n    Vision Attention:")
    print(f"    Input shape: {x.shape}")

    seq_len, hidden_size = x.shape
    head_dim = hidden_size // num_heads
    print(f"    seq_len: {seq_len}, hidden_size: {hidden_size}, head_dim: {head_dim}")

    # QKV projection and reshape
    qkv = F.linear(x, state_dict["qkv"]["weight"], state_dict["qkv"]["bias"])
    q, k, v = qkv.view(seq_len, 3, num_heads, head_dim).permute(1, 0, 2, 3)
    print(f"    After QKV: q={q.shape}, k={k.shape}, v={v.shape}")

    # Apply rotary embeddings
    if rotary_pos_emb is not None:
        print("    Using rotary_pos_emb")
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().float()
        sin = emb.sin().float()
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    elif position_embeddings is not None:
        print("    Using position_embeddings")
        print(f"    Position embeddings shapes: {[p.shape for p in position_embeddings]}")
        q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
    print(f"    After rotary: q={q.shape}, k={k.shape}")

    # Create attention mask
    mask = torch.ones(1, seq_len, seq_len, device=q.device, dtype=torch.bool)
    if cu_seqlens is not None:
        print(f"    Creating mask from cu_seqlens: {cu_seqlens}")
        mask.zero_()
        for i in range(1, len(cu_seqlens)):
            s, e = cu_seqlens[i - 1 : i + 1]
            print(f"    Setting mask[{s}:{e}, {s}:{e}] = True")
            mask[..., s:e, s:e] = True
    print(f"    Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"    Mask non-zero ratio: {mask.sum() / mask.numel():.2%}")

    # Attention and output projection
    attn_out = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), mask, dropout_p=0.0
    )
    print(f"    After attention: {attn_out.shape}")

    output = F.linear(
        attn_out.transpose(0, 1).reshape(seq_len, -1), state_dict["proj"]["weight"], state_dict["proj"]["bias"]
    )
    print(f"    Final output: {output.shape}")

    return output


def qwen2_5_vl_patch_merger(x: torch.Tensor, state_dict: Dict) -> torch.Tensor:
    """Functional stub for Qwen2_5_VLPatchMerger.

    state_dict structure:
    - ln_q:
      - weight: Layer norm weight
    - mlp:
      - 0:
        - weight: First linear layer weight
      - 2:
        - weight: Second linear layer weight
    """
    # Get the hidden size from the weight matrix shape
    hidden_size = state_dict["mlp"]["0"]["weight"].shape[0]

    # Normalize first
    x = qwen2_rms_norm(x, {"weight": state_dict["ln_q"]["weight"]})

    # Reshape to match the expected input size
    x = x.view(-1, hidden_size)

    # Apply MLP layers
    x = F.linear(x, state_dict["mlp"]["0"]["weight"])
    x = F.gelu(x)
    x = F.linear(x, state_dict["mlp"]["2"]["weight"])

    return x


def qwen2_5_vision_patch_embed(
    hidden_states: torch.Tensor,
    state_dict: Dict,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
) -> torch.Tensor:
    """Validate input shape before processing."""
    if hidden_states.dim() == 1:
        total_elements = hidden_states.size(0)
        expected_elements = 3 * temporal_patch_size * patch_size * patch_size
        if total_elements != expected_elements:
            raise ValueError(
                f"Input tensor size {total_elements} does not match expected size "
                f"{expected_elements} for patch_size={patch_size}, temporal_patch_size={temporal_patch_size}"
            )
    """Functional stub for Qwen2_5_VisionPatchEmbed.

    state_dict structure:
    - proj:
      - weight: 3D convolution weight matrix
    """
    # Get the original shape information
    in_channels = state_dict["proj"]["weight"].shape[1]  # Number of input channels

    # Convert input to the target dtype of the projection weight
    target_dtype = state_dict["proj"]["weight"].dtype

    # Get input shape and handle reshaping appropriately
    if hidden_states.dim() == 1:
        # Handle single dimension input
        side_length = int((hidden_states.size(0) / (in_channels * temporal_patch_size)) ** 0.5)
        hidden_states = hidden_states.view(1, in_channels, temporal_patch_size, side_length, side_length)
    else:
        # Handle multi-dimensional input
        hidden_states = hidden_states.view(-1, in_channels, temporal_patch_size, patch_size, patch_size)

    # Apply 3D convolution and reshape output
    hidden_states = F.conv3d(
        hidden_states.to(dtype=target_dtype),
        state_dict["proj"]["weight"],
        bias=None,
        stride=[temporal_patch_size, patch_size, patch_size],
    )

    # Flatten the output
    hidden_states = hidden_states.view(-1, state_dict["proj"]["weight"].shape[0])

    return hidden_states


def qwen2_5_vision_transformer(
    hidden_states: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    state_dict: Dict,
    num_heads: int,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Functional implementation of Qwen2.5 Vision Transformer.

    Args:
        hidden_states: Input tensor
        cu_seqlens: Cumulative sequence lengths
        state_dict: Dictionary containing model weights
        num_heads: Number of attention heads
        patch_size: Size of image patches
        temporal_patch_size: Size of temporal patches
        rotary_pos_emb: Optional rotary position embeddings
        position_embeddings: Optional position embeddings tuple (cos, sin)

    state_dict structure:
    - patch_embed: Patch embedding weights
    - blocks: List of transformer block weights
    - merger: Patch merger weights
    """
    print("\nStarting functional implementation:")
    print(f"Initial hidden_states shape: {hidden_states.shape}")

    # Patch embedding
    hidden_states = qwen2_5_vision_patch_embed(
        hidden_states, state_dict["patch_embed"], patch_size, temporal_patch_size
    )
    print(f"After patch embedding shape: {hidden_states.shape}")

    # Get window indices and window-level cumulative sequence lengths
    window_index = state_dict["window_index"]
    cu_window_seqlens = torch.tensor(state_dict["cu_window_seqlens"], device=hidden_states.device, dtype=torch.int32)
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    # Compute reverse indices right after getting window_index
    reverse_indices = torch.argsort(window_index)
    print(f"reverse_indices shape: {reverse_indices.shape}")
    print(f"reverse_indices min/max: {reverse_indices.min()}, {reverse_indices.max()}")

    # Apply windowing
    seq_len, _ = hidden_states.size()
    spatial_merge_unit = state_dict["spatial_merge_unit"]
    print(f"\nBefore windowing:")
    print(f"seq_len: {seq_len}")
    print(f"spatial_merge_unit: {spatial_merge_unit}")
    print(f"window_index shape: {window_index.shape}")
    print(f"window_index min/max: {window_index.min()}, {window_index.max()}")

    hidden_states = hidden_states.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
    print(f"After first reshape: {hidden_states.shape}")
    hidden_states = hidden_states[window_index, :, :]
    print(f"After window indexing: {hidden_states.shape}")
    hidden_states = hidden_states.reshape(seq_len, -1)
    print(f"After final reshape: {hidden_states.shape}")

    # Apply same windowing to position embeddings if present
    if rotary_pos_emb is not None:
        print("\nProcessing rotary embeddings:")
        print(f"Initial rotary_pos_emb shape: {rotary_pos_emb.shape}")
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
        print(f"After first reshape: {rotary_pos_emb.shape}")
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        print(f"After window indexing: {rotary_pos_emb.shape}")
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        print(f"After final reshape: {rotary_pos_emb.shape}")
        # Create final position embeddings
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        print(f"After concatenation: {emb.shape}")
        position_embeddings = (emb.cos(), emb.sin())
        print(f"Final position embeddings shapes: {[p.shape for p in position_embeddings]}")

    # Process blocks in order
    num_blocks = len(state_dict["blocks"])
    fullatt_block_indexes = state_dict["fullatt_block_indexes"]
    print(f"\nProcessing {num_blocks} blocks:")
    print(f"Full attention block indexes: {fullatt_block_indexes}")

    # If cu_seqlens is not provided, create it from the sequence length
    if cu_seqlens is None:
        # In the model, this is computed from grid_thw, but since we don't have that,
        # we'll create a single sequence length tensor
        cu_seqlens = torch.tensor([0, seq_len], device=hidden_states.device, dtype=torch.int32)

    for block_idx in range(num_blocks):
        # Choose between full attention and windowed attention
        cu_seqlens_now = cu_seqlens if block_idx in fullatt_block_indexes else cu_window_seqlens
        print(f"\nBlock {block_idx}:")
        print(f"Using {'full' if block_idx in fullatt_block_indexes else 'windowed'} attention")
        print(f"cu_seqlens shape: {cu_seqlens_now.shape}")
        print(f"cu_seqlens values: {cu_seqlens_now}")

        hidden_states = qwen2_5_vl_vision_block(
            hidden_states,
            cu_seqlens_now,
            state_dict["blocks"][str(block_idx)],
            num_heads,
            rotary_pos_emb,
            position_embeddings,
        )
        print(f"Output shape: {hidden_states.shape}")

    # Patch merging
    print("\nPatch merging:")
    print(f"Input shape: {hidden_states.shape}")
    hidden_states = qwen2_5_vl_patch_merger(hidden_states, state_dict["merger"])
    print(f"After merging: {hidden_states.shape}")

    # Restore original token order using pre-computed reverse_indices
    print("\nRestoring token order:")
    hidden_states = hidden_states[reverse_indices, :]
    print(f"Final output shape: {hidden_states.shape}")

    return hidden_states


def qwen2_5_vl_vision_block(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_dict: Dict,
    num_heads: int,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Functional stub for Qwen2_5_VLVisionBlock.

    Args:
        hidden_states: Input tensor
        cu_seqlens: Cumulative sequence lengths
        state_dict: Dictionary containing model weights
        num_heads: Number of attention heads
        rotary_pos_emb: Optional rotary position embeddings
        position_embeddings: Optional position embeddings tuple (cos, sin)

    state_dict structure:
    - norm1:
      - weight: First layer norm weight
    - norm2:
      - weight: Second layer norm weight
    - attn: Attention module weights
    - mlp: MLP module weights
    """
    print("\n  Vision Block:")
    print(f"  Input shape: {hidden_states.shape}")

    # Layer norm 1
    residual = hidden_states
    hidden_states = qwen2_rms_norm(hidden_states, {"weight": state_dict["norm1"]["weight"]})
    print(f"  After norm1: {hidden_states.shape}")

    # Attention
    hidden_states = qwen2_5_vl_vision_sdpa_attention(
        hidden_states, cu_seqlens, state_dict["attn"], num_heads, rotary_pos_emb, position_embeddings
    )
    print(f"  After attention: {hidden_states.shape}")
    hidden_states = residual + hidden_states
    print(f"  After residual1: {hidden_states.shape}")

    # Layer norm 2
    residual = hidden_states
    hidden_states = qwen2_rms_norm(hidden_states, {"weight": state_dict["norm2"]["weight"]})
    print(f"  After norm2: {hidden_states.shape}")

    # MLP
    hidden_states = qwen2_mlp(hidden_states, state_dict["mlp"])
    print(f"  After MLP: {hidden_states.shape}")
    hidden_states = residual + hidden_states
    print(f"  After residual2: {hidden_states.shape}")

    return hidden_states

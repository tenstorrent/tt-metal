# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Functional stubs for Qwen2.5-VL modules that match input/output shapes.
These are lightweight implementations for testing and development.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# This is a no-op function for compatibility with the TTNN implementation
def set_mesh_device(device):
    """No-op function for compatibility with TTNN implementation."""


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
    seq_len, hidden_size = x.shape
    head_dim = hidden_size // num_heads

    # QKV projection and reshape
    qkv = F.linear(x, state_dict["qkv"]["weight"], state_dict["qkv"]["bias"])
    q, k, v = qkv.view(seq_len, 3, num_heads, head_dim).permute(1, 0, 2, 3)

    # Apply rotary embeddings
    if rotary_pos_emb is not None:
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().float()
        sin = emb.sin().float()
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    elif position_embeddings is not None:
        q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)

    # Create attention mask
    mask = torch.ones(1, seq_len, seq_len, device=q.device, dtype=torch.bool)
    if cu_seqlens is not None:
        mask.zero_()
        for i in range(1, len(cu_seqlens)):
            s, e = cu_seqlens[i - 1 : i + 1]
            mask[..., s:e, s:e] = True

    # Attention and output projection
    attn_out = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), mask, dropout_p=0.0
    )

    output = F.linear(
        attn_out.transpose(0, 1).reshape(seq_len, -1), state_dict["proj"]["weight"], state_dict["proj"]["bias"]
    )

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


def qwen3_vl_rot_pos_emb(grid_thw: torch.Tensor, spatial_merge_size: int, head_dim: int) -> torch.Tensor:
    """Rotary position embedding for Qwen2.5 Vision Transformer.

    Args:
        grid_thw: Temporal, height, width dimensions for each image/video
        spatial_merge_size: Spatial merge size parameter
        head_dim: Attention head dimension
    Returns:
        Rotary position embeddings
    """
    merge_size = spatial_merge_size

    max_hw = int(grid_thw[:, 1:].max().item())
    freq_table = qwen2_5_vision_rotary_embedding(max_hw, head_dim)  # (max_hw, dim // 2)
    device = freq_table.device

    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    offset = 0
    for num_frames, height, width in grid_thw:
        merged_h, merged_w = height // merge_size, width // merge_size

        block_rows = torch.arange(merged_h, device=device)  # block row indices
        block_cols = torch.arange(merged_w, device=device)  # block col indices
        intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
        intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

        # Compute full-resolution positions
        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)

        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    embeddings = freq_table[pos_ids]  # lookup rotary embeddings
    embeddings = embeddings.flatten(1)
    return embeddings


def qwen3_vl_fast_pos_embed_interpolation(grid_thw: torch.Tensor, num_grid_per_side: int, pos_embed: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
    device = grid_thw.device

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for t, h, w in zip(grid_ts, grid_hs, grid_ws):
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * num_grid_per_side
        base_h_ceil = h_idxs_ceil * num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weight_list, dtype=pos_embed.weight.dtype, device=device)
    pos_embeds = pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

    patch_pos_embeds_permute = []
    merge_size = spatial_merge_size
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
    return patch_pos_embeds


def qwen2_5_vl_vision_block(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_dict: Dict,
    num_heads: int,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Functional implementation of a Qwen2.5 Vision Block.

    Args:
        hidden_states: Input tensor
        cu_seqlens: Cumulative sequence lengths
        state_dict: Dictionary containing model weights
        num_heads: Number of attention heads
        rotary_pos_emb: Optional rotary position embeddings
        position_embeddings: Optional position embeddings tuple (cos, sin)

    Returns:
        Transformed hidden states
    """
    # Layer norm 1
    residual = hidden_states
    hidden_states = qwen2_rms_norm(hidden_states, state_dict["norm1"])

    # Self-attention
    attn_output = qwen2_5_vl_vision_sdpa_attention(
        hidden_states, cu_seqlens, state_dict["attn"], num_heads, rotary_pos_emb, position_embeddings
    )

    # Add residual connection
    hidden_states = residual + attn_output

    # Layer norm 2
    residual = hidden_states
    hidden_states = qwen2_rms_norm(hidden_states, state_dict["norm2"])

    # MLP
    mlp_output = qwen2_5_vl_mlp(hidden_states, state_dict["mlp"])

    # Add residual connection
    hidden_states = residual + mlp_output

    return hidden_states


def qwen2_5_vision_transformer_preprocess(
    seq_len: int,
    grid_thw: torch.Tensor,
    head_dim: int,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
    num_grid_per_side: int,
    pos_embedding: torch.nn.Embedding,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Preprocesses input for Qwen2.5 Vision Transformer.

    Returns:
        Tuple containing:
        - cu_seqlens
        - position_embeddings tuple (cos, sin)
    """

    rotary_pos_emb = qwen3_vl_rot_pos_emb(grid_thw, spatial_merge_size, head_dim)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    return cu_seqlens, position_embeddings


def qwen2_5_vision_transformer_process(
    hidden_states: torch.Tensor,
    state_dict: Dict,
    cu_seqlens: torch.Tensor,
    cu_window_seqlens: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    window_index: torch.Tensor,
    num_heads: int,
    fullatt_block_indexes: List[int],
) -> torch.Tensor:
    """Processes the preprocessed input through transformer layers.

    Returns:
        Processed hidden states
    """
    for layer_num in sorted(state_dict["blocks"].keys()):
        state_blk = state_dict["blocks"][layer_num]
        if layer_num in fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
        hidden_states = qwen2_5_vl_vision_block(
            hidden_states, cu_seqlens_now, state_blk, num_heads, None, position_embeddings
        )

    hidden_states = qwen2_5_vl_patch_merger(hidden_states, state_dict["merger"])
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    return hidden_states


def qwen2_5_vision_transformer(
    hidden_states: torch.Tensor,
    state_dict: Dict,
    grid_thw: torch.Tensor,
    num_heads: int,
    head_dim: int,
    spatial_merge_size: int,
    window_size: int,
    fullatt_block_indexes: List[int],
    patch_size: int,
    temporal_patch_size: int,
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
        grid_thw: Optional grid dimensions for window calculations

    Returns:
        Transformed hidden states
    """

    # Apply patch embedding
    print(f"hidden_states: {hidden_states.shape}")
    hidden_states = qwen2_5_vision_patch_embed(
        hidden_states, state_dict["patch_embed"], patch_size, temporal_patch_size
    )
    print(f"hidden_states after patch embed: {hidden_states.shape}")
    seq_len, _ = hidden_states.size()

    # Preprocess the input
    cu_seqlens, cu_window_seqlens, position_embeddings, window_index = qwen2_5_vision_transformer_preprocess(
        seq_len=seq_len,
        grid_thw=grid_thw,
        head_dim=head_dim,
        spatial_merge_size=spatial_merge_size,
        window_size=window_size,
        patch_size=patch_size,
    )

    spatial_merge_unit = spatial_merge_size * spatial_merge_size
    hidden_states = hidden_states.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)

    # Process through transformer layers
    hidden_states = qwen2_5_vision_transformer_process(
        hidden_states=hidden_states,
        state_dict=state_dict,
        cu_seqlens=cu_seqlens,
        cu_window_seqlens=cu_window_seqlens,
        position_embeddings=position_embeddings,
        window_index=window_index,
        num_heads=num_heads,
        fullatt_block_indexes=fullatt_block_indexes,
    )

    return hidden_states


def qwen2_5_vision_rotary_embedding(seqlen: int, dim, theta: float = 10000.0, device=None) -> torch.Tensor:
    """Functional implementation of Qwen2_5_VisionRotaryEmbedding.

    Args:
        seqlen: Sequence length to generate embeddings for
        dim: Dimension of the embeddings
        theta: Base for the frequencies
        device: Device to create the embeddings on

    Returns:
        Rotary position embeddings
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
    seq = torch.arange(seqlen, device=device, dtype=torch.float)
    freqs = torch.outer(seq, inv_freq)
    return freqs

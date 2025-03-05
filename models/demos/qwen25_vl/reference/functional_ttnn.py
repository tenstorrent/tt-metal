# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Functional stubs for Qwen2.5-VL modules that match input/output shapes.
TTNN implementation version.
"""

import torch
import ttnn
from typing import Optional, Tuple, List, Dict
import atexit

# Global device variable to be set from outside
mesh_device = None


def set_mesh_device(device):
    """Set the mesh device to use for all operations."""
    global mesh_device
    mesh_device = device


# Function to get the device - creates a device if none was provided
def get_device():
    global mesh_device
    if mesh_device is None:
        # Create a default device if none was provided
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))

        # Register cleanup function to ensure device is closed properly
        def cleanup_ttnn_device():
            global mesh_device
            if mesh_device is not None:
                ttnn.close_mesh_device(mesh_device)
                mesh_device = None

        atexit.register(cleanup_ttnn_device)

    return mesh_device


def qwen2_rms_norm(x: torch.Tensor, state_dict: Dict, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm with weight scaling using TTNN."""
    # Get the current device
    device = get_device()

    # Convert input and weights to TTNN tensors
    tt_x = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    tt_weight = ttnn.from_torch(state_dict["weight"], layout=ttnn.TILE_LAYOUT, device=device)

    # Apply RMS norm
    tt_output = ttnn.rms_norm(tt_x, tt_weight, eps=eps, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Convert back to PyTorch
    output = ttnn.to_torch(tt_output)

    # Clean up
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_weight)
    ttnn.deallocate(tt_output)

    return output


def _mlp_forward(x: torch.Tensor, state_dict: Dict) -> torch.Tensor:
    """Common MLP forward pass for both Qwen2MLP and Qwen2_5_VLMLP using TTNN."""
    # Get the current device
    device = get_device()

    # Convert input tensor to TTNN
    tt_x = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)

    # In PyTorch, F.linear does: y = xA^T + b (where A is the weight matrix)
    # In TTNN, linear does: y = xA + b
    # So we need to transpose weights in PyTorch before converting to TTNN

    # Gate and Up weights are [intermediate_size, hidden_size] in PyTorch
    # Transpose in PyTorch to [hidden_size, intermediate_size]
    gate_weight_transposed = state_dict["gate_proj"]["weight"].transpose(0, 1)
    up_weight_transposed = state_dict["up_proj"]["weight"].transpose(0, 1)

    # Down weight is [hidden_size, intermediate_size] in PyTorch
    # Transpose in PyTorch to [intermediate_size, hidden_size]
    down_weight_transposed = state_dict["down_proj"]["weight"].transpose(0, 1)

    # Convert transposed weights to TTNN
    tt_gate_weight = ttnn.from_torch(gate_weight_transposed, layout=ttnn.TILE_LAYOUT, device=device)
    tt_up_weight = ttnn.from_torch(up_weight_transposed, layout=ttnn.TILE_LAYOUT, device=device)
    tt_down_weight = ttnn.from_torch(down_weight_transposed, layout=ttnn.TILE_LAYOUT, device=device)

    # Load biases if available
    tt_gate_bias = None
    if "bias" in state_dict["gate_proj"]:
        tt_gate_bias = ttnn.from_torch(state_dict["gate_proj"]["bias"], layout=ttnn.TILE_LAYOUT, device=device)

    tt_up_bias = None
    if "bias" in state_dict["up_proj"]:
        tt_up_bias = ttnn.from_torch(state_dict["up_proj"]["bias"], layout=ttnn.TILE_LAYOUT, device=device)

    tt_down_bias = None
    if "bias" in state_dict["down_proj"]:
        tt_down_bias = ttnn.from_torch(state_dict["down_proj"]["bias"], layout=ttnn.TILE_LAYOUT, device=device)

    # Define compute kernel config for good performance
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2)

    # First linear transformation for gate
    tt_gate_out = ttnn.linear(
        tt_x,
        tt_gate_weight,  # Already transposed in PyTorch
        bias=tt_gate_bias,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    # First linear transformation for up
    tt_up_out = ttnn.linear(
        tt_x,
        tt_up_weight,  # Already transposed in PyTorch
        bias=tt_up_bias,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    # Apply SiLU and multiply in one operation
    tt_gate_up = ttnn.multiply(
        tt_gate_out,
        tt_up_out,
        input_tensor_a_activation=ttnn.UnaryOpType.SILU,  # SiLU activation fused with multiply
        memory_config=tt_gate_out.memory_config(),
        dtype=ttnn.bfloat8_b,  # Use lower precision for intermediate result
    )

    # Final linear transformation
    tt_output = ttnn.linear(
        tt_gate_up,
        tt_down_weight,  # Already transposed in PyTorch
        bias=tt_down_bias,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    tt_output = ttnn.unsqueeze_to_4D(tt_output)

    output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(1, 3), mesh_shape=device.shape),
    )

    # Extract the first slice from dim 1 to match expected output shape
    output = output[:, :1, :, :]

    # Clean up
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_gate_weight)
    ttnn.deallocate(tt_up_weight)
    ttnn.deallocate(tt_down_weight)
    ttnn.deallocate(tt_gate_out)
    ttnn.deallocate(tt_up_out)
    ttnn.deallocate(tt_gate_up)
    ttnn.deallocate(tt_output)
    if tt_gate_bias is not None:
        ttnn.deallocate(tt_gate_bias)
    if tt_up_bias is not None:
        ttnn.deallocate(tt_up_bias)
    if tt_down_bias is not None:
        ttnn.deallocate(tt_down_bias)

    return output


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
    """Vision self-attention with rotary embeddings using TTNN."""
    seq_len, hidden_size = x.shape
    head_dim = hidden_size // num_heads

    # Get the current device
    device = get_device()

    # Convert input tensor to TTNN
    tt_x = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)

    # Convert weights to TTNN
    tt_qkv_weight = ttnn.from_torch(state_dict["qkv"]["weight"], layout=ttnn.TILE_LAYOUT, device=device)
    tt_qkv_bias = ttnn.from_torch(state_dict["qkv"]["bias"], layout=ttnn.TILE_LAYOUT, device=device)
    tt_proj_weight = ttnn.from_torch(state_dict["proj"]["weight"], layout=ttnn.TILE_LAYOUT, device=device)
    tt_proj_bias = ttnn.from_torch(state_dict["proj"]["bias"], layout=ttnn.TILE_LAYOUT, device=device)

    # Define compute kernel config
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4  # Higher precision for attention
    )

    # QKV projection
    tt_qkv = ttnn.linear(
        tt_x,
        tt_qkv_weight,
        bias=tt_qkv_bias,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    # For the reshape and rotary embedding, we need to go back to PyTorch
    qkv = ttnn.to_torch(tt_qkv)
    q, k, v = qkv.view(seq_len, 3, num_heads, head_dim).permute(1, 0, 2, 3)

    # Apply rotary embeddings using PyTorch
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

    # Attention and output projection using PyTorch
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), mask, dropout_p=0.0
    )

    # Convert attention output to TTNN
    reshaped_attn_out = attn_out.transpose(0, 1).reshape(seq_len, -1)
    tt_attn_out = ttnn.from_torch(reshaped_attn_out, layout=ttnn.TILE_LAYOUT, device=device)

    # Output projection
    tt_output = ttnn.linear(
        tt_attn_out,
        tt_proj_weight,
        bias=tt_proj_bias,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    # Convert back to PyTorch
    output = ttnn.to_torch(tt_output)

    # Clean up
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_qkv_weight)
    ttnn.deallocate(tt_qkv_bias)
    ttnn.deallocate(tt_proj_weight)
    ttnn.deallocate(tt_proj_bias)
    ttnn.deallocate(tt_qkv)
    ttnn.deallocate(tt_attn_out)
    ttnn.deallocate(tt_output)

    return output


def qwen2_5_vl_patch_merger(x: torch.Tensor, state_dict: Dict) -> torch.Tensor:
    """Functional stub for Qwen2_5_VLPatchMerger using TTNN."""
    # Get the hidden size from the weight matrix shape
    hidden_size = state_dict["mlp"]["0"]["weight"].shape[0]

    # Get the current device
    device = get_device()

    # Convert input tensor to TTNN
    tt_x = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)

    # Convert weight to TTNN
    tt_ln_weight = ttnn.from_torch(state_dict["ln_q"]["weight"], layout=ttnn.TILE_LAYOUT, device=device)

    # Normalize first
    tt_norm = ttnn.rms_norm(tt_x, tt_ln_weight, eps=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Reshape to match the expected input size (NOTE: current reshape only in PyTorch)
    norm = ttnn.to_torch(tt_norm)
    reshaped = norm.view(-1, hidden_size)
    tt_reshaped = ttnn.from_torch(reshaped, layout=ttnn.TILE_LAYOUT, device=device)

    # Convert MLP weights to TTNN
    tt_mlp0_weight = ttnn.from_torch(state_dict["mlp"]["0"]["weight"], layout=ttnn.TILE_LAYOUT, device=device)
    tt_mlp2_weight = ttnn.from_torch(state_dict["mlp"]["2"]["weight"], layout=ttnn.TILE_LAYOUT, device=device)

    # Define compute kernel config
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2)

    # Apply MLP layers
    tt_mlp0 = ttnn.linear(
        tt_reshaped,
        tt_mlp0_weight,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    # Apply GELU (need to go to PyTorch since TTNN doesn't have direct GELU support)
    mlp0 = ttnn.to_torch(tt_mlp0)
    gelu = torch.nn.functional.gelu(mlp0)
    tt_gelu = ttnn.from_torch(gelu, layout=ttnn.TILE_LAYOUT, device=device)

    # Apply second linear layer
    tt_output = ttnn.linear(
        tt_gelu,
        tt_mlp2_weight,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    # Convert back to PyTorch
    output = ttnn.to_torch(tt_output)

    # Clean up
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_ln_weight)
    ttnn.deallocate(tt_norm)
    ttnn.deallocate(tt_reshaped)
    ttnn.deallocate(tt_mlp0_weight)
    ttnn.deallocate(tt_mlp2_weight)
    ttnn.deallocate(tt_mlp0)
    ttnn.deallocate(tt_gelu)
    ttnn.deallocate(tt_output)

    return output


def qwen2_5_vision_patch_embed(
    hidden_states: torch.Tensor,
    state_dict: Dict,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
) -> torch.Tensor:
    """Patch embedding for Qwen2.5 Vision Transformer using TTNN."""
    # Validate input shape
    if hidden_states.dim() == 1:
        total_elements = hidden_states.size(0)
        expected_elements = 3 * temporal_patch_size * patch_size * patch_size
        if total_elements != expected_elements:
            raise ValueError(
                f"Input tensor size {total_elements} does not match expected size "
                f"{expected_elements} for patch_size={patch_size}, temporal_patch_size={temporal_patch_size}"
            )

    # Get the original shape information
    in_channels = state_dict["proj"]["weight"].shape[1]  # Number of input channels

    # Convert input to the target dtype of the projection weight
    target_dtype = state_dict["proj"]["weight"].dtype

    # Need to use PyTorch for reshaping and conv3d
    # Get input shape and handle reshaping appropriately
    if hidden_states.dim() == 1:
        # Handle single dimension input
        side_length = int((hidden_states.size(0) / (in_channels * temporal_patch_size)) ** 0.5)
        hidden_states = hidden_states.view(1, in_channels, temporal_patch_size, side_length, side_length)
    else:
        # Handle multi-dimensional input
        hidden_states = hidden_states.view(-1, in_channels, temporal_patch_size, patch_size, patch_size)

    # Apply 3D convolution using PyTorch
    hidden_states = torch.nn.functional.conv3d(
        hidden_states.to(dtype=target_dtype),
        state_dict["proj"]["weight"],
        bias=None,
        stride=[temporal_patch_size, patch_size, patch_size],
    )

    # Flatten the output
    hidden_states = hidden_states.view(-1, state_dict["proj"]["weight"].shape[0])

    # This operation remains in PyTorch as TTNN doesn't have direct support for 3D convolution
    return hidden_states


def qwen2_5_vl_rot_pos_emb(grid_thw: torch.Tensor, spatial_merge_size: int, head_dim: int) -> torch.Tensor:
    """Rotary position embedding for Qwen2.5 Vision Transformer."""
    # This function relies on indexing and reshaping operations that are more
    # naturally expressed in PyTorch, so we'll keep it as PyTorch implementation
    pos_ids = []
    for t, h, w in grid_thw:
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid_size = grid_thw[:, 1:].max()
    rotary_pos_emb_full = qwen2_5_vision_rotary_embedding(max_grid_size, head_dim // 2)
    rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
    return rotary_pos_emb


def qwen2_5_vl_get_window_index(
    grid_thw: torch.Tensor, window_size: int, spatial_merge_size: int, patch_size: int
) -> Tuple[torch.Tensor, List[int]]:
    """Get window index and cu_window_seqlens for Qwen2.5 Vision Transformer."""
    # This function relies heavily on complex PyTorch indexing and reshaping,
    # so we'll keep it as PyTorch implementation
    window_index: list = []
    cu_window_seqlens: list = [0]
    window_index_id = 0
    vit_merger_window_size = window_size // spatial_merge_size // patch_size
    spatial_merge_unit = spatial_merge_size * spatial_merge_size

    for grid_t, grid_h, grid_w in grid_thw:
        llm_grid_h, llm_grid_w = (
            grid_h // spatial_merge_size,
            grid_w // spatial_merge_size,
        )
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = torch.nn.functional.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_seqlens_tmp = seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
    window_index = torch.cat(window_index, dim=0)

    return window_index, cu_window_seqlens


def qwen2_5_vl_vision_block(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_dict: Dict,
    num_heads: int,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Functional implementation of a Qwen2.5 Vision Block using TTNN."""
    # Get the current device
    device = get_device()

    # Convert input tensor to TTNN
    tt_hidden_states = ttnn.from_torch(hidden_states, layout=ttnn.TILE_LAYOUT, device=device)

    # Convert norm1 weight to TTNN
    tt_norm1_weight = ttnn.from_torch(state_dict["norm1"]["weight"], layout=ttnn.TILE_LAYOUT, device=device)

    # Save residual
    tt_residual = tt_hidden_states

    # Layer norm 1
    tt_norm1 = ttnn.rms_norm(tt_hidden_states, tt_norm1_weight, eps=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Convert back to PyTorch for attention layer
    norm1 = ttnn.to_torch(tt_norm1)

    # Self-attention
    attn_output = qwen2_5_vl_vision_sdpa_attention(
        norm1, cu_seqlens, state_dict["attn"], num_heads, rotary_pos_emb, position_embeddings
    )

    # Convert attention output to TTNN
    tt_attn_output = ttnn.from_torch(attn_output, layout=ttnn.TILE_LAYOUT, device=device)

    # Add residual connection
    tt_hidden_states = ttnn.add(tt_residual, tt_attn_output, memory_config=tt_residual.memory_config())

    # Save residual for second block
    tt_residual = tt_hidden_states

    # Convert norm2 weight to TTNN
    tt_norm2_weight = ttnn.from_torch(state_dict["norm2"]["weight"], layout=ttnn.TILE_LAYOUT, device=device)

    # Layer norm 2
    tt_norm2 = ttnn.rms_norm(tt_hidden_states, tt_norm2_weight, eps=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Convert back to PyTorch for MLP
    norm2 = ttnn.to_torch(tt_norm2)

    # MLP
    mlp_output = qwen2_5_vl_mlp(norm2, state_dict["mlp"])

    # Convert MLP output to TTNN
    tt_mlp_output = ttnn.from_torch(mlp_output, layout=ttnn.TILE_LAYOUT, device=device)

    # Add residual connection
    tt_hidden_states = ttnn.add(tt_residual, tt_mlp_output, memory_config=tt_residual.memory_config())

    # Convert final output back to PyTorch
    hidden_states = ttnn.to_torch(tt_hidden_states)

    # Clean up
    ttnn.deallocate(tt_hidden_states)
    ttnn.deallocate(tt_norm1_weight)
    ttnn.deallocate(tt_residual)
    ttnn.deallocate(tt_norm1)
    ttnn.deallocate(tt_attn_output)
    ttnn.deallocate(tt_norm2_weight)
    ttnn.deallocate(tt_norm2)
    ttnn.deallocate(tt_mlp_output)

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
    """Functional implementation of Qwen2.5 Vision Transformer using TTNN where possible."""
    # Many of the operations here involve complex reshaping and indexing,
    # so we'll keep the overall structure in PyTorch and use TTNN for computational blocks

    spatial_merge_unit = spatial_merge_size * spatial_merge_size

    # Step 1: Apply patch embedding
    hidden_states = qwen2_5_vision_patch_embed(
        hidden_states, state_dict["patch_embed"], patch_size, temporal_patch_size
    )

    rotary_pos_emb = qwen2_5_vl_rot_pos_emb(grid_thw, spatial_merge_size, head_dim)
    window_index, cu_window_seqlens = qwen2_5_vl_get_window_index(grid_thw, window_size, spatial_merge_size, patch_size)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

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


def qwen2_5_vision_rotary_embedding(seqlen: int, dim, theta: float = 10000.0, device=None) -> torch.Tensor:
    """Functional implementation of Qwen2_5_VisionRotaryEmbedding."""
    # Using PyTorch for this operation as it's more of a setup step than a compute-intensive one
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
    seq = torch.arange(seqlen, device=device, dtype=torch.float)
    freqs = torch.outer(seq, inv_freq)
    return freqs

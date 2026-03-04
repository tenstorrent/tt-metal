# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Attention implementation for Qwen3-TTS.

Note: Qwen3-TTS uses non-interleaved RoPE (pairs dims i and i+64),
while TTNN rotary_embedding_llama uses interleaved format (pairs dims 2i and 2i+1).
This module handles the necessary dimension rearrangement.

Supports both prefill mode (full sequence) and decode mode (single token with KV cache).

IMPORTANT: Decode mode uses PyTorch-based RoPE computation because TTNN's
rotary_embedding_llama requires HEIGHT_SHARDED memory layout for decode mode,
which conflicts with the dimension rearrangement operations needed for
Qwen3's non-interleaved RoPE format.
"""

from typing import Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.rope import ttnn_rearrange_to_interleaved, ttnn_rearrange_to_noninterleaved


def apply_rope_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to Q and K tensors using PyTorch (for decode mode).

    Handles Qwen3's non-interleaved RoPE format.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine tensor [1, 1, seq_len, head_dim] (interleaved format)
        sin: Sine tensor [1, 1, seq_len, head_dim] (interleaved format)

    Returns:
        Tuple of (q_rotated, k_rotated) tensors
    """
    head_dim = q.shape[-1]
    half_dim = head_dim // 2

    def rotate_half_noninterleaved(x):
        """Rotate using non-interleaved format (pairs i, i+64)."""
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat([-x2, x1], dim=-1)

    # Convert cos/sin from interleaved [c0,c0,c1,c1,...] to non-interleaved [c0,c1,...,c0,c1,...]
    # cos/sin in TTNN format: [c0, c0, c1, c1, ...] - need unique values
    cos_unique = cos[..., 0::2]  # [c0, c1, c2, ...]
    sin_unique = sin[..., 0::2]  # [s0, s1, s2, ...]

    # Non-interleaved format: first half applies to dims 0-63, second half to dims 64-127
    cos_ni = torch.cat([cos_unique, cos_unique], dim=-1)
    sin_ni = torch.cat([sin_unique, sin_unique], dim=-1)

    # Apply rotation: x * cos + rotate_half(x) * sin
    q_rotated = q * cos_ni + rotate_half_noninterleaved(q) * sin_ni
    k_rotated = k * cos_ni + rotate_half_noninterleaved(k) * sin_ni

    return q_rotated, k_rotated


class Attention(LightweightModule):
    """
    Multi-head attention with GQA and QK-norm for Qwen3-TTS.

    Features:
    - Grouped Query Attention (GQA) with 16 heads and 8 KV heads
    - QK-normalization for stable training
    - RoPE positional embeddings (applied externally)

    This is a simplified implementation for single device (N150/N300).
    """

    def __init__(
        self,
        device,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        state_dict: dict,
        layer_prefix: str,
        rms_norm_eps: float = 1e-6,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = head_dim**-0.5
        self.rms_norm_eps = rms_norm_eps

        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        def get_cache_name(name):
            if weight_cache_path is None:
                return None
            return weight_cache_path / f"{layer_prefix}_{name}".replace(".", "_")

        # Load Q, K, V projection weights and create fused QKV weight
        q_proj_weight = state_dict[f"{layer_prefix}.self_attn.q_proj.weight"]
        k_proj_weight = state_dict[f"{layer_prefix}.self_attn.k_proj.weight"]
        v_proj_weight = state_dict[f"{layer_prefix}.self_attn.v_proj.weight"]
        o_proj_weight = state_dict[f"{layer_prefix}.self_attn.o_proj.weight"]

        # Fuse QKV weights: [hidden_size, (num_heads + 2*num_kv_heads) * head_dim]
        qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
        qkv_weight = torch.transpose(qkv_weight, -2, -1).unsqueeze(0).unsqueeze(0)

        self.wqkv = ttnn.as_tensor(
            qkv_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_cache_name("wqkv"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        # Output projection
        o_proj_weight = torch.transpose(o_proj_weight, -2, -1).unsqueeze(0).unsqueeze(0)
        self.wo = ttnn.as_tensor(
            o_proj_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_cache_name("wo"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        # QK-norm weights
        q_norm_weight = state_dict[f"{layer_prefix}.self_attn.q_norm.weight"]
        k_norm_weight = state_dict[f"{layer_prefix}.self_attn.k_norm.weight"]

        # Store as row-major for rms_norm
        TILE = 32
        q_norm_torch = q_norm_weight.unsqueeze(0).view(1, 1, head_dim).reshape([1, 1, head_dim // TILE, TILE])
        k_norm_torch = k_norm_weight.unsqueeze(0).view(1, 1, head_dim).reshape([1, 1, head_dim // TILE, TILE])

        self.q_norm_weight = ttnn.as_tensor(
            q_norm_torch,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_cache_name("q_norm"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.k_norm_weight = ttnn.as_tensor(
            k_norm_torch,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_cache_name("k_norm"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        transformation_mat: ttnn.Tensor,
        attention_mask: ttnn.Tensor = None,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        start_pos: int = 0,
        mode: str = "prefill",
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Apply multi-head attention with QK-norm and RoPE.

        Supports both prefill (full sequence) and decode (single token) modes.

        Args:
            x: Input tensor of shape [batch, 1, seq_len, hidden_size]
            cos: Cosine frequencies for RoPE [1, 1, seq_len, head_dim]
            sin: Sine frequencies for RoPE [1, 1, seq_len, head_dim]
            transformation_mat: Transformation matrix for RoPE
            attention_mask: Optional attention mask
            kv_cache: Optional tuple of (k_cache, v_cache) tensors for decode mode
                      Each has shape [batch, num_kv_heads, max_seq_len, head_dim]
            start_pos: Starting position in sequence (for KV cache indexing)
            mode: "prefill" for full sequence or "decode" for single token generation

        Returns:
            Tuple of (output, updated_kv_cache) where:
            - output: tensor of shape [batch, 1, seq_len, hidden_size]
            - updated_kv_cache: tuple of (k_cache, v_cache) or None if not using cache
        """
        batch_size = x.shape[0]
        seq_len = x.shape[-2]
        is_decode = mode == "decode"

        # Project QKV
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Split into Q, K, V heads
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        # Apply QK-norm
        q = ttnn.rms_norm(
            q,
            epsilon=self.rms_norm_eps,
            weight=self.q_norm_weight,
            compute_kernel_config=self.compute_kernel_config,
        )

        k = ttnn.rms_norm(
            k,
            epsilon=self.rms_norm_eps,
            weight=self.k_norm_weight,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Apply RoPE with dimension rearrangement
        # Qwen3-TTS uses non-interleaved RoPE (pairs i, i+64)
        # TTNN rotary_embedding_llama uses interleaved (pairs 2i, 2i+1)

        if q.dtype != ttnn.bfloat16:
            q = ttnn.typecast(q, dtype=ttnn.bfloat16)
        if k.dtype != ttnn.bfloat16:
            k = ttnn.typecast(k, dtype=ttnn.bfloat16)

        if is_decode:
            # Decode mode: Use PyTorch-based RoPE
            # TTNN's rotary_embedding_llama requires HEIGHT_SHARDED for decode,
            # but our dimension rearrangement ops don't work well with sharded tensors.
            # So we compute RoPE on CPU and transfer back.
            q_torch = ttnn.to_torch(q).float()
            k_torch = ttnn.to_torch(k).float()
            cos_torch = ttnn.to_torch(cos).float()
            sin_torch = ttnn.to_torch(sin).float()

            q_rotated, k_rotated = apply_rope_pytorch(q_torch, k_torch, cos_torch, sin_torch)

            q = ttnn.from_torch(
                q_rotated.to(torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            k = ttnn.from_torch(
                k_rotated.to(torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            # Prefill mode: Use TTNN RoPE (works with DRAM_MEMORY_CONFIG)
            # Rearrange to interleaved format for TTNN RoPE (pure TTNN - trace compatible)
            q = ttnn_rearrange_to_interleaved(q)
            k = ttnn_rearrange_to_interleaved(k)

            # Apply TTNN RoPE
            q = ttnn.experimental.rotary_embedding_llama(
                q,
                cos,
                sin,
                transformation_mat,
                is_decode_mode=False,
            )

            k = ttnn.experimental.rotary_embedding_llama(
                k,
                cos,
                sin,
                transformation_mat,
                is_decode_mode=False,
            )

            # Rearrange back to non-interleaved format (pure TTNN - trace compatible)
            q = ttnn_rearrange_to_noninterleaved(q)
            k = ttnn_rearrange_to_noninterleaved(k)

        # Handle KV cache for decode mode
        updated_kv_cache = None
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Update cache with new K, V at position start_pos
            # K, V shape: [batch, num_kv_heads, seq_len, head_dim]
            # Cache shape: [batch, num_kv_heads, max_seq_len, head_dim]
            # Use update_cache with update_idx for positional updates
            ttnn.update_cache(k_cache, k, update_idx=start_pos)
            ttnn.update_cache(v_cache, v, update_idx=start_pos)
            updated_kv_cache = (k_cache, v_cache)

            # For decode mode, use the full cached K, V
            # Slice the cache up to current position for attention
            cache_len = start_pos + seq_len
            # Use the cached values for attention computation
            k = ttnn.slice(k_cache, [0, 0, 0, 0], [batch_size, self.num_kv_heads, cache_len, self.head_dim])
            v = ttnn.slice(v_cache, [0, 0, 0, 0], [batch_size, self.num_kv_heads, cache_len, self.head_dim])

        # Keep bfloat16 for better precision (bfloat8_b can lose accuracy)
        # Note: For production, consider bfloat8_b for performance if PCC is acceptable

        # Scaled dot-product attention
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=self.scale,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )

        ttnn.deallocate(q)
        if kv_cache is None:
            # Only deallocate if not using cache (cache owns the memory)
            ttnn.deallocate(k)
            ttnn.deallocate(v)

        # Reshape: [batch, num_heads, seq_len, head_dim] -> [batch, 1, seq_len, hidden_size]
        attn_output = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Output projection
        output = ttnn.linear(
            attn_output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(attn_output)

        return output, updated_kv_cache

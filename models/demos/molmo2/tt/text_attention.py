# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Grouped-Query Attention for Molmo2 Text Model.

Implements GQA with:
- 32 query heads, 8 key/value heads (4:1 ratio)
- QK-norm (Qwen3-style normalization on Q and K)
- RoPE (Rotary Position Embeddings) with θ=1M using TTNN-native ops
- KV cache support for autoregressive generation

Weight layout:
- att_proj: fused QKV projection [hidden_dim, (num_heads + 2*num_kv_heads) * head_dim]
- attn_out: output projection [num_heads * head_dim, hidden_dim]

Uses ttnn.experimental.rotary_embedding_llama for efficient device-side RoPE.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TextAttention(LightweightModule):
    """
    Grouped-Query Attention with QK-norm for Molmo2 text model.
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        layer_num: int,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        max_seq_len: int = 8192,
        rope_theta: float = 1000000.0,
        weight_cache_path=None,
        state_dict_prefix: str = "model.transformer.blocks",
        dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize TextAttention.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Model state dict containing weights
            layer_num: Layer number (0-35)
            hidden_dim: Hidden dimension (4096)
            num_heads: Number of query heads (32)
            num_kv_heads: Number of key/value heads (8)
            head_dim: Dimension per head (128)
            max_seq_len: Maximum sequence length (8192)
            rope_theta: RoPE theta (1,000,000)
            weight_cache_path: Path to cache weights
            state_dict_prefix: Prefix for state dict keys
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.dtype = dtype

        self.scale = head_dim**-0.5

        # Tile alignment
        self.tile_size = 32
        self.padded_head_dim = math.ceil(head_dim / self.tile_size) * self.tile_size

        # Layer prefix
        prefix = f"{state_dict_prefix}.{layer_num}.self_attn"

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{prefix}.{name}"

        is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

        # Load fused QKV projection: att_proj
        # Shape: [hidden_dim, (num_heads + 2*num_kv_heads) * head_dim]
        att_proj = state_dict[f"{prefix}.att_proj.weight"]
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim

        # Split into Q, K, V
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        wq = att_proj[:q_dim, :]
        wk = att_proj[q_dim : q_dim + kv_dim, :]
        wv = att_proj[q_dim + kv_dim :, :]

        # Transpose for TTNN linear
        wq_t = torch.transpose(wq, -2, -1).unsqueeze(0).unsqueeze(0)
        wk_t = torch.transpose(wk, -2, -1).unsqueeze(0).unsqueeze(0)
        wv_t = torch.transpose(wv, -2, -1).unsqueeze(0).unsqueeze(0)

        self.wq = ttnn.as_tensor(
            wq_t,
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wq.weight"),
        )

        self.wk = ttnn.as_tensor(
            wk_t,
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wk.weight"),
        )

        self.wv = ttnn.as_tensor(
            wv_t,
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wv.weight"),
        )

        # Load output projection: attn_out
        wo = state_dict[f"{prefix}.attn_out.weight"]
        wo_t = torch.transpose(wo, -2, -1).unsqueeze(0).unsqueeze(0)

        self.wo = ttnn.as_tensor(
            wo_t,
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wo.weight"),
        )

        # Load QK-norm weights
        q_norm = state_dict[f"{prefix}.q_norm.weight"]
        k_norm = state_dict[f"{prefix}.k_norm.weight"]

        self.q_norm_weight = ttnn.as_tensor(
            q_norm.reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("q_norm.weight"),
        )

        self.k_norm_weight = ttnn.as_tensor(
            k_norm.reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("k_norm.weight"),
        )

        # Compute kernel configs
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        rot_mats: List[ttnn.Tensor],
        transformation_mats: Dict[str, ttnn.Tensor],
        attn_mask: Optional[ttnn.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Forward pass through GQA attention (prefill mode).

        Args:
            x: Input tensor of shape [1, 1, seq_len, hidden_dim]
            rot_mats: List of [cos, sin] rotation matrices
            transformation_mats: Dict with 'decode' and 'prefill' transformation matrices
            attn_mask: Optional causal mask
            start_pos: Starting position for KV cache
            kv_cache: Optional (k_cache, v_cache) tuple

        Returns:
            Tuple of (output, updated_kv_cache)
        """
        seq_len = x.shape[-2]

        # Q, K, V projections
        q = ttnn.linear(
            x,
            self.wq,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        k = ttnn.linear(
            x,
            self.wk,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        v = ttnn.linear(
            x,
            self.wv,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Reshape for multi-head attention
        # Q: [1, 1, seq_len, num_heads * head_dim] -> [1, num_heads, seq_len, head_dim]
        q = ttnn.reshape(q, [1, seq_len, self.num_heads, self.head_dim])
        q = ttnn.permute(q, (0, 2, 1, 3))

        # K, V: [1, 1, seq_len, num_kv_heads * head_dim] -> [1, num_kv_heads, seq_len, head_dim]
        k = ttnn.reshape(k, [1, seq_len, self.num_kv_heads, self.head_dim])
        k = ttnn.permute(k, (0, 2, 1, 3))

        v = ttnn.reshape(v, [1, seq_len, self.num_kv_heads, self.head_dim])
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Apply QK-norm (RMSNorm on Q and K)
        q = ttnn.rms_norm(q, weight=self.q_norm_weight, epsilon=1e-5)
        k = ttnn.rms_norm(k, weight=self.k_norm_weight, epsilon=1e-5)

        # Apply RoPE using TTNN-native op (prefill mode)
        # Ensure bfloat16 for rotary_embedding_llama
        if q.dtype != ttnn.bfloat16:
            q = ttnn.typecast(q, dtype=ttnn.bfloat16)
        if k.dtype != ttnn.bfloat16:
            k = ttnn.typecast(k, dtype=ttnn.bfloat16)

        q = ttnn.experimental.rotary_embedding_llama(
            q,
            rot_mats[0],  # cos
            rot_mats[1],  # sin
            transformation_mats["prefill"],
            is_decode_mode=False,
        )

        k = ttnn.experimental.rotary_embedding_llama(
            k,
            rot_mats[0],  # cos
            rot_mats[1],  # sin
            transformation_mats["prefill"],
            is_decode_mode=False,
        )

        # Handle KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Update cache at start_pos
            # For simplicity, concatenate for now
            k = ttnn.concat([k_cache, k], dim=2)
            v = ttnn.concat([v_cache, v], dim=2)

        new_kv_cache = (k, v)

        # Repeat K, V for GQA
        if self.num_kv_groups > 1:
            # [1, num_kv_heads, seq_len, head_dim] -> [1, num_heads, seq_len, head_dim]
            k = ttnn.repeat_interleave(k, self.num_kv_groups, dim=1)
            v = ttnn.repeat_interleave(v, self.num_kv_groups, dim=1)

        # Convert to bfloat8_b for SDPA
        q = ttnn.typecast(q, dtype=ttnn.bfloat8_b)
        k = ttnn.typecast(k, dtype=ttnn.bfloat8_b)
        v = ttnn.typecast(v, dtype=ttnn.bfloat8_b)

        # Scaled dot-product attention
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True if attn_mask is None else False,
            scale=self.scale,
            attn_mask=attn_mask,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )

        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Reshape back: [1, num_heads, seq_len, head_dim] -> [1, 1, seq_len, hidden_dim]
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, [1, 1, seq_len, self.num_heads * self.head_dim])

        # Output projection
        output = ttnn.linear(
            attn_output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(attn_output)

        return output, new_kv_cache

    def _apply_rotary_emb_torch(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor (PyTorch implementation for decode mode).

        Uses the formula: x_out = x * cos + rotate_half(x) * sin

        Args:
            x: Input tensor [1, num_heads, seq_len, head_dim]
            cos: Cosine values [1, 1, seq_len, head_dim]
            sin: Sine values [1, 1, seq_len, head_dim]

        Returns:
            Tensor with rotary embedding applied
        """
        # rotate_half: [x1, x2] -> [-x2, x1]
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        x_rotated = torch.cat((-x2, x1), dim=-1)

        return x * cos + x_rotated * sin

    def forward_decode(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        kv_cache: Tuple[ttnn.Tensor, ttnn.Tensor],
        current_pos: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Decode-mode forward pass with KV cache update.

        Uses paged_update_cache and scaled_dot_product_attention_decode
        for efficient autoregressive generation.

        Note: Uses PyTorch RoPE for decode mode to avoid HEIGHT_SHARDED requirement.
        Prefill mode uses TTNN-native RoPE for better performance.

        Args:
            x: Input tensor of shape [1, 1, 1, hidden_dim] (single token)
            cos: RoPE cosine values
            sin: RoPE sine values
            kv_cache: Tuple of (k_cache, v_cache) pre-allocated tensors
                      Shape: [batch, num_kv_heads, max_seq_len, head_dim]
            current_pos: Current decode position tensor [batch]

        Returns:
            Output tensor [1, 1, 1, hidden_dim]
        """
        batch_size = 1
        seq_len = 1  # Decode mode processes one token at a time

        # Q, K, V projections
        q = ttnn.linear(
            x,
            self.wq,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        k = ttnn.linear(
            x,
            self.wk,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        v = ttnn.linear(
            x,
            self.wv,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Reshape for multi-head attention
        # Q: [1, 1, 1, num_heads * head_dim] -> [1, num_heads, 1, head_dim]
        q = ttnn.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        q = ttnn.permute(q, (0, 2, 1, 3))

        # K, V: [1, 1, 1, num_kv_heads * head_dim] -> [1, num_kv_heads, 1, head_dim]
        k = ttnn.reshape(k, [batch_size, seq_len, self.num_kv_heads, self.head_dim])
        k = ttnn.permute(k, (0, 2, 1, 3))

        v = ttnn.reshape(v, [batch_size, seq_len, self.num_kv_heads, self.head_dim])
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Apply QK-norm (RMSNorm on Q and K)
        q = ttnn.rms_norm(q, weight=self.q_norm_weight, epsilon=1e-5)
        k = ttnn.rms_norm(k, weight=self.k_norm_weight, epsilon=1e-5)

        # Apply RoPE using PyTorch (for decode mode to avoid HEIGHT_SHARDED requirement)
        q_torch = ttnn.to_torch(q)
        k_torch = ttnn.to_torch(k)
        cos_torch = ttnn.to_torch(cos)
        sin_torch = ttnn.to_torch(sin)

        q_torch = self._apply_rotary_emb_torch(q_torch, cos_torch, sin_torch)
        k_torch = self._apply_rotary_emb_torch(k_torch, cos_torch, sin_torch)

        ttnn.deallocate(q)
        ttnn.deallocate(k)

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

        q = ttnn.from_torch(
            q_torch,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        k = ttnn.from_torch(
            k_torch,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # Get KV cache references
        k_cache, v_cache = kv_cache

        # Transpose Q, K, V for decode SDPA: [B, H, S, d] -> [S, B, H, d]
        q = ttnn.transpose(q, 0, 2)  # [B, H, S, d] -> [S, H, B, d]
        q = ttnn.transpose(q, 1, 2)  # [S, H, B, d] -> [S, B, H, d]
        k = ttnn.transpose(k, 0, 2)
        k = ttnn.transpose(k, 1, 2)
        v = ttnn.transpose(v, 0, 2)
        v = ttnn.transpose(v, 1, 2)

        # Update KV cache at current position
        ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=current_pos, page_table=None)
        ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=current_pos, page_table=None)

        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Convert to bfloat8_b for SDPA
        q = ttnn.typecast(q, dtype=ttnn.bfloat8_b)

        # Scaled dot-product attention decode
        # Uses the full KV cache up to current_pos
        attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=current_pos,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # Output: [1, B, H, d]

        ttnn.deallocate(q)

        # Transpose back: [1, B, H, d] -> [B, H, 1, d]
        attn_output = ttnn.transpose(attn_output, 1, 2)  # [1, B, H, d] -> [1, H, B, d]
        attn_output = ttnn.transpose(attn_output, 0, 2)  # [1, H, B, d] -> [B, H, 1, d]

        # Reshape back: [B, num_heads, 1, head_dim] -> [1, 1, 1, hidden_dim]
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, [1, 1, seq_len, self.num_heads * self.head_dim])

        # Output projection
        output = ttnn.linear(
            attn_output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(attn_output)

        return output

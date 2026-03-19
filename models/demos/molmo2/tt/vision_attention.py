# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Attention implementation for Molmo2 Vision Transformer.

The Molmo2 ViT uses standard multi-head attention with:
- Separate wq, wk, wv, wo linear projections (all with bias)
- Bidirectional attention (no causal mask)
- No rotary embeddings (positional info via learned pos embeddings)
"""

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class VisionAttention(LightweightModule):
    """
    Multi-head attention for Molmo2 Vision Transformer.

    Architecture:
        - Separate Q, K, V projections with bias
        - Scaled dot-product attention (bidirectional)
        - Output projection with bias
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix: str,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        weight_cache_path=None,
        dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize VisionAttention.

        Args:
            mesh_device: TTNN mesh device
            state_dict: Model state dict containing weights
            state_dict_prefix: Prefix for weight keys (e.g., "image_vit.transformer.resblocks.0.attn")
            hidden_dim: Model hidden dimension (1152 for Molmo2 ViT)
            num_heads: Number of attention heads (16 for Molmo2 ViT)
            head_dim: Dimension per head (72 for Molmo2 ViT)
            weight_cache_path: Path to cache weights
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.tile_size = 32

        # Pad head_dim to tile boundary if needed
        self.padded_head_dim = math.ceil(head_dim / self.tile_size) * self.tile_size

        # Scale factor for attention
        self.scale = head_dim**-0.5

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

        # Load separate Q, K, V, O weights and combine Q, K, V for efficient computation
        # We'll create a fused QKV weight for efficiency
        wq = state_dict[f"{state_dict_prefix}.wq.weight"]  # [hidden_dim, hidden_dim]
        wk = state_dict[f"{state_dict_prefix}.wk.weight"]  # [hidden_dim, hidden_dim]
        wv = state_dict[f"{state_dict_prefix}.wv.weight"]  # [hidden_dim, hidden_dim]

        bq = state_dict[f"{state_dict_prefix}.wq.bias"]  # [hidden_dim]
        bk = state_dict[f"{state_dict_prefix}.wk.bias"]  # [hidden_dim]
        bv = state_dict[f"{state_dict_prefix}.wv.bias"]  # [hidden_dim]

        # Handle head_dim padding if needed
        if self.head_dim != self.padded_head_dim:
            # Reshape and pad weights
            def pad_weight(w):
                # w: [num_heads * head_dim, hidden_dim]
                w = w.reshape(self.num_heads, self.head_dim, -1)
                w = torch.nn.functional.pad(w, (0, 0, 0, self.padded_head_dim - self.head_dim))
                return w.reshape(self.num_heads * self.padded_head_dim, -1)

            def pad_bias(b):
                # b: [num_heads * head_dim]
                b = b.reshape(self.num_heads, self.head_dim)
                b = torch.nn.functional.pad(b, (0, self.padded_head_dim - self.head_dim))
                return b.reshape(-1)

            wq = pad_weight(wq)
            wk = pad_weight(wk)
            wv = pad_weight(wv)
            bq = pad_bias(bq)
            bk = pad_bias(bk)
            bv = pad_bias(bv)

        # Transpose weights for linear: [hidden_dim, qkv_dim] -> [qkv_dim, hidden_dim]
        # Then concatenate Q, K, V
        wq_t = torch.transpose(wq, -2, -1)
        wk_t = torch.transpose(wk, -2, -1)
        wv_t = torch.transpose(wv, -2, -1)

        # Fused QKV: [hidden_dim, 3 * num_heads * padded_head_dim]
        wqkv = torch.cat([wq_t, wk_t, wv_t], dim=-1)
        bqkv = torch.cat([bq, bk, bv], dim=-1)

        self.wqkv = ttnn.as_tensor(
            wqkv.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wqkv.weight"),
        )

        self.bqkv = ttnn.as_tensor(
            bqkv,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wqkv.bias"),
        )

        # Output projection
        wo = state_dict[f"{state_dict_prefix}.wo.weight"]  # [hidden_dim, hidden_dim]
        bo = state_dict[f"{state_dict_prefix}.wo.bias"]  # [hidden_dim]

        # Handle padding for output projection input dimension
        if self.head_dim != self.padded_head_dim:
            # Pad input dimension (from concat_heads output)
            wo_reshaped = wo.reshape(-1, self.num_heads, self.head_dim)
            wo_padded = torch.nn.functional.pad(wo_reshaped, (0, self.padded_head_dim - self.head_dim))
            wo = wo_padded.reshape(-1, self.num_heads * self.padded_head_dim)

        wo_t = torch.transpose(wo, -2, -1)

        self.wo = ttnn.as_tensor(
            wo_t.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wo.weight"),
        )

        self.bo = ttnn.as_tensor(
            bo,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wo.bias"),
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

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass through attention.

        Args:
            x: Input tensor of shape [1, 1, seq_len, hidden_dim]

        Returns:
            Output tensor of shape [1, 1, seq_len, hidden_dim]
        """
        seq_len = x.shape[-2]

        # Reshape for long sequences
        if seq_len > 2048:
            x = ttnn.reshape(x, [1, seq_len // 2048, 2048, -1])

        # QKV projection (bias fused into linear to reduce matmul + binary op)
        qkv = ttnn.linear(
            x,
            self.wqkv,
            bias=self.bqkv,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Reshape back if needed
        if seq_len > 2048:
            qkv = ttnn.reshape(qkv, [1, 1, seq_len, -1])

        ttnn.deallocate(x)

        # Split Q, K, V and reshape to heads
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,  # Full attention, not GQA
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(qkv)

        # Convert to bfloat8 for SDPA
        q = ttnn.typecast(q, dtype=ttnn.bfloat8_b)
        k = ttnn.typecast(k, dtype=ttnn.bfloat8_b)
        v = ttnn.typecast(v, dtype=ttnn.bfloat8_b)

        # Scaled dot-product attention (bidirectional - no causal mask)
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )

        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Reshape attention output: [1, num_heads, seq_len, head_dim] -> [1, 1, seq_len, hidden_dim]
        attn_output = ttnn.reshape(attn_output, [1, self.num_heads, -1, self.padded_head_dim])

        # Concatenate heads
        attn_output = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Output projection
        if seq_len > 1024:
            attn_output = ttnn.reshape(attn_output, [1, seq_len // 1024, 1024, -1])

        # Output projection (bias fused into linear to reduce matmul + binary op)
        output = ttnn.linear(
            attn_output,
            self.wo,
            bias=self.bo,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if seq_len > 1024:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        ttnn.deallocate(attn_output)

        return output

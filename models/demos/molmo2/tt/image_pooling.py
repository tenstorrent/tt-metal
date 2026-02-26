# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Image Pooling for Molmo2 Vision Adapter.

This module implements 2D image pooling using cross-attention, where:
- Query: mean of gathered patch features from pooled_patches_idx neighborhoods
- Keys/Values: gathered patch features

The pooled_patches_idx tensor maps each output visual token to a neighborhood
of K source ViT patches. This is computed by the image processor on CPU.

Dimensions:
    - input_dim: 2304 (1152 * 2, concat of ViT layers 18 and 24)
    - hidden_dim: 1152 (adapter hidden size)
    - num_heads: 16
    - head_dim: 72 (1152 / 16)

All wq, wk, wv, wo projections have bias.
"""

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class ImagePooling(LightweightModule):
    """
    Cross-attention based image pooling for Molmo2.

    Pools multi-scale ViT features using attention with gathered patch neighborhoods.
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        input_dim: int = 2304,
        hidden_dim: int = 1152,
        num_heads: int = 16,
        head_dim: int = 72,
        weight_cache_path=None,
        state_dict_prefix: str = "model.vision_backbone.image_pooling_2d",
        dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize ImagePooling.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Model state dict containing weights
            input_dim: Input dimension (2304 = 1152 * 2 for multi-scale)
            hidden_dim: Hidden dimension for attention (1152)
            num_heads: Number of attention heads (16)
            head_dim: Dimension per head (72)
            weight_cache_path: Path to cache weights
            state_dict_prefix: Prefix for state dict keys
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.input_dim = input_dim
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

        # Load wq: input_dim (2304) -> hidden_dim (1152)
        wq = state_dict[f"{state_dict_prefix}.wq.weight"]
        bq = state_dict[f"{state_dict_prefix}.wq.bias"]

        # Handle head_dim padding if needed
        if self.head_dim != self.padded_head_dim:
            wq = self._pad_weight(wq)
            bq = self._pad_bias(bq)

        wq_t = torch.transpose(wq, -2, -1)

        self.wq = ttnn.as_tensor(
            wq_t.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wq.weight"),
        )

        self.bq = ttnn.as_tensor(
            bq,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wq.bias"),
        )

        # Load wk: input_dim (2304) -> hidden_dim (1152)
        wk = state_dict[f"{state_dict_prefix}.wk.weight"]
        bk = state_dict[f"{state_dict_prefix}.wk.bias"]

        if self.head_dim != self.padded_head_dim:
            wk = self._pad_weight(wk)
            bk = self._pad_bias(bk)

        wk_t = torch.transpose(wk, -2, -1)

        self.wk = ttnn.as_tensor(
            wk_t.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wk.weight"),
        )

        self.bk = ttnn.as_tensor(
            bk,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wk.bias"),
        )

        # Load wv: input_dim (2304) -> hidden_dim (1152)
        wv = state_dict[f"{state_dict_prefix}.wv.weight"]
        bv = state_dict[f"{state_dict_prefix}.wv.bias"]

        if self.head_dim != self.padded_head_dim:
            wv = self._pad_weight(wv)
            bv = self._pad_bias(bv)

        wv_t = torch.transpose(wv, -2, -1)

        self.wv = ttnn.as_tensor(
            wv_t.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wv.weight"),
        )

        self.bv = ttnn.as_tensor(
            bv,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wv.bias"),
        )

        # Load wo: hidden_dim (1152) -> hidden_dim (1152)
        wo = state_dict[f"{state_dict_prefix}.wo.weight"]
        bo = state_dict[f"{state_dict_prefix}.wo.bias"]

        # Handle padding for output projection
        if self.head_dim != self.padded_head_dim:
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

    def _pad_weight(self, w):
        """Pad weight tensor for non-tile-aligned head_dim."""
        w = w.reshape(self.num_heads, self.head_dim, -1)
        w = torch.nn.functional.pad(w, (0, 0, 0, self.padded_head_dim - self.head_dim))
        return w.reshape(self.num_heads * self.padded_head_dim, -1)

    def _pad_bias(self, b):
        """Pad bias tensor for non-tile-aligned head_dim."""
        b = b.reshape(self.num_heads, self.head_dim)
        b = torch.nn.functional.pad(b, (0, self.padded_head_dim - self.head_dim))
        return b.reshape(-1)

    def forward(
        self,
        query: ttnn.Tensor,
        key_value: ttnn.Tensor,
        attn_mask: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        """
        Forward pass through cross-attention pooling.

        Args:
            query: Query tensor of shape [1, 1, num_queries, input_dim]
                   (typically mean of gathered features)
            key_value: Key/Value tensor of shape [1, 1, pool_size, input_dim]
                       (gathered patch features)
            attn_mask: Optional attention mask [1, 1, 1, pool_size]

        Returns:
            Pooled features of shape [1, 1, num_queries, hidden_dim]
        """
        num_queries = query.shape[-2]
        pool_size = key_value.shape[-2]

        # Q projection
        q = ttnn.linear(
            query,
            self.wq,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q = q + self.bq

        # K projection
        k = ttnn.linear(
            key_value,
            self.wk,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k = k + self.bk

        # V projection
        v = ttnn.linear(
            key_value,
            self.wv,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v = v + self.bv

        # Reshape Q, K, V for multi-head attention
        # Note: Using ttnn ops for head splitting

        # For cross-attention, we need special handling
        # Get batch dimensions from input tensors
        batch_seq = query.shape[1]  # B*N_out from vision_backbone
        padded_hidden = self.num_heads * self.padded_head_dim

        # Q: [1, batch_seq, num_queries, padded_hidden] -> [batch_seq, num_heads, num_queries, head_dim]
        # K: [1, batch_seq, pool_size, padded_hidden] -> [batch_seq, num_heads, pool_size, head_dim]
        # V: [1, batch_seq, pool_size, padded_hidden] -> [batch_seq, num_heads, pool_size, head_dim]

        q = ttnn.reshape(q, [batch_seq, num_queries, self.num_heads, self.padded_head_dim])
        q = ttnn.permute(q, (0, 2, 1, 3))
        q = ttnn.typecast(q, dtype=ttnn.bfloat8_b)

        k = ttnn.reshape(k, [batch_seq, pool_size, self.num_heads, self.padded_head_dim])
        k = ttnn.permute(k, (0, 2, 1, 3))
        k = ttnn.typecast(k, dtype=ttnn.bfloat8_b)

        v = ttnn.reshape(v, [batch_seq, pool_size, self.num_heads, self.padded_head_dim])
        v = ttnn.permute(v, (0, 2, 1, 3))
        v = ttnn.typecast(v, dtype=ttnn.bfloat8_b)

        # Scaled dot-product attention
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            attn_mask=attn_mask,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )

        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Reshape back: [batch_seq, num_heads, num_queries, head_dim] -> [1, batch_seq, num_queries, hidden_dim]
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, [1, batch_seq, num_queries, self.num_heads * self.padded_head_dim])

        # Output projection
        output = ttnn.linear(
            attn_output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output = output + self.bo

        ttnn.deallocate(attn_output)

        return output

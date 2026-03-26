# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Attention implementation for Molmo2 Vision Transformer.

The Molmo2 ViT uses standard multi-head attention with:
- Separate wq, wk, wv linear projections (all with bias), fused as QKV for efficiency
- Bidirectional attention (no causal mask)
- No rotary embeddings (positional info via learned pos embeddings)

On a multi-device mesh, tensor parallelism matches Molmo2 text attention:
- Column-parallel fused QKV (heads sharded along the fused projection dim)
- Row-parallel output projection with all-reduce
Prefill ViT does not use forward_decode-style L1 width-sharded linears: those assume
very small M (e.g. one decode token); native seq_len ~729 hits width-sharded matmul
output spec limits. DRAM linears preserve PCC and correctness.
"""

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class VisionAttention(LightweightModule):
    """
    Multi-head attention for Molmo2 Vision Transformer.

    Architecture:
        - Fused QKV projection with bias
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
        self.is_mesh_device = is_mesh_device
        self.num_devices = mesh_device.get_num_devices() if is_mesh_device else 1

        # Same TP pattern as text attention: column QKV, row output, all-reduce.
        self.use_tensor_parallel = is_mesh_device and self.num_devices > 1 and num_heads % self.num_devices == 0
        if self.use_tensor_parallel:
            self.num_heads_per_device = num_heads // self.num_devices
            col_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
            row_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
            norm_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
            mesh_mapper_replicated = None
        else:
            self.num_heads_per_device = num_heads
            col_mesh_mapper = None
            row_mesh_mapper = None
            norm_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None
            mesh_mapper_replicated = norm_mesh_mapper

        # Load separate Q, K, V, O weights and combine Q, K, V for efficient computation
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
        wq_t = torch.transpose(wq, -2, -1)
        wk_t = torch.transpose(wk, -2, -1)
        wv_t = torch.transpose(wv, -2, -1)

        if self.use_tensor_parallel:
            # Multi-device: fused QKV with per-device head chunks (same layout as text_attention wqkv)
            qkv_list = []
            for i in range(self.num_devices):
                wq_chunk = torch.chunk(wq, self.num_devices, dim=0)[i]
                wk_chunk = torch.chunk(wk, self.num_devices, dim=0)[i]
                wv_chunk = torch.chunk(wv, self.num_devices, dim=0)[i]
                wq_chunk_t = torch.transpose(wq_chunk, -2, -1)
                wk_chunk_t = torch.transpose(wk_chunk, -2, -1)
                wv_chunk_t = torch.transpose(wv_chunk, -2, -1)
                qkv_chunk = torch.cat([wq_chunk_t, wk_chunk_t, wv_chunk_t], dim=-1)
                qkv_list.append(qkv_chunk)
            wqkv = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

            bias_parts = []
            bq_r = bq.reshape(self.num_heads, self.padded_head_dim)
            bk_r = bk.reshape(self.num_heads, self.padded_head_dim)
            bv_r = bv.reshape(self.num_heads, self.padded_head_dim)
            for i in range(self.num_devices):
                sl = slice(
                    i * self.num_heads_per_device,
                    (i + 1) * self.num_heads_per_device,
                )
                piece = torch.cat(
                    [
                        bq_r[sl].reshape(-1),
                        bk_r[sl].reshape(-1),
                        bv_r[sl].reshape(-1),
                    ],
                    dim=0,
                )
                bias_parts.append(piece)
            bqkv_torch = torch.cat(bias_parts, dim=0).reshape(1, 1, 1, -1)
            bqkv_mapper = col_mesh_mapper
            wqkv_mapper = col_mesh_mapper
        else:
            wqkv = torch.cat([wq_t, wk_t, wv_t], dim=-1).unsqueeze(0).unsqueeze(0)
            bqkv_torch = torch.cat([bq, bk, bv], dim=0)
            bqkv_mapper = mesh_mapper_replicated
            wqkv_mapper = mesh_mapper_replicated

        self.wqkv = ttnn.as_tensor(
            wqkv,
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=wqkv_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wqkv.weight"),
        )

        self.bqkv = ttnn.as_tensor(
            bqkv_torch,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=bqkv_mapper,
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
            mesh_mapper=row_mesh_mapper if self.use_tensor_parallel else mesh_mapper_replicated,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wo.weight"),
        )

        self.bo = ttnn.as_tensor(
            bo,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=norm_mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wo.bias"),
        )

        # Match text_attention.py forward / forward_decode linear fidelity
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
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
        heads = self.num_heads_per_device

        # Reshape for long sequences (only when divisible for memory optimization)
        if seq_len > 2048 and seq_len % 2048 == 0:
            x = ttnn.reshape(x, [1, seq_len // 2048, 2048, -1])

        # QKV projection — DRAM for prefill (see module docstring); matches PCC
        qkv = ttnn.linear(
            x,
            self.wqkv,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Add bias
        qkv = qkv + self.bqkv

        # Reshape back if needed
        if seq_len > 2048 and seq_len % 2048 == 0:
            qkv = ttnn.reshape(qkv, [1, 1, seq_len, -1])

        ttnn.deallocate(x)

        # Split Q, K, V and reshape to heads (per-device head count when tensor-parallel)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=heads,
            num_kv_heads=heads,
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

        # Reshape attention output: [1, num_heads, seq_len, head_dim] -> concat input layout
        attn_output = ttnn.reshape(attn_output, [1, heads, -1, self.padded_head_dim])

        # Concatenate heads
        attn_output = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Output projection (reshape only when divisible for memory optimization)
        if seq_len > 1024 and seq_len % 1024 == 0:
            attn_output = ttnn.reshape(attn_output, [1, seq_len // 1024, 1024, -1])

        output = ttnn.linear(
            attn_output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(attn_output)

        if self.use_tensor_parallel:
            output = ttnn.all_reduce(
                output,
                cluster_axis=1,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Add output bias (full hidden_dim; after all-reduce when tensor-parallel)
        output = output + self.bo

        if seq_len > 1024 and seq_len % 1024 == 0:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        return output

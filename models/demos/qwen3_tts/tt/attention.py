# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Attention implementation for Qwen3-TTS.

Note: Qwen3-TTS uses non-interleaved RoPE (pairs dims i and i+64),
while TTNN rotary_embedding_llama uses interleaved format (pairs dims 2i and 2i+1).
This module handles the necessary dimension rearrangement.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.rope import rearrange_to_interleaved, rearrange_to_noninterleaved


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
    ) -> ttnn.Tensor:
        """
        Apply multi-head attention with QK-norm and RoPE.

        Args:
            x: Input tensor of shape [batch, 1, seq_len, hidden_size]
            cos: Cosine frequencies for RoPE [1, 1, seq_len, head_dim]
            sin: Sine frequencies for RoPE [1, 1, seq_len, head_dim]
            transformation_mat: Transformation matrix for RoPE
            attention_mask: Optional attention mask

        Returns:
            Output tensor of shape [batch, 1, seq_len, hidden_size]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[-2]

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
        # We rearrange Q/K before RoPE and back after

        if q.dtype != ttnn.bfloat16:
            q = ttnn.typecast(q, dtype=ttnn.bfloat16)
        if k.dtype != ttnn.bfloat16:
            k = ttnn.typecast(k, dtype=ttnn.bfloat16)

        # Rearrange to interleaved format for TTNN RoPE
        # Convert to PyTorch, rearrange, convert back (TODO: optimize with TTNN ops)
        q_torch = ttnn.to_torch(q)
        k_torch = ttnn.to_torch(k)

        q_interleaved = rearrange_to_interleaved(q_torch)
        k_interleaved = rearrange_to_interleaved(k_torch)

        q = ttnn.from_torch(
            q_interleaved.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k = ttnn.from_torch(
            k_interleaved.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

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

        # Rearrange back to non-interleaved format
        q_torch = ttnn.to_torch(q)
        k_torch = ttnn.to_torch(k)

        q_final = rearrange_to_noninterleaved(q_torch)
        k_final = rearrange_to_noninterleaved(k_torch)

        q = ttnn.from_torch(
            q_final.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k = ttnn.from_torch(
            k_final.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

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

        return output

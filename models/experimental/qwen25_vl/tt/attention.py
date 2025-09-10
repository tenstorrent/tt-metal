"""
This is the vision attention implementation for Qwen-VL-7B.

We couldn't reuse the LLaMA version from tt_transformers because it expects separate q, k, v weights,
but Qwen-VL uses fused qkv weights. So this has been rewritten to support that,
based on the original code at:
models/tt_transformers/tt/multimodal/llama_image_attention.py
"""


import torch
import ttnn
from models.common.lightweightmodule import LightweightModule


def rotate_half(x):
    x1 = ttnn.slice(x, (0, 0, 0), (x.shape[0], x.shape[1], x.shape[2] // 2))
    x2 = ttnn.slice(x, (0, 0, x.shape[-1] // 2), (x.shape[0], x.shape[1], x.shape[2]))
    return ttnn.concat([ttnn.mul(x2, -1, use_legacy=False), x1], dim=-1)


def apply_rotary_pos_emb_vision_tt(q, k, cos, sin):
    cos = ttnn.unsqueeze(cos, -2)
    sin = ttnn.unsqueeze(sin, -2)

    q_embed = ttnn.add(ttnn.mul(q, cos), ttnn.mul(rotate_half(q), sin))
    k_embed = ttnn.add(ttnn.mul(k, cos), ttnn.mul(rotate_half(k), sin))
    return q_embed, k_embed


class TtQwen2_5_VLVisionSdpaAttention(LightweightModule):
    def __init__(self, mesh_device, state_dict, state_dict_prefix, dtype, configuration):
        super().__init__()

        self.mesh_device = mesh_device
        self.dtype = dtype
        self.hidden_size = 1280
        self.num_heads = 16
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        # Load qkv weight & bias (fused): shape [hidden_size, hidden_size*3]
        qkv_weight = state_dict[f"{state_dict_prefix}qkv.weight"]
        qkv_bias = state_dict[f"{state_dict_prefix}qkv.bias"]

        # Transpose to [hidden_size, 3*hidden_size] for matmul
        self.qkv_weight = ttnn.as_tensor(
            torch.transpose(qkv_weight, -2, -1),
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.qkv_bias = ttnn.as_tensor(
            qkv_bias, device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Output projection: proj
        proj_weight = state_dict[f"{state_dict_prefix}proj.weight"]  # shape [hidden_size, hidden_size]
        proj_bias = state_dict[f"{state_dict_prefix}proj.bias"]  # shape [hidden_size]

        self.proj_weight = ttnn.as_tensor(
            torch.transpose(proj_weight, -2, -1),
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.proj_bias = ttnn.as_tensor(
            proj_bias, device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=False,
        )

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        """
        hidden_states: ttnn.Tensor of shape [batch, seq_len, hidden_size]
        position_embeddings: tuple (cos, sin) each of shape [seq_len, head_dim]
        """
        seq_len = hidden_states.shape[-2]
        cos, sin = position_embeddings
        # Fused qkv projection
        qkv = ttnn.linear(
            hidden_states,
            self.qkv_weight,
            bias=self.qkv_bias,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )  # shape [batch, seq_len, hidden_size*3]

        (q, k, v) = ttnn.permute(ttnn.reshape(qkv, [seq_len, 3, self.num_heads, -1]), [1, 0, 2, 3])
        ttnn.deallocate(qkv)

        # Apply rotary position embeddings
        q, k = apply_rotary_pos_emb_vision_tt(q, k, cos, sin)
        # return q

        seq_len = cu_seqlens[-1].item()

        q = ttnn.unsqueeze(ttnn.permute(ttnn.pad(q, [(0, 0), (0, 0), (0, 16)], 0), [1, 0, 2]), 0)
        k = ttnn.unsqueeze(ttnn.permute(ttnn.pad(k, [(0, 0), (0, 0), (0, 16)], 0), [1, 0, 2]), 0)
        v = ttnn.unsqueeze(ttnn.permute(ttnn.pad(v, [(0, 0), (0, 0), (0, 16)], 0), [1, 0, 2]), 0)

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=self.scale
        )  # shape [1, seq_len, num_heads, head_dim]

        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # attn_output shape: [1, 16, 4096, 96]
        # Need to slice back from 96 → 80
        attn_output = ttnn.slice(
            attn_output,
            (0, 0, 0, 0),
            (attn_output.shape[0], attn_output.shape[1], attn_output.shape[2], self.head_dim),  # head_dim=80
        )

        attn_output = ttnn.permute(ttnn.squeeze(attn_output, 0), [1, 0, 2])
        attn_output = ttnn.reshape(attn_output, [seq_len, -1])

        # Final projection
        output = ttnn.linear(
            attn_output,
            self.proj_weight,
            bias=self.proj_bias,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        ttnn.deallocate(attn_output)

        return output

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_gn_mask,
    prepare_gn_beta_gamma,
    prepare_linear_params,
)


class TtAttention(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        query_dim: int,
        heads: int = 8,
        out_dim: int = None,
        kv_heads=None,
        dim_head: int = 64,
        weights_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device

        self.norm_core_grid = ttnn.CoreGrid(y=4, x=8)
        self.norm_groups = 32
        self.norm_eps = 1e-6
        self.num_out_blocks = 4

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=64,
            k_chunk_size=64,
            exp_approx_mode=False,
        )

        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        norm_weights = state_dict[f"{module_path}.group_norm.weight"]
        norm_bias = state_dict[f"{module_path}.group_norm.bias"]
        self.gamma_t, self.beta_t = prepare_gn_beta_gamma(device, norm_weights, norm_bias, self.norm_core_grid.y)
        self.input_mask = prepare_gn_mask(self.device, norm_weights.shape[0], self.norm_groups, self.norm_core_grid.y)

        q_weights = state_dict[f"{module_path}.to_q.weight"].unsqueeze(0).unsqueeze(0)
        q_bias = state_dict[f"{module_path}.to_q.bias"]
        k_weights = state_dict[f"{module_path}.to_k.weight"].unsqueeze(0).unsqueeze(0)
        k_bias = state_dict[f"{module_path}.to_k.bias"]
        v_weights = state_dict[f"{module_path}.to_v.weight"].unsqueeze(0).unsqueeze(0)
        v_bias = state_dict[f"{module_path}.to_v.bias"]

        fused_qkv_weights = torch.cat(
            [
                torch.transpose(q_weights, -2, -1),
                torch.transpose(k_weights, -2, -1),
                torch.transpose(v_weights, -2, -1),
            ],
            dim=-1,
        )
        self.tt_qkv_weights = ttnn.from_torch(fused_qkv_weights, weights_dtype, device=device, layout=ttnn.TILE_LAYOUT)
        fused_qkv_bias = torch.concat(
            [q_bias, k_bias, v_bias],
            dim=-1,
        ).unsqueeze(0)
        self.tt_qkv_bias = ttnn.from_torch(fused_qkv_bias, weights_dtype, device=device, layout=ttnn.TILE_LAYOUT)

        out_weights = state_dict[f"{module_path}.to_out.0.weight"].unsqueeze(0).unsqueeze(0)
        out_bias = state_dict[f"{module_path}.to_out.0.bias"]

        self.tt_out_weights, self.tt_out_bias = prepare_linear_params(device, out_weights, out_bias, ttnn.bfloat16)

    def forward(self, input_tensor, input_shape, encoder_hidden_states=None):
        hidden_states = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=self.norm_groups,
            input_mask=self.input_mask,
            weight=self.gamma_t,
            bias=self.beta_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=self.norm_core_grid,
            epsilon=self.norm_eps,
            inplace=False,
            num_out_blocks=self.num_out_blocks,
        )

        assert encoder_hidden_states is None, "VAE does self attention only"
        encoder_hidden_states = hidden_states

        qkv_fused = ttnn.linear(
            hidden_states,
            self.tt_qkv_weights,
            bias=self.tt_qkv_bias,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

        (
            q_heads,
            k_heads,
            v_heads,
        ) = ttnn.experimental.nlp_create_qkv_heads(
            qkv_fused, num_heads=self.heads, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv_fused)

        hidden_states = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=False,
            attn_mask=None,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        hidden_states = ttnn.experimental.nlp_concat_heads(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_out_weights,
            bias=self.tt_out_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        hidden_states = ttnn.add(hidden_states, input_tensor)

        return hidden_states

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

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
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
        )

        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
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

        out_weights = state_dict[f"{module_path}.to_out.0.weight"].unsqueeze(0).unsqueeze(0)
        out_bias = state_dict[f"{module_path}.to_out.0.bias"]

        self.tt_q_weights, self.tt_q_bias = prepare_linear_params(device, q_weights, q_bias, ttnn.bfloat16)
        self.tt_k_weights, self.tt_k_bias = prepare_linear_params(device, k_weights, k_bias, ttnn.bfloat16)
        self.tt_v_weights, self.tt_v_bias = prepare_linear_params(device, v_weights, v_bias, ttnn.bfloat16)
        self.tt_out_weights, self.tt_out_bias = prepare_linear_params(device, out_weights, out_bias, ttnn.bfloat16)

    def forward(self, input_tensor, input_shape, encoder_hidden_states=None):
        B, C, H, W = input_shape

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
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = ttnn.linear(
            hidden_states,
            self.tt_q_weights,
            bias=self.tt_q_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        key = ttnn.linear(
            encoder_hidden_states,
            self.tt_k_weights,
            bias=self.tt_k_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        value = ttnn.linear(
            encoder_hidden_states,
            self.tt_v_weights,
            bias=self.tt_v_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        inner_dim = list(key.shape)[-1]
        head_dim = inner_dim // self.heads

        query = ttnn.reshape(query, [B, -1, self.heads, head_dim])
        query = ttnn.transpose(query, 1, 2)

        key = ttnn.reshape(key, [B, -1, self.heads, head_dim])
        key = ttnn.transpose(key, 1, 2)

        value = ttnn.reshape(value, [B, -1, self.heads, head_dim])
        value = ttnn.transpose(value, 1, 2)

        hidden_states = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=False,
            attn_mask=None,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        hidden_states = ttnn.transpose(hidden_states, 1, 2)
        hidden_states = ttnn.reshape(hidden_states, [B, -1, self.heads * head_dim])

        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_out_weights,
            bias=self.tt_out_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        hidden_states = ttnn.add(hidden_states, input_tensor)

        return hidden_states

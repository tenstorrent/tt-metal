# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params


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

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        q_weights = state_dict[f"{module_path}.to_q.weight"].unsqueeze(0).unsqueeze(0)
        k_weights = state_dict[f"{module_path}.to_k.weight"].unsqueeze(0).unsqueeze(0)
        v_weights = state_dict[f"{module_path}.to_v.weight"].unsqueeze(0).unsqueeze(0)

        out_weights = state_dict[f"{module_path}.to_out.0.weight"].unsqueeze(0).unsqueeze(0)
        out_bias = state_dict[f"{module_path}.to_out.0.bias"]

        self.tt_q_weights, _ = prepare_linear_params(device, q_weights, None, ttnn.bfloat8_b)
        self.tt_k_weights, _ = prepare_linear_params(device, k_weights, None, ttnn.bfloat8_b)
        self.tt_v_weights, _ = prepare_linear_params(device, v_weights, None, ttnn.bfloat8_b)
        self.tt_out_weights, self.tt_out_bias = prepare_linear_params(device, out_weights, out_bias, ttnn.bfloat8_b)

    def forward(self, hidden_states, attention_mask, encoder_hidden_states=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        B = list(hidden_states.shape)[0]

        query = ttnn.linear(
            hidden_states,
            self.tt_q_weights,
            bias=None,
        )
        key = ttnn.linear(
            encoder_hidden_states,
            self.tt_k_weights,
            bias=None,
        )
        value = ttnn.linear(
            encoder_hidden_states,
            self.tt_v_weights,
            bias=None,
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
            attn_mask=attention_mask,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        hidden_states = ttnn.transpose(hidden_states, 1, 2)
        hidden_states = ttnn.reshape(hidden_states, [B, -1, self.heads * head_dim])

        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_out_weights,
            bias=self.tt_out_bias,
        )

        return hidden_states

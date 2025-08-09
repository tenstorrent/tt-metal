# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params


class TtClipAttention(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
        num_attention_heads,
        hidden_size,
    ):
        super().__init__()
        self.device = device

        self.heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim**-0.5

        q_weights = state_dict[f"{module_path}.q_proj.weight"].unsqueeze(0).unsqueeze(0)
        q_bias = state_dict[f"{module_path}.q_proj.bias"]
        k_weights = state_dict[f"{module_path}.k_proj.weight"].unsqueeze(0).unsqueeze(0)
        k_bias = state_dict[f"{module_path}.k_proj.bias"]
        v_weights = state_dict[f"{module_path}.v_proj.weight"].unsqueeze(0).unsqueeze(0)
        v_bias = state_dict[f"{module_path}.v_proj.bias"]

        out_weights = state_dict[f"{module_path}.out_proj.weight"].unsqueeze(0).unsqueeze(0)
        out_bias = state_dict[f"{module_path}.out_proj.bias"]

        self.tt_q_weights, self.tt_q_bias = prepare_linear_params(
            device, q_weights, q_bias, model_config.attention_weights_dtype
        )
        self.tt_k_weights, self.tt_k_bias = prepare_linear_params(
            device, k_weights, k_bias, model_config.attention_weights_dtype
        )
        self.tt_v_weights, self.tt_v_bias = prepare_linear_params(
            device, v_weights, v_bias, model_config.attention_weights_dtype
        )

        self.tt_out_weights, self.tt_out_bias = prepare_linear_params(
            device, out_weights, out_bias, model_config.attention_weights_dtype
        )

        self.default_compute_kernel_config = model_config.get_mm_compute_config(module_path)

    def forward(self, hidden_states, causal_mask=None):
        B = hidden_states.shape[0]

        query_states = ttnn.linear(
            hidden_states,
            self.tt_q_weights,
            bias=self.tt_q_bias,
            compute_kernel_config=self.default_compute_kernel_config,
        )
        query_states = query_states * self.scale

        key_states = ttnn.linear(
            hidden_states,
            self.tt_k_weights,
            bias=self.tt_k_bias,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        value_states = ttnn.linear(
            hidden_states,
            self.tt_v_weights,
            bias=self.tt_v_bias,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        query_states = ttnn.reshape(query_states, [1, -1, self.heads, self.head_dim])
        query_states = ttnn.transpose(query_states, 1, 2)

        key_states = ttnn.reshape(key_states, [1, -1, self.heads, self.head_dim])
        key_states = ttnn.transpose(key_states, 1, 2)

        value_states = ttnn.reshape(value_states, [1, -1, self.heads, self.head_dim])
        value_states = ttnn.transpose(value_states, 1, 2)

        key_states = ttnn.transpose(key_states, -2, -1)

        attn_weights = ttnn.matmul(query_states, key_states, compute_kernel_config=self.default_compute_kernel_config)
        attn_weights = attn_weights + causal_mask
        attn_weights = ttnn.softmax(attn_weights, dim=-1)

        attn_output = ttnn.matmul(attn_weights, value_states, compute_kernel_config=self.default_compute_kernel_config)

        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, [1, -1, self.heads * self.head_dim])

        attn_output = ttnn.linear(
            attn_output,
            self.tt_out_weights,
            bias=self.tt_out_bias,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        return attn_output

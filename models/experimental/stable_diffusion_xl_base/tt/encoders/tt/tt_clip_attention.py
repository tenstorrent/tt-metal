# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
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

        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

        q_weights = state_dict[f"{module_path}.q_proj.weight"].unsqueeze(0).unsqueeze(0)
        # q_weights = q_weights * self.scale

        k_weights = state_dict[f"{module_path}.k_proj.weight"].unsqueeze(0).unsqueeze(0)
        v_weights = state_dict[f"{module_path}.v_proj.weight"].unsqueeze(0).unsqueeze(0)

        out_weights = state_dict[f"{module_path}.out_proj.weight"].unsqueeze(0).unsqueeze(0)
        out_bias = state_dict[f"{module_path}.out_proj.bias"]

        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        fused_qkv_weights = torch.cat(
            [
                torch.transpose(q_weights, -2, -1),
                torch.transpose(k_weights, -2, -1),
                torch.transpose(v_weights, -2, -1),
            ],
            dim=-1,
        )
        self.tt_qkv_weights = ttnn.from_torch(
            fused_qkv_weights, model_config.attention_weights_dtype, device=device, layout=ttnn.TILE_LAYOUT
        )

        self.tt_out_weights, self.tt_out_bias = prepare_linear_params(
            device, out_weights, out_bias, model_config.attention_weights_dtype
        )

        self.default_compute_kernel_config = model_config.get_mm_compute_config(module_path)

    def forward(self, hidden_states):
        qkv_fused = ttnn.matmul(
            hidden_states,
            self.tt_qkv_weights,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.default_compute_kernel_config,
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
            is_causal=True,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
            scale=1.0,
        )
        hidden_states = ttnn.experimental.nlp_concat_heads(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_out_weights,
            bias=self.tt_out_bias,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return hidden_states

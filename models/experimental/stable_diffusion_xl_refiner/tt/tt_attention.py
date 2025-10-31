# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params
from .components.weight_loader import WeightLoader


class TtAttention(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        num_attn_heads: int,
    ):
        super().__init__()
        self.device = device
        self.heads = num_attn_heads

        self.weight_loader = WeightLoader(self, state_dict, module_path)

        self.is_self_attention = (
            self.weight_loader.q_weights.shape[-1] == self.weight_loader.k_weights.shape[-1]
            and self.weight_loader.q_weights.shape[-1] == self.weight_loader.v_weights.shape[-1]
        )

        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        if self.is_self_attention == True:
            fused_qkv_weights = torch.cat(
                [
                    torch.transpose(self.weight_loader.q_weights, -2, -1),
                    torch.transpose(self.weight_loader.k_weights, -2, -1),
                    torch.transpose(self.weight_loader.v_weights, -2, -1),
                ],
                dim=-1,
            )
            self.tt_qkv_weights = ttnn.from_torch(
                fused_qkv_weights, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
            )
        else:
            self.tt_q_weights, _ = prepare_linear_params(device, self.weight_loader.q_weights, None, ttnn.bfloat16)
            self.tt_k_weights, _ = prepare_linear_params(device, self.weight_loader.k_weights, None, ttnn.bfloat16)
            self.tt_v_weights, _ = prepare_linear_params(device, self.weight_loader.v_weights, None, ttnn.bfloat16)

        self.tt_out_weights, self.tt_out_bias = prepare_linear_params(
            device, self.weight_loader.out_weights, self.weight_loader.out_bias, ttnn.bfloat16
        )

    def forward(self, hidden_states, encoder_hidden_states=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        B = list(hidden_states.shape)[0]

        if self.is_self_attention:
            qkv_fused = ttnn.matmul(
                hidden_states,
                self.tt_qkv_weights,
                dtype=ttnn.bfloat16,
            )

            (
                q_heads,
                k_heads,
                v_heads,
            ) = ttnn.experimental.nlp_create_qkv_heads(
                qkv_fused, num_heads=self.heads, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(qkv_fused)
        else:
            q_heads = ttnn.matmul(
                hidden_states,
                self.tt_q_weights,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            k_heads = ttnn.matmul(
                encoder_hidden_states,
                self.tt_k_weights,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            v_heads = ttnn.matmul(
                encoder_hidden_states,
                self.tt_v_weights,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            q_heads, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                q_heads,
                num_heads=self.heads,
                num_kv_heads=0,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            v_heads, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                v_heads,
                num_heads=self.heads,
                num_kv_heads=0,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            k_heads, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                k_heads,
                num_heads=self.heads,
                num_kv_heads=0,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        hidden_states = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=False,
        )
        hidden_states = ttnn.experimental.nlp_concat_heads(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_out_weights,
            bias=self.tt_out_bias,
        )

        return hidden_states

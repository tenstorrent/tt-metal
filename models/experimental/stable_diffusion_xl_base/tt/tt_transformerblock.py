# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_attention import TtAttention
from models.experimental.stable_diffusion_xl_base.tt.tt_feedforward import TtFeedForward


class TtBasicTransformerBlock(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
        query_dim,
        num_attn_heads,
        out_dim,
    ):
        super().__init__()

        self.attn1 = TtAttention(
            device,
            state_dict,
            f"{module_path}.attn1",
            model_config,
            query_dim,
            num_attn_heads,
            out_dim,
        )
        self.attn2 = TtAttention(
            device,
            state_dict,
            f"{module_path}.attn2",
            model_config,
            query_dim,
            num_attn_heads,
            out_dim,
        )

        self.ff = TtFeedForward(device, state_dict, f"{module_path}.ff", model_config)

        norm1_weights = state_dict[f"{module_path}.norm1.weight"]
        norm1_bias = state_dict[f"{module_path}.norm1.bias"]

        # Always use bfloat16 for norm weights and bias
        self.tt_norm1_weights = ttnn.from_torch(norm1_weights, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        self.tt_norm1_bias = (
            ttnn.from_torch(norm1_bias, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            if norm1_bias is not None
            else None
        )

        norm2_weights = state_dict[f"{module_path}.norm2.weight"]
        norm2_bias = state_dict[f"{module_path}.norm2.bias"]
        self.tt_norm2_weights = ttnn.from_torch(norm2_weights, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        self.tt_norm2_bias = (
            ttnn.from_torch(norm2_bias, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            if norm2_bias is not None
            else None
        )

        norm3_weights = state_dict[f"{module_path}.norm3.weight"]
        norm3_bias = state_dict[f"{module_path}.norm3.bias"]
        self.tt_norm3_weights = ttnn.from_torch(norm3_weights, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        self.tt_norm3_bias = (
            ttnn.from_torch(norm3_bias, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            if norm3_bias is not None
            else None
        )

        self.ln_eps = 1e-5
        self.ln_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, input_tensor, attention_mask=None, encoder_hidden_states=None):
        attn_hidden_states = ttnn.layer_norm(
            input_tensor,
            weight=self.tt_norm1_weights,
            bias=self.tt_norm1_bias,
            epsilon=self.ln_eps,
            compute_kernel_config=self.ln_compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        attn_hidden_states = self.attn1(attn_hidden_states, attention_mask, None)
        hidden_states = ttnn.add(input_tensor, attn_hidden_states, use_legacy=False)
        ttnn.deallocate(input_tensor)

        attn_hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.tt_norm2_weights,
            bias=self.tt_norm2_bias,
            epsilon=self.ln_eps,
            compute_kernel_config=self.ln_compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        attn_hidden_states = self.attn2(attn_hidden_states, attention_mask, encoder_hidden_states)
        hidden_states = ttnn.add(hidden_states, attn_hidden_states, use_legacy=False)

        attn_hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.tt_norm3_weights,
            bias=self.tt_norm3_bias,
            epsilon=self.ln_eps,
            compute_kernel_config=self.ln_compute_kernel_config,
        )
        attn_hidden_states = self.ff(attn_hidden_states)
        hidden_states = ttnn.add(hidden_states, attn_hidden_states, use_legacy=False)

        return hidden_states

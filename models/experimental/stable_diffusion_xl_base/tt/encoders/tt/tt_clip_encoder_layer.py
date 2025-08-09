# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.encoders.tt.tt_clip_mlp import TtClipMLP
from models.experimental.stable_diffusion_xl_base.tt.encoders.tt.tt_clip_attention import TtClipAttention


class TtClipEncoderLayer(nn.Module):
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

        self.self_attn = TtClipAttention(
            device,
            state_dict,
            f"{module_path}.self_attn",
            model_config,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
        )

        self.mlp = TtClipMLP(
            device,
            state_dict,
            f"{module_path}.mlp",
            model_config,
        )

        norm1_weights = state_dict[f"{module_path}.layer_norm1.weight"]
        norm1_bias = state_dict[f"{module_path}.layer_norm1.bias"]
        self.tt_norm1_weights = ttnn.from_torch(norm1_weights, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        self.tt_norm1_bias = (
            ttnn.from_torch(norm1_bias, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            if norm1_bias is not None
            else None
        )

        norm2_weights = state_dict[f"{module_path}.layer_norm2.weight"]
        norm2_bias = state_dict[f"{module_path}.layer_norm2.bias"]
        self.tt_norm2_weights = ttnn.from_torch(norm2_weights, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        self.tt_norm2_bias = (
            ttnn.from_torch(norm2_bias, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            if norm2_bias is not None
            else None
        )

        self.ln_eps = 1e-5
        self.ln_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, hidden_states, causal_attention_mask):
        residual = hidden_states

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.tt_norm1_weights,
            bias=self.tt_norm1_bias,
            epsilon=self.ln_eps,
            compute_kernel_config=self.ln_compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        hidden_states = self.self_attn.forward(hidden_states, causal_attention_mask)
        hidden_states = ttnn.add(hidden_states, residual)

        residual = hidden_states

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.tt_norm2_weights,
            bias=self.tt_norm2_bias,
            epsilon=self.ln_eps,
            compute_kernel_config=self.ln_compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = ttnn.add(hidden_states, residual)

        return hidden_states

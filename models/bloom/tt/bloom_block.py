"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

import torch.nn as nn

import tt_lib
from models.utility_functions import (
    torch_to_tt_tensor_rm,
)
from models.bloom.tt.bloom_attention import TtBloomAttention
from models.bloom.tt.bloom_mlp import TtBloomMLP
from typing import Optional, Tuple


class TtBloomBlock(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.config = config
        self.state_dict = state_dict
        self.base_address = base_address
        self.device = device

        self.hidden_size = config.hidden_size

        self.input_layernorm_gamma = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.input_layernorm.weight"], self.device
        )
        self.input_layernorm_beta = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.input_layernorm.bias"], self.device
        )

        self.input_layernorm = tt_lib.tensor.layernorm

        self.num_heads = config.n_head

        self.self_attention = TtBloomAttention(
            self.config,
            state_dict=self.state_dict,
            base_address=f"{base_address}.self_attention",
            device=self.device,
        )

        self.post_attention_layernorm_gamma = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.post_attention_layernorm.weight"], self.device
        )
        self.post_attention_layernorm_beta = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.post_attention_layernorm.bias"], self.device
        )

        self.post_attention_layernorm = tt_lib.tensor.layernorm

        self.mlp = TtBloomMLP(
            self.config,
            state_dict=self.state_dict,
            base_address=f"{base_address}.mlp",
            device=self.device,
        )

        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        alibi: tt_lib.tensor.Tensor,
        attention_mask: tt_lib.tensor.Tensor,
        layer_past: Optional[Tuple[tt_lib.tensor.Tensor, tt_lib.tensor.Tensor]] = None,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[tt_lib.tensor.Tensor]:
        layernorm_output = self.input_layernorm(
            hidden_states,
            eps=self.config.layer_norm_epsilon,
            gamma=self.input_layernorm_gamma,
            beta=self.input_layernorm_beta,
        )

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(
            attention_output,
            eps=self.config.layer_norm_epsilon,
            gamma=self.post_attention_layernorm_gamma,
            beta=self.post_attention_layernorm_beta,
        )

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from models.experimental.t5.tt.t5_attention import TtT5Attention
from models.experimental.t5.tt.t5_layer_norm import TtT5LayerNorm


class TtT5LayerSelfAttention(torch.nn.Module):
    def __init__(
        self,
        config,
        state_dict,
        base_address,
        device,
        has_relative_attention_bias=False,
    ):
        super().__init__()
        self.SelfAttention = TtT5Attention(
            config,
            state_dict,
            f"{base_address}.SelfAttention",
            device,
            has_relative_attention_bias,
        )
        self.layer_norm = TtT5LayerNorm(config, state_dict, f"{base_address}.layer_norm", device)
        # self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # hidden_states = hidden_states + self.dropout(attention_output[0])
        hidden_states = ttnn.add(hidden_states, attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

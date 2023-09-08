# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
from typing import Optional
from models.t5.tt.t5_attention import TtT5Attention
from models.t5.tt.t5_layer_norm import TtT5LayerNorm


class TtT5LayerCrossAttention(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.EncDecAttention = TtT5Attention(
            config=config,
            state_dict=state_dict,
            base_address=f"{base_address}.EncDecAttention",
            device=device,
            has_relative_attention_bias=False,
        )

        # Cross attention is only in decoder
        assert config.is_decoder

        self.layer_norm = TtT5LayerNorm(
            config, state_dict, f"{base_address}.layer_norm", device
        )

    def forward(
        self,
        hidden_states: Optional[tt_lib.tensor.Tensor],
        key_value_states: Optional[tt_lib.tensor.Tensor],
        attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        position_bias: Optional[tt_lib.tensor.Tensor] = None,
        layer_head_mask: Optional[tt_lib.tensor.Tensor] = None,
        past_key_value: Optional[tt_lib.tensor.Tensor] = None,
        use_cache: Optional[bool] = False,
        query_length: Optional[int] = None,
        output_attentions: Optional[bool] = False,
    ) -> tt_lib.tensor.Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = tt_lib.tensor.add(hidden_states, attention_output[0])
        outputs = (layer_output,) + attention_output[1:]
        return outputs

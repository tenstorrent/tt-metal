# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from models.experimental.t5.tt.t5_layer_self_attention import TtT5LayerSelfAttention
from models.experimental.t5.tt.t5_layer_cross_attention import TtT5LayerCrossAttention
from models.experimental.t5.tt.t5_layer_ff import TtT5LayerFF


class TtT5Block(nn.Module):
    def __init__(
        self,
        config,
        state_dict,
        base_address,
        device,
        has_relative_attention_bias=False,
    ):
        super().__init__()
        self.is_decoder = config["is_decoder"]
        self.device = device

        self.layer = nn.ModuleList()
        self.layer.append(
            TtT5LayerSelfAttention(
                config,
                state_dict,
                f"{base_address}.layer.0",
                device,
                has_relative_attention_bias,
            )
        )
        layer_cnt = 1

        if self.is_decoder:
            self.layer.append(TtT5LayerCrossAttention(config, state_dict, f"{base_address}.layer.1", device))
            layer_cnt += 1

        self.layer.append(TtT5LayerFF(config, state_dict, f"{base_address}.layer.{layer_cnt}", device))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        # Always true for Tt
        if True:  # hidden_states.dtype == torch.float16:
            hidden_states = tt2torch_tensor(hidden_states)

            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(torch.float16).max - 1000,
                torch.finfo(torch.float16).max,
            )

            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
            hidden_states = torch2tt_tensor(hidden_states, self.device)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape.with_tile_padding()[3]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            # Always true for Tt
            if False:  # hidden_states.dtype == torch.float16:
                hidden_states = tt2torch_tensor(hidden_states)

                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(torch.float16).max - 1000,
                    torch.finfo(torch.float16).max,
                )

                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
                hidden_states = torch2tt_tensor(hidden_states, self.device)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        # Always true for Tt
        if False:  # hidden_states.dtype == torch.float16:
            hidden_states = tt2torch_tensor(hidden_states)

            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(torch.float16).max - 1000,
                torch.finfo(torch.float16).max,
            )

            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
            hidden_states = torch2tt_tensor(hidden_states, self.device)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from typing import List, Optional, Tuple, Union
from loguru import logger
import ttnn
from models.experimental.llama.llama_utils import _make_causal_mask, _expand_mask, linear

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

from models.experimental.llama.tt.llama_decoder import TtLlamaDecoderLayer
from models.experimental.llama.tt.llama_layer_norm import TtLlamaRMSNorm


def build_decoders(
    device,
    state_dict,
    base_url,
    max_position_embeddings,
    config,
    num_decoder_start,
    num_decoders,
):
    decoder_list = torch.nn.Sequential(
        *[
            TtLlamaDecoderLayer(
                device,
                state_dict,
                base_url,
                decoder_idx,
                max_position_embeddings,
                config,
            )
            for decoder_idx in range(num_decoder_start, num_decoder_start + num_decoders)
        ]
    )
    return decoder_list


class TtLlamaModelSecondHFModel(torch.nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        config,
        num_decoders_start,
        num_decoders,
        is_causallm=True,
    ):
        super().__init__()

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.device = device
        self.state_dict = state_dict
        self.base_url = base_url
        self.max_position_embeddings = max_position_embeddings
        self.num_decoders_start = num_decoders_start
        self.num_decoders = num_decoders
        self.config = config
        self.is_causallm = is_causallm

        # stack all decoders
        self.decoders_second = build_decoders(
            self.device,
            self.state_dict,
            self.base_url,
            self.max_position_embeddings,
            self.config,
            self.num_decoders_start,
            self.num_decoders,
        )

        # add final normalization layer
        self.layer_num = None
        self.layer_position = "norm"
        self.final_layernorm = TtLlamaRMSNorm(
            self.device,
            state_dict=self.state_dict,
            base_url=self.base_url,
            layer_num=self.layer_num,
            layer_position=self.layer_position,
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # get weights for linear layer
        self.weight = torch_to_tt_tensor_rm(self.state_dict["lm_head.weight"], self.device)
        self.bias = None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: ttnn.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            # batch_size, seq_length = input_ids.shape
            batch_size = input_ids.get_legacy_shape()[0]
            seq_length = input_ids.get_legacy_shape()[2]
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # No embeddings $$ just take hidden states from the first half of the model
        inputs_embeds = tt_to_torch_tensor(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = input_ids

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.decoders_second):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            # decoder layer returns tuple
            hidden_states = layer_outputs[0]

        # apply norm layer
        hidden_states = self.final_layernorm(hidden_states)

        # apply linear layer for causallm model
        if self.is_causallm:
            hidden_states = linear(hidden_states, self.weight, self.bias, self.device)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return (
            hidden_states,
            next_cache,
            all_hidden_states,
            all_self_attns,
        )

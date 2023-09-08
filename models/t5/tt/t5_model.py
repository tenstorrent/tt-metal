# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import copy
from torch import nn
from typing import Optional
from dataclasses import dataclass
from models.t5.tt.t5_stack import TtT5Stack
import tt_lib


@dataclass
class Seq2SeqModelOutput:
    last_hidden_state: Optional[tt_lib.tensor.Tensor] = None
    past_key_values: Optional[tt_lib.tensor.Tensor] = None
    decoder_hidden_states: Optional[tt_lib.tensor.Tensor] = None
    decoder_attentions: Optional[tt_lib.tensor.Tensor] = None
    cross_attentions: Optional[tt_lib.tensor.Tensor] = None
    encoder_last_hidden_state: Optional[tt_lib.tensor.Tensor] = None
    encoder_hidden_states: Optional[tt_lib.tensor.Tensor] = None
    encoder_attentions: Optional[tt_lib.tensor.Tensor] = None


class TtT5Model(nn.Module):
    def __init__(self, config, state_dict, device):
        super().__init__()

        self.config_use_cache = config.use_cache
        self.config_use_return_dict = config.use_return_dict

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.shared.weight = nn.Parameter(state_dict["shared.weight"])

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = TtT5Stack(
            encoder_config, state_dict, "encoder", device, self.shared
        )

        if config.num_decoder_layers is None:
            config.num_decoder_layers = config.num_layers

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = TtT5Stack(
            decoder_config, state_dict, "decoder", device, self.shared
        )

        self.config = config
        self.device = device

        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[tt_lib.tensor.Tensor] = None,
        attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        decoder_input_ids: Optional[tt_lib.tensor.Tensor] = None,
        decoder_attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        decoder_head_mask: Optional[tt_lib.tensor.Tensor] = None,
        cross_attn_head_mask: Optional[tt_lib.tensor.Tensor] = None,
        encoder_outputs: Optional[tt_lib.tensor.Tensor] = None,
        past_key_values: Optional[tt_lib.tensor.Tensor] = None,
        inputs_embeds: Optional[tt_lib.tensor.Tensor] = None,
        decoder_inputs_embeds: Optional[tt_lib.tensor.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tt_lib.tensor.Tensor:
        use_cache = use_cache if use_cache is not None else self.config_use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config_use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs.last_hidden_state

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

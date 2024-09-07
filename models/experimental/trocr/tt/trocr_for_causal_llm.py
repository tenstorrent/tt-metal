# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import copy
from typing import Optional, Tuple, Union
from dataclasses import dataclass

import ttnn
from models.utility_functions import torch_to_tt_tensor_rm
from models.experimental.trocr.tt.trocr_decoder_wrapper import TtTrOCRDecoderWrapper
from models.helper_funcs import Linear


@dataclass
class TtCausalLMOutputWithCrossAttentions(nn.Module):
    loss: Optional[ttnn.Tensor] = None
    logits: ttnn.Tensor = None
    past_key_values: Optional[Tuple[Tuple[ttnn.Tensor]]] = None
    hidden_states: Optional[Tuple[ttnn.Tensor]] = None
    attentions: Optional[Tuple[ttnn.Tensor]] = None
    cross_attentions: Optional[Tuple[ttnn.Tensor]] = None


class TtTrOCRForCausalLM(nn.Module):
    def __init__(
        self,
        config,
        state_dict=None,
        base_address=None,
        device=None,
    ):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        self.config = config
        self.device = device
        super().__init__()
        self.model = TtTrOCRDecoderWrapper(config, base_address=base_address, state_dict=state_dict, device=device)

        self.output_projection_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.output_projection.weight"],
            self.device,
            put_on_device=False,
        )
        self.output_projection = Linear(
            config.d_model,
            config.vocab_size,
            self.output_projection_weight,
        )

    def forward(
        self,
        input_ids: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        encoder_hidden_states: Optional[ttnn.Tensor] = None,
        encoder_attention_mask: Optional[ttnn.Tensor] = None,
        head_mask: Optional[ttnn.Tensor] = None,
        cross_attn_head_mask: Optional[ttnn.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[ttnn.Tensor]]] = None,
        inputs_embeds: Optional[ttnn.Tensor] = None,
        labels: Optional[ttnn.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TtCausalLMOutputWithCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.output_projection(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return TtCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

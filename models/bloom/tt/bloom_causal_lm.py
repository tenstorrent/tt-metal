"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

import torch
import torch.nn as nn
import tt_lib
import models.bloom.tt.bloom_model as bloom_model
from typing import Optional, Tuple, Union
from models.helper_funcs import Linear as TtLinear

from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor

from dataclasses import dataclass


@dataclass
class TtCausalLMOutputWithCrossAttentions:
    loss: Optional[tt_lib.tensor.Tensor] = None
    logits: tt_lib.tensor.Tensor = None
    past_key_values: Optional[Tuple[Tuple[tt_lib.tensor.Tensor]]] = None
    hidden_states: Optional[Tuple[tt_lib.tensor.Tensor]] = None
    attentions: Optional[Tuple[tt_lib.tensor.Tensor]] = None
    cross_attentions: Optional[Tuple[tt_lib.tensor.Tensor]] = None


class TtBloomForCausalLM(nn.Module):
    def __init__(self, config, state_dict, device):
        super().__init__()
        self.config = config
        self.state_dict = state_dict
        self.device = device

        self.use_return_dict = False
        self.transformer = bloom_model.TtBloomModel(
            self.config,
            state_dict=self.state_dict,
            base_address=f"transformer",
            device=self.device,
        )
        self.lm_head_weight = torch_to_tt_tensor_rm(
            state_dict["lm_head.weight"], device
        )

        self.lm_head = TtLinear(
            self.lm_head_weight.shape()[-1],
            self.lm_head_weight.shape()[-2],
            weight=self.lm_head_weight,
            bias=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[tt_lib.tensor.Tensor] = None,
        attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        inputs_embeds: Optional[tt_lib.tensor.Tensor] = None,
        **kwargs,
    ) -> dict:
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            if past_key_values[0][0].shape()[-3] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            Tuple[Tuple[tt_lib.tensor.Tensor, tt_lib.tensor.Tensor], ...]
        ] = None,
        attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        inputs_embeds: Optional[tt_lib.tensor.Tensor] = None,
        labels: Optional[tt_lib.tensor.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[tt_lib.tensor.Tensor, ...], TtCausalLMOutputWithCrossAttentions]:
        if deprecated_arguments.pop("position_ids", False) is not False:
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )

        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None

        if labels is not None:
            lm_logits = tt_to_torch_tensor(lm_logits).squeeze(0)
            labels = labels.to(lm_logits.device)
            labels = tt_to_torch_tensor(labels)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape

            shift_logits = torch_to_tt_tensor_rm(shift_logits, device=self.device)
            shift_labels = torch_to_tt_tensor_rm(shift_labels, device=self.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits=tt_lib.tensor.reshape(
                    shift_logits, 1, 1, batch_size * seq_length, vocab_size
                ),
                shift_labels=tt_lib.tensor.reshape(
                    shift_labels, 1, 1, 1, batch_size * seq_length
                ),
            )
            lm_logits = torch_to_tt_tensor_rm(lm_logits, device=self.device)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TtCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

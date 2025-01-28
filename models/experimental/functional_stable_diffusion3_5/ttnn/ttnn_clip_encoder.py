# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from torch import nn
from typing import Optional, Union, Tuple

from models.experimental.functional_stable_diffusion3_5.reference.clip_encoder_layer import CLIPEncoderLayer
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_clip_encoder_layer import ttnn_CLIPEncoderLayer

from dataclasses import dataclass


@dataclass
class ttnn_BaseModelOutput:
    last_hidden_state: ttnn.Tensor = None
    hidden_states: Optional[Tuple[ttnn.Tensor, ...]] = None
    attentions: Optional[Tuple[ttnn.Tensor, ...]] = None


class ttnn_CLIPEncoder:
    def __init__(self, config):
        self.config = config
        self.layers = [ttnn_CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.gradient_checkpointing = False

    def __call__(
        self,
        inputs_embeds,
        attention_mask: Optional[ttnn.Tensor] = None,
        causal_attention_mask: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        parameters=None,
    ) -> Union[Tuple, ttnn_BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
                parameters=parameters.layers[idx],
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return ttnn_BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

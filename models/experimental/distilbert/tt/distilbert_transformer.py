# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
import torch.nn as nn

import ttnn
from dataclasses import dataclass
from models.experimental.distilbert.tt.distilbert_transformer_block import TtTransformerBlock


@dataclass
class TtBaseModelOutput:
    last_hidden_state: ttnn.Tensor = None
    hidden_states: Optional[Tuple[ttnn.Tensor]] = None
    attentions: Optional[Tuple[ttnn.Tensor]] = None


class TtTransformer(nn.Module):
    def __init__(self, config, state_dict=None, base_address="", device=None):
        super().__init__()
        self.config = config
        self.n_layers = self.config.n_layers
        self.state_dict = state_dict
        self.base_address = base_address
        self.device = device

        self.layer = nn.ModuleList(
            [
                TtTransformerBlock(
                    config,
                    state_dict=self.state_dict,
                    base_address=f"{base_address}.layer.{i}",
                    device=self.device,
                )
                for i in range(config.n_layers)
            ]
        )

    def forward(
        self,
        input: ttnn.Tensor,
        attn_mask: Optional[ttnn.Tensor] = None,
        head_mask: Optional[ttnn.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[TtBaseModelOutput, Tuple[ttnn.Tensor, ...]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = input
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(
                input=hidden_state,
                attn_mask=attn_mask,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                if len(layer_outputs) != 2:
                    raise ValueError(f"The length of the layer_outputs should be 2, but it is {len(layer_outputs)}")

                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)

            else:
                if len(layer_outputs) != 1:
                    raise ValueError(f"The length of the layer_outputs should be 1, but it is {len(layer_outputs)}")

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return TtBaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

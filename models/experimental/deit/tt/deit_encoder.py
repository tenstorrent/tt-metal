# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union

import torch
from torch import nn

import ttnn
from models.experimental.deit.tt.deit_config import DeiTConfig
from models.experimental.deit.tt.deit_layer import TtDeiTLayer


class TtDeiTEncoder(nn.Module):
    def __init__(self, config: DeiTConfig(), device, state_dict=None, base_address="") -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [
                TtDeiTLayer(config, device, state_dict, f"{base_address}.layer.{_}")
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        head_mask: Optional[ttnn.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                assert False, "No support for training yet!"
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        output = tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return output

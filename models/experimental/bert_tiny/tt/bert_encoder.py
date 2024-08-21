# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
from typing import Optional
from models.experimental.bert_tiny.tt.bert_layer import TtBertlayer


class TtBertencoder(nn.Module):
    def __init__(
        self,
        config,
        state_dict=None,
        device=None,
        mem_config=None,
    ):
        super().__init__()

        self.config = config
        self.device = device
        self.state_dict = state_dict

        self.output_mem_config = mem_config

        self.layers = nn.ModuleList(
            [
                TtBertlayer(
                    config=self.config,
                    encoder_idx=idx,
                    state_dict=self.state_dict,
                    device=self.device,
                    mem_config=self.output_mem_config,
                )
                for idx in range(self.config.num_hidden_layers)
            ]
        )

    def forward(self, input: Optional[ttnn.Tensor] = None, attention_mask: Optional[ttnn.Tensor] = None):
        for i, layer_module in enumerate(self.layers):
            input = layer_module(input, attention_mask)
        return input

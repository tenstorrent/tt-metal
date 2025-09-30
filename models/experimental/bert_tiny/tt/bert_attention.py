# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.common.lightweightmodule import LightweightModule
import ttnn
from typing import Optional
from models.experimental.bert_tiny.tt.bert_self_attention import TtBertselfattention
from models.experimental.bert_tiny.tt.bert_output import TtBertoutput


class TtBertattention(LightweightModule):
    def __init__(self, config, encoder_idx: int, state_dict=None, device=None, mem_config=None):
        super().__init__()

        self.config = config
        self.device = device
        self.state_dict = state_dict

        self.output_mem_config = mem_config

        self.bert_self_attention = TtBertselfattention(
            config=self.config,
            encoder_idx=encoder_idx,
            state_dict=self.state_dict,
            device=self.device,
            mem_config=self.output_mem_config,
        )
        self.bert_output = TtBertoutput(
            self.config, encoder_idx, "attention.output", state_dict, self.device, mem_config=self.output_mem_config
        )

    def forward(self, input: Optional[ttnn.Tensor] = None, attention_mask: Optional[ttnn.Tensor] = None):
        out = self.bert_self_attention(input, attention_mask)
        return self.bert_output(out, input)

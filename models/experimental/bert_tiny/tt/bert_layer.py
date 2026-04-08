# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
from typing import Optional
from models.experimental.bert_tiny.tt.bert_attention import TtBertattention
from models.experimental.bert_tiny.tt.bert_output import TtBertoutput
from models.experimental.bert_tiny.tt.bert_intermediate import TtBertintermediate


class TtBertlayer(nn.Module):
    def __init__(
        self,
        config,
        encoder_idx: int,
        state_dict=None,
        device=None,
        mem_config=None,
    ):
        super().__init__()

        self.config = config
        self.device = device
        self.state_dict = state_dict
        self.output_mem_config = mem_config

        self.bert_attention = TtBertattention(
            config=self.config,
            encoder_idx=encoder_idx,
            state_dict=self.state_dict,
            device=self.device,
            mem_config=self.output_mem_config,
        )
        self.bert_intermediate = TtBertintermediate(
            self.config, encoder_idx, state_dict, self.device, mem_config=self.output_mem_config
        )
        self.bert_output = TtBertoutput(
            self.config, encoder_idx, "output", state_dict, self.device, mem_config=self.output_mem_config
        )

    def forward(self, input: Optional[ttnn.Tensor] = None, attention_mask: Optional[ttnn.Tensor] = None):
        out = self.bert_attention(input, attention_mask)
        intermediate_output = self.bert_intermediate(out)
        return self.bert_output(intermediate_output, out)

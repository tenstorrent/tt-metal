# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
from typing import Optional
from models.experimental.bert_tiny.tt.bert import TtBert

from models.experimental.bert_tiny.bert_tiny_helper_funcs import Linear as TtLinear


class TtBertforqa(nn.Module):
    def __init__(
        self,
        config,
        state_dict=None,
        device=None,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.output_mem_config = ttnn.DRAM_MEMORY_CONFIG

        self.bert = TtBert(config=self.config, state_dict=state_dict, device=device, mem_config=self.output_mem_config)

        self.weight = ttnn.to_device(
            ttnn.from_torch(state_dict[f"qa_outputs.weight"], dtype=ttnn.bfloat16),
            device=self.device,
        )

        self.bias = ttnn.to_device(
            ttnn.from_torch(state_dict[f"qa_outputs.bias"], dtype=ttnn.bfloat16),
            device=self.device,
        )

        self.dense = TtLinear(
            self.weight,
            self.bias,
            device=self.device,
            output_mem_config=self.output_mem_config,
        )

    def forward(
        self,
        input_ids: Optional[ttnn.Tensor] = None,
        token_type_ids: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
    ):
        bert_out = self.bert(input_ids, token_type_ids, attention_mask)
        return self.dense(bert_out)

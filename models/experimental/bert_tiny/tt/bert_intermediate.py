# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn
from typing import Optional
from models.experimental.bert_tiny.bert_tiny_helper_funcs import Linear as TtLinear


class TtBertintermediate(nn.Module):
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

        self.output_mem_config = mem_config

        self.weight = ttnn.to_device(
            ttnn.from_torch(
                state_dict[f"bert.encoder.layer.{encoder_idx}.intermediate.dense.weight"], dtype=ttnn.bfloat16
            ),
            device=self.device,
        )

        self.bias = ttnn.to_device(
            ttnn.from_torch(
                state_dict[f"bert.encoder.layer.{encoder_idx}.intermediate.dense.bias"], dtype=ttnn.bfloat16
            ),
            memory_config=self.output_mem_config,
            device=self.device,
        )

        self.dense = TtLinear(
            self.weight,
            self.bias,
            device=self.device,
            output_mem_config=self.output_mem_config,
        )

    def forward(self, activation: Optional[ttnn.Tensor] = None):
        out = self.dense(activation)
        return ttnn.gelu(out, memory_config=self.output_mem_config)

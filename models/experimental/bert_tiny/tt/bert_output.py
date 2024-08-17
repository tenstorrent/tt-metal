# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn
from typing import Optional
from models.experimental.bert_tiny.bert_tiny_helper_funcs import Linear as TtLinear
from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor_rm


class TtBertoutput(nn.Module):
    def __init__(
        self,
        config,
        encoder_idx: int,
        base_address=None,
        state_dict=None,
        device=None,
        mem_config=None,
    ):
        super().__init__()

        self.config = config
        self.device = device
        self.base_address = base_address

        self.output_mem_config = mem_config

        self.weight = ttnn.to_device(
            ttnn.from_torch(
                state_dict[f"bert.encoder.layer.{encoder_idx}.{self.base_address}.dense.weight"], dtype=ttnn.bfloat16
            ),
            device=self.device,
        )

        self.bias = ttnn.to_device(
            ttnn.from_torch(
                state_dict[f"bert.encoder.layer.{encoder_idx}.{self.base_address}.dense.bias"], dtype=ttnn.bfloat16
            ),
            memory_config=self.output_mem_config,
            device=self.device,
        )

        self.output = TtLinear(
            self.weight,
            self.bias,
            device=self.device,
            output_mem_config=self.output_mem_config,
        )

        self.beta = torch_to_tt_tensor_rm(
            state_dict[f"bert.encoder.layer.{encoder_idx}.{self.base_address}.LayerNorm.bias"],
            device=self.device,
            put_on_device=False,
        )
        self.gamma = torch_to_tt_tensor_rm(
            state_dict[f"bert.encoder.layer.{encoder_idx}.{self.base_address}.LayerNorm.weight"],
            device=self.device,
            put_on_device=False,
        )
        self.ln = ttnn.layer_norm

    def forward(self, hidden_state: Optional[ttnn.Tensor] = None, input: Optional[ttnn.Tensor] = None):
        out = self.output(hidden_state)
        out = out + input
        out = ttnn.to_layout(out, layout=ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.from_device(out)
        out = ttnn.to_torch(out)
        out = torch_to_tt_tensor_rm(out, self.device, put_on_device=False)
        ln_out = self.ln(out, epsilon=1e-12, weight=self.gamma, bias=self.beta)
        ln_out = tt_to_torch_tensor(ln_out)
        return ttnn.to_device(ttnn.from_torch(ln_out, dtype=ttnn.bfloat16), device=self.device)

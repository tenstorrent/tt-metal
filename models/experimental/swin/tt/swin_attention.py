# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

import ttnn

from models.experimental.swin.tt.swin_self_attention import (
    TtSwinSelfAttention,
)
from models.experimental.swin.tt.swin_self_output import TtSwinSelfOutput


class TtSwinAttention(nn.Module):
    def __init__(
        self,
        config,
        dim,
        num_heads,
        window_size,
        state_dict,
        base_address,
        device,
    ):
        super().__init__()
        self.self = TtSwinSelfAttention(
            config,
            dim,
            num_heads,
            window_size,
            state_dict,
            base_address=f"{base_address}.self",
            device=device,
        )
        self.output = TtSwinSelfOutput(config, dim, state_dict, f"{base_address}.output", device)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        head_mask: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ttnn.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

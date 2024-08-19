# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from typing import Union, Optional, Tuple


import tt_lib

from models.experimental.deit.tt.deit_config import DeiTConfig
from models.experimental.deit.tt.deit_self_attention import TtDeiTSelfAttention
from models.experimental.deit.tt.deit_self_output import TtDeiTSelfOutput


class TtDeiTAttention(nn.Module):
    def __init__(self, config: DeiTConfig(), device, state_dict=None, base_address="") -> None:
        super().__init__()

        self.attention = TtDeiTSelfAttention(config, device, state_dict, f"{base_address}.attention")

        self.output = TtDeiTSelfOutput(config, device, state_dict, f"{base_address}.output")

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[tt_lib.tensor.Tensor, tt_lib.tensor.Tensor], Tuple[tt_lib.tensor.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

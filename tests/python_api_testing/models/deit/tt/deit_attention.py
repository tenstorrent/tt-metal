from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
from torch import nn
from typing import Union, Optional, Tuple, Dict, Set, List
from deit_config import DeiTConfig

import tt_lib
from tt_lib.fallback_ops import fallback_ops
from deit_self_attention import TtDeiTSelfAttention
from deit_self_output import TtDeiTSelfOutput
from utility_functions_new import torch_to_tt_tensor, torch_to_tt_tensor_rm, tt_to_torch_tensor


class TtDeiTAttention(nn.Module):
    def __init__(self, config: DeiTConfig(), host, device, state_dict=None, base_address="") -> None:
        super().__init__()
        self.attention = TtDeiTSelfAttention(config, host, device, state_dict, f"{base_address}.attention")
        self.output = TtDeiTSelfOutput(config, host, device, state_dict, f"{base_address}.output")

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

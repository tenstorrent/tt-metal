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
from deit_config import DeiTConfig

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm
from helper_funcs import Linear as TtLinear

class TtDeiTOutput(nn.Module):
    def __init__(self, config: DeiTConfig() , device, state_dict=None, base_address="") -> None:
        super().__init__()

        dense_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.dense.weight"], device)
        dense_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.dense.bias"], device)
        self.dense = TtLinear(config.intermediate_size, config.hidden_size, dense_weight, dense_bias)

    def forward(self,
                hidden_states: tt_lib.tensor.Tensor,
                input_tensor: tt_lib.tensor.Tensor)-> tt_lib.tensor.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = tt_lib.tensor.add(hidden_states , input_tensor)

        return hidden_states

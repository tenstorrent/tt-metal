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
from helper_funcs import make_linear

class TtDeiTOutput(nn.Module):
    def __init__(self, config: DeiTConfig() , host, device, state_dict=None, base_address="") -> None:
        super().__init__()
        dense_weight = state_dict[f"{base_address}.weight"]
        dense_bias = state_dict[f"{base_address}.bias"]
        self.dense = make_linear(config.intermediate_size, config.hidden_size, dense_weight, dense_bias, device)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = tt_lib.tensor.add(hidden_states , input_tensor)

        return hidden_states

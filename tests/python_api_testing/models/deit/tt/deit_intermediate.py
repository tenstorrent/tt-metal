from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union

import torch
from torch import nn

from activations import ACT2FN
from deit_config import DeiTConfig

import tt_lib
from helper_funcs import make_linear

class ttDeiTIntermediate(nn.Module):
    def __init__(self, config: DeiTConfig(), host, device, state_dict=None, base_address="") -> None:
        super().__init__()

        dense_weight = state_dict[f"{base_address}.weight"]
        dense_bias = state_dict[f"{base_address}.bias"]
        self.dense = make_linear(config.hidden_size, config.intermediate_size, dense_weight, dense_bias, device)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

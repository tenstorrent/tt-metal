from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from activations import ACT2FN
from deit_config import DeiTConfig

import tt_lib
from helper_funcs import make_linear

class TtDeiTSelfOutput(nn.Module):
    """
    The residual connection is defined in DeiTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: DeiTConfig(), host, device, state_dict=None, base_address="") -> None:
        super().__init__()
        dense_weight = state_dict[f"{base_address}.weight"]
        dense_bias = state_dict[f"{base_address}.bias"]
        self.dense = make_linear(config.hidden_size, config.hidden_size, dense_weight, dense_bias, device)

        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)

        return hidden_states

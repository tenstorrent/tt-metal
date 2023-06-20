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
from deit_helper_funcs import make_linear

class TtDeiTPooler(nn.Module):
    def __init__(self, config: DeiTConfig(), host, device, state_dict=None, base_address=""):
        super().__init__()
        # dense_weight = state_dict[f"{base_address}.weight"]
        # dense_bias = state_dict[f"{base_address}.bias"]
        self.dense = make_linear(config.hidden_size, config.hidden_size, "dense", state_dict, base_address, device)

        # self.dense = make_linear(config.hidden_size, config.hidden_size, dense_weight, dense_bias, device)
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = tt_lib.tensor.tanh()
        # self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = tt_lib.tensor.tanh(pooled_output)
        # pooled_output = self.activation(pooled_output)
        return pooled_output

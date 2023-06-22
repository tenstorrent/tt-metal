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

class TtDeiTPooler(nn.Module):
    def __init__(self, config: DeiTConfig(), device, state_dict=None, base_address=""):
        super().__init__()
        dense_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.dense.weight"], device)
        dense_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.dense.bias"], device)
        self.dense = TtLinear(config.hidden_size, config.hidden_size, dense_weight, dense_bias)

        self.activation = tt_lib.tensor.tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = tt_lib.tensor.tanh(pooled_output)
        return pooled_output

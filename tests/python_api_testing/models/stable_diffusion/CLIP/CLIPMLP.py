import numpy as np
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")


import torch.nn as nn
import torch

from libs import tt_lib as ttl
from libs.tt_lib.utils import tilize_to_list, pad_weight

from python_api_testing.fused_ops.linear import Linear as tt_linear
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc


class TtCLIPMLP(torch.nn.Module):
    def __init__(self,  device, state_dict, config=None, hidden_size=None, intermediate_size=None, base_address="text_model.encoder.layers.10.mlp"):
        super().__init__()
        self.device=device

        self.linear1_weight = tilize_to_list(pad_weight(state_dict[f"{base_address}.fc1.weight"]))
        self.linear1_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.fc1.bias"]))

        self.linear2_weight = tilize_to_list(pad_weight(state_dict[f"{base_address}.fc2.weight"]))
        self.linear2_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.fc2.bias"]))

        self.config = config
        hidden_size = config.hidden_size if config else hidden_size
        intermediate_size = config.intermediate_size if config else intermediate_size

        self.linear_1 = tt_linear(hidden_size, intermediate_size, self.linear1_weight, bias=self.linear1_bias, device=device)
        self.linear_2 = tt_linear(intermediate_size, hidden_size, self.linear2_weight, bias=self.linear2_bias, device=device)

    def forward(self, x):

        x = self.linear_1(x)
        x = ttl.tensor.gelu(x)
        x = self.linear_2(x)
        return x

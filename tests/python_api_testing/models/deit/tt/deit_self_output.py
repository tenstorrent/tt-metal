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
from deit_helper_funcs import make_linear

class TtDeiTSelfOutput(nn.Module):
    """
    The residual connection is defined in DeiTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: DeiTConfig(), host, device, state_dict=None, base_address="") -> None:
        super().__init__()
        # print('self output base address:::', base_address)
        # dense_weight = state_dict[f"{base_address}.dense.weight"]
        # dense_bias = state_dict[f"{base_address}.dense.bias"]
        # self.dense = make_linear(config.hidden_size, config.hidden_size, dense_weight, dense_bias, device)
        self.dense = make_linear(
                            config.hidden_size,
                            config.hidden_size,
                            "dense",
                            state_dict=state_dict,
                            base_address=base_address,
                            device=device
                            )

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)

        return hidden_states

import torch
from torch.nn import functional as F

import tt_lib
from python_api_testing.models.nanogpt.tt.nanogpt_config import GPTConfig
from python_api_testing.models.helper_funcs import Linear


from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

from transformers import GPT2LMHeadModel

class TtMLP(torch.nn.Module):
    def __init__(self, base_address, config: GPTConfig(), state_dict, device):
        super().__init__()
        # Get the weights
        self.tt_weight_c_fc = state_dict[f"{base_address}.c_fc.weight"]
        self.tt_weight_c_proj = state_dict[f"{base_address}.c_proj.weight"]
        self.config = config
        self.device = device

        # Push weights to Tt device
        self.tt_weight_c_fc = torch2tt_tensor(
            self.tt_weight_c_fc, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        self.tt_weight_c_proj = torch2tt_tensor(
            self.tt_weight_c_proj, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        # Load biases
        self.tt_bias_c_fc = torch2tt_tensor(
            state_dict[f"{base_address}.c_fc.bias"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        self.tt_bias_c_proj = torch2tt_tensor(
            state_dict[f"{base_address}.c_proj.bias"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        self.tt_weight_c_fc = tt_lib.tensor.transpose(self.tt_weight_c_fc)
        self.tt_weight_c_proj = tt_lib.tensor.transpose(self.tt_weight_c_proj)

        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, self.tt_weight_c_fc, self.tt_bias_c_fc)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, self.tt_weight_c_proj, self.tt_bias_c_proj)

    def forward(self, x):

        x1 = self.c_fc(x)

        x2 = tt_lib.tensor.gelu(x1)

        x3 = self.c_proj(x2)

        return x3

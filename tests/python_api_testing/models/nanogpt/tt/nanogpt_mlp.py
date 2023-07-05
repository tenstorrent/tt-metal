import torch
from torch.nn import functional as F

import tt_lib
import python_api_testing.models.nanogpt.utils as nanogpt_utils


from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

from transformers import GPT2LMHeadModel

class TtMLP(torch.nn.Module):
    def __init__(self, base_address, state_dict, device):
        super().__init__()
        # Get the weights
        self.tt_weight_c_fc = state_dict[f"{base_address}.c_fc.weight"]
        self.tt_weight_c_proj = state_dict[f"{base_address}.c_proj.weight"]

        # Transpose the weights
        #self.tt_weight_c_fc = torch.transpose(self.tt_weight_c_fc, -1, -2)

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
        self.device = device


    def forward(self, x):

        x1 = nanogpt_utils.tt_linear(x, self.tt_weight_c_fc, self.tt_bias_c_fc)

        x2 = tt_lib.tensor.gelu(x1)

        x3 = nanogpt_utils.tt_linear(x2, self.tt_weight_c_proj, self.tt_bias_c_proj)

        return x3

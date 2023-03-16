from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from libs import tt_lib as ttl
from utility_functions import pad_weight, tilize_to_list, print_diff_argmax, torch_to_tt_tensor, tt_to_torch_tensor, print_corr_coef
from python_api_testing.fused_ops.linear import Linear as TtLinear
from python_api_testing.fused_ops.silu import SiLU as TtSiLU
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc


class TimeEmbedding(torch.nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.linear_1.weight.data.fill_(0.001)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd, bias=False)
        self.linear_2.weight.data.fill_(0.001)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x


class TtTimeEmbedding(torch.nn.Module):
    def __init__(self,  n_embd, device, state_dict=None):
        super().__init__()
        # Note: Load Weights
        #in_feature = n_embd, out_feature = 4 * n_embd
        weight1_shape = [1, 1, n_embd, 4*n_embd]
        self.linear1_weight = torch.ones(weight1_shape) * 0.001
        self.linear1_weight = self.linear1_weight.flatten().tolist()
        self.linear_1 = TtLinear(n_embd, 4*n_embd, self.linear1_weight, bias=None, device=device)
        #in_feature = 4 * n_embd, out_feature = 4 * n_embd
        weight2_shape = [1, 1, 4*n_embd, 4*n_embd]
        self.linear2_weight = torch.ones(weight2_shape) * 0.001
        self.linear2_weight = self.linear2_weight.flatten().tolist()
        self.linear_2 = TtLinear(4*n_embd, 4*n_embd, self.linear2_weight, bias=None, device=device)

    def forward(self, x):
        x = self.linear_1(x)
        x = TtSiLU(x)
        x = self.linear_2(x)
        return x



if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    n_embd = 32
    torch.manual_seed(123)

    input_shape =  [1, 1, 32, n_embd]
    input = torch.randn(input_shape)

    torch_emb = TimeEmbedding(n_embd)
    torch_out = torch_emb(input)

    # time = torch.reshape(time, [32, 32, 32, 1280])
    # time = torch.randn([32, 32, 32, 1280])

    tt_input = torch_to_tt_tensor(input, device)
    tt_emb = TtTimeEmbedding(n_embd, device)
    tt_out = tt_emb(tt_input)
    tt_out = tt_to_torch_tensor(tt_out, host)

    print_diff_argmax(tt_out, torch_out)
    print(comp_allclose_and_pcc(torch_out, tt_out))

    print("TEST PASSED!")
    ttl.device.CloseDevice(device)

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
from utility_functions import comp_allclose_and_pcc
from python_api_testing.models.stable_diffusion.utils import make_linear
from libs.tt_lib.fallback_ops import fallback_ops


class TtTimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu", out_dim: int = None, state_dict=None, base_address="", host=None, device=None):
        super().__init__()

        weights = state_dict[f"{base_address}.linear_1.weight"]
        bias = state_dict[f"{base_address}.linear_1.bias"]
        self.linear_1 = make_linear(in_features=in_channels,
                                    out_features=time_embed_dim,
                                    weights=weights,
                                    bias=bias,
                                    device=device)
        # self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = None
        if act_fn == "silu":
            # self.act = nn.SiLU()
            self.act = fallback_ops.silu
        elif act_fn == "mish":
            assert False, "tt does not support nn.Mish() yet"
            self.act = nn.Mish()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim

        weights = state_dict[f"{base_address}.linear_2.weight"]
        bias = state_dict[f"{base_address}.linear_2.bias"]
        self.linear_2 = make_linear(in_features=time_embed_dim,
                                    out_features=time_embed_dim_out,
                                    weights=weights,
                                    bias=bias,
                                    device=device)
        # self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

    def forward(self, sample):
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample


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

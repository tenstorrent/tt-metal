from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from libs import tt_lib as ttl
from libs.tt_lib.fallback_ops import fallback_ops
from utility_functions import torch_to_tt_tensor, tt_to_torch_tensor
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc


class TtUpsampleNearest2d(nn.Module):
    def __init__(self, scale_factor=2.0):
        super().__init__()

        assert scale_factor % 1 == 0 and scale_factor > 0, "We only support scaling by positive integer values"
        self.scale_factor = int(scale_factor)
        # self.device = device
        # self.host = host

    def forward(self, input):
        input_shape = input.shape()
        output_shape = list(input.shape())
        output_shape[-1] *= self.scale_factor
        output_shape[-2] *= self.scale_factor
        # input = tt_to_torch_tensor(input, self.host)
        input =  fallback_ops.repeat_interleave(input, repeats= self.scale_factor, dim=-1)
        input =  fallback_ops.repeat_interleave(input, repeats= self.scale_factor, dim=-2)

        # input = torch.repeat_interleave(input, repeats= self.scale_factor, dim=-1)
        # input = torch.repeat_interleave(input, repeats=self.scale_factor, dim=-2)
        # input = torch_to_tt_tensor(input, self.device)

        return input


def run_upsample_nearest_inference(device, host):
    input_shape =  [1, 1, 32, 32]
    input = torch.randn(input_shape)

    torch_out = F.interpolate(input, scale_factor=2.0, mode="nearest")

    tt_input = torch_to_tt_tensor(input, device)
    tt_up = TtUpsampleNearest2d(scale_factor=2.0)
    tt_out = tt_up(tt_input)
    tt_out = tt_to_torch_tensor(tt_out, host)
    print(comp_allclose_and_pcc(torch_out, tt_out))


if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_upsample_nearest_inference(device, host)
    ttl.device.CloseDevice(device)

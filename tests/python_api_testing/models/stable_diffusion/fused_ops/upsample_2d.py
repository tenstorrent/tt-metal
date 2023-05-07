from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
from torch import nn
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline
import numpy as np

from libs import tt_lib as ttl
from libs.tt_lib.fallback_ops import fallback_ops
from utility_functions import torch_to_tt_tensor, tt_to_torch_tensor
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc

from upsample_nearest2d import TtUpsampleNearest2d


class TtUpsample2D(nn.Module):
    def __init__(self, channels, out_channels, use_conv, use_conv_transpose, name, state_dict, base_address):
        super().__init__()
        assert not use_conv_transpose, "StableDiffusion's Unet does not use convTranspose, so leaving it out"
        self.in_channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.name = name


        self.conv = None
        if self.use_conv:
            # self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)
            self.conv_weight = state_dict[f"{base_address}.conv.weight"]
            self.conv_bias = state_dict[f"{base_address}.conv.bias"]
            self.conv = fallback_ops.Conv2d(self.conv_weight, self.conv_bias, self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, hidden_states, output_size=None):
        # conv Transpose is not our concern
        # TT's execution is done on bfloat16 - casting makes no sense
        assert hidden_states.shape()[1] == self.in_channels

        if output_size is None:
            upsampler_nearest2d = TtUpsampleNearest2d()
            hidden_states = upsampler_nearest2d(hidden_states)
        else:
            assert False, "we are not expected to support upsample 2d with output_size yet"
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")


        if self.use_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states

def run_upsample2d_inference(device, host):

    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)

    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_upblock = pipe.unet.up_blocks[0]
    resnet_upsampler = unet_upblock.upsamplers[0]

    input_shape =  [1, 1280, 32, 32]
    input = torch.randn(input_shape)
    in_channels = 1280
    out_channels = 1280
    unet_out = resnet_upsampler(input)

    print('unet_out size:', unet_out.shape)
    print('unet_out:', unet_out[0][0][0][:12])

    tt_input = torch_to_tt_tensor(input, device)

    tt_up = TtUpsample2D(channels=in_channels,
                        out_channels=out_channels,
                        use_conv=True,
                        use_conv_transpose=False,
                        name="conv",
                        state_dict=state_dict,
                        base_address="up_blocks.0.upsamplers.0")
    tt_out = tt_up(tt_input)
    tt_out = tt_to_torch_tensor(tt_out, host)
    print('tt_out size:', tt_out.shape)
    print('tt_out:', tt_out[0][0][0][:12])

    print('unet vs tt', comp_allclose_and_pcc(unet_out, tt_out), '\n')


if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_upsample2d_inference(device, host)
    ttl.device.CloseDevice(device)

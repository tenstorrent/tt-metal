from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch.nn as nn
import torch.nn.functional as F
import torch

from libs import tt_lib as ttl
from libs import tt_lib as ttl
from libs.tt_lib.fallback_ops import fallback_ops
from utility_functions import torch_to_tt_tensor, tt_to_torch_tensor
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc

from diffusers import StableDiffusionPipeline


class TtDownsample2D(nn.Module):
    """
    A downsampling layer with an optional convolution.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        out_channels:
        padding:
    """

    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv", base_address='down_blocks.0.downsamplers.0', state_dict=None):
        super().__init__()
        self.base_address = base_address
        self.state_dict=state_dict
        self.in_channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv_weight = self.state_dict[f"{base_address}.conv.weight"]
            conv_bias = self.state_dict[f"{base_address}.conv.bias"]
            conv = fallback_ops.Conv2d(conv_weight, conv_bias, self.in_channels, self.out_channels, kernel_size=3, stride=stride, padding=padding)

        else:
            assert self.in_channels == self.out_channels
            assert False, " we don't support AvgPool2d, and we should not need it either"
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states):
        assert hidden_states.shape()[1] == self.in_channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            # hidden_states = tt_to_torch_tensor(hidden_states, self.host)
            # hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)
            hidden_states = fallback_ops.pad(hidden_states, pad, mode="constant", value=0)
            # hidden_states = torch_to_tt_tensor(hidden_states, self.device)

        assert hidden_states.shape()[1] == self.in_channels
        # hidden_states = tt_to_torch_tensor(hidden_states, self.host)
        hidden_states = self.conv(hidden_states)
        # hidden_states = torch_to_tt_tensor(hidden_states, self.device)

        return hidden_states


def run_downsample2d_inference(device, host):

    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)

    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_downblock = pipe.unet.down_blocks[0]
    resnet_downsampler = unet_downblock.downsamplers[0]

    input_shape =  [1, 320, 32, 32]
    input = torch.randn(input_shape)
    in_channels = 320
    out_channels = 320
    unet_out = resnet_downsampler(input)

    print('unet_out size:', unet_out.shape)
    print('unet_out:', unet_out[0][0][0][:12])

    tt_input = torch_to_tt_tensor(input, device)

    tt_down = TtDownsample2D(channels=in_channels, out_channels=out_channels, use_conv=True, state_dict=state_dict)
    tt_out = tt_down(tt_input)
    tt_out = tt_to_torch_tensor(tt_out, host)
    print('tt_out size:', tt_out.shape)
    print('tt_out:', tt_out[0][0][0][:12])

    print('unet vs tt', comp_allclose_and_pcc(unet_out, tt_out))


if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_downsample2d_inference(device, host)
    ttl.device.CloseDevice(device)

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

from diffusers import StableDiffusionPipeline

from libs import tt_lib as ttl
from libs.tt_lib.fallback_ops import fallback_ops
from utility_functions import pad_weight, tilize_to_list, print_diff_argmax, torch_to_tt_tensor, tt_to_torch_tensor, print_corr_coef
from python_api_testing.fused_ops.linear import Linear as TtLinear
from python_api_testing.fused_ops.silu import SiLU as TtSiLU
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc


class TtResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        device=None,
        host=None,
        state_dict=None,
        base_address="encoder.mid_block.resnets.0"
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True # this is part of the original code
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.device = device
        self.host = host

        if groups_out is None:
            groups_out = groups

        norm1_weights = state_dict[f"{base_address}.norm1.weight"]
        norm1_bias = state_dict[f"{base_address}.norm1.bias"]

        self.norm1 = fallback_ops.GroupNorm(norm1_weights, norm1_bias, num_groups=groups, num_channels=in_channels, eps=eps, affine=True)


        conv1_weights = state_dict[f"{base_address}.conv1.weight"]
        conv1_bias = state_dict[f"{base_address}.conv1.bias"]

        self.conv1 = fallback_ops.Conv2d(conv1_weights, conv1_bias, in_channels, out_channels, kernel_size=3, stride=1, padding=1)


        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.time_emb_proj.weight"]))
            bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.time_emb_proj.bias"]))
            self.time_emb_proj = TtLinear(temb_channels, time_emb_proj_out_channels, weights, bias)
        else:
            self.time_emb_proj = None

        norm2_weights = state_dict[f"{base_address}.norm2.weight"]
        norm2_bias = state_dict[f"{base_address}.norm2.bias"]


        self.norm2 = fallback_ops.GroupNorm(norm2_weights, norm2_bias, num_groups=groups, num_channels=in_channels, eps=eps, affine=True)


        # self.dropout = torch.nn.Dropout(dropout)

        conv2_weights = state_dict[f"{base_address}.conv2.weight"]
        conv2_bias = state_dict[f"{base_address}.conv2.bias"]
        # self.conv2.weight = nn.Parameter(conv2_weights)
        # self.conv2.bias = nn.Parameter(conv2_bias)

        self.conv2 = fallback_ops.Conv2d(conv2_weights, conv2_bias, in_channels, out_channels, kernel_size=3, stride=1, padding=1)


        if non_linearity == "swish":
            self.nonlinearity = TtSiLU
        elif non_linearity == "mish":
            assert False, "mish is not implemented!"
            # self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = TtSiLU

        self.upsample = self.downsample = None
        if self.up:
            assert False, "we do not have tests that required this yet"
            # if kernel == "fir":
            #     fir_kernel = (1, 3, 3, 1)
            #     self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            # elif kernel == "sde_vp":
            #     self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            # else:
            #     self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            assert False, "we do not have tests that required this yet"
            # if kernel == "fir":
            #     fir_kernel = (1, 3, 3, 1)
            #     self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            # elif kernel == "sde_vp":
            #     self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            # else:
            #     self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")


        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = None
        if self.use_in_shortcut:
            # self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) # TODO
            pass

    def  forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            assert False, "we do not support upsample in resnet"
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            # if hidden_states.shape[0] >= 64:
            #     input_tensor = input_tensor.contiguous()
            #     hidden_states = hidden_states.contiguous()
            # input_tensor = self.upsample(input_tensor)
            # hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            assert False, "we do not support downsample in resnet"
            # input_tensor = self.downsample(input_tensor)
            # hidden_states = self.downsample(hidden_states)


        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            assert False, "not tested since we dont have tests for it yet"
            temp = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = ttl.tensor.add(hidden_states, temb)

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            assert False, "this is support but not tested!"
            temb = tt_to_torch_tensor(temb, self.host)
            scale, shift = torch.chunk(temb, 2, dim=1)
            temb = torch_to_tt_tensor(temb, self.device)
            shift = torch_to_tt_tensor(shift, self.device)
            scale = torch_to_tt_tensor(scale, self.device)

            ones = torch.ones(scale.shape)
            ones = torch_to_tt_tensor(ones, self.device)

            scale = ttl.tensor.add(ones, scale)
            hidden_states = ttl.tensor.mul(hidden_states, scale)
            hidden_states = ttl.tensor.add(hidden_states, shift)

        hidden_states = self.nonlinearity(hidden_states)

        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)


        # create a tensor of size output_scale_factor

        output_sc_recip = 1 / self.output_scale_factor
        output_sc_recip = torch.full(input_tensor.shape(), output_sc_recip)
        output_sc_recip = torch_to_tt_tensor(output_sc_recip, self.device)

        output_tensor = ttl.tensor.add(input_tensor, hidden_states)
        output_tensor = ttl.tensor.mul(output_tensor, output_sc_recip)


        return output_tensor


def run_resnet_inference(device):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)

    vae = pipe.vae
    vae.eval()
    state_dict = vae.state_dict()
    vae_encoder = pipe.vae.encoder
    resnet = vae_encoder.mid_block.resnets[0]

    in_channels = 512
    temb_channels = None
    eps = 1e-06
    resnet_groups = 32

    input_shape  = [1, 512, 32, 32]
    input = torch.randn(input_shape, dtype=torch.float32)
    # Note: Temb is none.

    torch_out = resnet(input, None)

    tt_input = torch_to_tt_tensor(input, device)
    tt_resnet = TtResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, groups=resnet_groups,  state_dict=state_dict, device=device, host=host)
    tt_out = tt_resnet(tt_input, None)
    tt_out = tt_to_torch_tensor(tt_out, host)

    print_diff_argmax(tt_out, torch_out)
    print(comp_allclose_and_pcc(torch_out, tt_out))



if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_resnet_inference(device)
    ttl.device.CloseDevice(device)

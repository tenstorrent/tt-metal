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

from typing import Optional
from libs import tt_lib as ttl
from utility_functions import pad_weight, print_diff_argmax, torch_to_tt_tensor, tt_to_torch_tensor, print_corr_coef
from python_api_testing.fused_ops.linear import Linear as TtLinear
from python_api_testing.fused_ops.silu import SiLU as TtSiLU
from python_api_testing.models.stable_diffusion.residual_block import TtResnetBlock2D as ResnetBlock2D
from python_api_testing.models.stable_diffusion.attention_block import TtAttentionBlock as AttentionBlock
from python_api_testing.models.stable_diffusion.unet.unet_2d_blocks import TtUNetMidBlock2D, TtDownEncoderBlock2D

from diffusers import StableDiffusionPipeline



class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
        device=None,
        host=None,
        state_dict=None,
        base_address=None,
    ):
        super().__init__()
        self.device = device
        self.host = host
        self.state_dict = state_dict

        self.layers_per_block = layers_per_block

        self.conv_in = torch.nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)
        self.conv_in.weight = nn.Parameter(state_dict[f"{base_address}.conv_in.weight"])
        self.conv_in.bias = nn.Parameter(state_dict[f"{base_address}.conv_in.bias"])

        # TODO: the weights for conv2d
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            assert down_block_type == "DownEncoderBlock2D", "other downblocks are not supported"
            down_block = TtDownEncoderBlock2D(num_layers=self.layers_per_block,
                            in_channels=input_channel,
                            out_channels=output_channel,
                            add_downsample=not is_final_block,
                            resnet_eps=1e-6,
                            downsample_padding=0,
                            resnet_act_fn=act_fn,
                            resnet_groups=norm_num_groups,
                            device=self.device,
                            host=self.host,
                            state_dict=self.state_dict,
                            base_address=f"{base_address}.down_blocks.{i}")


            # down_block = get_down_block(
            #     down_block_type,
            #     num_layers=self.layers_per_block,
            #     in_channels=input_channel,
            #     out_channels=output_channel,
            #     add_downsample=not is_final_block,
            #     resnet_eps=1e-6,
            #     downsample_padding=0,
            #     resnet_act_fn=act_fn,
            #     resnet_groups=norm_num_groups,
            #     attn_num_head_channels=None,
            #     temb_channels=None,

            # )

            self.down_blocks.append(down_block)


        self.mid_block = TtUNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
            device=device,
            host=host,
            state_dict=state_dict,
            base_address=f"{base_address}.mid_block"

        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_norm_out.weight = nn.Parameter(state_dict["encoder.conv_norm_out.weight"])
        self.conv_norm_out.bias = nn.Parameter(state_dict["encoder.conv_norm_out.bias"])

        self.conv_act = TtSiLU

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)
        self.conv_out.weight = nn.Parameter(state_dict["encoder.conv_out.weight"])
        self.conv_out.bias = nn.Parameter(state_dict["encoder.conv_out.bias"])

    def forward(self, x):
        sample = x

        sample = tt_to_torch_tensor(sample, self.host)
        sample = self.conv_in(sample)
        sample = torch_to_tt_tensor(sample, self.device)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = tt_to_torch_tensor(sample, self.host)
        sample = self.conv_norm_out(sample)
        sample = torch_to_tt_tensor(sample, self.device)

        sample = self.conv_act(sample)

        sample = tt_to_torch_tensor(sample, self.host)
        sample = self.conv_out(sample)
        sample = torch_to_tt_tensor(sample, self.device)

        return sample


def run_vae_encoder_inference(device):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)

    vae = pipe.vae
    vae.eval()
    state_dict = vae.state_dict()
    vae_encoder = pipe.vae.encoder


    in_channels = 3
    out_channels = 4
    down_block_types = ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D']
    layers_per_block = 2
    act_fn = "silu"
    norm_num_groups = 32
    block_out_channels = [128, 256, 512, 512]

    input_shape  = [1, 3, 256, 256]
    input = torch.randn(input_shape)


    torch_out = vae_encoder(input)
    print("pytorch is done, moving on to device")
    tt_input = torch_to_tt_tensor(input, device)

    tt_encoder = Encoder(in_channels=in_channels,
                        out_channels=out_channels,
                        down_block_types=down_block_types,
                        block_out_channels=block_out_channels,
                        layers_per_block= layers_per_block,
                        act_fn=act_fn,
                        norm_num_groups=norm_num_groups,
                        state_dict=state_dict,
                        device=device,
                        host=host,
                        base_address="encoder")

    tt_out = tt_encoder(tt_input)
    tt_out = tt_to_torch_tensor(tt_out, host)
    print_diff_argmax(tt_out, torch_out)
    print(comp_allclose_and_pcc(torch_out, tt_out))
    print("encoder executed successfully")



if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_vae_encoder_inference(device)
    ttl.device.CloseDevice(device)

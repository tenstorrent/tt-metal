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
from utility_functions import print_diff_argmax, torch_to_tt_tensor, tt_to_torch_tensor
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc

from python_api_testing.fused_ops.silu import SiLU as TtSiLU

from python_api_testing.models.stable_diffusion.residual_block import TtResnetBlock2D as ResnetBlock2D
from python_api_testing.models.stable_diffusion.attention_block import TtAttentionBlock as AttentionBlock
from python_api_testing.models.stable_diffusion.unet.unet_2d_blocks import TtUNetMidBlock2D, TtUpDecoderBlock2D

from diffusers import StableDiffusionPipeline


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        device=None,
        host=None,
        state_dict=None,
        base_address="decoder"
    ):
        super().__init__()
        self.device = device
        self.host = host
        self.base_address = base_address
        self.state_dict = state_dict
        assert act_fn == "silu", "other act_fn are not supported"
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)
        self.conv_in.weight = nn.Parameter(self.state_dict[f"{base_address}.conv_in.weight"])
        self.conv_in.bias = nn.Parameter(self.state_dict[f"{base_address}.conv_in.bias"])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # mid
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

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            assert up_block_type == "UpDecoderBlock2D", "Only UpDecoderBlock2D is supported"
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            # TODO: fix this

            up_block = TtUpDecoderBlock2D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift="default",
                device=self.device,
                host=self.host,
                state_dict=self.state_dict,
                base_address=f"{base_address}.up_blocks.{i}"
            )
            '''
            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            '''

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_norm_out.weight = nn.Parameter(self.state_dict[f"{base_address}.conv_norm_out.weight"])
        self.conv_norm_out.bias = nn.Parameter(self.state_dict[f"{base_address}.conv_norm_out.bias"])

        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)
        self.conv_out.weight = nn.Parameter(self.state_dict[f"{base_address}.conv_out.weight"])
        self.conv_out.bias = nn.Parameter(self.state_dict[f"{base_address}.conv_out.bias"])

    def forward(self, z):
        sample = z
        print("0. z.shape ", z.shape())
        sample = tt_to_torch_tensor(sample, self.host)
        sample = self.conv_in(sample)
        sample = torch_to_tt_tensor(sample, self.device)
        print("1. sample", sample.shape())
        # middle
        sample = self.mid_block(sample)
        print("2. sample", sample.shape())
        # up
        for up_block in self.up_blocks:
            sample = up_block(sample)
        print("3. sample", sample.shape())
        # post-process
        sample = tt_to_torch_tensor(sample, self.host)
        sample = self.conv_norm_out(sample)
        sample = torch_to_tt_tensor(sample, self.device)

        sample = self.conv_act(sample)

        sample = tt_to_torch_tensor(sample, self.host)
        sample = self.conv_out(sample)
        sample = torch_to_tt_tensor(sample, self.device)

        return sample



def run_vae_decoder_inference(device):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)

    vae = pipe.vae
    vae.eval()
    state_dict = vae.state_dict()
    vae_decoder = pipe.vae.decoder


    in_channels = 4
    out_channels = 3
    up_block_types = ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D']
    layers_per_block = 2
    act_fn = "silu"
    norm_num_groups = 32
    block_out_channels = [128, 256, 512, 512]

    input_shape  = [1, 4, 64, 64]
    input = torch.randn(input_shape)


    torch_out = vae_decoder(input)
    print("pytorch is done, moving on to device")
    tt_input = torch_to_tt_tensor(input, device)

    tt_encoder = Decoder(in_channels=in_channels,
                        out_channels=out_channels,
                        up_block_types=up_block_types,
                        block_out_channels=block_out_channels,
                        layers_per_block= layers_per_block,
                        act_fn=act_fn,
                        norm_num_groups=norm_num_groups,
                        state_dict=state_dict,
                        device=device,
                        host=host,
                        base_address="decoder")

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
    run_vae_decoder_inference(device)
    ttl.device.CloseDevice(device)

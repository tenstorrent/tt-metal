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
from diffusers import StableDiffusionPipeline

from libs import tt_lib as ttl
from libs.tt_lib.fallback_ops import fallback_ops
from utility_functions import torch_to_tt_tensor, tt_to_torch_tensor
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc

from python_api_testing.models.stable_diffusion.residual_block import TtResnetBlock2D
from python_api_testing.models.stable_diffusion.fused_ops.upsample_2d import TtUpsample2D


class TtUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        device=None,
        host=None,
        state_dict=None,
        base_address=None,
    ):
        super().__init__()
        resnets = []
        self.device = device
        self.host = host
        self.state_dict = state_dict


        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                TtResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    device=self.device,
                    host=self.host,
                    state_dict=self.state_dict,
                    base_address=f"{base_address}.resnets.{i}"
                )
            )

        # self.resnets = nn.ModuleList(resnets)
        self.resnets = resnets
        if add_upsample:
            self.upsamplers = [TtUpsample2D(channels=out_channels, out_channels=out_channels, use_conv=True, use_conv_transpose = False,name = 'op',
                                            state_dict=self.state_dict, base_address=f"{base_address}.upsamplers.0")]

        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        print('tt upblock resnets:', self.resnets)
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # res_hidden_states = tt_to_torch_tensor(res_hidden_states, self.host)
            # res_hidden_states_tuple = tt_to_torch_tensor(res_hidden_states_tuple, self.host)

            hidden_states = fallback_ops.concat([hidden_states, res_hidden_states], dim=1)

            # hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            # hidden_states = torch_to_tt_tensor(hidden_states, self.device)


            # if self.training and self.gradient_checkpointing:
            #     assert False, "we do not support training"
            #     # def create_custom_forward(module):
            #     #     def custom_forward(*inputs):
            #     #         return module(*inputs)

            #     #     return custom_forward

            #     # hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            # else:
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                print('adding upsampler..')
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


def run_upblock_inference(host, device):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)

    base_address = 'up_blocks.0'
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_upblock = pipe.unet.up_blocks[0]
    unet_resnet_upblock_module_list = unet_upblock.resnets

    in_channels = 1280
    out_channels = 1280
    prev_output_channel = 1280

    temb_channels = None
    eps = 1e-05
    resnet_groups = 32

    input_shape  = [2, 1280, 8, 8]
    hidden_state = torch.randn(input_shape, dtype=torch.float32)

    res_hidden_states_tuple = (hidden_state , hidden_state , hidden_state )


    # temb_shape  = [out_channels, out_channels]
    # temb = torch.randn(temb_shape, dtype=torch.float32)

###
    # hidden_states_shape = [2, 1280, 8, 8]
    temb_shape = [1, 1, 2, 1280]

    # input = torch.randn(hidden_states_shape)
    temb = torch.randn(temb_shape)

###
    # print('my input:', input.shape)
    unet_out = unet_upblock(hidden_state, res_hidden_states_tuple, None, None)
    # unet_out = unet_upblock.upsamplers[0](unet_out)
    print('\n\nRun tt upblock\n\n')

    # tt_hidden_states = torch_to_tt_tensor(hidden_state, device)
    tt_upblock = TtUpBlock2D(in_channels=in_channels, prev_output_channel = prev_output_channel, out_channels=out_channels, temb_channels=temb_channels, dropout= 0.0, num_layers= 3, resnet_eps= 1e-6,
                                  resnet_time_scale_shift = "default", resnet_act_fn= "silu", resnet_groups=resnet_groups, resnet_pre_norm= True, output_scale_factor=1.0,
                                  add_upsample=True, state_dict=state_dict, base_address = base_address)

    tt_out = tt_upblock(hidden_state, res_hidden_states_tuple, None, None)
    tt_out = tt_to_torch_tensor(tt_out, host)

    print('unet out shape:', unet_out.shape)
    print('tt out shape:', tt_out.shape)
    print('unet out:', unet_out[0,0,0,:12])

    print('tt out:', tt_out[0,0,0,:12])

    print(comp_allclose_and_pcc(unet_out, tt_out))

if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()
    run_upblock_inference(host, device)
    ttl.device.CloseDevice(device)

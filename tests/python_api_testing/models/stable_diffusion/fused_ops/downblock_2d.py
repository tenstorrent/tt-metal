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
from python_api_testing.models.stable_diffusion.fused_ops.downsample_2d import TtDownsample2D


class TtDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor:float=1.0,
        add_downsample:bool=True,
        downsample_padding:int=1,
        state_dict=None,
        base_address=None,
    ):
        super().__init__()
        self.state_dict = state_dict
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                TtResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    state_dict=self.state_dict,
                    base_address=f"{base_address}.resnets.{i}"
                )
            )

        # self.resnets = nn.ModuleList(resnets)
        self.resnets = resnets
        print('resnets:', self.resnets)

        if add_downsample:
            self.downsamplers = [TtDownsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op", state_dict=self.state_dict, base_address=f"{base_address}.downsamplers.0")]
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb):
        output_states = ()

        for resnet in self.resnets:
            # if self.training and self.gradient_checkpointing:
            #     assert False, "we do not support training"
            #     # def create_custom_forward(module):
            #     #     def custom_forward(*inputs):
            #     #         return module(*inputs)

            #     #     return custom_forward

            #     # hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            # else:
            hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


def run_downblock_inference(host, device):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)

    base_address = 'down_blocks.0'
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_downblock = pipe.unet.down_blocks[0]
    unet_resnet_downblock_module_list = unet_downblock.resnets

    in_channels = unet_resnet_downblock_module_list[0].conv1.in_channels
    out_channels = unet_resnet_downblock_module_list[0].conv2.in_channels
    temb_channels = 1280
    eps = 1e-05
    resnet_groups = 32

    input_shape  = [1, in_channels, 32, 32]
    input = torch.randn(input_shape, dtype=torch.float32)

    temb_shape  = [out_channels, out_channels]
    temb = torch.randn(temb_shape, dtype=torch.float32)

    unet_out = unet_resnet_downblock_module_list[0](input, None)
    unet_out = unet_resnet_downblock_module_list[1](unet_out, None)
    unet_out = unet_downblock.downsamplers[0](unet_out)


    tt_input = torch_to_tt_tensor(input, device)
    tt_downblock = TtDownBlock2D(in_channels=in_channels,
                                out_channels=out_channels,
                                temb_channels=temb_channels,
                                dropout= 0.0,
                                num_layers= 2,
                                resnet_eps= 1e-6,
                                resnet_time_scale_shift = "default",
                                resnet_act_fn= "silu",
                                resnet_groups=resnet_groups,
                                resnet_pre_norm= True,
                                output_scale_factor=1.0,
                                add_downsample=True,
                                downsample_padding=1,
                                state_dict=state_dict,
                                base_address = base_address)


    tt_out = tt_downblock(tt_input, None)[0]
    tt_out = tt_to_torch_tensor(tt_out, host)
    print('unet_out:', unet_out[0,0,0,:12])

    print('torch tt_out:', tt_out[0,0,0,:12])

    print(comp_allclose_and_pcc(unet_out, tt_out))

if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_downblock_inference(host, device)
    ttl.device.CloseDevice(device)

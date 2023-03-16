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
from utility_functions import print_diff_argmax
from utility_functions import tilize_to_list, print_diff_argmax, torch_to_tt_tensor, tt_to_torch_tensor, print_corr_coef


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
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        device=None,
        host=None,
        state_dict=None,
        base_address=None,
    ):
        super().__init__()
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
                    device=self.device,
                    host=self.host,
                    state_dict=self.state_dict,
                    base_address=f"{base_address}.resnets.{i}"
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    TtDownsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op",
                        device=self.device, host=self.host, state_dict=self.state_dict, base_address=f"{base_address}.downsamplers.0"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:
                assert False, "we do not support training"
                # def create_custom_forward(module):
                #     def custom_forward(*inputs):
                #         return module(*inputs)

                #     return custom_forward

                # hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states

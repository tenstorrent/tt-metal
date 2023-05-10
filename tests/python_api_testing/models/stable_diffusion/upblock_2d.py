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

from residual_block import TtResnetBlock2D
from upsample_2d import TtUpsample2D


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
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]


            hidden_states = fallback_ops.concat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states

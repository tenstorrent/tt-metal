from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/")
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

from typing import Optional

from python_api_testing.models.stable_diffusion.residual_block import TtResnetBlock2D as ResnetBlock2D
# from python_api_testing.models.stable_diffusion.attention_block import TtAttentionBlock as AttentionBlock
from python_api_testing.models.stable_diffusion.fused_ops.upsample_2d import TtUpsample2D as Upsample2D
from python_api_testing.models.stable_diffusion.fused_ops.downsample_2d import TtDownsample2D as Downsample2D
from python_api_testing.models.stable_diffusion.unet.transformer_2d import TtTransformer2DModel


class TtUNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
        state_dict=None,
        base_address="",
        device=None,
        host=None,
    ):
        super().__init__()
        self.base_address_with_dot = "" if base_address=="" else f"{base_address}."
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        self.device = device
        self.host = host
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
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
                state_dict=state_dict,
                base_address=f"{self.base_address_with_dot}resnets.0"

            )
        ]
        attentions = []

        for _ in range(num_layers):
            if not dual_cross_attention:

                attentions.append(
                    TtTransformer2DModel(
                        num_attention_heads= attn_num_head_channels,
                        attention_head_dim =in_channels // attn_num_head_channels,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        state_dict=state_dict,
                        base_address=f"{self.base_address_with_dot}attentions.{len(attentions)}",
                        device=self.device,
                        host=self.host
                    )
                )
            else:
                assert False, "this should not be triggered"
                attentions.append(
                    DualTransformer2DModel(
                        attn_num_head_channels,
                        in_channels // attn_num_head_channels,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    state_dict=state_dict,
                    base_address=f"{self.base_address_with_dot}resnets.{len(resnets)}",
                    device=self.device,
                    host=self.host
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None, cross_attention_kwargs=None
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            )
            hidden_states = resnet(hidden_states, temb)

        return hidden_states




# in_channels = 1280
# temb_channels = 1280
# dropout = 0.0
# num_layers = 1
# resnet_eps = 1e-05
# resnet_time_scale_shift = default
# resnet_act_fn = silu
# resnet_groups = 32
# resnet_pre_norm = True
# attn_num_head_channels = 8
# output_scale_factor = 1
# cross_attention_dim = 768
# dual_cross_attention = False
# use_linear_projection = False
# upcast_attention = False


# sample shape torch.Size([2, 1280, 8, 8])
# emb shape torch.Size([2, 1280])
# encoder hidden state shape torch.Size([2, 77, 768])
# attention mask None
# cross attentino kwargs None




# class TtUpBlock2d(nn.Module):
#     pass
#
# class TtDownBlock2d(nn.Module):
#     pass
#
# class TtUpDecoderBlock2D(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         dropout: float = 0.0,
#         num_layers: int = 1,
#         resnet_eps: float = 1e-6,
#         resnet_time_scale_shift: str = "default",
#         resnet_act_fn: str = "swish",
#         resnet_groups: int = 32,
#         resnet_pre_norm: bool = True,
#         output_scale_factor=1.0,
#         add_upsample=True,
#         device=None,
#         host=None,
#         state_dict=None,
#         base_address=None,
#     ):
#         super().__init__()
#         resnets = []
#         self.device = device
#         self.host = host
#         self.state_dict = state_dict

#         for i in range(num_layers):
#             input_channels = in_channels if i == 0 else out_channels

#             resnets.append(
#                 ResnetBlock2D(
#                     in_channels=input_channels,
#                     out_channels=out_channels,
#                     temb_channels=None,
#                     eps=resnet_eps,
#                     groups=resnet_groups,
#                     dropout=dropout,
#                     time_embedding_norm=resnet_time_scale_shift,
#                     non_linearity=resnet_act_fn,
#                     output_scale_factor=output_scale_factor,
#                     pre_norm=resnet_pre_norm,
#                     device=self.device,
#                     host=self.host,
#                     state_dict=self.state_dict,
#                     base_address=f"{base_address}.resnets.{i}"
#                 )
#             )

#         self.resnets = nn.ModuleList(resnets)

#         if add_upsample:
#             self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True,
#                                             out_channels=out_channels,
#                                             device=self.device,
#                                             host=self.host,
#                                             state_dict=self.state_dict,
#                                             base_address=f"{base_address}.upsamplers.0")])
#         else:
#             self.upsamplers = None

#     def forward(self, hidden_states):
#         for resnet in self.resnets:
#             hidden_states = resnet(hidden_states, temb=None)

#         if self.upsamplers is not None:
#             for upsampler in self.upsamplers:
#                 hidden_states = upsampler(hidden_states)

#         return hidden_states


# class TtDownEncoderBlock2D(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         dropout: float = 0.0,
#         num_layers: int = 1,
#         resnet_eps: float = 1e-6,
#         resnet_time_scale_shift: str = "default",
#         resnet_act_fn: str = "swish",
#         resnet_groups: int = 32,
#         resnet_pre_norm: bool = True,
#         output_scale_factor=1.0,
#         add_downsample=True,
#         downsample_padding=1,
#         device=None,
#         host=None,
#         state_dict=None,
#         base_address=None
#     ):
#         super().__init__()

#         self.device = device
#         self.host = host
#         self.base_address = base_address
#         self.state_dict = state_dict

#         resnets = []

#         for i in range(num_layers):
#             in_channels = in_channels if i == 0 else out_channels
#             resnets.append(
#                 ResnetBlock2D(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     temb_channels=None,
#                     eps=resnet_eps,
#                     groups=resnet_groups,
#                     dropout=dropout,
#                     time_embedding_norm=resnet_time_scale_shift,
#                     non_linearity=resnet_act_fn,
#                     output_scale_factor=output_scale_factor,
#                     pre_norm=resnet_pre_norm,
#                     device=self.device,
#                     host=self.host,
#                     base_address=f"{base_address}.resnets.{i}",
#                     state_dict=self.state_dict
#                 )
#             )

#         self.resnets = nn.ModuleList(resnets)

#         if add_downsample:
#             self.downsamplers = nn.ModuleList(
#                 [
#                     Downsample2D(
#                         out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op", device=self.device, host=self.host, base_address=f"{base_address}.downsamplers.0", state_dict=self.state_dict
#                     )
#                 ]
#             )
#         else:
#             self.downsamplers = None


#     def forward(self, hidden_states):
#         for resnet in self.resnets:
#             hidden_states = resnet(hidden_states, temb=None)

#         if self.downsamplers is not None:
#             for downsampler in self.downsamplers:
#                 hidden_states = downsampler(hidden_states)

#         return hidden_states


# class TtUNetMidBlock2D(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         temb_channels: int,
#         dropout: float = 0.0,
#         num_layers: int = 1,
#         resnet_eps: float = 1e-6,
#         resnet_time_scale_shift: str = "default",
#         resnet_act_fn: str = "swish",
#         resnet_groups: int = 32,
#         resnet_pre_norm: bool = True,
#         add_attention: bool = True,
#         attn_num_head_channels=1,
#         output_scale_factor=1.0,
#         host=None,
#         device=None,
#         state_dict=None,
#         base_address="encoder.mid_block"
#     ):
#         super().__init__()
#         resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
#         self.add_attention = add_attention
#         self.host = host
#         self.device = device
#         self.state_dict = state_dict
#         assert resnet_act_fn=="silu", "we do not support all the activations at this time"
#         # there is always at least one resnet
#         resnets = [
#             ResnetBlock2D(
#                 in_channels=in_channels,
#                 out_channels=in_channels,
#                 temb_channels=temb_channels,
#                 eps=resnet_eps,
#                 groups=resnet_groups,
#                 dropout=dropout,
#                 time_embedding_norm=resnet_time_scale_shift,
#                 non_linearity=resnet_act_fn,
#                 output_scale_factor=output_scale_factor,
#                 pre_norm=resnet_pre_norm,
#                 host=self.host,
#                 device=self.device,
#                 state_dict=self.state_dict,
#                 base_address=f"{base_address}.resnets.0"
#             )
#         ]
#         attentions = []

#         for _ in range(num_layers):
#             if self.add_attention:
#                 attentions.append(
#                     AttentionBlock(
#                         in_channels,
#                         num_head_channels=attn_num_head_channels,
#                         rescale_output_factor=output_scale_factor,
#                         eps=resnet_eps,
#                         norm_num_groups=resnet_groups,
#                         host=self.host,
#                         device=self.device,
#                         state_dict=self.state_dict,
#                         base_address=f"{base_address}.attentions.{_}"
#                     )
#                 )
#             else:
#                 attentions.append(None)

#             resnets.append(
#                 ResnetBlock2D(
#                     in_channels=in_channels,
#                     out_channels=in_channels,
#                     temb_channels=temb_channels,
#                     eps=resnet_eps,
#                     groups=resnet_groups,
#                     dropout=dropout,
#                     time_embedding_norm=resnet_time_scale_shift,
#                     non_linearity=resnet_act_fn,
#                     output_scale_factor=output_scale_factor,
#                     pre_norm=resnet_pre_norm,
#                     host=self.host,
#                     device=self.device,
#                     state_dict=self.state_dict,
#                     base_address=f"{base_address}.resnets.1"
#                 )
#             )

#         self.attentions = nn.ModuleList(attentions)
#         self.resnets = nn.ModuleList(resnets)


#     def forward(self, hidden_states, temb=None):
#         hidden_states = self.resnets[0](hidden_states, temb)
#         for attn, resnet in zip(self.attentions, self.resnets[1:]):
#             if attn is not None:
#                 hidden_states = attn(hidden_states)
#             hidden_states = resnet(hidden_states, temb)

#         return hidden_states

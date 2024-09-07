# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

import ttnn

from typing import Optional

from models.experimental.stable_diffusion.tt.residual_block import TtResnetBlock2D as ResnetBlock2D
from models.experimental.stable_diffusion.tt.upsample_2d import TtUpsample2D as Upsample2D
from models.experimental.stable_diffusion.tt.downsample_2d import TtDownsample2D as Downsample2D
from models.experimental.stable_diffusion.tt.transformer_2d import (
    TtTransformer2DModel as Transformer2DModel,
)

from models.experimental.stable_diffusion.tt.experimental_ops import concat

####################### UNet Mid Block Cross Attention #######################


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
        self.base_address_with_dot = "" if base_address == "" else f"{base_address}."
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
                base_address=f"{self.base_address_with_dot}resnets.0",
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads=attn_num_head_channels,
                        attention_head_dim=in_channels // attn_num_head_channels,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        state_dict=state_dict,
                        base_address=f"{self.base_address_with_dot}attentions.{len(attentions)}",
                        device=self.device,
                        host=self.host,
                    )
                )
            else:
                assert False, "We do not support Dual Transformer"

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
                    host=self.host,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        cross_attention_kwargs=None,
    ) -> ttnn.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            )
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


####################### Cross Attention Up Block #######################


class TtCrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        state_dict=None,
        base_address="",
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
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
                    state_dict=state_dict,
                    base_address=f"{base_address}.resnets.{i}",
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        state_dict=state_dict,
                        base_address=f"{base_address}.attentions.{i}",
                    )
                )
            else:
                assert False, "We do not support Dual Transformer2DModel"

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        state_dict=state_dict,
                        base_address=f"{base_address}.upsamplers.0",
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        cross_attention_kwargs=None,
        upsample_size=None,
        attention_mask=None,
    ) -> ttnn.Tensor:
        # TODO(Patrick, William) - attention mask is not used
        device = ttnn.GetDefaultDevice()
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            if isinstance(res_hidden_states, (ttnn.Tensor,)):
                on_dev_res_hidden_states = res_hidden_states
            else:
                on_dev_res_hidden_states = ttnn.Tensor(
                    res_hidden_states.reshape(-1).tolist(),
                    res_hidden_states.shape,
                    ttnn.bfloat16,
                    ttnn.ROW_MAJOR_LAYOUT,
                ).to(device)

            hidden_states = concat([hidden_states, on_dev_res_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:
                assert False, "We do not support Training"
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


####################### Cross Attention Down Block #######################


class TtCrossAttnDownBlock2D(nn.Module):
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
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        state_dict=None,
        base_address="",
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels
        self.dropout = dropout
        self.num_layers = num_layers
        self.resnet_eps = resnet_eps
        self.resnet_time_scale_shift = resnet_time_scale_shift
        self.resnet_act_fn = resnet_act_fn
        self.resnet_groups = resnet_groups
        self.resnet_pre_norm = resnet_pre_norm
        self.attn_num_head_channels = attn_num_head_channels
        self.cross_attention_dim = cross_attention_dim
        self.output_scale_factor = output_scale_factor
        self.downsample_padding = downsample_padding
        self.add_downsample = add_downsample
        self.dual_cross_attention = dual_cross_attention
        self.use_linear_projection = use_linear_projection
        self.only_cross_attention = only_cross_attention
        self.upcast_attention = upcast_attention

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
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
                    state_dict=state_dict,
                    base_address=f"{base_address}.resnets.{i}",
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        state_dict=state_dict,
                        base_address=f"{base_address}.attentions.{i}",
                    )
                )
            else:
                assert False, "We do not support Dual Transformer"

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                        state_dict=state_dict,
                        base_address=f"{base_address}.downsamplers.0",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def __str__(self):
        lst = ["##### info on crossattndownblock #####"]

        lst.append(f"in_channels  = {self.in_channels}")
        lst.append(f"out_channels  = {self.out_channels}")
        lst.append(f"temb_channels  = {self.temb_channels}")
        lst.append(f"dropout  = {self.dropout}")
        lst.append(f"num_layers  = {self.num_layers}")
        lst.append(f"resnet_eps  = {self.resnet_eps}")
        lst.append(f"resnet_time_scale_shift  = {self.resnet_time_scale_shift}")
        lst.append(f"resnet_act_fn  = {self.resnet_act_fn}")
        lst.append(f"resnet_groups  = {self.resnet_groups}")
        lst.append(f"resnet_pre_norm  = {self.resnet_pre_norm}")
        lst.append(f"attn_num_head_channels  = {self.attn_num_head_channels}")
        lst.append(f"cross_attention_dim  = {self.cross_attention_dim}")
        lst.append(f"output_scale_factor  = {self.output_scale_factor}")
        lst.append(f"downsample_padding  = {self.downsample_padding}")
        lst.append(f"add_downsample  = {self.add_downsample}")
        lst.append(f"dual_cross_attention  = {self.dual_cross_attention}")
        lst.append(f"use_linear_projection  = {self.use_linear_projection}")
        lst.append(f"only_cross_attention  = {self.only_cross_attention}")
        lst.append(f"upcast_attention  = {self.upcast_attention}")

        lst.append("##### end of cross att down blck ##### ")
        return "\n".join(lst)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        cross_attention_kwargs=None,
    ) -> ttnn.Tensor:
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:
                assert False, "we are not training"

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    cross_attention_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states

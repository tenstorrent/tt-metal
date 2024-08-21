# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn

from models.experimental.vovnet.tt.conv_norm_act import TtConvNormAct
from models.experimental.vovnet.tt.sequential_append_list import (
    TtSequentialAppendList,
)
from models.experimental.vovnet.tt.effective_se_module import (
    TtEffectiveSEModule,
)


class TtOsaBlock(nn.Module):
    def __init__(
        self,
        in_chs=1,
        mid_chs=64,
        out_chs=64,
        layer_per_block=3,
        groups=64,
        residual=True,
        depthwise=True,
        base_address=None,
        state_dict=None,
        device=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.residual = residual
        self.depthwise = depthwise
        self.state_dict = state_dict

        next_in_chs = in_chs
        if self.depthwise and next_in_chs != mid_chs:
            assert not residual
            self.conv_reduction = TtConvNormAct(
                in_channels=next_in_chs,
                out_channels=mid_chs,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                apply_act=True,
                norm_kwargs=None,
                act_kwargs=None,
                state_dict=state_dict,
                base_address=f"{base_address}.conv_reduction",
                device=self.device,
            )
        else:
            self.conv_reduction = None

        self.conv_mid = TtSequentialAppendList(
            layer_per_block=layer_per_block,
            state_dict=state_dict,
            base_address=f"{base_address}",
            in_channels=mid_chs,
            groups=groups,
        )

        # feature aggregation
        next_in_chs = in_chs + layer_per_block * mid_chs
        self.conv_concat = TtConvNormAct(
            in_channels=next_in_chs,
            out_channels=out_chs,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            apply_act=True,
            norm_kwargs=None,
            act_kwargs=None,
            state_dict=state_dict,
            base_address=f"{base_address}.conv_concat",
            device=device,
        )

        self.attn = TtEffectiveSEModule(
            in_channels=out_chs,
            out_channels=out_chs,
            kernel_size=1,
            stride=1,
            dilation=1,
            padding=0,
            bias=False,
            state_dict=state_dict,
            base_address=f"{base_address}.attn",
            device=device,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        output = [x]
        if self.conv_reduction is not None:
            x = self.conv_reduction(x)
        x = self.conv_mid(x, output)
        x = self.conv_concat(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.residual:
            x = x + output[0]
        return x

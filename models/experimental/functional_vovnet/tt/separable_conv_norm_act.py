# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch.nn as nn

import ttnn

from tt_lib import fallback_ops
from models.utility_functions import (
    torch_to_tt_tensor_rm,
)
from models.experimental.vovnet.vovnet_utils import create_batchnorm
from models.experimental.functional_vovnet.tt.common import Conv


class TtSeparableConvNormAct(nn.Module):
    def __init__(
        self,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        base_address=None,
        device=None,
        torch_model=None,
        channel_multiplier: float = 1.0,
        split_conv=False,
    ):
        super(TtSeparableConvNormAct, self).__init__()
        self.device = device

        bias = None
        self.conv_dw = Conv(
            device=device,
            model=torch_model,
            # parameters,
            path=base_address,
            # input_params = input_params,
            conv_params=[stride, stride, padding, padding],
            # enable_split_reader=True,
            # enable_act_double_buffer=True,
            split_conv=split_conv,
            fused_op=False,
            activation="",
        )

        self.conv_pw = Conv(
            device=device,
            model=torch_model,
            # parameters,
            path=base_address,
            # input_params = input_params,
            conv_params=[1, 1, 0, 0],
            # enable_split_reader=True,
            # enable_act_double_buffer=True
            fused_op=False,
            activation="",
            seperable_conv_norm_act=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        conv_dw = self.conv_dw(x)
        print("Shape of conv_dw final check:", conv_dw[0].shape)
        conv_pw = self.conv_pw(conv_dw[0])

        return conv_pw

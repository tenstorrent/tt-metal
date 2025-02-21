# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch.nn as nn
import ttnn
from models.experimental.functional_vovnet.tt.common import Conv


class TtSeparableConvNormAct(nn.Module):
    def __init__(
        self,
        stride: int = 1,
        padding: int = 1,
        base_address=None,
        device=None,
        # torch_model=None,
        parameters=None,
        split_conv=False,
    ):
        super(TtSeparableConvNormAct, self).__init__()
        self.device = device

        self.conv_dw = Conv(
            device=device,
            # model=torch_model,
            path=base_address,
            conv_params=[stride, stride, padding, padding],
            split_conv=split_conv,
            fused_op=False,
            activation="",
            parameters=parameters,
        )

        self.conv_pw = Conv(
            device=device,
            # model=torch_model,
            path=base_address,
            conv_params=[1, 1, 0, 0],
            fused_op=True,
            activation="relu",
            seperable_conv_norm_act=True,
            pw=True,
            parameters=parameters,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.conv_dw(x)
        x = self.conv_pw(x[0])

        return x

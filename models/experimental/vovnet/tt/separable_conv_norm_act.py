# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.vovnet.tt.common import Conv


class TtSeparableConvNormAct:
    def __init__(
        self,
        stride: int = 1,
        padding: int = 1,
        base_address=None,
        device=None,
        parameters=None,
        split_conv=False,
    ):
        self.device = device

        self.conv_dw = Conv(
            device=device,
            path=base_address,
            conv_params=[stride, stride, padding, padding],
            split_conv=split_conv,
            fused_op=False,
            activation="",
            parameters=parameters,
        )

        self.conv_pw = Conv(
            device=device,
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

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.vovnet.tt.common import Conv

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


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
            activation=None,
            parameters=parameters,
        )

        self.conv_pw = Conv(
            device=device,
            path=base_address,
            conv_params=[1, 1, 0, 0],
            fused_op=True,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            pw=True,
            parameters=parameters,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if use_signpost:
            signpost(header="seperable_conv_norm")

        x = self.conv_dw(x)
        x = self.conv_pw(x[0])

        return x

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.functional_vovnet.tt.common import Conv


class TtConvNormAct:
    def __init__(
        self,
        stride: int = 1,
        padding: int = 1,
        base_address=None,
        device=None,
        parameters=None,
    ) -> None:
        self.device = device
        self.conv = Conv(
            device=device,
            path=base_address,
            conv_params=[stride, stride, padding, padding],
            activation="relu",
            parameters=parameters,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.conv(x)
        return x

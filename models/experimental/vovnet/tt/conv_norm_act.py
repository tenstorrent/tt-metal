# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.vovnet.tt.common import Conv

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class TtConvNormAct:
    def __init__(
        self,
        stride: int = 1,
        padding: int = 1,
        base_address=None,
        device=None,
        parameters=None,
        deallocate_activation=False,
        lay_idx=1000,
    ) -> None:
        self.device = device
        self.lay_idx = lay_idx
        self.base_address = base_address
        self.conv = Conv(
            device=device,
            path=base_address,
            conv_params=[stride, stride, padding, padding],
            activation="relu",
            parameters=parameters,
            deallocate_activation=deallocate_activation,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if use_signpost:
            signpost(header="conv_norm_act")

        x = self.conv(x)
        return x

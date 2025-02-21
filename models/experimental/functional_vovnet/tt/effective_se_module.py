# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.functional_vovnet.tt.common import Conv


class TtEffectiveSEModule:
    def __init__(
        self,
        stride: int = 1,
        padding: int = 1,
        # torch_model=None,
        parameters=None,
        device=None,
        base_address=None,
        **_,
    ):
        self.device = device
        self.base_address = base_address

        self.fc = Conv(
            device=device,
            # model=torch_model,
            parameters=parameters,
            path=base_address,
            conv_params=[stride, stride, padding, padding],
            fused_op=False,
            effective_se=True,
        )

        self.activation = ttnn.hardsigmoid

    def forward(self, input: ttnn.Tensor) -> ttnn.Tensor:
        out = ttnn.mean(input, dim=[2, 3], keepdim=True)
        out = self.fc(out)[0]
        out = self.activation(out)
        out = ttnn.multiply(input, out)
        return out

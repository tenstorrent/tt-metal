# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.functional_vovnet.tt.common import Conv


class TtEffectiveSEModule:
    def __init__(
        self,
        stride: int = 1,
        padding: int = 1,
        parameters=None,
        device=None,
        base_address=None,
        **_,
    ):
        self.device = device
        self.base_address = base_address

        self.fc = Conv(
            device=device,
            parameters=parameters,
            path=base_address,
            conv_params=[stride, stride, padding, padding],
            fused_op=False,
            effective_se=True,
        )

        self.activation = ttnn.hardsigmoid

    def forward(self, input: ttnn.Tensor) -> ttnn.Tensor:
        out = ttnn.mean(input, dim=[2, 3], keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = self.fc(out)[0]
        out = self.activation(out, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.multiply(input, out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(input)
        return out

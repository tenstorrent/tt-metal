# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn.torch_tracer
from models.experimental.functional_vovnet.tt.select_adaptive_pool2d import (
    TtSelectAdaptivePool2d,
)
import ttnn


class TtClassifierHead:
    def __init__(
        self,
        device=None,
        # torch_model=None
        parameters=None,
        base_address=None,
    ):
        self.device = device
        # self.torch_model = torch_model

        self.base_address = base_address
        self.weight = parameters[f"{base_address}.fc.weight"]
        self.bias = parameters[f"{base_address}.fc.bias"]

        self.global_pool = TtSelectAdaptivePool2d(output_size=1, device=self.device)

    def forward(self, x):
        x = self.global_pool.forward(x)

        # weight = ttnn.from_torch(
        #     self.torch_model[self.base_address + ".fc.weight"].unsqueeze(0).unsqueeze(0),
        #     device=self.device,
        #     dtype=ttnn.bfloat16,
        #     layout=ttnn.TILE_LAYOUT,
        # )
        # weight = ttnn.permute(weight, (0, 1, 3, 2))
        # bias = ttnn.from_torch(
        #     self.torch_model[self.base_address + ".fc.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0),
        #     device=self.device,
        #     dtype=ttnn.bfloat16,
        #     layout=ttnn.TILE_LAYOUT,
        # )
        # x = ttnn.linear(x, weight, bias=bias, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        print("Shape of  xx:", x.shape, " ", self.weight.shape, " ", self.bias.shape)
        x = ttnn.linear(x, self.weight, bias=self.bias, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

        x = ttnn.reshape(x, [x.shape[0], 1000])
        return x

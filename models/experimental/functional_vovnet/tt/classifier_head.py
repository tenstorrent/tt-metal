# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn.torch_tracer
import ttnn


class TtClassifierHead:
    def __init__(
        self,
        device=None,
        parameters=None,
        base_address=None,
    ):
        self.device = device

        self.base_address = base_address
        self.weight = parameters[f"{base_address}.fc.weight"]
        self.bias = parameters[f"{base_address}.fc.bias"]

    def forward(self, x):
        x = ttnn.permute(x, (0, 2, 3, 1))
        x = ttnn.global_avg_pool2d(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.permute(x, (0, 3, 1, 2))
        x = ttnn.reshape(x, [x.shape[0], 1, 1, x.shape[1] * x.shape[2] * x.shape[3]])
        x = ttnn.linear(
            x,
            self.weight,
            bias=self.bias,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=x.shape[0], x=8),
        )

        x = ttnn.reshape(x, [x.shape[0], 1000])
        return x

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.transfuser.tt.self_attn import TTSelfAttention


class TTMlp(LightweightModule):
    def __init__(self, device, parameters=None, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG):
        self.device = device
        self.dtype = dtype
        self.memory_config = memory_config
        self.parameters = parameters

    def forward(self, x):
        if x.memory_config().buffer_type != ttnn.BufferType.L1:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.linear(
            x,
            self.parameters[0]["weight"],
            bias=self.parameters[0]["bias"],
            memory_config=self.memory_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            activation="relu",
            dtype=self.dtype,
        )

        x = ttnn.linear(
            x,
            self.parameters[2]["weight"],
            bias=self.parameters[2]["bias"],
            memory_config=self.memory_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=self.dtype,
        )

        return x


class TTGptBlock(LightweightModule):
    def __init__(self, device, parameters, n_head, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG):
        self.parameters = parameters
        self.device = device
        self.n_head = n_head
        self.dtype = ttnn.bfloat16
        self.memory_config = memory_config
        self.attn = TTSelfAttention(device, parameters["attn"], n_head, dtype=dtype, memory_config=memory_config)
        self.mlp = TTMlp(device, parameters["mlp"], dtype=dtype, memory_config=memory_config)

    def forward(self, x):
        B, T, C = x.shape

        ln1 = ttnn.layer_norm(x, weight=self.parameters["ln1"]["weight"], bias=self.parameters["ln1"]["bias"])
        x = x + self.attn(ln1)
        ln2 = ttnn.layer_norm(x, weight=self.parameters["ln2"]["weight"], bias=self.parameters["ln2"]["bias"])
        x = x + ln2
        x = x + self.mlp(x)
        return x

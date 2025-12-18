# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.transfuser.tt.self_attn import TTSelfAttention


class TTMlp(LightweightModule):
    def __init__(
        self,
        device,
        parameters=None,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=None,
    ):
        self.device = device
        self.dtype = dtype
        self.memory_config = memory_config
        self.parameters = parameters
        self.compute_kernel_config = compute_kernel_config

    def forward(self, x):
        if x.memory_config().buffer_type != ttnn.BufferType.L1:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)

        x = ttnn.linear(
            x,
            self.parameters["mlp_0_weight"],
            bias=self.parameters["mlp_0_bias"],
            compute_kernel_config=self.compute_kernel_config,
            memory_config=self.memory_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            activation="relu",
            dtype=self.dtype,
        )

        x = ttnn.linear(
            x,
            self.parameters["mlp_2_weight"],
            bias=self.parameters["mlp_2_bias"],
            compute_kernel_config=self.compute_kernel_config,
            memory_config=self.memory_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=self.dtype,
        )

        return x


class TTGptBlock(LightweightModule):
    def __init__(
        self,
        device,
        parameters,
        n_embed,
        n_head,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=None,
    ):
        self.parameters = parameters
        self.device = device
        self.n_embed = n_embed
        self.n_head = n_head
        self.dtype = ttnn.bfloat16
        self.memory_config = memory_config
        self.compute_kernel_config = compute_kernel_config
        self.attn = TTSelfAttention(
            device,
            parameters["attn"],
            n_embed,
            n_head,
            dtype=dtype,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )
        self.mlp = TTMlp(
            device, parameters, dtype=dtype, memory_config=memory_config, compute_kernel_config=compute_kernel_config
        )

    def forward(self, x):
        ln1 = ttnn.layer_norm(x, weight=self.parameters["ln1_weight"], bias=self.parameters["ln1_bias"])
        x = ttnn.add(x, self.attn(ln1), memory_config=self.memory_config)
        ln2 = ttnn.layer_norm(x, weight=self.parameters["ln2_weight"], bias=self.parameters["ln2_bias"])
        # x = ttnn.add(x, ln2, memory_config=self.memory_config)
        # mlp = self.mlp(x)
        mlp = self.mlp(ln2)
        x = ttnn.add(x, mlp, memory_config=self.memory_config)
        return x

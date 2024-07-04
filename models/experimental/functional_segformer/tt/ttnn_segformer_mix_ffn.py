# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_segformer.tt.ttnn_segformer_dwconv import TtSegformerDWConv


class TtSegformerMixFFN:
    def __init__(self, parameters, hidden_features):
        super().__init__()
        self.dwconv = TtSegformerDWConv(parameters["dwconv"], hidden_features)

    def __call__(self, hidden_states: ttnn.Tensor, height: int, width: int, parameters, device):
        hidden_states = ttnn.linear(
            hidden_states,
            parameters.dense1.weight,
            bias=parameters.dense1.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            dtype=ttnn.bfloat8_b,
        )
        hidden_states = self.dwconv(hidden_states, height, width, device)
        hidden_states = ttnn.gelu(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.linear(
            hidden_states,
            parameters.dense2.weight,
            bias=parameters.dense2.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            dtype=ttnn.bfloat16,
        )
        return hidden_states

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtSegformerSelfOutput:
    def __init__(self):
        super().__init__()

    def __call__(self, hidden_states: ttnn.Tensor, parameters):
        hidden_states = ttnn.linear(
            hidden_states,
            parameters.dense.weight,
            bias=parameters.dense.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            dtype=ttnn.bfloat8_b,
        )
        return hidden_states

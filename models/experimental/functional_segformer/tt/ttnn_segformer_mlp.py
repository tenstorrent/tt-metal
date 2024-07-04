# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtSegformerMLP:
    def __init__(self):
        super().__init__()

    def __call__(self, hidden_states: ttnn.Tensor, parameters):
        device = hidden_states.device()
        hidden_states = ttnn.from_device(hidden_states)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(
            hidden_states,
            (hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2] * hidden_states.shape[3]),
        )
        hidden_states = ttnn.to_device(hidden_states, device=device)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
        if len(hidden_states.shape) == 2:  # This is due to while permuting 1,x,y we are getting 2D as output
            hidden_states = ttnn.reshape(hidden_states, (1, hidden_states.shape[0], hidden_states.shape[1]))
        hidden_states = ttnn.linear(
            hidden_states,
            parameters.proj.weight,
            bias=parameters.proj.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
        )
        return hidden_states

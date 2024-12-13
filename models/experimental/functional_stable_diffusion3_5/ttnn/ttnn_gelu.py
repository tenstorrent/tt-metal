# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import ttnn


class ttnn_GELU:
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        self.proj = ttnn.linear
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if True:  # gate.device.type != "mps":  , In torch its executed
            if self.approximate == "tanh":
                approximate_bool = True
            else:
                approximate_bool = False
            return ttnn.gelu(gate, fast_and_approximate_mode=approximate_bool)
        # This is not invoked in our call
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)

    def __call__(self, hidden_states, parameters=None):
        hidden_states = self.proj(
            hidden_states, input_tensor_b=parameters["proj"]["weight"], bias=parameters["proj"]["bias"]
        )
        hidden_states = self.gelu(hidden_states)
        return hidden_states

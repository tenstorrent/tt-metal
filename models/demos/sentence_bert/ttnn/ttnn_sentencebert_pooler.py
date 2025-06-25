# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtnnSentenceBertPooler:
    def __init__(self, parameters):
        self.dense = ttnn.linear
        self.activation = ttnn.tanh
        self.parameters = parameters

    def __call__(self, hidden_states: ttnn.Tensor):
        hidden_states = ttnn.squeeze(hidden_states, dim=1)
        first_token_tensor = hidden_states[:, 0, :]
        pooled_output = self.dense(
            first_token_tensor,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
        pooled_output = self.activation(pooled_output)
        return pooled_output

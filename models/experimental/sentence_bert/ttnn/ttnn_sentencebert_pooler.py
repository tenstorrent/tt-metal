# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtnnSentenceBertPooler:
    def __init__(self, parameters):
        self.dense = ttnn.linear
        self.activation = ttnn.tanh
        self.parameters = parameters

    def __call__(self, hidden_states: ttnn.Tensor):
        first_token_tensor = hidden_states[:, 0, :]
        pooled_output = self.dense(
            first_token_tensor,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=first_token_tensor.shape[0], x=8),
        )
        pooled_output = self.activation(pooled_output)
        return pooled_output

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.bge_large_en.ttnn.ttnn_bge_self_attention import TtnnBGESelfAttention
from models.demos.bge_large_en.ttnn.ttnn_bge_self_output import TtnnBGESelfOutput


class TtnnBGEAttention:
    def __init__(self, parameters, config):
        self.self = TtnnBGESelfAttention(parameters.self, config)
        self.output = TtnnBGESelfOutput(parameters.output, config)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        device=None,
    ):
        self_outputs = self.self(hidden_states, attention_mask, device=device)
        self_outputs = self.output(self_outputs, hidden_states)
        return self_outputs

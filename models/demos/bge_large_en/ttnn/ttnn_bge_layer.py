# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.bge_large_en.ttnn.ttnn_bge_attention import TtnnBGEAttention
from models.demos.bge_large_en.ttnn.ttnn_bge_intermediate import TtnnBGEIntermediate
from models.demos.bge_large_en.ttnn.ttnn_bge_output import TtnnBGEOutput


class TtnnBGELayer:
    def __init__(self, parameters, config):
        self.attention = TtnnBGEAttention(parameters.attention, config)
        self.intermediate = TtnnBGEIntermediate(parameters.intermediate)
        self.output = TtnnBGEOutput(parameters.output, config)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        device=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, device=device)
        ttnn.deallocate(hidden_states)
        self_attention_outputs = ttnn.reallocate(self_attention_outputs)
        intermediate_output = self.intermediate(self_attention_outputs)
        layer_output = self.output(intermediate_output, self_attention_outputs)
        layer_output = ttnn.reallocate(layer_output)
        return layer_output

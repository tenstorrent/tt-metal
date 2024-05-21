# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.functional_segformer.tt.ttnn_segformer_efficient_selfattention import (
    TtSegformerEfficientSelfAttention,
)
from models.experimental.functional_segformer.tt.ttnn_segformer_selfoutput import TtSegformerSelfOutput
import ttnn


class TtSegformerAttention:
    def __init__(self, hidden_size, num_attention_heads, parameters, sequence_reduction_ratio, model):
        super().__init__()
        self.self = TtSegformerEfficientSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            parameters=parameters.self,
            sequence_reduction_ratio=sequence_reduction_ratio,
            model=model.self,
        )
        self.output = TtSegformerSelfOutput()

    def __call__(self, hidden_states: ttnn.Tensor, height: int, width: int, parameters, output_attentions=False):
        self_outputs = self.self(hidden_states, height, width, parameters.self, output_attentions)

        attention_output = self.output(self_outputs[0], parameters.output)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

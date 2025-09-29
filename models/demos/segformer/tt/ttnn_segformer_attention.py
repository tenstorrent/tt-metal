# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.segformer.tt.ttnn_segformer_efficient_selfattention import TtSegformerEfficientSelfAttention
from models.demos.segformer.tt.ttnn_segformer_selfoutput import TtSegformerSelfOutput


class TtSegformerAttention:
    def __init__(self, hidden_size, num_attention_heads, parameters, sequence_reduction_ratio):
        super().__init__()
        self.self = TtSegformerEfficientSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            parameters=parameters.self,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.output = TtSegformerSelfOutput()

    def __call__(
        self, device, hidden_states: ttnn.Tensor, height: int, width: int, parameters, output_attentions=False
    ):
        self_outputs = self.self(device, hidden_states, height, width, parameters.self, output_attentions)
        attention_output = self.output(device, self_outputs[0], parameters.output)
        outputs = (attention_output,) + self_outputs[1:]
        ttnn.deallocate(self_outputs[0])

        return outputs

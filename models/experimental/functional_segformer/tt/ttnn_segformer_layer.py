# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_segformer.tt.ttnn_segformer_attention import TtSegformerAttention
from models.experimental.functional_segformer.tt.ttnn_segformer_mix_ffn import TtSegformerMixFFN


class TtSegformerLayer:
    def __init__(self, hidden_size, num_attention_heads, sequence_reduction_ratio, parameters, model):
        super().__init__()
        self.attention = TtSegformerAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
            parameters=parameters.attention,
            model=model.attention,
        )
        self.mlp = TtSegformerMixFFN(parameters=parameters.mlp, model=model.mlp)

    def __call__(
        self, hidden_states: ttnn.Tensor, height: int, width: int, parameters, device, output_attentions=False
    ):
        self_attention_outputs = self.attention(
            ttnn.layer_norm(
                hidden_states,
                weight=parameters.layer_norm_1.weight,
                bias=parameters.layer_norm_1.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            ),
            height,
            width,
            parameters.attention,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        hidden_states = attention_output + hidden_states

        mlp_output = self.mlp(
            ttnn.layer_norm(
                hidden_states,
                weight=parameters.layer_norm_2.weight,
                bias=parameters.layer_norm_2.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            ),
            height,
            width,
            parameters=parameters.mlp,
            device=device,
        )

        layer_output = mlp_output + hidden_states

        outputs = (layer_output,) + outputs

        return outputs

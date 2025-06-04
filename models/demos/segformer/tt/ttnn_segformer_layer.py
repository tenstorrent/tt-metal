# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.segformer.tt.ttnn_segformer_attention import TtSegformerAttention
from models.demos.segformer.tt.ttnn_segformer_mix_ffn import TtSegformerMixFFN


class TtSegformerLayer:
    def __init__(self, hidden_size, num_attention_heads, sequence_reduction_ratio, parameters, mlp_ratio):
        super().__init__()
        self.attention = TtSegformerAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            parameters=parameters["attention"],
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = TtSegformerMixFFN(parameters["mlp"], mlp_hidden_size)

    def __call__(
        self, device, hidden_states: ttnn.Tensor, height: int, width: int, parameters, output_attentions=False
    ):
        self_attention_outputs = self.attention(
            device,
            ttnn.layer_norm(
                hidden_states,
                weight=parameters.layer_norm_1.weight,
                bias=parameters.layer_norm_1.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
            ),
            height,
            width,
            parameters.attention,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        hidden_states = attention_output + hidden_states

        ttnn.deallocate(self_attention_outputs[0])
        ttnn.deallocate(attention_output)

        mlp_output = self.mlp(
            device,
            ttnn.layer_norm(
                hidden_states,
                weight=parameters.layer_norm_2.weight,
                bias=parameters.layer_norm_2.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
            ),
            height,
            width,
            parameters=parameters.mlp,
        )
        layer_output = mlp_output + hidden_states

        ttnn.deallocate(hidden_states)
        ttnn.deallocate(mlp_output)

        outputs = (layer_output,) + outputs

        return outputs

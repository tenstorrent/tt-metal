# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.utility_functions import is_blackhole

if is_blackhole():
    from models.demos.blackhole.bge_large_en.ttnn.common import ff2_program_config, layernorm_program_config
else:
    from models.demos.wormhole.bge_large_en.ttnn.common import ff2_program_config, layernorm_program_config


class TtnnBGEOutput:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config
        self.dense = ttnn.linear
        self.LayerNorm = ttnn.layer_norm

    def __call__(self, hidden_states: ttnn.Tensor, input_tensor: ttnn.Tensor):
        bert_output_lin = self.dense(
            hidden_states,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=ff2_program_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=True,  # FP32 accumulation only on output layer (most critical)
            ),
            dtype=ttnn.bfloat8_b,  # Keep BF8 for storage; compute will use BF16 with FP32 accumulation
        )

        ttnn.deallocate(hidden_states)
        # bert_output_lin = ttnn.reshard(bert_output_lin, input_tensor.memory_config())
        bert_output_lin = ttnn.reallocate(bert_output_lin)

        bert_output_lin = self.LayerNorm(
            bert_output_lin,
            residual_input_tensor=input_tensor,
            weight=self.parameters.LayerNorm.weight,
            bias=self.parameters.LayerNorm.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            epsilon=self.config.layer_norm_eps,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                fp32_dest_acc_en=True,  # FP32 accumulation per engineering guidance (layer norm)
            ),
            program_config=layernorm_program_config,
        )

        ttnn.deallocate(input_tensor)
        return bert_output_lin

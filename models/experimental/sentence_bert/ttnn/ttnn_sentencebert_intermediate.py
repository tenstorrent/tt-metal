# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.sentence_bert.ttnn.common import ff1_matmul_program_config


class TtnnSentenceBertIntermediate:
    def __init__(self, parameters):
        self.dense = ttnn.linear
        self.parameters = parameters

    def __call__(self, hidden_states: ttnn.Tensor):
        out_intermediate = self.dense(
            hidden_states,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=ff1_matmul_program_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                packer_l1_acc=False,
            ),
            dtype=ttnn.bfloat8_b,
        )
        return out_intermediate

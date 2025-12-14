# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.utility_functions import is_blackhole

if is_blackhole():
    from models.demos.blackhole.bge_large_en.ttnn.common import ff1_matmul_program_config
else:
    from models.demos.wormhole.bge_large_en.ttnn.common import ff1_matmul_program_config


class TtnnBGEIntermediate:
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
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                packer_l1_acc=False,
            ),
            dtype=ttnn.bfloat8_b,  # Keep BF8 for storage; compute will use BF16 with FP32 accumulation
        )
        return out_intermediate

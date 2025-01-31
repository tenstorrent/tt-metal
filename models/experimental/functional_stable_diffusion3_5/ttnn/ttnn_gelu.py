# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import ttnn


class ttnn_GELU:
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        self.proj = ttnn.linear
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if True:  # gate.device.type != "mps":  , In torch its executed
            if self.approximate == "tanh":
                approximate_bool = True
            else:
                approximate_bool = False
            return ttnn.gelu(gate, fast_and_approximate_mode=approximate_bool)
        # This is not invoked in our call
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)

    def __call__(self, hidden_states, parameters=None):
        hifi2_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
        )

        mm_a_y = 8
        mm_a_x = 8
        mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
        mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        if hidden_states.shape[-2] < 512:
            mm_a_y = 6
            mm_a_x_strategy = ttnn.ShardStrategy.WIDTH
            mm_a_x_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

        hidden_states_pre_gelu = self.proj(
            hidden_states,
            input_tensor_b=parameters["proj"]["weight"],
            bias=parameters["proj"]["bias"],
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
            compute_kernel_config=hifi2_kernel_config,
        )
        ttnn.deallocate(hidden_states)

        # needed only for 1024 resolution, if it helps!
        hidden_states_pre_gelu = ttnn.reallocate(hidden_states_pre_gelu)

        hidden_states = self.gelu(hidden_states_pre_gelu)  # , memory_config=ttnn.DRAM_MEMORY_CONFIG) #
        ttnn.deallocate(hidden_states_pre_gelu)
        hidden_states = ttnn.reallocate(hidden_states)

        return hidden_states

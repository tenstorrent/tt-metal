# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import ttnn


class ttnn_GELU:
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        self.proj = ttnn.linear
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor, memory_config=ttnn.L1_MEMORY_CONFIG) -> torch.Tensor:
        if True:  # gate.device.type != "mps":  , In torch its executed
            if self.approximate == "tanh":
                approximate_bool = True
            else:
                approximate_bool = False
            return ttnn.gelu(gate, fast_and_approximate_mode=approximate_bool, memory_config=memory_config)
        # This is not invoked in our call
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)

    def __call__(self, hidden_states, parameters=None):
        hifi2_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
        )

        if hidden_states.shape[-2] < 512:
            mm_a_x = 8
            mm_a_y = 6
            mm_a_x_strategy = ttnn.ShardStrategy.WIDTH
            mm_a_x_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        else:
            mm_a_x_memory_config = ttnn.DRAM_MEMORY_CONFIG
            mm_a_y = 8
            mm_a_x = 8
        print("config befor gleu op", hidden_states.memory_config())
        hidden_states_pre_gelu = self.proj(  # expects in dram for 4096,width sharded for 352
            hidden_states,
            input_tensor_b=parameters["proj"]["weight"],
            bias=parameters["proj"]["bias"],
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
            compute_kernel_config=hifi2_kernel_config,
        )
        ttnn.deallocate(hidden_states)
        hidden_states = self.gelu(hidden_states_pre_gelu, memory_config=hidden_states_pre_gelu.memory_config())
        ttnn.deallocate(hidden_states_pre_gelu)
        hidden_states = ttnn.reallocate(hidden_states)
        return hidden_states

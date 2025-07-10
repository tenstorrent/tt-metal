# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params


class TtGEGLU(nn.Module):
    def __init__(self, device, state_dict, module_path, model_config):
        super().__init__()

        self.device = device
        weights = state_dict[f"{module_path}.proj.weight"]
        bias = state_dict[f"{module_path}.proj.bias"]
        w1, w2 = weights.chunk(2, dim=0)  # Each: [out_dim // 2, in_dim]
        b1, b2 = bias.chunk(2, dim=0)  # Each: [out_dim // 2]

        w1 = w1.unsqueeze(0).unsqueeze(0)  # [1, 1, out_dim // 2, in_dim]
        w2 = w2.unsqueeze(0).unsqueeze(0)  # same

        self.tt_weights_1, self.tt_bias_1 = prepare_linear_params(device, w1, b1, model_config.ff_weights_dtype)
        self.tt_weights_2, self.tt_bias_2 = prepare_linear_params(device, w2, b2, model_config.ff_weights_dtype)
        self.program_config = model_config.get_matmul_config(matmul_path=f"{module_path}.proj.split")
        self.program_config_gelu = model_config.get_matmul_config(matmul_path=f"{module_path}.proj.split.gelu")
        self.compute_config = model_config.get_mm_compute_config(f"{module_path}.proj")

        assert self.program_config is not None, "Program config for split weights is None"
        assert self.program_config_gelu is not None, "Program config for split weights with GELU is None"

    def forward(self, input_tensor):
        if input_tensor.shape[2] == 4096:
            # due to block sharded mm constraints, if we block shard the input tensor, we can only run it on 56 cores
            # hence using L1 memory config instead
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG)
        else:
            # here we can run the block sharded matmul on 64 cores
            block_sharded_mem_config = ttnn.create_sharded_memory_config(
                (input_tensor.shape[2] // 8, input_tensor.shape[3] // 8),
                core_grid=ttnn.CoreGrid(y=8, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            input_tensor = ttnn.to_memory_config(input_tensor, block_sharded_mem_config)

        hidden_states = ttnn.linear(
            input_tensor,
            self.tt_weights_1,
            bias=self.tt_bias_1,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=self.program_config,
            compute_kernel_config=self.compute_config,
        )
        gate = ttnn.linear(
            input_tensor,
            self.tt_weights_2,
            bias=self.tt_bias_2,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=self.program_config_gelu,
            compute_kernel_config=self.compute_config,
        )
        ttnn.deallocate(input_tensor)
        hidden_states = ttnn.mul_(hidden_states, gate, use_legacy=False)
        return hidden_states

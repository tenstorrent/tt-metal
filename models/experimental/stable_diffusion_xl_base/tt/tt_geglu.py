# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params


class TtGEGLU(nn.Module):
    def __init__(self, device, state_dict, module_path, model_config, weights_dtype=ttnn.bfloat16):
        super().__init__()

        self.device = device
        weights = state_dict[f"{module_path}.proj.weight"].unsqueeze(0).unsqueeze(0)
        bias = state_dict[f"{module_path}.proj.bias"]

        self.tt_weights, self.tt_bias = prepare_linear_params(device, weights, bias, weights_dtype)
        self.program_config = model_config.get_matmul_config(matmul_path=f"{module_path}.proj")
        self.compute_config = model_config.get_mm_compute_config(f"{module_path}.proj")

    def forward(self, input_tensor):
        output_memory_config = ttnn.create_sharded_memory_config(
            (self.program_config.per_core_M * 32, self.program_config.per_core_N * 32),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            strategy=ttnn.ShardStrategy.WIDTH if input_tensor.shape[-1] == 1280 else ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        hidden_states = ttnn.linear(
            input_tensor,
            self.tt_weights,
            bias=self.tt_bias,
            memory_config=output_memory_config,
            program_config=self.program_config,
            compute_kernel_config=self.compute_config,
        )
        ttnn.deallocate(input_tensor)

        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
        gate = hidden_states[:, :, :, hidden_states.shape[3] // 2 :]
        hidden_states = hidden_states[:, :, :, : hidden_states.shape[3] // 2]
        gate = ttnn.gelu(gate)

        # ttnn.split not working properly
        # hidden_states, gate = ttnn.split(hidden_states, ceil(hidden_states.shape[3] / 2), 3)

        return ttnn.multiply(hidden_states, gate)

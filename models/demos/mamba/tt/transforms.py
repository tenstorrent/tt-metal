# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


class MambaSsmBlockTransformer:
    def __init__(self, device, hidden_size, latent_size, dtype=ttnn.bfloat4_b):
        permute_mask = torch.repeat_interleave(torch.eye(hidden_size), latent_size, dim=0)
        self.reduce_mask = ttnn.from_torch(
            permute_mask,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=dtype,
        )

        repeat_interleave_mask = torch.repeat_interleave(torch.eye(hidden_size), latent_size, dim=1)
        self.repeat_interleave_mask = ttnn.from_torch(
            repeat_interleave_mask,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=dtype,
        )

        repeat_mask = torch.eye(latent_size).repeat(1, hidden_size).unsqueeze(0).unsqueeze(0)
        self.repeat_mask = ttnn.from_torch(
            repeat_mask,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=dtype,
        )

    def repeat_interleave(self, x, memory_config, compute_kernel_config, core_grid):
        return ttnn.linear(
            x,
            self.repeat_interleave_mask,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
            core_grid=core_grid,
        )

    def repeat(self, x, memory_config, compute_kernel_config, core_grid):
        return ttnn.linear(
            x,
            self.repeat_mask,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
            core_grid=core_grid,
        )

    def reduce(self, x, memory_config, compute_kernel_config, core_grid):
        return ttnn.linear(
            x,
            self.reduce_mask,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
            core_grid=core_grid,
        )

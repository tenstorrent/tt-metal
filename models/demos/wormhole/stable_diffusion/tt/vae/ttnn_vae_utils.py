# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


def prepare_group_norm(device, in_channels, core_grid, torch_weights, torch_bias, num_groups=32):
    num_cores_across_channel = core_grid.y
    input_mask = ttnn.create_group_norm_input_mask(in_channels, num_groups, num_cores_across_channel, ttnn.bfloat16)
    input_mask = ttnn.to_device(input_mask, device)
    weights = ttnn.from_torch(
        ttnn.create_group_norm_weight_bias_rm(torch_weights, in_channels, num_cores_across_channel),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bias = ttnn.from_torch(
        ttnn.create_group_norm_weight_bias_rm(torch_bias, in_channels, num_cores_across_channel),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return input_mask, weights, bias

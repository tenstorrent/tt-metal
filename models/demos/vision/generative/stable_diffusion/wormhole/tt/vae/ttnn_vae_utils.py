# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def prepare_group_norm(device, in_channels, core_grid, torch_weights, torch_bias, num_groups=32):
    num_cores_across_channel = core_grid.y
    # Mask is now synthesized in the writer kernel (see #48640); no DRAM tensor needed.
    input_mask = None
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

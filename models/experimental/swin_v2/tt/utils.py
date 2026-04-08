# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


def ttnn_custom_normalize(x, dim, device):
    # Convert input to tiled layout
    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Square the tensor using multiply
    x_squared = ttnn.multiply(x, x)

    # Sum along the specified dimension
    if dim == 1:
        sum_squared = ttnn.sum(x_squared, dim=1, keepdim=True)
    else:
        sum_squared = ttnn.sum(x_squared, dim=-1, keepdim=True)

    # Add small epsilon and calculate square root
    sum_squared = ttnn.add(sum_squared, 1e-12)
    norm = ttnn.sqrt(sum_squared)

    # Create a tensor of ones with the same shape as x
    ones = ttnn.ones_like(
        tensor=x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # Multiply norm by ones to match input shape
    norm_expanded = ttnn.multiply(norm, ones)

    # Divide input by expanded norm
    normalized = ttnn.divide(x, norm_expanded)

    return normalized

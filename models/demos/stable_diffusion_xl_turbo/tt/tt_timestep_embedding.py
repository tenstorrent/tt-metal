# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def timestep_embedding(x, parameters, device):
    x = ttnn.linear(
        x,
        parameters.linear_1.weight,
        bias=parameters.linear_1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation="silu",
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    x = ttnn.linear(
        x,
        parameters.linear_2.weight,
        bias=parameters.linear_2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    return x

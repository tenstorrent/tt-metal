# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def mnist(device, batch_size, x, parameters):
    x = ttnn.reshape(x, (x.shape[0], -1))
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = ttnn.linear(
        x,
        parameters.fc1.weight,
        bias=parameters.fc1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation="relu",
    )
    x = ttnn.linear(
        x,
        parameters.fc2.weight,
        bias=parameters.fc2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation="relu",
    )
    x = ttnn.linear(
        x,
        parameters.fc3.weight,
        bias=parameters.fc3.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation="relu",
    )

    x = ttnn.softmax(x)

    return x

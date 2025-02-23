# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def mnist(mesh_device, batch_size, x, parameters):
    x = ttnn.reshape(x, (x.shape[0], -1))

    x = ttnn.from_device(x)
    x = ttnn.to_device(x, device=mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)

    memory_config = ttnn.create_sharded_memory_config(
        x.shape,
        core_grid=ttnn.CoreGrid(y=mesh_device.core_grid.y, x=mesh_device.core_grid.x),
        strategy=ttnn.ShardStrategy.BLOCK,
        # orientation=ttnn.ShardOrientation.TILE,
    )

    x = ttnn.linear(
        x,
        parameters.fc1.weight,
        bias=parameters.fc1.bias,
        memory_config=memory_config,
        activation="relu",
        core_grid=mesh_device.core_grid,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
        ),
    )

    x = ttnn.linear(
        x,
        parameters.fc2.weight,
        bias=parameters.fc2.bias,
        memory_config=memory_config,
        activation="relu",
        core_grid=mesh_device.core_grid,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
        ),
    )

    x = ttnn.linear(
        x,
        parameters.fc3.weight,
        bias=parameters.fc3.bias,
        memory_config=memory_config,
        activation="relu",
        core_grid=mesh_device.core_grid,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
        ),
    )

    if x.is_sharded():
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

    x = ttnn.softmax(x)

    return x

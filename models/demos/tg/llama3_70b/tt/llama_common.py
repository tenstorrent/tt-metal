# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def tt_all_reduce(input_tensor, mesh_device, cluster_axis, dim=0, num_links=2, memory_config=None):
    # Ensure the input tensor is in the correct memory configuration
    input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    gathered_tensor = ttnn.line_all_gather(
        input_tensor, dim, num_links=num_links, cluster_axis=cluster_axis, mesh_device=mesh_device
    )
    reduced_tensors = ttnn.experimental.fast_reduce_nc(
        gathered_tensor, dims=[dim], output=None, compute_kernel_config=None
    )

    return reduced_tensors


def tt_all_gather(input_tensor, mesh_device, cluster_axis, dim, num_links=2, memory_config=None):
    # Ensure the input tensor is in the correct memory configuration
    input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    return ttnn.line_all_gather(
        input_tensor, dim, num_links=num_links, cluster_axis=cluster_axis, mesh_device=mesh_device
    )

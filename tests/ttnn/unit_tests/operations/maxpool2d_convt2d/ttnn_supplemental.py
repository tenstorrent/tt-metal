# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pure Python implementation of tt-mlir wrappers of TT-NN CCL ops.
"""

import ttnn
from typing import List


def collective_permute(input: ttnn.Tensor, source_target_pairs: List[int]) -> ttnn.Tensor:
    """
    Perform collective permute operation to remap tensor shards across devices.

    Redistributes tensor data by sending shards from source devices to target devices.
    Devices not participating receive zeros.

    Args:
        input: The input device tensor
        source_target_pairs: Flat list of [source_id, target_id, ...] pairs
                            Must have even length

    Returns:
        The permuted tensor with remapped shards
    """
    mesh_device = input.device()
    assert mesh_device is not None, "Tensor must belong to a mesh device"
    assert len(source_target_pairs) % 2 == 0, "Expected source_target_pairs to have size multiple of 2"
    assert input.storage_type() == ttnn.StorageType.DEVICE, "Input of collective_permute must be device storage."

    # Get individual device tensors
    host_tensors = ttnn.get_device_tensors(ttnn.from_device(input))
    num_devices = len(host_tensors)

    # Initialize output tensors
    new_host_tensors = [None] * num_devices
    found_dest_devices = [False] * num_devices

    # Process source-target pairs
    for i in range(0, len(source_target_pairs), 2):
        src = source_target_pairs[i]
        dest = source_target_pairs[i + 1]

        assert 0 <= src < num_devices, f"Source device id {src} is out of bounds!"
        assert 0 <= dest < num_devices, f"Destination device id {dest} is out of bounds!"

        new_host_tensors[dest] = host_tensors[src]
        found_dest_devices[dest] = True

    # Zero out tensors for devices that didn't participate
    for i in range(num_devices):
        if not found_dest_devices[i]:
            # Create zero tensor with same shape as source
            new_host_tensors[i] = ttnn.zeros_like(host_tensors[i])
            found_dest_devices[i] = True

    # Combine all host tensor shards and send back to device
    output = ttnn.from_host_shards(new_host_tensors, mesh_device.shape)
    output = ttnn.to_device(output, mesh_device, input.memory_config())

    return output

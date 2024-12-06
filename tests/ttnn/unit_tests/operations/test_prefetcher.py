# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def run_prefetcher(device):
    sender_cores = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 4)]
    receiver_cores = [
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 0),
                    ttnn.CoreCoord(2, 0),
                ),
            }
        ),
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 4),
                    ttnn.CoreCoord(2, 4),
                ),
            }
        ),
    ]
    sender_receiver_mapping = dict(zip(sender_cores, receiver_cores))

    global_circular_buffer = ttnn.create_global_circular_buffer(device, sender_receiver_mapping, 3200)

    torch_tensor1 = torch.randn(10, 10)
    torch_tensor2 = torch.randn(10, 10)
    tensors = [
        ttnn.from_torch(torch_tensor1, dtype=ttnn.bfloat16, device=device),
        ttnn.from_torch(torch_tensor2, dtype=ttnn.bfloat16, device=device),
    ]

    output_tensor = ttnn.dram_prefetcher(tensors, global_circular_buffer)

    output_tensor = ttnn.to_torch(output_tensor)
    print(output_tensor)
    assert_with_pcc(torch_tensor1, output_tensor, 0.9999)


def test_global_circular_buffer(device):
    run_prefetcher(device)


# def test_global_circular_buffer_mesh(mesh_device):
#     run_prefetcher(mesh_device)

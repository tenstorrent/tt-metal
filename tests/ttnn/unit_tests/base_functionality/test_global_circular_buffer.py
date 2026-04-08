# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def run_global_circular_buffer(device):
    sender_cores = [ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 2)]
    receiver_cores = [
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(4, 4),
                    ttnn.CoreCoord(4, 4),
                ),
            }
        ),
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(2, 3),
                    ttnn.CoreCoord(2, 4),
                ),
            }
        ),
    ]
    sender_receiver_mapping = list(zip(sender_cores, receiver_cores))

    global_circular_buffer = ttnn.create_global_circular_buffer(device, sender_receiver_mapping, 3200)


def test_global_circular_buffer(device):
    run_global_circular_buffer(device)


def test_global_circular_buffer_mesh(mesh_device):
    run_global_circular_buffer(mesh_device)

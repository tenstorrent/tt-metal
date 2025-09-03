# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.utility_functions import torch_random


def test_command_queue(device):
    """Test command_queue context manager and cq_id parameter functionality"""
    torch.manual_seed(0)

    torch_input_tensor = torch_random((1, 32, 32), -1, 1, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    # Save initial command queue ID
    initial_cq_id = ttnn.get_current_command_queue_id_for_thread()

    # Test command_queue context manager
    with ttnn.command_queue(1):
        assert ttnn.get_current_command_queue_id_for_thread() == 1

        # Test operation within context
        result = ttnn.exp(input_tensor)
        assert ttnn.get_current_command_queue_id_for_thread() == 1

        # Test operation with explicit cq_id (should override context)
        result = ttnn.exp(input_tensor, cq_id=0)
        assert ttnn.get_current_command_queue_id_for_thread() == 1  # Should restore to context's cq_id

    # Verify command queue is restored to initial value
    assert ttnn.get_current_command_queue_id_for_thread() == initial_cq_id

    # Test operation with cq_id parameter outside context
    result = ttnn.exp(input_tensor, cq_id=1)
    assert ttnn.get_current_command_queue_id_for_thread() == initial_cq_id

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Stress test to reproduce: Device timeout in gather operation

Original failure: ttnn nightly data_movement tests wormhole_b0 N300 - 2026-01-30
Error: RuntimeError: TT_THROW @ system_memory_manager.cpp:627: TIMEOUT: device timeout, potential hang detected

This test amplifies the gather operation with large tensors that caused
device timeout during ttnn.to_torch() when reading results back from device.

Run with:
    # CRITICAL: Set the timeout environment variable first!
    export TT_METAL_OPERATION_TIMEOUT_SECONDS=5
    export ARCH_NAME=wormhole_b0
    export TT_METAL_HOME=/tt-metal
    source /opt/venv/bin/activate

    # Run with parallel workers (matches CI, higher stress)
    pytest test_gather_timeout_stress.py -n auto -x -v --timeout=300 2>&1 | tee ../logs/run_1.log

    # Or sequential for clean stack traces after reproducing
    pytest test_gather_timeout_stress.py -x -v --timeout=300 2>&1 | tee ../logs/sequential_run.log
"""

import pytest
import torch
import ttnn
import numpy as np


# The exact failing parameters from CI
INPUT_SHAPE = [1, 151936]
INDEX_SHAPE = [1, 151936]
DIM = -1


@pytest.mark.parametrize("iteration", range(50))
def test_gather_long_tensor_stress(iteration, device):
    """
    Stress test for gather with large tensor (the exact failing case from CI).

    The failure occurred during ttnn.to_torch() when reading data back from device.
    This is a device timeout, not a test timeout - the device's internal watchdog
    detected a hang during the read operation.

    Each iteration runs the full gather + to_torch cycle.
    With pytest -n auto, multiple workers hit the device simultaneously,
    increasing the chance of reproducing the race condition or resource contention.
    """
    torch.manual_seed(iteration)  # Different random data each iteration

    torch_dtype = torch.bfloat16
    max_uint32 = np.iinfo(np.uint32).max
    max_idx_val = min(INPUT_SHAPE[DIM], max_uint32)

    # Create input tensors
    input_tensor = torch.randn(INPUT_SHAPE, dtype=torch_dtype)
    index = torch.randint(0, max_idx_val, INDEX_SHAPE, dtype=torch.int64)

    # Compute reference on CPU
    torch_gather = torch.gather(input_tensor, DIM, index)

    # Move to device
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)

    # Execute gather on device
    ttnn_gather = ttnn.gather(ttnn_input, DIM, index=ttnn_index)

    # Read result back - THIS IS WHERE THE TIMEOUT OCCURS
    # The device timeout happens during copy_completion_queue_data_into_user_space
    result = ttnn.to_torch(ttnn_gather)

    # Basic shape validation (if we get here, no timeout occurred)
    assert result.shape == torch.Size(INDEX_SHAPE), f"Shape mismatch: {result.shape} vs {INDEX_SHAPE}"


@pytest.mark.parametrize("iteration", range(20))
def test_gather_back_to_back_stress(iteration, device):
    """
    Back-to-back gather operations without waiting for cleanup.

    This stresses the device command queue and completion queue handling,
    which is where the original timeout occurred.
    """
    torch.manual_seed(42)
    torch_dtype = torch.bfloat16

    # Create fixed input tensors
    input_tensor = torch.randn(INPUT_SHAPE, dtype=torch_dtype)
    max_idx_val = INPUT_SHAPE[DIM]

    # Move input to device once
    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    # Run multiple gather operations back-to-back
    for sub_iter in range(5):
        # New random index each time
        index = torch.randint(0, max_idx_val, INDEX_SHAPE, dtype=torch.int64)
        ttnn_index = ttnn.from_torch(index, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)

        ttnn_gather = ttnn.gather(ttnn_input, DIM, index=ttnn_index)

        # Read back result - potential timeout point
        result = ttnn.to_torch(ttnn_gather)

        assert result.shape == torch.Size(INDEX_SHAPE)


@pytest.mark.parametrize(
    "input_shape,index_shape",
    [
        # The exact failing case
        ([1, 151936], [1, 151936]),
        # Other long tensor cases from the same test
        ([1, 128256], [1, 128256]),
        # Variations that stress similar code paths
        ([32, 256 * 32], [32, 64 * 32]),  # 256 * TILE_HEIGHT
        ([1, 1, 32, 256 * 32], [1, 1, 32, 128 * 32]),
    ],
)
@pytest.mark.parametrize("iteration", range(10))
def test_gather_multiple_shapes_stress(input_shape, index_shape, iteration, device):
    """
    Test multiple shape configurations that exercise the same gather code path.
    """
    torch.manual_seed(iteration)
    torch_dtype = torch.bfloat16

    dim = -1
    max_idx_val = min(input_shape[dim], np.iinfo(np.uint32).max)

    input_tensor = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(0, max_idx_val, index_shape, dtype=torch.int64)

    ttnn_input = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)
    result = ttnn.to_torch(ttnn_gather)

    assert result.shape == torch.Size(index_shape)


if __name__ == "__main__":
    import sys

    # Allow running directly with custom iteration count
    pytest.main([__file__, "-x", "-v", "--timeout=300"] + sys.argv[1:])

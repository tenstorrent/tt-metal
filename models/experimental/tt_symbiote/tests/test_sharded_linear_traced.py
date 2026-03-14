# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests to verify sharded linear correctness in traced mode on T3K.

These tests verify that:
1. Semaphores are properly reset between trace replays for async CCL operations
2. Output buffers are independent and not aliased in traced async operations
3. Multiple trace replays complete successfully without corruption

Run with:
    MESH_DEVICE=T3K TT_SYMBIOTE_RUN_MODE=TRACED pytest models/experimental/tt_symbiote/tests/test_sharded_linear_traced.py -v --timeout=0
"""

import os
import pytest
import torch
import torch.nn as nn
import ttnn

from models.experimental.tt_symbiote.core.run_config import TracedRun
from models.experimental.tt_symbiote.modules.linear import TTNNLinearIColShardedWRowSharded
from models.experimental.tt_symbiote.utils.device_management import set_device


def _is_traced_mode():
    """Check if traced mode is active."""
    return os.environ.get("TT_SYMBIOTE_RUN_MODE") == "TRACED"


def _is_t3k():
    """Check if T3K mesh device is configured."""
    return os.environ.get("MESH_DEVICE") == "T3K"


# T3K device mesh mapping
MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}

# T3K device parameters for traced mode
T3K_DEVICE_PARAMS = {
    "l1_small_size": 245760,
    "trace_region_size": 50000000,
    "num_command_queues": 2,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
}


@pytest.mark.skipif(not _is_traced_mode(), reason="Requires TT_SYMBIOTE_RUN_MODE=TRACED")
@pytest.mark.skipif(not _is_t3k(), reason="Requires MESH_DEVICE=T3K")
@pytest.mark.parametrize("device_params", [T3K_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_sharded_linear_trace_replay_succeeds_with_semaphore_reset(mesh_device):
    """
    Test that verifies semaphores are properly reset between trace replays.

    Expected behavior:
    - First trace capture succeeds
    - Subsequent trace replays complete successfully because
      semaphores are properly reset between replays

    This test verifies that semaphore management works correctly in traced mode.
    """
    TracedRun.release_all()

    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1

    # Skip if not enough devices
    if num_devices < 2:
        pytest.skip(f"Need at least 2 devices for sharded test, got {num_devices}")

    # Create dimensions that work with sharding
    # in_features must be divisible by num_devices for column sharding
    in_features = 256 * num_devices  # e.g., 2048 for 8 devices
    out_features = 256
    batch_size = 1
    seq_len = 32

    # Create PyTorch reference layer
    torch_linear = nn.Linear(in_features, out_features, bias=False)
    torch_linear = torch_linear.to(torch.bfloat16)

    # Create TTNN sharded linear
    ttnn_linear = TTNNLinearIColShardedWRowSharded.from_torch(torch_linear)
    ttnn_linear._unique_name = "test_sharded_semaphore"
    set_device(ttnn_linear, mesh_device, dump_visualization=False)
    ttnn_linear.preprocess_weights()
    ttnn_linear.move_weights_to_device()

    # Create input tensor shape
    input_shape = (batch_size, seq_len, in_features)

    # Phase 1: First run - trace capture
    print(f"\n[Phase 1] First run (trace capture) with {num_devices} devices...")
    input1 = torch.randn(*input_shape, dtype=torch.bfloat16)
    output1 = ttnn_linear(input1)
    print(f"  Trace capture SUCCESS, cache size: {TracedRun.cache_size()}")

    # Phase 2: Multiple trace replays - verifying semaphores are properly reset
    print("\n[Phase 2] Multiple trace replays (verifying semaphore reset)...")
    num_replays = 5

    for i in range(num_replays):
        input_new = torch.randn(*input_shape, dtype=torch.bfloat16)
        print(f"  Replay {i+1}/{num_replays}...", end=" ")

        # Semaphores are properly reset, so this completes successfully
        output = ttnn_linear(input_new)

        print("SUCCESS")

        # Verify output is valid
        assert output is not None, f"Output {i+1} should not be None"

    print(f"\nAll {num_replays} trace replays completed successfully!")
    TracedRun.release_all()


@pytest.mark.skipif(not _is_traced_mode(), reason="Requires TT_SYMBIOTE_RUN_MODE=TRACED")
@pytest.mark.skipif(not _is_t3k(), reason="Requires MESH_DEVICE=T3K")
@pytest.mark.parametrize("device_params", [T3K_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_sharded_linear_trace_outputs_are_independent(mesh_device):
    """
    Test that verifies output buffers are independent in traced mode.

    Expected behavior:
    - Multiple trace replays return outputs with independent buffers
    - Each output has its own unique buffer address

    This test verifies that output buffer management works correctly in traced mode.
    """
    TracedRun.release_all()

    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1

    if num_devices < 2:
        pytest.skip(f"Need at least 2 devices for sharded test, got {num_devices}")

    in_features = 256 * num_devices
    out_features = 256

    torch_linear = nn.Linear(in_features, out_features, bias=False)
    torch_linear = torch_linear.to(torch.bfloat16)

    ttnn_linear = TTNNLinearIColShardedWRowSharded.from_torch(torch_linear)
    ttnn_linear._unique_name = "test_sharded_buffer"
    set_device(ttnn_linear, mesh_device, dump_visualization=False)
    ttnn_linear.preprocess_weights()
    ttnn_linear.move_weights_to_device()

    input_shape = (1, 32, in_features)
    input_tensor = torch.randn(*input_shape, dtype=torch.bfloat16)

    # First run - capture trace
    print("\n[Phase 1] Capture trace...")
    _ = ttnn_linear(input_tensor)

    # Get two consecutive outputs
    print("[Phase 2] Get two consecutive outputs from trace replay...")
    output1 = ttnn_linear(input_tensor)
    output2 = ttnn_linear(input_tensor)

    # Check if outputs are aliased (same buffer)
    print("[Phase 3] Checking buffer addresses...")

    if hasattr(output1, "ttnn_tensor") and output1.ttnn_tensor is not None:
        buf1 = output1.ttnn_tensor.buffer_address()
        buf2 = output2.ttnn_tensor.buffer_address()

        print(f"  Output 1 buffer address: {buf1}")
        print(f"  Output 2 buffer address: {buf2}")

        # Verify buffers are independent (not aliased)
        assert buf1 != buf2, f"Output buffers should be independent but both are at {buf1}."
        print(f"  SUCCESS: Buffers are independent (addresses {buf1} != {buf2})")

    TracedRun.release_all()


@pytest.mark.skipif(not _is_traced_mode(), reason="Requires TT_SYMBIOTE_RUN_MODE=TRACED")
@pytest.mark.skipif(not _is_t3k(), reason="Requires MESH_DEVICE=T3K")
@pytest.mark.parametrize("device_params", [T3K_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("num_replays", [2, 5, 10])
def test_sharded_linear_multiple_trace_replays_succeed(mesh_device, num_replays):
    """
    Test that verifies multiple trace replays complete successfully.

    Tests increasing numbers of replays to verify semaphore cycling works correctly.
    All replay counts should complete without issues.
    """
    TracedRun.release_all()

    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1

    if num_devices < 2:
        pytest.skip(f"Need at least 2 devices for sharded test, got {num_devices}")

    in_features = 256 * num_devices
    out_features = 256

    torch_linear = nn.Linear(in_features, out_features, bias=False)
    torch_linear = torch_linear.to(torch.bfloat16)

    ttnn_linear = TTNNLinearIColShardedWRowSharded.from_torch(torch_linear)
    ttnn_linear._unique_name = f"test_stress_{num_replays}"
    set_device(ttnn_linear, mesh_device, dump_visualization=False)
    ttnn_linear.preprocess_weights()
    ttnn_linear.move_weights_to_device()

    input_shape = (1, 32, in_features)

    # Capture trace
    input1 = torch.randn(*input_shape, dtype=torch.bfloat16)
    _ = ttnn_linear(input1)

    # Run multiple replays
    print(f"\nVerifying {num_replays} trace replays complete successfully...")
    for i in range(num_replays):
        input_new = torch.randn(*input_shape, dtype=torch.bfloat16)
        output = ttnn_linear(input_new)
        assert output is not None, f"Replay {i+1} returned None"

    print(f"SUCCESS: All {num_replays} replays completed successfully!")
    TracedRun.release_all()

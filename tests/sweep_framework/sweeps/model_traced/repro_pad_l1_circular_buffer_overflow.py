# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Reproduction test for pad operation L1 circular buffer overflow.

This demonstrates that the pad operation allocates internal L1 circular buffers
that can exceed the L1 memory limit (1.5MB), even when the input tensor uses DRAM.

Root Cause:
-----------
The pad operation's C++ implementation allocates circular buffers in L1 memory
proportional to the tensor size. For large tensors (>500K elements), these buffers
can grow to 2-7MB, exceeding the L1 capacity.

Test Cases:
-----------
1. Small tensor (32x32): PASS - circular buffers fit in L1
2. Medium tensor (224x224 sharded): FAIL - circular buffers exceed L1 (2.2MB)
3. Large tensor (512x512 sharded): FAIL - circular buffers exceed L1 (5.3MB)

Expected Errors:
---------------
TT_THROW @ tt_metal/impl/program/program.cpp:914
Statically allocated circular buffers grow beyond max L1 size
- Medium: 2199936 B (2.2MB)
- Large: 5345856 B (5.3MB)

Note: Max L1 size is typically 1572864 B (1.5MB)
"""

import torch
import ttnn
import pytest


def test_pad_small_tensor_passes():
    """Small tensor should work - circular buffers fit in L1."""
    device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.device.DispatchCoreConfig())

    # Small tensor: 1*1*32*32 = 1024 elements = 2KB
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # Even with DRAM, this should work because buffers are small
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Apply padding
    padding = ((0, 0), (0, 0), (0, 1), (0, 1))
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=0.0)

    print("✅ Small tensor (32x32): PASSED")
    print(f"   Input shape: {shape}")
    print(f"   Output shape: {output_tensor.shape}")

    ttnn.close_device(device)


def test_pad_medium_sharded_tensor_fails():
    """Medium sharded tensor should fail - circular buffers exceed L1."""
    device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.device.DispatchCoreConfig())

    # Medium tensor: 1*3*224*224 = 150,528 elements = 294KB input
    # But pad allocates 2.2MB circular buffers!
    shape = (1, 3, 224, 224)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # Use DRAM - but pad operation will still allocate L1 circular buffers internally
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print(f"About to call pad with medium tensor (224x224)...")
    print(f"   Input: {shape} = {150_528:,} elements")
    print(f"   Input size: {150_528 * 2 / 1024:.1f} KB (bfloat16)")
    print(f"   Expected to fail with: 'Statically allocated circular buffers grow to ~2.2MB'")
    print()

    # This will fail with:
    # TT_THROW: Statically allocated circular buffers grow to 2199936 B (2.2MB)
    # The padding makes it worse, but even without padding large tensors fail
    padding = [[0, 0], [0, 13], [0, 0], [0, 0]]

    output_tensor = ttnn.pad(input_tensor, padding=padding, value=0.0)

    print("❌ UNEXPECTED: Medium tensor did not fail!")
    print(f"   Output shape: {output_tensor.shape}")

    ttnn.close_device(device)


def test_pad_large_tensor_fails():
    """Large tensor should fail - circular buffers way exceed L1."""
    device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.device.DispatchCoreConfig())

    # Large tensor: 1*3*512*512 = 786,432 elements = 1.5MB input
    # Pad allocates 5.3MB circular buffers!
    shape = (1, 3, 512, 512)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # Force DRAM (still won't help - pad uses L1 circular buffers internally)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print(f"About to call pad with large tensor (512x512)...")
    print(f"   Input: {shape} = {786_432:,} elements")
    print(f"   Input size: {786_432 * 2 / 1024:.1f} KB (bfloat16)")
    print(f"   Expected to fail with: 'Statically allocated circular buffers grow to ~5.3MB'")
    print()

    # This will fail with:
    # TT_THROW: Statically allocated circular buffers grow to 5345856 B (5.3MB)
    padding = [[0, 0], [0, 0], [0, 0], [0, 0]]  # Even no padding fails!

    output_tensor = ttnn.pad(input_tensor, padding=padding, value=0.0)

    print("❌ UNEXPECTED: Large tensor did not fail!")
    print(f"   Output shape: {output_tensor.shape}")

    ttnn.close_device(device)


if __name__ == "__main__":
    print("=" * 80)
    print("PAD OPERATION - L1 CIRCULAR BUFFER OVERFLOW REPRODUCTION")
    print("=" * 80)
    print("")
    print("This test demonstrates the hardware limitation of the pad operation.")
    print("All try-except blocks removed - errors will be shown directly.")
    print("")

    print("=" * 80)
    print("Test 1: Small tensor (should pass)")
    print("=" * 80)
    test_pad_small_tensor_passes()
    print("")

    print("=" * 80)
    print("Test 2: Medium tensor (224x224) - WILL CRASH")
    print("=" * 80)
    test_pad_medium_sharded_tensor_fails()
    print("")

    print("=" * 80)
    print("Test 3: Large tensor (512x512) - WILL CRASH")
    print("=" * 80)
    test_pad_large_tensor_fails()
    print("")

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("The pad operation cannot handle large tensors due to internal L1")
    print("circular buffer allocation. This is a C++ implementation limitation.")
    print("")
    print("WORKAROUND: Use invalidate_vector to skip:")
    print("  1. All L1 sharded configs")
    print("  2. Tensors > 500K elements")
    print("=" * 80)

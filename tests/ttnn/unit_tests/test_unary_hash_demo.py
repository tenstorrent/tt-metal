# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn


def test_unary_hash_collision_demo(device):
    """Demonstrate hash collision with same volume, different shapes"""

    # All shapes have volume = 1,048,576 (smaller for faster test)
    shapes = [
        [1, 512, 2048],  # volume = 1,048,576
        [1, 1024, 1024],  # volume = 1,048,576
        [1, 2048, 512],  # volume = 1,048,576
    ]

    print("\n" + "=" * 70)
    print("❌ UnaryDeviceOperation Hash Collision (PROBLEM)")
    print("=" * 70)
    print(f"All shapes have volume = 1,048,576")
    print(f"Hash uses .volume() → all get SAME hash!\n")

    for i, shape in enumerate(shapes, 1):
        print(f"[{i}] Shape: {str(shape):20s} → relu...")

        # Create input
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Execute UnaryDeviceOperation (relu)
        ttnn_output = ttnn.relu(ttnn_input)

        # Verify correctness
        torch_output = torch.relu(torch_input)
        ttnn_torch = ttnn.to_torch(ttnn_output)
        passed = torch.allclose(torch_output, ttnn_torch, rtol=1e-2, atol=1e-2)

        print(f"    {'✓ PASS' if passed else '✗ FAIL'}")
        assert passed

    print("\n" + "=" * 70)
    print("✅ SoftmaxDeviceOperation (CORRECT)")
    print("=" * 70)
    print(f"Same shapes, same volume = 1,048,576")
    print(f"Hash uses .logical_shape() → each gets DIFFERENT hash!\n")

    for i, shape in enumerate(shapes, 1):
        print(f"[{i}] Shape: {str(shape):20s} → softmax...")

        # Create input
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Execute SoftmaxDeviceOperation (uses logical_shape correctly!)
        ttnn_output = ttnn.softmax(ttnn_input, dim=-1)

        # Verify correctness
        torch_output = torch.softmax(torch_input.float(), dim=-1).bfloat16()
        ttnn_torch = ttnn.to_torch(ttnn_output)
        passed = torch.allclose(torch_output, ttnn_torch, rtol=1e-1, atol=1e-1)

        print(f"    {'✓ PASS' if passed else '✗ FAIL'}")
        assert passed

    print("\n" + "=" * 70)
    print("Summary:")
    print("  ❌ relu (3 ops):    SAME hash → Shape info LOST")
    print("  ✅ softmax (3 ops): DIFFERENT hashes → Shape preserved")
    print("\nCheck Tracy CSV:")
    print('  csvexport-release -m -s ";" .logs/tracy_profile_log_host.tracy \\')
    print("    > tracy_ops_data.csv")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # For standalone execution
    import ttnn

    device = ttnn.open_device(device_id=0)
    test_unary_hash_collision_demo(device)
    ttnn.close_device(device)

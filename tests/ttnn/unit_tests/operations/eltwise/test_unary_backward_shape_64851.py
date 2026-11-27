# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test for issue #64851: rdiv_bw and rsqrt_bw use wrong tensor for output shape.

Bug 1: rdiv_bw line 305 used zeros_like(grad) instead of zeros_like(input)
       when rounding_mode is 'trunc' or 'floor'.
Bug 2: rsqrt_bw line 607 used empty_like(grad) instead of empty_like(input)
       when input_grad is not provided.

Both bugs cause incorrect output shape when grad and input have different
shapes due to broadcasting.
"""

import pytest
import torch
import ttnn


@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
def test_rdiv_bw_shape_with_broadcast(rounding_mode):
    """Test that rdiv_bw returns gradient with correct shape when broadcasting."""
    dev = ttnn.open_device(device_id=0)
    try:
        # Create tensors where grad and input have different shapes
        # input: [2, 3, 4], grad: [1, 3, 1] (broadcasts to [2, 3, 4])
        input_torch = torch.randn(2, 3, 4, dtype=torch.float32)
        grad_torch = torch.randn(1, 3, 1, dtype=torch.float32)
        scalar = 2.0

        input_ttnn = ttnn.from_torch(
            input_torch.reshape(1, 1, 2, 3, 4),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )
        grad_ttnn = ttnn.from_torch(
            grad_torch.reshape(1, 1, 1, 3, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )

        # Compute backward
        result = ttnn.rdiv_bw(
            grad_ttnn,
            input_ttnn,
            scalar=scalar,
            rounding_mode=rounding_mode,
        )

        # Result should have same shape as input (after broadcasting)
        result_torch = ttnn.to_torch(result[0])
        assert result_torch.shape[-3:] == input_torch.shape, (
            f"rdiv_bw shape mismatch: got {result_torch.shape}, "
            f"expected input shape {input_torch.shape}"
        )

        ttnn.deallocate(input_ttnn)
        ttnn.deallocate(grad_ttnn)
    finally:
        ttnn.close_device(dev)


def test_rsqrt_bw_shape_with_broadcast():
    """Test that rsqrt_bw returns gradient with correct shape when broadcasting."""
    dev = ttnn.open_device(device_id=0)
    try:
        # Create tensors where grad and input have different shapes
        input_torch = torch.randn(2, 3, 4, dtype=torch.float32).abs() + 0.1
        grad_torch = torch.randn(1, 3, 1, dtype=torch.float32)

        input_ttnn = ttnn.from_torch(
            input_torch.reshape(1, 1, 2, 3, 4),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )
        grad_ttnn = ttnn.from_torch(
            grad_torch.reshape(1, 1, 1, 3, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )

        # Compute backward without providing input_grad
        result = ttnn.rsqrt_bw(
            grad_ttnn,
            input_ttnn,
        )

        # Result should have same shape as input
        result_torch = ttnn.to_torch(result[0])
        assert result_torch.shape[-3:] == input_torch.shape, (
            f"rsqrt_bw shape mismatch: got {result_torch.shape}, "
            f"expected input shape {input_torch.shape}"
        )

        ttnn.deallocate(input_ttnn)
        ttnn.deallocate(grad_ttnn)
    finally:
        ttnn.close_device(dev)


def test_rdiv_bw_correctness_trunc():
    """Test that rdiv_bw with trunc returns correct values (zeros for non-differentiable)."""
    dev = ttnn.open_device(device_id=0)
    try:
        input_torch = torch.tensor([1.5, 2.7, -3.2], dtype=torch.float32)
        grad_torch = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

        input_ttnn = ttnn.from_torch(
            input_torch.reshape(1, 1, 1, 3),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )
        grad_ttnn = ttnn.from_torch(
            grad_torch.reshape(1, 1, 1, 3),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )

        result = ttnn.rdiv_bw(
            grad_ttnn,
            input_ttnn,
            scalar=1.0,
            rounding_mode="trunc",
        )

        result_torch = ttnn.to_torch(result[0]).flatten()[:3]

        # For trunc/floor rounding, gradient should be zero
        assert torch.allclose(result_torch, torch.zeros_like(result_torch), atol=1e-6), (
            f"rdiv_bw with trunc should return zeros, got {result_torch}"
        )

        ttnn.deallocate(input_ttnn)
        ttnn.deallocate(grad_ttnn)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    print("Testing rdiv_bw and rsqrt_bw shape fixes (#64851)...")

    for mode in ["trunc", "floor"]:
        print(f"\nTesting rdiv_bw shape with broadcast (mode={mode})...")
        try:
            test_rdiv_bw_shape_with_broadcast(mode)
            print("  ✓ Passed")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\nTesting rsqrt_bw shape with broadcast...")
    try:
        test_rsqrt_bw_shape_with_broadcast()
        print("  ✓ Passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print("\nTesting rdiv_bw correctness with trunc...")
    try:
        test_rdiv_bw_correctness_trunc()
        print("  ✓ Passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

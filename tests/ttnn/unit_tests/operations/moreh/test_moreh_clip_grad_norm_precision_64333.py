# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test for issue #64333: moreh_clip_grad_norm bf16 precision fix.

The bug: Step 1 accumulator used input dtype (bf16), causing precision loss
when summing many tiles. Step 2 sqrt computation also in bf16. Result: norm
underestimated by 1.3-7.6% for bf16 inputs.

The fix: Use FLOAT32 for tmp_pow_sum accumulator and output_total_norm_fp32,
then typecast back to input dtype for step3 compatibility.
"""

import pytest
import torch
import ttnn


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shape", [(32, 64), (128, 256), (512, 512)])
def test_moreh_clip_grad_norm_precision(dtype, shape):
    """Test that clip_grad_norm returns accurate norms for both bf16 and fp32."""
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(42)

        # Create random gradient tensor
        grad_torch = torch.randn(shape, dtype=torch.float32)
        if dtype == ttnn.bfloat16:
            grad_torch = grad_torch.to(torch.bfloat16)

        # Compute reference norm using PyTorch
        reference_norm = torch.norm(grad_torch.float(), p=2.0).item()

        # Convert to ttnn tensor
        grad_ttnn = ttnn.from_torch(
            grad_torch.reshape(1, 1, *shape),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )

        # Compute norm using moreh_clip_grad_norm
        max_norm = 1.0
        result_norm = ttnn.moreh_clip_grad_norm(
            [grad_ttnn],
            max_norm=max_norm,
            norm_type=2.0,
            error_if_nonfinite=False,
        )

        # Convert result back to torch
        result_norm_torch = ttnn.to_torch(result_norm).item()

        # Check accuracy
        # Before fix: bf16 had 1.3-7.6% error
        # After fix: should be < 0.5% error for both dtypes
        relative_error = abs(result_norm_torch - reference_norm) / reference_norm

        assert relative_error < 0.005, (
            f"shape={shape}, dtype={dtype}: relative error {relative_error:.4%} "
            f"(result={result_norm_torch:.6f}, reference={reference_norm:.6f}). "
            f"Expected < 0.5% error."
        )

        ttnn.deallocate(grad_ttnn)
    finally:
        ttnn.close_device(dev)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_moreh_clip_grad_norm_multiple_tensors(dtype):
    """Test clip_grad_norm with multiple gradient tensors (common in training)."""
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(123)

        # Create multiple gradient tensors (simulating a small model)
        shapes = [(64, 64), (128, 128), (32, 256)]
        grads_torch = []
        grads_ttnn = []

        for shape in shapes:
            grad = torch.randn(shape, dtype=torch.float32)
            if dtype == ttnn.bfloat16:
                grad = grad.to(torch.bfloat16)
            grads_torch.append(grad)
            grads_ttnn.append(
                ttnn.from_torch(
                    grad.reshape(1, 1, *shape),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=dev,
                )
            )

        # Compute reference total norm
        reference_norm = torch.sqrt(
            sum(torch.norm(g.float(), p=2.0) ** 2 for g in grads_torch)
        ).item()

        # Compute norm using moreh_clip_grad_norm
        max_norm = 1.0
        result_norm = ttnn.moreh_clip_grad_norm(
            grads_ttnn,
            max_norm=max_norm,
            norm_type=2.0,
            error_if_nonfinite=False,
        )

        result_norm_torch = ttnn.to_torch(result_norm).item()

        # Check accuracy
        relative_error = abs(result_norm_torch - reference_norm) / reference_norm

        assert relative_error < 0.005, (
            f"dtype={dtype}: relative error {relative_error:.4%} "
            f"(result={result_norm_torch:.6f}, reference={reference_norm:.6f}). "
            f"Expected < 0.5% error."
        )

        for g in grads_ttnn:
            ttnn.deallocate(g)
    finally:
        ttnn.close_device(dev)


def test_moreh_clip_grad_norm_clipping():
    """Test that clipping actually works correctly after precision fix."""
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(456)

        # Create a gradient with known norm
        shape = (128, 128)
        grad_torch = torch.randn(shape, dtype=torch.bfloat16)
        grad_norm = torch.norm(grad_torch.float(), p=2.0).item()

        grad_ttnn = ttnn.from_torch(
            grad_torch.reshape(1, 1, *shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )

        # Clip to half the norm
        max_norm = grad_norm / 2.0
        result_norm = ttnn.moreh_clip_grad_norm(
            [grad_ttnn],
            max_norm=max_norm,
            norm_type=2.0,
            error_if_nonfinite=False,
        )

        result_norm_torch = ttnn.to_torch(result_norm).item()

        # After clipping, the norm should be close to max_norm
        # Allow some tolerance for bf16 precision
        assert abs(result_norm_torch - max_norm) / max_norm < 0.01, (
            f"Clipping failed: result_norm={result_norm_torch:.6f}, "
            f"max_norm={max_norm:.6f}. Expected result ≈ max_norm."
        )

        # Check that the gradient was actually scaled
        grad_after = ttnn.to_torch(grad_ttnn).reshape(shape)
        expected_scale = max_norm / grad_norm
        grad_expected = grad_torch.float() * expected_scale

        # Allow 1% tolerance for bf16 rounding
        assert torch.allclose(
            grad_after.float(), grad_expected.to(torch.bfloat16).float(), rtol=0.01, atol=0.01
        ), "Gradient was not scaled correctly"

        ttnn.deallocate(grad_ttnn)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Testing moreh_clip_grad_norm precision fix...")

    for dtype in [ttnn.bfloat16, ttnn.float32]:
        for shape in [(32, 64), (128, 256), (512, 512)]:
            print(f"\nTesting shape={shape}, dtype={dtype}...")
            try:
                test_moreh_clip_grad_norm_precision(dtype, shape)
                print(f"  ✓ Passed")
            except AssertionError as e:
                print(f"  ✗ Failed: {e}")
            except Exception as e:
                print(f"  ✗ Error: {e}")

    print("\nTesting multiple tensors...")
    for dtype in [ttnn.bfloat16, ttnn.float32]:
        print(f"\nTesting dtype={dtype}...")
        try:
            test_moreh_clip_grad_norm_multiple_tensors(dtype)
            print(f"  ✓ Passed")
        except AssertionError as e:
            print(f"  ✗ Failed: {e}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\nTesting clipping...")
    try:
        test_moreh_clip_grad_norm_clipping()
        print("  ✓ Passed")
    except AssertionError as e:
        print(f"  ✗ Failed: {e}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test for issue #63940: ttnn.permute uint32 overflow at 2^32 padded elements.

The bug: permute_tiled_program_factory.cpp computed padded_xw_volume in uint32_t,
which wraps to 0 at exactly 2^32 elements, causing the permute to return all zeros
(or garbage) without any error.

The fix: use uint64_t for volume calculations and add TT_FATAL if the result exceeds
supported limits.
"""

import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "n",
    [2047, 2048, 2049],  # Below, at, and above the 2^32 boundary
)
def test_permute_large_tensor_no_overflow(n):
    """Test that permute works correctly for tensors near the 2^32 element boundary."""
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        # Input shape (n, 5, n, 1), permute to (n, n, 1, 5)
        # Tile-padded output volume: n * n * 32 * 32
        # At n=2048: 2048 * 2048 * 32 * 32 = 2^32 exactly (the overflow point)
        x = torch.randn(n, 5, n, 1)
        t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        y = ttnn.to_torch(ttnn.permute(t, [2, 0, 3, 1]))
        ref = x.permute(2, 0, 3, 1)

        # Check shapes match
        assert y.shape == ref.shape, f"Shape mismatch: {y.shape} vs {ref.shape}"

        # Check values match (allowing for floating point tolerance)
        mismatched = (y != ref).sum().item()
        total = ref.numel()
        mismatch_ratio = mismatched / total if total > 0 else 0

        # The bug caused 100% mismatch at n=2048. After the fix, should be 0 or near-0.
        assert mismatch_ratio < 0.01, (
            f"n={n}: {mismatched}/{total} elements mismatched ({mismatch_ratio:.2%}). "
            f"This indicates the uint32 overflow bug is present."
        )

        # Check for non-finite values (another symptom of the bug)
        non_finite = (~torch.isfinite(y)).sum().item()
        assert non_finite == 0, f"n={n}: {non_finite} non-finite values in output"

        ttnn.deallocate(t)
    finally:
        ttnn.close_device(dev)


def test_permute_volume_calculation_overflow():
    """
    Direct test of the overflow condition: padded volume = n^2 * 32 * 32.
    At n=2048, this is exactly 2^32, which wrapped to 0 in uint32_t.
    """
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(42)
        n = 2048  # The exact overflow point
        x = torch.randn(n, 5, n, 1)

        # This should NOT silently fail or return garbage
        t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        y = ttnn.to_torch(ttnn.permute(t, [2, 0, 3, 1]))
        ref = x.permute(2, 0, 3, 1)

        # Verify the output is valid
        assert y.shape == ref.shape
        assert torch.isfinite(y).all(), "Output contains non-finite values"

        # The bug would cause y to be all zeros or garbage
        # After fix, should match reference closely
        assert torch.allclose(y, ref, rtol=1e-5, atol=1e-5), (
            "Output does not match reference — uint32 overflow may still be present"
        )

        ttnn.deallocate(t)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Testing permute overflow fix...")
    for n in [2047, 2048, 2049]:
        print(f"\nTesting n={n}...")
        try:
            test_permute_large_tensor_no_overflow(n)
            print(f"  ✓ n={n} passed")
        except AssertionError as e:
            print(f"  ✗ n={n} failed: {e}")
        except Exception as e:
            print(f"  ✗ n={n} error: {e}")

    print("\nTesting exact overflow point...")
    try:
        test_permute_volume_calculation_overflow()
        print("  ✓ Overflow test passed")
    except AssertionError as e:
        print(f"  ✗ Overflow test failed: {e}")
    except Exception as e:
        print(f"  ✗ Overflow test error: {e}")

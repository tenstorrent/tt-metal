# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test for issue #64187: rms_norm_pre_all_gather 2D core grid validation.

Bug: LayerNormPreAllGather2DProgramFactory only supports RMSNorm, but validation
didn't reject LAYERNORM + 2D grid combination. This would cause silent incorrect
results or crashes.

Fix: Add TT_FATAL to reject LAYERNORM when use_2d_core_grid=True.
"""

import pytest
import torch
import ttnn


def test_layernorm_pre_all_gather_2d_rejects_layernorm():
    """Test that layer_norm_pre_all_gather with 2D grid rejects LAYERNORM."""
    dev = ttnn.open_device(device_id=0)
    try:
        # Create input tensor
        x_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
        x_ttnn = ttnn.from_torch(x_torch, device=dev)

        # This should raise an error because LAYERNORM + 2D grid is not supported
        with pytest.raises(RuntimeError, match="LayerNorm does not support 2D core grid"):
            ttnn.layer_norm_pre_all_gather(
                x_ttnn,
                epsilon=1e-5,
                weight=None,
                bias=None,
                use_2d_core_grid=True,
                program_config=ttnn.LayerNormDefaultProgramConfig(use_welford=False),
            )

        ttnn.deallocate(x_ttnn)
    finally:
        ttnn.close_device(dev)


def test_rmsnorm_pre_all_gather_2d_accepts_rmsnorm():
    """Test that rms_norm_pre_all_gather with 2D grid accepts RMSNORM."""
    dev = ttnn.open_device(device_id=0)
    try:
        # Create input tensor
        x_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
        x_ttnn = ttnn.from_torch(x_torch, device=dev)

        # This should work because RMSNORM + 2D grid is supported
        result = ttnn.rms_norm_pre_all_gather(
            x_ttnn,
            epsilon=1e-5,
            weight=None,
            use_2d_core_grid=True,
        )

        # Verify output shape
        result_torch = ttnn.to_torch(result)
        assert result_torch.shape == x_torch.shape

        ttnn.deallocate(x_ttnn)
        ttnn.deallocate(result)
    finally:
        ttnn.close_device(dev)


def test_layernorm_pre_all_gather_without_2d_accepts_layernorm():
    """Test that layer_norm_pre_all_gather without 2D grid accepts LAYERNORM."""
    dev = ttnn.open_device(device_id=0)
    try:
        # Create input tensor
        x_torch = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
        x_ttnn = ttnn.from_torch(x_torch, device=dev)

        # This should work because LAYERNORM without 2D grid is supported
        result = ttnn.layer_norm_pre_all_gather(
            x_ttnn,
            epsilon=1e-5,
            weight=None,
            bias=None,
            use_2d_core_grid=False,  # or omit this parameter
            program_config=ttnn.LayerNormDefaultProgramConfig(use_welford=False),
        )

        # Verify output shape
        result_torch = ttnn.to_torch(result)
        assert result_torch.shape == x_torch.shape

        ttnn.deallocate(x_ttnn)
        ttnn.deallocate(result)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    print("Testing layer_norm_pre_all_gather 2D grid validation (#64187)...")

    print("\nTest 1: LAYERNORM + 2D grid should be rejected...")
    try:
        test_layernorm_pre_all_gather_2d_rejects_layernorm()
        print("  ✓ Passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print("\nTest 2: RMSNORM + 2D grid should be accepted...")
    try:
        test_rmsnorm_pre_all_gather_2d_accepts_rmsnorm()
        print("  ✓ Passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print("\nTest 3: LAYERNORM without 2D grid should be accepted...")
    try:
        test_layernorm_pre_all_gather_without_2d_accepts_layernorm()
        print("  ✓ Passed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

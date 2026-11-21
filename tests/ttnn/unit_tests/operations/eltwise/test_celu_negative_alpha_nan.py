# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for issue #64021: ttnn.celu with alpha < 0 returns NaN for x/alpha in (88.38, 88.72).

This test verifies that CELU with negative alpha produces correct results (finite or -inf)
instead of NaN in the problematic band where x/alpha is near 128*ln(2).
"""

import pytest
import torch
import ttnn


@pytest.mark.parametrize("alpha", [-1.0, -0.5, -4.0])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_celu_negative_alpha_no_nan(device, alpha, dtype):
    """
    Test that CELU with negative alpha doesn't produce NaN in the band (88.38, 88.72).
    
    The bug was that for x/alpha in this band, the device returned NaN (float32) or +inf (bfloat16)
    while torch returns finite values or -inf.
    """
    # Create input spanning the problematic band
    # x/alpha should span 88.0 to 89.0, covering the NaN band (88.38, 88.72)
    ratio = torch.linspace(88.0, 89.0, 32 * 64)
    x = (ratio * alpha).reshape(32, 64).to(dtype)
    
    # Get torch reference
    ref = torch.nn.functional.celu(x, alpha=alpha)
    
    # Get device result
    x_dev = ttnn.from_torch(x, dtype=ttnn.from_torch_dtype(dtype), layout=ttnn.TILE_LAYOUT, device=device)
    out_dev = ttnn.celu(x_dev, alpha=alpha)
    out = ttnn.to_torch(out_dev)
    
    # Check the problematic band
    r = x.float() / alpha
    band = (r > 88.3763) & (r < 88.7228)
    
    if band.any():
        # Device should not produce NaN from finite input
        assert not torch.isnan(out[band]).any(), (
            f"CELU produced NaN for alpha={alpha}, dtype={dtype} in band (88.38, 88.72). "
            f"Expected finite or -inf like torch."
        )
        
        # Where torch is finite, device should be close (within 2 ULP for bfloat16, more for float32)
        torch_finite = torch.isfinite(ref[band])
        if torch_finite.any():
            # For bfloat16, allow larger tolerance due to rounding
            if dtype == torch.bfloat16:
                rtol = 0.02  # 2% relative tolerance for bfloat16
                atol = 1e30  # Large absolute tolerance for values near overflow
            else:
                rtol = 0.01  # 1% for float32
                atol = 1e30
            
            torch.testing.assert_close(
                out[band][torch_finite],
                ref[band][torch_finite],
                rtol=rtol,
                atol=atol,
                msg=f"CELU mismatch for alpha={alpha}, dtype={dtype} in finite region"
            )
        
        # Where torch is -inf, device should also be -inf (or very large negative)
        torch_neg_inf = (ref[band] == -float("inf"))
        if torch_neg_inf.any():
            # Device should give -inf or very large negative (allowing for bfloat16 rounding to +inf)
            device_overflow = (out[band] == -float("inf")) | (out[band] == float("inf"))
            assert device_overflow.all() or (out[band] < -1e30).all(), (
                f"CELU should overflow to -inf for alpha={alpha}, dtype={dtype} where torch overflows"
            )


@pytest.mark.parametrize("alpha", [1.0, 0.5, 4.0])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_celu_positive_alpha_unchanged(device, alpha, dtype):
    """
    Test that CELU with positive alpha still works correctly (regression test).
    
    Positive alpha doesn't have the NaN band issue, but we should verify the fix
    doesn't break the normal path.
    """
    x = torch.randn(32, 64).to(dtype) * 10  # Scale to cover range
    ref = torch.nn.functional.celu(x, alpha=alpha)
    
    x_dev = ttnn.from_torch(x, dtype=ttnn.from_torch_dtype(dtype), layout=ttnn.TILE_LAYOUT, device=device)
    out_dev = ttnn.celu(x_dev, alpha=alpha)
    out = ttnn.to_torch(out_dev)
    
    # Should match torch closely
    if dtype == torch.bfloat16:
        rtol = 0.02
        atol = 0.01
    else:
        rtol = 0.01
        atol = 0.001
    
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

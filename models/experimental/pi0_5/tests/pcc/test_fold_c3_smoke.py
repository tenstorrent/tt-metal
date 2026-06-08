"""Quick smoke test: does ttnn.fold work with C=3 (no padding to 4)?

If yes, we avoid the matmul inflation (608 → 784 in-features) and the fold path
could finally beat the linear+unfold baseline.
"""
import torch
import ttnn
import pytest


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_fold_c3(device):
    torch.manual_seed(0)
    B, H, W, C = 3, 224, 224, 3
    patch = 14
    x_torch = torch.randn(B, H, W, C, dtype=torch.float32)
    # Pre-reshape on host: (B, H, W/patch, C*patch) = (B, 224, 16, 42)
    x_re = x_torch.reshape(B, H, W // patch, C * patch).to(torch.bfloat16)
    print(f"\ninput shape: {tuple(x_re.shape)}  (C*patch = {C*patch})")
    x_ttnn = ttnn.from_torch(
        x_re,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    try:
        out = ttnn.fold(x_ttnn, patch, 1)
        print(f"fold output shape: {tuple(out.shape)}")
        out_torch = ttnn.to_torch(out)
        print(f"out torch shape: {tuple(out_torch.shape)}, dtype={out_torch.dtype}")
        expected_inner = patch * patch * C
        print(f"expected inner dim (no pad): {expected_inner}, actual: {out.shape[-1]}")
        assert torch.isfinite(out_torch).all(), "fold output has NaN/Inf"
        if out.shape[-1] == expected_inner:
            print("✓ ttnn.fold WORKS with C=3, no padding needed")
        else:
            print(f"⚠ ttnn.fold padded to {out.shape[-1]} (input C*patch was {C*patch})")
    except Exception as e:
        print(f"✗ fold failed with C=3: {type(e).__name__}: {e}")
        raise

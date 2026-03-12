# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test tilize -> untilize round-trip (compute kernel modified to skip all math).
If this passes, the untilize works after tilize in isolation.
If this fails, tilize+untilize is broken for fp32+Wt>1.
"""

import torch
import ttnn
from ttnn.operations.rms_norm import rms_norm


def test_roundtrip_fp32_wt2(device):
    """Tilize->untilize round-trip, fp32, Wt=2."""
    torch.manual_seed(42)
    shape = (1, 1, 32, 64)
    x = torch.randn(*shape, dtype=torch.float32)

    x_tt = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = rms_norm(x_tt, gamma=None, epsilon=1e-6)
    actual = ttnn.to_torch(out).float()

    diff = (actual - x).abs()
    print(f"\nRound-trip fp32 Wt=2: max_diff={diff.max().item():.6e}")
    print(f"  Row 0 max:  {diff[0,0,0].max().item():.6e}")
    print(f"  Row 15 max: {diff[0,0,15].max().item():.6e}")
    print(f"  Row 16 max: {diff[0,0,16].max().item():.6e}")
    print(f"  Row 31 max: {diff[0,0,31].max().item():.6e}")
    assert diff.max().item() < 0.01, f"Round-trip failed: max_diff={diff.max().item()}"


def test_roundtrip_bf16_wt2(device):
    """Tilize->untilize round-trip, bf16, Wt=2."""
    torch.manual_seed(42)
    shape = (1, 1, 32, 64)
    x = torch.randn(*shape, dtype=torch.bfloat16)

    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = rms_norm(x_tt, gamma=None, epsilon=1e-6)
    actual = ttnn.to_torch(out).bfloat16()

    diff = (actual.float() - x.float()).abs()
    print(f"\nRound-trip bf16 Wt=2: max_diff={diff.max().item():.6e}")
    assert diff.max().item() < 0.01, f"Round-trip failed: max_diff={diff.max().item()}"

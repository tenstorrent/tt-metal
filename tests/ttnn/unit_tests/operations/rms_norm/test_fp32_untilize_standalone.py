# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test if ttnn.untilize built-in also has fp32+Wt>1 bug."""

import torch
import ttnn


def test_ttnn_untilize_fp32_wt2(device):
    """Round-trip tilize->untilize for fp32 Wt=2."""
    torch.manual_seed(42)
    x = torch.randn(1, 1, 32, 64, dtype=torch.float32)
    x_tile = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    x_rm = ttnn.untilize(x_tile)
    actual = ttnn.to_torch(x_rm).float()
    diff = (actual - x).abs()
    print(f"\nttnn.untilize fp32 Wt=2: max_diff={diff.max().item():.6e}")
    print(f"  Row 0 max:  {diff[0,0,0].max().item():.6e}")
    print(f"  Row 15 max: {diff[0,0,15].max().item():.6e}")
    print(f"  Row 16 max: {diff[0,0,16].max().item():.6e}")
    print(f"  Row 31 max: {diff[0,0,31].max().item():.6e}")
    assert diff.max().item() < 0.01


def test_ttnn_untilize_fp32_wt1(device):
    """Sanity: fp32 Wt=1 should work."""
    torch.manual_seed(42)
    x = torch.randn(1, 1, 32, 32, dtype=torch.float32)
    x_tile = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    x_rm = ttnn.untilize(x_tile)
    actual = ttnn.to_torch(x_rm).float()
    diff = (actual - x).abs()
    print(f"\nttnn.untilize fp32 Wt=1: max_diff={diff.max().item():.6e}")
    assert diff.max().item() < 0.01


def test_ttnn_untilize_bf16_wt2(device):
    """Sanity: bf16 Wt=2 should work."""
    torch.manual_seed(42)
    x = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    x_tile = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    x_rm = ttnn.untilize(x_tile)
    actual = ttnn.to_torch(x_rm).float()
    diff = (actual.float() - x.float()).abs()
    print(f"\nttnn.untilize bf16 Wt=2: max_diff={diff.max().item():.6e}")
    assert diff.max().item() < 0.01

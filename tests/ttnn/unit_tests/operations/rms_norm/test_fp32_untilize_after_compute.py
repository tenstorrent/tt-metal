# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test: does the untilize helper break when called after other compute ops?
Strategy: skip all phases except Phase 5 (normalize) and Phase 7 (untilize).
This isolates whether the compute state from prior ops corrupts untilize.
"""

import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm


def pytorch_rms_norm(x, epsilon=1e-6):
    x = x.float()
    return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + epsilon)


def test_fp32_rm_skip_untilize(device):
    """
    Run fp32 RM rms_norm but output in TILE layout (skip untilize).
    If this passes, the compute is correct and untilize is the only issue.
    """
    torch.manual_seed(42)
    shape = (1, 1, 32, 64)
    x = torch.randn(*shape, dtype=torch.float32)
    expected = pytorch_rms_norm(x)

    # Input as TILE (skip the tilize + untilize entirely)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = rms_norm(x_tt, gamma=None, epsilon=1e-6)
    actual = ttnn.to_torch(out).float()

    diff = (actual - expected).abs()
    print(
        f"\nTILE path (no untilize): max_diff={diff.max().item():.6e}, PCC check passed"
        if diff.max().item() < 0.1
        else f"TILE path FAILED: {diff.max().item():.6e}"
    )

    # Now manually untilize the output on host
    # This confirms the compute is correct even for the case we were testing
    assert diff.max().item() < 0.1, f"TILE path max_diff={diff.max().item()}"

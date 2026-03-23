#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for custom DeltaNet recurrence kernel.
"""

import torch
from ttnn._ttnn.operations.experimental import deltanet_recurrence

import ttnn


def run_deltanet_kernel_test(num_heads, head_k_dim, head_v_dim, num_k_heads, gqa_ratio, device):
    """Run DeltaNet kernel with given params and verify output shape."""
    scale = 1.0 / (head_k_dim**0.5)
    norm_eps = 1e-6
    B_pad = 32

    k_tiles = head_k_dim // 32
    v_tiles = head_v_dim // 32

    # conv_out: Q + K + V sections
    conv_dim = num_k_heads * head_k_dim + num_k_heads * head_k_dim + num_heads * head_v_dim
    value_dim = num_heads * head_v_dim

    conv_out_torch = torch.randn(1, 1, B_pad, conv_dim, dtype=torch.bfloat16)
    # Pad small dims to tile-aligned (32)
    b_proj_dim = max(num_heads, 32)
    b_proj_torch = torch.randn(1, 1, B_pad, b_proj_dim, dtype=torch.bfloat16)
    a_proj_torch = torch.randn(1, 1, B_pad, b_proj_dim, dtype=torch.bfloat16)
    z_proj_torch = torch.randn(1, 1, B_pad, value_dim, dtype=torch.bfloat16)
    dt_bias_torch = torch.randn(1, 1, 32, b_proj_dim, dtype=torch.bfloat16)
    A_exp_torch = torch.randn(1, 1, 32, b_proj_dim, dtype=torch.bfloat16)
    norm_weight_torch = torch.ones(1, 1, 32, head_v_dim, dtype=torch.bfloat16)
    state_torch = torch.randn(1, num_heads, head_k_dim, head_v_dim, dtype=torch.bfloat16)

    conv_out_tt = ttnn.from_torch(conv_out_torch, layout=ttnn.TILE_LAYOUT, device=device)
    b_proj_tt = ttnn.from_torch(b_proj_torch, layout=ttnn.TILE_LAYOUT, device=device)
    a_proj_tt = ttnn.from_torch(a_proj_torch, layout=ttnn.TILE_LAYOUT, device=device)
    z_proj_tt = ttnn.from_torch(z_proj_torch, layout=ttnn.TILE_LAYOUT, device=device)
    dt_bias_tt = ttnn.from_torch(dt_bias_torch, layout=ttnn.TILE_LAYOUT, device=device)
    A_exp_tt = ttnn.from_torch(A_exp_torch, layout=ttnn.TILE_LAYOUT, device=device)
    norm_weight_tt = ttnn.from_torch(norm_weight_torch, layout=ttnn.TILE_LAYOUT, device=device)
    state_tt = ttnn.from_torch(state_torch, layout=ttnn.TILE_LAYOUT, device=device)

    output_tt = deltanet_recurrence(
        conv_out_tt,
        b_proj_tt,
        a_proj_tt,
        z_proj_tt,
        dt_bias_tt,
        A_exp_tt,
        norm_weight_tt,
        state_tt,
        num_heads=num_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        num_k_heads=num_k_heads,
        gqa_ratio=gqa_ratio,
        scale=scale,
        norm_eps=norm_eps,
    )

    output_torch = ttnn.to_torch(output_tt)
    assert output_torch.shape == (
        1,
        1,
        B_pad,
        value_dim,
    ), f"Expected (1,1,{B_pad},{value_dim}), got {output_torch.shape}"

    # Verify output is not all zeros or NaN
    assert not torch.isnan(output_torch).any(), "Output contains NaN"
    print(f"  Output min={output_torch.min().item():.4f}, max={output_torch.max().item():.4f}")

    return output_torch


def test_minimal():
    """1 head, 32-dim — minimal test."""
    device = ttnn.open_device(device_id=0)
    try:
        print("Test 1: Minimal (1 head, 32-dim)")
        out = run_deltanet_kernel_test(1, 32, 32, 1, 1, device)
        print(f"  Output shape: {out.shape} -- PASSED")
    finally:
        ttnn.close_device(device)


def test_full():
    """48 heads, 128-dim — full Qwen3.5-27B config."""
    device = ttnn.open_device(device_id=0)
    try:
        print("Test 2: Full (48 heads, 128-dim)")
        out = run_deltanet_kernel_test(48, 128, 128, 16, 3, device)
        print(f"  Output shape: {out.shape} -- PASSED")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_minimal()
    print()
    test_full()
    print("\nAll tests PASSED!")

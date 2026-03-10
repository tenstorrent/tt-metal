# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Correctness tests for group_norm with numerical verification.

Tests sub-tile groups (C/G < 32), tile-aligned groups (C/G >= 32),
various G values, and edge cases. Reports actual max diff per test case.
"""

import pytest
import torch
import ttnn

from .group_norm import group_norm


def pytorch_group_norm(x, num_groups, eps=1e-5, gamma=None, beta=None):
    """Reference group norm in PyTorch.

    Input x has shape (N, 1, HW, C). Groups split the C dimension.
    """
    N, _, HW, C = x.shape
    # Split C into groups: (N, 1, HW, C) -> (N, HW, G, C//G) -> (N, G, C//G, HW)
    x_r = x[:, 0, :, :].reshape(N, HW, num_groups, C // num_groups).permute(0, 2, 3, 1).float()
    m = x_r.mean(dim=[2, 3], keepdim=True)
    v = x_r.var(dim=[2, 3], unbiased=False, keepdim=True)
    normed = (x_r - m) / torch.sqrt(v + eps)
    # Reshape back to (N, 1, HW, C)
    normed = normed.permute(0, 3, 1, 2).reshape(N, HW, C).unsqueeze(1)
    if gamma is not None and beta is not None:
        normed = gamma.float() * normed + beta.float()
    return normed


def pearson_cc(a, b):
    """Compute Pearson correlation coefficient between two flat tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    a_mean = a_flat.mean()
    b_mean = b_flat.mean()
    a_c = a_flat - a_mean
    b_c = b_flat - b_mean
    num = (a_c * b_c).sum()
    den = torch.sqrt((a_c**2).sum() * (b_c**2).sum())
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return (num / den).item()


def run_group_norm_test(device, shape, num_groups, eps=1e-5, use_affine=True, seed=42):
    """Run a single group_norm test and return (max_diff, mean_diff, pcc)."""
    torch.manual_seed(seed)
    N, _, HW, C = shape

    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    if use_affine:
        gamma = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
        beta = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
    else:
        gamma = torch.ones(1, 1, 1, C, dtype=torch.bfloat16)
        beta = torch.zeros(1, 1, 1, C, dtype=torch.bfloat16)

    expected = pytorch_group_norm(torch_input, num_groups, eps, gamma, beta)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = group_norm(ttnn_input, num_groups=num_groups, gamma=gamma, beta=beta, eps=eps)
    torch_output = ttnn.to_torch(ttnn_output).float()

    diff = (torch_output - expected).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pcc = pearson_cc(torch_output, expected)

    return max_diff, mean_diff, pcc


# ============================================================
# Test 1: Tile-aligned groups (C/G >= 32) WITHOUT affine
# These should have tight tolerances since no gamma amplification
# ============================================================
@pytest.mark.parametrize(
    "shape,num_groups",
    [
        pytest.param((1, 1, 32, 64), 2, id="64C_2G_CpG32"),  # C/G=32, exact tile
        pytest.param((1, 1, 64, 128), 2, id="128C_2G_CpG64"),  # C/G=64
        pytest.param((1, 1, 64, 128), 4, id="128C_4G_CpG32"),  # C/G=32
        pytest.param((1, 1, 32, 256), 4, id="256C_4G_CpG64"),  # C/G=64
        pytest.param((2, 1, 64, 128), 2, id="batch2_128C_2G"),  # multi-batch
        pytest.param((1, 1, 32, 128), 1, id="128C_1G_layernorm"),  # G=1 = layernorm
    ],
)
def test_tile_aligned_no_affine(device, shape, num_groups):
    """Tile-aligned groups without affine. Should be very tight."""
    max_diff, mean_diff, pcc = run_group_norm_test(device, shape, num_groups, use_affine=False)
    print(f"\n  tile-aligned no-affine: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, pcc={pcc:.6f}")
    assert max_diff < 0.05, f"max_diff={max_diff:.4f} exceeds 0.05"
    assert pcc > 0.999, f"pcc={pcc:.6f} below 0.999"


# ============================================================
# Test 2: Tile-aligned groups WITH affine
# Gamma amplification expected, but should still be reasonable
# ============================================================
@pytest.mark.parametrize(
    "shape,num_groups",
    [
        pytest.param((1, 1, 32, 64), 2, id="64C_2G_CpG32"),
        pytest.param((1, 1, 64, 128), 2, id="128C_2G_CpG64"),
        pytest.param((1, 1, 64, 128), 4, id="128C_4G_CpG32"),
        pytest.param((2, 1, 64, 128), 2, id="batch2_128C_2G"),
    ],
)
def test_tile_aligned_with_affine(device, shape, num_groups):
    """Tile-aligned groups with random gamma/beta."""
    max_diff, mean_diff, pcc = run_group_norm_test(device, shape, num_groups, use_affine=True)
    print(f"\n  tile-aligned affine: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, pcc={pcc:.6f}")
    assert max_diff < 0.1, f"max_diff={max_diff:.4f} exceeds 0.1"
    assert pcc > 0.999, f"pcc={pcc:.6f} below 0.999"


# ============================================================
# Test 3: Sub-tile groups (C/G < 32) WITHOUT affine
# This is the case the masking was built for
# ============================================================
@pytest.mark.parametrize(
    "shape,num_groups",
    [
        pytest.param((1, 1, 32, 32), 2, id="32C_2G_CpG16"),  # C/G=16
        pytest.param((1, 1, 32, 32), 4, id="32C_4G_CpG8"),  # C/G=8
        pytest.param((1, 1, 32, 32), 8, id="32C_8G_CpG4"),  # C/G=4
        pytest.param((1, 1, 32, 32), 16, id="32C_16G_CpG2"),  # C/G=2
        pytest.param((1, 1, 32, 32), 32, id="32C_32G_CpG1"),  # C/G=1, each channel is own group
        pytest.param((1, 1, 32, 64), 4, id="64C_4G_CpG16"),  # C/G=16, multi-tile
        pytest.param((1, 1, 64, 64), 4, id="64C_4G_CpG16_tall"),  # C/G=16, Ht=2
        pytest.param((2, 1, 32, 32), 2, id="batch2_32C_2G_CpG16"),  # multi-batch sub-tile
    ],
)
def test_subtile_groups_no_affine(device, shape, num_groups):
    """Sub-tile groups (C/G < 32) without affine. Tests the masking correctness."""
    max_diff, mean_diff, pcc = run_group_norm_test(device, shape, num_groups, use_affine=False)
    print(f"\n  sub-tile no-affine: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, pcc={pcc:.6f}")
    assert max_diff < 0.05, f"max_diff={max_diff:.4f} exceeds 0.05"
    assert pcc > 0.999, f"pcc={pcc:.6f} below 0.999"


# ============================================================
# Test 4: Sub-tile groups WITH affine
# ============================================================
@pytest.mark.parametrize(
    "shape,num_groups",
    [
        pytest.param((1, 1, 32, 32), 2, id="32C_2G_CpG16"),
        pytest.param((1, 1, 32, 32), 4, id="32C_4G_CpG8"),
        pytest.param((1, 1, 32, 64), 4, id="64C_4G_CpG16"),
        pytest.param((2, 1, 32, 32), 2, id="batch2_32C_2G"),
    ],
)
def test_subtile_groups_with_affine(device, shape, num_groups):
    """Sub-tile groups with random gamma/beta."""
    max_diff, mean_diff, pcc = run_group_norm_test(device, shape, num_groups, use_affine=True)
    print(f"\n  sub-tile affine: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, pcc={pcc:.6f}")
    assert max_diff < 0.1, f"max_diff={max_diff:.4f} exceeds 0.1"
    assert pcc > 0.999, f"pcc={pcc:.6f} below 0.999"


# ============================================================
# Test 5: G=1 is equivalent to layer norm over (H*W, C)
# ============================================================
def test_g1_is_layernorm(device):
    """G=1 should produce layer norm result."""
    shape = (1, 1, 64, 128)
    max_diff, mean_diff, pcc = run_group_norm_test(device, shape, num_groups=1, use_affine=False)
    print(f"\n  G=1 layernorm: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, pcc={pcc:.6f}")
    assert max_diff < 0.05, f"max_diff={max_diff:.4f} exceeds 0.05"
    assert pcc > 0.999, f"pcc={pcc:.6f} below 0.999"


# ============================================================
# Test 6: G=C means each channel is its own group (instance-norm-like)
# Only valid when C is divisible by 32... and G. With C=32, G=32 -> C/G=1
# ============================================================
def test_g_equals_c(device):
    """G=C: each channel normalized independently over spatial dims only."""
    shape = (1, 1, 64, 32)
    max_diff, mean_diff, pcc = run_group_norm_test(device, shape, num_groups=32, use_affine=False)
    print(f"\n  G=C instance-norm: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, pcc={pcc:.6f}")
    assert max_diff < 0.05, f"max_diff={max_diff:.4f} exceeds 0.05"
    assert pcc > 0.999, f"pcc={pcc:.6f} below 0.999"


# ============================================================
# Test 7: Determinism — same inputs should produce same outputs
# ============================================================
def test_determinism(device):
    """Two runs with same seed should produce identical output."""
    shape = (1, 1, 32, 64)
    max_diff1, _, _ = run_group_norm_test(device, shape, num_groups=2, seed=123)
    max_diff2, _, _ = run_group_norm_test(device, shape, num_groups=2, seed=123)
    # Both should have exact same diff from reference
    assert max_diff1 == max_diff2, f"Non-deterministic: {max_diff1} vs {max_diff2}"


# ============================================================
# Test 8: Uniform input — output should be zero (or beta if affine)
# ============================================================
def test_uniform_input(device):
    """Uniform input: variance=0, output should be beta (or 0 without affine)."""
    shape = (1, 1, 32, 64)
    N, _, HW, C = shape
    torch.manual_seed(42)

    # All-same input: mean=5.0, var=0
    torch_input = torch.full(shape, 5.0, dtype=torch.bfloat16)
    gamma = torch.ones(1, 1, 1, C, dtype=torch.bfloat16)
    beta = torch.zeros(1, 1, 1, C, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = group_norm(ttnn_input, num_groups=2, gamma=gamma, beta=beta, eps=1e-5)
    torch_output = ttnn.to_torch(ttnn_output).float()

    # With uniform input, (x - mean) = 0 for all elements, so output should be beta = 0
    max_diff = torch_output.abs().max().item()
    print(f"\n  uniform input: max_val={max_diff:.6f} (should be ~0)")
    assert max_diff < 0.01, f"Uniform input should give ~0 output, got max={max_diff}"

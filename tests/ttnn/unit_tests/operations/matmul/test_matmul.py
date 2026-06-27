# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the 2D dual-multicast matmul op (IMMUTABLE — the spec).

C = A @ B, fused on-device kernels (2D grid + dual orthogonal multicast).
Phase 0 contract: float32 activation + float32 weight, TILE_LAYOUT,
tile-aligned M/K/N, shared 2D weight (and batched activation against it),
maxed precision (HiFi4, fp32_dest_acc_en=True).

The implementer is told NOT to modify this file.

- Reference: torch.matmul (computed in fp32).
- Reproducible: torch.manual_seed(42), torch.randn.
- PCC tolerance keyed by dtype (float32 -> 0.999, bfloat16 -> 0.995,
  bfloat8_b -> 0.99) — same thresholds as the golden suite.
- Device comes from the root `device` fixture (module-scoped via this dir's
  conftest.py); never open a device manually.
"""

import pytest
import torch
import ttnn

from ttnn.operations.matmul import matmul


# PCC threshold by activation dtype (same bands as the golden suite).
PCC_BY_DTYPE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # no native torch bf8b
}


def _pcc(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """Pearson correlation between two tensors (flattened, fp64)."""
    a = golden.flatten().to(torch.float64)
    b = calculated.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        # Both (centered) constant — match iff numerically identical.
        return 1.0 if torch.allclose(golden.to(torch.float64), calculated.to(torch.float64)) else 0.0
    return float((a @ b) / denom)


def _reference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """C = A @ B in fp32 (torch.matmul broadcasts a 2D B over A's batches)."""
    return torch.matmul(A.to(torch.float32), B.to(torch.float32))


# (A_shape, B_shape) — all tile-aligned (multiples of 32), shared 2D weight.
# Covers: single-tile, multi-tile, non-square, square, larger (per-core block
# may exceed one core), rank-3 batched activation, rank-4 batched activation.
SHAPES = [
    pytest.param((32, 32), (32, 32), id="single_tile"),
    pytest.param((64, 128), (128, 256), id="multi_tile"),
    pytest.param((128, 64), (64, 256), id="non_square"),
    pytest.param((256, 256), (256, 256), id="square_multi_tile"),
    pytest.param((512, 512), (512, 512), id="large_square"),
    pytest.param((2, 128, 64), (64, 128), id="batched_rank3_shared_weight"),
    pytest.param((1, 2, 128, 256), (256, 128), id="batched_rank4_shared_weight"),
]


@pytest.mark.parametrize("a_shape, b_shape", SHAPES)
def test_matmul(device, a_shape, b_shape):
    """matmul(A, B) with default (maxed) compute config matches torch.matmul."""
    dtype = ttnn.float32
    torch.manual_seed(42)

    torch_a = torch.randn(a_shape, dtype=_TORCH_DTYPE[dtype])
    torch_b = torch.randn(b_shape, dtype=_TORCH_DTYPE[dtype])
    expected = _reference(torch_a, torch_b)

    ttnn_a = ttnn.from_torch(torch_a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(torch_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_out = matmul(ttnn_a, ttnn_b)

    # Output shape: A's leading dims with last dim swapped to N.
    expected_shape = list(a_shape[:-1]) + [b_shape[-1]]
    assert list(ttnn_out.shape) == expected_shape, f"{list(ttnn_out.shape)} != {expected_shape}"
    assert ttnn_out.dtype == dtype
    assert ttnn_out.layout == ttnn.TILE_LAYOUT

    out = ttnn.to_torch(ttnn_out).to(torch.float32)
    pcc = _pcc(expected, out)
    assert pcc >= PCC_BY_DTYPE[dtype], f"PCC {pcc} < {PCC_BY_DTYPE[dtype]} for {a_shape}@{b_shape}"


@pytest.mark.parametrize("a_shape, b_shape", SHAPES)
def test_matmul_explicit_maxed_config(device, a_shape, b_shape):
    """Same, but with an explicit maxed ComputeConfigDescriptor (HiFi4, fp32 acc)."""
    dtype = ttnn.float32
    torch.manual_seed(42)

    torch_a = torch.randn(a_shape, dtype=_TORCH_DTYPE[dtype])
    torch_b = torch.randn(b_shape, dtype=_TORCH_DTYPE[dtype])
    expected = _reference(torch_a, torch_b)

    ttnn_a = ttnn.from_torch(torch_a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(torch_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
    ttnn_out = matmul(ttnn_a, ttnn_b, compute_kernel_config=config)

    out = ttnn.to_torch(ttnn_out).to(torch.float32)
    pcc = _pcc(expected, out)
    assert pcc >= PCC_BY_DTYPE[dtype], f"PCC {pcc} < {PCC_BY_DTYPE[dtype]} for {a_shape}@{b_shape}"


def test_matmul_rejects_rank1_activation(device):
    """A rank < 2 is a shape-contract violation -> ValueError/RuntimeError."""
    torch.manual_seed(42)
    a = ttnn.from_torch(torch.randn(32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises((ValueError, RuntimeError)):
        matmul(a, b)


def test_matmul_rejects_k_mismatch(device):
    """A[-1] != B[-2] is a shape-contract violation -> ValueError/RuntimeError."""
    torch.manual_seed(42)
    a = ttnn.from_torch(torch.randn(64, 128), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(96, 256), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises((ValueError, RuntimeError)):
        matmul(a, b)

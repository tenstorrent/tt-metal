# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for permute — IMMUTABLE. Do not modify; it is the spec.

Phase 0 regime: whole-tile relocation (TILE layout, fp32, tile-aligned, rank 4,
dims keeping the innermost two dims in place, interleaved DRAM output).
Regimes are PINNED by passing explicit `dims` rather than inferring a path
from the shape.
"""

import pytest
import torch

import ttnn

from ttnn.operations.permute import permute

PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

SHAPES = [
    (1, 1, 32, 32),  # single tile
    (2, 4, 64, 128),  # multi-tile, multi-batch
    (1, 2, 128, 4096),  # non-square, wide inner
    (4, 8, 128, 256),  # larger grid fill
]

# Inner-pair-preserving permutations: the pure tile-movement regime.
DIMS = [
    (1, 0, 2, 3),  # outer swap
    (0, 1, 2, 3),  # identity
]


def _reference(torch_input, dims):
    return torch.permute(torch_input.to(torch.float32), dims).contiguous()


def _assert_pcc(actual, expected, dtype):
    a = actual.flatten().to(torch.float64)
    e = expected.flatten().to(torch.float64)
    ac, ec = a - a.mean(), e - e.mean()
    den = ac.norm() * ec.norm()
    pcc = (ac * ec).sum().item() / den.item() if den.item() > 1e-30 else 1.0
    assert not torch.isnan(a).any(), "output contains NaN"
    assert not torch.isinf(a).any(), "output contains Inf"
    assert pcc >= PCC[dtype], f"PCC {pcc:.6f} < {PCC[dtype]}"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dims", DIMS)
def test_permute_tile_movement(device, shape, dims):
    """Phase 0 acceptance: fp32 / TILE / tile-aligned / rank 4 / DRAM interleaved."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    expected = _reference(torch_input, dims)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out = permute(tt_in, dims)

    assert list(tt_out.shape) == [shape[d] for d in dims]
    assert tt_out.dtype == ttnn.float32
    assert tt_out.layout == ttnn.TILE_LAYOUT

    _assert_pcc(ttnn.to_torch(tt_out), expected, ttnn.float32)


@pytest.mark.parametrize("shape", [(2, 4, 64, 128), (1, 2, 128, 4096)])
def test_permute_round_trip_is_identity(device, shape):
    """permute(permute(x, dims), inverse(dims)) == x — the correctness invariant."""
    dims = (1, 0, 2, 3)
    inverse = tuple(sorted(range(len(dims)), key=lambda i: dims[i]))

    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out = permute(permute(tt_in, dims), inverse)

    assert list(tt_out.shape) == list(shape)
    _assert_pcc(ttnn.to_torch(tt_out), torch_input.to(torch.float32), ttnn.float32)

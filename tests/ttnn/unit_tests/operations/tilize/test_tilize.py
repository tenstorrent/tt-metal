# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for tilize — Phase 0 corner.

bfloat16 ROW_MAJOR interleaved DRAM in -> bfloat16 TILE interleaved DRAM out,
tile-aligned, no padding, 32x32 tiles. Immutable spec for the implementer.
"""

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize

PCC_BY_DTYPE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}


def torch_tilize_reference(x: torch.Tensor) -> torch.Tensor:
    """tilize keeps every element at its logical position: the reference is identity."""
    return x.clone()


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    if torch.equal(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 0.0
    return float((a @ b) / denom)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),  # single tile
        (1, 1, 64, 128),  # multi-tile
        (1, 1, 96, 64),  # non-square
        (2, 3, 64, 64),  # multi-batch: leading dims fold into tile-rows
        (1, 1, 2048, 64),  # tall_narrow at grid scale
    ],
    ids=["single_tile", "multi_tile", "non_square", "multi_batch", "tall_narrow"],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_tilize(device, shape, dtype):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=TORCH_DTYPE[dtype])

    tt_in = ttnn.from_torch(
        x,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_out = tilize(tt_in)

    assert tt_out.layout == ttnn.TILE_LAYOUT
    assert tt_out.dtype == dtype
    assert tuple(tt_out.shape) == tuple(shape), "logical shape must be unchanged"
    assert tt_out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert tt_out.memory_config().buffer_type == ttnn.BufferType.DRAM

    out = ttnn.to_torch(tt_out)
    expected = torch_tilize_reference(x)

    assert _pcc(expected, out) >= PCC_BY_DTYPE[dtype]
    # No arithmetic and no cast: the re-lay must be bit-identical.
    assert torch.equal(out, expected)


def test_tilize_explicit_memory_config(device):
    torch.manual_seed(42)
    shape = (1, 1, 64, 128)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = tilize(tt_in, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert tt_out.layout == ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(tt_out), x)


def test_tilize_rejects_non_aligned_without_padding(device, expect_error):
    x = torch.randn((1, 1, 32, 50), dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with expect_error((ValueError, RuntimeError), ""):
        tilize(tt_in)


def test_tilize_rejects_bad_tile_width(device, expect_error):
    x = torch.randn((1, 1, 32, 64), dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with expect_error((ValueError, RuntimeError), ""):
        tilize(tt_in, tile=ttnn.Tile([32, 16]))


def test_tilize_rejects_host_tensor(expect_error):
    x = torch.randn((1, 1, 32, 32), dtype=torch.bfloat16)
    tt_host = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    with expect_error((ValueError, RuntimeError), ""):
        tilize(tt_host)

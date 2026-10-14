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


# --- Axes the verifier found already realized by the Phase 0 kernels (knob-turns) ---------


@pytest.mark.parametrize("tile_h", [16, 8, 4, 2, 1])
@pytest.mark.parametrize("shape", [(1, 1, 32, 64), (1, 1, 4096, 64)], ids=["small", "tall_narrow"])
def test_tilize_tiny_tile(device, shape, tile_h):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = tilize(tt_in, tile=ttnn.Tile([tile_h, 32]))
    assert list(tt_out.tile.tile_shape) == [tile_h, 32]
    assert torch.equal(ttnn.to_torch(tt_out), x)


@pytest.mark.parametrize(
    "in_mc, out_mc",
    [
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        (ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ],
    ids=["dram_to_l1", "l1_to_l1", "l1_to_dram"],
)
@pytest.mark.parametrize("low_l1", [False, True])
def test_tilize_buffers_low_l1(device, in_mc, out_mc, low_l1):
    torch.manual_seed(0)
    x = torch.randn((1, 1, 4608, 288), dtype=torch.bfloat16)  # C=9: low_l1 -> blocks 8 + ragged 1
    tt_in = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)
    tt_out = tilize(tt_in, memory_config=out_mc, low_l1=low_l1)
    assert tt_out.memory_config().buffer_type == out_mc.buffer_type
    assert torch.equal(ttnn.to_torch(tt_out), x)


@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 64), (1, 2, 1, 32, 64), (2, 1, 1, 1, 32, 64)])
def test_tilize_ranks(device, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = tilize(tt_in)
    assert tuple(tt_out.shape) == tuple(shape)
    assert torch.equal(ttnn.to_torch(tt_out), x)

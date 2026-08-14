# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 5 — tile geometry: tiny tiles (a 32-wide tile with FEWER than 32
rows) and the retile path (an already-TILE input re-tiled to another height).

DO NOT DELETE. These are the pins for the four Refinement-5 claims:

  * the requested `tile` reaches the OUTPUT TENSOR SPEC, not only the CBs — the
    host allocator's shape/dtype/layout overload silently builds a 32x32 tile,
    which is the one host-side gap the whole tiny-tile axis was blocked on;
  * a tiny tile is EXACT at every height (1/2/4/8/16), interleaved and sharded,
    for every non-block-float dtype — tilize is a byte permutation, so PCC is the
    wrong instrument here;
  * `tile_height=16` x a BLOCK-FLOAT output is a PLATFORM pack gap, reproduced by
    an op that has nothing to do with tilize, hence an EXCLUSION rather than a
    kernel bug (`test_bfp8_16x32_is_a_platform_pack_gap` is the disqualifier the
    op file cites);
  * H-alignment is measured against the REQUESTED tile height, not a hardcoded
    32, so a tiny tile re-aligns its own cells.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import ExcludedCell
from ttnn.operations.tilize import EXCLUSIONS, SUPPORTED, tilize
from ttnn.operations.tilize import tilize_program_descriptor as pd


TINY_HEIGHTS = [16, 8, 4, 2, 1]

_EXACT_DTYPES = [
    pytest.param(ttnn.bfloat16, id="bf16"),
    pytest.param(ttnn.float32, id="fp32"),
    pytest.param(ttnn.uint32, id="uint32"),
    pytest.param(ttnn.uint8, id="uint8"),
]


def _make_input(dtype, shape):
    if dtype == ttnn.uint8:
        return torch.randint(0, 256, shape, dtype=torch.uint8)
    if dtype in (ttnn.uint16, ttnn.uint32):
        return torch.randint(0, 1000, shape, dtype=torch.int32)
    if dtype == ttnn.int32:
        return torch.randint(-1000, 1000, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.randn(shape, dtype=torch.float32)
    return torch.randn(shape).bfloat16()


def _tilize_and_read(torch_input, dtype, tile_height, *, device, memory_config=None, out_dtype=None):
    memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    tt_output = tilize(
        tt_input,
        memory_config=memory_config,
        dtype=out_dtype if out_dtype is not None else dtype,
        tile=ttnn.Tile([tile_height, 32]),
    )
    return tt_output, ttnn.to_torch(tt_output)


# --- the host-side gap this refinement existed to close ---------------------


@pytest.mark.parametrize("tile_height", TINY_HEIGHTS + [32])
def test_output_tensor_carries_the_requested_tile(tile_height, device):
    """The output TENSOR SPEC — not just the CB TileDescriptor — must carry the
    requested geometry.

    `ttnn.allocate_tensor_on_device(shape, dtype, layout, device, mem_config)`
    has no `tile=` parameter, so it always builds a 32x32 tile: the kernels would
    pack tiny pages into a buffer laid out for 32-row tiles. The TensorSpec
    overload is what threads it through, and the page size is the observable.
    """
    torch_input = _make_input(ttnn.bfloat16, [1, 1, 64, 64])
    tt_output, _ = _tilize_and_read(torch_input, ttnn.bfloat16, tile_height, device=device)
    assert list(tt_output.tile.tile_shape) == [tile_height, 32]
    assert tt_output.buffer_page_size() == tile_height * 32 * 2


# --- identity at every tiny height ------------------------------------------


@pytest.mark.parametrize("tile_height", TINY_HEIGHTS)
@pytest.mark.parametrize("dtype", _EXACT_DTYPES)
@pytest.mark.parametrize("shape", [[1, 1, 128, 256], [2, 3, 32, 64]], ids=["multicore", "rank4_small"])
def test_tiny_tile_identity_is_exact(tile_height, dtype, shape, device):
    """A permutation op has no error budget — compare exactly, at every height
    down to the degenerate 1x32 (one source stick per output tile)."""
    torch_input = _make_input(dtype, shape)
    _, got = _tilize_and_read(torch_input, dtype, tile_height, device=device)
    # int64 on both sides: an unsigned readback will not promote against a torch
    # int32 source, and this comparison must stay EXACT (tilize does no arithmetic).
    if dtype in (ttnn.uint8, ttnn.uint16, ttnn.uint32, ttnn.int32):
        got, torch_input = got.to(torch.int64), torch_input.to(torch.int64)
    assert torch.equal(got, torch_input), f"tile_height={tile_height} {dtype} max diff on {shape}"


@pytest.mark.parametrize("tile_height", TINY_HEIGHTS)
def test_tiny_tile_identity_is_exact_on_a_local_shard(tile_height, device):
    """The sharded tiny-tile factory: both CBs aliased on the resident L1 shard
    (zero NoC), with the shard's own rows folded into tile_height-row tiles."""
    shape = [1, 1, 32, 1024]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    torch_input = _make_input(ttnn.bfloat16, shape)
    _, got = _tilize_and_read(torch_input, ttnn.bfloat16, tile_height, device=device, memory_config=memory_config)
    assert torch.equal(got, torch_input)


@pytest.mark.parametrize("tile_height", [16, 4])
def test_tiny_tile_pad_region_is_exact_including_a_widening_cast(tile_height, device):
    """A tiny tile redefines H-alignment (the multiple is `tile_height`, not 32),
    and the writer's OUTPUT-format pad stamp has to address the tile through the
    TINY face geometry — a sub-16-row tile's face height IS its tile height.
    `pad_value=10.2` is the fill bf16 cannot hold, i.e. the one that reaches the
    stamp at all.
    """
    shape = [1, 1, 50, 50]
    torch_input = _make_input(ttnn.bfloat16, shape)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input, dtype=ttnn.float32, pad_value=10.2, tile=ttnn.Tile([tile_height, 32]))
    got = tt_output.cpu().to_torch_with_padded_shape()

    padded_h = ((50 + tile_height - 1) // tile_height) * tile_height
    expected = torch.nn.functional.pad(torch_input.float(), (0, 64 - 50, 0, padded_h - 50), value=10.2)
    assert list(got.shape) == [1, 1, padded_h, 64]
    assert torch.equal(got, expected)


def test_alignment_is_measured_against_the_requested_tile_height():
    """`tag_alignment` gates H on the tile height the CALL asks for — a 8-row tile
    makes H=56 tile-aligned, which a hardcoded 32 would mis-gate as h_non_aligned."""
    from ttnn.operations.tilize.tilize import tag_alignment

    assert tag_alignment(({"input_shape": [1, 1, 56, 64], "tile_height": 8},), {}) == "tile_aligned"
    assert tag_alignment(({"input_shape": [1, 1, 56, 64], "tile_height": 32},), {}) == "h_non_aligned"
    assert tag_alignment(({"input_shape": [1, 1, 56, 50], "tile_height": 8},), {}) == "w_non_aligned"


# --- the one EXCLUSION, and the platform behaviour that justifies it ---------


def test_bfp8_16x32_is_a_platform_pack_gap(device):
    """The disqualifier for `EXCLUSIONS[{tile_height:16, output_dtype:bfloat8_b}]`.

    Metal flags EVERY sub-32-row tile as `partial_face` (tile.cpp), but a 16x32
    tile's two faces are FULL 16-row faces. That routes the packer to the
    partial-face BFP MOP, whose DEST walk over-advances at face_r_dim == 16. The
    proof that this is not tilize's bug: a plain eltwise `ttnn.mul` on a 16x32
    bfloat8_b tile — no tilize anywhere in the program — returns the same wrong
    bytes, while the 32-row tile is fine.

    If this test ever starts FAILING (i.e. the platform packs a 16x32 block-float
    tile correctly), delete the EXCLUSION: the cell is unblocked.
    """
    source = torch.arange(16 * 32).reshape(1, 1, 16, 32).float()

    def roundtrip(tile_height):
        tt = ttnn.from_torch(
            source.bfloat16(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile([tile_height, 32]),
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.to_torch(ttnn.mul(tt, 1.0)).float()

    assert torch.allclose(roundtrip(32), source, atol=8.0), "the 32-row control must be sane"
    got16 = roundtrip(16)
    assert not torch.allclose(got16, source, atol=8.0), (
        "a 16x32 block-float tile now survives a plain eltwise op — the platform gap is fixed, "
        "so the tilize EXCLUSION should be removed"
    )


def test_block_float_output_is_refused_only_at_tile_height_16(device, expect_error):
    """The EXCLUSION is exactly one cell wide: every other tiny height packs
    bfloat8_b correctly (their faces really are partial), and 16 is fine for every
    non-block-float dtype."""
    assert {"tile_height": 16, "output_dtype": ttnn.bfloat8_b} in EXCLUSIONS
    torch_input = torch.randn(1, 1, 64, 64).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(ExcludedCell, "(?i)tile_height"):
        tilize(tt_input, dtype=ttnn.bfloat8_b, tile=ttnn.Tile([16, 32]))

    # bfloat8_b is lossy at pack time (7-bit mantissa on a shared exponent), so
    # this one is a relative check — the failure mode it guards against is the
    # tile_height=16 signature (values off by a whole face and two binades), not
    # a last-bit difference.
    reference = torch_input.float()
    scale = reference.abs().max()
    for tile_height in (8, 4, 2, 1):
        got = ttnn.to_torch(tilize(tt_input, dtype=ttnn.bfloat8_b, tile=ttnn.Tile([tile_height, 32]))).float()
        assert (got - reference).abs().max() < 0.05 * scale, tile_height


def test_supported_lists_every_tile_height():
    assert SUPPORTED["tile_height"] == [32] + TINY_HEIGHTS


# --- the blocking model still derives from the tile height -------------------


@pytest.mark.parametrize("tile_height", TINY_HEIGHTS + [32])
def test_cb_geometry_tracks_the_tile_height(tile_height):
    """`cb_bytes()` stays the single source and scales with the tile height — a
    tiny tile is a smaller page, never a hardcoded 32-row one."""
    in_tile_bytes = tile_height * pd.DEFAULT_TILE_WIDTH * 2
    assert pd.cb_bytes(2, 4, in_tile_bytes, in_tile_bytes) == 2 * pd.NT_BLK * 4 * 2 * in_tile_bytes
    # The L1 cap therefore ADMITS a wider chunk on a tinier tile, which is what
    # keeps a tiny-tile call from collapsing its block factor.
    assert pd.wt_cap(2, in_tile_bytes, in_tile_bytes) >= pd.wt_cap(2, 32 * 32 * 2, 32 * 32 * 2)

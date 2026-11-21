# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: the OFT encoder's layout conversions and BEV-grid slices from the **second** kernel
cache (in-tree ``tt_encoder`` / demo at ``628a4797bf7``). The data-movement owner left these to
the encoder group.

In cache #2 the BEV grid ``grid_`` is [160, 160, 3] bf16 **TILE** DRAM (``from_torch`` in the
demo). In cache #1 it was ROW_MAJOR, which ``test_slice.py`` covers (422600). The positions
decode slices it twice (``centers = grid[1:, 1:] + grid[:-1, :-1]``):

  * ``grid[:-1, :-1]``: tile-aligned begins, so this is the tiled slice. The kernel is
    ``reader_unary_unpad_dims_interleaved_start_id/351409`` (ct: rank 3, DRAM, 2048-B tiles) [H]
  * ``grid[1:, 1:]``: begin 1 on the tiled H dim is not tile-aligned. ``ttnn.slice`` at
    ``628a4797bf7`` then takes its row-major path. The steps are:
      1. untilize the whole grid: ``untilize/237994`` [40, 1] = 800 tile-rows = 160 x 5 over 20 cores,
         with ``writer_unary_stick_layout_split_rows_multicore/895982`` (6-B rows in 64-B DRAM pages)
      2. row-major slice: ``slice_reader_unary_unpad_dims_rm_interleaved_start_id/529889`` (DRAM, 64 B)
      3. tilize back: ``tilize`` 106517 / 184243 [40, 1] / [35, 1] = 795 tile-rows = 159 x 5, through
         the 6-B ``reader_unary_pad_dims_split_rows_multicore``                                        [H/M]
    Current main may lower this differently, so the 3-channel untilize also gets a standalone case.

A second pad-dims reader, ``reader_unary_pad_dims_split_rows_multicore/980752``, reads 2-B
(1-channel) DRAM rows into the same [40, 1] / [35, 1] tilize: an RM [1, 159, 159, 1] -> TILE
conversion. Its call site is not recoverable; the closest candidate is the NMS heatmap /
``nms_conv`` path. The case is [L].

Not added (already covered):
  * the 3-channel tilize of the sliced grid: same reader / factory / memcfg as
    ``test_to_layout.py`` ``image-c3-bf16-dram-366418``
  * the ``ang_offsets[:, i]`` TILE L1 slices (``test_slice.py`` 119790)
  * the head-part tilizes (``test_to_layout.py`` 522636)
  * the decode TILE permutes (``test_exp_decode.py``)

Value-preserving: checked bit-exact.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

BEV_GRID = (U.BEV + 1, U.BEV + 1, 3)  # [160, 160, 3] BEV grid corners


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, begins, ends",
    [
        pytest.param(BEV_GRID, (0, 0, 0), (U.BEV, U.BEV, 3), id="grid-tile-:-1,:-1-351409"),
        pytest.param(BEV_GRID, (1, 1, 0), BEV_GRID, id="grid-tile-1:,1:-unaligned-237994"),
    ],
)
def test_grid_slice_tile(ttnn_mesh_device, reset_seeds, shape, begins, ends):
    """``grid[...]`` on a bf16 TILE DRAM grid. Output stays TILE DRAM (the slice re-tilizes on the RM path)."""
    mesh = ttnn_mesh_device
    x = U.torch_rand(shape)
    tt_x = U.to_tt(x, mesh)  # bf16 TILE DRAM, as the demo uploads grid_
    out = U.op("slice")(tt_x, list(begins), list(ends))
    ref = x[tuple(slice(b, e) for b, e in zip(begins, ends))]
    U.assert_shape_dtype(out, shape=ref.shape, finite=False)
    assert out.layout == ttnn.TILE_LAYOUT
    U.assert_lossless(ref, out, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize("shape", [pytest.param(BEV_GRID, id="grid-untilize-c3-160x160x3-237994")])
def test_untilize_narrow_c3(ttnn_mesh_device, reset_seeds, shape):
    """TILE DRAM -> ROW_MAJOR DRAM with W = 3: untilize_with_unpadding to 6-B rows."""
    mesh = ttnn_mesh_device
    x = U.torch_rand(shape)
    out = U.op("to_layout")(U.to_tt(x, mesh), ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    U.assert_lossless(x, out, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize("shape", [pytest.param((1, U.BEV, U.BEV, 1), id="c1-tilize-1x159x159x1-980752")])
def test_tilize_narrow_c1(ttnn_mesh_device, reset_seeds, shape):
    """[L] ROW_MAJOR DRAM -> TILE DRAM with C = 1: tilize_with_val_padding from 2-B rows."""
    mesh = ttnn_mesh_device
    x = U.torch_rand(shape)
    tt_x = U.to_tt(x, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)
    out = U.op("to_layout")(tt_x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert out.layout == ttnn.TILE_LAYOUT
    U.assert_lossless(x, out, mesh_device=mesh)

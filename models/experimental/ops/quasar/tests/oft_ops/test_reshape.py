# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.reshape`` — the non-view reshapes the OFT model ran.

Shape source (kernel cache, ``reshape_view`` factories at ``1088b10b014``):
  * OFT linear-weight prep: ``reshape(W, [256, 7, 128])`` of the bf16 TILE DRAM [1792, 128]
    weight (``reader/writer_reshape_tiled`` 136476 / 150108; 512-B mapping pages, 32 entries;
    the follow-on transpose args pin the shape)                                            [H]
  * conv DRAM-slicing input reshape ``[1, 1, 25281, 128] -> [1, 159, 159, 128]``, bf16 TILE
    DRAM (142673 / 176769; 136 mapping entries, nearest simulated candidate)              [M]
  * head ``reshape(outputs, (1, -1, 9, 159, 159))``: bf16 ROW_MAJOR L1 [1, 9, 1, 25281]
    (``rm_reshape_interleaved`` 135821 / 972639: src page 50562 B, dst page 318 B)          [H]

Second cache (in-tree model, factories at ``628a4797bf7``):
  * precomputed grid_sample grid reshape ``reshape(grid, [1, 1, rows, 7 * 6])`` (tt_oft.py
    ``reshape_corners_lambda``): bf16 ROW_MAJOR DRAM [1, rows, 7, 6] -> [1, 1, rows, 42]
    (``rm_reshape_interleaved`` 107626 / 832696: src page 12 B, dst page 84 B); rows = 1408 /
    2112 / 2304 for the 18 / 12 / 11 OFT slices                                              [H]
  * OFT ``reshape(features, [1, H, W, -1])`` of the bf16 TILE DRAM lateral features for the
    1/16 and 1/32 scales ([1, 1, 1920, 256] -> [1, 24, 80, 256], [1, 1, 480, 256] ->
    [1, 12, 40, 256]: W not a tile multiple, so not a view). ``reshape_tiled`` has 5 cached
    variants (mapping pages 512 / 1024 / 1216 (L1 input) / 1536 / 2176 B); which variant is
    which scale is not pinned down: the hash suffixes are candidates                           [L]
  * the 1/8 scale ([1, 1, 7680, 256] -> [1, 48, 160, 256]) is tile-aligned: a view, no kernel.

Value-preserving: checked bit-exact.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

BEV = U.BEV
LAT_C = 256  # in-tree model lateral channels (second cache)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "in_shape, out_shape, layout_rm, mem",
    [
        pytest.param(
            (U.VOX_C, U.TD_C),
            (U.OFT_C, U.GRID_POINTS, U.TD_C),
            False,
            "dram",
            id="oft-weight-1792x128->256x7x128-136476",
        ),
        pytest.param(
            (1, 1, U.BEV_NHW, U.TD_C), (1, BEV, BEV, U.TD_C), False, "dram", id="topdown-in-25281->159x159-142673"
        ),
        pytest.param((1, 1, 32 * 32, 64), (1, 32, 32, 64), False, "dram", id="proxy-flat->nhwc-32x32"),
        pytest.param(
            (1, U.HEAD_OUT, 1, U.BEV_NHW),
            (1, 1, U.HEAD_OUT, BEV, BEV),
            True,
            "l1",
            id="head-out-rm-1x9x1x25281->5d-135821",
        ),
        # second cache
        pytest.param(
            (1, 1408, U.GRID_POINTS, 6),
            (1, 1, 1408, U.GRID_POINTS * 6),
            True,
            "dram",
            id="grid8-rm-1408x7x6->1408x42-107626",
        ),
        pytest.param(
            (1, 64, U.GRID_POINTS, 6), (1, 1, 64, U.GRID_POINTS * 6), True, "dram", id="proxy-grid-rm-64x7x6->64x42"
        ),
        pytest.param((1, 1, 24 * 80, LAT_C), (1, 24, 80, LAT_C), False, "dram", id="oft16-feats-1920->24x80-142827"),
        pytest.param((1, 1, 12 * 40, LAT_C), (1, 12, 40, LAT_C), False, "dram", id="oft32-feats-480->12x40-143492"),
    ],
)
def test_reshape(ttnn_mesh_device, reset_seeds, in_shape, out_shape, layout_rm, mem):
    mesh = ttnn_mesh_device
    x = U.torch_rand(in_shape)
    layout = ttnn.ROW_MAJOR_LAYOUT if layout_rm else ttnn.TILE_LAYOUT
    tt_x = U.to_tt_mem(x, mesh, mem, layout=layout)
    out = U.op("reshape")(tt_x, out_shape)
    U.assert_shape_dtype(out, shape=out_shape, finite=False)
    U.assert_lossless(x.reshape(out_shape), out, mesh_device=mesh)

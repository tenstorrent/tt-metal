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

Value-preserving: checked bit-exact.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

BEV = U.BEV


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

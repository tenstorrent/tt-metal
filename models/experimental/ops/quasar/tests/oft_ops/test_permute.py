# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.permute`` — every permute the OFT model ran.

Shape source (kernel cache, permute / transpose factories at ``1088b10b014``):
  * OFT linear-weight prep ``permute(W, (1, 0, 2))`` on bf16 TILE DRAM [256, 7, 128] ->
    [7, 256, 128]: lowered to transpose_hc (``reader/writer_unary_transpose_hc_interleaved_
    tiled_padding_aware`` 717327 / 148610; writer ct C=256, H=7, W=128, DRAM)                  [H]
  * head ``permute(outputs, (0, 3, 1, 2), memory_config=L1)``: bf16 ROW_MAJOR DRAM
    [1, 1, 25281, 9] -> L1 [1, 9, 1, 25281] (``*_permute_interleaved_rm_blocked_generic``
    156026 / 110535 + ``transpose_xw_rm_single_tile_size/147729``; named args num_rows=25281,
    in page 18 B, out page 50562 B)                                                              [H]
  * encoder ``permute(heatmaps_4d, (0, 2, 3, 1))``: bf16 ROW_MAJOR L1 [1, 1, 159, 159] ->
    [1, 159, 159, 1] (148039 / 134397; num_rows=159, page 318 B)                                [H]

Value-preserving: checked bit-exact.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

BEV = U.BEV


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, dims, layout_rm, in_mem, out_mem",
    [
        pytest.param(
            (U.OFT_C, U.GRID_POINTS, U.TD_C), (1, 0, 2), False, "dram", "dram", id="oft-weight-256x7x128-102-717327"
        ),
        pytest.param((1, 1, U.BEV_NHW, U.HEAD_OUT), (0, 3, 1, 2), True, "dram", "l1", id="head-out-rm-0312-156026"),
        pytest.param((1, 1, 1024, U.HEAD_OUT), (0, 3, 1, 2), True, "dram", "l1", id="proxy-head-out-rm-0312"),
        pytest.param((1, 1, BEV, BEV), (0, 2, 3, 1), True, "l1", "l1", id="nms-heatmap-rm-0231-148039"),
    ],
)
def test_permute(ttnn_mesh_device, reset_seeds, shape, dims, layout_rm, in_mem, out_mem):
    mesh = ttnn_mesh_device
    x = U.torch_rand(shape)
    layout = ttnn.ROW_MAJOR_LAYOUT if layout_rm else ttnn.TILE_LAYOUT
    tt_x = U.to_tt_mem(x, mesh, in_mem, layout=layout)
    out = ttnn.permute(tt_x, dims, memory_config=U.resolve_memcfg(out_mem, mesh))
    ref = x.permute(dims)
    U.assert_shape_dtype(out, shape=ref.shape, finite=False)
    U.assert_lossless(ref, out, mesh_device=mesh)

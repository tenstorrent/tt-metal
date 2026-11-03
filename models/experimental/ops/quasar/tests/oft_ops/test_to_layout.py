# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.to_layout`` (tilize / tilize_with_val_padding / untilize) and
``ttnn.untilize`` — every standalone layout conversion the OFT model ran.

Shape source (kernel cache, factories at ``1088b10b014``):
  * integral image -> ROW_MAJOR for grid_sample: float32 [1, 48, 160, 256] TILE DRAM ->
    ROW_MAJOR DRAM (``pack_untilize_variable_num_blocks/173317`` [8, 0, 16] with fp32 dest,
    writer stick 1024 B)                                                                    [H]
  * grid_sample output -> TILE: float32 [1, 1, 576, 1792] ROW_MAJOR HEIGHT_SHARDED (32, 1792)
    on the 5x4 grid -> TILE, same shard spec (``tilize/129205`` [1, 56], fp32 dest)           [H]
  * head output parts -> TILE: bf16 ROW_MAJOR L1 [1, 159, 159] / [1, 3, 159, 159] /
    [1, 2, 159, 159] -> TILE L1 (tilize_with_val_padding, reader row 318 B; ``tilize/522636``) [H]
  * NMS indices -> TILE: uint16 [1, 1, 25281, 16] ROW_MAJOR L1 -> TILE L1, padded to
    [.., 25312, 32] (791 tile-rows = 19 x 40 + 31: ``tilize`` 156927 / 166157)              [H]
  * conv-internal untilize before ``slice_write`` (head conv): bf16 TILE HEIGHT_SHARDED
    (640, 32) x 20 -> ROW_MAJOR, same shard spec (``pack_untilize/803521`` [20, 1, 0, 16])   [H]

Second cache (in-tree model at ``628a4797bf7``: bf16 throughout, 3 OFT scales, cumsum integral
image, on-device GroupNorm; factories at ``628a4797bf7``):
  * lateral features -> TILE after ``sharded_to_interleaved``: bf16 ROW_MAJOR DRAM
    [1, 1, H*W, 256] -> TILE DRAM with no padding (plain tilize, reader stick 512 B DRAM;
    lat8 7680 rows = ``tilize/388401`` [12, 8] on 20 cores, lat32 480 rows = ``423576`` [1, 8])  [H]
  * input image -> TILE: bf16 ROW_MAJOR DRAM [1, 384, 1280, 3] -> TILE (tilize_with_val_padding,
    6-B rows / 64-B aligned pages: ``reader_unary_pad_dims_split_rows_multicore/167465``,
    15360 tile-rows = ``tilize/366418`` [768, 1])                                          [H/M]
  * grid_sample output -> TILE in bf16: ROW_MAJOR HEIGHT_SHARDED (96, 1792) for the 18-slice
    1/8 scale (1408 rows, ``tilize/150546`` [3, 56]) and (128, 1792) for the 12/11-slice 1/16,
    1/32 scales (2112 / 2304 rows, ``171257`` [4, 56])                                      [H]
  * stem GroupNorm split (negative-mask path) back to TILE: bf16 ROW_MAJOR HEIGHT_SHARDED
    (6144, 32) = 122880 rows x 32 channels over 20 cores (``tilize/162864`` [192, 1])        [H/M]
  * integral image -> ROW_MAJOR in bf16: [1, 48, 160, 256] TILE DRAM (tile-aligned:
    ``untilize_variable_num_blocks/106193``), [1, 24, 80, 256] (W padded to 96:
    untilize_with_unpadding, 72 tile-rows = ``untilize/180509`` [4, 8])                     [H]
  * ortho -> ROW_MAJOR: bf16 TILE DRAM [1, 1, 25281, 256] (untilize_with_unpadding, 791
    tile-rows = 19 x 40 + 31: ``untilize`` 791191 / 261867, stick 512 B DRAM)               [H]
  * GroupNormDRAM input: ``tilize_with_val_padding`` of bf16 ROW_MAJOR DRAM [1, 1, 25281, 256]
    to padded [1, 1, 25344, 256] (792 tile-rows = 19 x 40 + 32: ``tilize`` 127508 / 412582)   [H]

Layout conversion is value-preserving: checked bit-exact.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

VOX = (1, 1, U.SLICE_ROWS, U.VOX_C)

# Second-cache (in-tree model) constants. Kept local; candidates for op_utils.
LAT_C = 256  # lateral / OFT / topdown channels in the in-tree model
# grid_sample rows per OFT slice: (25281 + 63) / num_slices for the 1/8, 1/16, 1/32 scales.
VOX8_ROWS, VOX16_ROWS = U.BEV_ROWS // 18, U.BEV_ROWS // 12  # 1408, 2112
STEM_NHW = U.STEM_H * U.STEM_W  # 122880 conv1 output rows


def _rand(shape, dt):
    if dt == "uint16":
        return torch.randint(0, U.BEV_NHW, shape, dtype=torch.int32)
    return U.torch_rand(shape, dtype=torch.float32 if dt == "fp32" else torch.bfloat16)


_DT = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32, "uint16": ttnn.uint16}

# fmt: off
# (shape, dtype, in_mem, out_mem)
TILIZE_CASES = [
    pytest.param((1, U.BEV, U.BEV), "bf16", "l1", "l1", id="head-score-1x159x159-522636"),
    pytest.param((1, 3, U.BEV, U.BEV), "bf16", "l1", "l1", id="head-pos-dim-1x3x159x159-522636"),
    pytest.param((1, 2, U.BEV, U.BEV), "bf16", "l1", "l1", id="head-ang-1x2x159x159-522636"),
    pytest.param((1, 1, U.BEV_NHW, 16), "uint16", "l1", "l1", id="nms-indices-uint16-156927"),
    pytest.param((1, 1, 64, U.VOX_C), "fp32", "l1", "l1", id="proxy-vox-fp32-l1"),
    # second cache (bf16, DRAM interleaved)
    pytest.param((1, 1, U.L2_H * U.L2_W, LAT_C), "bf16", "dram", "dram", id="lat8-bf16-dram-388401"),
    pytest.param((1, 1, 12 * 40, LAT_C), "bf16", "dram", "dram", id="lat32-bf16-dram-423576"),
    pytest.param((1, U.IMG_H, U.IMG_W, 3), "bf16", "dram", "dram", id="image-c3-bf16-dram-366418"),
    pytest.param((1, 32, 64, 3), "bf16", "dram", "dram", id="proxy-image-c3-bf16-dram"),
    pytest.param((1, 1, 96, U.VOX_C), "bf16", "l1", "l1", id="proxy-vox-bf16-l1"),
]
TILIZE_SHARDED_CASES = [
    pytest.param(VOX, "fp32", ("hs", (U.TILE, U.VOX_C)), ("hs", (U.TILE, U.VOX_C)), id="vox-fp32-hs-129205"),
    # second cache: bf16, multi-tile-row shards (partly empty grids: 15 / 17 of 20 shards used)
    pytest.param((1, 1, VOX8_ROWS, U.VOX_C), "bf16", ("hs", (96, U.VOX_C)), ("hs", (96, U.VOX_C)), id="vox8-bf16-hs-96x1792-150546"),
    pytest.param((1, 1, VOX16_ROWS, U.VOX_C), "bf16", ("hs", (128, U.VOX_C)), ("hs", (128, U.VOX_C)), id="vox16-bf16-hs-128x1792-171257"),
    pytest.param((1, 1, STEM_NHW, 32), "bf16", ("hs", (6144, 32)), ("hs", (6144, 32)), id="stem-gn-split-bf16-hs-6144x32-162864"),
]

UNTILIZE_CASES = [
    pytest.param((1, U.L2_H, U.L2_W, U.OFT_C), "fp32", "dram", "dram", id="intimg-fp32-1x48x160x256-173317"),
    pytest.param((1, 12, 40, U.OFT_C), "fp32", "dram", "dram", id="proxy-intimg-fp32-1x12x40x256"),
    # second cache (bf16): tile-aligned untilize and untilize_with_unpadding
    pytest.param((1, U.L2_H, U.L2_W, LAT_C), "bf16", "dram", "dram", id="intimg8-bf16-1x48x160x256-106193"),
    pytest.param((1, 24, 80, LAT_C), "bf16", "dram", "dram", id="intimg16-bf16-unpad-1x24x80x256-180509"),
    pytest.param((1, 1, U.BEV_NHW, LAT_C), "bf16", "dram", "dram", id="ortho-bf16-unpad-1x1x25281x256-261867"),
    pytest.param((1, 1, 1000, LAT_C), "bf16", "dram", "dram", id="proxy-ortho-bf16-unpad-1x1x1000x256"),
]
# fmt: on


def _run_tilize(mesh, shape, dt, in_mem, out_mem):
    x = _rand(shape, dt)
    tt_x = U.to_tt_mem(x, mesh, in_mem, dtype=_DT[dt], layout=ttnn.ROW_MAJOR_LAYOUT)
    out = U.op("to_layout")(tt_x, ttnn.TILE_LAYOUT, memory_config=U.resolve_memcfg(out_mem, mesh))
    assert out.layout == ttnn.TILE_LAYOUT
    U.assert_lossless(x, out, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize("shape, dt, in_mem, out_mem", TILIZE_CASES)
def test_to_layout_tile(ttnn_mesh_device, reset_seeds, shape, dt, in_mem, out_mem):
    _run_tilize(ttnn_mesh_device, shape, dt, in_mem, out_mem)


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape, dt, in_mem, out_mem", TILIZE_SHARDED_CASES)
def test_to_layout_tile_sharded(ttnn_mesh_device, reset_seeds, shape, dt, in_mem, out_mem):
    _run_tilize(ttnn_mesh_device, shape, dt, in_mem, out_mem)


@U.with_default_mesh()
@pytest.mark.parametrize("shape, dt, in_mem, out_mem", UNTILIZE_CASES)
def test_to_layout_row_major(ttnn_mesh_device, reset_seeds, shape, dt, in_mem, out_mem):
    mesh = ttnn_mesh_device
    x = _rand(shape, dt)
    tt_x = U.to_tt_mem(x, mesh, in_mem, dtype=_DT[dt])
    out = U.op("to_layout")(tt_x, ttnn.ROW_MAJOR_LAYOUT, memory_config=U.resolve_memcfg(out_mem, mesh))
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    U.assert_lossless(x, out, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, mem",
    [
        pytest.param(
            (1, 1, 20 * 640, 32),
            ("hs", (640, 32)),
            id="head-conv-out-hs-640x32-803521",
            marks=pytest.mark.blackhole_scale,
        ),
        pytest.param((1, 1, 640, 32), "l1", id="proxy-head-conv-out-l1"),
    ],
)
def test_untilize(ttnn_mesh_device, reset_seeds, shape, mem):
    """``ttnn.untilize`` keeping the input's memory config (sharded in the model)."""
    mesh = ttnn_mesh_device
    x = U.torch_rand(shape)
    tt_x = U.to_tt_mem(x, mesh, mem)
    out = U.op("untilize")(tt_x, memory_config=U.resolve_memcfg(mem, mesh))
    U.assert_lossless(x, out, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "in_shape, out_shape, mem",
    [
        pytest.param(
            (1, 1, U.BEV_NHW, LAT_C),
            (1, 1, U.BEV_ROWS, LAT_C),
            "dram",
            id="topdown-gn-in-25281->25344-412582",
        ),
        pytest.param((1, 1, 100, LAT_C), (1, 1, 128, LAT_C), "dram", id="proxy-gn-in-100->128"),
    ],
)
def test_tilize_with_val_padding(ttnn_mesh_device, reset_seeds, in_shape, out_shape, mem):
    """GroupNormDRAM's explicit ``tilize_with_val_padding`` to a per-core-rounded padded shape.

    ``_nearest_32_per_core(25281, 4)`` = 25344: padding goes past the next tile boundary
    (25312), so the op writes a whole extra tile-row of pad values.
    """
    mesh = ttnn_mesh_device
    x = U.torch_rand(in_shape)
    tt_x = U.to_tt_mem(x, mesh, mem, layout=ttnn.ROW_MAJOR_LAYOUT)
    out = U.op("tilize_with_val_padding")(tt_x, list(out_shape), 0.0, memory_config=U.resolve_memcfg(mem, mesh))
    assert out.layout == ttnn.TILE_LAYOUT
    assert tuple(out.padded_shape) == tuple(out_shape), f"padded shape {tuple(out.padded_shape)}"
    U.assert_lossless(x, out, mesh_device=mesh)

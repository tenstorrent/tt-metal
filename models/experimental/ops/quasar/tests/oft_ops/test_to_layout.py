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

Layout conversion is value-preserving: checked bit-exact.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

VOX = (1, 1, U.SLICE_ROWS, U.VOX_C)


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
]
TILIZE_SHARDED_CASES = [
    pytest.param(VOX, "fp32", ("hs", (U.TILE, U.VOX_C)), ("hs", (U.TILE, U.VOX_C)), id="vox-fp32-hs-129205"),
]

UNTILIZE_CASES = [
    pytest.param((1, U.L2_H, U.L2_W, U.OFT_C), "fp32", "dram", "dram", id="intimg-fp32-1x48x160x256-173317"),
    pytest.param((1, 12, 40, U.OFT_C), "fp32", "dram", "dram", id="proxy-intimg-fp32-1x12x40x256"),
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

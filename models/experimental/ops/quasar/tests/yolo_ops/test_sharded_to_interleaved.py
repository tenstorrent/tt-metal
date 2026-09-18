# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.sharded_to_interleaved`` (YOLOv8, 640x640, batch 1).

Every neck stage un-shards its conv output back to an interleaved L1/DRAM tensor
before the next reshape / concat:

    c2f_6 = ttnn.sharded_to_interleaved(c2f_6, ttnn.L1_MEMORY_CONFIG)

It is the inverse of ``interleaved_to_sharded`` and value-preserving, so the full
round-trip interleaved -> shard -> unshard must return the original (PCC 0.999).

Model call sites (branch ``origin/sdawle/yolov8_bh``, representative):
  * models/demos/yolov8l/tt/ttnn_yolov8l.py:65,72,76,88,106,131,265,319,609,
    1004,1042,1063,1081,1085,1088,1095,1099,1107
  * models/demos/yolov8s/tt/ttnn_yolov8s.py:68,105,235,737,774,802,806,809,816,820,829

Single-core grid, so the passed shape [1,1,32,C] is the exact per-core shard
shape. Neck channel widths are the standard YOLOv8l/s sizes (not load-bearing).
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

_GRID = ttnn.CoreGrid(y=1, x=1)


def _sharded_cfg(h, w, strategy):
    return ttnn.create_sharded_memory_config(
        (h, w),
        _GRID,
        strategy,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# (id, shape, strategy) — shard shape is (shape[-2], shape[-1]).
_CASES = [
    pytest.param((1, 1, U.TILE, 512), ttnn.ShardStrategy.WIDTH, id="l-width-32x512"),  # yolov8l neck
    pytest.param((1, 1, U.TILE, 256), ttnn.ShardStrategy.WIDTH, id="s-width-32x256"),  # yolov8s neck
    pytest.param((1, 1, U.TILE, 128), ttnn.ShardStrategy.HEIGHT, id="height-32x128"),  # height-shard round-trip
]


@U.with_default_mesh()
@pytest.mark.parametrize("shape, strategy", _CASES)
def test_sharded_to_interleaved(ttnn_mesh_device, reset_seeds, shape, strategy):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)  # interleaved DRAM

    memcfg = _sharded_cfg(shape[-2], shape[-1], strategy)
    x_sharded = ttnn.interleaved_to_sharded(x, memcfg)
    out = ttnn.sharded_to_interleaved(x_sharded, ttnn.L1_MEMORY_CONFIG)

    assert not out.is_sharded(), "expected an interleaved output"
    U.assert_lossless(x_torch, out, mesh_device=mesh)


# --- model-faithful: un-shard a HEIGHT-sharded concat output on the SPPF grid. ------------
#
# Every ``sharded_concat`` ends by un-sharding its HEIGHT-sharded output back to
# interleaved L1 (ttnn_yolov8l.py:65,88,131 ... ``sharded_to_interleaved(out, L1)``). The
# source is HEIGHT-sharded over the full SPPF grid (_SPPF_NUM_CORES = 64 / 80,
# ttnn_yolov8l.py:21-22,777-781) at the neck's flattened feature shape [1,1,H*W,C].
# Value-preserving, so interleaved -> HEIGHT-shard -> unshard must return the original.
@U.with_default_mesh()
@pytest.mark.blackhole_scale
@pytest.mark.parametrize(
    "variant, num_cores, shape",
    [
        # yolov8l: _SPPF_NUM_CORES=64 (8x8), sppf concat @40x40 -> 1600 // 64 = 25.
        pytest.param("yolov8l", 64, (1, 1, 1600, 512), id="l-height-64c-1600x512"),
        # yolov8s: _SPPF_NUM_CORES=80 (10x8), 1600 // 80 = 20.
        pytest.param("yolov8s", 80, (1, 1, 1600, 256), id="s-height-80c-1600x256"),
    ],
)
def test_sharded_to_interleaved_height(ttnn_mesh_device, reset_seeds, variant, num_cores, shape):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    sharded = U.height_sharded_memcfg(mesh, num_cores, shape)  # skips if device < num_cores
    x_sharded = ttnn.to_memory_config(x, sharded)
    assert x_sharded.is_sharded(), "expected a HEIGHT_SHARDED source"

    out = ttnn.sharded_to_interleaved(x_sharded, ttnn.L1_MEMORY_CONFIG)
    assert not out.is_sharded(), "expected an interleaved output"
    U.assert_lossless(x_torch, out, mesh_device=mesh)


# --- model-faithful: un-shard a BLOCK-sharded conv output. -------------------------------
#
# The ``block_shard=True`` convs (c2f_6 :1004, c2f_12 :1042, conv_16 :1085, conv_19 :1099,
# c2f_21 :1107) emit BLOCK-sharded activations that the neck un-shards to interleaved L1
# right after (ttnn_yolov8l.py:1004 ``sharded_to_interleaved(c2f_6, L1)``). We reproduce a
# BLOCK-sharded source on the compute grid and round-trip it. The exact conv block grid is
# chosen internally by conv2d and not recoverable here, so we use the full 8x8 grid (a
# documented best estimate; the helper skips if the device is smaller) with a flattened
# feature shape whose H*W and C divide the grid.
@U.with_default_mesh()
@pytest.mark.blackhole_scale
@pytest.mark.parametrize(
    "site, grid_y, grid_x, shape",
    [
        # c2f_6 @40x40, C=512 (configs.json c2f_configs["model.6"].cv2 out=512); 1600/8=200, 512/8=64.
        pytest.param("c2f_6", 8, 8, (1, 1, 1600, 512), id="c2f6-block-8x8-1600x512"),
        # c2f_21 @20x20, C=512 (configs.json c2f_configs["model.21"].cv2 out=512); 400/8=50, 512/8=64.
        pytest.param("c2f_21", 8, 8, (1, 1, 400, 512), id="c2f21-block-8x8-400x512"),
    ],
)
def test_sharded_to_interleaved_block(ttnn_mesh_device, reset_seeds, site, grid_y, grid_x, shape):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    sharded = U.block_sharded_memcfg(mesh, grid_y, grid_x, shape)  # skips if grid doesn't fit
    x_sharded = ttnn.to_memory_config(x, sharded)
    assert x_sharded.is_sharded(), "expected a BLOCK_SHARDED source"

    out = ttnn.sharded_to_interleaved(x_sharded, ttnn.L1_MEMORY_CONFIG)
    assert not out.is_sharded(), "expected an interleaved output"
    U.assert_lossless(x_torch, out, mesh_device=mesh)

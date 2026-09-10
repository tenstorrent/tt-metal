# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.interleaved_to_sharded`` (YOLOv8, 640x640, batch 1).

Before each upsample the neck moves an interleaved NHWC feature map into a
HEIGHT-sharded config:

    sppf_9 = ttnn.interleaved_to_sharded(sppf_9, shardspec)   # HEIGHT shard over N*H*W

It only re-lays-out data, so moving an interleaved tensor into a sharded config
and reading it back must preserve values (PCC 0.999).

Model call sites (branch ``origin/sdawle/yolov8_bh``):
  * models/demos/yolov8l/tt/ttnn_yolov8l.py:1027 (sppf_9), :1059 (c2f_12)
  * models/demos/yolov8s/tt/ttnn_yolov8s.py:758 (sppf_9), :789 (c2f_12)

Single-core grid, so the passed shape [1,1,32,C] is the exact per-core shard
shape. The model uses HEIGHT sharding here; a WIDTH case is added for coverage.
Neck channel widths are the standard YOLOv8l/s sizes (not load-bearing).
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


# (id, shape, strategy)
_CASES = [
    pytest.param((1, 1, U.TILE, 512), ttnn.ShardStrategy.HEIGHT, id="l-sppf-height-32x512"),  # ttnn_yolov8l.py:1027
    pytest.param((1, 1, U.TILE, 256), ttnn.ShardStrategy.HEIGHT, id="s-sppf-height-32x256"),  # ttnn_yolov8s.py:758
    pytest.param((1, 1, U.TILE, 512), ttnn.ShardStrategy.WIDTH, id="width-32x512"),  # width-shard coverage
]


@U.with_default_mesh()
@pytest.mark.parametrize("shape, strategy", _CASES)
def test_interleaved_to_sharded(ttnn_mesh_device, reset_seeds, shape, strategy):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)  # interleaved DRAM

    memcfg = _sharded_cfg(shape[-2], shape[-1], strategy)
    out = ttnn.interleaved_to_sharded(x, memcfg)

    assert out.is_sharded(), "expected a sharded output"
    U.assert_lossless(x_torch, out, mesh_device=mesh)


# --- model-faithful: interleaved L1 RM -> HEIGHT sharded on the real multi-core grid. -----
#
# The single-core case above is a shape-only stand-in. Before each neck upsample the model
# moves an interleaved L1 row-major feature map (reshaped to (batch, out_h, out_w, C)) into
# a HEIGHT-sharded config over a MULTI-core grid (ttnn_yolov8l.py:1017-1028 sppf_9,
# :1051-1061 c2f_12):
#
#     num_cores = determine_num_cores(nhw, sppf_9.shape[2])   # width arg = out_w, NOT C
#     shardspec = create_sharded_memory_config_(shape, get_core_grid_from_num_cores(...), HEIGHT)
#     sppf_9    = ttnn.interleaved_to_sharded(sppf_9, shardspec)
#
# determine_num_cores (yolo_utils.py:17) with width=out_w yields one image row per core:
# sppf_9 nhw=400,w=20 -> 20 cores; c2f_12 nhw=1600,w=40 -> 40 cores.
@U.with_default_mesh()
@pytest.mark.blackhole_scale
@pytest.mark.parametrize(
    "site, num_cores, shape",
    [
        # sppf_9 @20x20, C=512 (ttnn_yolov8l.py:1027); determine_num_cores(400, out_w=20)=20.
        pytest.param("sppf_9", 20, (1, 20, 20, 512), id="sppf9-height-20c-20x20x512"),
        # c2f_12 @40x40, C=512 (ttnn_yolov8l.py:1059); determine_num_cores(1600, out_w=40)=40.
        pytest.param("c2f_12", 40, (1, 40, 40, 512), id="c2f12-height-40c-40x40x512"),
    ],
)
def test_interleaved_to_sharded_multicore(ttnn_mesh_device, reset_seeds, site, num_cores, shape):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    memcfg = U.height_sharded_memcfg(mesh, num_cores, shape)  # skips if device < num_cores
    out = ttnn.interleaved_to_sharded(x, memcfg)

    assert out.is_sharded(), "expected a sharded output"
    U.assert_lossless(x_torch, out, mesh_device=mesh)

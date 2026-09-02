# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.reshard`` (YOLOv8, 640x640, batch 1).

When a pre-upsample feature map is already sharded, the neck reshards it into the
target HEIGHT-shard config instead of going through interleaved:

    if sppf_9.is_sharded():
        sppf_9 = ttnn.reshard(sppf_9, shardspec)     # sharded -> sharded (new grid)
    else:
        sppf_9 = ttnn.interleaved_to_sharded(sppf_9, shardspec)

``reshard`` moves shards between core grids without changing values, so a
round-trip interleaved -> shard(grid A) -> reshard(grid B) -> interleaved must
return the original (PCC 0.999).

Model call sites (branch ``origin/sdawle/yolov8_bh``):
  * models/demos/yolov8l/tt/ttnn_yolov8l.py:1025 (sppf_9), :1057 (c2f_12)
  * models/demos/yolov8s/tt/ttnn_yolov8s.py:756 (sppf_9), :787 (c2f_12)

Uses HEIGHT sharding of a [1,1,64,C] tensor across 2 cores, then reshards onto a
single core (shard height 32 -> 64). Requires a device with >=2 cores. Neck
channel widths are the standard YOLOv8l/s sizes (not load-bearing).
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U


def _height_sharded(core_grid, shard_h, shard_w):
    return ttnn.create_sharded_memory_config(
        (shard_h, shard_w),
        core_grid,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 64, 512), id="l-64x512"),  # ttnn_yolov8l.py:1025 (yolov8l neck)
        pytest.param((1, 1, 64, 256), id="s-64x256"),  # ttnn_yolov8s.py:756 (yolov8s neck)
    ],
)
def test_reshard(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device

    grid = mesh.compute_with_storage_grid_size()
    if grid.x * grid.y < 2:
        pytest.skip("reshard round-trip needs >=2 cores")

    h, w = shape[-2], shape[-1]
    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)  # interleaved DRAM

    # grid A: 2 cores (shard height h//2); grid B: 1 core (shard height h).
    cfg_a = _height_sharded(ttnn.CoreGrid(y=1, x=2), h // 2, w)
    cfg_b = _height_sharded(ttnn.CoreGrid(y=1, x=1), h, w)

    x_sharded = ttnn.interleaved_to_sharded(x, cfg_a)
    x_reshard = ttnn.reshard(x_sharded, cfg_b)
    assert x_reshard.is_sharded(), "expected a sharded output"

    out = ttnn.sharded_to_interleaved(x_reshard, ttnn.L1_MEMORY_CONFIG)
    U.assert_lossless(x_torch, out, mesh_device=mesh)


# --- model-faithful: HEIGHT-shard -> reshard to the real neck HEIGHT grid. ----------------
#
# When a pre-upsample feature map is already sharded, the neck reshards it (HEIGHT -> HEIGHT
# on a new grid) instead of going through interleaved (ttnn_yolov8l.py:1024-1028 sppf_9,
# :1056-1061 c2f_12):
#
#     num_cores = determine_num_cores(nhw, sppf_9.shape[2])   # width arg = out_w, NOT C
#     shardspec = create_sharded_memory_config_(shape, get_core_grid_from_num_cores(...), HEIGHT)
#     if sppf_9.is_sharded():
#         sppf_9 = ttnn.reshard(sppf_9, shardspec)
#
# The model target grids are 20 cores (sppf_9 nhw=400,w=20) and 40 cores (c2f_12 nhw=1600,
# w=40). The prior (source) shard grid isn't uniquely recoverable, so we shard onto a
# coarser grid first, then reshard onto the model's target grid; both grids divide nhw
# evenly (one/several image rows per core). Value-preserving, so the interleaved round-trip
# must return the original.
@U.with_default_mesh()
@pytest.mark.blackhole_scale
@pytest.mark.parametrize(
    "site, src_cores, dst_cores, shape",
    [
        # sppf_9 @20x20, C=512 (ttnn_yolov8l.py:1025); target determine_num_cores(400,20)=20.
        # source 10 cores (400/10=40 -> 2 rows/core) -> reshard to 20 cores (1 row/core).
        pytest.param("sppf_9", 10, 20, (1, 1, 400, 512), id="sppf9-reshard-10c-to-20c-400x512"),
        # c2f_12 @40x40, C=512 (ttnn_yolov8l.py:1057); target determine_num_cores(1600,40)=40.
        # source 20 cores (1600/20=80 -> 2 rows/core) -> reshard to 40 cores (1 row/core).
        pytest.param("c2f_12", 20, 40, (1, 1, 1600, 512), id="c2f12-reshard-20c-to-40c-1600x512"),
    ],
)
def test_reshard_multicore(ttnn_mesh_device, reset_seeds, site, src_cores, dst_cores, shape):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    cfg_src = U.height_sharded_memcfg(mesh, src_cores, shape)  # skips if device < src_cores
    cfg_dst = U.height_sharded_memcfg(mesh, dst_cores, shape)  # skips if device < dst_cores

    x_sharded = ttnn.interleaved_to_sharded(x, cfg_src)
    x_reshard = ttnn.reshard(x_sharded, cfg_dst)
    assert x_reshard.is_sharded(), "expected a sharded output"

    out = ttnn.sharded_to_interleaved(x_reshard, ttnn.L1_MEMORY_CONFIG)
    U.assert_lossless(x_torch, out, mesh_device=mesh)

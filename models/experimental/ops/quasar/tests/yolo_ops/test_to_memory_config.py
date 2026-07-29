# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.to_memory_config`` (YOLOv8, 640x640, batch 1).

YOLOv8 uses ``to_memory_config`` all over the neck to relocate activations
(DRAM<->L1) and to move interleaved tensors into sharded configs before conv /
concat. It only relocates data, so a round-trip must preserve values exactly (assert_lossless).

Model call sites (branch ``origin/sdawle/yolov8_bh``):
  * DRAM <-> L1 placement:
      - models/demos/yolov8l/tt/ttnn_yolov8l.py:74,90,321,447,451,459,524,611,987,1006,1015,1065,1112
      - models/demos/yolov8s/tt/ttnn_yolov8s.py:379,387,440,746,765
  * interleaved -> sharded (per-core conv input configs):
      - ttnn_yolov8l.py:55,119  to_memory_config(tensor, input_sharded_memory_configs[i])
      - ttnn_yolov8s.py:56,92
  * with dtype cast (anchors/strides -> bfloat8_b):
      - ttnn_yolov8l.py:751,754 ; ttnn_yolov8s.py:572,575

Neck activations are flattened feature maps [1, 1, H*W, C]; the placement cases
use those shapes, the interleaved->sharded cases use a per-tile-row width shard
[1,1,32,C] (single-core, so the shard shape equals the tensor). Neck channel
widths are the standard YOLOv8l/s sizes and are not load-bearing here.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

_GRID = ttnn.CoreGrid(y=1, x=1)


def _width_sharded(h, w):
    return ttnn.create_sharded_memory_config(
        (h, w),
        _GRID,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# --- DRAM <-> L1 placement round-trip (neck feature maps). -------------------
@U.with_default_mesh()
@pytest.mark.parametrize(
    "target_memcfg",
    [
        pytest.param(ttnn.L1_MEMORY_CONFIG, id="to-l1"),  # ttnn_yolov8l.py:451,987,1015
        pytest.param(ttnn.DRAM_MEMORY_CONFIG, id="to-dram"),  # ttnn_yolov8l.py:74,321,611,1065
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 400, 512), id="p5-400x512"),  # yolov8l P5
        pytest.param((1, 1, 1600, 256), id="p4-1600x256"),  # yolov8s neck
    ],
)
def test_to_memory_config_placement(ttnn_mesh_device, reset_seeds, target_memcfg, shape):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    source_memcfg = ttnn.DRAM_MEMORY_CONFIG if target_memcfg == ttnn.L1_MEMORY_CONFIG else ttnn.L1_MEMORY_CONFIG
    x = U.to_tt(x_torch, mesh, memory_config=source_memcfg)
    out = ttnn.to_memory_config(x, target_memcfg)
    U.assert_lossless(x_torch, out, mesh_device=mesh)


# --- interleaved -> WIDTH-sharded move (per-core conv input config). ---------
@U.with_default_mesh()
@pytest.mark.parametrize(
    "width",
    [
        pytest.param(512, id="width-512"),  # yolov8l neck channels
        pytest.param(256, id="width-256"),  # yolov8s neck channels
        pytest.param(128, id="width-128"),  # yolov8s shallow neck channels
    ],
)
def test_to_memory_config_interleaved_to_sharded(ttnn_mesh_device, reset_seeds, width):
    mesh = ttnn_mesh_device
    shape = (1, 1, U.TILE, width)
    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)  # interleaved DRAM
    out = ttnn.to_memory_config(x, _width_sharded(U.TILE, width))
    assert out.is_sharded(), "expected a sharded output"
    U.assert_lossless(x_torch, out, mesh_device=mesh)


# --- model-faithful: interleaved L1 RM -> HEIGHT_SHARDED on the SPPF core grid. -----------
#
# The single-core WIDTH shard above is a shape-only stand-in. The model's real
# ``to_memory_config`` sharding happens inside ``sharded_concat`` /
# ``sharded_concat_sppf`` (ttnn_yolov8l.py:55, :119): each concat input is a flattened
# neck feature map [1, 1, H*W, C] moved from interleaved L1 into a HEIGHT-sharded config
# over the full SPPF grid — ``_SPPF_NUM_CORES`` = 64 (yolov8l, 8x8) / 80 (yolov8s, 10x8)
# (ttnn_yolov8l.py:21-22, :777-781). Value-preserving relocation, so PCC vs the original.
@U.with_default_mesh()
@pytest.mark.blackhole_scale
@pytest.mark.parametrize(
    "variant, num_cores, shape",
    [
        # yolov8l sharded_concat: _SPPF_NUM_CORES=64 (8x8); sppf concat @40x40 -> H*W=1600.
        # shard_height = 1600 // 64 = 25 (ttnn_yolov8l.py:55,119).
        pytest.param("yolov8l", 64, (1, 1, 1600, 512), id="l-height-64c-1600x512"),
        # yolov8s sharded_concat: _SPPF_NUM_CORES=80 (10x8); 1600 // 80 = 20 (ttnn_yolov8s.py:56,92).
        pytest.param("yolov8s", 80, (1, 1, 1600, 256), id="s-height-80c-1600x256"),
    ],
)
def test_to_memory_config_to_height_sharded(ttnn_mesh_device, reset_seeds, variant, num_cores, shape):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    sharded = U.height_sharded_memcfg(mesh, num_cores, shape)  # skips if device < num_cores
    out = ttnn.to_memory_config(x, sharded)
    assert out.is_sharded(), "expected a HEIGHT_SHARDED output"
    U.assert_lossless(x_torch, out, mesh_device=mesh)


# Fused relocation + dtype cast: to_memory_config(..., dtype=bfloat8_b). The model casts
# the detect-head anchors/strides to bfloat8_b *during* the L1 relocation
# (ttnn_yolov8l.py:751,754 / ttnn_yolov8s.py:572,575) — a distinct overload from the
# plain relocation and from the separate ttnn.to_dtype path.
@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 2, 8400), id="anchors-640"),
        pytest.param((1, 1, 8400), id="strides-640"),
    ],
)
def test_to_memory_config_dtype_cast(ttnn_mesh_device, reset_seeds, shape):
    """relocation fused with bf16->bfloat8_b cast (ttnn_yolov8l.py:751,754)."""
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    # Source in DRAM (tilize into DRAM works for these odd shapes). The cast must ride a
    # REAL relocation: an L1->L1 no-op short-circuits and ignores dtype, leaving bf16.
    x = U.to_tt(x_torch, mesh, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    out = ttnn.to_memory_config(x, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

    assert out.dtype == ttnn.bfloat8_b, f"expected bfloat8_b, got {out.dtype}"
    U.assert_pcc(x_torch, out, pcc=0.99, mesh_device=mesh)  # 0.99: bfloat8_b precision

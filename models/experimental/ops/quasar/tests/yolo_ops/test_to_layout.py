# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.to_layout`` (YOLOv8, 640x640, batch 1).

YOLOv8 flips activations between ROW_MAJOR and TILE layouts around the neck
concats / detect head:

    tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)                 # to RM
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=L1, dtype=bf8)   # to TILE (+bf8)

``to_layout`` is a value-preserving re-tiling, so a TILE<->ROW_MAJOR round-trip
must return the original values. The plain bf16 round-trip is checked at PCC
0.999; the bf8 TILE variant (matching :327 which also casts to bfloat8_b) at 0.99.

Model call sites (branch ``origin/sdawle/yolov8_bh``):
  * models/demos/yolov8l/tt/ttnn_yolov8l.py:35   to_layout(tensor, ROW_MAJOR_LAYOUT)
  * ttnn_yolov8l.py:327  to_layout(x, TILE_LAYOUT, memory_config=L1, dtype=bfloat8_b)
  * models/demos/yolov8s/tt/ttnn_yolov8s.py:441,747,764,775,794,807,821,822  to_layout(..., ROW_MAJOR_LAYOUT)
  * ttnn_yolov8s.py:288  to_layout(x, TILE_LAYOUT, memory_config=L1, dtype=bfloat8_b)

Tensors are flattened neck feature maps [1, 1, H*W, C]; H==W parametrized by
``hw`` (20/40 emulator, 80 real-BH). Neck channels are the standard YOLOv8l/s
sizes and are not load-bearing (to_layout is channel-independent).
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

_FMAPS = [
    (20, 512, "l_p5_20x512"),
    (40, 256, "s_p4_40x256"),
    (80, 256, "l_p3_80x256"),
]


# TILE -> ROW_MAJOR -> TILE round-trip (bf16). ttnn_yolov8l.py:35 / ttnn_yolov8s.py:441
@U.with_default_mesh()
@pytest.mark.parametrize("hw, c", [pytest.param(hw, c, id=i) for (hw, c, i) in _FMAPS])
def test_to_layout_tile_rowmajor_roundtrip(ttnn_mesh_device, reset_seeds, hw, c):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand((1, 1, hw * hw, c))
    x = U.to_tt(x_torch, mesh)  # TILE, bf16

    rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    assert rm.layout == ttnn.ROW_MAJOR_LAYOUT
    back = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    U.assert_lossless(x_torch, back, mesh_device=mesh)


# ROW_MAJOR -> TILE (+ bf8 cast) into L1. ttnn_yolov8l.py:327 / ttnn_yolov8s.py:288
@U.with_default_mesh()
@pytest.mark.parametrize("hw, c", [pytest.param(hw, c, id=i) for (hw, c, i) in _FMAPS])
def test_to_layout_rowmajor_to_tile_bf8(ttnn_mesh_device, reset_seeds, hw, c):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand((1, 1, hw * hw, c))
    x = U.to_tt(x_torch, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    assert out.layout == ttnn.TILE_LAYOUT

    U.assert_pcc(x_torch, out, pcc=0.99, mesh_device=mesh)  # 0.99: bfloat8_b block-float cast

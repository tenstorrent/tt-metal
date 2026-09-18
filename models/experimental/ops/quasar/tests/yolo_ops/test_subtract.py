# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.subtract``  (YOLOv8 dist2bbox math).

Model call sites:
  * x1y1 = anchor_points - lt   (tt_yolov8l_utils.py:94 / tt_yolov8s_utils.py:92)
        via `-` operator -> ttnn.subtract; [1,2,A] - [1,2,A]
  * wh   = ttnn.subtract(x2y2, x1y1, dtype=bfloat8_b, ...)
        (tt_yolov8l_utils.py:99 / tt_yolov8s_utils.py:97); [1,2,A] - [1,2,A]

``lt, rb = ttnn.split(distance, 2, 1)`` -> each [1,2,A]; ``anchor_points`` /
``x1y1`` / ``x2y2`` are all [1,2,A] (make_anchors ``a``, tt_yolov8l_utils.py:80).

A = total detect anchors: 8400 @ 640, 33600 @ 1280 (yolov8l only). dist2bbox is
identical for yolov8l/yolov8s. Model uses bfloat8_b; we use default bfloat16.
Reference: torch a - b.  PCC 0.999.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

# All subtract operands are [1, 2, A]. shape tuple param -> element-count classification.
_SITES = [
    pytest.param((1, 2, 8400), id="640"),  # yolov8l & yolov8s
    pytest.param((1, 2, 33600), id="1280"),  # yolov8l only
]


@U.with_default_mesh()
@pytest.mark.parametrize("shape", _SITES)
def test_subtract(ttnn_mesh_device, reset_seeds, shape):
    """anchor-lt / x2y2-x1y1 (tt_yolov8l_utils.py:94,99)."""
    mesh = ttnn_mesh_device

    a_torch = U.torch_rand(shape)
    b_torch = U.torch_rand(shape)
    a = U.to_tt(a_torch, mesh)
    b = U.to_tt(b_torch, mesh)

    out = ttnn.subtract(a, b)

    ref = a_torch.float() - b_torch.float()
    U.assert_pcc(ref, out, pcc=0.999, mesh_device=mesh)


# ---------------------------------------------------------------------------
# Model-faithful (class C): L1-interleaved TILE inputs. dist2bbox runs inside the
# detect head; its operands live on _DETECT_MEM_CONFIG = L1_MEMORY_CONFIG for
# res <= 640 (ttnn_yolov8l.py:772-773). Same shapes as test_subtract, DRAM -> L1.
# ---------------------------------------------------------------------------
@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape", _SITES)
def test_subtract_l1(ttnn_mesh_device, reset_seeds, shape):
    """anchor-lt / x2y2-x1y1 on L1 detect buffers (tt_yolov8l_utils.py:94,99; mem cfg ttnn_yolov8l.py:772-773)."""
    mesh = ttnn_mesh_device

    a_torch = U.torch_rand(shape)
    b_torch = U.torch_rand(shape)
    a = U.to_tile_l1(a_torch, mesh)
    b = U.to_tile_l1(b_torch, mesh)

    out = ttnn.subtract(a, b, dtype=ttnn.bfloat8_b)

    ref = a_torch.float() - b_torch.float()
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)

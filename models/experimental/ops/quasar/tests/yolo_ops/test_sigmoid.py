# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.sigmoid``  (YOLOv8 detect head).

Model call site:
  * yolov8l  ttnn_yolov8l.py:757  return [ttnn.concat((dbox, ttnn.sigmoid(cls)), dim=1, ...), x]
  * yolov8s  ttnn_yolov8s.py:578  (identical detect head)

``cls`` is the class-logits half of the concatenated detect output, sliced out at
ttnn_yolov8l.py:748 (``cls = ttnn.slice(x_cat, [0, 64, 0], [1, 144, A])``), i.e.
shape ``[1, nc=80, A]`` where ``A`` is the total anchor count over the 3 detect
scales (80/40/20 for 640x640 -> 8400; 160/80/40 for 1280x1280 -> 33600).

nc/reg_max come from TtDetect.__call__ (yolov8l.py:725-729: nc=80, no=nc+reg_max*4).
The detect head is IDENTICAL for yolov8l and yolov8s; only yolov8l also runs at
1280 (test_yolov8l.py parametrizes input_res [640, 1280]); yolov8s is 640 only
(test_yolov8s.py: torch.rand((1, 3, 640, 640))).

Reference: torch.sigmoid.  PCC 0.99.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

# Total detect-head anchor count A = sum(h*w) over the 3 scales, per input res.
#   640  -> 80^2 + 40^2 + 20^2  = 8400
#   1280 -> 160^2 + 80^2 + 40^2 = 33600
_NC = 80  # yolov8{l,s} TtDetect nc (ttnn_yolov8l.py:662 / ttnn_yolov8s.py:519)

# (id, cls_shape) — cls = [1, nc, A]. shape tuple param so conftest classifies by
# element count (only the 640 case is small enough for the emulator).
_CLS_SITES = [
    pytest.param((1, _NC, 8400), id="l_s-640-cls"),  # yolov8l & yolov8s @ 640
    pytest.param((1, _NC, 33600), id="l-1280-cls"),  # yolov8l @ 1280 only
]


@U.with_default_mesh()
@pytest.mark.parametrize("shape", _CLS_SITES)
def test_sigmoid(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)

    out = ttnn.sigmoid(x)

    ref = torch.sigmoid(x_torch.float())
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)


# ---------------------------------------------------------------------------
# Model-faithful (class C): L1-interleaved TILE input, matching the real detect
# head buffer state. TtDetectionModel sets _DETECT_MEM_CONFIG = L1_MEMORY_CONFIG
# for res <= 640 (ttnn_yolov8l.py:772-773); the sigmoid at ttnn_yolov8l.py:757
# runs on _DETECT_MEM_CONFIG, i.e. L1 (not DRAM) at 640. Same shapes as
# test_sigmoid, only the input memory_config changes DRAM -> L1.
# ---------------------------------------------------------------------------
@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape", _CLS_SITES)
def test_sigmoid_l1(ttnn_mesh_device, reset_seeds, shape):
    """sigmoid(cls) on L1 detect buffer (ttnn_yolov8l.py:757, mem cfg :772-773)."""
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tile_l1(x_torch, mesh)

    out = ttnn.sigmoid(x)

    ref = torch.sigmoid(x_torch.float())
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)

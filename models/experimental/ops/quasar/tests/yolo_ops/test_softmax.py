# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.softmax``  (YOLOv8 detect-head DFL).

Model call site (Distribution Focal Loss decode):
  * yolov8l  ttnn_yolov8l.py:644-645
        x = ttnn.reshape(x, (b, 4, c1, a), ...)   # c1 = reg_max = 16
        x = ttnn.softmax(x, dim=2, ...)           # softmax over the reg_max axis
  * yolov8s  ttnn_yolov8s.py:507-508  (identical)

The DFL input is ``box`` = ``[1, 64, A]`` (ttnn_yolov8l.py:747, 64 = reg_max*4),
reshaped to ``[b, 4, c1=16, a=A]`` (TtDFL.__call__, c1=16 default at :642) and
softmaxed over dim=2 (the reg_max=16 axis). ``A`` = total detect anchors
(8400 @ 640, 33600 @ 1280). Detect head is identical for yolov8l/yolov8s; only
yolov8l also runs 1280.

Note: reg_max=16 is not tile-aligned; ttnn pads to 32 in TILE layout but tracks the
logical width, so the softmax reduction must respect the logical 16. Reference is
torch.softmax over the same logical shape.  PCC 0.99.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

_REG_MAX = 16  # c1 (TtDFL.__call__ default, ttnn_yolov8l.py:642)

# (id, dfl_shape) — [b, 4, reg_max, A]; softmax dim=2. shape tuple param -> conftest
# classifies by element count (both are large -> Blackhole, not emulator).
_DFL_SITES = [
    pytest.param((1, 4, _REG_MAX, 8400), id="l_s-640-dfl"),  # yolov8l & yolov8s @ 640
    pytest.param((1, 4, _REG_MAX, 33600), id="l-1280-dfl"),  # yolov8l @ 1280 only
]


@U.with_default_mesh()
@pytest.mark.parametrize("shape", _DFL_SITES)
def test_softmax(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)

    out = ttnn.softmax(x, dim=2)

    ref = torch.softmax(x_torch.float(), dim=2)
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)


# ---------------------------------------------------------------------------
# Model-faithful (class C): L1-interleaved TILE input, matching the real DFL
# buffer state. The DFL softmax at ttnn_yolov8l.py:645 runs on _DETECT_MEM_CONFIG,
# which TtDetectionModel pins to L1_MEMORY_CONFIG for res <= 640
# (ttnn_yolov8l.py:772-773). Same shapes as test_softmax, DRAM -> L1 input only.
# ---------------------------------------------------------------------------
@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape", _DFL_SITES)
def test_softmax_l1(ttnn_mesh_device, reset_seeds, shape):
    """DFL softmax(dim=2) on L1 detect buffer (ttnn_yolov8l.py:645, mem cfg :772-773)."""
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tile_l1(x_torch, mesh)

    out = ttnn.softmax(x, dim=2)

    ref = torch.softmax(x_torch.float(), dim=2)
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.reallocate`` (YOLOv8, 640x640, batch 1).

After padding the stem input to 16 channels and permuting NCHW->NHWC, the model
defragments the tensor before reshaping it into the conv input:

    nhwc = ttnn.permute(nchw, (0, 2, 3, 1))   # NCHW -> NHWC
    nhwc = ttnn.reallocate(nhwc)

``reallocate`` just moves a tensor to a fresh (defragmented) allocation; it is a
pure value-preserving relocation, so the result must equal the input (PCC 0.999).

Model call site (branch ``origin/sdawle/yolov8_bh``):
  * models/demos/yolov8s/tt/ttnn_yolov8s.py:717
    (yolov8l performs the same NCHW->NHWC prep at ttnn_yolov8l.py:986-987 via
    permute + to_memory_config; reallocate is the yolov8s spelling.)

The tensor is the row-major NHWC stem input (1, H, W, 16). H==W is a spatial
feature map, parametrized by ``hw`` (20 fits the emulator, 640 is the real stem
resolution and runs on real Blackhole).
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize("hw", [20, 640])
def test_reallocate(ttnn_mesh_device, reset_seeds, hw):
    """ttnn_yolov8s.py:717 — reallocate the NHWC stem input (1, hw, hw, 16)."""
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand((1, hw, hw, U.IMG_CH_PADDED))
    x = U.to_tt(x_torch, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.reallocate(x)

    U.assert_lossless(x_torch, out, mesh_device=mesh)

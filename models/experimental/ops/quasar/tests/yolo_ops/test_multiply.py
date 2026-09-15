# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.multiply``  (YOLOv8 detect-head box de-scaling).

Model call site:
  * yolov8l  ttnn_yolov8l.py:755   dbox = ttnn.multiply(dbox, strides, dtype=bfloat8_b)
  * yolov8s  ttnn_yolov8s.py:576   (identical)

``dbox`` is the decoded box tensor ``[1, 4, A]`` (dist2bbox output,
tt_yolov8l_utils.py:100). ``strides`` is make_anchors' ``b`` = per-anchor stride,
``torch.cat(stride_tensor).transpose(0,1)`` -> logical ``[1, A]``
(tt_yolov8l_utils.py:81,85); it broadcasts across the 4 box coords. Here strides is
built as ``[1, 1, A]`` (rank-aligned equivalent of the model's ``[1, A]``) so the
broadcast over dim=1 (4 vs 1) is unambiguous.

A = total detect anchors: 8400 @ 640, 33600 @ 1280 (yolov8l only). Detect head is
identical for yolov8l/yolov8s. Model uses bfloat8_b; we use default bfloat16.
Reference: torch broadcast multiply.  PCC 0.99.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

# dbox=[1,4,A], strides=[1,1,A]. shape tuple param -> element-count classification.
_SITES = [
    pytest.param((1, 4, 8400), id="640"),  # yolov8l & yolov8s
    pytest.param((1, 4, 33600), id="1280"),  # yolov8l only
]


@U.with_default_mesh()
@pytest.mark.parametrize("shape", _SITES)
def test_multiply_dbox_strides(ttnn_mesh_device, reset_seeds, shape):
    """dbox * strides (ttnn_yolov8l.py:755)."""
    mesh = ttnn_mesh_device
    a = shape[-1]

    dbox_torch = U.torch_rand(shape)  # [1, 4, A]
    strides_torch = U.torch_rand((1, 1, a))  # broadcasts over the 4 coords
    dbox = U.to_tt(dbox_torch, mesh)
    strides = U.to_tt(strides_torch, mesh)

    out = ttnn.multiply(dbox, strides)

    ref = dbox_torch.float() * strides_torch.float()
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)


# ---------------------------------------------------------------------------
# Model-faithful (class C): L1-interleaved TILE inputs, matching the real detect
# head buffer state. dbox * strides (ttnn_yolov8l.py:755) runs inside the detect
# head, whose tensors live on _DETECT_MEM_CONFIG = L1_MEMORY_CONFIG for res <= 640
# (ttnn_yolov8l.py:772-773). Same shapes as test_multiply_dbox_strides, DRAM -> L1.
# ---------------------------------------------------------------------------
@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape", _SITES)
def test_multiply_l1(ttnn_mesh_device, reset_seeds, shape):
    """dbox * strides on L1 detect buffers (ttnn_yolov8l.py:755, mem cfg :772-773)."""
    mesh = ttnn_mesh_device
    a = shape[-1]

    dbox_torch = U.torch_rand(shape)  # [1, 4, A]
    strides_torch = U.torch_rand((1, 1, a))  # broadcasts over the 4 coords
    dbox = U.to_tile_l1(dbox_torch, mesh)
    strides = U.to_tile_l1(strides_torch, mesh)

    out = ttnn.multiply(dbox, strides, dtype=ttnn.bfloat8_b)

    ref = dbox_torch.float() * strides_torch.float()
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.div``  (YOLOv8 dist2bbox center computation).

Model call site:
  * tt_yolov8l_utils.py:98   c_xy = ttnn.div(c_xy, 2, dtype=bfloat8_b, ...)
  * tt_yolov8s_utils.py:96   c_xy = ttnn.div(c_xy, 2, dtype=bfloat8_b)

``c_xy = x1y1 + x2y2`` is the summed box-center tensor ``[1, 2, A]``
(tt_yolov8l_utils.py:97), divided by the python scalar 2 to get the mean center.

A = total detect anchors: 8400 @ 640, 33600 @ 1280 (yolov8l only). dist2bbox is
identical for yolov8l/yolov8s. Model uses bfloat8_b; we use default bfloat16.
Reference: torch x / 2.  PCC 0.99.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

# c_xy = [1, 2, A]. shape tuple param -> element-count classification.
_SITES = [
    pytest.param((1, 2, 8400), id="640"),  # yolov8l & yolov8s
    pytest.param((1, 2, 33600), id="1280"),  # yolov8l only
]


@U.with_default_mesh()
@pytest.mark.parametrize("shape", _SITES)
def test_div_scalar(ttnn_mesh_device, reset_seeds, shape):
    """c_xy / 2 (tt_yolov8l_utils.py:98)."""
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)

    out = U.op("div")(x, 2)

    ref = x_torch.float() / 2
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)


# ---------------------------------------------------------------------------
# Model-faithful (class C): L1-interleaved TILE input. dist2bbox runs inside the
# detect head; its c_xy tensor lives on _DETECT_MEM_CONFIG = L1_MEMORY_CONFIG for
# res <= 640 (ttnn_yolov8l.py:772-773). Same shapes as test_div_scalar, DRAM -> L1.
# ---------------------------------------------------------------------------
@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize("shape", _SITES)
def test_div_l1(ttnn_mesh_device, reset_seeds, shape):
    """c_xy / 2 on L1 detect buffer (tt_yolov8l_utils.py:98, mem cfg ttnn_yolov8l.py:772-773)."""
    U.skip_bfp_on_quasar()  # model's bf8 output dtype; no block-float formats on Quasar
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tile_l1(x_torch, mesh)

    out = U.op("div")(x, 2, dtype=ttnn.bfloat8_b)
    assert out.dtype == ttnn.bfloat8_b, f"expected bfloat8_b, got {out.dtype}"

    ref = x_torch.float() / 2
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)


# ---------------------------------------------------------------------------
# Regression for the fork's output dtype forwarding (#62730): the model asks ``div`` for a
# dtype, and the Quasar composite used to drop it on every path. bfloat16 inputs with a float32
# output are supported on every target, so this runs on Quasar as well as WH/BH; the bfloat8_b
# model case above only checks the dtype where the format exists.
# ---------------------------------------------------------------------------
@U.with_default_mesh()
@pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"], ids=["none", "floor", "trunc"])
@pytest.mark.parametrize("by_tensor", [False, True], ids=["scalar", "tensor"])
def test_div_output_dtype(ttnn_mesh_device, reset_seeds, by_tensor, rounding_mode):
    """div(..., dtype=float32) returns float32 for scalar/tensor divisors and every rounding mode."""
    mesh = ttnn_mesh_device
    shape = (1, 2, 8400)

    x_torch = U.torch_rand(shape) * 8
    x = U.to_tt(x_torch, mesh)
    if by_tensor:
        d_torch = U.torch_rand(shape).abs() + 0.5  # divisor away from zero
        out = U.op("div")(x, U.to_tt(d_torch, mesh), rounding_mode=rounding_mode, dtype=ttnn.float32)
        ref = x_torch.float() / d_torch.float()
    else:
        out = U.op("div")(x, 2, rounding_mode=rounding_mode, dtype=ttnn.float32)
        ref = x_torch.float() / 2
    assert out.dtype == ttnn.float32, f"expected float32, got {out.dtype}"

    if rounding_mode == "floor":
        ref = ref.floor()
    elif rounding_mode == "trunc":
        ref = ref.trunc()
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)

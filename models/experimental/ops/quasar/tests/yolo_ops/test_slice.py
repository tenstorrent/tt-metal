# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.slice`` (YOLOv8, 640x640, batch 1).

The detect head splits the concatenated per-anchor prediction tensor
``x_cat`` [1, 144, 8400] along the channel dim into the 64 box-regression
channels and the 80 class channels:

    box = ttnn.slice(x_cat, [0, 0, 0],  [1, 64, N])
    cls = ttnn.slice(x_cat, [0, 64, 0], [1, 144, N])

``ttnn.slice`` end coords are exclusive, so this is a plain channel narrow.
Value-preserving -> torch reference is ``x[..., start:end, :]``; PCC 0.999.
Uploaded ROW_MAJOR so the narrow is exact.

Model call sites (branch ``origin/sdawle/yolov8_bh``):
  * models/demos/yolov8l/tt/ttnn_yolov8l.py:747 (box), :748 (cls)
  * models/demos/yolov8s/tt/ttnn_yolov8s.py:568 (box), :569 (cls)

Detect head at 640 input -> anchor count 8400 (80*80+40*40+20*20),
no = nc + reg_max*4 = 80 + 64 = 144.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

_N = 8400  # anchor count at 640x640

# (id, start_c, end_c) — channel slice of x_cat [1, 144, 8400].
_SLICE_SITES = [
    ("box_0_64", 0, 64),  # ttnn_yolov8l.py:747 / ttnn_yolov8s.py:568
    ("cls_64_144", 64, 144),  # ttnn_yolov8l.py:748 / ttnn_yolov8s.py:569
]


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, start_c, end_c",
    [pytest.param(*s, id=s[0]) for s in _SLICE_SITES],
)
def test_slice(ttnn_mesh_device, reset_seeds, name, start_c, end_c):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand((1, 144, _N))
    x = U.to_tt(x_torch, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.slice(x, [0, start_c, 0], [1, end_c, _N])

    ref = x_torch.float()[0:1, start_c:end_c, 0:_N]
    U.assert_lossless(ref, out, mesh_device=mesh)


# =============================================================================
# Model-faithful variant: reproduce the model's REAL layout / buffer state.
# =============================================================================


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, start_c, end_c",
    [pytest.param(*s, id=s[0] + "_tile") for s in _SLICE_SITES],
)
def test_slice_tile(ttnn_mesh_device, reset_seeds, name, start_c, end_c):
    """Model-faithful ``ttnn.slice`` on TILE / L1 — the detect box/cls channel split.

    The detect head slices ``x_cat`` while it is in TILE_LAYOUT, passing
    ``memory_config=_DETECT_MEM_CONFIG`` (== ttnn.L1_MEMORY_CONFIG at 640x640):

        box = ttnn.slice(x_cat, [0, 0, 0],  [1, 64, N],  memory_config=_DETECT_MEM_CONFIG)  # :747
        cls = ttnn.slice(x_cat, [0, 64, 0], [1, 144, N], memory_config=_DETECT_MEM_CONFIG)  # :748

    This narrows the channel dim (dim 1) of a TILE-padded, non-tile-aligned (N=8400)
    tensor. The box slice (0:64) and cls slice (64:144) both start on a 32-aligned
    channel boundary. Slice is value-preserving; reference is the torch narrow at
    PCC 0.999. The existing ``test_slice`` covers the ROW_MAJOR path.

    Model call sites (branch ``origin/sdawle/yolov8_bh``):
      * models/demos/yolov8l/tt/ttnn_yolov8l.py:747 (box), :748 (cls)
        (same at models/demos/yolov8s/tt/ttnn_yolov8s.py:568 / :569)
    """
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand((1, 144, _N))
    x = U.to_tile_l1(x_torch, mesh)

    out = ttnn.slice(x, [0, start_c, 0], [1, end_c, _N])

    ref = x_torch.float()[0:1, start_c:end_c, 0:_N]
    U.assert_lossless(ref, out, mesh_device=mesh)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.reshape`` (YOLOv8, 640x640, batch 1).

``ttnn.reshape`` in YOLOv8 only folds / unfolds axes (DFL reg reshape, detect
flatten, NHWC<->flattened neck reshapes, bias reshape); it is value-preserving,
so the torch reference is ``torch.reshape``. Uploaded ROW_MAJOR so no tile
padding perturbs the flat value comparison.

Model call sites (branch ``origin/sdawle/yolov8_bh``):
  * models/demos/yolov8l/tt/ttnn_yolov8l.py:644  DFL reg reshape (b,64,a)->(b,4,c1,a), c1=16
    (same op at models/demos/yolov8s/tt/ttnn_yolov8s.py:507)
  * ttnn_yolov8l.py:650  DFL post-conv reshape (b,1,4,a4)
  * ttnn_yolov8l.py:651  DFL final reshape (b, 1*4, a)
  * ttnn_yolov8l.py:741  detect flatten x[i] -> (b, -1, no)  (no = nc+reg_max*4 = 144)
    (same at ttnn_yolov8s.py:562)
  * ttnn_yolov8l.py:1017 SPPF NHWC reshape (b, out_h, out_w, C)
    (same at ttnn_yolov8s.py:748)
  * ttnn_yolov8l.py:1031 post-upsample flatten (1,1,b*H*W,C)
    (same at ttnn_yolov8s.py:762)
  * models/demos/yolov8l/tt/tt_yolov8l_utils.py:110 conv-bias reshape (C,)->(1,1,1,-1)
    (same at tt_yolov8s_utils.py:108)

Detect head runs at feature-map sizes {80,40,20} for a 640 input, so the anchor
count is 80*80+40*40+20*20 = 8400 and no = 80 + 16*4 = 144.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

# (id, in_shape, out_shape). -1 is resolved by both torch.reshape and ttnn.reshape.
# Neck channel widths are the standard YOLOv8l/s neck sizes; reshape is
# channel-independent so the exact value is not load-bearing.
_RESHAPE_SITES = [
    # ttnn_yolov8l.py:644 — DFL reg reshape (1,64,8400) -> (1,4,16,8400)
    ("dfl_reg_split", (1, 64, 8400), (1, 4, 16, 8400)),
    # ttnn_yolov8l.py:650 — DFL post-conv reshape (1,1,4,8400) -> (1,1,4,8400) width-fold form
    ("dfl_post_conv", (1, 1, 4, 8400), (1, 1, 4, 8400)),
    # ttnn_yolov8l.py:651 — DFL final reshape (1,1,4,8400) -> (1,4,8400)
    ("dfl_final", (1, 1, 4, 8400), (1, 4, 8400)),
    # ttnn_yolov8l.py:741 — detect flatten per scale (p3/p4/p5): [1,1,H*W,144] -> (1,H*W,144)
    ("detect_flatten_p3", (1, 1, 6400, 144), (1, 6400, 144)),
    ("detect_flatten_p4", (1, 1, 1600, 144), (1, 1600, 144)),
    ("detect_flatten_p5", (1, 1, 400, 144), (1, 400, 144)),
    # ttnn_yolov8l.py:1017 — SPPF NHWC reshape (1,1,400,512) -> (1,20,20,512)  [P5, C=512]
    ("sppf_nhwc_l", (1, 1, 400, 512), (1, 20, 20, 512)),
    # ttnn_yolov8s.py:748 — SPPF NHWC reshape (yolov8s narrower neck, C=256)
    ("sppf_nhwc_s", (1, 1, 400, 256), (1, 20, 20, 256)),
    # ttnn_yolov8l.py:1031 — post-upsample flatten (1,40,40,512) -> (1,1,1600,512)
    ("upsample_flatten_l", (1, 40, 40, 512), (1, 1, 1600, 512)),
    # ttnn_yolov8s.py:762 — post-upsample flatten (yolov8s C=256)
    ("upsample_flatten_s", (1, 40, 40, 256), (1, 1, 1600, 256)),
    # tt_yolov8l_utils.py:110 — conv-bias reshape (C,) -> (1,1,1,C)
    ("bias_reshape_l", (512,), (1, 1, 1, 512)),
    ("bias_reshape_s", (256,), (1, 1, 1, 256)),
]


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, in_shape, out_shape",
    [pytest.param(*s, id=s[0]) for s in _RESHAPE_SITES],
)
def test_reshape(ttnn_mesh_device, reset_seeds, name, in_shape, out_shape):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(in_shape)
    x = U.to_tt(x_torch, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.reshape(x, list(out_shape))

    ref = torch.reshape(x_torch.float(), out_shape)
    U.assert_lossless(ref, out, mesh_device=mesh)


# =============================================================================
# Model-faithful variant: reproduce the model's REAL layout / buffer state.
# =============================================================================

# The DFL reshapes run on TILE_LAYOUT in L1 (memory_config=_DETECT_MEM_CONFIG, which
# is ttnn.L1_MEMORY_CONFIG at 640x640). The existing ``test_reshape`` covers the safe
# ROW_MAJOR path; these reproduce the model's actual TILE/L1 state.
#
# (id, in_shape, out_shape, model_line).
_DFL_RESHAPE_SITES = [
    # ttnn_yolov8l.py:644 — DFL reg reshape (b,64,a)->(b,4,c1,a), c1=16: (1,64,8400)->(1,4,16,8400)
    ("dfl_reg_split_tile", (1, 64, 8400), (1, 4, 16, 8400), 644),
    # ttnn_yolov8l.py:651 — DFL final reshape (b,1,4,a)->(b,4,a): (1,1,4,8400)->(1,4,8400)
    ("dfl_final_tile", (1, 1, 4, 8400), (1, 4, 8400), 651),
]


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, in_shape, out_shape, model_line",
    [pytest.param(*s, id=s[0]) for s in _DFL_RESHAPE_SITES],
)
def test_reshape_tile_dfl(ttnn_mesh_device, reset_seeds, name, in_shape, out_shape, model_line):
    """Model-faithful ``ttnn.reshape`` on TILE / L1 — the DFL reg / final reshapes.

    The DFL head reshapes the reg-distance tensor while still in TILE_LAYOUT in L1:

        x = ttnn.reshape(x, (b, 4, c1, a), memory_config=_DETECT_MEM_CONFIG)   # :644
        ...
        x = ttnn.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3])) # :651

    At 640x640 ``_DETECT_MEM_CONFIG == ttnn.L1_MEMORY_CONFIG``. The anchor dim (8400)
    is NOT tile-aligned (8400 = 262.5 * 32), so a TILE-layout reshape that re-splits
    the row/channel dims of this tensor is finicky (the same class of hazard as the
    split issue #17017). We build it exactly as the model does; reshape is
    value-preserving so the reference is ``torch.reshape`` at PCC 0.999.

    If a Quasar/ttnn TILE reshape rejects the non-tile-aligned shape at runtime, this
    documents that the isolated op cannot reproduce the layout the model relies on
    (the reason surfaces as the op's own tile-alignment error).

    Model call sites (branch ``origin/sdawle/yolov8_bh``):
      * models/demos/yolov8l/tt/ttnn_yolov8l.py:644 (reg split), :651 (final)
        (same op at models/demos/yolov8s/tt/ttnn_yolov8s.py:507 / :519)
    """
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(in_shape)
    x = U.to_tile_l1(x_torch, mesh)

    out = ttnn.reshape(x, list(out_shape))

    ref = torch.reshape(x_torch.float(), out_shape)
    U.assert_lossless(ref, out, mesh_device=mesh)

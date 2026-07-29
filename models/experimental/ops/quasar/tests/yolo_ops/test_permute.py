# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.permute`` (YOLOv8, 640x640, batch 1).

YOLOv8 uses ``ttnn.permute`` for NCHW<->NHWC layout swaps at the stem and inside
the DFL / detect head. It is a value-preserving axis permutation, so the torch
reference is ``torch.permute``. Uploaded ROW_MAJOR to avoid tile-padding effects.

Model call sites (branch ``origin/sdawle/yolov8_bh``):
  * models/demos/yolov8l/tt/ttnn_yolov8l.py:986  stem NCHW->NHWC permute(0,2,3,1) of (1,16,H,W)
    (same at models/demos/yolov8s/tt/ttnn_yolov8s.py:714)
  * ttnn_yolov8l.py:646  DFL permute(0,1,3,2) of (1,4,16,8400) -> (1,4,8400,16)
    (same at ttnn_yolov8s.py:509)
  * ttnn_yolov8l.py:649  DFL post-conv NHWC->NCHW permute(0,3,1,2)
  * ttnn_yolov8l.py:745  detect x_cat permute(0,2,1) of (1,8400,144) -> (1,144,8400)
    (same at ttnn_yolov8s.py:566)

Detect head at 640 input -> anchor count 8400, no = nc + reg_max*4 = 144.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U


# --- Stem NCHW->NHWC: spatial feature map (H==W) -> parametrize by `hw`. ------
# hw=20 fits the emulator; hw=640 is the real stem resolution (real-BH only).
# Channel dim is 16 (3-channel input already padded to 16 upstream).
@U.with_default_mesh()
@pytest.mark.parametrize("hw", [20, 640])
def test_permute_nchw_to_nhwc(ttnn_mesh_device, reset_seeds, hw):
    """ttnn_yolov8l.py:986 / ttnn_yolov8s.py:714 — permute(0,2,3,1)."""
    mesh = ttnn_mesh_device
    dims = (0, 2, 3, 1)

    x_torch = U.torch_rand((1, U.IMG_CH_PADDED, hw, hw))
    x = U.to_tt(x_torch, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.permute(x, dims)

    ref = torch.permute(x_torch.float(), dims).contiguous()
    U.assert_lossless(ref, out, mesh_device=mesh)


# =============================================================================
# Model-faithful variant: reproduce the model's REAL layout / buffer state.
# =============================================================================

# The DFL / detect permutes run on TILE_LAYOUT in L1 (memory_config=_DETECT_MEM_CONFIG,
# == ttnn.L1_MEMORY_CONFIG at 640x640). The existing tests above cover ROW_MAJOR; these
# reproduce the model's actual TILE/L1 state.
#
# (id, shape, dims, model_line).
_PERMUTE_TILE_SITES = [
    # ttnn_yolov8l.py:646 / ttnn_yolov8s.py:509 — DFL (1,4,16,8400) -> (1,4,8400,16)
    ("dfl_transpose_tile", (1, 4, 16, 8400), (0, 1, 3, 2), 646),
    # ttnn_yolov8l.py:649 — DFL post-conv NHWC->NCHW (1,4,8400,1) -> (1,1,4,8400)
    ("dfl_nhwc_to_nchw_tile", (1, 4, 8400, 1), (0, 3, 1, 2), 649),
    # ttnn_yolov8l.py:745 / ttnn_yolov8s.py:566 — detect x_cat (1,8400,144) -> (1,144,8400)
    ("detect_xcat_tile", (1, 8400, 144), (0, 2, 1), 745),
]


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, shape, dims, model_line",
    [pytest.param(*s, id=s[0]) for s in _PERMUTE_TILE_SITES],
)
def test_permute_tile(ttnn_mesh_device, reset_seeds, name, shape, dims, model_line):
    """Model-faithful ``ttnn.permute`` on TILE / L1 — the DFL / detect-head permutes.

    Every detect-head permute passes ``memory_config=_DETECT_MEM_CONFIG`` (== L1 at
    640x640) and runs while the tensor is in TILE_LAYOUT:

        x     = ttnn.permute(x, (0, 1, 3, 2), memory_config=_DETECT_MEM_CONFIG)  # :646
        x     = ttnn.permute(x, (0, 3, 1, 2))                                    # :649
        x_cat = ttnn.permute(x_cat, (0, 2, 1), memory_config=_DETECT_MEM_CONFIG) # :745

    The anchor dim (8400) is not tile-aligned, so these transpose the padded row/col
    dims of a non-tile-aligned tensor. Permute is value-preserving; reference is
    ``torch.permute`` at PCC 0.999.

    Model call sites (branch ``origin/sdawle/yolov8_bh``):
      * models/demos/yolov8l/tt/ttnn_yolov8l.py:646, :649, :745
        (:646/:745 also at models/demos/yolov8s/tt/ttnn_yolov8s.py:509 / :566)
    """
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tile_l1(x_torch, mesh)

    out = ttnn.permute(x, dims)

    ref = torch.permute(x_torch.float(), dims).contiguous()
    U.assert_lossless(ref, out, mesh_device=mesh)


# --- DFL / detect-head permutes: non-spatial (H != W) -> shape-tuple params. --
_PERMUTE_SITES = [
    # ttnn_yolov8l.py:646 / ttnn_yolov8s.py:509 — DFL (1,4,16,8400) -> (1,4,8400,16)
    ("dfl_transpose", (1, 4, 16, 8400), (0, 1, 3, 2)),
    # ttnn_yolov8l.py:649 — DFL post-conv NHWC->NCHW (1,4,8400,1) -> (1,1,4,8400)
    ("dfl_nhwc_to_nchw", (1, 4, 8400, 1), (0, 3, 1, 2)),
    # ttnn_yolov8l.py:745 / ttnn_yolov8s.py:566 — detect x_cat (1,8400,144) -> (1,144,8400)
    ("detect_xcat", (1, 8400, 144), (0, 2, 1)),
]


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, shape, dims",
    [pytest.param(*s, id=s[0]) for s in _PERMUTE_SITES],
)
def test_permute_detect(ttnn_mesh_device, reset_seeds, name, shape, dims):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.permute(x, dims)

    ref = torch.permute(x_torch.float(), dims).contiguous()
    U.assert_lossless(ref, out, mesh_device=mesh)

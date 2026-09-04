# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.split`` (YOLOv8, 640x640, batch 1).

``dist2bbox`` (``ttnn_decode_bboxes``) splits the DFL distance tensor
[1, 4, 8400] along dim 1 into the left-top and right-bottom offset pairs
(2 coords each):

    lt, rb = ttnn.split(distance, 2, 1)   # split_size=2, dim=1 -> two [1,2,8400]

``ttnn.split`` here is value-preserving (a straight partition), so each half is
compared against ``torch.split`` with PCC 0.999. Uploaded ROW_MAJOR so the
partition is exact.

Model call sites (branch ``origin/sdawle/yolov8_bh``):
  * models/demos/yolov8l/tt/tt_yolov8l_utils.py:92
  * models/demos/yolov8s/tt/tt_yolov8s_utils.py:90

The DFL distance tensor is (1, 4, 8400): 4 = reg_max(16) reduced to 1 per each of
the 4 box edges, over 8400 anchors at 640x640.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [
        # tt_yolov8l_utils.py:92 / tt_yolov8s_utils.py:90 — distance (1,4,8400)
        pytest.param((1, 4, 8400), id="dist2bbox_4x8400"),
    ],
)
def test_split(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device
    split_size, dim = 2, 1

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh, layout=ttnn.ROW_MAJOR_LAYOUT)

    parts = ttnn.split(x, split_size, dim)
    assert len(parts) == 2, f"expected 2 splits, got {len(parts)}"

    refs = torch.split(x_torch.float(), split_size, dim=dim)
    for ref, part in zip(refs, parts):
        U.assert_lossless(ref, part, mesh_device=mesh)


# =============================================================================
# Model-faithful variant: reproduce the model's REAL layout / buffer state.
# =============================================================================


@pytest.mark.blackhole_scale
@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [
        # tt_yolov8l_utils.py:92 / tt_yolov8s_utils.py:90 — distance (1,4,8400)
        pytest.param((1, 4, 8400), id="dist2bbox_4x8400_tile_l1"),
    ],
)
def test_split_tile(ttnn_mesh_device, reset_seeds, shape):
    """Model-faithful ``ttnn.split`` on TILE / L1 (the model's actual layout).

    ``dist2bbox`` runs ``lt, rb = ttnn.split(distance, 2, 1, memory_config=_DETECT_MEM_CONFIG)``
    where at 640x640 ``_DETECT_MEM_CONFIG == ttnn.L1_MEMORY_CONFIG`` and ``distance``
    reaches this call still in TILE_LAYOUT — the source itself flags this as risky:

        lt, rb = ttnn.split(distance, 2, 1, ...)  # if done in tile : tt-metal issue #17017

    This deliberately exercises that flagged TILE/L1 state (the existing ``test_split``
    covers the safe ROW_MAJOR path). Split is value-preserving; each half is compared
    against ``torch.split`` at PCC 0.999. Note the split is along dim=1 (size 4, split
    size 2), i.e. across the tile-padded row dim of a non-tile-aligned (8400) tensor —
    exactly the configuration issue #17017 warns about.

    Model call sites (branch ``origin/sdawle/yolov8_bh``):
      * models/demos/yolov8l/tt/tt_yolov8l_utils.py:92
      * models/demos/yolov8s/tt/tt_yolov8s_utils.py:90
    """
    mesh = ttnn_mesh_device
    split_size, dim = 2, 1

    x_torch = U.torch_rand(shape)
    x = U.to_tile_l1(x_torch, mesh)

    parts = ttnn.split(x, split_size, dim)
    assert len(parts) == 2, f"expected 2 splits, got {len(parts)}"

    refs = torch.split(x_torch.float(), split_size, dim=dim)
    for ref, part in zip(refs, parts):
        U.assert_lossless(ref, part, mesh_device=mesh)

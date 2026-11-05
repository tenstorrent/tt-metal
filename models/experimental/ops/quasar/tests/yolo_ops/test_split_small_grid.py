# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Small-grid companion to test_split.py.
#
# The test_split.py::test_split_tile cases are marked blackhole_scale: they reproduce the model's
# exact TILE L1 dist2bbox tensor at full size on 10-80 cores, so -m emulator never runs them. These call
# the same test body with sizes that fit 1-2 cores; everything else (layouts, memory
# configs, checks) is unchanged.
# ---------------------------------------------------------------------------
"""Small-grid cases for the model-faithful ``split`` call sites (fit a 1- or 2-compute-node device)."""

import pytest

from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U
from models.experimental.ops.quasar.tests.yolo_ops import test_split as T


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 4, 400), id="dist2bbox_4x400_tile_l1"),
    ],
)
def test_split_tile_small_grid(ttnn_mesh_device, reset_seeds, shape):
    T.test_split_tile(ttnn_mesh_device, reset_seeds, shape)

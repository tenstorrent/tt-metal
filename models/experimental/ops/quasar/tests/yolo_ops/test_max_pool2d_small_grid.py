# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Small-grid companion to test_max_pool2d.py.
#
# The test_max_pool2d.py::test_max_pool2d_sharded cases are marked blackhole_scale: they reproduce the model's
# exact SPPF triple pool on HEIGHT_SHARDED input at full size on 10-80 cores, so -m emulator never runs them. These call
# the same test body with sizes that fit 1-2 cores; everything else (layouts, memory
# configs, checks) is unchanged.
# ---------------------------------------------------------------------------
"""Small-grid cases for the model-faithful ``max_pool2d`` call sites (fit a 1- or 2-compute-node device)."""

import pytest

from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U
from models.experimental.ops.quasar.tests.yolo_ops import test_max_pool2d as T


@U.with_mesh_l1small()
@pytest.mark.parametrize(
    "variant, channels, hw",
    [
        pytest.param("yolov8l", 64, 8, id="sppf-sharded-64x8x8"),
    ],
)
def test_max_pool2d_sharded_small_grid(ttnn_mesh_device, reset_seeds, variant, channels, hw):
    T.test_max_pool2d_sharded(ttnn_mesh_device, reset_seeds, variant, channels, hw)

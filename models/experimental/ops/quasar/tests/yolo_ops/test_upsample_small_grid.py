# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Small-grid companion to test_upsample.py.
#
# The test_upsample.py::test_upsample_sharded cases are marked blackhole_scale: they reproduce the model's
# exact HEIGHT_SHARDED neck upsample at full size on 10-80 cores, so -m emulator never runs them. These call
# the same test body with sizes that fit 1-2 cores; everything else (layouts, memory
# configs, checks) is unchanged.
# ---------------------------------------------------------------------------
"""Small-grid cases for the model-faithful ``upsample`` call sites (fit a 1- or 2-compute-node device)."""

import pytest

from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U
from models.experimental.ops.quasar.tests.yolo_ops import test_upsample as T


@U.with_mesh_l1small()
@pytest.mark.parametrize(
    "sites, channels, hw, num_cores",
    [
        pytest.param("yolov8l+yolov8s", 64, 4, 2, id="sharded-64x4x4-2c"),
    ],
)
def test_upsample_sharded_small_grid(ttnn_mesh_device, reset_seeds, sites, channels, hw, num_cores):
    T.test_upsample_sharded(ttnn_mesh_device, reset_seeds, sites, channels, hw, num_cores)

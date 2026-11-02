# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Small-grid companion to test_sharded_to_interleaved.py.
#
# The test_sharded_to_interleaved.py::test_sharded_to_interleaved_height cases are marked blackhole_scale: they reproduce the model's
# exact HEIGHT_SHARDED SPPF concat at full size on 10-80 cores, so -m emulator never runs them. These call
# the same test body with sizes that fit 1-2 cores; everything else (layouts, memory
# configs, checks) is unchanged.
#
# The test_sharded_to_interleaved.py::test_sharded_to_interleaved_block cases are marked blackhole_scale: they reproduce the model's
# exact BLOCK_SHARDED conv outputs at full size on 10-80 cores, so -m emulator never runs them. These call
# the same test body with sizes that fit 1-2 cores; everything else (layouts, memory
# configs, checks) is unchanged.
# ---------------------------------------------------------------------------
"""Small-grid cases for the model-faithful ``sharded_to_interleaved`` call sites (fit a 1- or 2-compute-node device)."""

import pytest

from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U
from models.experimental.ops.quasar.tests.yolo_ops import test_sharded_to_interleaved as T


@U.with_default_mesh()
@pytest.mark.parametrize(
    "variant, num_cores, shape",
    [
        pytest.param("yolov8l", 2, (1, 1, 64, 64), id="height-2c-64x64"),
    ],
)
def test_sharded_to_interleaved_height_small_grid(ttnn_mesh_device, reset_seeds, variant, num_cores, shape):
    T.test_sharded_to_interleaved_height(ttnn_mesh_device, reset_seeds, variant, num_cores, shape)


@U.with_default_mesh()
@pytest.mark.parametrize(
    "site, grid_y, grid_x, shape",
    [
        pytest.param("c2f_6", 1, 2, (1, 1, 64, 128), id="block-1x2-64x128"),
    ],
)
def test_sharded_to_interleaved_block_small_grid(ttnn_mesh_device, reset_seeds, site, grid_y, grid_x, shape):
    T.test_sharded_to_interleaved_block(ttnn_mesh_device, reset_seeds, site, grid_y, grid_x, shape)

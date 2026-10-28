# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Small-grid companion to test_to_memory_config.py.
#
# The test_to_memory_config.py::test_to_memory_config_to_height_sharded cases are marked blackhole_scale: they reproduce the model's
# exact HEIGHT_SHARDED SPPF concat at full size on 10-80 cores, so -m emulator never runs them. These call
# the same test body with sizes that fit 1-2 cores; everything else (layouts, memory
# configs, checks) is unchanged.
# ---------------------------------------------------------------------------
"""Small-grid cases for the model-faithful ``to_memory_config`` call sites (fit a 1- or 2-compute-node device)."""

import pytest

from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U
from models.experimental.ops.quasar.tests.yolo_ops import test_to_memory_config as T


@U.with_default_mesh()
@pytest.mark.parametrize(
    "variant, num_cores, shape",
    [
        pytest.param("yolov8l", 2, (1, 1, 64, 64), id="height-2c-64x64"),
    ],
)
def test_to_memory_config_to_height_sharded_small_grid(ttnn_mesh_device, reset_seeds, variant, num_cores, shape):
    T.test_to_memory_config_to_height_sharded(ttnn_mesh_device, reset_seeds, variant, num_cores, shape)

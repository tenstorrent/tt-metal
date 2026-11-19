# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Small-grid companion to test_interleaved_to_sharded.py.
#
# The test_interleaved_to_sharded.py::test_interleaved_to_sharded_multicore cases are marked blackhole_scale: they reproduce the model's
# exact HEIGHT_SHARDED neck layout at full size on 10-80 cores, so -m emulator never runs them. These call
# the same test body with sizes that fit 1-2 cores; everything else (layouts, memory
# configs, checks) is unchanged.
# ---------------------------------------------------------------------------
"""Small-grid cases for the model-faithful ``interleaved_to_sharded`` call sites (fit a 1- or 2-compute-node device)."""

import pytest

from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U
from models.experimental.ops.quasar.tests.yolo_ops import test_interleaved_to_sharded as T


@U.with_default_mesh()
@pytest.mark.parametrize(
    "site, num_cores, shape",
    [
        pytest.param("sppf_9", 2, (1, 4, 4, 64), id="height-2c-4x4x64"),
        pytest.param("sppf_9", 1, (1, 4, 4, 64), id="height-1c-4x4x64"),
    ],
)
def test_interleaved_to_sharded_multicore_small_grid(ttnn_mesh_device, reset_seeds, site, num_cores, shape):
    T.test_interleaved_to_sharded_multicore(ttnn_mesh_device, reset_seeds, site, num_cores, shape)

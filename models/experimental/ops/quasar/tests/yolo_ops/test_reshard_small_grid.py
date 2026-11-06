# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Small-grid companion to test_reshard.py.
#
# The test_reshard.py::test_reshard_multicore cases are marked blackhole_scale: they reproduce the model's
# exact HEIGHT_SHARDED reshard at full size on 10-80 cores, so -m emulator never runs them. These call
# the same test body with sizes that fit 1-2 cores; everything else (layouts, memory
# configs, checks) is unchanged.
# ---------------------------------------------------------------------------
"""Small-grid cases for the model-faithful ``reshard`` call sites (fit a 1- or 2-compute-node device)."""

import pytest

from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U
from models.experimental.ops.quasar.tests.yolo_ops import test_reshard as T


@U.with_default_mesh()
@pytest.mark.parametrize(
    "site, src_cores, dst_cores, shape",
    [
        pytest.param("sppf_9", 1, 2, (1, 1, 64, 64), id="reshard-1c-to-2c-64x64"),
        pytest.param("sppf_9", 2, 1, (1, 1, 64, 64), id="reshard-2c-to-1c-64x64"),
    ],
)
def test_reshard_multicore_small_grid(ttnn_mesh_device, reset_seeds, site, src_cores, dst_cores, shape):
    T.test_reshard_multicore(ttnn_mesh_device, reset_seeds, site, src_cores, dst_cores, shape)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Small-grid companion to test_permute.py.
#
# The test_permute.py::test_permute_tile cases are marked blackhole_scale: they reproduce the model's
# exact TILE L1 DFL / detect tensors at full size on 10-80 cores, so -m emulator never runs them. These call
# the same test body with sizes that fit 1-2 cores; everything else (layouts, memory
# configs, checks) is unchanged.
# ---------------------------------------------------------------------------
"""Small-grid cases for the model-faithful ``permute`` call sites (fit a 1- or 2-compute-node device)."""

import pytest

from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U
from models.experimental.ops.quasar.tests.yolo_ops import test_permute as T


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, shape, dims, model_line",
    [
        pytest.param("dfl_transpose_tile", (1, 4, 16, 400), (0, 1, 3, 2), 646, id="dfl_transpose_tile-400"),
        pytest.param("dfl_nhwc_to_nchw_tile", (1, 4, 400, 1), (0, 3, 1, 2), 649, id="dfl_nhwc_to_nchw_tile-400"),
        pytest.param("detect_xcat_tile", (1, 400, 144), (0, 2, 1), 745, id="detect_xcat_tile-400"),
    ],
)
def test_permute_tile_small_grid(ttnn_mesh_device, reset_seeds, name, shape, dims, model_line):
    T.test_permute_tile(ttnn_mesh_device, reset_seeds, name, shape, dims, model_line)

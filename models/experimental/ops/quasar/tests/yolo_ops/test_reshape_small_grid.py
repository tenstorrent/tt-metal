# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Small-grid companion to test_reshape.py.
#
# The test_reshape.py::test_reshape_tile_dfl cases are marked blackhole_scale: they reproduce the model's
# exact TILE L1 DFL tensors at full size on 10-80 cores, so -m emulator never runs them. These call
# the same test body with sizes that fit 1-2 cores; everything else (layouts, memory
# configs, checks) is unchanged.
# ---------------------------------------------------------------------------
"""Small-grid cases for the model-faithful ``reshape`` call sites (fit a 1- or 2-compute-node device)."""

import pytest

from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U
from models.experimental.ops.quasar.tests.yolo_ops import test_reshape as T


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, in_shape, out_shape, model_line",
    [
        pytest.param("dfl_reg_split_tile", (1, 64, 400), (1, 4, 16, 400), 644, id="dfl_reg_split_tile-400"),
        pytest.param("dfl_final_tile", (1, 1, 4, 400), (1, 4, 400), 651, id="dfl_final_tile-400"),
    ],
)
def test_reshape_tile_dfl_small_grid(ttnn_mesh_device, reset_seeds, name, in_shape, out_shape, model_line):
    T.test_reshape_tile_dfl(ttnn_mesh_device, reset_seeds, name, in_shape, out_shape, model_line)

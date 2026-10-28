# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Small-grid companion to test_clone.py.
#
# The test_clone.py::test_clone_rm_l1 cases are marked blackhole_scale: they reproduce the model's
# exact ROW_MAJOR L1 neck feature map at full size on 10-80 cores, so -m emulator never runs them. These call
# the same test body with sizes that fit 1-2 cores; everything else (layouts, memory
# configs, checks) is unchanged.
# ---------------------------------------------------------------------------
"""Small-grid cases for the model-faithful ``clone`` call sites (fit a 1- or 2-compute-node device)."""

import pytest

from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U
from models.experimental.ops.quasar.tests.yolo_ops import test_clone as T


@U.with_default_mesh()
@pytest.mark.parametrize(
    "hw, c",
    [
        pytest.param(20, 64, id="rm_l1-20x20x64"),
    ],
)
def test_clone_rm_l1_small_grid(ttnn_mesh_device, reset_seeds, hw, c):
    T.test_clone_rm_l1(ttnn_mesh_device, reset_seeds, hw, c)

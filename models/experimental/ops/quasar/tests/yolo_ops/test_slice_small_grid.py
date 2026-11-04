# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Small-grid companion to test_slice.py.
#
# test_slice.py::test_slice_tile hard-codes 8400 anchors. This keeps the box / cls channel
# slices of the TILE L1 [1, 144, A] detect tensor at A = 400.
# ---------------------------------------------------------------------------
"""Small-grid cases for the model-faithful ``slice`` call sites (fit a 1- or 2-compute-node device)."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, start_c, end_c",
    [pytest.param("box", 0, 64, id="box_0_64_tile"), pytest.param("cls", 64, 144, id="cls_64_144_tile")],
)
def test_slice_tile_small_grid(ttnn_mesh_device, reset_seeds, name, start_c, end_c):
    mesh = ttnn_mesh_device
    n = 400
    x_torch = U.torch_rand((1, 144, n))
    out = ttnn.slice(U.to_tile_l1(x_torch, mesh), [0, start_c, 0], [1, end_c, n])
    U.assert_lossless(x_torch.float()[0:1, start_c:end_c, 0:n], out, mesh_device=mesh)

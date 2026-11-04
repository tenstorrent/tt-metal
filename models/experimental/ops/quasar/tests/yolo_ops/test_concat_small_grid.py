# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Small-grid companion to test_concat.py.
#
# The test_concat.py::test_concat_detect_l1 cases are marked blackhole_scale: they reproduce the model's
# exact TILE L1 detect buffers at full size on 10-80 cores, so -m emulator never runs them. These call
# the same test body with sizes that fit 1-2 cores; everything else (layouts, memory
# configs, checks) is unchanged.
#
# test_concat.py::test_concat_neck_sharded hard-codes the 80x80 P3 level on 64 cores. This
# keeps its form (unequal-width ROW_MAJOR HEIGHT_SHARDED operands joined on the channel dim)
# at 8x8 on 2 cores.
# ---------------------------------------------------------------------------
"""Small-grid cases for the model-faithful ``concat`` call sites (fit a 1- or 2-compute-node device)."""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U
from models.experimental.ops.quasar.tests.yolo_ops import test_concat as T


@U.with_default_mesh()
@pytest.mark.parametrize(
    "name, dim, in_shapes, out_shape",
    [
        pytest.param("cv2cv3", 3, ((1, 1, 64, 64), (1, 1, 64, 80)), (1, 1, 64, 144), id="cv2cv3-l1-64"),
        pytest.param("detect_xi", 1, ((1, 64, 144), (1, 16, 144), (1, 4, 144)), (1, 84, 144), id="detect_xi-l1-84"),
        pytest.param("dist2bbox", 1, ((1, 2, 400), (1, 2, 400)), (1, 4, 400), id="dist2bbox-l1-400"),
    ],
)
def test_concat_detect_l1_small_grid(ttnn_mesh_device, reset_seeds, name, dim, in_shapes, out_shape):
    T.test_concat_detect_l1(ttnn_mesh_device, reset_seeds, name, dim, in_shapes, out_shape)


@U.with_default_mesh()
@pytest.mark.parametrize("num_cores", [pytest.param(2, id="2c"), pytest.param(1, id="1c")])
def test_concat_neck_sharded_small_grid(ttnn_mesh_device, reset_seeds, num_cores):
    """upsample(Ca) + backbone skip(Cb) channel concat on HEIGHT_SHARDED ROW_MAJOR operands."""
    mesh = ttnn_mesh_device
    rows, Ca, Cb = 8 * 8, 64, 32
    a_torch = U.torch_rand((1, 1, rows, Ca))
    b_torch = U.torch_rand((1, 1, rows, Cb))
    a = U.to_tt(
        a_torch,
        mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=U.height_sharded_memcfg(mesh, num_cores, (1, 1, rows, Ca)),
    )
    b = U.to_tt(
        b_torch,
        mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=U.height_sharded_memcfg(mesh, num_cores, (1, 1, rows, Cb)),
    )
    out = ttnn.concat([a, b], dim=-1)
    U.assert_lossless(torch.cat([a_torch.float(), b_torch.float()], dim=-1), out, mesh_device=mesh)

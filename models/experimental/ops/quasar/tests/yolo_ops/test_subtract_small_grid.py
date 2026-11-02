# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Small-grid companion to test_subtract.py.
#
# The model's subtract_l1 call site writes a bfloat8_b output, which Quasar does not have, so
# test_subtract.py::test_subtract_l1 always skips there (and is blackhole_scale besides). This keeps
# the call site's TILE L1 operands at 400 anchors and writes bfloat16 instead.
# ---------------------------------------------------------------------------
"""Small-grid cases for the model-faithful ``subtract`` call sites (fit a 1- or 2-compute-node device)."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize("shape", [pytest.param((1, 2, 400), id="1x2x400")])
def test_subtract_l1_small_grid(ttnn_mesh_device, reset_seeds, shape):
    """anchor - lt / x2y2 - x1y1 on L1 detect buffers, bfloat16 output."""
    mesh = ttnn_mesh_device
    a_torch = U.torch_rand(shape)
    b_torch = U.torch_rand(shape)
    out = U.op("subtract")(U.to_tile_l1(a_torch, mesh), U.to_tile_l1(b_torch, mesh), dtype=ttnn.bfloat16)
    assert out.dtype == ttnn.bfloat16, f"expected bfloat16, got {out.dtype}"
    U.assert_pcc(a_torch.float() - b_torch.float(), out, pcc=0.999, mesh_device=mesh)

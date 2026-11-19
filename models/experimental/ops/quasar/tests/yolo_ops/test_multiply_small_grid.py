# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Small-grid companion to test_multiply.py.
#
# The model's multiply_l1 call site writes a bfloat8_b output, which Quasar does not have, so
# test_multiply.py::test_multiply_l1 always skips there (and is blackhole_scale besides). This keeps
# the call site's TILE L1 operands at 400 anchors and writes bfloat16 instead.
# ---------------------------------------------------------------------------
"""Small-grid cases for the model-faithful ``multiply`` call sites (fit a 1- or 2-compute-node device)."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize("shape", [pytest.param((1, 4, 400), id="1x4x400")])
def test_multiply_l1_small_grid(ttnn_mesh_device, reset_seeds, shape):
    """dbox * strides on L1 detect buffers (strides broadcast over the 4 coords), bfloat16 output."""
    mesh = ttnn_mesh_device
    dbox_torch = U.torch_rand(shape)
    strides_torch = U.torch_rand((1, 1, shape[-1]))
    out = U.op("multiply")(U.to_tile_l1(dbox_torch, mesh), U.to_tile_l1(strides_torch, mesh), dtype=ttnn.bfloat16)
    assert out.dtype == ttnn.bfloat16, f"expected bfloat16, got {out.dtype}"
    U.assert_pcc(dbox_torch.float() * strides_torch.float(), out, pcc=0.999, mesh_device=mesh)

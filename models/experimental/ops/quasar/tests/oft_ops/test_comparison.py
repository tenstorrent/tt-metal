# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: comparison / logical binary ops from the composite ``ttnn.atan2`` in the OFT
encoder's angle decode, on [1, 159, 159] bf16 TILE, L1 interleaved.

Shape source (kernel cache, binary_ng):
  * ``ge(y, 0.0)``: ``eltwise_binary_scalar/964555`` — FPU sub with ``gez`` as post-activation,
    scalar operand (``writer_interleaved_scalar``)                                      [H]
  * ``logical_and(a, b)``: ``eltwise_binary_no_bcast/148942`` — FPU mul with ``nez`` on lhs,
    rhs and output; both operands L1                                                     [H]

Results are 0/1, so the check is exact.

Cache #1 only. The second cache (in-tree OFT at ``628a4797bf7``) runs ``atan2`` natively
(``test_atan2.py``) and has no comparison or logical kernels.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U


def _with_zeros(x):
    x = x.clone()
    x.view(-1)[::5] = 0
    return x


@U.with_default_mesh()
@pytest.mark.parametrize("shape", [pytest.param(U.HEATMAP, id="ge-scalar0-1x159x159-964555")])
def test_ge_scalar(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device
    y = _with_zeros(U.torch_rand(shape))
    out = U.op("ge")(U.to_tt_mem(y, mesh, "l1"), 0.0, memory_config=ttnn.L1_MEMORY_CONFIG)
    U.assert_lossless((y.float() >= 0).float(), out, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize("shape", [pytest.param(U.HEATMAP, id="logical_and-1x159x159-148942")])
def test_logical_and(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device
    # atan2 ANDs 0/1 masks; mix in non-0/1 values too since logical_and tests != 0.
    a = _with_zeros((torch.rand(shape) > 0.3).to(torch.bfloat16) * U.torch_rand(shape))
    b = (torch.rand(shape) > 0.5).to(torch.bfloat16)
    out = U.op("logical_and")(
        U.to_tt_mem(a, mesh, "l1"), U.to_tt_mem(b, mesh, "l1"), memory_config=ttnn.L1_MEMORY_CONFIG
    )
    U.assert_lossless(((a != 0) & (b != 0)).float(), out, mesh_device=mesh)

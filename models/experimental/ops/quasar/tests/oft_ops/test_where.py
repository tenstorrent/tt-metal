# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.where`` in its three operand forms — all from the composite
``ttnn.atan2`` in the OFT encoder's angle decode, on [1, 159, 159] bf16 TILE, L1
interleaved in and out.

Shape source (kernel cache):
  * tensor-scalar-scalar ``where(c, pi/2, 0.0)``: unary ``where_tss_kernel``
    (726108 / 146021; 2 and 1 tiles per core -> 25 tiles)                            [H]
  * tensor-scalar-tensor ``where(c, -pi/2, t)``: binary_ng ``eltwise_where_no_bcast/182526``
    (``WHERE_TST 1``)                                                                    [H]
  * tensor-tensor-tensor ``where(c, t, f)``: ``ternary_sfpu_no_bcast_ttt/534125`` with
    ``ternary_reader_nosubtilebcast_ttt`` / ``ternary_writer_nobcast``, all L1        [H]

``where`` selects values, so the check is exact.

Cache #1 only. The second cache (in-tree OFT at ``628a4797bf7``) runs ``atan2`` natively
(``test_atan2.py``) and has no ``where`` kernels.
"""

import math

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize(
    "form, shape",
    [
        pytest.param("tss", U.HEATMAP, id="where-tensor-scalar-scalar-726108"),
        pytest.param("tst", U.HEATMAP, id="where-tensor-scalar-tensor-182526"),
        pytest.param("ttt", U.HEATMAP, id="where-tensor-tensor-tensor-534125"),
    ],
)
def test_where(ttnn_mesh_device, reset_seeds, form, shape):
    mesh = ttnn_mesh_device
    cond = (torch.rand(shape) > 0.5).to(torch.bfloat16)
    t = U.torch_rand(shape)
    f = U.torch_rand(shape)
    tt_cond = U.to_tt_mem(cond, mesh, "l1")
    tt_t = U.to_tt_mem(t, mesh, "l1")
    tt_f = U.to_tt_mem(f, mesh, "l1")
    # The scalars atan2 uses; round to bf16 so the exact comparison is fair.
    half_pi = float(torch.tensor(math.pi / 2, dtype=torch.bfloat16))

    if form == "tss":
        out = ttnn.where(tt_cond, half_pi, 0.0, memory_config=ttnn.L1_MEMORY_CONFIG)
        ref = torch.where(cond.bool(), torch.tensor(half_pi), torch.tensor(0.0))
    elif form == "tst":
        out = ttnn.where(tt_cond, -half_pi, tt_f, memory_config=ttnn.L1_MEMORY_CONFIG)
        ref = torch.where(cond.bool(), torch.tensor(-half_pi), f.float())
    else:
        out = ttnn.where(tt_cond, tt_t, tt_f, memory_config=ttnn.L1_MEMORY_CONFIG)
        ref = torch.where(cond.bool(), t.float(), f.float())
    U.assert_lossless(ref, out, mesh_device=mesh)

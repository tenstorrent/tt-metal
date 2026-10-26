# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.sum`` — the spatial-cross-attention hit-count reductions.

Shape source (kernel cache):
  * ``sum(dim=-1)`` on ``bev_mask`` [6, 1, 10000, 4] (cams, 1, queries, Z-anchors):
    ``reduce_w`` (REDUCE_ROW SUM via MM) + ``reader_unary_reduce_universal_start_id``
    (scaler 0x3F80 = 1.0). Compute ct rows_per_core 94/93 over 20 cores = 1878 = 6 x 313 tile rows [H]
  * ``sum(dim=0)`` over the 6 cameras: ``reduce_nc`` / ``reader_reduce_nc`` (fast_reduce_nc,
    reduced dim 6, 313 output tiles) [H]; width <= 32 [M]
Both bf16, L1 interleaved; HiFi4 + fp32 dest acc. Inputs are 0/1 masks so sums are exact.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

C, Q = U.NUM_CAMS, U.NUM_QUERY


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape, dim",
    [
        pytest.param((C, 1, Q, 4), -1, id="R1-bev_mask-sum-zanchors-135669"),
        pytest.param((C, 1, Q, 32), 0, id="R2-sum-over-cams-300812"),
    ],
)
def test_sum(ttnn_mesh_device, reset_seeds, shape, dim):
    mesh = ttnn_mesh_device
    x_torch = torch.randint(0, 2, shape).to(torch.bfloat16)
    x = U.to_tt_mem(x_torch, mesh, "l1")
    out = ttnn.sum(x, dim=dim, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
    U.assert_pcc(x_torch.float().sum(dim=dim, keepdim=True), out, pcc=0.9999, mesh_device=mesh)

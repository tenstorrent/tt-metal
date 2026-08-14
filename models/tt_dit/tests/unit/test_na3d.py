# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device parity for 3D neighborhood attention.

The host executor in ``layers.na3d`` is already checked against upstream's own natten-free
na3d, so this compares device against host: same plan, same masks, so any gap is ttnn
execution (gather ordering, layout round trips, bf16 rounding) rather than window arithmetic.

Shapes are the ones the DiffVAE decoder actually uses — kernels ``(3,7,7)``, ``(3,5,5)`` and
``(11,11,11)`` — plus the boundary cases that a window rule can get wrong: an axis shorter
than the kernel, and a grid large enough to force more than one tile per axis.
"""

import pytest
import torch

import ttnn

from ...layers.na3d import build_device_plan, na3d_torch, neighborhood_attention_3d, plan_na3d
from ...utils.check import assert_quality


@pytest.mark.parametrize(
    "dims, kernel",
    [
        ((3, 7, 7), (3, 7, 7)),  # stage-1/2 kernel, exact fit
        ((2, 5, 5), (3, 7, 7)),  # every axis shorter than the kernel
        ((6, 9, 9), (3, 7, 7)),  # interior plus both boundary regimes
        ((8, 12, 6), (3, 5, 5)),  # stage-3/4 kernel, non-cubic
        ((12, 16, 16), (11, 11, 11)),  # stage-5 kernel, multiple tiles
    ],
)
@pytest.mark.parametrize("heads, head_dim", [(4, 64)])
def test_na3d_matches_host(*, device, dims, kernel, heads, head_dim):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, *dims, heads, head_dim, dtype=torch.float32) for _ in range(3))

    plan = plan_na3d(dims, kernel)
    expected = na3d_torch(q, k, v, kernel, scale=1.0, plan=plan)
    expected = expected.reshape(1, *dims, heads * head_dim)

    tt_q, tt_k, tt_v = (
        ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for x in (q, k, v)
    )
    device_plan = build_device_plan(plan, mesh_device=device, dtype=ttnn.bfloat16)
    actual = neighborhood_attention_3d(tt_q, tt_k, tt_v, kernel_size=kernel, scale=1.0, device_plan=device_plan)

    assert tuple(actual.shape) == tuple(expected.shape), f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    assert_quality(expected, ttnn.to_torch(actual), pcc=0.999)

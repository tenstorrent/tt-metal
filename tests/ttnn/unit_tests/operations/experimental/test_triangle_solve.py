# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit test for ttnn.experimental.triangle_solve.

Solves  L X = RHS  for a single 32x32 tile via SFPU forward substitution. L is a unit
lower-triangular matrix; the op takes it NEGATED on the strict-lower part (diagonal is an
implicit 1, upper triangle ignored).
"""
import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def test_triangle_solve(device):
    torch.manual_seed(0)

    N = 32

    # Well-conditioned unit lower-triangular L: diagonal exactly 1, small strict-lower part.
    L = torch.eye(N) + torch.tril(torch.randn(N, N) * 0.1, diagonal=-1)

    # L_neg: negate the strict-lower part, keep the (unit) diagonal, zero the upper triangle.
    L_neg = -torch.tril(L, diagonal=-1) + torch.eye(N)

    RHS = torch.randn(N, N)

    # Reference solve (unit-triangular: diagonal treated as 1).
    X_ref = torch.linalg.solve_triangular(L, RHS, upper=False, unitriangular=True)

    L_neg_t = L_neg.reshape(1, 1, N, N).to(torch.bfloat16)
    RHS_t = RHS.reshape(1, 1, N, N).to(torch.bfloat16)

    l_neg_tt = ttnn.from_torch(L_neg_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    rhs_tt = ttnn.from_torch(RHS_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    x_tt = ttnn.experimental.triangle_solve(l_neg_tt, rhs_tt)

    assert list(x_tt.shape) == [1, 1, N, N]
    assert x_tt.dtype == ttnn.bfloat16

    X_out = ttnn.to_torch(x_tt).reshape(N, N).to(torch.float32)

    # Loose PCC threshold to accommodate bf16 accumulation error in the solve.
    assert_with_pcc(X_ref, X_out, 0.99)

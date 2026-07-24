# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Unit test for the fused Flash KDA (Kimi Delta Attention) recurrent state-update op
(ttnn.transformer.flash_kda). Validates the device op against a torch reference of the
exact per-core recurrence from tenstorrent/tt-blaze#2431:

    S_tilde = S_prev * g            (g scales each key-dim row, broadcast over value cols)
    pred    = k @ S_tilde
    err     = v - pred
    delta   = beta * err
    S_new   = S_tilde + (k outer delta)
    out     = q @ S_new
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def _flash_kda_reference(S_prev, g, k, v, beta, q):
    """Torch golden for a batch of N independent per-core state updates.

    Shapes: S_prev [N,Dk,Dv], g [N,Dk,1], k [N,1,Dk], v [N,1,Dv], beta [N,1,1], q [N,1,Dk].
    Returns (S_new [N,Dk,Dv], out [N,1,Dv]).
    """
    S_tilde = S_prev * g  # [N,Dk,Dv] — g broadcasts over the Dv columns of each row
    pred = torch.matmul(k, S_tilde)  # [N,1,Dk] @ [N,Dk,Dv] -> [N,1,Dv]
    err = v - pred  # [N,1,Dv]
    delta = beta * err  # [N,1,1] * [N,1,Dv] -> [N,1,Dv]
    outer = torch.matmul(k.transpose(-1, -2), delta)  # [N,Dk,1] @ [N,1,Dv] -> [N,Dk,Dv]
    S_new = S_tilde + outer  # [N,Dk,Dv]
    out = torch.matmul(q, S_new)  # [N,1,Dk] @ [N,Dk,Dv] -> [N,1,Dv]
    return S_new, out


def _to_dev(t, device):
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


# Exact per-core shape from tt-blaze#2431: S_prev (128,32), g/k/q (128), v (32), beta (1).
@pytest.mark.parametrize("N", [1])
@pytest.mark.parametrize("Dk", [128])
@pytest.mark.parametrize("Dv", [32])
def test_flash_kda(N, Dk, Dv, device):
    torch.manual_seed(0)

    S_prev = torch.randn(N, Dk, Dv, dtype=torch.float32)
    g = torch.rand(N, Dk, 1, dtype=torch.float32)  # decay factor already applied upstream
    k = torch.randn(N, 1, Dk, dtype=torch.float32)
    v = torch.randn(N, 1, Dv, dtype=torch.float32)
    beta = torch.rand(N, 1, 1, dtype=torch.float32)
    q = torch.randn(N, 1, Dk, dtype=torch.float32)

    ref_S_new, ref_out = _flash_kda_reference(S_prev, g, k, v, beta, q)

    tt_S_new, tt_out = ttnn.transformer.flash_kda(
        _to_dev(S_prev, device),
        _to_dev(g, device),
        _to_dev(k, device),
        _to_dev(v, device),
        _to_dev(beta, device),
        _to_dev(q, device),
    )

    tt_S_new = ttnn.to_torch(ttnn.from_device(tt_S_new))
    tt_out = ttnn.to_torch(ttnn.from_device(tt_out))

    assert_with_pcc(ref_S_new, tt_S_new, 0.999)
    assert_with_pcc(ref_out, tt_out, 0.999)

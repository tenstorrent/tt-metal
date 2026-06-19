# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal isolated dtype probe for R1 (float32 + bfloat8_b).

One small tile-aligned MHA self-attention shape per dtype, no mask, auto
scale. Cheapest signal for whether the dtype lands at the descriptor level
before running the full golden suite. NOT an acceptance gate.
"""

from __future__ import annotations

import math

import pytest
import torch

import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _ref(Q, K, V, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    D = Qf.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(D)
    sc = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    w = torch.softmax(sc, dim=-1)
    return torch.matmul(w, Vf)


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize(
    "dt",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
        pytest.param(ttnn.bfloat8_b, id="bf8b"),
    ],
)
def test_dtype_probe(dt, device):
    shape = (1, 2, 128, 64)
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
    torch.manual_seed(0)
    Q = torch.randn(shape, dtype=tdt)
    K = torch.randn(shape, dtype=tdt)
    V = torch.randn(shape, dtype=tdt)
    exp = _ref(Q, K, V)

    tq = ttnn.from_torch(Q, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(tq, tk, tv)
    res = ttnn.to_torch(out).float()

    pcc = _pcc(exp, res)
    max_abs = (exp - res).abs().max().item()
    print(f"\n[probe] dt={dt} out.dtype={out.dtype} shape={list(out.shape)} PCC={pcc:.6f} max_abs={max_abs:.5f}")

    assert out.dtype == dt, f"output dtype {out.dtype} != input {dt}"
    assert pcc > 0.99, f"PCC too low: {pcc}"

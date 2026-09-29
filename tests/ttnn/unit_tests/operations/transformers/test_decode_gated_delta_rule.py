# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device test for fused T=1 ``ttnn.transformer.decode_gated_delta_rule``.

Two shapes (the silicon-proven decode geometries), bf16 TILE DRAM, against a
host golden of ``recurrent_gated_delta_rule_decode_ttnn``:

  B=1 H=32 K=32 V=32  and  B=1 H=24 K=128 V=128

pcc >= 0.99 per case on both outputs (o and new_state).
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _l2_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    # l2_norm_ttnn last-dim path: x * rsqrt(sum(x^2) + eps)
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def recurrent_gated_delta_rule_decode_golden(q, k, v, beta, g, scale=None, initial_state=None):
    """Host golden of recurrent_gated_delta_rule_decode_ttnn (ops.py:383-481)."""
    B, _, H, K = q.shape
    V = v.shape[-1]
    q, k, v, beta, g = (t.float() for t in (q, k, v, beta, g))
    q = _l2_norm(q, dim=-1)
    k = _l2_norm(k, dim=-1)
    if scale is None:
        scale = K**-0.5
    q = q * scale
    q_row = q.reshape(B, H, 1, K)
    k_row = k.reshape(B, H, 1, K)
    v_t = v.reshape(B, H, V)
    beta_t = beta.reshape(B, H)
    g_t = g.reshape(B, H)
    if initial_state is None:
        h = torch.zeros(B, H, K, V, dtype=torch.float32)
    else:
        h = initial_state.float()
    h = h * g_t[:, :, None, None].exp()
    v_read = k_row @ h
    delta = v_t.reshape(B, H, 1, V) - v_read
    # Canonical ttnn order: beta on the outer product, not folded into delta.
    outer = k_row.reshape(B, H, K, 1) @ delta
    h = h + beta_t[:, :, None, None] * outer
    o = (q_row @ h).reshape(B, 1, H, V)
    return o, h


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    am, bm = a.mean(), b.mean()
    num = ((a - am) * (b - bm)).sum()
    den = torch.sqrt(((a - am) ** 2).sum() * ((b - bm) ** 2).sum())
    if float(den) == 0.0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float(num / den)


def _to_torch_one_device(t):
    # MeshShape collect without a composer is pytensor.cpp:295, not an op fail.
    shards = ttnn.get_device_tensors(t)
    return ttnn.to_torch(shards[0]).float()


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="p300_1x2")], indirect=True)
def test_decode_gated_delta_rule_t1_vs_golden(mesh_device):
    assert hasattr(
        ttnn.transformer, "decode_gated_delta_rule"
    ), "ttnn.transformer.decode_gated_delta_rule is not bound in this tree"

    cases = ((1, 32, 32, 32, 0), (1, 24, 128, 128, 1))
    worst_o = worst_h = 1.0
    pairs = []
    for B, H, K, V, seed in cases:
        torch.manual_seed(seed)
        q = torch.randn(B, 1, H, K)
        k = torch.randn(B, 1, H, K)
        v = torch.randn(B, 1, H, V)
        beta = torch.rand(B, 1, H)
        g = -torch.rand(B, 1, H) * 0.5
        scale = K**-0.5
        h0 = torch.randn(B, H, K, V)
        gold_o, gold_h = recurrent_gated_delta_rule_decode_golden(q, k, v, beta, g, scale=scale, initial_state=h0)

        kw = dict(
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        o_t, h1_t = ttnn.transformer.decode_gated_delta_rule(
            ttnn.from_torch(q, **kw),
            ttnn.from_torch(k, **kw),
            ttnn.from_torch(v, **kw),
            ttnn.from_torch(beta, **kw),
            ttnn.from_torch(g, **kw),
            scale=scale,
            initial_state=ttnn.from_torch(h0, **kw),
        )
        o = _to_torch_one_device(o_t)
        h1 = _to_torch_one_device(h1_t)
        pcc_o = _pcc(o, gold_o)
        pcc_h = _pcc(h1, gold_h)
        print(
            f"I_GDN_DECODE case B={B} H={H} K={K} V={V} "
            f"o_shape={list(o.shape)} h_shape={list(h1.shape)} "
            f"pcc_o={pcc_o:.6f} pcc_h={pcc_h:.6f}"
        )
        worst_o, worst_h = min(worst_o, pcc_o), min(worst_h, pcc_h)
        pairs.append((gold_o, o, gold_h, h1))

    ok = worst_o >= 0.99 and worst_h >= 0.99
    print(
        f"I_GDN_DECODE: B=1 H=32/24 K=32/128 V=32/128 bf16 TILE DRAM "
        f"vs recurrent_gated_delta_rule_decode_golden "
        f"min_pcc_o={worst_o:.6f} min_pcc_h={worst_h:.6f} "
        f"VERDICT: {'PASS' if ok else 'FAIL'}"
    )
    for gold_o, o, gold_h, h1 in pairs:
        assert_with_pcc(gold_o, o, pcc=0.99)
        assert_with_pcc(gold_h, h1, pcc=0.99)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Minimal repro for the non-causal SDPA >32768 Q-length cliff (#48549 / W2b).

Sweeps the QUERY length of a maskless non-causal SDPA (square Q=K) across 32768
and compares the ttnn op against a tiled fp32 oracle. The denoise canvas (Q=256)
is unaffected — this isolates the cliff to the Q axis, which a non-causal long
prefill (Q>32768) would hit.

This is an opt-in diagnostic script, intentionally not collected by pytest:

  DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 pytest -q -s w2b_repro.py
"""
import os

import pytest
import torch
import ttnn

pytestmark = [
    pytest.mark.skipif(os.environ.get("DG_RUN_DEVICE") != "1", reason="device only"),
    pytest.mark.use_module_device,
]

HEAD_DIM = int(os.environ.get("W2B_HEAD_DIM", "256"))
Q_SWEEP = tuple(int(x) for x in os.environ.get("W2B_Q_SWEEP", "32768,65536,131072,262144").split(","))


def _tiled_fp32_oracle(q, k, v, k_chunk=2048, scale=1.0):
    # scale=1.0 matches the ttnn op call below (and the DiffusionGemma denoise path).
    rmax = torch.full(q.shape[:-1], -torch.inf)
    rsum = torch.zeros(q.shape[:-1])
    rout = torch.zeros_like(q, dtype=torch.float32)
    q = q.float()
    for s in range(0, k.shape[-2], k_chunk):
        kc, vc = k[:, :, s : s + k_chunk, :].float(), v[:, :, s : s + k_chunk, :].float()
        sc = torch.einsum("bhqd,bhkd->bhqk", q, kc) * scale
        cmax = sc.max(dim=-1).values
        nmax = torch.maximum(rmax, cmax)
        osc = torch.exp(rmax - nmax)
        e = torch.exp(sc - nmax.unsqueeze(-1))
        rout = rout * osc.unsqueeze(-1) + torch.einsum("bhqk,bhkd->bhqd", e, vc)
        rsum = rsum * osc + e.sum(dim=-1)
        rmax = nmax
    return rout / rsum.unsqueeze(-1)


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize("seq", Q_SWEEP)
def test_noncausal_sdpa_q_cliff(device, seq):
    torch.manual_seed(seq)
    q = torch.randn(1, 1, seq, HEAD_DIM)
    k = torch.randn(1, 1, seq, HEAD_DIM)
    v = torch.randn(1, 1, seq, HEAD_DIM)
    golden = _tiled_fp32_oracle(q, k, v)

    tt = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.transformer.scaled_dot_product_attention(tt(q), tt(k), tt(v), is_causal=False, scale=1.0)
    out = ttnn.to_torch(out)[:, :, :seq, :]
    pcc = _pcc(golden, out)
    print(f"\n[w2b-repro] non-causal SDPA Q=K={seq} head_dim={HEAD_DIM} -> PCC={pcc:.5f}")
    assert pcc > 0.99, f"non-causal SDPA wrong at seq={seq}: PCC={pcc:.5f}"

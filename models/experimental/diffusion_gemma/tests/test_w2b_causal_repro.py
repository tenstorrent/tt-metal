# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does regular (non-chunked) CAUSAL SDPA have a >32768 Q correctness cliff?

Tests the claim in chunked_prefill_sdpa's docstring: that
`ttnn.transformer.scaled_dot_product_attention(is_causal=True)` "silently returns
wrong results" once Q > 32768. Square Q=K, valid chunked program config
(q=256,k=128), vs a tiled fp32 causal oracle.

  DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 W2B_SEQ=8192,32768,33024,65536 pytest -q -s
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
SEQ_SWEEP = tuple(int(x) for x in os.environ.get("W2B_SEQ", "8192,32768,33024,65536").split(","))
GRID = ttnn.CoreCoord(8, 8) if HEAD_DIM < 512 else ttnn.CoreCoord(8, 4)


def _tiled_causal_oracle(q, k, v, k_chunk=2048, scale=1.0):
    """fp32 online-softmax causal oracle (q_pos attends k_pos <= q_pos)."""
    sq = q.shape[-2]
    q_pos = torch.arange(sq).view(1, 1, sq, 1)
    rmax = torch.full(q.shape[:-1], -torch.inf)
    rsum = torch.zeros(q.shape[:-1])
    rout = torch.zeros_like(q, dtype=torch.float32)
    qf = q.float()
    for s in range(0, k.shape[-2], k_chunk):
        e = min(s + k_chunk, k.shape[-2])
        kc, vc = k[:, :, s:e, :].float(), v[:, :, s:e, :].float()
        sc = torch.einsum("bhqd,bhkd->bhqk", qf, kc) * scale
        k_pos = torch.arange(s, e).view(1, 1, 1, e - s)
        sc = sc.masked_fill(k_pos > q_pos, -torch.inf)  # causal
        cmax = sc.max(dim=-1).values
        nmax = torch.maximum(rmax, cmax)
        osc = torch.exp(rmax - nmax)
        ex = torch.exp(sc - nmax.unsqueeze(-1))
        rout = rout * osc.unsqueeze(-1) + torch.einsum("bhqk,bhkd->bhqd", ex, vc)
        rsum = rsum * osc + ex.sum(dim=-1)
        rmax = nmax
    return rout / rsum.unsqueeze(-1)


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize("seq", SEQ_SWEEP)
def test_causal_sdpa_past_32768(device, seq):
    torch.manual_seed(seq)
    q = torch.randn(1, 1, seq, HEAD_DIM)
    k = torch.randn(1, 1, seq, HEAD_DIM)
    v = torch.randn(1, 1, seq, HEAD_DIM)
    golden = _tiled_causal_oracle(q, k, v)

    # Match production chunked_prefill_sdpa tiles: head_dim>=512 -> q=128 on (8,4)
    # grid (q=256 overflows L1 there); head_dim<=256 -> q=256 on (8,8).
    q_chunk = 128 if HEAD_DIM >= 512 else 256
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=GRID, q_chunk_size=q_chunk, k_chunk_size=128, exp_approx_mode=False
    )
    tt = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.transformer.scaled_dot_product_attention(
        tt(q), tt(k), tt(v), is_causal=True, scale=1.0, program_config=cfg
    )
    out = ttnn.to_torch(out)[:, :, :seq, :]
    pcc = _pcc(golden, out)
    print(f"\n[causal-repro] CAUSAL SDPA Q=K={seq} d={HEAD_DIM} -> PCC={pcc:.5f}", flush=True)

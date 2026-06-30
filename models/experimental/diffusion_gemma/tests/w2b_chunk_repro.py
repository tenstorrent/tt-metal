# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chunked vs non-chunked non-causal SDPA across the 32768 boundary (#48549).

Same maskless non-causal SDPA (square Q=K), two program configs:
  - chunked     : small q/k chunk (512) -> op flash-tiles internally
  - non-chunked : single q chunk == seq (>= 32768) -> the documented cliff

Reports PCC for each vs a tiled fp32 oracle so we can see exactly which knob
triggers the >32768 garbage.

This is an opt-in diagnostic script, intentionally not collected by pytest:

  DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 pytest -q -s w2b_chunk_repro.py
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
SEQ_SWEEP = tuple(int(x) for x in os.environ.get("W2B_SEQ", "8192,32768,65536").split(","))
GRID = ttnn.CoreCoord(8, 8) if HEAD_DIM < 512 else ttnn.CoreCoord(8, 4)


def _oracle(q, k, v, k_chunk=2048, scale=1.0):
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


def _run(device, q, k, v, q_chunk, k_chunk):
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=GRID,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    tt = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        out = ttnn.transformer.scaled_dot_product_attention(
            tt(q), tt(k), tt(v), is_causal=False, scale=1.0, program_config=cfg
        )
        return ttnn.to_torch(out)[:, :, : q.shape[-2], :]
    except Exception as e:  # noqa: BLE001 - diagnostic records op-level rejections
        return repr(e)


@pytest.mark.parametrize("seq", SEQ_SWEEP)
def test_chunked_vs_nonchunked(device, seq):
    torch.manual_seed(seq)
    q = torch.randn(1, 1, seq, HEAD_DIM)
    k = torch.randn(1, 1, seq, HEAD_DIM)
    v = torch.randn(1, 1, seq, HEAD_DIM)
    golden = _oracle(q, k, v)

    chunked = _run(device, q, k, v, q_chunk=256, k_chunk=128)
    nonchunked = _run(device, q, k, v, q_chunk=seq, k_chunk=128)

    cp = f"{_pcc(golden, chunked):.5f}" if torch.is_tensor(chunked) else f"ERR:{chunked[:90]}"
    np_ = f"{_pcc(golden, nonchunked):.5f}" if torch.is_tensor(nonchunked) else f"ERR:{nonchunked[:90]}"
    print(f"\n[chunk-repro] seq={seq} d={HEAD_DIM}  chunked(q=256,k=128)={cp}  non-chunked(q={seq},k=128)={np_}")

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single op-only ttnn.experimental.indexer_score_dsa data point (TEN-4859 context).

DeepSeek-like indexer score: q [1,Hi,Sq,D], k [1,1,T,D], chunk_start = T - Sq. The learned-gate weights
layout changed with PR 55614 (merged 2026-09-08): the op now takes the projection's native [B, 1, Sq, Hi]
and the permute is gone, where 73e0e25 and the 2026-07-22 kernels took [B, Hi, Sq, 1]. IX_WLAYOUT picks
which one the probe builds, so the same script runs on both checkouts: "native" (default, main tip) or
"permuted" (the calibration checkout).

Env: IX_HEADS (64), IX_D (128), IX_SQ (2048), IX_T (8192), IX_ITERS (4), IX_WLAYOUT (native|permuted).
"""
from __future__ import annotations

import os


def test_indexer_probe(device):
    import torch
    import ttnn

    hi = int(os.environ.get("IX_HEADS", "64"))
    d = int(os.environ.get("IX_D", "128"))
    sq = int(os.environ.get("IX_SQ", "2048"))
    t = int(os.environ.get("IX_T", "8192"))
    iters = int(os.environ.get("IX_ITERS", "4"))
    wlayout = os.environ.get("IX_WLAYOUT", "native")
    chunk_start = t - sq

    q = torch.randn(1, hi, sq, d, dtype=torch.bfloat16)
    k = torch.randn(1, 1, t, d, dtype=torch.bfloat16)
    w = (
        torch.randn(1, 1, sq, hi, dtype=torch.bfloat16)
        if wlayout == "native"
        else torch.randn(1, hi, sq, 1, dtype=torch.bfloat16)
    )
    tt_q = ttnn.from_torch(q, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_k = ttnn.from_torch(k, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_w = ttnn.from_torch(w, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    print(
        f"\n[indexer_probe] Hi={hi} D={d} Sq={sq} T={t} chunk_start={chunk_start} " f"w={tuple(w.shape)} ({wlayout})",
        flush=True,
    )
    for _ in range(iters):
        out = ttnn.experimental.indexer_score_dsa(tt_q, tt_k, tt_w, chunk_start_idx=chunk_start)
        ttnn.synchronize_device(device)
        out.deallocate()
    for x in (tt_q, tt_k, tt_w):
        x.deallocate()
    print("[indexer_probe] OK", flush=True)

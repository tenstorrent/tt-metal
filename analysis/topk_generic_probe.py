# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generic ttnn.topk (multi core factory) timing probe. Env: TG_ROWS (32), TG_N (16384), TG_K (32), TG_ITERS (6),
TG_LARGEST (1). Run under tracy and read the TopK op duration."""
import os

import torch


def test_topk_generic_probe(device):
    import ttnn

    rows = int(os.environ.get("TG_ROWS", "32"))
    n = int(os.environ.get("TG_N", "16384"))
    k = int(os.environ.get("TG_K", "32"))
    iters = int(os.environ.get("TG_ITERS", "6"))
    largest = os.environ.get("TG_LARGEST", "1") == "1"
    # TG_LOCAL_X x TG_LOCAL_Y pins the local core count the way the tree merge test does (one spare column,
    # two spare rows for the final core)
    lx, ly = int(os.environ.get("TG_LOCAL_X", "0")), int(os.environ.get("TG_LOCAL_Y", "0"))
    sub = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(lx, ly + 1))]) if lx and ly else None
    torch.manual_seed(1)
    x = torch.randn(1, 1, rows, n, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(f"\n[topk_generic] rows={rows} N={n} k={k} largest={largest} local={lx}x{ly}", flush=True)
    for _ in range(iters):
        vals, idx = ttnn.topk(tt_x, k=k, dim=-1, largest=largest, sub_core_grids=sub)
        ttnn.synchronize_device(device)
        vals.deallocate()
        idx.deallocate()
    ref_v, _ = torch.topk(x.float(), k, dim=-1, largest=largest)
    vals, idx = ttnn.topk(tt_x, k=k, dim=-1, largest=largest, sub_core_grids=sub)
    got = ttnn.to_torch(vals).float()
    assert torch.allclose(got.sort(dim=-1).values, ref_v.sort(dim=-1).values.to(torch.bfloat16).float(), atol=0.05, rtol=0.02)
    print("[topk_generic] OK", flush=True)

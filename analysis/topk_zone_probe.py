# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generic ttnn.topk on one core, for the L7 zone decomposition (TEN-4859).

Same call shape as the campaign's single-core generic cells (topk_campaign.py run_topk with
grid="single"): a [1, 1, rows, N] bf16 TILE tensor in DRAM, dim -1, largest, sorted, sub_core_grids
one core. The point of this probe is the row axis on ONE core: with TKP_ROWS 64 the kernel runs two
TK_ROW zones in the same launch, so the first row and the second row can be compared and the Ht-law
intercept (c_launch_g, 7.2 us) can be attributed.

Env: TKP_ROWS (default 32, a multiple of 32), TKP_N (4096), TKP_K (32), TKP_ITERS (3; the first is
discarded in analysis as everywhere in this campaign).
"""
from __future__ import annotations

import os

import torch

ROWS = int(os.environ.get("TKP_ROWS", "32"))
N = int(os.environ.get("TKP_N", "4096"))
K = int(os.environ.get("TKP_K", "32"))
ITERS = int(os.environ.get("TKP_ITERS", "3"))


def test_topk_zone_probe(device):
    import ttnn

    torch.manual_seed(0)
    x = torch.randn(1, 1, ROWS, N, dtype=torch.float32)
    tx = ttnn.from_torch(
        x, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    one_core = ttnn.num_cores_to_corerangeset(1, device.compute_with_storage_grid_size(), row_wise=True)
    print(f"\n[topk_zone_probe] rows={ROWS} N={N} K={K} iters={ITERS} cores=1", flush=True)
    for it in range(ITERS):
        out = ttnn.topk(tx, K, dim=-1, largest=True, sorted=True, sub_core_grids=one_core)
        ttnn.synchronize_device(device)
        for t in out if isinstance(out, (list, tuple)) else [out]:
            t.deallocate()
        print(f"[topk_zone_probe] iter {it} done", flush=True)
    tx.deallocate()

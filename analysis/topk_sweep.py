# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Op-only ttnn.experimental.topk_large_indices sweep (TEN-4859).

Mirrors analysis/sparse_sweep.py: builds a [1, 1, R, N] bf16 ROW_MAJOR interleaved
logits tensor (the shape the DeepSeek indexer feeds it, models/demos/deepseek_v3_d_p/
tt/mla/indexer.py), calls the op in a loop with no golden, under tracy device profiling
(plain single pass for durations, or --perf-counter-multipass for engine counters).

Constraints honored (topk_large_indices_device_operation.cpp): Blackhole only, ROW_MAJOR
bf16 interleaved input, k in [16, 2048] multiple of 16, k <= N. The kernel snaps k up to
an LLK bucket K' in {512, 1024, 2048} and scans the row in ceil(N/K') chunks, so the
sweep includes non-power-of-two N (5000/5300) straddling a K'=512 chunk boundary and
single-chunk N (256/512) where only the local sort plus index rebuild runs.

Env: TK_ROWS (single int), TK_NS (csv), TK_KS (csv), TK_ITERS (default 6; analysis
discards iteration 1), TK_MEM (dram|l1, default dram; L1-interleaved input is the
DRAM-bound vs SFPU-bound disambiguator at fixed shape).
Run one TK_ROWS/TK_MEM value per tracy invocation so run-id -> config mapping stays
trivial (ascending run host IDs = parametrize order x iters).
"""
from __future__ import annotations

import os

import pytest

_NS = [int(x) for x in os.environ.get("TK_NS", "256,512,1024,4096,5000,5300,16384,65536,131072").split(",")]
_KS = [int(x) for x in os.environ.get("TK_KS", "16,32,64,256,512,1024,2048").split(",")]
_COMBOS = [(n, k) for n in _NS for k in _KS if k <= n]


@pytest.mark.parametrize("n,k", _COMBOS)
def test_topk(device, n, k):
    import torch
    import ttnn

    r = int(os.environ.get("TK_ROWS", "1"))
    iters = int(os.environ.get("TK_ITERS", "6"))
    mem = os.environ.get("TK_MEM", "dram")
    memcfg = ttnn.L1_MEMORY_CONFIG if mem == "l1" else ttnn.DRAM_MEMORY_CONFIG

    grid = device.compute_with_storage_grid_size()
    x = torch.randn(1, 1, r, n, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memcfg,
    )
    print(f"\n[topk_sweep] R={r} N={n} K={k} mem={mem} grid={grid.x}x{grid.y}", flush=True)
    for _ in range(iters):
        idx = ttnn.experimental.topk_large_indices(tt_x, k=k)
        ttnn.synchronize_device(device)
        idx.deallocate()
    tt_x.deallocate()
    print(f"[topk_sweep] OK R={r} N={n} K={k}", flush=True)

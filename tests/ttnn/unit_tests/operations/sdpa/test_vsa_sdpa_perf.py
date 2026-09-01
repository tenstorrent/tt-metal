# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""vsa_sdpa microbenchmark: single device, host-timed (op time >> dispatch), production shard shapes.

Prints per-variant time, effective FLOPs, and math utilization against the tt-perf-report
Blackhole peak (4096 FLOP/cyc x 1.35 GHz / 2 for HiFi2). Run with -s.
"""

import time

import pytest
import torch

import ttnn
from models.common.utility_functions import skip_for_wormhole_b0

BLOCK = 64
SENTINEL = 0xFFFFFFFF
HEADS = 14
DIM = 128


def make_inputs(device, s_local, n_blocks, row_blocks, dense_rows, order, seed=0):
    """Worst-shard-shaped inputs: `dense_rows` rows list every block, the rest list `row_blocks`."""
    torch.manual_seed(seed)
    n_q = s_local // BLOCK
    kv_len = n_blocks * BLOCK
    w = ((n_blocks + 15) // 16) * 16
    q = torch.randn(1, HEADS, s_local, DIM, dtype=torch.bfloat16)
    k = torch.randn(1, HEADS, kv_len, DIM, dtype=torch.bfloat16)
    v = torch.randn(1, HEADS, kv_len, DIM, dtype=torch.bfloat16)
    idx = torch.full((1, HEADS, n_q, w), SENTINEL, dtype=torch.int64)
    gen = torch.Generator().manual_seed(seed + 1)
    for h in range(HEADS):
        for r in range(n_q):
            if r < dense_rows:
                listed = torch.arange(n_blocks)
            elif order == "topk":
                listed = torch.randperm(n_blocks, generator=gen)[:row_blocks]
            elif order == "sorted":
                listed = torch.randperm(n_blocks, generator=gen)[:row_blocks].sort().values
            elif order == "model":
                # Realistic VSA selection: every row lists the exempt prefix (~13% of the budget),
                # and neighboring q-tiles' top-k sets are spatially correlated -- scores are a
                # smooth per-block field plus per-row noise, like real coarse-stage attention maps.
                n_exempt = max(1, row_blocks // 8)
                if r % 64 == 0 or "model_field" not in locals():
                    model_field = torch.randn(n_blocks, generator=gen)
                block_of_row = int(r * n_blocks / max(1, n_q))
                dist = (torch.arange(n_blocks) - block_of_row).abs().float() / n_blocks
                row_scores = model_field - 3.0 * dist + 0.35 * torch.randn(n_blocks, generator=gen)
                row_scores[:n_exempt] = float("inf")  # exempt prefix always listed
                listed = row_scores.topk(row_blocks).indices
            else:  # sequential: contiguous run, maximal DRAM locality
                start = int(torch.randint(0, n_blocks - row_blocks, (1,), generator=gen))
                listed = torch.arange(start, start + row_blocks)
            idx[0, h, r, : listed.numel()] = listed
    counts = torch.full((1, 1, 1, w), BLOCK, dtype=torch.int32)
    counts[..., n_blocks:] = 0

    tt = lambda x: ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    visits = HEADS * (dense_rows * n_blocks + (n_q - dense_rows) * row_blocks)
    flops = visits * 4 * BLOCK * BLOCK * DIM
    return (
        tt(q),
        tt(k),
        tt(v),
        ttnn.from_torch(
            idx.to(torch.uint32).view(torch.int32), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
        ),
        ttnn.from_torch(counts, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32),
        flops,
    )


def bench(device, args, m, iters=8, streaming=False):
    q, k, v, idx, counts, flops = args
    out = ttnn.transformer.vsa_sdpa(q, k, v, idx, counts, k_chunk_blocks=m, streaming=streaming)  # compile
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = ttnn.transformer.vsa_sdpa(q, k, v, idx, counts, k_chunk_blocks=m, streaming=streaming)
    ttnn.synchronize_device(device)
    ms = (time.perf_counter() - t0) / iters * 1e3
    ttnn.deallocate(out)
    grid = device.compute_with_storage_grid_size()
    peak = grid.x * grid.y * 4096 * 1.35e9 / 2
    util = flops / (ms * 1e-3) / peak * 100
    return ms, util, (grid.x, grid.y)


@skip_for_wormhole_b0("vsa_sdpa is Blackhole-only")
def test_vsa_sdpa_bench(device):
    # 15s/768p production shard: 226 q-tiles/device, 1808 global blocks, 197 listed/row.
    # median shard: no dense rows; worst shard (identity placement): 18 dense rows.
    cases = [
        ("median 15s topk", dict(s_local=14464, n_blocks=1808, row_blocks=197, dense_rows=0, order="topk")),
        ("median 15s model", dict(s_local=14464, n_blocks=1808, row_blocks=197, dense_rows=0, order="model")),
        ("worst  15s model", dict(s_local=14464, n_blocks=1808, row_blocks=197, dense_rows=18, order="model")),
        ("median 10s model", dict(s_local=9216, n_blocks=1152, row_blocks=125, dense_rows=0, order="model")),
        ("median 5s  topk", dict(s_local=4800, n_blocks=688, row_blocks=80, dense_rows=0, order="topk")),
        ("median 5s  model", dict(s_local=4800, n_blocks=688, row_blocks=80, dense_rows=0, order="model")),
    ]
    print()
    for label, spec in cases:
        args = make_inputs(device, **spec)
        for mode, m in (("v1", 2), ("stream", 1)):
            ms, util, grid = bench(device, args, m, streaming=(mode == "stream"))
            print(f"{label}  {mode:<6s}  {ms:8.3f} ms   util {util:5.2f} %   grid {grid}")
        for t in args[:5]:
            ttnn.deallocate(t)

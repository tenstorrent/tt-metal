# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What running every local expert in ONE program actually buys.

The main perf matrix pins experts_per_chip=1, so it measures the kernel and not
the dispatch model. This sweeps the expert count and times two ways of doing the
same work:

  fused   -- one call, N-element weight lists
  looped  -- N calls, one-element lists each (what the per-expert op forced)

Both enqueue identical device work, so the difference is per-dispatch overhead
and the inter-op gap. Wall clock over a saturated queue is the metric: device
kernel duration cannot see a dispatch gap.

The `sparse` case is the one the hybrid routing depends on -- N experts declared,
only some with a non-zero count. Zero-count experts are skipped device-side, so
`fused` should approach the cost of the live experts alone while `looped` still
pays N launches.
"""

import os
import time

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole

TILE = 32
EMB = 7168
HIDDEN = 2048
GRID = ttnn.CoreCoord(11, 8)
NUM_ROUTED_EXPERTS = 384
CAPACITY = 512
REPS = int(os.environ.get("MOE_FUSED_SWIGLU_SWEEP_REPS", "20"))
EXPERT_COUNTS = tuple(int(v) for v in os.environ.get("MOE_FUSED_SWIGLU_SWEEP_EXPERTS", "1,2,4,8").split(","))
TOKEN_COUNTS = tuple(int(v) for v in os.environ.get("MOE_FUSED_SWIGLU_SWEEP_TOKENS", "128,256,512").split(","))


def _to_device(tensor, dtype, layout, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(tensor.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _bench(call, device):
    """Per-iteration wall clock with the queue saturated, warmup excluded."""
    call()
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(REPS):
        call()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - start) / REPS * 1e6  # us


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="moe_fused_swiglu is Blackhole-only")
def test_experts_per_chip_sweep(device):
    torch.manual_seed(20260827)
    op = ttnn.experimental.deepseek_prefill.moe_fused_swiglu
    max_experts = max(EXPERT_COUNTS)

    # One weight set, reused by every expert slot. Placement and shape are what drive the
    # schedule; distinct values would only cost DRAM and slow the upload.
    host = (
        torch.randn((EMB, HIDDEN), dtype=torch.bfloat16) * 2.0e-2,
        torch.randn((EMB, HIDDEN), dtype=torch.bfloat16) * 2.0e-2,
        torch.randn((HIDDEN, EMB), dtype=torch.bfloat16) * 2.0e-2,
    )
    gates, ups, downs = ([], [], [])
    for _ in range(max_experts):
        gates.append(_to_device(host[0], ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device))
        ups.append(_to_device(host[1], ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device))
        downs.append(_to_device(host[2], ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device))

    total_rows = CAPACITY * max_experts
    x = _to_device(
        torch.randn((1, 1, total_rows, EMB), dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device
    )
    out = ttnn.empty(
        [1, 1, total_rows, EMB],
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    global_ids = [(11 + 37 * e) % NUM_ROUTED_EXPERTS for e in range(max_experts)]
    idx_all = _to_device(torch.tensor(global_ids, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    offsets_host = torch.zeros(NUM_ROUTED_EXPERTS, dtype=torch.int32)
    for e, gid in enumerate(global_ids):
        offsets_host[gid] = e * CAPACITY
    offsets = _to_device(offsets_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)

    def counts_tensor(per_expert):
        host_counts = torch.zeros(NUM_ROUTED_EXPERTS, dtype=torch.int32)
        for e, c in enumerate(per_expert):
            host_counts[global_ids[e]] = c
        return _to_device(host_counts, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)

    # The idx table a single-expert call needs: local 0 must map to expert e's global id.
    idx_one = [
        _to_device(torch.tensor([global_ids[e]], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
        for e in range(max_experts)
    ]

    common = dict(
        input_m_tiles=CAPACITY // TILE, core_grid=GRID, output=out, expert_region_offsets=offsets, read_x_at_offset=True
    )

    rows = []
    for n in EXPERT_COUNTS:
        for tokens in TOKEN_COUNTS:
            live = counts_tensor([tokens] * n + [0] * (max_experts - n))
            fused_us = _bench(lambda: op(x, gates[:n], ups[:n], downs[:n], live, idx_all, **common), device)

            def looped():
                for e in range(n):
                    op(x, [gates[e]], [ups[e]], [downs[e]], live, idx_one[e], **common)

            looped_us = _bench(looped, device)
            rows.append((n, tokens, "dense", fused_us, looped_us))

    # Hybrid routing: N slots declared, only every other expert live.
    for n in EXPERT_COUNTS:
        if n < 2:
            continue
        for tokens in TOKEN_COUNTS:
            per = [tokens if e % 2 == 0 else 0 for e in range(n)] + [0] * (max_experts - n)
            sparse = counts_tensor(per)
            fused_us = _bench(lambda: op(x, gates[:n], ups[:n], downs[:n], sparse, idx_all, **common), device)

            def looped_sparse():
                for e in range(n):
                    op(x, [gates[e]], [ups[e]], [downs[e]], sparse, idx_one[e], **common)

            looped_us = _bench(looped_sparse, device)
            rows.append((n, tokens, f"sparse({n // 2}/{n} live)", fused_us, looped_us))

    print(f"\n{'experts':>7} {'tok/exp':>7} {'fill':>16} {'fused us':>10} {'looped us':>10} {'speedup':>8}")
    for n, tokens, fill, f_us, l_us in rows:
        print(f"{n:>7} {tokens:>7} {fill:>16} {f_us:>10.1f} {l_us:>10.1f} {l_us / f_us:>7.2f}x")

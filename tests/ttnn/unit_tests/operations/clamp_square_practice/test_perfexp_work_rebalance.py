# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""work_rebalance perf experiment: work-distribution variants of clamp_square_practice.

Env filters (run_safe_pytest word-splits -k):
  WR_VARIANTS=baseline,dyn_k1   WR_SHAPES=1x1x4096x4096   WR_STATS=1 (dyn: per-core claim counts)
Correctness is bit-exact vs torch bf16 clamp(x,-1,1)**2 (output pre-filled with NaN so a missed block
fails). Perf is never asserted; run under --profile for DEVICE KERNEL DURATION.
"""
import os

import pytest
import torch
import ttnn

from ttnn.operations.clamp_square_practice.perf_experiments.work_rebalance import descriptor as wr

VARIANTS = [
    "baseline",
    "strided",
    "dyn_k1",
    "dyn_k2",
    "dyn_k4",
    "cores_8x8",
    "cores_10x8",
    "cores_10x10",
]
SHAPES = [(1, 1, 4096, 4096), (8, 1, 1024, 1024), (2, 4, 256, 512), (1, 1, 32, 8192), (1, 1, 32, 32)]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize(
    "variant",
    VARIANTS
    + [
        "dyn_k1_g8x8",
        "dyn_k2_g10x10",
        "dyn_k8",
        "dyn_k16",
        "skew",
        "dyn_k1_jit",
        "dyn_k1_b4",
        "dyn_k1_b4_jit",
        "dyn_k1_b2_jit",
        "dyn_k1_g8x8_jit",
        "guided_f1",
        "guided_f2",
        "guided_f4",
        "dyn_k1_b1_jit",
        "dyn_k1_b1_g10x10_jit",
        "dyn_k1_b1_g8x8_jit",
    ],
)
def test_work_rebalance(device, shape, variant):
    vs, ss = os.environ.get("WR_VARIANTS"), os.environ.get("WR_SHAPES")
    if (vs and variant not in vs.split(",")) or (ss and "x".join(map(str, shape)) not in ss.split(",")):
        pytest.skip("filtered")
    torch.manual_seed(0)
    xt = torch.randn(shape, dtype=torch.bfloat16) * 2
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    reps = int(os.environ.get("WR_REPEAT", "1"))  # >1: stress the dynamic scheme for races
    for rep in range(reps):
        out = ttnn.from_torch(
            torch.full(shape, float("nan"), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        stats = None
        want_stats = os.environ.get("WR_STATS") == "1" and variant.startswith(("dyn", "guided"))
        if want_stats:
            grid = device.compute_with_storage_grid_size()
            stats = ttnn.from_torch(
                torch.zeros((2 * grid.x * grid.y, 256), dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        desc, info = wr.build(variant, x, out, -1.0, 1.0, stats=stats)
        ttnn.generic_op([x, out], desc)
        ttnn.synchronize_device(device)
        print(f"BENCH_ROW variant={variant} shape={shape} info={info}")

        got = ttnn.to_torch(out)
        ref = torch.clamp(xt.float(), -1.0, 1.0).pow(2).to(torch.bfloat16)
        nmis = int((got.view(torch.int16) != ref.view(torch.int16)).sum())
        if want_stats:
            st = ttnn.to_torch(stats).to(torch.int64)
            counts = st[:, 0]
            n = info["cores"]
            G = st.shape[0] // 2
            claims = [v for i in range(n) for v in st[i, 1 : 1 + min(255, int(st[i, 0]))].tolist()]
            writes = [v for i in range(n) for v in st[G + i, 1 : 1 + min(255, int(st[G + i, 0]))].tolist()]
            from collections import Counter

            dup_c = [k for k, c in Counter(claims).items() if c > 1]
            dup_w = [k for k, c in Counter(writes).items() if c > 1]
            if dup_c or dup_w or sorted(claims) != sorted(writes):
                wc = [int(st[G + i, 0]) for i in range(n)]
                print(
                    f"CLAIMLOG variant={variant} rep={rep} dup_claims={dup_c[:8]} dup_writes={dup_w[:8]} "
                    f"claims_vs_writes_equal={sorted(claims) == sorted(writes)} rd_counts_sum={int(counts[:n].sum())} wr_counts_sum={sum(wc)}"
                )
                for i in range(n):
                    rc = st[i, 1 : 1 + min(255, int(st[i, 0]))].tolist()
                    wl = st[G + i, 1 : 1 + min(255, int(st[G + i, 0]))].tolist()
                    if rc != wl or any(v in dup_c for v in rc):
                        print(f"  core {i}: claims={rc} writes={wl}")
            print(
                f"STATS variant={variant} shape={shape} sum={int(counts[:n].sum())} nb={info['nb']} counts={counts[:n].tolist()}"
            )
            assert (
                variant.startswith("guided") or int(counts[:n].sum()) == info["nb"]
            ), "claim total != num_blocks (duplicate or missed claims)"
        if nmis:
            W = shape[-1]
            bad = (got.view(torch.int16) != ref.view(torch.int16)).reshape(-1, shape[-1])
            rows, cols = bad.nonzero(as_tuple=True)
            tiles = sorted({int(r // 32) * (W // 32) + int(c // 32) for r, c in zip(rows.tolist(), cols.tolist())})
            nans = int(torch.isnan(got.float()).sum())
            print(f"MISMATCH variant={variant} shape={shape} rep={rep} tiles={tiles[:16]} nan_elems={nans}")
        assert nmis == 0, f"rep {rep}: {nmis} mismatching elements vs torch"

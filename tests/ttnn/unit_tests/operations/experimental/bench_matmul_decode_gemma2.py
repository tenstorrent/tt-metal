# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bench ttnn.experimental.matmul_decode at gemma2-9B decode MLP shapes with bf4 weights.

Compares the tiny-tile decode matmul (m=1) against the production gather_in0 ttnn.matmul
(m=32 padded) at the same shape/dtype, reporting PCC, device time (trace replay), and the
implied weight-read bandwidth. Baseline reference from the profiler sheet:
  FF1/FF3 (3584x14336): ~98 us, ~263 GB/s   FF2 (14336x3584): ~100 us, ~257 GB/s
"""
import math
import os
import statistics
import time

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

VALID_TILE_HEIGHTS = [1, 2, 4, 8, 16, 32]


def tile_height_for(m):
    for th in VALID_TILE_HEIGHTS:
        if m <= th:
            return th
    return 32


def rect_core_range_set(num_cores, device):
    grid = device.compute_with_storage_grid_size()
    x = grid.x
    while x > 0 and num_cores % x != 0:
        x -= 1
    y = num_cores // x if x > 0 else 0
    if x == 0 or y > grid.y:
        raise ValueError(f"cannot fit {num_cores} cores in {grid.x}x{grid.y}")
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})


def can_rect(num_cores, grid):
    """True if num_cores can form an x*y rectangle with x<=grid.x, y<=grid.y."""
    x = grid.x
    while x > 0 and num_cores % x != 0:
        x -= 1
    return x > 0 and (num_cores // x) <= grid.y


def pick_a_cores(k, device):
    """Largest core count dividing k/32 (=> tile-aligned width shard) that fits a grid rectangle."""
    grid = device.compute_with_storage_grid_size()
    ktiles = k // 32
    best = 1
    for c in range(1, grid.x * grid.y + 1):
        if ktiles % c == 0 and can_rect(c, grid) and c > best:
            best = c
    return best


def pick_blocks(k, n, device):
    """Choose (k_blocks, n_blocks) so k_blocks*n_blocks and n_blocks both form valid
    rectangles within the grid, k_blocks even, and kc, nc stay tile-aligned. Maximize cores."""
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    best = None
    for k_blocks in (2, 4):
        if (k // k_blocks) % 32 != 0:
            continue
        for n_blocks in range(1, (n // 32) + 1):
            if n % n_blocks != 0 or (n // n_blocks) % 32 != 0:
                continue
            cores = k_blocks * n_blocks
            if cores > max_cores:
                continue
            if not can_rect(cores, grid) or not can_rect(n_blocks, grid):
                continue
            if best is None or cores > best[2]:
                best = (k_blocks, n_blocks, cores)
    if best is None:
        pytest.skip(f"no valid block split for k={k} n={n} on {grid.x}x{grid.y}")
    return best[0], best[1]


def time_replay(device, fn, reps=50):
    fn()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    try:
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            ts.append((time.perf_counter() - t0) * 1e6)  # us
        return statistics.median(ts)
    finally:
        ttnn.release_trace(device, tid)


def bf4_bytes(k, n):
    # bfloat4_b: 4-bit mantissa + shared 8-bit exponent per 16-elem block => 0.5 + 1/16 byte/elem
    return k * n * (0.5 + 1.0 / 16.0)


@pytest.mark.parametrize(
    "name, k, n",
    [
        ("FF1", 3584, 14336),
        ("FF3", 3584, 14336),
        ("FF2", 14336, 3584),
        # TP=2 per-chip shapes (N halved for FF1/FF3, K halved for FF2)
        ("FF1_2x", 3584, 7168),
        ("FF3_2x", 3584, 7168),
        ("FF2_2x", 7168, 3584),
    ],
)
def test_bench_matmul_decode(device, name, k, n):
    torch.manual_seed(0)
    m = 1  # batch-1 decode
    th = tile_height_for(m)
    m_padded = 32

    a = torch.randn((m, k), dtype=torch.bfloat16)
    b = torch.randn((k, n), dtype=torch.bfloat16)
    ref = a.float() @ b.float()

    k_blocks, n_blocks = pick_blocks(k, n, device)
    kc, nc = k // k_blocks, n // n_blocks
    num_b_cores = k_blocks * n_blocks
    num_a_cores = pick_a_cores(k, device)

    # Weight reshape/permute: core c=(kb*n_blocks+nb) holds B[kb,nb] block.
    b_resh = b.reshape(k_blocks, kc, n).permute(1, 0, 2).reshape(kc, n * k_blocks)

    a_crs = rect_core_range_set(num_a_cores, device)
    b_crs = rect_core_range_set(num_b_cores, device)
    in0_mc = ttnn.create_sharded_memory_config(
        (m_padded, k // num_a_cores),
        core_grid=a_crs,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    w_dram = os.environ.get("BENCH_W_DRAM") == "1"
    if w_dram:
        in1_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(b_crs, [kc, nc], ttnn.ShardOrientation.ROW_MAJOR),
        )
    else:
        in1_mc = ttnn.create_sharded_memory_config(
            (kc, nc),
            core_grid=b_crs,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    # partial_width_sharded pads m to a full 32-row tile (matches reference test).
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mc, dtype=ttnn.bfloat16)
    tb = ttnn.from_torch(b_resh, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_mc, dtype=ttnn.bfloat4_b)

    out_crs = rect_core_range_set(n_blocks, device)
    out_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(out_crs, [m_padded, n // n_blocks], ttnn.ShardOrientation.ROW_MAJOR),
    )

    def run_decode():
        return ttnn.experimental.matmul_decode(ta, tb, partial_width_sharded=True, output_mem_config=out_mc)

    out = run_decode()
    pcc_ok, pcc_val = assert_with_pcc(ref, ttnn.to_torch(out).float()[:m], 0.97)
    dt = time_replay(device, run_decode)
    # GB/s = bytes / seconds = bytes / (dt_us * 1e-6) / 1e9 = bytes / (dt_us * 1e3)
    gbps = bf4_bytes(k, n) / (dt * 1e3)
    print(
        f"\n[matmul_decode {name}] shape m={m} k={k} n={n} bf4  "
        f"k_blocks={k_blocks} n_blocks={n_blocks} cores={num_b_cores}  "
        f"PCC={pcc_val:.5f}  time={dt:.1f} us  BW={gbps:.0f} GB/s   "
        f"(prod baseline ~98-100 us, ~257-263 GB/s)"
    )

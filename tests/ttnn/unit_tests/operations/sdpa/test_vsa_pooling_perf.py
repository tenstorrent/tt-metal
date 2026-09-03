# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for the VSA coarse-stage pooling ops on one device (15 s / 768p shard shapes):
the heads-folded pooling matmul [H*d, S_local] @ [S_local, slots] under several program configs, and
the transposes around it. Run: ./scripts/run_safe_pytest.sh <this file> -q -s"""

import time

import pytest
import torch
import ttnn

from models.common.utility_functions import skip_for_wormhole_b0

H, D, S_LOCAL, SLOTS = 14, 128, 14464, 256


def _bench(dev, fn, n_iters=10):
    out = fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(n_iters):
        out = fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(3):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) * 1e3 / (3 * n_iters)
    ttnn.release_trace(dev, tid)
    return ms, out


@skip_for_wormhole_b0("Blackhole shapes")
@pytest.mark.parametrize("device_params", [{"trace_region_size": 60_000_000, "l1_small_size": 65536}], indirect=True)
def test_pooling_matmul_sweep(device):
    x = ttnn.from_torch(torch.randn(1, H, S_LOCAL, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    a_t = ttnn.from_torch(
        torch.randn(1, 1, S_LOCAL, SLOTS) * 0.01, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    x_t = ttnn.reshape(ttnn.transpose(x, 2, 3), [1, 1, H * D, S_LOCAL])
    results = []

    ms, _ = _bench(device, lambda: ttnn.transpose(x, 2, 3))
    results.append(("transpose [1,H,S,d]->[1,H,d,S]", ms))
    ms, _ = _bench(device, lambda: ttnn.matmul(x_t, a_t))
    results.append(("matmul default", ms))

    grid = device.compute_with_storage_grid_size()
    m_tiles, n_tiles, k_tiles = H * D // 32, SLOTS // 32, S_LOCAL // 32
    for gx, gy in ((grid.x, grid.y), (8, 8), (12, 8)):
        for in0_block_w in (2, 4):
            per_core_m = -(-m_tiles // gy)
            per_core_n = -(-n_tiles // gx)
            cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=per_core_m,
                per_core_N=per_core_n,
                transpose_mcast=False,
                fused_activation=None,
            )
            try:
                ms, _ = _bench(device, lambda: ttnn.matmul(x_t, a_t, program_config=cfg))
                results.append((f"mcast grid {gx}x{gy} in0_block_w={in0_block_w} M/N per core {per_core_m}/{per_core_n}", ms))
            except Exception as e:  # noqa: BLE001
                results.append((f"mcast grid {gx}x{gy} in0_block_w={in0_block_w}", str(e)[:70]))

    print(f"\nPOOLING shapes x_t=[{H*D},{S_LOCAL}] a_t=[{S_LOCAL},{SLOTS}] bf16")
    for name, ms in results:
        print(f"POOLING {name:60s} {ms if isinstance(ms, str) else f'{ms:7.3f} ms'}")

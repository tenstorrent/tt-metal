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


@skip_for_wormhole_b0("Blackhole shapes")
@pytest.mark.parametrize("device_params", [{"trace_region_size": 60_000_000, "l1_small_size": 65536}], indirect=True)
def test_coarse_matmuls_sweep(device):
    """The two batched coarse matmuls (scores = q_c @ k_c_t_g, o_c = probs @ v_c_g) at 15 s / 768p with
    padded pooling: [1,H,256,128] @ [1,H,128,2048] and [1,H,256,2048] @ [1,H,2048,128]."""
    n_cols = SLOTS * 8
    q_c = ttnn.from_torch(torch.randn(1, H, SLOTS, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_t = ttnn.from_torch(torch.randn(1, H, D, n_cols), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    probs = ttnn.from_torch(torch.rand(1, H, SLOTS, n_cols), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_g = ttnn.from_torch(torch.randn(1, H, n_cols, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grid = device.compute_with_storage_grid_size()
    results = []
    ms, _ = _bench(device, lambda: ttnn.matmul(q_c, k_t))
    results.append(("scores default", ms))
    ms, _ = _bench(device, lambda: ttnn.matmul(probs, v_g))
    results.append(("o_c default", ms))
    # batched reuse config: cores split the batch x M x N output tiles
    for name, a, b, m_t, n_t, k_t_ in (("scores", q_c, k_t, SLOTS // 32, n_cols // 32, D // 32), ("o_c", probs, v_g, SLOTS // 32, D // 32, n_cols // 32)):
        for in0_block_w in (1, 2, 4):
            if k_t_ % in0_block_w:
                continue
            for per_core_m, per_core_n in ((1, n_t), (2, n_t), (m_t, 1), (1, 4), (2, 2)):
                if per_core_n > n_t or per_core_m > m_t:
                    continue
                cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=(grid.x, grid.y),
                    in0_block_w=in0_block_w,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=per_core_m,
                    per_core_N=per_core_n,
                )
                try:
                    ms, _ = _bench(device, lambda: ttnn.matmul(a, b, program_config=cfg))
                    results.append((f"{name} reuse in0_block_w={in0_block_w} per_core M/N={per_core_m}/{per_core_n}", ms))
                except Exception as e:  # noqa: BLE001
                    results.append((f"{name} reuse in0_block_w={in0_block_w} per_core M/N={per_core_m}/{per_core_n}", str(e)[:60]))
    print("\nCOARSE_MM shapes: scores [1,14,256,128]@[1,14,128,2048], o_c [1,14,256,2048]@[1,14,2048,128]")
    for name, ms in results:
        print(f"COARSE_MM {name:58s} {ms if isinstance(ms, str) else f'{ms:7.3f} ms'}")

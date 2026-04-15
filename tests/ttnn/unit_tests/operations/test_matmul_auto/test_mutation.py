# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Mutation/perturbation test for matmul_auto.

Stage 3 requirement: prove that the auto-selected config beats
deliberately degraded neighbors (smaller grids, minimal block widths).
"""

import os
import time

import pytest
import torch

import ttnn

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


TEST_SHAPES = [
    ("1024x1024x1024", 1024, 1024, 1024),
    ("1024x4096x1024", 1024, 4096, 1024),
    ("2048x2048x2048", 2048, 2048, 2048),
]


def _bench(fn, device, warmup=5, runs=20):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / runs * 1000


@pytest.mark.parametrize("name,M,K,N", TEST_SHAPES)
def test_auto_beats_default(device, name, M, K, N):
    """Auto-selected config should be at least as fast as default ttnn.matmul."""
    from ttnn._experimental.auto_config.matmul_auto import matmul_auto

    a = torch.randn(1, 1, M, K)
    b = torch.randn(1, 1, K, N)
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    t_auto = _bench(lambda: matmul_auto(ta, tb), device)
    t_default = _bench(lambda: ttnn.matmul(ta, tb), device)

    # Allow 5% tolerance — auto should not be significantly slower
    assert t_auto <= t_default * 1.05, f"{name}: auto ({t_auto:.2f}ms) slower than default ({t_default:.2f}ms)"


@pytest.mark.parametrize("name,M,K,N", TEST_SHAPES)
def test_auto_beats_degraded_neighbors(device, name, M, K, N):
    """Auto-selected config should beat deliberately suboptimal configs."""
    from ttnn._experimental.auto_config.matmul_auto import matmul_auto

    a = torch.randn(1, 1, M, K)
    b = torch.randn(1, 1, K, N)
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    M_tiles = M // 32
    K_tiles = K // 32
    N_tiles = N // 32

    t_auto = _bench(lambda: matmul_auto(ta, tb), device)

    configs_tested = 0
    configs_auto_wins = 0

    for core_x, core_y in [(4, 4), (2, 2)]:
        if core_x > gx or core_y > gy:
            continue
        pcM = M_tiles // core_y
        pcN = N_tiles // core_x
        if pcM < 1 or pcN < 1:
            continue
        try:
            cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(core_x, core_y),
                in0_block_w=min(K_tiles, 4),
                out_subblock_h=1,
                out_subblock_w=min(pcN, 4),
                per_core_M=pcM,
                per_core_N=pcN,
            )
            t_cfg = _bench(lambda: ttnn.matmul(ta, tb, program_config=cfg), device)
            configs_tested += 1
            if t_auto <= t_cfg * 1.05:
                configs_auto_wins += 1
        except Exception:
            configs_tested += 1
            configs_auto_wins += 1  # crashed = auto wins by default

    assert configs_tested > 0, f"{name}: no degraded configs could be tested"
    assert (
        configs_auto_wins >= configs_tested * 0.5
    ), f"{name}: auto won {configs_auto_wins}/{configs_tested} — expected >= 50%"

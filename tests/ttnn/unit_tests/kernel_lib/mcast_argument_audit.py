# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicitly collected, matched before/after operation timing measurements.

Run through scripts/run_safe_pytest.sh, optionally with --profile, and retain
stdout. These measurements are not performance assertions.

The dense multicast matmul factories build a Metal 2.0 ProgramSpec, which is not
exposed to Python. Their emitted argument snapshots (complete named CT, CT
varargs, per-node named RT and varargs, resource bindings, and placement) and
warmed artifact-construction timings come from the matching C++ cases instead:

    TT_MCAST_ARGUMENT_AUDIT=1 build_Release/test/ttnn/unit_tests_ttnn \
        --gtest_filter='McastDenseMatmul/*'

Cases fixed_1d_mcast_in0, rotating_1d_mcast_in0, fixed_2d, and rotating_2d there
use the same workloads as fixed_1d, rotating_1d, fixed_2d, and rotating_2d here.
"""

import json
import statistics
import time

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("kind", ["fixed_1d", "rotating_1d", "fixed_2d", "rotating_2d"])
def test_matmul_argument_measurement(device, kind):
    torch.manual_seed(7)
    rotating = kind.startswith("rotating")
    two_dimensional = kind.endswith("2d")
    m, k, n = (256, 1024, 256) if two_dimensional else (256, 1024, 1024)
    grid = (4, 4) if two_dimensional else (8, 1)
    a_host = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    b_host = torch.randn((1, 1, k, n), dtype=torch.bfloat16)
    a_memory = ttnn.DRAM_MEMORY_CONFIG
    if rotating:
        a_memory = ttnn.create_sharded_memory_config(
            (m, k),
            core_grid=ttnn.CoreGrid(x=grid[0], y=grid[1]),
            strategy=ttnn.ShardStrategy.BLOCK if two_dimensional else ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    a = ttnn.from_torch(a_host, layout=ttnn.TILE_LAYOUT, device=device, memory_config=a_memory)
    b = ttnn.from_torch(b_host, layout=ttnn.TILE_LAYOUT, device=device)
    common = dict(
        compute_with_storage_grid_size=grid,
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=m // 32 // (grid[1] if two_dimensional else 1),
        per_core_N=n // 32 // grid[0],
        fused_activation=None,
        fuse_batch=True,
    )
    config = (
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**common, transpose_mcast=False)
        if two_dimensional
        else ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(**common, mcast_in0=True)
    )
    config.allowed_worker_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))]
    )

    start = time.perf_counter_ns()
    result = ttnn.matmul(a, b, program_config=config, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(device)
    first_call_ns = time.perf_counter_ns() - start
    assert_with_pcc(a_host @ b_host, ttnn.to_torch(result), 0.999)
    cache_hit_ns = []
    for iteration in range(25):
        start = time.perf_counter_ns()
        result = ttnn.matmul(a, b, program_config=config, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(device)
        if iteration >= 5:
            cache_hit_ns.append(time.perf_counter_ns() - start)
    assert_with_pcc(a_host @ b_host, ttnn.to_torch(result), 0.999)

    print(
        "MCAST_ARGUMENT_AUDIT "
        + json.dumps(
            {
                "case": kind,
                "shape_mkn": [m, k, n],
                "grid": grid,
                "input_sharded": rotating,
                "dtype": "bfloat16",
                "output": "interleaved_dram",
                "fusion": False,
                "first_call_including_sync_ns": first_call_ns,
                "cache_hit_including_sync_ns": cache_hit_ns,
                "cache_hit_including_sync_median_ns": statistics.median(cache_hit_ns),
            },
            sort_keys=True,
        )
    )

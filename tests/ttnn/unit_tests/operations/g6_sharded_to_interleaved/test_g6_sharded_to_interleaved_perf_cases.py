# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf harness: the two production cases, a few calls each, for `run_safe_pytest.sh --profile`.

Measurement only (device kernel duration from the profiler CSV); correctness is gated elsewhere.
p1 runs through the dispatch path because it is outside Phase 0 SUPPORTED.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved import _check_structure, _dispatch

CASES = {
    "p2_flux": dict(
        shape=(1, 5120),
        dtype=ttnn.bfloat16,
        scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        shard=(32, 160),
        grid=(((0, 0), (7, 3)),),
        out=ttnn.DRAM_MEMORY_CONFIG,
    ),
    "p1_panoptic": dict(
        shape=(1, 128, 256, 32),
        dtype=ttnn.bfloat8_b,
        scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        shard=(320, 32),
        grid=(((0, 0), (10, 8)), ((0, 9), (3, 9))),
        out=ttnn.L1_MEMORY_CONFIG,
    ),
}
CALLS = 4


@pytest.mark.parametrize("name", list(CASES))
def test_perf_case(device, name):
    case = CASES[name]
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for (x0, y0), (x1, y1) in case["grid"]]
    )
    in_mc = ttnn.MemoryConfig(
        case["scheme"], ttnn.BufferType.L1, ttnn.ShardSpec(grid, case["shard"], ttnn.ShardOrientation.ROW_MAJOR)
    )
    tt_input = ttnn.from_torch(
        torch.randn(case["shape"]).to(torch.bfloat16),
        dtype=case["dtype"],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in_mc,
    )
    _check_structure(tt_input, case["out"])
    for _ in range(CALLS):
        ttnn.deallocate(_dispatch(tt_input, case["out"]))
    ttnn.synchronize_device(device)
    ttnn.deallocate(tt_input)

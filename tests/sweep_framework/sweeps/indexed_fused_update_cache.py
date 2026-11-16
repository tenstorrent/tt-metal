# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import random

import torch
import ttnn

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time


TIMEOUT = 60


parameters = {
    "nightly": {
        "geometry": [
            {"cache_shape": (1, 1, 32, 32), "source_rows": 1, "position_pattern": "sequential"},
            {"cache_shape": (2, 2, 64, 64), "source_rows": 9, "position_pattern": "cross_page"},
            {"cache_shape": (4, 2, 128, 128), "source_rows": 17, "position_pattern": "skip_oob"},
            {"cache_shape": (2, 4, 64, 256), "source_rows": 35, "position_pattern": "duplicates"},
            {"cache_shape": (2, 8, 64, 512), "source_rows": 40, "position_pattern": "worker_stride"},
        ],
    },
}


def _physical_positions(pattern, source_rows, total_cache_rows, rows_per_page):
    if pattern == "sequential":
        return list(range(source_rows))
    if pattern == "cross_page":
        return [(rows_per_page - 3 + row) % total_cache_rows for row in range(source_rows)]
    if pattern == "skip_oob":
        positions = [(row * 17) % total_cache_rows for row in range(source_rows)]
        positions[1] = -1
        positions[-1] = total_cache_rows
        return positions
    if pattern == "duplicates":
        return [(row // 3) % total_cache_rows for row in range(source_rows)]
    if pattern == "worker_stride":
        return list(range(total_cache_rows - source_rows, total_cache_rows))
    raise ValueError(f"Unknown position pattern: {pattern}")


def run(geometry, *, device) -> list:
    random.seed(0)
    torch.manual_seed(0)

    cache_shape = tuple(geometry["cache_shape"])
    source_rows = geometry["source_rows"]
    input_shape = (1, cache_shape[1], source_rows, cache_shape[3])
    total_cache_rows = cache_shape[0] * cache_shape[2]
    positions = _physical_positions(geometry["position_pattern"], source_rows, total_cache_rows, cache_shape[2])

    torch_cache1 = torch.randn(cache_shape, dtype=torch.bfloat16)
    torch_cache2 = torch.randn(cache_shape, dtype=torch.bfloat16)
    torch_input1 = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_input2 = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_positions = torch.tensor([positions], dtype=torch.int32)

    golden_function = ttnn.get_golden_function(ttnn.experimental.indexed_fused_update_cache)
    expected1, expected2 = golden_function(torch_cache1, torch_input1, torch_cache2, torch_input2, torch_positions)

    def to_device(tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            tensor,
            device=device,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    cache1 = to_device(torch_cache1)
    cache2 = to_device(torch_cache2)
    input1 = to_device(torch_input1)
    input2 = to_device(torch_input2)
    positions_tensor = to_device(torch_positions, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    start_time = start_measuring_time()
    output1, output2 = ttnn.experimental.indexed_fused_update_cache(cache1, input1, cache2, input2, positions_tensor)
    output1 = ttnn.to_torch(output1)
    output2 = ttnn.to_torch(output2)
    elapsed = stop_measuring_time(start_time)

    passed = torch.equal(output1, expected1) and torch.equal(output2, expected2)
    message = None if passed else "indexed fused cache output did not exactly match the golden function"
    return [(passed, message), elapsed]

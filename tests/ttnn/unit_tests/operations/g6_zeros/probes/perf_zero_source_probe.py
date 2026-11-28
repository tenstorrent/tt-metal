# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 probe: zero-source size x placement on the four production cases.

The zero source is filled serially before the first DRAM write, so its size is fixed cost on small
runs (p4) and, past the size where packet issue keeps up with the bank, buys nothing on large runs.
Dispatched past validate() through the plan. Run with
`scripts/run_safe_pytest.sh --profile --no-py-zones <this file>`; REPEATS rows per test in the ops
CSV, in collection order (cases outer, variants inner).
"""

import pytest
import ttnn

from ttnn.operations.g6_zeros import g6_zeros_program_descriptor as pdm

REPEATS = 2

CASES = [
    pytest.param((32831, 1, 32, 256), ttnn.bfloat8_b, id="p1"),
    pytest.param((16, 12, 128, 128), ttnn.float32, id="p2"),
    pytest.param((8, 12, 128, 128), ttnn.float32, id="p3"),
    pytest.param((1, 4, 256, 128), ttnn.bfloat16, id="p4"),
]

# (placement, writers_per_bank, zero_source_bytes); writers_per_bank None = the grid-filling rule.
VARIANTS = [
    pytest.param("row_wise", None, 16384, id="phase0"),
    pytest.param("bank_affine", 1, 16384, id="ba1-z16k"),
    pytest.param("bank_affine", 1, 8192, id="ba1-z8k"),
    pytest.param("bank_affine", 1, 4096, id="ba1-z4k"),
    pytest.param("bank_affine", 1, 2048, id="ba1-z2k"),
    pytest.param("bank_affine", 1, 1024, id="ba1-z1k"),
    pytest.param("bank_affine", 2, 2048, id="ba2-z2k"),
]


@pytest.mark.parametrize("placement, writers_per_bank, zero_source_bytes", VARIANTS)
@pytest.mark.parametrize("shape, dtype", CASES)
def test_zero_source(shape, dtype, placement, writers_per_bank, zero_source_bytes, device):
    num_banks = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    grid = device.compute_with_storage_grid_size()
    seeds = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_1)
    for _ in range(REPEATS):
        output = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        num_pages = output.buffer_num_pages()
        wpb = writers_per_bank or pdm.grid_fill_writers_per_bank(num_pages, num_banks, grid.x * grid.y)
        plan = pdm.plan_bank_runs(
            num_pages,
            num_banks,
            output.buffer_aligned_page_size(),
            grid,
            writers_per_bank=wpb,
            placement=placement,
            bank_seeds=seeds,
            zero_source_bytes=zero_source_bytes,
        )
        ttnn.generic_op([output], pdm.create_program_descriptor(output, plan))
        ttnn.synchronize_device(device)
        ttnn.deallocate(output)

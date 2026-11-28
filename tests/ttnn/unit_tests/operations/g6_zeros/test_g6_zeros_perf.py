# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-time harness for g6_zeros: the four production cases under the plan's knobs.

Measurement only; correctness is the acceptance suite's job. Each test builds the op's own
descriptor and dispatches it past validate(), so a dtype outside SUPPORTED can still be timed.
Run under `scripts/run_safe_pytest.sh --profile --no-py-zones`. The ops CSV holds REPEATS rows
per test, in collection order; read the last row of each test (the first can carry a cold start).
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

# None is the plan's default rule; the others are the design's grid-synchronization lamp.
WRITERS_PER_BANK = [None, 1, 2, 4, 8]


@pytest.mark.parametrize("writers_per_bank", WRITERS_PER_BANK, ids=lambda w: f"wpb={w if w else 'default'}")
@pytest.mark.parametrize("shape, dtype", CASES)
def test_g6_zeros_device_time(shape, dtype, writers_per_bank, device):
    num_banks = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    grid = device.compute_with_storage_grid_size()
    for _ in range(REPEATS):
        output = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        plan = pdm.plan_bank_runs(
            output.buffer_num_pages(),
            num_banks,
            output.buffer_aligned_page_size(),
            grid,
            writers_per_bank=writers_per_bank,
        )
        ttnn.generic_op([output], pdm.create_program_descriptor(output, plan))
        ttnn.synchronize_device(device)
        ttnn.deallocate(output)

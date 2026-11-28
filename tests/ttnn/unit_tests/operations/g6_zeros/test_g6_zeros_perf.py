# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-time harness for g6_zeros: the four production cases, shipped plan against Phase 0.

Measurement only; correctness is the acceptance suite's job. "shipped" calls the public entry
point with the module's knob defaults. "phase0" dispatches the op's own descriptor with the Phase 0
knob values (13 writers per bank, row-wise, 16 KiB zero source) for an A/B in the same session.
Run under `scripts/run_safe_pytest.sh --profile --no-py-zones`; one ops-CSV row per test, in
collection order (cases outer, plans inner).
"""

import pytest
import ttnn

from ttnn.operations.g6_zeros import g6_zeros
from ttnn.operations.g6_zeros import g6_zeros_program_descriptor as pdm

CASES = [
    pytest.param((32831, 1, 32, 256), ttnn.bfloat8_b, id="p1"),
    pytest.param((16, 12, 128, 128), ttnn.float32, id="p2"),
    pytest.param((8, 12, 128, 128), ttnn.float32, id="p3"),
    pytest.param((1, 4, 256, 128), ttnn.bfloat16, id="p4"),
]

PHASE0_ZERO_SOURCE_BYTES = 16384


def _phase0(shape, dtype, device):
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    num_banks = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    grid = device.compute_with_storage_grid_size()
    num_pages = output.buffer_num_pages()
    plan = pdm.plan_bank_runs(
        num_pages,
        num_banks,
        output.buffer_aligned_page_size(),
        grid,
        writers_per_bank=pdm.grid_fill_writers_per_bank(num_pages, num_banks, grid.x * grid.y),
        placement="row_wise",
        zero_source_bytes=PHASE0_ZERO_SOURCE_BYTES,
    )
    return ttnn.generic_op([output], pdm.create_program_descriptor(output, plan))


@pytest.mark.parametrize("plan", ["shipped", "phase0"])
@pytest.mark.parametrize("shape, dtype", CASES)
def test_g6_zeros_device_time(shape, dtype, plan, device):
    if plan == "shipped":
        output = g6_zeros(
            shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    else:
        output = _phase0(shape, dtype, device)
    ttnn.synchronize_device(device)
    ttnn.deallocate(output)

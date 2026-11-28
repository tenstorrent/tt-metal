# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Verifier probe: headroom of the design's core<->bank placement lamp on the production cases.

Times the Phase 0 plan against one writer per bank placed on that bank's NoC1-optimal worker core,
dispatched past validate() so bfloat16 / bfloat8_b can be timed before they enter SUPPORTED. Each
variant also checks the output over a poisoned range. Run with
`scripts/run_safe_pytest.sh --profile --no-py-zones <this file>`; the ops CSV holds REPEATS rows per
test in collection order.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_zeros import g6_zeros_program_descriptor as pdm

REPEATS = 2

CASES = [
    pytest.param((32831, 1, 32, 256), ttnn.bfloat8_b, id="p1"),
    pytest.param((16, 12, 128, 128), ttnn.float32, id="p2"),
    pytest.param((8, 12, 128, 128), ttnn.float32, id="p3"),
    pytest.param((1, 4, 256, 128), ttnn.bfloat16, id="p4"),
]

_TORCH = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}


def _plan(output, device, variant):
    num_banks = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    grid = device.compute_with_storage_grid_size()
    num_pages = output.buffer_num_pages()
    if variant == "default":
        knobs = dict(placement="row_wise", writers_per_bank=pdm.grid_fill_writers_per_bank(num_pages, num_banks, grid.x * grid.y))
    else:
        knobs = dict(placement="bank_affine", writers_per_bank=1)
    plan = pdm.plan_bank_runs(
        num_pages,
        num_banks,
        output.buffer_aligned_page_size(),
        grid,
        bank_seeds=ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_1),
        zero_source_bytes=16384,
        **knobs,
    )
    print(f"\n{variant} cores={[(c.x, c.y) for c in plan.cores][:16]} ranges={plan.core_ranges}")
    return plan


def _poisoned_output(shape, dtype, device):
    pattern = torch.full(tuple(shape), 3.0, dtype=_TORCH[dtype])
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(pattern, dtype=dtype, layout=ttnn.TILE_LAYOUT), out)
    return out


@pytest.mark.parametrize("variant", ["default", "bank_affine_1"])
@pytest.mark.parametrize("shape, dtype", CASES)
def test_placement(shape, dtype, variant, device):
    for _ in range(REPEATS):
        output = _poisoned_output(shape, dtype, device)
        plan = _plan(output, device, variant)
        ttnn.generic_op([output], pdm.create_program_descriptor(output, plan))
        ttnn.synchronize_device(device)
        readback = ttnn.to_torch(output).to(torch.float32)
        assert int((readback.view(torch.int32) != 0).sum()) == 0
        ttnn.deallocate(output)

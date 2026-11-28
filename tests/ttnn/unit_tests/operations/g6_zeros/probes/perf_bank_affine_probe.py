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


def _bank_affine_plan(output, device):
    num_banks = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    grid = device.compute_with_storage_grid_size()
    optimal = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_1)
    cores = tuple(optimal[b] for b in range(num_banks))
    num_pages = output.buffer_num_pages()
    blocks = tuple((pdm.BankRun(b, b, pdm.bank_slots(num_pages, num_banks, b)),) for b in range(num_banks))
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    print(f"\nBANK_AFFINE cores={[(c.x, c.y) for c in cores]} ranges={core_ranges}")
    return pdm.WriterPlan(
        num_pages=num_pages,
        num_banks=num_banks,
        aligned_page_size=output.buffer_aligned_page_size(),
        writers_per_bank=1,
        zero_source_bytes=pdm.ZERO_SOURCE_BYTES,
        zero_source_regions=pdm.ZERO_SOURCE_REGIONS,
        grid=(grid.x, grid.y),
        core_ranges=core_ranges,
        cores=cores,
        blocks=blocks,
    )


def _default_plan(output, device):
    num_banks = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    return pdm.plan_bank_runs(
        output.buffer_num_pages(),
        num_banks,
        output.buffer_aligned_page_size(),
        device.compute_with_storage_grid_size(),
    )


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
        plan = _default_plan(output, device) if variant == "default" else _bank_affine_plan(output, device)
        ttnn.generic_op([output], pdm.create_program_descriptor(output, plan))
        ttnn.synchronize_device(device)
        readback = ttnn.to_torch(output).to(torch.float32)
        assert int((readback.view(torch.int32) != 0).sum()) == 0
        ttnn.deallocate(output)

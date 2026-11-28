# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Verifier probe: fixed-cost floor of the p4 case (1, 4, 256, 128) bfloat16, 128 pages.

Times the real write against an empty launch (every writer's run has zero slots, so the kernel does a
0-byte fill and one barrier) on the Phase 0 core set (104) and on one NoC1-optimal writer per bank
(8), at two zero-source sizes. The gap between the write and the empty launch is the most a kernel
change can recover. Dispatched past validate(). Run with
`scripts/run_safe_pytest.sh --profile --no-py-zones <this file>`; REPEATS rows per test in the ops
CSV, in collection order.
"""

import pytest
import ttnn

from ttnn.operations.g6_zeros import g6_zeros_program_descriptor as pdm

REPEATS = 3
SHAPE, DTYPE = (1, 4, 256, 128), ttnn.bfloat16


def _phase0_plan(output, device, zero_source_bytes, empty):
    num_banks = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    plan = pdm.plan_bank_runs(
        output.buffer_num_pages(),
        num_banks,
        output.buffer_aligned_page_size(),
        device.compute_with_storage_grid_size(),
        zero_source_bytes=zero_source_bytes,
    )
    if empty:
        # Keep the runtime-arg count per core: generic_op's cache hashes structure only, and a cache
        # hit with a different arg count is a TT_FATAL.
        blocks = tuple(
            tuple(pdm.BankRun(r.bank, r.first_page_id, 0) for r in core_blocks) for core_blocks in plan.blocks
        )
        plan = pdm.WriterPlan(**{**plan.__dict__, "blocks": blocks})
    return plan


def _bank_affine_plan(output, device, zero_source_bytes, empty):
    num_banks = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    grid = device.compute_with_storage_grid_size()
    optimal = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_1)
    cores = tuple(optimal[b] for b in range(num_banks))
    num_pages = output.buffer_num_pages()
    blocks = tuple(
        (pdm.BankRun(b, b, 0 if empty else pdm.bank_slots(num_pages, num_banks, b)),) for b in range(num_banks)
    )
    return pdm.WriterPlan(
        num_pages=num_pages,
        num_banks=num_banks,
        aligned_page_size=output.buffer_aligned_page_size(),
        writers_per_bank=1,
        zero_source_bytes=zero_source_bytes,
        zero_source_regions=1,
        grid=(grid.x, grid.y),
        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]),
        cores=cores,
        blocks=blocks,
    )


@pytest.mark.parametrize("empty", [False, True], ids=["write", "empty"])
@pytest.mark.parametrize("zero_source_bytes", [16384, 4096], ids=["z16k", "z4k"])
@pytest.mark.parametrize("placement", ["phase0_104", "bank_affine_8"])
def test_p4_floor(placement, zero_source_bytes, empty, device):
    make = _phase0_plan if placement == "phase0_104" else _bank_affine_plan
    for _ in range(REPEATS):
        output = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(SHAPE)), DTYPE, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        plan = make(output, device, zero_source_bytes, empty)
        ttnn.generic_op([output], pdm.create_program_descriptor(output, plan))
        ttnn.synchronize_device(device)
        ttnn.deallocate(output)

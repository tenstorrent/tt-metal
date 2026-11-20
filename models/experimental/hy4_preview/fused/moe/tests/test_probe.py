# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What one chip tells Python about the things the MoE program's placement relies on: the worker grid, the logical ->
NOC coordinates of its corners (the multicast rectangle), the DRAM banks and each bank's optimal reader core. Logs
lines starting with HY4_MK5; runs on the mock Galaxy and on the board."""

from loguru import logger

import ttnn


def test_probe_chip(device):
    grid = device.compute_with_storage_grid_size()
    banks = device.dram_grid_size()
    logger.info(f"HY4_MK5 probe {type(device).__name__} grid {grid.x}x{grid.y} dram grid {banks.x}x{banks.y}")
    row = [device.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)) for x in range(grid.x)]
    col = [device.worker_core_from_logical_core(ttnn.CoreCoord(0, y)) for y in range(grid.y)]
    logger.info(f"HY4_MK5 probe noc x of logical x: {[c.x for c in row]}")
    logger.info(f"HY4_MK5 probe noc y of logical y: {[c.y for c in col]}")
    try:
        a = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    except TypeError:
        a = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0, ttnn.MeshCoordinate(0, 0))
    workers = [a[b] for b in sorted(a)] if isinstance(a, dict) else list(a)
    logger.info(f"HY4_MK5 probe NOC0 bank workers (logical): {[(c.x, c.y) for c in workers]}")
    for kind in (ttnn.BufferType.L1, ttnn.BufferType.DRAM):
        view = ttnn.get_memory_view(device, kind)
        logger.info(f"HY4_MK5 probe {kind}: banks {view.num_banks} per bank {view.total_bytes_per_bank}")

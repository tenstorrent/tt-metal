# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What one chip tells Python about its worker grid and DRAM banks; logged, nothing asserted beyond what the
sparse-MLA layout relies on (lines start with HY4_MK3)."""

from loguru import logger

import ttnn


def test_probe_grid(device):
    grid = device.compute_with_storage_grid_size()
    logger.info(f"HY4_MK3 probe grid {grid.x}x{grid.y} dram banks {device.dram_grid_size().x}")
    for y in range(grid.y):
        row = []
        for x in range(grid.x):
            v = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            row.append(f"({v.x},{v.y})")
        logger.info(f"HY4_MK3 probe row {y}: " + " ".join(row))
    for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
        workers = device.get_optimal_dram_bank_to_logical_worker_assignment(noc)
        logger.info(
            f"HY4_MK3 probe {noc} bank->worker " + " ".join(f"{b}:({w.x},{w.y})" for b, w in enumerate(workers))
        )
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    logger.info(f"HY4_MK3 probe L1 per bank {view.total_bytes_per_bank}")
    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    logger.info(f"HY4_MK3 probe DRAM banks {view.num_banks} per bank {view.total_bytes_per_bank}")

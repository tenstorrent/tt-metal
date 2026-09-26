# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Print where the optimal DRAM-bank reader cores sit (logical worker grid + physical NoC coords of cores and banks)."""

import pytest
from loguru import logger

import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_dram_core_map(device):
    grid = device.compute_with_storage_grid_size()
    banks = device.dram_grid_size().x
    out = {"grid": (grid.x, grid.y), "banks": banks}
    for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
        cores = device.get_optimal_dram_bank_to_logical_worker_assignment(noc)
        out[str(noc)] = [
            (c.x, c.y, device.worker_core_from_logical_core(c).x, device.worker_core_from_logical_core(c).y)
            for c in cores
        ]
    try:
        out["dram_phys"] = [
            (
                b,
                device.dram_core_from_logical_core(ttnn.CoreCoord(b, 0)).x,
                device.dram_core_from_logical_core(ttnn.CoreCoord(b, 0)).y,
            )
            for b in range(banks)
        ]
    except Exception as e:  # noqa: BLE001
        out["dram_phys"] = repr(e)
    out["col_x"] = [device.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x for x in range(grid.x)]
    out["row_y"] = [device.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y for y in range(grid.y)]
    logger.info(f"DRAMMAP {out}")

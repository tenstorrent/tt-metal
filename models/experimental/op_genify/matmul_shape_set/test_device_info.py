# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Print the per-chip facts the matmul roofline needs (grid, DRAM banks, L1)."""

import ttnn


def test_device_info(device):
    grid = device.compute_with_storage_grid_size()
    dram = device.dram_grid_size()
    print(
        f"DEVICE_INFO arch={device.arch()} compute_grid={grid.x}x{grid.y} ({grid.x * grid.y} cores) "
        f"dram_grid={dram.x}x{dram.y} num_devices={ttnn.GetNumAvailableDevices()}"
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Experiment scaffold: mirror TtMoe's 2-sub-device split (row 0 = dispatch,
rows 1.. = shared) and run dummy_op on the dispatch sub-device.

dummy_op's program factory already targets row 0 only, which is exactly the
dispatch_sd's CoreRangeSet -- so loading the sub-device manager confines
fast-dispatch's per-sub-device counters to dispatch_sd while the op runs.
"""

import pytest
import torch
import ttnn


def test_dummy_op_on_dispatch_subdevice(device):
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = grid.x, grid.y
    assert grid_y > 1, f"need grid_y > 1 for a dispatch/shared split, got grid_y={grid_y}"

    # Same topology as TtMoe (tt_moe.py:268-283 with overlap enabled).
    dispatch_sd_rows = 1
    dispatch_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, dispatch_sd_rows - 1))}
    )
    shared_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, dispatch_sd_rows), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}
    )
    dispatch_sd = ttnn.SubDevice([dispatch_cores])
    shared_sd = ttnn.SubDevice([shared_cores])

    sd_manager_id = device.create_sub_device_manager([dispatch_sd, shared_sd], 0)
    device.load_sub_device_manager(sd_manager_id)
    dispatch_sd_id = ttnn.SubDeviceId(0)
    try:
        # dummy_op constraint: tile count divisible by num_cores * 8.
        # num_cores for the dispatch sub-device == grid_x. tiles_per_core=8
        # satisfies the constraint.
        tiles_per_core = 8
        num_tiles = grid_x * tiles_per_core
        t = ttnn.from_torch(
            torch.zeros((32, 32 * num_tiles), dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.experimental.deepseek_prefill.dummy_op(t, num_iter=1, subdevice_id=dispatch_sd_id)
        ttnn.synchronize_device(device)
    finally:
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(sd_manager_id)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Experiment scaffold: mirror TtMoe's 2-sub-device split (row 0 = dispatch,
rows 1.. = shared) and run dummy_op on the dispatch sub-device while a host
for-loop runs a "routed-expert-like" matmul on shared_sd and bumps a global
semaphore between iterations to drive the reader kernel's per-iter wait.
"""

import pytest
import torch
import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
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
    shared_sd_id = ttnn.SubDeviceId(1)
    try:
        num_iter = 4

        # Monotonic counter of completed iterations: starts at 0, host bumps
        # to 1, 2, ..., num_iter. Reader's iter-i wait is for *sem >= i+1.
        sem = ttnn.create_global_semaphore(
            device, dispatch_cores, initial_value=0, buffer_type=ttnn.BufferType.L1_SMALL
        )

        # dummy_op input: tile count must be divisible by num_cores * 8.
        tiles_per_core = 8 * 8
        num_tiles = grid_x * tiles_per_core
        t = ttnn.from_torch(
            torch.zeros((32, 32 * num_tiles), dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Two DRAM-interleaved tensors for the pointless matmul that simulates
        # the routed-expert workload running concurrent with dummy_op.
        mm_a = ttnn.from_torch(
            torch.randn((128, 128), dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mm_b = ttnn.from_torch(
            torch.randn((128, 128), dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Launch dummy_op (non-blocking). The reader kernel will block on the
        # semaphore between its iter loop iterations.
        ttnn.experimental.deepseek_prefill.dummy_op(
            t, num_iter=num_iter, global_semaphore=sem, subdevice_id=dispatch_sd_id
        )

        # Stall only on shared_sd so host commands don't wait for dispatch_sd
        # (which is busy running dummy_op and would deadlock).
        device.set_sub_device_stall_group([shared_sd_id])

        # Each iter: run a "pointless" matmul confined to shared_sd (so it
        # doesn't fight dummy_op for dispatch_sd cores), then bump the
        # semaphore to unblock dummy_op's reader for iter i.
        for i in range(num_iter):
            ttnn.matmul(mm_a, mm_b, sub_device_id=shared_sd_id)
            ttnn.reset_global_semaphore_value(sem, i + 1)

        device.reset_sub_device_stall_group()

        ttnn.synchronize_device(device)
    finally:
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(sd_manager_id)

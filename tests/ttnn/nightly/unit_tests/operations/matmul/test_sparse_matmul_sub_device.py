# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Regression test: sparse_matmul must give identical results on program-cache hits regardless of which
sub-device index its cores belong to.

sparse_matmul zero-fills its (expanded) output with a separate fill program before the matmul. That fill must run
on the matmul's own sub-device: the command queue only orders a program against earlier work on its own sub-device,
so a fill on the default (first) sub-device races with the matmul when the matmul runs on sub-device index >= 1.
On a program-cache miss host-side program build hides the race; on cache hits it shows up as a few percent of
wrong output tiles that change from call to call.
"""

import math

import pytest
import torch

import ttnn

NUM_CALLS = 5
# The expanded output (E * M * N bf16, ~256 MB) is zero-filled by a separate program. The race needs that fill to
# still be running when the matmul is launched (~1 ms after it), so make the fill big and the matmul small (tiny K).
M, K, E = 128, 32, 128
PER_CORE_N_TILES = 8
TILE_H, TILE_W = 32, 32
CORE_Y = 4  # matmul grid is rows 0..3, anchored at (0, 0)
# Sub-devices split the grid by rows: [0, grid_y - 1) and the last row only. The unfixed fill runs on sub-device 0's
# workers; with the single-row sub-device first it uses only a few cores, which makes it slow.


def _make_inputs(device, seed):
    g = torch.Generator().manual_seed(seed)
    in0 = torch.randn((1, 1, M, K), generator=g).to(torch.bfloat16)
    in1 = torch.randn((1, E, K, N_PER_CALL[0]), generator=g).to(torch.bfloat16)
    sparsity = torch.zeros((1, 1, 1, E))
    sparsity.view(-1)[torch.randperm(E, generator=g)[: E // 2]] = 1.0
    sparsity = sparsity.to(torch.bfloat16)
    in0_t = ttnn.from_torch(
        in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    in1_t = ttnn.from_torch(
        in1, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    sp_t = ttnn.from_torch(
        sparsity,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return in0_t, in1_t, sp_t


N_PER_CALL = [0]  # set per test (depends on the grid)


def _run(device, seed, core_x):
    in0_t, in1_t, sp_t = _make_inputs(device, seed)
    per_core_n = N_PER_CALL[0] // TILE_W // (core_x * CORE_Y)
    out = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sp_t,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, CORE_Y),
            in0_block_w=K // TILE_W,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=M // TILE_H,
            per_core_N=per_core_n,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        ),
    )
    return ttnn.to_torch(out)


@pytest.mark.parametrize("matmul_sub_device_index", [0, 1])
def test_sparse_matmul_program_cache_hits_on_sub_device(device, matmul_sub_device_index):
    grid = device.compute_with_storage_grid_size()
    split_row = grid.y - 1
    if split_row < CORE_Y:
        pytest.skip(f"needs at least {CORE_Y + 1} worker rows, got {grid.y}")
    core_x = min(grid.x, 9)
    N_PER_CALL[0] = PER_CORE_N_TILES * core_x * CORE_Y * TILE_W

    # Reference: no sub-device manager.
    reference = [_run(device, seed, core_x) for seed in range(NUM_CALLS)]

    def cores(y0, y1):
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, y0), ttnn.CoreCoord(grid.x - 1, y1))})

    # The matmul always runs on rows 0..3 (inside the large sub-device); only the order of the sub-devices changes, so it lands on
    # sub-device `matmul_sub_device_index`.
    lower = ttnn.SubDevice([cores(0, split_row - 1)])
    upper = ttnn.SubDevice([cores(split_row, grid.y - 1)])
    sub_devices = [lower, upper] if matmul_sub_device_index == 0 else [upper, lower]
    manager = device.create_sub_device_manager(sub_devices, 0)
    device.load_sub_device_manager(manager)
    try:
        # Call 0 is a program-cache miss, later calls are cache hits.
        for seed in range(NUM_CALLS):
            result = _run(device, seed, core_x)
            wrong = int((result != reference[seed]).sum().item())
            assert wrong == 0, (
                f"sparse_matmul on sub-device index {matmul_sub_device_index}, call {seed}: "
                f"{wrong} elements differ from the no-sub-device-manager result"
            )
    finally:
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)

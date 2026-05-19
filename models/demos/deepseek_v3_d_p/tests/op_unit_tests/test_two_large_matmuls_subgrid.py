# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests running large L1-heavy matmul(s) on constrained 8x4 compute
sub-grids using MatmulMultiCoreReuseMultiCastProgramConfig (block-style 2D
core distribution).

Both operands are DRAM-interleaved bfloat16 in TILE layout. The program
config is sized so per-core CBs consume ~1024 KB of L1.

Matmul shape: [1024, 4096] @ [4096, 4096] -> [1024, 4096]

Per-core CB footprint (bfloat16, tile = 2 KB):
    in0 (double buffered) = 2 * in0_block_w * per_core_M * 2 KB = 256 KB
    in1 (double buffered) = 2 * in0_block_w * per_core_N * 2 KB = 512 KB
    output                = per_core_M * per_core_N * 2 KB      = 256 KB
    total per core                                              = ~1024 KB

Tests:
    - test_large_matmul              : one matmul on a single 8x4 sub-grid
    - test_large_matmul_on_subdevices: two 8x4 sub-devices stacked (y=0..3 and
                                       y=4..7), one matmul dispatched on each
                                       back-to-back so they can overlap
"""

import torch
from loguru import logger

import ttnn

# Each sub-device is GRID_X x GRID_Y cores.
GRID_X = 8
GRID_Y = 4

M, K, N = 1024, 4096, 4096

# M_tiles=32, K_tiles=128, N_tiles=128
#   per_core_M = M_tiles / GRID_Y = 32 / 4 = 8
#   per_core_N = N_tiles / GRID_X = 128 / 8 = 16
#   in0_block_w=8 divides K_tiles (larger block_w => bigger in0/in1 CBs)
#   out_subblock h*w = 1*8 = 8 tiles (DEST limit for bfloat16 w/ packer_l1_acc)
prog_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(GRID_X, GRID_Y),
    in0_block_w=8,
    out_subblock_h=1,
    out_subblock_w=8,
    per_core_M=8,
    per_core_N=16,
    transpose_mcast=False,
    fused_activation=None,
)


def _make_inputs(device):
    """Build a fresh (in0, in1) DRAM-interleaved bfloat16 pair for one matmul."""
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N), dtype=torch.bfloat16) * 0.02
    kwargs = dict(
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.from_torch(torch_a, **kwargs), ttnn.from_torch(torch_b, **kwargs)


def _compute_kernel_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def test_large_matmul(device):
    """Run one L1-heavy matmul on the 8x4 compute sub-grid."""
    torch.manual_seed(0)

    tt_a, tt_b = _make_inputs(device)
    ckc = _compute_kernel_config(device)

    tt_out = ttnn.matmul(
        tt_a,
        tt_b,
        program_config=prog_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=ckc,
    )
    ttnn.synchronize_device(device)
    logger.info(f"matmul done, output shape: {tt_out.shape}")


def test_large_matmul_on_subdevices(device):
    """Create two 8x4 sub-devices stacked vertically (y=0..3 and y=4..7) and
    dispatch one L1-heavy matmul on each, back-to-back. The two ops live on
    disjoint cores, so the dispatcher can run them concurrently."""
    torch.manual_seed(0)

    grid = device.compute_with_storage_grid_size()
    assert grid.x >= GRID_X and grid.y >= 2 * GRID_Y, (
        f"Need at least {GRID_X}x{2 * GRID_Y} compute grid for two stacked "
        f"{GRID_X}x{GRID_Y} sub-devices; got {grid.x}x{grid.y}"
    )

    sd1_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))})
    sd2_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, GRID_Y), ttnn.CoreCoord(GRID_X - 1, 2 * GRID_Y - 1))}
    )
    sd_manager_id = device.create_sub_device_manager([ttnn.SubDevice([sd1_cores]), ttnn.SubDevice([sd2_cores])], 0)
    sd1_id = ttnn.SubDeviceId(0)
    sd2_id = ttnn.SubDeviceId(1)

    device.load_sub_device_manager(sd_manager_id)
    try:
        tt_a1, tt_b1 = _make_inputs(device)
        tt_a2, tt_b2 = _make_inputs(device)
        ckc = _compute_kernel_config(device)

        # Dispatch both matmuls without syncing in between so they can overlap.
        tt_out1 = ttnn.matmul(
            tt_a1,
            tt_b1,
            program_config=prog_config,
            sub_device_id=sd1_id,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ckc,
        )
        tt_out2 = ttnn.matmul(
            tt_a2,
            tt_b2,
            program_config=prog_config,
            sub_device_id=sd2_id,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ckc,
        )
        ttnn.synchronize_device(device)
        logger.info(f"both matmuls done, shapes: sd1={tt_out1.shape}, sd2={tt_out2.shape}")
    finally:
        device.clear_loaded_sub_device_manager()

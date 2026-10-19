# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression for the mainline DRAM-sharded matmul factory on Quasar (#54630).

The llama graph captures that select MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig need 12 DRAM
banks, so they skip on the Quasar simulator (2 banks). This test sizes itself from the device's DRAM bank
count instead: one in0 L1 width shard and one in1 DRAM width shard per bank, several K blocks per shard so
the multicast / credit path is exercised more than once. It runs on any arch (WH/BH too).
"""

import pytest
import torch
import ttnn
from loguru import logger

M = 32
K_TILES_PER_BANK = 4  # in0 shard width in tiles -> 4 K blocks per shard with in0_block_w=1
N_TILES_PER_BANK = 2  # per_core_N


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize("in0_block_w", [1, K_TILES_PER_BANK], ids=["kblocks4", "kblocks1"])
def test_matmul_dram_sharded_bank_sized(mesh_device, in0_block_w):
    nb = int(mesh_device.dram_grid_size().x)
    grid = mesh_device.compute_with_storage_grid_size()
    if nb > int(grid.x) * int(grid.y):
        pytest.skip(f"needs one worker core per DRAM bank; {nb} banks, grid {grid.x}x{grid.y}")
    K, N = K_TILES_PER_BANK * 32 * nb, N_TILES_PER_BANK * 32 * nb
    # The first nb compute cores in row-major order: full rows plus a partial last row.
    gx = int(grid.x)
    ranges = set()
    if nb // gx > 0:
        ranges.add(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, nb // gx - 1)))
    if nb % gx > 0:
        ranges.add(ttnn.CoreRange(ttnn.CoreCoord(0, nb // gx), ttnn.CoreCoord(nb % gx - 1, nb // gx)))
    cores = ttnn.CoreRangeSet(ranges)
    assert cores.num_cores() == nb
    # DRAM shard grids are 1D: bank_id == logical x, every shard core on row 0.
    dram_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))})
    logger.info(f"arch={mesh_device.arch()} banks={nb} M={M} K={K} N={N} in0_block_w={in0_block_w}")

    in0 = torch.randn(1, 1, M, K).bfloat16().float()
    in1 = torch.randn(1, 1, K, N).bfloat16().float()
    in0_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [M, K // nb], ttnn.ShardOrientation.ROW_MAJOR),
    )
    in1_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_cores, [K, N // nb], ttnn.ShardOrientation.ROW_MAJOR),
    )
    mapper = ttnn.replicate_tensor_to_mesh_mapper(mesh_device)
    # Host tilize, then a plain write: the on-device from_torch(layout=TILE) path is not under test here.
    in0_t = ttnn.to_device(
        ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper), mesh_device, in0_cfg
    )
    in1_t = ttnn.to_device(
        ttnn.from_torch(in1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper), mesh_device, in1_cfg
    )
    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w, per_core_M=M // 32, per_core_N=N_TILES_PER_BANK, fused_activation=None
    )
    ck = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    out_t = ttnn.matmul(
        in0_t,
        in1_t,
        program_config=pc,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
        dtype=ttnn.bfloat16,
        compute_kernel_config=ck,
    )
    out = ttnn.to_torch(out_t).float()
    ref = in0 @ in1
    pcc = _pcc(out, ref)
    zeros = int((out == 0).sum().item())
    logger.info(f"pcc={pcc:.6f} max_abs_err={(out - ref).abs().max().item():.4f} zeros={zeros}/{out.numel()}")
    assert pcc > 0.999, f"PCC {pcc} below 0.999"
    assert zeros < out.numel() // 8, f"{zeros} zero outputs: a pack-destination / write-back miss"

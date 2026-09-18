# Standalone matmul geometry probe for the SDXL FF-up shape (1024x1280 @ 1280x5120, bf16 act / bf8b weights,
# HiFi2, fp32 acc off, packer_l1_acc on = the model's DEFAULT_MM_COMPUTE_CONFIG). Compares the shipped 10x8
# config with a padded-M split over the 11 grid columns (transpose_mcast). Run under the tracy probe harness.
import os

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

M, K, N = int(os.environ.get("MM_M", "1024")), int(os.environ.get("MM_K", "1280")), int(os.environ.get("MM_N", "5120"))
KPC = K // 10 // 32  # K tiles per core on a 10-way K split
IBW = int(os.environ.get("MM_IN0_BLOCK_W", str(min(KPC, 4))))
PCN = N // 10 // 32  # per_core_N on a 10-way N split
SBW = min(PCN, 8)
COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("MM_FIDELITY", "HiFi2")),  # diagnostics only
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=os.environ.get("MM_L1_ACC", "1") == "1",
)


def block_shard(shard, grid_xy, orientation):
    gx, gy = grid_xy
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard, orientation)
    )


def width_shard(shard, grid_xy):
    gx, gy = grid_xy
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )


IN1_MC = {  # name -> in1 memory config (None = DRAM interleaved)
    "in1_l1_interleaved": ttnn.L1_MEMORY_CONFIG,
    "in1_l1_wsharded": width_shard([K, N // 10], (10, 1)),
}

CASES = {
    # name: (in0 memory config or None for L1 interleaved, program config, out memory config)
    "ship_10x8_sharded": (
        block_shard([128, 128], (10, 8), ttnn.ShardOrientation.ROW_MAJOR),
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=4,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
        ),
        ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    ),
    "ship_10x8_in0_il": (
        None,
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=4,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
        ),
        ttnn.L1_MEMORY_CONFIG,
    ),
    "dram8x8_sharded": (  # per_core_N = 20 tiles = one DRAM bank's shard width (5120/8 = 640)
        block_shard([128, 160], (8, 8), ttnn.ShardOrientation.ROW_MAJOR),
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=5,
            out_subblock_h=1,
            out_subblock_w=5,
            per_core_M=4,
            per_core_N=20,
            transpose_mcast=False,
            fused_activation=None,
        ),
        ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    ),
    "ship_10x8_k2": (  # K-block overhead sensitivity: in0_block_w 4 -> 2 (20 K blocks instead of 10)
        block_shard([128, 128], (10, 8), ttnn.ShardOrientation.ROW_MAJOR),
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=4,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
        ),
        ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    ),
    "ship_10x8_sub2x4": (  # subblock shape sensitivity (w != per_core_N and h != 1 needs interleaved out)
        block_shard([128, 128], (10, 8), ttnn.ShardOrientation.ROW_MAJOR),
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=4,
            out_subblock_h=2,
            out_subblock_w=4,
            per_core_M=4,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
        ),
        ttnn.L1_MEMORY_CONFIG,
    ),
    "n11x8_il_pad": (  # N padded over the 11 columns (165 >= 160 tiles), non-transposed, in0 interleaved
        None,
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=5,
            per_core_M=4,
            per_core_N=15,
            transpose_mcast=False,
            fused_activation=None,
        ),
        ttnn.L1_MEMORY_CONFIG,
    ),
    "t8x10_sharded": (  # transposed mcast with the SAME per-core work as the shipped config (4x16), 80 cores
        block_shard([128, 128], (8, 10), ttnn.ShardOrientation.COL_MAJOR),
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 10),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=4,
            per_core_N=16,
            transpose_mcast=True,
            fused_activation=None,
        ),
        ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    ),
    "ship_10x8_N": (  # shipped layout, per_core_N from MM_N, K from MM_K
        block_shard([128, K // 10], (10, 8), ttnn.ShardOrientation.ROW_MAJOR),
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=IBW,
            out_subblock_h=1,
            out_subblock_w=SBW,
            per_core_M=4,
            per_core_N=PCN,
            transpose_mcast=False,
            fused_activation=None,
        ),
        ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    ),
    "t11x10_interleaved": (
        None,
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=SBW,
            per_core_M=3,
            per_core_N=PCN,
            transpose_mcast=True,
            fused_activation=None,
        ),
        ttnn.L1_MEMORY_CONFIG,
    ),
    "t11x10_sharded": (
        block_shard([96, K // 10], (11, 10), ttnn.ShardOrientation.COL_MAJOR),
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            in0_block_w=IBW,
            out_subblock_h=1,
            out_subblock_w=SBW,
            per_core_M=3,
            per_core_N=PCN,
            transpose_mcast=True,
            fused_activation=None,
        ),
        ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    ),
}


@pytest.mark.parametrize("name", os.environ.get("MM_CASES", ",".join(CASES)).split(","))
def test_mm(device, name):
    torch.manual_seed(0)
    in0_mc, pc, out_mc = CASES[name]
    a = torch.randn(1, 1, M, K) * 0.5
    b = torch.randn(1, 1, K, N) * 0.05
    tt_a = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    if in0_mc is not None:
        tt_a = ttnn.to_memory_config(tt_a, in0_mc)
    tt_b = ttnn.from_torch(
        b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    in1_variant = os.environ.get("MM_IN1")
    if in1_variant == "in1_dram_wsharded":
        # weights width-sharded across the DRAM banks (the 2D factory's IN1_DRAM_WIDTH_SHARDED path):
        # the in1 sender reads its N block as contiguous rows instead of 1 KB interleaved tiles.
        dg = device.dram_grid_size()
        nbanks = dg.x * dg.y
        shard_w = -(-(N // 32) // nbanks) * 32
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))})
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid, [K, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
        )
        tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        print(f"\nMMPROBE in1 dram width-sharded: banks={nbanks} shard=[{K},{shard_w}]")
    elif in1_variant:
        tt_b = ttnn.to_memory_config(tt_b, IN1_MC[in1_variant])
    for _ in range(3):
        out = ttnn.matmul(
            tt_a, tt_b, program_config=pc, memory_config=out_mc, compute_kernel_config=COMPUTE, dtype=ttnn.bfloat16
        )
    ref = a.float() @ ttnn.to_torch(tt_b).float()
    got = ttnn.to_torch(out).float()
    print(
        f"\nMMPROBE {name} out_shape={tuple(out.shape)} padded={tuple(out.padded_shape)} mem={out.memory_config().memory_layout}"
    )
    assert_with_pcc(ref, got[..., :M, :N], 0.99)

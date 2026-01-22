# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

# Ring computation cores (16 cores per ring: 2 rows x 8 columns)
GRID = [
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),
    (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1),
]


def create_ring_config(M, K, N, num_cores, num_global_cb_receivers, untilize_out=False):
    """Create ring matmul config similar to matmul_1d_ring_config in model_config.py"""
    in0_block_w = K // num_cores // ttnn.TILE_SIZE
    out_block_h = M // ttnn.TILE_SIZE
    out_block_w = N // num_cores // ttnn.TILE_SIZE

    out_subblock_h = 1
    out_subblock_w = 8
    while out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1

    # Calculate grid size from num_cores
    if num_cores % 8 == 0:
        grid = ttnn.CoreGrid(y=num_cores // 8, x=8)
    elif num_cores == 12:
        grid = ttnn.CoreGrid(y=2, x=6)
    elif num_cores == 20:
        grid = ttnn.CoreGrid(y=4, x=5)
    else:
        grid = ttnn.CoreGrid(y=1, x=num_cores)

    hop_grid = []
    hop_core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in hop_grid
        }
    )
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=hop_core_range_set,
        num_global_cb_receivers=num_global_cb_receivers,
        untilize_out=untilize_out,
    )


def run_matmul_1d_dram_sharded(device, num_iters=1):
    """
    Matmul 1D test with configs from test_prefetcher_BH.py
    Shapes: [1, 1, 128, 4096] x [1, 1, 4096, 512]
    Each ring (16 cores) works on one slice of the sequence length (M=128 = 4 tiles)
    """
    # Matmul dimensions (with ring_size=16, matches actual use case)
    # M = 128 = 4 tiles (sequence length slice per ring)
    # K // ring_size = 4096 // 16 = 256 (tile-aligned)
    # N // ring_size = 512 // 16 = 32 (tile-aligned)
    M = 32
    K = 4096
    N = 512
    
    # Ring configuration
    ring_size = len(GRID)  # Number of cores from GRID (16 cores per ring)
    
    # Core range set for in0 and output (using GRID)
    core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in GRID])

    # in0 sharded in L1: shape={128, 4096 // ring_size}
    in0_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range_set, [M, K // ring_size], ttnn.ShardOrientation.ROW_MAJOR),
    )

    # in1 interleaved in DRAM (not sharded to avoid Blackhole restriction with ring gather matmul)
    in1_mem_config = ttnn.DRAM_MEMORY_CONFIG

    # Output sharded in L1 (same grid as in0)
    output_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range_set, [M, N // ring_size], ttnn.ShardOrientation.ROW_MAJOR),
    )

    # Program config for ring matmul using create_ring_config from test_prefetcher_BH.py
    # Note: num_global_cb_receivers must be 1 when global_cb is not provided
    program_config = create_ring_config(M, K, N, ring_size, num_global_cb_receivers=1, untilize_out=False)

    # Compute kernel config from test_prefetcher_BH.py
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    # Create test data with specified shapes
    torch.manual_seed(0)
    in0_original = torch.randn([1, 1, M, K], dtype=torch.bfloat16)
    in1_original = torch.randn([1, 1, K, N], dtype=torch.bfloat16)

    # Multiple iterations to test with program cache
    for _ in range(3):
        in0 = in0_original[0, 0].unsqueeze(0).unsqueeze(0)
        in1 = in1_original[0, 0].unsqueeze(0).unsqueeze(0)

        # --- Ring matmul ---
        in0_t = ttnn.from_torch(
            in0,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=in0_sharded_mem_config,
        )
        in1_t = ttnn.from_torch(
            in1,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=in1_mem_config,
        )

        for _ in range(num_iters):
            ring_output_t = ttnn.matmul(
                in0_t,
                in1_t,
                program_config=program_config,
                memory_config=output_sharded_mem_config,
                compute_kernel_config=compute_kernel_config,
                dtype=ttnn.bfloat16,
            )
        ring_out = ttnn.to_torch(ring_output_t)

        # --- Create input for plain matmul (like test_matmul.py). ---
        # --- Only creating input is needed to trigger failure. ---
        in0_plain = ttnn.from_torch(
            in0,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        in1_plain = ttnn.from_torch(
            in1,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # --- Torch reference ---
        torch_out = in0 @ in1.to(torch.bfloat16)

        # --- PCC comparisons ---
        assert_with_pcc(ring_out, torch_out, 0.9)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_multi_core_matmul_1d_wh_minimal(device, function_level_defaults):
    run_matmul_1d_dram_sharded(device)

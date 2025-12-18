import pytest
import torch
import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

# Hardcoded grid from PREFETCHER_NOC1_OUTPUT_GRID
GRID = [
    (1, 3),
    (2, 3),
    (1, 0),
    (2, 0),
    (1, 4),
    (2, 4),
    (1, 5),
    (2, 5),
    (5, 0),
    (6, 0),
    (5, 3),
    (6, 3),
    (5, 1),
    (6, 1),
    (5, 7),
    (6, 7),
    (5, 6),
    (6, 6),
    (5, 2),
    (6, 2),
    (5, 4),
    (6, 4),
    (5, 5),
    (6, 5),
]


def run_matmul_1d_dram_sharded(device, num_iters=1):
    """
    Minimal matmul 1D test with all configs hardcoded from debugger output.
    """
    # Core range set for in0 and output
    core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in GRID])

    # in0 sharded in L1: shape={32, 64}
    in0_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range_set, [32, 64], ttnn.ShardOrientation.ROW_MAJOR),
    )

    # in1 sharded in DRAM: grid={(x=0,y=0)-(x=11,y=0)}, shape={1280, 320}
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))})
    in1_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(in1_shard_grid, [1280, 320], ttnn.ShardOrientation.ROW_MAJOR),
    )

    # Output sharded in L1 (same grid as in0)
    output_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range_set, [32, 160], ttnn.ShardOrientation.ROW_MAJOR),
    )

    # Program config for ring matmul
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(6, 4),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=5,
        per_core_M=1,
        per_core_N=5,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=ttnn.CoreRangeSet([]),
        untilize_out=False,
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    # Create test data
    torch.manual_seed(0)
    in0_original = torch.randn([1, 1, 32, 1280], dtype=torch.bfloat16)
    in1_original = torch.randn([1, 1, 1280, 3200], dtype=torch.bfloat16)

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
            memory_config=in1_sharded_mem_config,
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


@pytest.mark.skipif(is_blackhole(), reason="Test suite for WH only")
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_multi_core_matmul_1d_wh_minimal(device, function_level_defaults):
    run_matmul_1d_dram_sharded(device)

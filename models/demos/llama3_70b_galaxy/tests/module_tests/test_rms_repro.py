import ttnn
import torch


def test_rms_repro(device):
    input_tensor = torch.randn([1, 1, 64, 128])
    input_weight = torch.randn([1, 1, 1, 128])

    input_tensor = ttnn.as_tensor(
        input_tensor,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 2), ttnn.CoreCoord(2, 5))]),
                [64, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        layout=ttnn.TILE_LAYOUT,
    )

    input_weight = ttnn.as_tensor(
        input_weight,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[1, 4],
        subblock_w=1,
        block_h=2,
        block_w=1,
        inplace=False,
    )

    normed = ttnn.rms_norm(
        input_tensor,
        weight=input_weight,
        epsilon=1e-5,
        program_config=program_config,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )

    breakpoint()

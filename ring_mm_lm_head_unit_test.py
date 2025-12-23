import ttnn
import torch
import math
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.prefetcher import Prefetcher


def num_to_coregrid(x):
    if x % 8 == 0:
        return ttnn.CoreGrid(y=x // 8, x=8)
    if x == 12:
        return ttnn.CoreGrid(y=2, x=6)
    if x == 20:
        return ttnn.CoreGrid(y=4, x=5)


def create_dram_sharded_mem_config(k, n):
    """Create DRAM-sharded memory config for width-sharded tensors"""
    dram_cores = 8  # WH has 12 dram cores, P150 has 8, P100 has 7
    padded_size = math.ceil(n / (ttnn.TILE_SIZE * dram_cores)) * (ttnn.TILE_SIZE * dram_cores)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))]),
        (k, padded_size // dram_cores),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def matmul_1d_ring_config(
    B,
    M,
    K,
    N,
    num_cores,
    num_global_cb_receivers,
    prefetch=True,
    untilize_out=False,
):
    M *= B  # Fuse batch always enabled

    in0_block_h = M // ttnn.TILE_SIZE  # 1
    in0_block_w = K // num_cores // ttnn.TILE_SIZE  # 2
    out_block_h = M // ttnn.TILE_SIZE  # 1
    out_block_w = N // num_cores // ttnn.TILE_SIZE  # (16384/32/32) = 16

    num_blocks_y = (M // ttnn.TILE_SIZE - 1) // out_block_h + 1  # 1
    num_blocks_x = (N // ttnn.TILE_SIZE - 1) // out_block_w + 1  # (4096/8/32 - 1)/2 + 1 = 8
    num_blocks_total = num_blocks_y * num_blocks_x  # 1 * 8 = 8

    if num_blocks_total != num_cores:
        assert False, f"num_blocks_total {num_blocks_total} != num_cores {num_cores}"

    out_subblock_h = 1
    out_subblock_w = 8
    while out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1

    hop_grid = []  # FIXME: Make not hard coded
    hop_core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in hop_grid
        }
    )
    grid = num_to_coregrid(num_cores)

    # tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_mcast_1d_program_factory.cpp
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,  # 1
        per_core_N=out_block_w,  # 14
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=hop_core_range_set,
        num_global_cb_receivers=num_global_cb_receivers if prefetch else 1,
        untilize_out=untilize_out,
    )

    return program_config


def test_ring_mm_lm_head():
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

    prefetcher = Prefetcher(mesh_device, num_tensors=0, num_layers=1)
    prefetcher.init("decode")

    # x is of shape 32, 4096
    x = torch.load("lm_head_x0.pt")
    # w is of shape 4096, 16032
    weights = torch.load("lm_head_weight0.pt")

    # x memory config
    x_memory_config = ttnn.create_sharded_memory_config(
        shape=(32, 4096 // prefetcher.ring_size),
        core_grid=prefetcher.to_core_range_set(prefetcher.receiver_cores(sender_active=True, receiver_active=True)),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # weights memory config
    weights_memory_config = create_dram_sharded_mem_config(k=4096, n=16384)

    # output memory config
    output_memory_config = ttnn.create_sharded_memory_config(
        shape=(32, 16384 // prefetcher.ring_size),
        core_grid=prefetcher.to_core_range_set(prefetcher.receiver_cores(sender_active=True, receiver_active=True)),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # ring mm program config
    breakpoint()
    ring_mm_program_config = matmul_1d_ring_config(
        B=1,
        M=32,
        K=4096,
        N=16384,
        num_cores=32,
        num_global_cb_receivers=1,
        prefetch=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )
    breakpoint()
    # torch implementation
    print("Running torch implementation....")
    torch_output = torch.matmul(x, weights)
    print(f"torch_output shape: {torch_output.shape}")
    breakpoint()
    x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, device=mesh_device, memory_config=x_memory_config)
    weights = ttnn.from_torch(weights, dtype=ttnn.bfloat8_b, device=mesh_device, memory_config=weights_memory_config)
    print("Running ttnn implementation....")
    breakpoint()
    output = ttnn.linear(
        x,
        weights,
        program_config=ring_mm_program_config,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat8_b,
        memory_config=output_memory_config,
        sub_device_id=prefetcher.worker_sub_device_id,
    )
    print(f"ttnn_output shape: {output.shape}")
    ttnn_output = ttnn.to_torch(ttnn.get_device_tensors(output)[0])  # 32,16032

    passing, pcc_message = comp_pcc(torch_output, ttnn_output, 0.99)
    print(f"PCC: {pcc_message}")
    breakpoint()


if __name__ == "__main__":
    test_ring_mm_lm_head()

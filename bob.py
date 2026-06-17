import ttnn
import torch

device = ttnn.open_device(device_id=0)

A = torch.rand([110, 256, 384], dtype=torch.bfloat16)
B = torch.rand([1, 384, 512], dtype=torch.bfloat16)

##############################################################################
core_ranges = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
tensor_spec = ttnn.TensorSpec(
    shape=(256, 256, 384),  # Batch=2, Height=256, Width=384
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    buffer_type=ttnn.BufferType.L1,
).block_sharded(core_ranges)

a_tile = ttnn.from_torch(
    A,
    device=device,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.L1_MEMORY_CONFIG,
    dtype=ttnn.bfloat8_b,  # override dtype to test compute kernel config's math_fidelity
    # spec=tensor_spec
)
b_tile = ttnn.from_torch(
    B,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
    dtype=ttnn.bfloat8_b,  # override dtype to test compute kernel config's math_fidelity
)

# program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
#     compute_with_storage_grid_size=(10, 11),
#     in0_block_w=12, # k param
#     out_subblock_h=2,
#     out_subblock_w=4,
#     per_core_M=8,
#     per_core_N=16,
#     # transpose_mcast=False,
#     # fused_activation=None,
#     fuse_batch=True,
# )
compute_kernel_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,  # LoFi, HiFi2, HiFi3, HiFi4
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)

o_tile = ttnn.matmul(
    a_tile,
    b_tile,
    # program_config=program_config,
    compute_kernel_config=compute_kernel_config,
    dtype=ttnn.bfloat8_b,  # override dtype to test compute kernel config's math_fidelity
)

# a_tile = ttnn.to_memory_config(a_tile, ttnn.DRAM_MEMORY_CONFIG)
# b_tile = ttnn.to_memory_config(b_tile, ttnn.DRAM_MEMORY_CONFIG)
# o_tile = a_tile @ b_tile


ttnn.close_device(device)

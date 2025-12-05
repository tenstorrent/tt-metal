# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


@pytest.mark.parametrize("num_iterations", [10])
@pytest.mark.parametrize("packer_l1_acc", [False, True])
@pytest.mark.parametrize(
    "m, k, n",
    [
        (32, 4096, 1792),
    ],
)
@pytest.mark.parametrize(
    "activation_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "weight_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "math_fidelity",
    [
        ttnn.MathFidelity.HiFi2,
    ],
)
def test_matmul_width_sharded_dram(
    device,
    m,
    k,
    n,
    activation_dtype,
    weight_dtype,
    math_fidelity,
    num_iterations,
    packer_l1_acc,
):
    """Test matmul with WIDTH_SHARDED layout matching LI_FF1 configuration"""

    # Generate input tensors
    torch.manual_seed(0)
    x_torch = torch.randn(1, 1, m, k)

    w_torch = torch.randn(k, n)

    # Golden output
    golden = torch.matmul(x_torch, w_torch)

    # Input grid: 8 cores (0,0)-(7,0) for x
    x_grid_size = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})

    # Weight grid: 12 cores (0,0)-(11,0) for w1
    w_grid_size = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))})

    # Program config matching the setup
    # in0_block_w in tiles: per_core_K = 8*32=256 elements, total K=4096
    # so we need k/(256*num_cores) = 4096/(256*8) = 2 blocks, but in0_block_w is per-core
    # Actually in0_block_w is the inner dimension block size per core = K_tiles_per_core
    # For DRAM sharded, in0_block_w should be divided by 4 (bfloat8 factor)
    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=8,  # K/(num_cores*32*4) = 4096/(8*32*4) = 4, but docs show 2
        per_core_M=1,
        per_core_N=7,
        fused_activation=None,
    )

    # Memory configs
    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )

    # x memory config: WIDTH_SHARDED, L1, shard [32, 512]
    x_shard_spec = ttnn.ShardSpec(x_grid_size, [32, 512], ttnn.ShardOrientation.ROW_MAJOR)
    x_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, x_shard_spec)

    # w1 memory config: WIDTH_SHARDED, DRAM, shard [4096, 160]
    w_shard_spec = ttnn.ShardSpec(w_grid_size, [4096, 160], ttnn.ShardOrientation.ROW_MAJOR)
    w_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, w_shard_spec)

    # Output sharded config
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    # Compute kernel config
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=packer_l1_acc,
    )

    # Convert to ttnn tensors with sharded memory configs
    x_ttnn = ttnn.from_torch(
        x_torch,
        dtype=activation_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=x_mem_config,
    )

    w_ttnn = ttnn.from_torch(
        w_torch,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=w_mem_config,
    )

    logger.info(f"Running matmul: [{m}, {k}] x [{k}, {n}]")
    logger.info(f"Activation dtype: {activation_dtype}, Weight dtype: {weight_dtype}")
    logger.info(f"Math fidelity: {math_fidelity}")

    for i in range(num_iterations):
        # Run matmul
        output_ttnn = ttnn.linear(
            x_ttnn,
            w_ttnn,
            dtype=activation_dtype,
            compute_kernel_config=compute_kernel_config,
            program_config=pc,
            memory_config=sharded_mem_config,
        )

    # Convert sharded output back to interleaved
    output_ttnn = ttnn.sharded_to_interleaved(output_ttnn, interleaved_mem_config)

    # Convert back to torch
    output_torch = ttnn.to_torch(output_ttnn)

    # Compare results
    if activation_dtype == ttnn.bfloat8_b or weight_dtype == ttnn.bfloat8_b:
        expected_pcc = 0.99
    else:
        expected_pcc = 0.9999

    eq, pcc_msg = comp_pcc(golden, output_torch, expected_pcc)
    logger.info(f"PCC check: {pcc_msg}")
    assert eq, f"PCC check failed: {pcc_msg}"

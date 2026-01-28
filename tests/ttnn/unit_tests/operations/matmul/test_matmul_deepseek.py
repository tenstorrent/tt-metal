# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

from loguru import logger
import pytest
import torch
import math
import ttnn

from tests.ttnn.unit_tests.operations.matmul.test_matmul import pad_to_dram_banks
from models.common.utility_functions import comp_pcc, skip_for_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_blackhole("Deepseek tests target Wormhole")
@pytest.mark.parametrize(
    "test_case",
    [
        # qkv_a
        {
            "m": 32,
            "k": 896,
            "n": 2112,
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (7, 1),
            "out_core_grid": (8, 1),
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat8_b,
            "out_dtype": ttnn.bfloat16,
            "expected_pcc": 0.999,
            "tile_h": 32,
            "tile_w": 32,
        },
        # wq_b
        {
            "m": 32,
            "k": 1536,
            "n": 3072,
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (8, 2),
            "out_core_grid": (8, 2),
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat8_b,
            "out_dtype": ttnn.bfloat16,
            "expected_pcc": 0.999,
            "tile_h": 32,
            "tile_w": 32,
        },
        # wo
        {
            "m": 32,
            "k": 16384,
            "n": 896,
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (8, 1),
            "out_core_grid": (7, 2),
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat8_b,
            "out_dtype": ttnn.bfloat16,
            "expected_pcc": 0.999,
            "tile_h": 32,
            "tile_w": 32,
        },
    ],
    ids=["qkv_a", "wq_b", "wo"],
)
@pytest.mark.parametrize("num_iters", [1])
def test_matmul_l1_dram_sharded(device, test_case, num_iters):
    """
    Test matmul with L1 sharded input1 and DRAM sharded input2.
    Supports both HEIGHT and WIDTH sharding strategies for input1.
    """
    torch.manual_seed(0)

    # Extract test case parameters
    m = test_case["m"]
    k = test_case["k"]
    n = test_case["n"]
    in0_shard_strategy = test_case["in0_shard_strategy"]
    in0_core_grid_y, in0_core_grid_x = test_case["in0_core_grid"]
    out_core_grid_y, out_core_grid_x = test_case["out_core_grid"]
    in0_dtype = test_case["in0_dtype"]
    in1_dtype = test_case["in1_dtype"]
    out_dtype = test_case["out_dtype"]
    expected_pcc = test_case["expected_pcc"]
    tile_h = test_case["tile_h"]
    tile_w = test_case["tile_w"]

    # Tensor shapes
    in0_shape = [1, 1, m, k]
    in1_shape = [1, 1, k, n]

    # DRAM configuration - 12 banks
    num_dram_banks = 12
    n_padded = pad_to_dram_banks(n, tile_w, tile_w * num_dram_banks)
    in1_shard_shape = [k, n_padded // num_dram_banks]

    # Core grid configurations
    in0_core_grid = ttnn.CoreGrid(y=in0_core_grid_y, x=in0_core_grid_x)
    out_core_grid = ttnn.CoreGrid(y=out_core_grid_y, x=out_core_grid_x)
    num_in0_cores = in0_core_grid_y * in0_core_grid_x
    num_out_cores = out_core_grid_y * out_core_grid_x

    # Calculate program config parameters based on sharding strategy
    if in0_shard_strategy == ttnn.ShardStrategy.WIDTH:
        # WIDTH sharding: shard along K dimension
        in0_block_w = k // num_in0_cores // tile_w
    else:  # HEIGHT sharding
        # HEIGHT sharding: shard along M dimension
        in0_block_w = k // tile_w
    per_core_M = m // tile_h
    # Calculate per_core_N ensuring we don't exceed available cores
    # The code calculates: num_blocks_x = ((N_tiles - 1) / per_core_N) + 1
    # and num_cores = num_blocks_x * num_blocks_y
    # We need to ensure num_cores <= available cores (64 for 8x8 grid)
    N_tiles = n // tile_w
    M_tiles = m // tile_h
    num_blocks_y = ((M_tiles - 1) // per_core_M) + 1
    # Maximum number of blocks in x direction to stay within available cores
    # Use device grid size (typically 8x8 = 64 cores) as the limit
    device_grid_size = device.compute_with_storage_grid_size()
    max_available_cores = device_grid_size.x * device_grid_size.y
    max_num_blocks_x = max_available_cores // num_blocks_y if num_blocks_y > 0 else max_available_cores
    if max_num_blocks_x == 0:
        max_num_blocks_x = 1
    # Calculate minimum per_core_N to ensure num_blocks_x <= max_num_blocks_x
    # num_blocks_x = ((N_tiles - 1) / per_core_N) + 1 <= max_num_blocks_x
    # Solving: per_core_N >= ceil((N_tiles - 1) / (max_num_blocks_x - 1))
    if max_num_blocks_x > 1:
        min_per_core_N = ((N_tiles - 1) // (max_num_blocks_x - 1)) + 1
    else:
        min_per_core_N = N_tiles
    # Use the larger of the two: what we want for sharding vs what fits in available cores
    desired_per_core_N = n // num_out_cores // tile_w
    per_core_N = max(min_per_core_N, desired_per_core_N)

    # Create torch tensors
    in0 = torch.randn(in0_shape, dtype=torch.bfloat16)
    in1 = torch.randn(in1_shape, dtype=torch.bfloat16)

    # Input1: L1 sharded memory config (HEIGHT or WIDTH)
    in0_memory_config = ttnn.create_sharded_memory_config(
        in0_shape,
        core_grid=in0_core_grid,
        strategy=in0_shard_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((tile_h, tile_w)),
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
    )

    # Input2: DRAM width-sharded memory config (always 12 banks)
    in1_shard_grid = ttnn.CoreCoord(num_dram_banks - 1, 0)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    in1_t = ttnn.from_torch(
        in1,
        tile=ttnn.Tile((tile_h, tile_w)),
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
    )

    # Output: L1 width-sharded memory config
    out_memory_config = ttnn.create_sharded_memory_config(
        [1, 1, m, n],
        core_grid=out_core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    # Program config
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fused_activation=None,
    )

    # Compute kernel config
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Run matmul - do it three times for perf
    for itr in range(num_iters):
        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=out_memory_config,
            dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
            output_tile=ttnn.Tile((tile_h, tile_w)),
        )

        if itr != num_iters - 1:
            output_t.deallocate()

    # Convert to torch and validate
    output_tensor = ttnn.to_torch(output_t)
    pt_out = in0 @ in1

    pcc_passed, pcc_message = comp_pcc(pt_out, output_tensor, expected_pcc)
    logger.info(pcc_message)
    assert_with_pcc(pt_out, output_tensor, expected_pcc)


@skip_for_blackhole("Deepseek tests target Wormhole")
def test_batched_mm_wkv_b1(device):
    """
    Test batched matmul with L1 width-sharded input0 and DRAM sharded input1.
    Input0: [1, 16, 32, 128] - L1 width-sharded on [1,8] core grid (K=128/8=16 per core)
    Input1: [1, 16, 128, 512] - DRAM sharded across 12 banks
    Batch=16, M=32, K=128, N=512
    """
    torch.manual_seed(0)

    # Tensor shapes
    batch = 16
    m = 32
    k = 128
    n = 512
    in0_shape = [1, batch, m, k]
    in1_shape = [1, batch, k, n]

    # Tile configuration
    tile_h = 32
    tile_w = 32

    # DRAM configuration - 12 banks
    # num_dram_banks = 12
    # n_padded = pad_to_dram_banks(n, tile_w, tile_w * num_dram_banks)
    # in1_shard_shape = [batch*k, n_padded // num_dram_banks]

    # Core grid configuration - [1,4] so K=128 divides evenly to one tile per core
    # in0_core_grid = ttnn.CoreGrid(y=8, x=8)

    # Create torch tensors
    in0 = torch.randn(in0_shape, dtype=torch.bfloat16)
    in1 = torch.randn(in1_shape, dtype=torch.bfloat16)

    # in0_memory_config = ttnn.create_sharded_memory_config(
    #     in0_shape,
    #     core_grid=in0_core_grid,
    #     strategy=ttnn.ShardStrategy.BLOCK,
    #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
    # )

    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((tile_h, tile_w)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        # memory_config=in0_memory_config,
    )

    # Input1: DRAM width-sharded memory config (12 banks)
    # in1_shard_grid = ttnn.CoreCoord(num_dram_banks - 1, 0)
    # in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    # in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    # in1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    in1_t = ttnn.from_torch(
        in1,
        tile=ttnn.Tile((tile_h, tile_w)),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        # memory_config=in1_memory_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Output: L1 width-sharded memory config
    # For width sharding, use 2D physical shape (batch * m, n) since shard height must equal full physical height
    # out_core_grid = ttnn.CoreGrid(y=1, x=4)
    # out_memory_config = ttnn.create_sharded_memory_config(
    #     (batch * m, n),
    #     core_grid=out_core_grid,
    #     strategy=ttnn.ShardStrategy.WIDTH,
    #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
    # )

    # Program config
    # num_in0_cores = in0_core_grid.y * in0_core_grid.x
    # num_out_cores = out_core_grid.y * out_core_grid.x
    # in0_block_w = k // num_in0_cores // tile_w
    # per_core_M = m // tile_h  # Per-batch M dimension (program operates per-batch)
    # per_core_N = n // num_out_cores // tile_w

    # program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
    #     in0_block_w=in0_block_w,
    #     per_core_M=per_core_M,
    #     per_core_N=per_core_N,
    #     fused_activation=None,
    # )

    # Compute kernel config
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Run matmul with DRAM sharded program config
    output_t = ttnn.matmul(
        in0_t,
        in1_t,
        # program_config=program_config,
        # memory_config=out_memory_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        output_tile=ttnn.Tile((tile_h, tile_w)),
    )

    # Convert to torch and validate
    output_tensor = ttnn.to_torch(output_t)
    pt_out = in0 @ in1

    expected_pcc = 0.999
    pcc_passed, pcc_message = comp_pcc(pt_out, output_tensor, expected_pcc)
    logger.info(pcc_message)
    assert_with_pcc(pt_out, output_tensor, expected_pcc)


@skip_for_blackhole("Deepseek tests target Wormhole")
def test_batched_mm_wkv_b2(device):
    """
    Test batched matmul with L1 width-sharded input0 and DRAM sharded input1.
    Input0: [1, 128, 4, 512] - L1 width-sharded on 1x8 core grid, tile layout
    Input1: [1, 128, 512, 128] - DRAM sharded across 1x12 grid, bf8 dtype
    Batch=128, M=4, K=512, N=128
    """
    torch.manual_seed(0)

    # Tensor shapes
    batch = 128
    m = 4
    k = 512
    n = 128
    in0_shape = [1, batch, m, k]
    in1_shape = [1, batch, k, n]

    # Tile configuration
    tile_h = 32
    tile_w = 32

    # DRAM configuration - 12 banks
    num_dram_banks = 12
    n_padded = pad_to_dram_banks(n, tile_w, tile_w * num_dram_banks)
    in1_shard_shape = [batch * k, n_padded // num_dram_banks]

    # Core grid configuration for input0 - 1x8 so K=512 divides to 64 per core (2 tiles)
    in0_core_grid = ttnn.CoreGrid(y=1, x=8)

    # Create torch tensors
    in0 = torch.randn(in0_shape, dtype=torch.bfloat16)
    in1 = torch.randn(in1_shape, dtype=torch.bfloat16)

    # Input0: L1 width-sharded memory config
    # in0_memory_config = ttnn.create_sharded_memory_config(
    #     in0_shape,
    #     core_grid=in0_core_grid,
    #     strategy=ttnn.ShardStrategy.WIDTH,
    #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
    # )
    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((tile_h, tile_w)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        # memory_config=in0_memory_config,
    )

    # Input1: DRAM width-sharded memory config (12 banks, 1x12 grid)
    # in1_shard_grid = ttnn.CoreCoord(num_dram_banks - 1, 0)
    # in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    # in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    # in1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    in1_t = ttnn.from_torch(
        in1,
        tile=ttnn.Tile((tile_h, tile_w)),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        # memory_config=in1_memory_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Output: L1 width-sharded memory config
    # For width sharding, use 2D physical shape (batch * m, n) since shard height must equal full physical height
    # out_core_grid = ttnn.CoreGrid(y=1, x=4)
    # num_out_cores = out_core_grid.y * out_core_grid.x
    # out_memory_config = ttnn.create_sharded_memory_config(
    #     (batch * m, n),
    #     core_grid=out_core_grid,
    #     strategy=ttnn.ShardStrategy.WIDTH,
    #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
    # )

    # Program config
    # Since input0 is width-sharded, in0_block_w is K per core in tiles
    # num_in0_cores = in0_core_grid.y * in0_core_grid.x
    # in0_block_w = k // num_in0_cores // tile_w
    # per_core_M = 1  # M=4 < tile_h=32, so use 1 tile per core
    # per_core_N = n // num_out_cores // tile_w

    # program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
    #     in0_block_w=in0_block_w,
    #     per_core_M=per_core_M,
    #     per_core_N=per_core_N,
    #     fused_activation=None,
    # )

    # Compute kernel config
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Run matmul with DRAM sharded program config
    output_t = ttnn.matmul(
        in0_t,
        in1_t,
        # program_config=program_config,
        # memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        output_tile=ttnn.Tile((tile_h, tile_w)),
    )

    # Convert to torch and validate
    output_tensor = ttnn.to_torch(output_t)
    pt_out = in0 @ in1

    expected_pcc = 0.999
    pcc_passed, pcc_message = comp_pcc(pt_out, output_tensor, expected_pcc)
    logger.info(pcc_message)
    assert_with_pcc(pt_out, output_tensor, expected_pcc)

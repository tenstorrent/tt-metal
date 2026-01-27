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


def pad_batch_to_dram_banks(batch, num_dram_banks=12):
    """Pad batch dimension to be divisible by number of DRAM banks."""
    if batch % num_dram_banks == 0:
        return batch
    padded_batch = ((batch + num_dram_banks - 1) // num_dram_banks) * num_dram_banks
    return padded_batch


def pad_to_tile(dim, tile_size):
    """Pad dimension to be a multiple of tile size."""
    if dim % tile_size == 0:
        return dim
    return ((dim + tile_size - 1) // tile_size) * tile_size


@skip_for_blackhole("Deepseek tests target Wormhole")
@pytest.mark.parametrize(
    "test_case",
    [
        # qkv_a Roofline 8.4, achieve 13.2
        # Compute bound -> could be 10.7us
        # Perhaps tweak transaction IDs, go ahead by 3 transactions instead of 2?
        {
            "m": 32,
            "k": 896,
            "n": 2112,  # this pads up to 2304
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (7, 1),
            "out_core_grid": (8, 1),
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat8_b,
            "out_dtype": ttnn.bfloat16,
            "expected_pcc": 0.999,
            "in0_tile_h": 32,
            "in0_tile_w": 32,
            "in1_tile_h": 32,
            "in1_tile_w": 32,
            "out_tile_h": 32,
            "out_tile_w": 32,
        },
        # wq_b Roofline 21, achieves 23
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
            "in0_tile_h": 32,
            "in0_tile_w": 32,
            "in1_tile_h": 32,
            "in1_tile_w": 32,
            "out_tile_h": 32,
            "out_tile_w": 32,
        },
        # wo Roofline 64.99, achieve 93
        # Can achieve 83 with in0_block_w // 2. This maps to ~250GBps
        {
            "m": 32,
            "k": 16384,
            "n": 896,  # this pads up to 1152
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (8, 1),
            "out_core_grid": (8, 1),
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat8_b,
            "out_dtype": ttnn.bfloat16,
            "expected_pcc": 0.999,
            "in0_tile_h": 32,
            "in0_tile_w": 32,
            "in1_tile_h": 32,
            "in1_tile_w": 32,
            "out_tile_h": 32,
            "out_tile_w": 32,
        },
        # llama shape
        {
            "m": 32,
            "k": 8192,
            "n": 1280,
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (8, 1),
            "out_core_grid": (8, 1),
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat8_b,
            "out_dtype": ttnn.bfloat16,
            "expected_pcc": 0.999,
            "in0_tile_h": 32,
            "in0_tile_w": 32,
            "in1_tile_h": 32,
            "in1_tile_w": 32,
            "out_tile_h": 32,
            "out_tile_w": 32,
        },
        # mlp ff1 / ff3
        {
            "m": 32,
            "k": 7168,
            "n": 3584,  # 2304, padded up
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (7, 8),
            "out_core_grid": (7, 8),
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat4_b,
            "out_dtype": ttnn.bfloat16,
            "expected_pcc": 0.99,
            "in0_tile_h": 32,
            "in0_tile_w": 32,
            "in1_tile_h": 32,
            "in1_tile_w": 32,
            "out_tile_h": 32,
            "out_tile_w": 32,
        },
        # mlp ff2
        {
            "m": 32,
            "k": 3584,  # 2304, padded up
            "n": 7168,
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (7, 8),
            "out_core_grid": (7, 8),
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat4_b,
            "out_dtype": ttnn.bfloat16,
            "expected_pcc": 0.99,
            "in0_tile_h": 32,
            "in0_tile_w": 32,
            "in1_tile_h": 32,
            "in1_tile_w": 32,
            "out_tile_h": 32,
            "out_tile_w": 32,
        },
        # moe gate
        {
            "m": 16,
            "k": 7168,
            "n": 256,
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (7, 8),
            "out_core_grid": (1, 8),
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat4_b,
            "out_dtype": ttnn.bfloat16,
            "expected_pcc": 0.99,
            "in0_tile_h": 16,
            "in0_tile_w": 32,
            "in1_tile_h": 32,
            "in1_tile_w": 32,
            "out_tile_h": 16,
            "out_tile_w": 32,
        },
        # shared expert ff1 / ff3
        {
            "m": 32,
            "k": 7168,
            "n": 384,  # 256, padded up
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (7, 8),
            "out_core_grid": (3, 4),
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat4_b,
            "out_dtype": ttnn.bfloat16,
            "expected_pcc": 0.99,
            "in0_tile_h": 32,
            "in0_tile_w": 32,
            "in1_tile_h": 32,
            "in1_tile_w": 32,
            "out_tile_h": 32,
            "out_tile_w": 32,
        },
        # shared expert ff2
        {
            "m": 32,
            "k": 256,
            "n": 7168,
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (1, 8),
            "out_core_grid": (7, 8),
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat4_b,
            "out_dtype": ttnn.bfloat16,
            "expected_pcc": 0.99,
            "in0_tile_h": 32,
            "in0_tile_w": 32,
            "in1_tile_h": 32,
            "in1_tile_w": 32,
            "out_tile_h": 32,
            "out_tile_w": 32,
        },
        # lm heads
        {
            "m": 32,
            "k": 7168,
            "n": 16512,  # 16160 padded
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (7, 8),
            "out_core_grid": (8, 8),
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat4_b,
            "out_dtype": ttnn.bfloat16,
            "expected_pcc": 0.99,
            "in0_tile_h": 32,
            "in0_tile_w": 32,
            "in1_tile_h": 32,
            "in1_tile_w": 32,
            "out_tile_h": 32,
            "out_tile_w": 32,
        },
    ],
    ids=[
        "qkv_a",
        "wq_b",
        "wo",
        "llama",
        "mlp_ff1_ff3",
        "mlp_ff2",
        "moe_gate",
        "shared_expert_ff1_ff3",
        "shared_expert_mlp_ff2",
        "lm_heads",
    ],
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
    in0_tile_h = test_case["in0_tile_h"]
    in0_tile_w = test_case["in0_tile_w"]
    in1_tile_h = test_case["in1_tile_h"]
    in1_tile_w = test_case["in1_tile_w"]
    out_tile_h = test_case["out_tile_h"]
    out_tile_w = test_case["out_tile_w"]

    # Tensor shapes
    in0_shape = [1, 1, m, k]
    in1_shape = [1, 1, k, n]

    # DRAM configuration - 12 banks
    num_dram_banks = 12
    n_padded = pad_to_dram_banks(n, in1_tile_w, in1_tile_w * num_dram_banks)
    in1_shard_shape = [k, n_padded // num_dram_banks]

    # Core grid configurations
    in0_core_grid = ttnn.CoreGrid(y=in0_core_grid_y, x=in0_core_grid_x)
    out_core_grid = ttnn.CoreGrid(y=out_core_grid_y, x=out_core_grid_x)
    num_in0_cores = in0_core_grid_y * in0_core_grid_x
    num_out_cores = out_core_grid_y * out_core_grid_x

    # Calculate program config parameters based on sharding strategy
    if in0_shard_strategy == ttnn.ShardStrategy.WIDTH:
        # WIDTH sharding: shard along K dimension
        in0_block_w = k // num_in0_cores // in0_tile_w
    else:  # HEIGHT sharding
        # HEIGHT sharding: shard along M dimension
        in0_block_w = k // in0_tile_w
    per_core_M = m // out_tile_h
    # Calculate per_core_N ensuring we don't exceed available cores
    # The code calculates: num_blocks_x = ((N_tiles - 1) / per_core_N) + 1
    # and num_cores = num_blocks_x * num_blocks_y
    # We need to ensure num_cores <= available cores (64 for 8x8 grid)
    N_tiles = n // out_tile_w
    M_tiles = m // out_tile_h
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
    desired_per_core_N = n // num_out_cores // out_tile_w
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
        tile=ttnn.Tile((in0_tile_h, in0_tile_w)),
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
        tile=ttnn.Tile((in1_tile_h, in1_tile_w)),
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
            output_tile=ttnn.Tile((out_tile_h, out_tile_w)),
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
@pytest.mark.parametrize(
    "test_case",
    [
        {
            "batch": 48,
            "m": 64,
            "n": 96,
            "k": 32,
            "tile_h": 32,
            "tile_w": 32,
            "in0_core_grid_x": 6,
            "in0_core_grid_y": 2,
            "out_core_grid_x": 6,
            "out_core_grid_y": 2,
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat8_b,
            "expected_pcc": 0.9997,
        },
        {
            "batch": 12,
            "m": 32,
            "n": 64,
            "k": 128,
            "tile_h": 32,
            "tile_w": 32,
            "in0_core_grid_x": 4,
            "in0_core_grid_y": 3,
            "out_core_grid_x": 4,
            "out_core_grid_y": 3,
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat8_b,
            "expected_pcc": 0.9997,
        },
        # wkv_b1 takes roughly 19us. Compute bound, could be 15us
        {
            "batch": 16,  # Pads to 24
            "m": 32,
            "n": 128,
            "k": 512,
            "tile_h": 32,
            "tile_w": 32,
            "in0_core_grid_x": 3,
            "in0_core_grid_y": 4,
            "out_core_grid_x": 3,
            "out_core_grid_y": 4,
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat8_b,
            "expected_pcc": 0.9997,
        },
        # wkv_b2 takes roughly 63us. Compute bound, could be 45us.
        {
            "batch": 128,  # Pads to 132
            "m": 4,
            "k": 512,
            "n": 128,
            "tile_h": 4,  # Tiny tile needed to avoid padding
            "tile_w": 32,
            "in0_core_grid_x": 3,
            "in0_core_grid_y": 4,
            "out_core_grid_x": 3,
            "out_core_grid_y": 4,
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat8_b,
            "expected_pcc": 0.9997,
        },
    ],
    ids=[
        "batch48_m64_n96_k32",
        "batch12_m32_n32_k64",
        "wkv_b1",
        "wkv_b2",
    ],
)
@skip_for_blackhole("Deepseek tests target Wormhole")
def test_matmul_batched_dram_sharded(device, test_case):
    """
    Test for Batch-Sharded DRAM Matmul.

    Batched matmul: [1, B, M, N] x [1, B, N, K] = [1, B, M, K]

    Batch sharding layout - each worker handles B/num_cores complete matmuls:
    - in0: L1 sharded by batch - each core has batches_per_core complete [M, N] matrices
    - in1: DRAM sharded by batch - each bank has batches_per_core complete [N, K] matrices
    - output: L1 sharded by batch - each core outputs batches_per_core complete [M, K] matrices
    """
    torch.manual_seed(0)

    # Extract test case parameters
    batch = test_case["batch"]
    m = test_case["m"]
    n = test_case["n"]
    k = test_case["k"]
    tile_h = test_case["tile_h"]
    tile_w = test_case["tile_w"]
    in0_core_grid_x = test_case["in0_core_grid_x"]
    in0_core_grid_y = test_case["in0_core_grid_y"]
    out_core_grid_x = test_case["out_core_grid_x"]
    out_core_grid_y = test_case["out_core_grid_y"]
    in0_dtype = test_case["in0_dtype"]
    in1_dtype = test_case["in1_dtype"]
    expected_pcc = test_case["expected_pcc"]

    # Dimensions for batched matmul
    num_dram_banks = 12  # Wormhole has 12 DRAM banks

    # Pad batch to be divisible by num_dram_banks (required for even sharding)
    batch_padded = pad_batch_to_dram_banks(batch, num_dram_banks)

    # Pad m, n, k to tile dimensions (required for tile-aligned shards)
    m_padded = pad_to_tile(m, tile_h)
    n_padded = pad_to_tile(n, tile_w)
    k_padded = pad_to_tile(k, tile_w)

    num_in0_cores = in0_core_grid_x * in0_core_grid_y
    num_out_cores = out_core_grid_x * out_core_grid_y

    # Core grids must equal num_dram_banks for 1:1 worker mapping
    assert (
        num_in0_cores == num_dram_banks
    ), f"in0 core grid ({num_in0_cores}) must equal num_dram_banks ({num_dram_banks})"
    assert (
        num_out_cores == num_dram_banks
    ), f"out core grid ({num_out_cores}) must equal num_dram_banks ({num_dram_banks})"

    batches_per_core_in0 = batch_padded // num_in0_cores
    batches_per_core_out = batch_padded // num_out_cores

    # Shapes with padded dimensions: [1, B_padded, M_padded, N_padded] x [1, B_padded, N_padded, K_padded] = [1, B_padded, M_padded, K_padded]
    in0_shape_padded = [1, batch_padded, m_padded, n_padded]
    in1_shape_padded = [1, batch_padded, n_padded, k_padded]
    out_shape_padded = [1, batch_padded, m_padded, k_padded]

    # Create random input data at original size
    in0_orig = torch.randn([1, batch, m, n], dtype=torch.bfloat16)
    in1_orig = torch.randn([1, batch, n, k], dtype=torch.bfloat16)

    # Pad tensors to padded dimensions (pad with zeros)
    in0 = torch.zeros(in0_shape_padded, dtype=torch.bfloat16)
    in0[:, :batch, :m, :n] = in0_orig
    in1 = torch.zeros(in1_shape_padded, dtype=torch.bfloat16)
    in1[:, :batch, :n, :k] = in1_orig

    print(f"\n=== Batch-sharded DRAM matmul test ===")
    print(f"original: batch={batch}, m={m}, n={n}, k={k}")
    print(f"padded:   batch={batch_padded}, m={m_padded}, n={n_padded}, k={k_padded}")
    print(f"in0_shape: {in0_shape_padded}, in1_shape: {in1_shape_padded}, out_shape: {out_shape_padded}")

    # in0 L1 shard grid
    in0_shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(in0_core_grid_x - 1, in0_core_grid_y - 1))}
    )

    # Output L1 shard grid
    out_shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(out_core_grid_x - 1, out_core_grid_y - 1))}
    )

    # DRAM shard grid: 12 DRAM banks (1D grid from 0 to 11)
    dram_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))})

    # in0: L1 sharded by batch
    # Each core gets batches_per_core_in0 complete [M_padded, N_padded] matrices
    # Shard shape: [batches_per_core_in0 * M_padded, N_padded]
    in0_shard_shape = [batches_per_core_in0 * m_padded, n_padded]
    in0_shard_spec = ttnn.ShardSpec(in0_shard_grid, in0_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in0_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec)

    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((tile_h, tile_w)),
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
    )
    print(f"Created in0_t (batch-sharded in L1 on {num_in0_cores} cores)")
    print(f"in0_shape: {in0_shape_padded}, shard_shape: {in0_shard_shape}")

    # in1: DRAM sharded by batch across 12 DRAM banks
    # Each DRAM bank gets batches_per_dram_bank complete [N_padded, K_padded] matrices
    # Shard shape: [batches_per_dram_bank * N_padded, K_padded]
    batches_per_dram_bank = batch_padded // num_dram_banks
    in1_shard_shape = [batches_per_dram_bank * n_padded, k_padded]
    in1_shard_spec = ttnn.ShardSpec(dram_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)

    in1_t = ttnn.from_torch(
        in1,
        tile=ttnn.Tile((32, tile_w)),  # in1 tile height must be 32 (inner dim constraint)
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
    )

    # Output: L1 sharded by batch
    # Each core outputs batches_per_core_out complete [M_padded, K_padded] matrices
    # Shard shape: [batches_per_core_out * M_padded, K_padded]
    out_shard_shape = [batches_per_core_out * m_padded, k_padded]
    out_shard_spec = ttnn.ShardSpec(out_shard_grid, out_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    out_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard_spec)

    # Program config for batch sharding:
    # - in0_block_w: N_padded tiles per block (inner loop over N dimension)
    # - per_core_M: M_padded tiles per matmul (full M)
    # - per_core_N: K_padded tiles per matmul (full K) - note: this is output width
    in0_block_w = n_padded // tile_w
    per_core_M = m_padded // tile_h
    per_core_N = k_padded // tile_w

    program_config = ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fused_activation=None,
    )

    # Run batched matmul
    output_t = ttnn.matmul(
        in0_t,
        in1_t,
        program_config=program_config,
        memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        output_tile=ttnn.Tile((tile_h, tile_w)),  # Output tile matches in0 tile height
    )

    # Validate
    output_tensor = ttnn.to_torch(output_t)

    # Slice off padding from output to get original dimensions [1, batch, m, k]
    output_tensor = output_tensor[:, :batch, :m, :k]

    # PyTorch batched matmul using original (unpadded) tensors
    pt_out = torch.matmul(in0_orig, in1_orig)

    # Lower PCC threshold due to bfloat8_b weights (lower precision than bfloat16)
    pcc_passed, pcc_message = comp_pcc(pt_out, output_tensor, expected_pcc)
    logger.info(f"Batch-sharded DRAM matmul test: {pcc_message}")
    assert_with_pcc(pt_out, output_tensor, expected_pcc)

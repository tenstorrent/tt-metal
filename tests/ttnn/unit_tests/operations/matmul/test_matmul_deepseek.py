# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

from loguru import logger
import pytest
import torch
import math
import ttnn

from tests.ttnn.unit_tests.operations.matmul.test_matmul import pad_to_dram_banks
from models.common.utility_functions import comp_pcc, skip_for_blackhole, skip_with_watcher
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
            "n": 2112,  # pads to 2304 (72 tiles), out_core_grid must divide 72
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (1, 7),
            "out_core_grid": (1, 8),
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
            "n": 3072,  # already aligned (96 tiles), out_core_grid must divide 96
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (2, 8),
            "out_core_grid": (2, 8),
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
            "n": 896,  # pads to 1152 (36 tiles), out_core_grid must divide 36
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (1, 8),
            "out_core_grid": (1, 6),
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
            "n": 1280,  # pads to 1536 (48 tiles), out_core_grid must divide 48
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (1, 8),
            "out_core_grid": (1, 8),
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
            "n": 3584,  # pads to 3840 (120 tiles), out_core_grid must divide 120
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (7, 8),
            "out_core_grid": (5, 8),
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
            "k": 3584,
            "n": 7168,  # pads to 7296 (228 tiles), out_core_grid must divide 228
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (7, 8),
            "out_core_grid": (2, 6),
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
            "n": 256,  # pads to 384 (12 tiles), out_core_grid must divide 12
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (7, 8),
            "out_core_grid": (1, 4),
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
            "n": 384,  # already aligned (12 tiles), out_core_grid must divide 12
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (7, 8),
            "out_core_grid": (2, 6),
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
            "n": 7168,  # pads to 7296 (228 tiles), out_core_grid must divide 228
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (1, 8),
            "out_core_grid": (2, 6),
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
            "n": 16512,  # already aligned (516 tiles), out_core_grid must divide 516
            "in0_shard_strategy": ttnn.ShardStrategy.WIDTH,
            "in0_core_grid": (7, 8),
            "out_core_grid": (2, 6),
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
@skip_with_watcher("Skipping test with watcher enabled due to failure, see github issue #36314")
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
    N_tiles = n_padded // out_tile_w
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
    desired_per_core_N = n_padded // num_out_cores // out_tile_w
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

    # Output: L1 width-sharded memory config using padded dimensions
    out_memory_config = ttnn.create_sharded_memory_config(
        [1, 1, m, n_padded],
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


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "batch": 48,
            "m": 64,
            "k": 32,
            "n": 96,
            "tile_h": 32,
            "tile_w": 32,
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat8_b,
            "expected_pcc": 0.9997,
        },
        {
            "batch": 12,
            "m": 32,
            "k": 128,
            "n": 64,
            "tile_h": 32,
            "tile_w": 32,
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat8_b,
            "expected_pcc": 0.9997,
        },
        {
            "batch": 7,
            "m": 32,
            "k": 32,
            "n": 32,
            "tile_h": 32,
            "tile_w": 32,
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat8_b,
            "expected_pcc": 0.9997,
        },
        # wkv_b1 takes roughly 19us. Compute bound, could be 15us
        {
            "batch": 16,  # Pads to 24 (Wormhole) or 21 (Blackhole)
            "m": 32,
            "k": 128,
            "n": 512,
            "tile_h": 32,
            "tile_w": 32,
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat8_b,
            "expected_pcc": 0.9997,
        },
        # wkv_b2 takes roughly 63us. Compute bound, could be 45us.
        {
            "batch": 128,  # Pads to 132 (Wormhole) or 133 (Blackhole)
            "m": 4,
            "k": 512,
            "n": 128,
            "tile_h": 4,  # Tiny tile needed to avoid padding
            "tile_w": 32,
            "in0_dtype": ttnn.bfloat16,
            "in1_dtype": ttnn.bfloat8_b,
            "expected_pcc": 0.9997,
        },
    ],
    ids=[
        "batch48_m64_n96_k32",
        "wh_sanity",
        "bh_sanity",
        "wkv_b1",
        "wkv_b2",
    ],
)
@skip_with_watcher("Skipping test with watcher enabled due to failure, see github issue #36314")
def test_matmul_batched_dram_sharded(device, test_case):
    """
    Test for Batch-Sharded DRAM Matmul.

    Batched matmul: [1, B, M, K] x [1, B, K, N] = [1, B, M, N]

    Batch sharding layout - each worker handles B/num_cores complete matmuls:
    - in0: L1 sharded by batch - each core has batches_per_core complete [M, K] matrices
    - in1: DRAM sharded by batch - each bank has batches_per_core complete [K, N] matrices
    - output: L1 sharded by batch - each core outputs batches_per_core complete [M, N] matrices
    """
    torch.manual_seed(0)

    # Extract test case parameters
    # Note: test case dict keys "k" and "n" are swapped relative to standard convention
    # "n" in dict = contracted dimension = K, "k" in dict = output cols = N
    batch = test_case["batch"]
    m = test_case["m"]
    k = test_case["k"]  # contracted dimension (cols of A, rows of B)
    n = test_case["n"]  # output cols (cols of B)
    tile_h = test_case["tile_h"]
    tile_w = test_case["tile_w"]
    in0_dtype = test_case["in0_dtype"]
    in1_dtype = test_case["in1_dtype"]
    expected_pcc = test_case["expected_pcc"]

    # Get the optimal DRAM bank-to-worker core assignment from the device.
    # The factory uses this same assignment to map workers to DRAM banks.
    optimal_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)

    num_dram_banks = len(optimal_worker_cores)
    num_in0_cores = num_dram_banks
    num_out_cores = num_dram_banks

    # Pad batch to be divisible by num_dram_banks (required for even sharding)
    batch_padded = pad_batch_to_dram_banks(batch, num_dram_banks)

    # Pad m, k, n to tile dimensions (required for tile-aligned shards)
    m_padded = pad_to_tile(m, tile_h)
    k_padded = pad_to_tile(k, tile_w)
    n_padded = pad_to_tile(n, tile_w)

    batches_per_core_in0 = batch_padded // num_in0_cores
    batches_per_core_out = batch_padded // num_out_cores

    # Shapes with padded dimensions: [1, B_padded, M_padded, K_padded] x [1, B_padded, K_padded, N_padded] = [1, B_padded, M_padded, N_padded]
    in0_shape_padded = [1, batch_padded, m_padded, k_padded]
    in1_shape_padded = [1, batch_padded, k_padded, n_padded]

    # Create random input data at original size
    in0_orig = torch.randn([1, batch, m, k], dtype=torch.bfloat16)
    in1_orig = torch.randn([1, batch, k, n], dtype=torch.bfloat16)

    # Pad tensors to padded dimensions (pad with zeros)
    in0 = torch.zeros(in0_shape_padded, dtype=torch.bfloat16)
    in0[:, :batch, :m, :k] = in0_orig
    in1 = torch.zeros(in1_shape_padded, dtype=torch.bfloat16)
    in1[:, :batch, :k, :n] = in1_orig

    # Use optimal worker cores for L1 shard grids - this ensures the shard ordering
    # matches the factory's worker ordering (critical for correct data routing!)
    in0_shard_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in optimal_worker_cores]
    )

    # Output L1 shard grid - same cores as input
    out_shard_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in optimal_worker_cores]
    )

    # DRAM shard grid: 12 DRAM banks (1D grid from 0 to 11)
    dram_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))})

    # in0: L1 sharded by batch
    # Each core gets batches_per_core_in0 complete [M_padded, K_padded] matrices
    # Shard shape: [batches_per_core_in0 * M_padded, K_padded]
    in0_shard_shape = [batches_per_core_in0 * m_padded, k_padded]
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

    # in1: DRAM sharded by batch across 12 DRAM banks
    # Each DRAM bank gets batches_per_dram_bank complete [K_padded, N_padded] matrices
    # Shard shape: [batches_per_dram_bank * K_padded, N_padded]
    batches_per_dram_bank = batch_padded // num_dram_banks
    in1_shard_shape = [batches_per_dram_bank * k_padded, n_padded]
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
    # Each core outputs batches_per_core_out complete [M_padded, N_padded] matrices
    # Shard shape: [batches_per_core_out * M_padded, N_padded]
    out_shard_shape = [batches_per_core_out * m_padded, n_padded]
    out_shard_spec = ttnn.ShardSpec(out_shard_grid, out_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    out_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard_spec)

    # Program config for batch sharding:
    # - in0_block_w: K_padded tiles per block (inner loop over K dimension)
    # - per_core_M: M_padded tiles per matmul (full M)
    # - per_core_N: N_padded tiles per matmul (full N) - output width
    in0_block_w = k_padded // tile_w
    per_core_M = m_padded // tile_h
    per_core_N = n_padded // tile_w

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

    # Slice off padding from output to get original dimensions [1, batch, m, n]
    output_tensor = output_tensor[:, :batch, :m, :n]

    # PyTorch batched matmul using original (unpadded) tensors
    pt_out = torch.matmul(in0_orig, in1_orig)

    # Lower PCC threshold due to bfloat8_b weights (lower precision than bfloat16)
    pcc_passed, pcc_message = comp_pcc(pt_out, output_tensor, expected_pcc)
    assert pcc_passed


@pytest.mark.parametrize(
    "batch, m, k, n",
    [
        (12, 32, 64, 32),
        (24, 64, 128, 64),
    ],
    ids=["batch12_m32_k64_n32", "batch24_m64_k128_n64"],
)
@skip_with_watcher("Skipping test with watcher enabled due to failure, see github issue #36314")
def test_matmul_batched_dram_sharded_program_cache(device, batch, m, k, n):
    """
    Test program cache behavior for Batch-Sharded DRAM Matmul.

    Runs the matmul twice with a dummy tensor allocation between runs to verify
    that the program is cached and reused (only 1 program cache entry).
    """
    torch.manual_seed(0)

    tile_h = 32
    tile_w = 32
    expected_pcc = 0.9997

    # Get the optimal DRAM bank-to-worker core assignment from the device.
    # The factory uses this same assignment to map workers to DRAM banks.
    optimal_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)

    num_dram_banks = len(optimal_worker_cores)
    num_in0_cores = num_dram_banks
    num_out_cores = num_dram_banks

    # Pad batch to be divisible by num_dram_banks (required for even sharding)
    batch_padded = pad_batch_to_dram_banks(batch, num_dram_banks)

    batches_per_core_in0 = batch_padded // num_in0_cores
    batches_per_core_out = batch_padded // num_out_cores

    in0_shape_padded = [1, batch_padded, m, k]
    in1_shape_padded = [1, batch_padded, k, n]

    # Use optimal worker cores for L1 shard grids - this ensures the shard ordering
    # matches the factory's worker ordering (critical for correct data routing!)
    in0_shard_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in optimal_worker_cores]
    )

    # Output L1 shard grid - same cores as input
    out_shard_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in optimal_worker_cores]
    )

    # DRAM shard grid: num_dram_banks DRAM banks (1D grid from 0 to num_dram_banks-1)
    dram_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))})

    # in0: L1 sharded by batch
    in0_shard_shape = [batches_per_core_in0 * m, k]
    in0_shard_spec = ttnn.ShardSpec(in0_shard_grid, in0_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in0_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec)

    # in1: DRAM sharded by batch
    batches_per_dram_bank = batch_padded // num_dram_banks
    in1_shard_shape = [batches_per_dram_bank * k, n]
    in1_shard_spec = ttnn.ShardSpec(dram_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)

    # Output: L1 sharded by batch
    out_shard_shape = [batches_per_core_out * m, n]
    out_shard_spec = ttnn.ShardSpec(out_shard_grid, out_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    out_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard_spec)

    # Program config
    in0_block_w = k // tile_w
    per_core_M = m // tile_h
    per_core_N = n // tile_w

    program_config = ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fused_activation=None,
    )

    for _ in range(2):
        # Create input tensors at original size
        in0_orig = torch.randn([1, batch, m, k], dtype=torch.bfloat16)
        in1_orig = torch.randn([1, batch, k, n], dtype=torch.bfloat16)

        # Pad tensors to padded dimensions (pad with zeros)
        in0 = torch.zeros(in0_shape_padded, dtype=torch.bfloat16)
        in0[:, :batch, :, :] = in0_orig
        in1 = torch.zeros(in1_shape_padded, dtype=torch.bfloat16)
        in1[:, :batch, :, :] = in1_orig

        in0_t = ttnn.from_torch(
            in0,
            tile=ttnn.Tile((tile_h, tile_w)),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=in0_memory_config,
        )

        in1_t = ttnn.from_torch(
            in1,
            tile=ttnn.Tile((32, tile_w)),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=in1_memory_config,
        )

        # Run batched matmul
        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=out_memory_config,
            dtype=ttnn.bfloat16,
            output_tile=ttnn.Tile((tile_h, tile_w)),
        )

        # Validate correctness
        output_tensor = ttnn.to_torch(output_t)
        # Slice off padding from output to get original dimensions [1, batch, m, n]
        output_tensor = output_tensor[:, :batch, :m, :n]
        pt_out = torch.matmul(in0_orig, in1_orig)
        assert_with_pcc(pt_out, output_tensor, expected_pcc)

        # Dummy tensor to change tensor allocation (tests program cache robustness)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.from_torch(
            py_dummy_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    assert device.num_program_cache_entries() == 1

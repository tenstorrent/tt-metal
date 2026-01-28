# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Multi-Core Fused SwiGLU with DISJOINT Gate/Up Grids Test

Tests multi-core fused SwiGLU with mcast input distribution where gate and up
weights are on DISJOINT core grids:
- W_gate: [K, N] WIDTH_SHARDED on gate_grid (e.g., 8x9 = 72 cores, 32 per core)
- W_up: [K, N] WIDTH_SHARDED on up_grid (e.g., 4x9 = 36 cores, 64 per core)

Architecture:
  - Input activations: HEIGHT_SHARDED on single sender core
  - W_gate: WIDTH_SHARDED across gate_grid (separate from up_grid)
  - W_up: WIDTH_SHARDED across up_grid (separate from gate_grid)
  - Output: WIDTH_SHARDED across up_grid
  - Mcast broadcasts input to both grids
  - Gate cores compute SiLU(input @ W_gate) and send results to up cores
  - Up cores compute input @ W_up, then multiply with received gate results

Computes: output = SiLU(input @ W_gate) * (input @ W_up)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.mcast_matmul.op import McastDisjointSwiGLUMultiCore


@pytest.mark.parametrize(
    "M, K, N, gate_cols, gate_rows, up_cols, up_rows, in0_dtype, weights_dtype",
    [
        # Small test case with row-oriented grid (similar to working test layout)
        # gate 2x4=8 cores, up 1x4=4 cores -> 3x4 total
        # N=384, gate 48/core (1.5 tiles), up 96/core (3 tiles)
        # Wait, need 32-tile alignment. Let's use:
        # N=256, gate_cols=2 (4 cores per row), up_cols=1 (2 cores per row)
        # Actually let's use a simpler config: N=256, 8 gate cores, 4 up cores
        # Per gate core: 256/8 = 32, per up core: 256/4 = 64
        (1, 256, 256, 2, 4, 1, 4, ttnn.bfloat16, ttnn.bfloat8_b),
        # Medium test case: gate 4x3=12 cores, up 2x3=6 cores
        # N=384, gate 32/core, up 64/core
        (1, 512, 384, 4, 3, 2, 3, ttnn.bfloat16, ttnn.bfloat8_b),
        # DeepSeek SwiGLU shape: gate 8x9=72 cores, up 4x9=36 cores
        # [1, 7168] x [7168, 2304] with gate 32/core, up 64/core
        (1, 7168, 2304, 8, 9, 4, 9, ttnn.bfloat16, ttnn.bfloat8_b),
        # DeepSeek SwiGLU with bfloat4_b weights
        (1, 7168, 2304, 8, 9, 4, 9, ttnn.bfloat16, ttnn.bfloat4_b),
    ],
)
def test_mcast_disjoint_swiglu(device, M, K, N, gate_cols, gate_rows, up_cols, up_rows, in0_dtype, weights_dtype):
    """Test multi-core fused SwiGLU with disjoint gate/up grids"""

    gate_num_cores = gate_cols * gate_rows
    up_num_cores = up_cols * up_rows
    N_per_gate_core = N // gate_num_cores
    N_per_up_core = N // up_num_cores

    # Validate the 2:1 ratio
    assert gate_cols == 2 * up_cols, f"Gate cols ({gate_cols}) must be 2x up cols ({up_cols})"
    assert gate_rows == up_rows, f"Gate rows ({gate_rows}) must equal up rows ({up_rows})"
    assert (
        N_per_up_core == 2 * N_per_gate_core
    ), f"Up per core ({N_per_up_core}) must be 2x gate per core ({N_per_gate_core})"

    # Tile dimensions
    a_tile = ttnn.Tile([M, 32])  # Input: M x 32 tiles
    b_tile = ttnn.Tile([32, 32])  # Weights: 32x32 tiles
    out_tile = ttnn.Tile([M, 32])  # Output: M x 32 tiles

    # Calculate tiles
    num_tiles_k = K // a_tile.tile_shape[1]
    num_tiles_n_per_gate_core = N_per_gate_core // b_tile.tile_shape[1]
    num_tiles_n_per_up_core = N_per_up_core // out_tile.tile_shape[1]

    logger.info(f"Testing multi-core fused SwiGLU with DISJOINT gate/up grids")
    logger.info(f"Shape: [{M}, {K}] x [{K}, {N}]")
    logger.info(f"Gate grid: {gate_cols}x{gate_rows} = {gate_num_cores} cores, {N_per_gate_core} per core")
    logger.info(f"Up grid: {up_cols}x{up_rows} = {up_num_cores} cores, {N_per_up_core} per core")
    logger.info(f"Dtypes: in0={in0_dtype}, weights={weights_dtype}")

    # Create input and weight tensors
    torch.manual_seed(42)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gate = torch.randn((K, N), dtype=torch.bfloat16)
    torch_up = torch.randn((K, N), dtype=torch.bfloat16)

    # Compute reference output using PyTorch: SiLU(input @ W_gate) * (input @ W_up)
    torch_expected = McastDisjointSwiGLUMultiCore.golden(
        torch_input.float(), torch_gate.float(), torch_up.float()
    ).bfloat16()

    # ========================================================================
    # Input tensor: HEIGHT_SHARDED on single sender core
    # Place sender at a core NOT in gate or up grids to test separate sender
    # ========================================================================
    # Gate grid: (0, 0) to (gate_cols-1, gate_rows-1)
    # Up grid: (gate_cols, 0) to (gate_cols+up_cols-1, up_rows-1)
    # Sender: beyond both grids with a gap to avoid proximity issues
    # NOTE: Placing sender immediately adjacent to mcast grid causes hangs.
    # Leave at least 4 columns of gap between mcast grid and sender.
    mcast_end_x = gate_cols + up_cols - 1  # Rightmost column of mcast grid
    sender_x = max(mcast_end_x + 5, 7)  # At least 5 columns gap, minimum column 7
    sender_core = ttnn.CoreCoord(sender_x, 0)
    sender_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(sender_core, sender_core)})

    input_shard_shape = (M, K)
    input_shard_spec = ttnn.ShardSpec(
        sender_core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        input_shard_spec,
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=a_tile,
    )

    logger.info(f"Created input tensor on sender core ({sender_x},0)")

    # ========================================================================
    # W_gate tensor: WIDTH_SHARDED across gate grid (separate from up grid)
    # Gate grid: (0, 0) to (gate_cols-1, gate_rows-1)
    # ========================================================================
    gate_grid_start = ttnn.CoreCoord(0, 0)
    gate_grid_end = ttnn.CoreCoord(gate_cols - 1, gate_rows - 1)
    gate_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(gate_grid_start, gate_grid_end)})

    gate_shard_shape = (K, N_per_gate_core)
    gate_shard_spec = ttnn.ShardSpec(
        gate_core_grid,
        gate_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gate_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        gate_shard_spec,
    )

    ttnn_gate = ttnn.from_torch(
        torch_gate,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_mem_config,
        tile=b_tile,
    )

    logger.info(f"Created W_gate tensor across gate grid (0,0)-({gate_cols-1},{gate_rows-1})")

    # ========================================================================
    # W_up tensor: WIDTH_SHARDED across up grid (DISJOINT from gate grid)
    # Up grid: (gate_cols, 0) to (gate_cols+up_cols-1, up_rows-1)
    # ========================================================================
    up_grid_start = ttnn.CoreCoord(gate_cols, 0)
    up_grid_end = ttnn.CoreCoord(gate_cols + up_cols - 1, up_rows - 1)
    up_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(up_grid_start, up_grid_end)})

    up_shard_shape = (K, N_per_up_core)
    up_shard_spec = ttnn.ShardSpec(
        up_core_grid,
        up_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    up_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        up_shard_spec,
    )

    ttnn_up = ttnn.from_torch(
        torch_up,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=up_mem_config,
        tile=b_tile,
    )

    logger.info(f"Created W_up tensor across up grid ({gate_cols},0)-({gate_cols+up_cols-1},{up_rows-1})")

    # ========================================================================
    # Output tensor: WIDTH_SHARDED across up grid (same as W_up)
    # ========================================================================
    output_shard_shape = (M, N_per_up_core)
    output_shard_spec = ttnn.ShardSpec(
        up_core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        output_shard_spec,
    )

    torch_output_zeros = torch.zeros((M, N), dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )

    logger.info(f"Created output tensor across up grid")

    # ========================================================================
    # Run multi-core fused SwiGLU with disjoint grids
    # ========================================================================
    logger.info("Running multi-core fused SwiGLU with disjoint gate/up grids...")
    ttnn_result = McastDisjointSwiGLUMultiCore.op(
        ttnn_input,
        ttnn_gate,
        ttnn_up,
        ttnn_output,
        fp32_dest_acc_en=False,
    )

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == (M, N), f"Expected shape ({M}, {N}), got {output_torch.shape}"

    # Verify fused SwiGLU results
    logger.info("Verifying fused SwiGLU results...")

    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.97)
    logger.info(pcc_message)

    assert passing, pcc_message

    logger.info("✓ Multi-core fused SwiGLU with disjoint grids test passed!")

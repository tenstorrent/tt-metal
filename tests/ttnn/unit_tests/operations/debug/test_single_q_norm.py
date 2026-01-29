# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple test that runs a single iteration of Q norm with:
1. ttnn.rms_norm() (direct)
2. launch_composite() with single-element list

This test verifies both methods produce the same results.
"""

import pytest
import torch
import ttnn


def create_q_norm_tensors(device):
    """
    Create q_norm tensors:
    - L1 width sharded on cores (0,0) to (3,3) = 16 cores
    - Shard shape: [32, 96]
    - Total tensor shape: [1, 1, 32, 1536] (16 cores × 96 = 1536)
    - Weights are DRAM interleaved
    """
    torch.manual_seed(42)

    # Core range: (0,0) to (3,3) = 4x4 = 16 cores
    q_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    num_cores = 16
    shard_width = 96
    total_width = num_cores * shard_width  # 1536

    # Tensor shapes
    input_shape = (1, 1, 32, total_width)
    weight_shape = (1, 1, 1, total_width)

    # Create torch tensors
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_weight = torch.rand(weight_shape, dtype=torch.bfloat16)

    # Create sharded memory config for input (L1 width sharded)
    shard_spec = ttnn.ShardSpec(q_cores, [32, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    # Move input to device with sharded memory config
    input_tensor = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_mem_config,
    )

    # Weights are DRAM interleaved
    weight_tensor = ttnn.from_torch(
        torch_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Program config for sharded RMS norm
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(4, 4),  # 4x4 grid
        subblock_w=shard_width // 32,  # tiles per shard width = 96/32 = 3
        block_h=1,  # 32/32 = 1 tile high
        block_w=shard_width // 32,  # 3 tiles wide per core
        inplace=False,
    )

    return {
        "input": input_tensor,
        "weight": weight_tensor,
        "cores": q_cores,
        "memory_config": sharded_mem_config,
        "program_config": program_config,
        "torch_input": torch_input,
        "torch_weight": torch_weight,
    }


def test_single_q_norm_comparison(device):
    """Test Q norm with both ttnn.rms_norm() and launch_composite()."""

    # Create Q tensors
    q_tensors = create_q_norm_tensors(device)

    # Clone input tensor so we can use the same data for both operations
    # (in case the first operation modifies the input)
    input_clone = ttnn.clone(q_tensors["input"])

    print("\n=== Running Q norm with ttnn.rms_norm() (direct) ===")
    # Run Q norm with direct ttnn.rms_norm()
    output_direct = ttnn.rms_norm(
        q_tensors["input"],
        epsilon=1e-5,
        weight=q_tensors["weight"],
        memory_config=q_tensors["memory_config"],
        program_config=q_tensors["program_config"],
    )
    ttnn.synchronize_device(device)

    print("\n=== Running Q norm with launch_composite() ===")
    # Run Q norm with launch_composite() using the cloned input
    q_branch = ttnn.experimental.programs.rms_norm(
        input_clone,
        epsilon=1e-5,
        weight=q_tensors["weight"],
        memory_config=q_tensors["memory_config"],
        core_range_set=q_tensors["cores"],
    )
    output_composite = ttnn.experimental.launch_composite([q_branch])[0]
    ttnn.synchronize_device(device)

    # Convert outputs to torch for comparison
    output_direct_torch = ttnn.to_torch(output_direct)
    output_composite_torch = ttnn.to_torch(output_composite)

    # Compare outputs
    print("\n=== Comparing outputs ===")
    max_error = torch.max(torch.abs(output_direct_torch - output_composite_torch)).item()
    mean_error = torch.mean(torch.abs(output_direct_torch - output_composite_torch)).item()

    print(f"Max error: {max_error:.6f}")
    print(f"Mean error: {mean_error:.6f}")

    # Check if outputs are close (allowing for bfloat16 precision)
    # Note: launch_composite() may have slightly different numerical precision than direct ttnn.rms_norm()
    # Using slightly relaxed tolerance to account for this
    is_close = torch.allclose(
        output_direct_torch,
        output_composite_torch,
        rtol=3e-2,  # 3% relative tolerance
        atol=3e-2,  # 3% absolute tolerance
    )

    print(f"Outputs match (allclose): {is_close}")

    assert is_close, f"Outputs do not match! Max error: {max_error}, Mean error: {mean_error}"

    print("\n✓ Test passed: Both methods produce equivalent results!")

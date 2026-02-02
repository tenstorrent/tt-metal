# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for MoE Routed Expert fused operation.

Tests the fused operation:
- Input: [1, 7168] tensor on sender core (outside compute grid)
- Mcast from sender to 8 compute cores
- Each compute core: [1, 7168] @ [7168, 32] -> [1, 32] + sigmoid
- Gather outputs back to sender core -> [1, 256] = [16, 16]
- Gate: top-8 expert selection with normalized scores
- Output: top8 scores [1, 16] + top8 indices [1, 16] on sender core
"""

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.fused_ops.moe_routed_expert.op import MoeRoutedExpert
from models.demos.deepseek_v3_b1.micro_ops.deepseek_moe_gate.op import DeepseekMoeGateSingleCore


def test_moe_routed_expert(device):
    """Test MoE routed expert fused operation"""

    # MoE router: [1, 7168] x [7168, 256] with 8 cores
    M = 1
    K = 7168
    N_per_core = 32
    num_cores = 8
    N = N_per_core * num_cores  # 256 total output width

    # Tile definitions
    tile_1x32 = ttnn.Tile([1, 32])
    tile_32x32 = ttnn.Tile([32, 32])  # For weights
    tile_16x16 = ttnn.Tile([16, 16])  # For gate 16x16 tensors
    tile_1x16 = ttnn.Tile([1, 16])  # For gate output tensors

    logger.info(f"Testing MoE routed expert: [{M}, {K}] x [{K}, {N}] with {num_cores} cores")

    # Gate parameters (must match op.py)
    gate_eps = 1e-20
    gate_scaling_factor = 2.5

    # Create input, weights, and gate tensors
    torch.manual_seed(0)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_weights = torch.randn((K, N), dtype=torch.bfloat16)
    torch_bias = torch.randn(
        (1, 8, 32), dtype=torch.bfloat16
    )  # Gate bias (batch=1, 8, 32) - matches golden expectation
    # Expert indices 0-255, transposed as expected by gate
    torch_indices = torch.arange(N, dtype=torch.int32).reshape(16, 16).T.contiguous().to(torch.uint16)

    # Compute reference output using DeepseekMoeGateSingleCore.golden
    # First compute matmul + sigmoid (what the kernel does before gate)
    expected_matmul_sigmoid = torch.sigmoid(torch_input.float() @ torch_weights.float())
    # Reshape to (1, 8, 32) for gate golden (golden expects 8 rows x 32 cols)
    expected_gate_input = expected_matmul_sigmoid.reshape(1, 8, 32)
    # Use the correct gate golden with enable_sigmoid=False (sigmoid already applied)
    torch_expected_scores, torch_expected_indices = DeepseekMoeGateSingleCore.golden(
        expected_gate_input, torch_bias.float(), gate_eps, gate_scaling_factor, enable_sigmoid=False
    )

    # Define core grid for compute (first column, 8 cores)
    compute_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_cores - 1))])

    # Input tensor: sharded on sender core OUTSIDE the compute grid
    # Same location as pre_sdpa mcast sender: (device_grid_x - 1, 9)
    device_grid_size = device.compute_with_storage_grid_size()
    input_core = ttnn.CoreCoord(device_grid_size.x - 1, 9)
    input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(input_core, input_core)])
    input_shard_spec = ttnn.ShardSpec(
        input_core_grid,
        (M, K),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile_1x32,
    )
    logger.info(f"Created input tensor with shard shape ({M}, {K}) on core ({input_core.x}, {input_core.y})")

    # Weights: width-sharded across 8 cores
    # Each core gets [K, N_per_core] = [7168, 32]
    weights_shard_spec = ttnn.ShardSpec(
        compute_core_grid,
        (K, N_per_core),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights_shard_spec
    )

    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=tile_32x32,
    )
    logger.info(f"Created weights tensor with shard shape ({K}, {N_per_core}) on {num_cores} cores")

    # Intermediate tensor: matmul output width-sharded across compute cores
    # Each core produces [1, N_per_core] = [1, 32]
    intermediate_shard_spec = ttnn.ShardSpec(
        compute_core_grid,
        (M, N_per_core),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    intermediate_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, intermediate_shard_spec
    )

    torch_intermediate_zeros = torch.zeros((M, N), dtype=torch.bfloat16)
    ttnn_intermediate = ttnn.from_torch(
        torch_intermediate_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=intermediate_mem_config,
        tile=tile_1x32,
    )
    logger.info(f"Created intermediate tensor with shard shape ({M}, {N_per_core}) on {num_cores} cores")

    # Gate input tensor: sharded on sender core (gathered from compute cores)
    # [16, 16] = 256 elements on single core (receives gathered matmul output)
    gate_input_shard_spec = ttnn.ShardSpec(
        input_core_grid,  # Same core as input (sender core)
        (16, 16),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gate_input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gate_input_shard_spec
    )

    torch_gate_input_zeros = torch.zeros((16, 16), dtype=torch.bfloat16)
    ttnn_gate_input = ttnn.from_torch(
        torch_gate_input_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_input_mem_config,
        tile=tile_16x16,
    )
    logger.info(f"Created gate input tensor with shard shape (16, 16) on sender core ({input_core.x}, {input_core.y})")

    # Gate bias tensor: sharded on sender core (transposed as expected by gate)
    # Reshape from (1, 8, 32) to (16, 16) and transpose (matches unit test pattern)
    torch_bias_reshaped = torch_bias.reshape(16, 16)
    torch_bias_transposed = torch.transpose(torch_bias_reshaped, 0, 1).contiguous()
    ttnn_gate_bias = ttnn.from_torch(
        torch_bias_transposed,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_input_mem_config,
        tile=tile_16x16,
    )
    logger.info(f"Created gate bias tensor with shard shape (16, 16) on sender core")

    # Gate indices tensor: sharded on sender core (uint16 indices, already transposed)
    ttnn_gate_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_input_mem_config,
        tile=tile_16x16,
    )
    logger.info(f"Created gate indices tensor with shard shape (16, 16) on sender core")

    # Gate output scores tensor: sharded on sender core [1, 16]
    gate_output_shard_spec = ttnn.ShardSpec(
        input_core_grid,
        (1, 16),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gate_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gate_output_shard_spec
    )

    torch_gate_output_scores_zeros = torch.zeros((1, 16), dtype=torch.bfloat16)
    ttnn_gate_output_scores = ttnn.from_torch(
        torch_gate_output_scores_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_output_mem_config,
        tile=tile_1x16,
    )
    logger.info(f"Created gate output scores tensor with shard shape (1, 16) on sender core")

    # Gate output indices tensor: sharded on sender core [1, 16]
    torch_gate_output_indices_zeros = torch.zeros((1, 16), dtype=torch.uint16)
    ttnn_gate_output_indices = ttnn.from_torch(
        torch_gate_output_indices_zeros,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_output_mem_config,
        tile=tile_1x16,
    )
    logger.info(f"Created gate output indices tensor with shard shape (1, 16) on sender core")

    # Run fused operation
    logger.info("Running MoE routed expert fused operation...")
    ttnn_result_scores, ttnn_result_indices = MoeRoutedExpert.op(
        ttnn_input,
        ttnn_weights,
        ttnn_intermediate,
        ttnn_gate_input,
        ttnn_gate_bias,
        ttnn_gate_indices,
        ttnn_gate_output_scores,
        ttnn_gate_output_indices,
        fp32_dest_acc_en=True,
    )

    # Convert back to torch for comparison
    output_scores_torch = ttnn.to_torch(ttnn_result_scores)
    output_indices_torch = ttnn.to_torch(ttnn_result_indices)

    # Verify output shapes
    assert output_scores_torch.shape == (1, 16), f"Expected scores shape (1, 16), got {output_scores_torch.shape}"
    assert output_indices_torch.shape == (1, 16), f"Expected indices shape (1, 16), got {output_indices_torch.shape}"

    # Extract top-8 values (first 8 elements are valid)
    output_scores_top8 = output_scores_torch[0, :8]
    output_indices_top8 = output_indices_torch[0, :8]

    # Verify results
    logger.info("Verifying results...")

    # Sort both actual and expected by indices to handle tie-breaking differences
    sorted_output_indices, sort_idx = torch.sort(output_indices_top8.to(torch.int64), dim=-1)
    sorted_output_scores = torch.gather(output_scores_top8, dim=-1, index=sort_idx)

    # Squeeze batch dimension from golden output and sort
    sorted_expected_indices, sort_idx_expected = torch.sort(torch_expected_indices.squeeze(0).to(torch.int64), dim=-1)
    sorted_expected_scores = torch.gather(torch_expected_scores.squeeze(0).bfloat16(), dim=-1, index=sort_idx_expected)

    logger.info(f"Expected indices (sorted): {sorted_expected_indices.tolist()}")
    logger.info(f"Actual indices (sorted): {sorted_output_indices.tolist()}")
    logger.info(f"Expected scores (sorted): {sorted_expected_scores.tolist()}")
    logger.info(f"Actual scores (sorted): {sorted_output_scores.tolist()}")

    # Check indices match (after sorting)
    assert torch.equal(sorted_output_indices, sorted_expected_indices), "Output indices do not match"

    # Check scores match (after sorting)
    assert torch.allclose(
        sorted_output_scores, sorted_expected_scores, atol=1e-2, rtol=1e-4
    ), "Output scores do not match"

    logger.info("MoE routed expert fused operation test passed!")

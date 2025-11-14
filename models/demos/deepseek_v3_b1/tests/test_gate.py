"""
TTNN Gate Operation Test for DeepSeek B1
Tests gate operation which performs router computation with shape [1, 7168] -> [1, 256]
Router performs: matmul, sigmoid, add bias, top-8 selection, and normalization
"""

import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_gate(device):
    """Test TTNN gate operation for DeepSeek B1 router"""

    # Router dimensions from gate operation:
    # Input activations: [1, 7168]
    # Router computation: [1, 7168] @ [7168, 256] -> [1, 256]
    # Then applies sigmoid, add bias, top-8, normalize
    tile_height = 1
    m = tile_height  # batch size of 1
    k = 7168  # input feature dimension
    router_experts = 256  # number of experts for router

    logger.info(f"Testing gate operation with activation shape [{m}, {k}]")
    logger.info(f"Expected router output shape: [{m}, {router_experts}]")

    # Create PyTorch tensors for input activations, router weights, and expert bias
    torch.manual_seed(0)
    torch_activations = torch.randn((m, k), dtype=torch.bfloat16)
    torch_router_weights = torch.randn((k, router_experts), dtype=torch.bfloat16)
    torch_expert_bias = torch.randn((m, router_experts), dtype=torch.bfloat16)

    # For reference, compute the expected output
    # (The actual gate op will do matmul + bias + sigmoid + top-k + normalize internally)
    torch_matmul_output = torch.matmul(torch_activations.float(), torch_router_weights.float())
    torch_sigmoid_output = torch.sigmoid(torch_matmul_output)
    torch_reference_output = (torch_sigmoid_output + torch_expert_bias.float()).bfloat16()

    # Create memory config for input activations
    # Using single core for simplicity
    grid_size = ttnn.CoreGrid(y=1, x=1)
    num_cores = grid_size.num_cores

    k_tiles = k // 32  # 224 tiles
    shard_shape = (m, k)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    # Create input tensor for activations
    ttnn_activations = ttnn.from_torch(
        torch_activations,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=ttnn.Tile([tile_height, 32]),
    )

    # Create memory config for router weights [7168, 256]
    router_shard_shape = (k, router_experts)
    router_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}),
        router_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    router_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, router_shard_spec)

    # Create input tensor for router weights
    ttnn_router_weights = ttnn.from_torch(
        torch_router_weights,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=router_mem_config,
    )

    # Create memory config for expert bias [1, 256]
    bias_shard_shape = (m, router_experts)
    bias_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}),
        bias_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    bias_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, bias_shard_spec)

    # Create input tensor for expert bias
    ttnn_expert_bias = ttnn.from_torch(
        torch_expert_bias,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=bias_mem_config,
        tile=ttnn.Tile([tile_height, 32]),
    )

    # Create output memory config
    # Output shape will be [1, 256] after router computation
    output_shard_shape = (m, router_experts)
    output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}),
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # Use the new DeepSeek B1 gate operation with three input tensors
    logger.info(f"Input activations tensor: {ttnn_activations}")
    logger.info(f"Input router weights tensor: {ttnn_router_weights}")
    logger.info(f"Input expert bias tensor: {ttnn_expert_bias}")
    logger.info("Calling ttnn.experimental.deepseek_b1.gate...")

    for i in range(1000):
        print(f"Iteration {i}")
        ttnn_output = ttnn.experimental.deepseek_b1.gate(
            ttnn_activations,
            ttnn_router_weights,
            ttnn_expert_bias,
            grid_size,
            memory_config=output_mem_config,
        )

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(ttnn_output)

    logger.info(f"Output shape: {output_torch.shape}")
    logger.info(
        f"Output tensor stats - min: {output_torch.min():.4f}, max: {output_torch.max():.4f}, mean: {output_torch.mean():.4f}"
    )

    # Verify output shape matches expected router output
    assert output_torch.shape == (
        m,
        router_experts,
    ), f"Expected shape ({m}, {router_experts}), got {output_torch.shape}"

    logger.info("✓ Gate operation test passed using ttnn.experimental.deepseek_b1.gate!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_rmsnorm(device):
    """Test TTNN rmsnorm with width-sharded input and mcast 1d using DeepSeek B1 op"""

    # Create PyTorch tensors for reference
    torch.manual_seed(0)

    torch_input = torch.randn((1, 1536), dtype=torch.bfloat16)
    torch_weight = torch.ones((1536,), dtype=torch.bfloat16)
    torch_bias = torch.zeros((1536,), dtype=torch.bfloat16)

    torch_output = torch.nn.functional.layer_norm(torch_input, (1536,), weight=torch_weight, bias=torch_bias)

    # Create width-sharded memory config for input A
    width_sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)

    # Create width-sharded input A directly
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=width_sharded_mem_config
    )

    ttnn_output = ttnn.experimental.deepseek_b1.layernorm(ttnn_input, memory_config=width_sharded_mem_config)

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(ttnn_output)

    torch_rms = torch.nn.RMSNorm(1536)
    torch_output = torch_rms(torch_input) * torch_weight + torch_bias

    # Compute PCC (Pearson Correlation Coefficient) for accuracy check
    output_flat = output_torch.flatten().float()
    expected_flat = torch_output.flatten().float()

    mean_output = output_flat.mean()
    mean_expected = expected_flat.mean()

    output_centered = output_flat - mean_output
    expected_centered = expected_flat - mean_expected

    correlation = (output_centered * expected_centered).sum()
    norm_output = torch.sqrt((output_centered**2).sum())
    norm_expected = torch.sqrt((expected_centered**2).sum())

    pcc = correlation / (norm_output * norm_expected)

    logger.info(f"PCC: {pcc.item():.6f}")

    # Check that PCC is above threshold
    assert pcc.item() > 0.99, f"PCC {pcc.item():.6f} is below threshold 0.99"

    logger.info("✓ Width-sharded mcast 1d matmul test passed using ttnn.experimental.deepseek_b1.matmul_1d!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

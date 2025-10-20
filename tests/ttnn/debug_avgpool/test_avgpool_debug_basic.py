#!/usr/bin/env python3

import pytest
import ttnn
import torch
import numpy as np


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_avgpool_basic_2x_kernel_positions(device):
    """Test basic avgpool with 2x kernel positions per iteration for DST optimization"""

    # Simple test case
    batch_size = 1
    input_channels = 64  # This should trigger the DST optimization path
    input_height = 16
    input_width = 16
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)

    # Create input tensor
    torch_input = torch.randn(batch_size, input_channels, input_height, input_width, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Run ttnn avgpool
    try:
        ttnn_output = ttnn.avg_pool2d(
            ttnn_input,
            batch_size=batch_size,
            input_h=input_height,
            input_w=input_width,
            channels=input_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=False,
            count_include_pad=True,
        )

        # Convert back to torch
        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        # Reference implementation
        torch_output = torch.nn.functional.avg_pool2d(
            torch_input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=False,
            count_include_pad=True,
        )

        # Check shapes match
        assert (
            ttnn_output_torch.shape == torch_output.shape
        ), f"Shape mismatch: {ttnn_output_torch.shape} vs {torch_output.shape}"

        # Check values are close (allowing for some numerical differences in bfloat16)
        assert torch.allclose(
            ttnn_output_torch, torch_output, rtol=1e-2, atol=1e-2
        ), "Output values don't match reference"

        print("✓ Basic avgpool test with DST optimization passed")

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

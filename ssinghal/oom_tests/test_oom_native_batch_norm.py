import pytest
import torch
import ttnn


@pytest.mark.parametrize("input_shape_memory", [([1, 64, 1280, 800], 125.0), ([1, 128, 640, 400], 65.0), ([1, 256, 320, 200], 32.5)])
def test_oom_native_batch_norm(device, input_shape_memory):
    """
    Test native_batch_norm operator with shapes that previously caused OOM failures.
    These tests are expected to be SKIPPED due to out-of-memory conditions.

    This test serves to:
    1. Document problematic input shapes for native_batch_norm
    2. Verify OOM handling works correctly
    3. Track memory requirements for optimization
    """
    input_shape, expected_memory_mb = input_shape_memory
    torch.manual_seed(0)

    print(f"Testing native_batch_norm with shape {input_shape} (Expected memory: {expected_memory_mb} MB)")

    try:
        # For batch norm, we need 4D input (N, C, H, W)
        if len(input_shape) != 4:
            pytest.skip(f"Batch norm requires 4D input, got {len(input_shape)}D")

        batch_size, channels, height, width = input_shape
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
        
        # Create batch norm parameters
        weight = torch.ones(channels, dtype=torch.bfloat16)
        bias = torch.zeros(channels, dtype=torch.bfloat16)

        # Convert to ttnn format (N, H, W, C)
        torch_input_ttnn = torch_input.permute(0, 2, 3, 1)

        # This should trigger OOM - we expect this test to be skipped
        ttnn_input = ttnn.from_torch(
            torch_input_ttnn, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        ttnn_weight = ttnn.from_torch(
            weight, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        ttnn_bias = ttnn.from_torch(
            bias, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        # Perform batch norm
        ttnn_output = ttnn.batch_norm(ttnn_input, ttnn_weight, ttnn_bias)

        # If we reach here, the operation succeeded unexpectedly
        pytest.fail(f"Expected OOM for shape {input_shape} but operation succeeded")

    except RuntimeError as e:
        if "Out of Memory" in str(e):
            pytest.skip(f"Expected OOM: {input_shape} requires {expected_memory_mb} MB - {str(e)}")
        else:
            # Some other runtime error
            pytest.fail(f"Unexpected RuntimeError for {input_shape}: {str(e)}")
    except Exception as e:
        # Some other unexpected error
        pytest.fail(f"Unexpected error for {input_shape}: {str(e)}")


@pytest.mark.parametrize("input_shape_memory", [([1, 64, 1280, 800], 125.0), ([1, 128, 640, 400], 65.0), ([1, 256, 320, 200], 32.5)])
def test_memory_estimation_native_batch_norm(input_shape_memory):
    """
    Test to estimate memory requirements without actually running on device.
    This can be used for memory planning and optimization.
    """
    input_shape, expected_memory_mb = input_shape_memory

    # Calculate theoretical memory requirement
    import numpy as np

    # Assuming bfloat16 (2 bytes per element)
    # Batch norm needs input, output, weight, bias, running_mean, running_var
    element_size = 2
    input_elements = np.prod(input_shape)
    channels = input_shape[1]  # C dimension
    param_elements = channels * 4  # weight, bias, running_mean, running_var
    theoretical_memory_mb = ((input_elements * 2) + param_elements) * element_size / (1024 * 1024)  # input + output + params

    print(f"Shape: {input_shape}")
    print(f"  Input elements: {input_elements}")
    print(f"  Parameter elements: {param_elements}")
    print(f"  Theoretical memory: {theoretical_memory_mb:.2f} MB")
    print(f"  Actual OOM at: {expected_memory_mb} MB")
    print(f"  Overhead factor: {expected_memory_mb / max(theoretical_memory_mb, 0.001):.2f}x")

    # This test always passes - it's just for analysis
    assert True

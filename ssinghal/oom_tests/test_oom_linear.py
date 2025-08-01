import pytest
import torch
import ttnn


@pytest.mark.parametrize("input_shape_memory", [([1, 8000], 122.07)])
def test_oom_linear(device, input_shape_memory):
    """
    Test linear operator with shapes that previously caused OOM failures.
    These tests are expected to be SKIPPED due to out-of-memory conditions.

    This test serves to:
    1. Document problematic input shapes for linear
    2. Verify OOM handling works correctly
    3. Track memory requirements for optimization
    """
    input_shape, expected_memory_mb = input_shape_memory
    torch.manual_seed(0)

    print(f"Testing linear with shape {input_shape} (Expected memory: {expected_memory_mb} MB)")

    try:
        batch_size, seq_len, hidden_size = input_shape if len(input_shape) == 3 else (1, input_shape[0], input_shape[1])
        torch_input = torch.rand((batch_size, seq_len, hidden_size), dtype=torch.bfloat16)
        torch_weight = torch.rand((hidden_size, hidden_size), dtype=torch.bfloat16)

        # This should trigger OOM - we expect this test to be skipped
        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        ttnn_weight = ttnn.from_torch(
            torch_weight, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.linear(ttnn_input, ttnn_weight)

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


@pytest.mark.parametrize("input_shape_memory", [([1, 8000], 122.07)])
def test_memory_estimation_linear(input_shape_memory):
    """
    Test to estimate memory requirements without actually running on device.
    This can be used for memory planning and optimization.
    """
    input_shape, expected_memory_mb = input_shape_memory

    # Calculate theoretical memory requirement
    import numpy as np

    # Assuming bfloat16 (2 bytes per element)
    element_size = 2
    total_elements = np.prod(input_shape)
    theoretical_memory_mb = (total_elements * element_size) / (1024 * 1024)

    print(f"Shape: {input_shape}")
    print(f"  Theoretical memory: {theoretical_memory_mb:.2f} MB")
    print(f"  Actual OOM at: {expected_memory_mb} MB")
    print(f"  Overhead factor: {expected_memory_mb / max(theoretical_memory_mb, 0.001):.2f}x")

    # This test always passes - it's just for analysis
    assert True

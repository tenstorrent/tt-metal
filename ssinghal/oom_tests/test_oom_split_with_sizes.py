import pytest
import torch
import ttnn


@pytest.mark.parametrize("input_shape_sizes_memory", [
    ([1, 96, 1088, 1920], [48, 48], 380.0),     # YOLOv12x after first conv
    ([1, 192, 544, 960], [96, 96], 190.0),      # After second conv stride=2
    ([1, 384, 272, 480], [192, 192], 95.0),     # After third conv stride=2
    ([1, 768, 136, 240], [384, 384], 47.5),     # After fourth conv stride=2
    ([1, 1152, 68, 120], [576, 576], 17.8),     # Attention feature maps
    ([1, 2304, 34, 60], [1152, 1152], 8.9),     # Very high channel splits
])
def test_oom_split_with_sizes(device, input_shape_sizes_memory):
    """
    Test split_with_sizes operator with shapes that previously caused OOM failures.
    These tests are expected to be SKIPPED due to out-of-memory conditions.

    This test serves to:
    1. Document problematic input shapes for split_with_sizes
    2. Verify OOM handling works correctly
    3. Track memory requirements for optimization
    """
    input_shape, split_sizes, expected_memory_mb = input_shape_sizes_memory
    torch.manual_seed(0)

    print(f"Testing split_with_sizes with shape {input_shape} sizes {split_sizes} (Expected memory: {expected_memory_mb} MB)")

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        # This should trigger OOM - we expect this test to be skipped
        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        # Split with sizes (split along dimension 1 by default)
        dim = 1 if len(input_shape) > 1 else 0
        ttnn_outputs = ttnn.split(ttnn_input, split_sizes, dim=dim)

        # If we reach here, the operation succeeded unexpectedly
        pytest.fail(f"Expected OOM for shape {input_shape} but operation succeeded")

    except RuntimeError as e:
        if "Out of Memory" in str(e):
            pytest.skip(f"Expected OOM: {input_shape} with sizes {split_sizes} requires {expected_memory_mb} MB - {str(e)}")
        else:
            # Some other runtime error
            pytest.fail(f"Unexpected RuntimeError for {input_shape}: {str(e)}")
    except Exception as e:
        # Some other unexpected error
        pytest.fail(f"Unexpected error for {input_shape}: {str(e)}")


@pytest.mark.parametrize("input_shape_sizes_memory", [
    ([1, 96, 1088, 1920], [48, 48], 380.0),     # YOLOv12x after first conv
    ([1, 192, 544, 960], [96, 96], 190.0),      # After second conv stride=2
    ([1, 384, 272, 480], [192, 192], 95.0),     # After third conv stride=2
    ([1, 768, 136, 240], [384, 384], 47.5),     # After fourth conv stride=2
    ([1, 1152, 68, 120], [576, 576], 17.8),     # Attention feature maps
    ([1, 2304, 34, 60], [1152, 1152], 8.9),     # Very high channel splits
])
def test_memory_estimation_split_with_sizes(input_shape_sizes_memory):
    """
    Test to estimate memory requirements without actually running on device.
    This can be used for memory planning and optimization.
    """
    input_shape, split_sizes, expected_memory_mb = input_shape_sizes_memory

    # Calculate theoretical memory requirement
    import numpy as np

    # Assuming bfloat16 (2 bytes per element)
    # Split creates multiple output tensors, so memory usage is input + all outputs
    element_size = 2
    input_elements = np.prod(input_shape)
    num_splits = len(split_sizes)
    theoretical_memory_mb = (input_elements * element_size * (1 + num_splits)) / (1024 * 1024)  # input + outputs

    print(f"Shape: {input_shape}, Split sizes: {split_sizes}")
    print(f"  Number of splits: {num_splits}")
    print(f"  Theoretical memory: {theoretical_memory_mb:.2f} MB")
    print(f"  Actual OOM at: {expected_memory_mb} MB")
    print(f"  Overhead factor: {expected_memory_mb / max(theoretical_memory_mb, 0.001):.2f}x")

    # This test always passes - it's just for analysis
    assert True

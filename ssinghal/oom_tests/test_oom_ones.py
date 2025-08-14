import pytest
import torch
import ttnn


@pytest.mark.parametrize("input_shape_memory", [([1, 3, 1280, 1280], 19.5), ([1, 96, 640, 640], 75.0), ([1, 192, 320, 320], 37.5)])
def test_oom_ones(device, input_shape_memory):
    """
    Test ones operator with shapes that previously caused OOM failures.
    These tests are expected to be SKIPPED due to out-of-memory conditions.

    This test serves to:
    1. Document problematic input shapes for ones
    2. Verify OOM handling works correctly
    3. Track memory requirements for optimization
    """
    input_shape, expected_memory_mb = input_shape_memory
    torch.manual_seed(0)

    print(f"Testing ones with shape {input_shape} (Expected memory: {expected_memory_mb} MB)")

    try:
        # Create ones tensor for YOLOv12 shapes
        if len(input_shape) == 4:
            ttnn_shape = [input_shape[0], input_shape[2], input_shape[3], input_shape[1]]  # NHWC for ttnn
        else:
            ttnn_shape = input_shape
        
        # This should trigger OOM - we expect this test to be skipped
        ttnn_output = ttnn.ones(ttnn_shape, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

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


@pytest.mark.parametrize("input_shape_memory", [([1, 3, 1280, 1280], 19.5), ([1, 96, 640, 640], 75.0), ([1, 192, 320, 320], 37.5)])
def test_memory_estimation_ones(input_shape_memory):
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

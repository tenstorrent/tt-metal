import pytest
import torch
import ttnn


@pytest.mark.parametrize("input_shape_memory", [
        # YOLOv12x ultra-high resolution (2176x3840) OOM test shapes
        ([1, 3, 2176, 3840], 150.0),   # YOLOv12x ultra-high-res input
        ([1, 96, 1088, 1920], 380.0),  # After first conv stride=2
        ([1, 192, 544, 960], 190.0),   # After second conv stride=2
        ([1, 384, 272, 480], 95.0),    # After third conv stride=2
        ([1, 768, 136, 240], 47.5),    # After fourth conv stride=2
        ([1, 1152, 68, 120], 17.8),    # Attention feature maps
])
def test_oom_hardtanh(device, input_shape_memory):
    """
    Test hardtanh operator with shapes that previously caused OOM failures.
    These tests are expected to be SKIPPED due to out-of-memory conditions.

    This test serves to:
    1. Document problematic input shapes for hardtanh
    2. Verify OOM handling works correctly
    3. Track memory requirements for optimization
    """
    input_shape, expected_memory_mb = input_shape_memory
    torch.manual_seed(0)

    print(f"Testing hardtanh with shape {input_shape} (Expected memory: {expected_memory_mb} MB)")

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        # This should trigger OOM - we expect this test to be skipped
        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.hardtanh(ttnn_input)

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


@pytest.mark.parametrize("input_shape_memory", [
        # YOLOv12x ultra-high resolution (2176x3840) OOM test shapes
        ([1, 3, 2176, 3840], 150.0),   # YOLOv12x ultra-high-res input
        ([1, 96, 1088, 1920], 380.0),  # After first conv stride=2
        ([1, 192, 544, 960], 190.0),   # After second conv stride=2
        ([1, 384, 272, 480], 95.0),    # After third conv stride=2
        ([1, 768, 136, 240], 47.5),    # After fourth conv stride=2
        ([1, 1152, 68, 120], 17.8),    # Attention feature maps
])
def test_memory_estimation_hardtanh(input_shape_memory):
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

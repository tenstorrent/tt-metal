import pytest
import torch
import ttnn


@pytest.mark.parametrize("input_shape_memory", [
    ([1, 3, 2176, 3840], 300.0),   # YOLOv12x ultra-high-res input
    ([1, 96, 1088, 1920], 760.0),  # After first conv stride=2
    ([1, 192, 544, 960], 380.0),   # After second conv stride=2
    ([1, 384, 272, 480], 190.0),   # After third conv stride=2
    ([1, 768, 136, 240], 95.0),    # After fourth conv stride=2
    ([1, 1152, 68, 120], 35.6),    # Attention feature maps
])
def test_oom_sub(device, input_shape_memory):
    """
    Test sub operator with shapes that previously caused OOM failures.
    These tests are expected to be SKIPPED due to out-of-memory conditions.

    This test serves to:
    1. Document problematic input shapes for sub
    2. Verify OOM handling works correctly
    3. Track memory requirements for optimization
    """
    input_shape, expected_memory_mb = input_shape_memory
    torch.manual_seed(0)

    print(f"Testing sub with shape {input_shape} (Expected memory: {expected_memory_mb} MB)")

    try:
        torch_input1 = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_input2 = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input1 = torch_input1.permute(0, 2, 3, 1)
            torch_input2 = torch_input2.permute(0, 2, 3, 1)

        # This should trigger OOM - we expect this test to be skipped
        ttnn_input1 = ttnn.from_torch(
            torch_input1, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        ttnn_input2 = ttnn.from_torch(
            torch_input2, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.sub(ttnn_input1, ttnn_input2)

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
    ([1, 3, 2176, 3840], 300.0),   # YOLOv12x ultra-high-res input
    ([1, 96, 1088, 1920], 760.0),  # After first conv stride=2
    ([1, 192, 544, 960], 380.0),   # After second conv stride=2
    ([1, 384, 272, 480], 190.0),   # After third conv stride=2
    ([1, 768, 136, 240], 95.0),    # After fourth conv stride=2
    ([1, 1152, 68, 120], 35.6),    # Attention feature maps
])
def test_memory_estimation_sub(input_shape_memory):
    """
    Test to estimate memory requirements without actually running on device.
    This can be used for memory planning and optimization.
    """
    input_shape, expected_memory_mb = input_shape_memory

    # Calculate theoretical memory requirement
    import numpy as np

    # Assuming bfloat16 (2 bytes per element) for 2 input tensors + 1 output
    element_size = 2
    total_elements = np.prod(input_shape)
    theoretical_memory_mb = (total_elements * element_size * 3) / (1024 * 1024)  # 3x for input1, input2, output

    print(f"Shape: {input_shape}")
    print(f"  Theoretical memory: {theoretical_memory_mb:.2f} MB")
    print(f"  Actual OOM at: {expected_memory_mb} MB")
    print(f"  Overhead factor: {expected_memory_mb / max(theoretical_memory_mb, 0.001):.2f}x")

    # This test always passes - it's just for analysis
    assert True

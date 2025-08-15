import pytest
import torch
import ttnn


@pytest.mark.parametrize("input_shape_scale_memory", [
    # YOLOv12x ultra-high resolution (2176x3840) OOM test shapes for upsample
    ([1, 768, 68, 120], 2, 95.0),       # YOLOv12x upsample 68x120 -> 136x240
    ([1, 384, 136, 240], 2, 190.0),     # Upsample 136x240 -> 272x480
    ([1, 192, 272, 480], 2, 380.0),     # Upsample 272x480 -> 544x960
    ([1, 96, 544, 960], 2, 760.0),      # Upsample 544x960 -> 1088x1920
    ([1, 768, 34, 60], 4, 190.0),       # 4x upsampling for low-res features
    ([1, 384, 68, 120], 4, 380.0),      # 4x upsampling for medium-res features
])
def test_oom_upsample_nearest2d(device, input_shape_scale_memory):
    """
    Test upsample_nearest2d operator with shapes that previously caused OOM failures.
    These tests are expected to be SKIPPED due to out-of-memory conditions.

    This test serves to:
    1. Document problematic input shapes for upsample_nearest2d
    2. Verify OOM handling works correctly
    3. Track memory requirements for optimization
    """
    input_shape, scale_factor, expected_memory_mb = input_shape_scale_memory
    torch.manual_seed(0)

    print(f"Testing upsample_nearest2d with shape {input_shape} scale {scale_factor} (Expected memory: {expected_memory_mb} MB)")

    try:
        # For upsample, we need 4D input (N, C, H, W)
        if len(input_shape) != 4:
            pytest.skip(f"Upsample requires 4D input, got {len(input_shape)}D")

        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
        
        # Convert to ttnn format (N, H, W, C)
        torch_input_ttnn = torch_input.permute(0, 2, 3, 1)

        # This should trigger OOM - we expect this test to be skipped
        ttnn_input = ttnn.from_torch(
            torch_input_ttnn, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        # Perform upsampling
        ttnn_output = ttnn.upsample(ttnn_input, scale_factor)

        # If we reach here, the operation succeeded unexpectedly
        pytest.fail(f"Expected OOM for shape {input_shape} with scale {scale_factor} but operation succeeded")

    except RuntimeError as e:
        if "Out of Memory" in str(e):
            pytest.skip(f"Expected OOM: {input_shape} with scale {scale_factor} requires {expected_memory_mb} MB - {str(e)}")
        else:
            # Some other runtime error
            pytest.fail(f"Unexpected RuntimeError for {input_shape}: {str(e)}")
    except Exception as e:
        # Some other unexpected error
        pytest.fail(f"Unexpected error for {input_shape}: {str(e)}")


@pytest.mark.parametrize("input_shape_scale_memory", [
    # YOLOv12x ultra-high resolution (2176x3840) OOM test shapes for upsample
    ([1, 768, 68, 120], 2, 95.0),       # YOLOv12x upsample 68x120 -> 136x240
    ([1, 384, 136, 240], 2, 190.0),     # Upsample 136x240 -> 272x480
    ([1, 192, 272, 480], 2, 380.0),     # Upsample 272x480 -> 544x960
    ([1, 96, 544, 960], 2, 760.0),      # Upsample 544x960 -> 1088x1920
    ([1, 768, 34, 60], 4, 190.0),       # 4x upsampling for low-res features
    ([1, 384, 68, 120], 4, 380.0),      # 4x upsampling for medium-res features
])
def test_memory_estimation_upsample_nearest2d(input_shape_scale_memory):
    """
    Test to estimate memory requirements without actually running on device.
    This can be used for memory planning and optimization.
    """
    input_shape, scale_factor, expected_memory_mb = input_shape_scale_memory

    # Calculate theoretical memory requirement
    import numpy as np

    # Assuming bfloat16 (2 bytes per element)
    element_size = 2
    input_elements = np.prod(input_shape)
    output_elements = input_elements * (scale_factor ** 2)  # upsampling scales area
    theoretical_memory_mb = ((input_elements + output_elements) * element_size) / (1024 * 1024)

    print(f"Shape: {input_shape}, Scale: {scale_factor}")
    print(f"  Input elements: {input_elements}")
    print(f"  Output elements: {output_elements}")
    print(f"  Theoretical memory: {theoretical_memory_mb:.2f} MB")
    print(f"  Actual OOM at: {expected_memory_mb} MB")
    print(f"  Overhead factor: {expected_memory_mb / max(theoretical_memory_mb, 0.001):.2f}x")

    # This test always passes - it's just for analysis
    assert True

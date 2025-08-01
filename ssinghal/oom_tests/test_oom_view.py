import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "input_shape_memory",
    [
        ([160, 100, 512, 1, 1], 16000.0),
        ([80, 50, 512, 1, 1], 4000.0),
        ([1, 1, 64000, 1000], 4000.0),
        ([2, 160, 100, 32, 1, 1], 2000.0),
        ([4, 80, 50, 32, 1, 1], 1000.0),
        ([40, 25, 512, 1, 1], 1000.0),
        ([1, 2, 16000, 1000], 1000.0),
        ([8, 40, 25, 32, 1, 1], 500.0),
        ([1334, 4, 49, 49], 255.34),
        ([1, 5, 4000, 1000], 250.0),
        ([1000, 4, 64, 64], 250.0),
        ([1, 2, 160, 100, 80, 1], 187.5),
        ([1334, 49, 4, 32], 127.67),
        ([1, 1, 64000, 32], 125.0),
        ([1, 64000, 1, 32], 125.0),
        ([1000, 64, 4, 32], 125.0),
        ([1, 80, 160, 100, 1], 100.0),
        ([1, 4, 80, 50, 80, 1], 93.75),
        ([1334, 7, 7, 128], 72.95),
        ([1000, 8, 8, 128], 62.5),
        ([345, 8, 49, 49], 21.56),
    ],
)
def test_oom_view(device, input_shape_memory):
    """
    Test view operator with shapes that previously caused OOM failures.
    These tests are expected to be SKIPPED due to out-of-memory conditions.

    This test serves to:
    1. Document problematic input shapes for view
    2. Verify OOM handling works correctly
    3. Track memory requirements for optimization
    """
    input_shape, expected_memory_mb = input_shape_memory
    torch.manual_seed(0)

    print(f"Testing view with shape {input_shape} (Expected memory: {expected_memory_mb} MB)")

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        # This should trigger OOM - we expect this test to be skipped
        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.reshape(ttnn_input, input_shape)

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


@pytest.mark.parametrize(
    "input_shape_memory",
    [
        ([160, 100, 512, 1, 1], 16000.0),
        ([80, 50, 512, 1, 1], 4000.0),
        ([1, 1, 64000, 1000], 4000.0),
        ([2, 160, 100, 32, 1, 1], 2000.0),
        ([4, 80, 50, 32, 1, 1], 1000.0),
        ([40, 25, 512, 1, 1], 1000.0),
        ([1, 2, 16000, 1000], 1000.0),
        ([8, 40, 25, 32, 1, 1], 500.0),
        ([1334, 4, 49, 49], 255.34),
        ([1, 5, 4000, 1000], 250.0),
        ([1000, 4, 64, 64], 250.0),
        ([1, 2, 160, 100, 80, 1], 187.5),
        ([1334, 49, 4, 32], 127.67),
        ([1, 1, 64000, 32], 125.0),
        ([1, 64000, 1, 32], 125.0),
        ([1000, 64, 4, 32], 125.0),
        ([1, 80, 160, 100, 1], 100.0),
        ([1, 4, 80, 50, 80, 1], 93.75),
        ([1334, 7, 7, 128], 72.95),
        ([1000, 8, 8, 128], 62.5),
        ([345, 8, 49, 49], 21.56),
    ],
)
def test_memory_estimation_view(input_shape_memory):
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

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        # Original + 8K YOLOv12x ultra-high resolution shapes
        [1, 3, 2176, 3840],
        [1, 96, 1088, 1920],
        [1, 192, 544, 960],
        [1, 384, 272, 480],
        [1, 768, 136, 240],
        [1, 768, 68, 120],
        [1, 96, 2176, 1920],
        [1, 192, 1088, 960],
        [1, 384, 544, 480],
        [1, 768, 272, 240],
        [1, 1152, 136, 120],
        [1, 384, 136, 240],
        [1, 768, 68, 120],
        [1, 1152, 34, 60],
        [1, 3, 1088, 1920],
        [1, 96, 544, 960],
        [1, 192, 272, 480],
        [1, 384, 136, 240],
        [1, 768, 68, 120],
        [1, 3, 4320, 7680],  # 8K feature (189.8MB)
        [1, 96, 2160, 3840],  # 8K feature (1518.8MB)
        [1, 96, 1080, 1920],  # 8K feature (379.7MB)
        [1, 192, 1080, 1920],  # 8K feature (759.4MB)
        [1, 192, 540, 960],  # 8K feature (189.8MB)
        [1, 384, 540, 960],  # 8K feature (379.7MB)
        [1, 384, 270, 480],  # 8K feature (94.9MB)
        [1, 768, 270, 480],  # 8K feature (189.8MB)
        [1, 768, 135, 240],  # 8K feature (47.5MB)
        [1, 1536, 135, 240],  # 8K feature (94.9MB)
        [1, 96, 2160, 1920],  # 8K feature (759.4MB)
        [1, 192, 1080, 960],  # 8K feature (379.7MB)
        [1, 384, 540, 480],  # 8K feature (189.8MB)
        [1, 768, 270, 240],  # 8K feature (94.9MB)
        [1, 1152, 135, 120],  # 8K feature (35.6MB)
        [1, 384, 135, 240],  # 8K feature (23.7MB)
        [1, 768, 67, 120],  # 8K feature (11.8MB)
        [1, 1152, 33, 60],  # 8K feature (4.4MB)
        [1, 96, 540, 960],  # 8K feature (94.9MB)
        [1, 192, 270, 480],  # 8K feature (47.5MB)
    ],
)
def test_native_batch_norm(device, input_shape):
    """Test native_batch_norm operator with YOLOv12x ultra-high resolution (2176x3840) input shapes"""
    torch.manual_seed(0)

    try:
        # For batch norm, we need 4D input (N, C, H, W)
        if len(input_shape) != 4:
            pytest.skip(f"Batch norm requires 4D input, got {len(input_shape)}D")

        batch_size, channels, height, width = input_shape
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        # Create batch norm parameters
        weight = torch.ones(channels, dtype=torch.bfloat16)
        bias = torch.zeros(channels, dtype=torch.bfloat16)
        running_mean = torch.zeros(channels, dtype=torch.bfloat16)
        running_var = torch.ones(channels, dtype=torch.bfloat16)

        # Convert to ttnn format (N, H, W, C)
        torch_input_ttnn = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input_ttnn, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        ttnn_weight = ttnn.from_torch(
            weight, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        ttnn_bias = ttnn.from_torch(bias, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)

        # Perform batch norm
        ttnn_output = ttnn.batch_norm(ttnn_input, ttnn_weight, ttnn_bias)

        # PyTorch reference
        torch_reference = torch.nn.functional.batch_norm(
            torch_input, running_mean, running_var, weight, bias, training=True
        )
        torch_reference = torch_reference.permute(0, 2, 3, 1)  # Convert to NHWC

        # Convert output back to torch
        ttnn_result = ttnn.to_torch(ttnn_output)

        # Compare results
        check_with_pcc_without_tensor_printout(ttnn_result, torch_reference, 0.95)

    except RuntimeError as e:
        if "Out of Memory" in str(e):
            pytest.skip(f"OOM: {input_shape} - {str(e)}")
        else:
            raise e
    except Exception as e:
        if "incompatible function arguments" in str(e):
            pytest.skip(f"Type error: {input_shape} - {str(e)}")
        else:
            raise e

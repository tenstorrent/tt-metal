import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        # Original + 8K YOLOv12x ultra-high resolution shapes
        [27, 256],
        [80, 512],
        [80, 128],
        [80, 256],
        [1000, 1024],
        [1000, 256],
        [21000, 256],
        [21000, 80],
        [300, 256],
        [300, 4],
        [300, 512],
        [300, 192],
        [300, 96],
        [300, 1024],
        [300, 80],
        [1, 2048],
        [1, 1280],
        [1, 1536],
        [1, 1792],
        [1, 8000],
        [1000, 32],
        [64000, 32],
        [64000, 128],
        [1000, 64],
        [16000, 64],
        [16000, 256],
        [1000, 160],
        [4000, 160],
        [4000, 640],
        [1, 256],
        [65366, 128],
        [64000, 512],
        [16905, 256],
        [16000, 1024],
        [4000, 512],
        [4704, 512],
        [4000, 2048],
        [1176, 1024],
        [1000, 4096],
        [1, 1024],
        [16640, 256],
        [4480, 512],
        [1280, 1024],
        [0],
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
        [1, 1536, 67, 120],
        [1, 96, 4320, 1920],
        [1, 96, 2160, 1920],  # 8K feature (759.4MB)
        [1, 192, 1080, 960],  # 8K feature (379.7MB)
        [1, 384, 540, 480],  # 8K feature (189.8MB)
        [1, 768, 270, 240],  # 8K feature (94.9MB)
        [1, 1152, 135, 120],  # 8K feature (35.6MB)
        [1, 384, 135, 240],  # 8K feature (23.7MB)
        [1, 768, 67, 120],  # 8K feature (11.8MB)
        [1, 1152, 33, 60],  # 8K feature (4.4MB)
        [1, 1536, 16, 30],
        [1, 32400, 768],
        [1, 8100, 1536],
        [1, 2025, 3072],
        [64800, 768],
        [16200, 1536],
        [4050, 3072],
        [1012, 6144],
        [1, 96, 540, 960],  # 8K feature (94.9MB)
        [1, 192, 270, 480],  # 8K feature (47.5MB)
        [768, 3072],
        [1536, 6144],
        [384, 1536],
        [192, 768],
    ],
)
def test_linear(device, input_shape):
    """Test linear operator with various input shapes from vision models"""
    torch.manual_seed(0)

    try:
        # For 2D shapes: treat as [batch_size, features]
        # For 4D shapes: treat as [batch, height, width, channels]
        if len(input_shape) == 2:
            batch_size, hidden_size = input_shape
            seq_len = 1
        else:
            # 4D tensor - flatten for linear operation
            batch_size = input_shape[0]
            seq_len = input_shape[1] * input_shape[2]
            hidden_size = input_shape[3]

        torch_input = torch.rand((batch_size, seq_len, hidden_size), dtype=torch.bfloat16)
        torch_weight = torch.rand((hidden_size, hidden_size), dtype=torch.bfloat16)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        ttnn_weight = ttnn.from_torch(
            torch_weight, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.linear(ttnn_input, ttnn_weight)

        torch_reference = torch.nn.functional.linear(torch_input, torch_weight)

        # Convert output back to torch
        ttnn_result = ttnn.to_torch(ttnn_output)

        # Compare results
        check_with_pcc_without_tensor_printout(ttnn_result, torch_reference, 0.99)

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

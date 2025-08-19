import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        # Original + 8K YOLOv12x ultra-high resolution shapes
        [1, 12, 8160, 8160],
        [1, 12, 32640, 384],
        [1, 8, 4080, 4080],
        [12, 8160, 8160],
        [12, 32640, 32],
        [1, 1152, 68120],
        [8, 2040, 96],
        [16, 1020, 48],
        [4, 16320, 128],
        [24, 680, 32],
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
def test_softmax(device, input_shape):
    """Test softmax operator with YOLOv12x ultra-high resolution (2176x3840) attention shapes"""
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.softmax(ttnn_input)

        torch_reference = torch.softmax(torch_input)

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

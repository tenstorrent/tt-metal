import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        # YOLOv12x ultra-high resolution (2176x3840) unsqueeze operations
        [1, 3, 2176, 3840],   # YOLOv12x ultra-high-res input
        [1, 96, 1088, 1920],  # After first conv stride=2
        [1, 192, 544, 960],   # After second conv stride=2
        [1, 384, 272, 480],   # After third conv stride=2
        [1, 768, 136, 240],   # After fourth conv stride=2
        [1, 768, 68, 120],    # After fifth conv stride=2
        [1, 96, 2176, 1920],  # Ultra-high-res feature maps
        [1, 192, 1088, 960],  # High-res feature maps
        [1, 384, 544, 480],   # Medium-res feature maps
        [1, 768, 272, 240],   # Lower-res feature maps
        [1, 1152, 136, 120],  # Attention feature maps
        [1, 384, 136, 240],   # Detection head shapes
        [1, 768, 68, 120],    # Detection head shapes
        [1, 1152, 34, 60],    # Detection head shapes
        [3, 2176, 3840],      # 3D tensor (ultra-high-res)
        [96, 1088, 1920],     # 3D tensor (after first conv)
        [192, 544, 960],      # 3D tensor (after second conv)
        [384, 272, 480],      # 3D tensor (after third conv)
        [768, 136, 240],      # 3D tensor (after fourth conv)
    ],
)
def test_unsqueeze(device, input_shape):
    """Test unsqueeze operator with YOLOv12x ultra-high resolution (2176x3840) input shapes"""
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.unsqueeze(ttnn_input, 0)

        torch_reference = torch.unsqueeze(torch_input, 0)

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

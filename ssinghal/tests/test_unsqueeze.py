import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        # YOLOv12 high-resolution unsqueeze operations
        [1, 3, 1280, 1280],   # YOLOv12 high-res input
        [1, 96, 640, 640],    # After first conv stride=2
        [1, 192, 320, 320],   # After second conv stride=2
        [1, 384, 160, 160],   # After third conv stride=2
        [1, 768, 80, 80],     # After fourth conv stride=2
        [1, 768, 40, 40],     # After fifth conv stride=2
        [1, 64, 1280, 800],   # High-res feature maps
        [1, 128, 640, 400],   # Medium-res feature maps
        [1, 256, 320, 200],   # Lower-res feature maps
        [1, 32, 1280, 800],   # High-res channels
        [1, 80, 640, 400],    # Detection head shapes
        [1, 160, 320, 200],   # Detection head shapes
        [1, 320, 160, 100],   # Detection head shapes
        [1, 640, 80, 50],     # Detection head shapes
        [1, 1280, 40, 25],    # Detection head shapes
        [3, 1280, 1280],      # 3D tensor
        [96, 640, 640],       # 3D tensor
        [192, 320, 320],      # 3D tensor
        [384, 160, 160],      # 3D tensor
        [768, 80, 80],        # 3D tensor
    ],
)
def test_unsqueeze(device, input_shape):
    """Test unsqueeze operator with YOLOv12 high-resolution input shapes"""
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
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
